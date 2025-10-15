#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
连续 I/O 测试：
1) 权重：按层 & 参数名（或全部 streamable 参数）连续读取到 pinned，测吞吐
2) KV：按 layer/slot 连续写入 & 读回校验，测吞吐
"""

import os
import re
import time
import math
import argparse
import hashlib
from typing import Dict, List, Optional

import torch

# --- 依赖你的工程内模块 ---
# 权重路径
from llama3.raw_param_store import ParamStore
# KV 路径
from SSDBacked import RawBlockKVBackend


# ========== 工具 ==========

def _human(nbytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    x = float(nbytes)
    while x >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {units[i]}"

def _md5_u8(t: torch.Tensor) -> str:
    # 只用于测试校验，不在热路径使用
    assert t.dtype == torch.uint8 and t.is_contiguous()
    return hashlib.md5(memoryview(t.cpu().numpy())).hexdigest()

def _sum_param_total_nbytes(store: ParamStore, layer_id: int, name: str) -> int:
    """
    兼容：若你已添加 ParamStore.param_total_nbytes() 就用官方接口；
    否则使用 _groups_for_layer() 汇总所有 segment 的 nbytes。
    """
    if hasattr(store, "param_total_nbytes"):
        return int(store.param_total_nbytes(layer_id, name))
    # fallback：汇总 segment
    groups = store._groups_for_layer(layer_id, only_stream=False, names=[name])  # noqa
    segs = groups.get(name, [])
    if not segs:
        raise KeyError(f"param not found: L{layer_id}:{name}")
    return sum(int(s["nbytes"]) for s in segs)


# ========== 1) 权重连续读取测试 ==========

def weights_read_sequential(
    manifest_path: str,
    layer_start: int,
    layer_end: int,
    param_regex: Optional[str],
    only_stream: bool,
    verify_hash: bool,
    staging_mb: int = 32,
) -> None:
    """
    连续把 [layer_start, layer_end) 的参数按顺序读入 pinned，统计吞吐。
    - param_regex：仅匹配参数名的正则；为空则加载层内所有(或仅 streamable)参数
    - only_stream：True=仅加载 policy=stream 的参数（推荐），False=层内所有参数
    - verify_hash：对每个读取的 pinned 计算 md5（有开销，仅用于演示/校验）
    """
    print(f"[WEIGHTS] manifest={manifest_path}")
    store = ParamStore(manifest_path, method="bytecopy", staging_mb=staging_mb)

    total_bytes = 0
    t0 = time.perf_counter()
    n_params = 0

    try:
        name_pat = re.compile(param_regex) if param_regex else None
        for L in range(layer_start, layer_end):
            # 列出该层的参数名字集合
            # 使用内部工具：返回 dict[name] = [segments...]；我们只需要名字
            groups = store._groups_for_layer(L, only_stream=only_stream, names=None)  # noqa
            names = sorted(groups.keys())
            if name_pat:
                names = [nm for nm in names if name_pat.search(nm)]
            if not names:
                print(f"[WEIGHTS] L{L}: no params match (only_stream={only_stream}, regex={param_regex})")
                continue

            for name in names:
                nbytes = _sum_param_total_nbytes(store, L, name)
                # 分配 pinned 目标，长度 = 该参数总分片字节数
                dst_u8 = torch.empty(int(nbytes), dtype=torch.uint8, pin_memory=True)
                assert dst_u8.is_pinned() and dst_u8.is_contiguous()

                # 读取（内部：对齐→READ_FIXED；不对齐→staging 包读 + memmove）
                n = store.fetch_param_into_pinned_u8(L, name, dst_u8)
                if n != nbytes:
                    raise RuntimeError(f"Short read for L{L}:{name}, got {n}, expect {nbytes}")

                if verify_hash:
                    h = _md5_u8(dst_u8)
                    print(f"[WEIGHTS] L{L} {name} read {_human(nbytes)} md5={h}")

                total_bytes += nbytes
                n_params += 1

                # 释放 pinned 内存（出作用域后由 GC 释放。你也可以复用一个环形缓冲）
                del dst_u8

            print(f"[WEIGHTS] L{L} done: {len(names)} params")

    finally:
        store.close()

    dt = time.perf_counter() - t0
    gbps = (total_bytes / dt) / (1024**3) if dt > 0 else 0.0
    print(f"[WEIGHTS] layers {layer_start}..{layer_end-1}, {n_params} params, "
          f"total { _human(total_bytes) }, time {dt:.3f}s, avg {gbps:.2f} GB/s")


# ========== 2) KV 连续写入/读取/校验测试 ==========

def kv_rw_sequential(
    dev_path: str,
    n_layers: int,
    blk_bytes: int,
    blk_per_layer: int,
    layer_start: int,
    layer_end: int,
    slots_per_layer: int,
    verify_readback: bool,
    sync_writes: bool,
) -> None:
    """
    KV 连续写入→可选读回校验→吞吐统计。
    - 每个 block 大小为 blk_bytes（设备 stride = ceil(blk_bytes, 4KiB)）
    - 为了简单，我们对每个 layer 的 [0..slots_per_layer-1] 做写入测试（或读回）
    """
    kv = RawBlockKVBackend(dev_path, n_layers=n_layers, blk_bytes=blk_bytes, blk_per_layer=blk_per_layer)
    stride = kv.stride
    print(f"[KV] dev={dev_path} stride={stride} blk_bytes={blk_bytes} blk_per_layer={blk_per_layer}")

    # 目标 pinned 缓冲（对齐检查）
    buf = torch.empty(stride, dtype=torch.uint8, pin_memory=True)
    assert buf.is_pinned() and buf.is_contiguous()
    # data_ptr 对齐仅用于核对（torch 的 pinned 一般是 page 对齐，我们依然检查）
    ptr_mod = (buf.numpy().ctypes.data % 4096)
    if ptr_mod != 0:
        print(f"[KV][WARN] pinned buffer ptr not 4KiB aligned (mod={ptr_mod}). "
              f"代码会自动回退到对齐 staging 路径。")

    total_write = 0
    total_read = 0
    t_write0 = time.perf_counter()

    # 连续写入（数据模式：layer/slot/循环序列，便于简单校验）
    for L in range(layer_start, layer_end):
        for slot in range(slots_per_layer):
            # 准备内容：写入特定模式（便于读回校验）
            buf[:stride].fill_(0)
            # 前 blk_bytes 区域写入简单模式（剩余 padding 区域保持 0）
            view = buf[:blk_bytes]
            # 填充一个重复模式：L, slot, idx
            # 注意：这里只是测试；实际 KV 会来自 GPU→pinned
            view[:min(16, blk_bytes)] = torch.tensor(
                [(L & 0xFF), (slot & 0xFF)] + [i & 0xFF for i in range(14)],
                dtype=torch.uint8
            )[:min(16, blk_bytes)]

            kv.write_from_pinned_aligned(L, slot, buf, sync=sync_writes)
            total_write += stride
        print(f"[KV] layer {L} write {slots_per_layer} slots")

    t_write = time.perf_counter() - t_write0
    w_gbps = (total_write / t_write) / (1024**3) if t_write > 0 else 0.0
    print(f"[KV] write total { _human(total_write) }, time {t_write:.3f}s, avg {w_gbps:.2f} GB/s "
          f"(stride bytes accounted)")

    if verify_readback:
        t_read0 = time.perf_counter()
        for L in range(layer_start, layer_end):
            for slot in range(slots_per_layer):
                buf.zero_()
                kv.read_into_pinned_aligned(L, slot, buf)
                total_read += stride
                # 简单校验：看前几个字节的模式是否一致
                head = bytes(buf[:min(16, blk_bytes)].cpu().numpy().tolist())
                if len(head) >= 2:
                    if head[0] != (L & 0xFF) or head[1] != (slot & 0xFF):
                        raise AssertionError(f"[KV][FAIL] L{L} slot{slot} pattern mismatch: head={list(head[:4])}")
            print(f"[KV] layer {L} read-back {slots_per_layer} slots (OK)")
        t_read = time.perf_counter() - t_read0
        r_gbps = (total_read / t_read) / (1024**3) if t_read > 0 else 0.0
        print(f"[KV] read total { _human(total_read) }, time {t_read:.3f}s, avg {r_gbps:.2f} GB/s "
              f"(stride bytes accounted)")

    # 资源回收
    del buf
    del kv


# ========== CLI ==========

def main():
    p = argparse.ArgumentParser(description="Sequential I/O tests for Weights (read-only) and KV (write+readback).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Weights
    pw = sub.add_parser("weights", help="weights read-only sequential test (RAW->pinned)")
    pw.add_argument("--manifest", required=True, help="path to runtime manifest (e.g., llama3.1-70b.runtime_manifest.json)")
    pw.add_argument("--layer-start", type=int, default=0)
    pw.add_argument("--layer-end", type=int, default=1, help="exclusive")
    pw.add_argument("--regex", default=None, help="regex to filter parameter names under a layer")
    pw.add_argument("--only-stream", action="store_true", help="only streamable params (recommended)")
    pw.add_argument("--verify-hash", action="store_true", help="compute md5 for each param (slow)")
    pw.add_argument("--staging-mb", type=int, default=32, help="staging buffer for unaligned reads")

    # KV
    pk = sub.add_parser("kv", help="kv write/readback sequential test (pinned-aligned fast path if possible)")
    pk.add_argument("--dev", required=True, help="NVMe block device or raw file path opened with O_DIRECT")
    pk.add_argument("--n-layers", type=int, required=True, help="backend total layers configured with the device layout")
    pk.add_argument("--blk-bytes", type=int, required=True, help="logical block bytes (effective data)")
    pk.add_argument("--blk-per-layer", type=int, required=True, help="slots per layer (e.g., max seq len)")
    pk.add_argument("--layer-start", type=int, default=0)
    pk.add_argument("--layer-end", type=int, default=1, help="exclusive")
    pk.add_argument("--slots-per-layer", type=int, default=8)
    pk.add_argument("--verify-readback", action="store_true")
    pk.add_argument("--sync-writes", action="store_true", help="fsync after each write (slow)")

    args = p.parse_args()

    if args.cmd == "weights":
        weights_read_sequential(
            manifest_path=args.manifest,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            param_regex=args.regex,
            only_stream=args.only_stream,
            verify_hash=args.verify_hash,
            staging_mb=args.staging_mb,
        )
    elif args.cmd == "kv":
        kv_rw_sequential(
            dev_path=args.dev,
            n_layers=args.n_layers,
            blk_bytes=args.blk_bytes,
            blk_per_layer=args.blk_per_layer,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            slots_per_layer=args.slots_per_layer,
            verify_readback=args.verify_readback,
            sync_writes=args.sync_writes,
        )
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
