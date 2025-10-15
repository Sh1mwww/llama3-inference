#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把指定层/slots 的 KV block 连续写入设备，同时把每个 block 的 pinned 缓冲保持一段时间（默认 5s）后释放。
用途：肉眼观察 htop 的 Lck（Locked）或 /proc/<pid>/status 的 VmLck 变化，确认 pinned 生效。
"""

import os
import time
import argparse
from typing import Dict, Tuple

import torch
from llama3.SSDBacked import RawBlockKVBackend  # 你的工程里的 KV 后端

def vmlck_kb() -> int:
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmLck:"):
                return int(line.split()[1])  # kB
    return 0

def human(nbytes: float) -> str:
    units = ["B","KB","MB","GB","TB"]
    i, x = 0, float(nbytes)
    while x >= 1024 and i < len(units)-1:
        x /= 1024.0; i += 1
    return f"{x:.2f} {units[i]}"

def hold_kv_in_pinned(
    dev: str,
    n_layers: int,
    blk_bytes: int,
    blk_per_layer: int,
    layer_start: int,
    layer_end: int,
    slots_per_layer: int,
    hold_seconds: float,
    verify_readback: bool,
    sync_writes: bool,
):
    print(f"[INFO] dev={dev}  n_layers={n_layers}  blk_bytes={blk_bytes}  blk_per_layer={blk_per_layer}")
    print(f"[INFO] layers=[{layer_start}..{layer_end-1}]  slots_per_layer={slots_per_layer}")
    print(f"[INFO] ulimit -l (memlock) = {os.popen('ulimit -l').read().strip() or '(unknown)'}")

    kv = RawBlockKVBackend(dev, n_layers=n_layers, blk_bytes=blk_bytes, blk_per_layer=blk_per_layer)
    stride = kv.stride
    print(f"[INFO] stride (physical per block) = {stride} bytes  ({human(stride)})")

    # 为每个 (L,slot) 准备一个 pinned 缓冲并写入设备；保持引用以防 GC
    held: Dict[Tuple[int,int], torch.Tensor] = {}

    print(f"[VmLck] before: {vmlck_kb():,} kB")
    total_bytes = 0

    try:
        for L in range(layer_start, layer_end):
            print(f"[L{L}] allocating & writing {slots_per_layer} slots ...")
            for slot in range(slots_per_layer):
                buf = torch.empty(stride, dtype=torch.uint8, pin_memory=True).contiguous()
                # 写入内容模式（前 blk_bytes 部分写入签名，padding 保持 0）
                buf.zero_()
                head = min(16, blk_bytes)
                buf[:head] = torch.tensor(
                    [(L & 0xFF), (slot & 0xFF)] + [i & 0xFF for i in range(14)],
                    dtype=torch.uint8
                )[:head]

                # 写入设备（优先 preadv/pwritev 的直达 pinned；否则自动回退）
                kv.write_from_pinned_aligned(L, slot, buf, sync=sync_writes)
                total_bytes += stride

                if verify_readback:
                    # 读回到同一缓冲区的尾部（或直接覆盖），做一个轻量校验
                    kv.read_into_pinned_aligned(L, slot, buf)
                    if head >= 2:
                        if int(buf[0].item()) != (L & 0xFF) or int(buf[1].item()) != (slot & 0xFF):
                            raise AssertionError(f"[KV][FAIL] L{L} slot{slot} pattern mismatch: "
                                                 f"{int(buf[0].item())}, {int(buf[1].item())}")

                # 关键：把 pinned 缓冲放到 held 里保持引用，防止被释放
                held[(L, slot)] = buf

            print(f"[L{L}] done.")

        print(f"[VmLck] after alloc: {vmlck_kb():,} kB  (hold {hold_seconds:.1f}s；此时看 htop 的 Lck 列)")
        time.sleep(hold_seconds)

    finally:
        n_bufs = len(held)
        held.clear()            # 释放引用（注意：pinned 可能有 allocator 缓存，不一定立刻降为 0）
        print(f"[INFO] released {n_bufs} pinned buffers; note: PyTorch may cache pinned pages.")
        print(f"[VmLck] after free: {vmlck_kb():,} kB")
        del kv                 # 关闭 fd / 线程池

def main():
    ap = argparse.ArgumentParser(description="Hold KV blocks in pinned memory for a while, then free.")
    ap.add_argument("--dev", required=True, help="NVMe block device or raw file path opened with O_DIRECT")
    ap.add_argument("--n-layers", type=int, required=True)
    ap.add_argument("--blk-bytes", type=int, required=True, help="logical block bytes (effective data per slot)")
    ap.add_argument("--blk-per-layer", type=int, required=True, help="slots per layer capacity in backend")
    ap.add_argument("--layer-start", type=int, required=True)
    ap.add_argument("--layer-end", type=int, required=True, help="exclusive")
    ap.add_argument("--slots-per-layer", type=int, default=8)
    ap.add_argument("--hold-seconds", type=float, default=5.0)
    ap.add_argument("--verify-readback", action="store_true", help="read back & check first bytes")
    ap.add_argument("--sync-writes", action="store_true", help="fsync after each write (slow)")
    args = ap.parse_args()

    hold_kv_in_pinned(
        dev=args.dev,
        n_layers=args.n_layers,
        blk_bytes=args.blk_bytes,
        blk_per_layer=args.blk_per_layer,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        slots_per_layer=args.slots_per_layer,
        hold_seconds=args.hold_seconds,
        verify_readback=args.verify_readback,
        sync_writes=args.sync_writes,
    )

if __name__ == "__main__":
    main()
