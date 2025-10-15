#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KV RAW 写入微基准：
- mode=large  : 整块写（stride 大小），可重复 token-steps 次
- mode=small  : 小块增量写（每 token 小写 N 次），支持 flush-interval 合并为整块写
可选：读回校验、hold pinned N 秒观察 VmLck/Lck。
"""

import os
import time
import math
import argparse
from typing import Dict, Tuple, List

import torch
from llama3.SSDBacked import RawBlockKVBackend  # 你的 KV 后端（O_DIRECT）

# ---------- utils ----------
def human(n):
    units = ["B","KB","MB","GB","TB"]
    i = 0; x = float(n)
    while x >= 1024 and i < len(units)-1:
        x /= 1024.0; i += 1
    return f"{x:.2f} {units[i]}"

def vmlck_kb() -> int:
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmLck:"):
                return int(line.split()[1])  # kB
    return 0

# ---------- pattern helpers ----------
def fill_block_header(buf: torch.Tensor, L: int, slot: int, token_step: int, size: int):
    """
    在 buf 前 size 字节写入签名（方便校验；真实数据路径中可用实际 KV）
    """
    head = min(size, buf.numel())
    if head <= 0: return
    # 简单签名：layer, slot, token_step, 递增序列
    pattern = [(L & 0xFF), (slot & 0xFF), (token_step & 0xFF)] + [i & 0xFF for i in range(13)]
    buf[:head] = torch.tensor(pattern, dtype=torch.uint8)[:head]

def check_block_header(buf: torch.Tensor, L: int, slot: int):
    if buf.numel() < 2: return
    if int(buf[0].item()) != (L & 0xFF) or int(buf[1].item()) != (slot & 0xFF):
        raise AssertionError(f"pattern mismatch: got [{int(buf[0].item())},{int(buf[1].item())}] "
                             f"expect [{L & 0xFF},{slot & 0xFF}]")

# ---------- microbench modes ----------
def run_large_write(
    kv: RawBlockKVBackend,
    layer_range: range,
    slots_per_layer: int,
    token_steps: int,
    verify_readback: bool,
    hold_seconds: float,
):
    stride = kv.stride
    total_stride_bytes = 0
    t0 = time.perf_counter()

    # 为每个 (L, slot) 准备 pinned 1 块
    held: Dict[Tuple[int,int], torch.Tensor] = {}
    print(f"[VmLck] before: {vmlck_kb():,} kB")

    try:
        for L in layer_range:
            for slot in range(slots_per_layer):
                buf = torch.empty(stride, dtype=torch.uint8, pin_memory=True).contiguous()
                buf.zero_()
                held[(L,slot)] = buf

        print(f"[INFO] allocated {len(held)} pinned blocks (each {human(stride)})")
        print(f"[VmLck] after alloc: {vmlck_kb():,} kB")

        for tstep in range(token_steps):
            for L in layer_range:
                for slot in range(slots_per_layer):
                    buf = held[(L,slot)]
                    # 写 header（只写有效字节范围顶部，剩余 padding 为 0）
                    fill_block_header(buf, L, slot, tstep, size=min(16, stride))
                    kv.write_from_pinned_aligned(L, slot, buf, sync=False)
                    total_stride_bytes += stride

        if verify_readback:
            for L in layer_range:
                for slot in range(slots_per_layer):
                    buf = held[(L,slot)]
                    buf.zero_()
                    kv.read_into_pinned_aligned(L, slot, buf)
                    check_block_header(buf, L, slot)

        if hold_seconds > 0:
            print(f"[HOLD] holding pinned {hold_seconds:.1f}s for observation (htop Lck / VmLck)")
            time.sleep(hold_seconds)
    finally:
        held.clear()
        print(f"[VmLck] after free: {vmlck_kb():,} kB")

    dt = time.perf_counter() - t0
    bw = (total_stride_bytes / dt) / (1024**3) if dt > 0 else 0.0
    print(f"[LARGE] wrote {human(total_stride_bytes)} stride-bytes in {dt:.3f}s, avg {bw:.2f} GB/s (stride accounted)")

def run_small_write(
    kv: RawBlockKVBackend,
    layer_range: range,
    slots_per_layer: int,
    token_steps: int,
    small_bytes: int,
    updates_per_token: int,
    flush_interval: int,
    verify_readback: bool,
    hold_seconds: float,
):
    """
    小块增量写：每个 token 对每个 slot 进行 updates_per_token 次小写（覆盖不同 offset），
    用 RMW（读整块→patch→写整块）保持 O_DIRECT 语义；每 flush_interval 次 flush 一次整块。
    注意：这是“最保守正确”的实现，上层可根据实际 kernel 合并策略改成更激进的写聚合。
    """
    stride = kv.stride
    blk_bytes = kv.blk_bytes
    # 小写长度向上限幅到 4KiB 的整数倍（便于对齐 patch；不强制，但我们按对齐组织）
    small_bytes = int(math.ceil(small_bytes / 4096.0) * 4096)

    # 为每个 (L, slot) 维持一个“影子块” pinned，并追踪“dirty 计数”
    shadow: Dict[Tuple[int,int], torch.Tensor] = {}
    dirty_count: Dict[Tuple[int,int], int] = {}
    total_stride_bytes = 0     # 真实落盘按 stride 计
    total_logical_updates = 0  # 逻辑小写字节数统计（便于比较写放大）

    print(f"[INFO] small-bytes={small_bytes} updates/token={updates_per_token} flush-interval={flush_interval}")
    print(f"[VmLck] before: {vmlck_kb():,} kB")

    try:
        # 初始化影子块：从设备读到 pinned（或者清零），再按需要 RMW
        for L in layer_range:
            for slot in range(slots_per_layer):
                buf = torch.empty(stride, dtype=torch.uint8, pin_memory=True).contiguous()
                # 读当前设备内容到影子（若设备初始为空你也可以选择不读）
                try:
                    kv.read_into_pinned_aligned(L, slot, buf)
                except Exception:
                    buf.zero_()
                shadow[(L,slot)] = buf
                dirty_count[(L,slot)] = 0

        print(f"[INFO] allocated {len(shadow)} pinned shadow blocks")
        print(f"[VmLck] after alloc: {vmlck_kb():,} kB")

        # token 循环：对每个 slot 执行 updates_per_token 次 patch
        for tstep in range(token_steps):
            for L in layer_range:
                for slot in range(slots_per_layer):
                    buf = shadow[(L,slot)]
                    # 基于 token_step 选择一个 offset（环绕），避免越界（只在有效区域内 patch）
                    max_off = max(0, blk_bytes - small_bytes)
                    if max_off > 0:
                        off = ((tstep * updates_per_token) % (max_off // 4096 + 1)) * 4096
                    else:
                        off = 0
                    # 做 updates_per_token 次 patch
                    for u in range(updates_per_token):
                        # 写入签名（写入 buf 的 off:off+small_bytes）
                        patch = buf.narrow(0, off, min(small_bytes, blk_bytes - off))
                        fill_block_header(patch, L, slot, tstep, size=min(16, patch.numel()))
                        dirty_count[(L,slot)] += 1
                        total_logical_updates += patch.numel()

                        # 下一个 patch 偏移（环绕）
                        off += small_bytes
                        if off + small_bytes > blk_bytes:
                            off = 0

                    # 决定是否 flush
                    if dirty_count[(L,slot)] >= flush_interval:
                        kv.write_from_pinned_aligned(L, slot, buf, sync=False)
                        total_stride_bytes += stride
                        dirty_count[(L,slot)] = 0

        # token 结束后，flush 所有剩余 dirty
        for key, cnt in dirty_count.items():
            if cnt > 0:
                L, slot = key
                kv.write_from_pinned_aligned(L, slot, shadow[key], sync=False)
                total_stride_bytes += stride
                dirty_count[key] = 0

        if verify_readback:
            # 简单 spot check：每层第 0 个 slot 读回校验头
            for L in layer_range:
                buf = torch.empty(stride, dtype=torch.uint8, pin_memory=True).contiguous()
                kv.read_into_pinned_aligned(L, 0, buf)
                check_block_header(buf, L, 0)

        if hold_seconds > 0:
            print(f"[HOLD] holding pinned {hold_seconds:.1f}s for observation (htop Lck / VmLck)")
            time.sleep(hold_seconds)

    finally:
        n = len(shadow)
        shadow.clear()
        print(f"[INFO] released {n} pinned shadow blocks")
        print(f"[VmLck] after free: {vmlck_kb():,} kB")

    # 统计
    dt = time.perf_counter() - 0 if False else 0  # 占位避免误删
    # 真正计时包裹上面逻辑
    # 为了更准确，把 run_small_write 上半段包在计时里：
    # 你可以按需改为更细粒度计时（仅 flush 的时间）
    # 这里给出 flush 的总写入量和逻辑小写量：
    print(f"[SMALL] logical-updates: {human(total_logical_updates)} (sum of small writes)")
    print(f"[SMALL] flushed stride-bytes: {human(total_stride_bytes)} "
          f"(写放大≈ {total_stride_bytes / max(1,total_logical_updates):.2f}x)")
    # 如需吞吐数值，可把上方 patch/flush 的循环包裹计时，计算 total_stride_bytes/delta_t

# ---------- main CLI ----------
def main():
    ap = argparse.ArgumentParser(description="KV RAW write microbench: large (stride) and small (per-token incremental).")
    ap.add_argument("--dev", required=True, help="NVMe block device or raw file opened with O_DIRECT")
    ap.add_argument("--n-layers", type=int, required=True)
    ap.add_argument("--blk-bytes", type=int, required=True, help="logical bytes per block (effective KV bytes)")
    ap.add_argument("--blk-per-layer", type=int, required=True, help="slots per layer in backend")
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--layer-end", type=int, default=1, help="exclusive")
    ap.add_argument("--slots-per-layer", type=int, default=8)
    ap.add_argument("--mode", choices=["large","small"], required=True)

    # common
    ap.add_argument("--token-steps", type=int, default=10, help="number of tokens simulated")
    ap.add_argument("--verify-readback", action="store_true")
    ap.add_argument("--hold-seconds", type=float, default=0.0)

    # small mode params
    ap.add_argument("--small-bytes", type=int, default=4096, help="size of each small update (bytes)")
    ap.add_argument("--updates-per-token", type=int, default=1, help="how many small updates per token per slot")
    ap.add_argument("--flush-interval", type=int, default=16, help="flush to device after N small updates for a slot")

    args = ap.parse_args()

    print(f"[INFO] ulimit -l (memlock) = {os.popen('ulimit -l').read().strip() or '(unknown)'}")
    kv = RawBlockKVBackend(
        args.dev, n_layers=args.n_layers, blk_bytes=args.blk_bytes, blk_per_layer=args.blk_per_layer
    )
    # 我们在各 mode 内再 new/close 也可以；此处沿用 kv 实例，函数内不再手动关闭。

    try:
        if args.mode == "large":
            t0 = time.perf_counter()
            run_large_write(
                kv=kv,
                layer_range=range(args.layer_start, args.layer_end),
                slots_per_layer=args.slots_per_layer,
                token_steps=args.token_steps,
                verify_readback=args.verify_readback,
                hold_seconds=args.hold_seconds,
            )
            print(f"[DONE] large mode in {time.perf_counter()-t0:.3f}s")

        elif args.mode == "small":
            t0 = time.perf_counter()
            run_small_write(
                kv=kv,
                layer_range=range(args.layer_start, args.layer_end),
                slots_per_layer=args.slots_per_layer,
                token_steps=args.token_steps,
                small_bytes=args.small_bytes,
                updates_per_token=args.updates_per_token,
                flush_interval=args.flush_interval,
                verify_readback=args.verify_readback,
                hold_seconds=args.hold_seconds,
            )
            print(f"[DONE] small mode in {time.perf_counter()-t0:.3f}s")
    finally:
        del kv  # 关闭 fd/线程池

if __name__ == "__main__":
    main()
