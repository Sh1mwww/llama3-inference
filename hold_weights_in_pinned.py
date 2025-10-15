#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把指定层的权重连续读入 Host pinned 内存，并保持一段时间（默认 5s）后释放。
用途：肉眼观察 htop 的 Lck（Locked）或 /proc/<pid>/status 的 VmLck 变化。
"""

import os
import re
import time
import argparse
import torch
from typing import Optional, Dict, List

from llama3.raw_param_store import ParamStore


def vmlck_kb() -> int:
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmLck:"):
                return int(line.split()[1])  # kB
    return 0


def sum_param_total_nbytes(store: ParamStore, layer_id: int, name: str) -> int:
    # 若你有 param_total_nbytes() 就用，没有就汇总分片
    if hasattr(store, "param_total_nbytes"):
        return int(store.param_total_nbytes(layer_id, name))
    groups = store._groups_for_layer(layer_id, only_stream=False, names=[name])  # noqa: 内部工具
    segs = groups.get(name, [])
    if not segs:
        raise KeyError(f"param not found: L{layer_id}:{name}")
    return sum(int(s["nbytes"]) for s in segs)


def hold_layers_in_pinned(
    manifest: str,
    layer_start: int,
    layer_end: int,
    regex: Optional[str],
    only_stream: bool,
    hold_seconds: float,
    staging_mb: int = 32,
) -> None:
    print(f"[INFO] manifest={manifest}")
    print(f"[INFO] layers=[{layer_start}..{layer_end-1}], only_stream={only_stream}, regex={regex}")
    print(f"[INFO] ulimit -l (memlock) = {os.popen('ulimit -l').read().strip() or '(unknown)'}")

    store = ParamStore(manifest, method="bytecopy", staging_mb=staging_mb)
    pinned_holds: Dict[tuple, torch.Tensor] = {}  # {(L,name): pinned_u8}
    name_pat = re.compile(regex) if regex else None

    total_bytes = 0
    try:
        print(f"[VmLck] before: {vmlck_kb():,} kB")
        for L in range(layer_start, layer_end):
            groups = store._groups_for_layer(L, only_stream=only_stream, names=None)  # name -> [segments...]
            names = sorted(groups.keys())
            if name_pat:
                names = [nm for nm in names if name_pat.search(nm)]
            if not names:
                print(f"[L{L}] no params matched.")
                continue

            print(f"[L{L}] loading {len(names)} params into pinned ...")
            for name in names:
                nbytes = sum_param_total_nbytes(store, L, name)
                dst_u8 = torch.empty(int(nbytes), dtype=torch.uint8, pin_memory=True).contiguous()
                assert dst_u8.is_pinned() and dst_u8.is_contiguous()
                # 读取到 pinned（对齐=READ_FIXED，不对齐=staging+memmove；函数内部处理）
                store.fetch_param_into_pinned_u8(L, name, dst_u8)
                pinned_holds[(L, name)] = dst_u8  # 保持引用，防 GC
                total_bytes += nbytes
            print(f"[L{L}] pinned hold ok: {len(names)} params")

        print(f"[VmLck] after alloc: {vmlck_kb():,} kB   (sleep {hold_seconds:.1f}s; 可观察 htop 的 Lck 列)")
        time.sleep(hold_seconds)

    finally:
        # 释放引用
        n_tensors = len(pinned_holds)
        pinned_holds.clear()
        # 注：PyTorch 会缓存 pinned 区，VmLck 可能不会立即降为 0 —— 这属正常现象
        print(f"[INFO] released {n_tensors} pinned buffers; note: pinned cache may keep pages locked.")
        print(f"[VmLck] after free: {vmlck_kb():,} kB")
        store.close()


def main():
    ap = argparse.ArgumentParser(description="Hold specified layers' weights in pinned memory for a while, then free.")
    ap.add_argument("--manifest", required=True, help="path to runtime manifest (e.g., llama3.1-70b.runtime_manifest.json)")
    ap.add_argument("--layer-start", type=int, required=True)
    ap.add_argument("--layer-end", type=int, required=True, help="exclusive")
    ap.add_argument("--regex", default=None, help="filter param names by regex (e.g. 'layers\\.5\\.attention\\.(wq|wk)\\.weight')")
    ap.add_argument("--only-stream", action="store_true", help="only streamable params")
    ap.add_argument("--hold-seconds", type=float, default=5.0)
    ap.add_argument("--staging-mb", type=int, default=32, help="staging buffer size for unaligned reads")
    args = ap.parse_args()

    hold_layers_in_pinned(
        manifest=args.manifest,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        regex=args.regex,
        only_stream=args.only_stream,
        hold_seconds=args.hold_seconds,
        staging_mb=args.staging_mb,
    )


if __name__ == "__main__":
    main()
