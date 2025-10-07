# 全层 + 大矩阵（更快）
# python validate_all_layers_sequential.py --manifest /data1/llama3.1-70b.runtime_manifest.json --only-stream

# 或全层所有参数
# python validate_all_layers_sequential.py --manifest /data1/llama3.1-70b.runtime_manifest.json

import argparse
import ctypes
import gc
import json
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

from llama3.weights_io_ssd_dram import (
    DirectIOFile,
    alloc_pinned_aligned,
    DTYPE_MAP,  # {"float16": torch.float16, "bfloat16": torch.bfloat16, ...}
)

NP_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.int8:    np.int8,
    torch.uint8:   np.uint8,
}

def read_bytes_odirect(dio: DirectIOFile, want_off: int, want_len: int, bsz: int) -> bytes:
    """O_DIRECT 对齐读取任意 offset 的 want_len 字节。"""
    aligned_off = (want_off // bsz) * bsz
    delta = want_off - aligned_off
    need = delta + want_len
    need_aligned = ((need + bsz - 1) // bsz) * bsz
    buf = alloc_pinned_aligned(max(bsz, need_aligned), bsz)
    dio.pread_into_tensor(buf, buf.numel(), aligned_off)
    return bytes(buf[delta:delta + want_len].cpu().numpy())

def _detect_concat_axis(shapes: List[Tuple[int, ...]]) -> int:
    """
    在多组张量形状中，自动判断它们能否在某一个轴上拼接，如果可以，返回该轴索引，否则返回 -1
    """
    base = shapes[0]
    for ax in range(len(base)):
        same_other = True
        for s in shapes:
            if len(s) != len(base):
                return -1
            for j in range(len(base)):
                if j == ax:
                    continue
                if s[j] != base[j]:
                    same_other = False
                    break
            if not same_other:
                break
        if same_other:
            return ax
    return -1

def load_one_layer_to_pinned_merged(
    manifest: dict,
    layer_id: int,
    method: str = "bytecopy",   # "bytecopy"（通用/bf16友好）或 "reinterpret"（fp16/fp32）
    staging_mb: int = 16,
    only_stream: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    从 raw 读取指定层，按 name 合并多分片，返回 {name: CPU pinned tensor}。
    """
    raw_dev = manifest["raw_device"]
    bsz     = int(manifest["block_size"])
    params  = manifest["params"]

    entries = [p for p in params if int(p["layer"]) == int(layer_id)]
    if only_stream:
        entries = [p for p in entries if p.get("policy", "resident") == "stream"]
    if not entries:
        return {}

    groups: Dict[str, List[dict]] = defaultdict(list)
    for p in entries:
        groups[p["name"]].append(p)

    dio = DirectIOFile(raw_dev, mode="r", block_size=bsz)
    staging_bytes = max(1, int(staging_mb)) * 1024 * 1024
    staging_bytes = (staging_bytes // bsz) * bsz
    staging = alloc_pinned_aligned(max(bsz, staging_bytes), bsz)  # uint8 pinned

    out: Dict[str, torch.Tensor] = {}

    for name, segs in groups.items():
        # 基本信息
        base_dtype = DTYPE_MAP[segs[0]["dtype"]]
        elem_size  = torch.tensor([], dtype=base_dtype).element_size()
        shapes     = [tuple(s["shape"]) for s in segs]
        concat_ax  = _detect_concat_axis(shapes)

        total_nbytes = sum(int(s["nbytes"]) for s in segs)
        total_elems  = total_nbytes // elem_size

        if concat_ax == 0:
            base_shape = list(shapes[0])
            base_shape[0] = sum(s[0] for s in shapes)
            new_shape = tuple(base_shape)
            assert np.prod(new_shape) == total_elems, f"[{name}] shape/product mismatch after concat"
        else:
            # 其它轴或无法判断：退化 1D，数据仍正确
            new_shape = (total_elems,)

        dst = torch.empty(new_shape, dtype=base_dtype, pin_memory=True)

        cursor_bytes = 0
        cursor_elems = 0

        for seg in segs:
            offset = int(seg["offset"])
            stride = int(seg["stride"])
            nbytes = int(seg["nbytes"])

            if stride > staging.numel():
                staging = alloc_pinned_aligned(((stride + bsz - 1)//bsz)*bsz, bsz)

            nread = dio.pread_into_tensor(staging, stride, offset)
            if nread != stride:
                raise RuntimeError(f"Short read on {name}: got {nread}, expect {stride}")

            seg_elems = nbytes // elem_size

            if method == "reinterpret" and base_dtype in NP_DTYPE and concat_ax == 0:
                src_u8  = staging[:nbytes].cpu().numpy()
                np_view = src_u8.view(NP_DTYPE[base_dtype])[:seg_elems].reshape((seg["shape"][0],) + tuple(shapes[0][1:]))
                dst_slice = dst[cursor_elems : cursor_elems + seg_elems].reshape(np_view.shape)
                dst_slice.copy_(torch.from_numpy(np_view), non_blocking=False)
            else:
                # 纯字节复制（bf16 推荐）
                ctypes.memmove(dst.data_ptr() + cursor_bytes, staging.data_ptr(), nbytes)

            cursor_bytes += nbytes
            cursor_elems += seg_elems

        out[name] = dst

    dio.close()
    return out

# -------------------- 整层逐分片校验 --------------------
def sanity_check_whole_layer(
    pinned_dict: Dict[str, torch.Tensor],
    manifest: dict,
    layer_id: int,
    verbose: bool = False
) -> Tuple[int, int]:
    """
    对该层所有参数的每个分片逐一校验（前 64 字节）：返回 (matched_pieces, total_pieces)。
    """
    raw, bsz = manifest["raw_device"], int(manifest["block_size"])
    entries = [p for p in manifest["params"] if int(p["layer"]) == int(layer_id)]

    groups: Dict[str, List[dict]] = defaultdict(list)
    for p in entries:
        groups[p["name"]].append(p)

    dio = DirectIOFile(raw, mode="r", block_size=bsz)

    total_pieces = 0
    matched_pieces = 0

    for name, segs in groups.items():
        if name not in pinned_dict:
            # 可能因为 only_stream；跳过
            continue

        dst = pinned_dict[name]
        elem_size = dst.element_size()
        total_nbytes = dst.numel() * elem_size

        cursor_bytes = 0
        for si, seg in enumerate(segs):
            nbytes = int(seg["nbytes"])
            need = min(64, nbytes)
            src_head = read_bytes_odirect(dio, int(seg["offset"]), need, bsz)

            ok = False
            if cursor_bytes + need <= total_nbytes:
                dst_head = (ctypes.c_char * need).from_address(dst.data_ptr() + cursor_bytes).raw
                ok = (src_head == dst_head)

            if verbose:
                print(f"[L{layer_id}][{name}][piece {si}] {'OK' if ok else 'FAIL'}")

            matched_pieces += int(ok)
            total_pieces += 1
            cursor_bytes += nbytes

    dio.close()
    return matched_pieces, total_pieces

# -------------------- 主循环：一次只验证一个 layer --------------------
def main():
    ap = argparse.ArgumentParser(description="Sequentially verify each layer from SSD to pinned (avoid DRAM OOM).")
    ap.add_argument("--manifest", required=True, help="runtime manifest json path")
    ap.add_argument("--method", choices=["bytecopy", "reinterpret"], default="bytecopy",
                    help="bytecopy: universal (incl. bf16); reinterpret: fp16/fp32")
    ap.add_argument("--staging-mb", type=int, default=16, help="staging buffer size MiB")
    ap.add_argument("--only-stream", action="store_true", help="only verify stream (QKV/O, W1/2/3)")
    ap.add_argument("--start", type=int, default=None, help="start layer id (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="end layer id (inclusive)")
    ap.add_argument("--stop-on-fail", action="store_true", help="stop as soon as a mismatch appears")
    ap.add_argument("--verbose", action="store_true", help="print per-piece result")
    args = ap.parse_args()

    # 读取 manifest；找出层数范围
    manifest = json.load(open(args.manifest, "r"))
    layers = sorted({int(p["layer"]) for p in manifest["params"]})
    if not layers:
        print("No params in manifest"); return
    min_layer, max_layer = min(layers), max(layers)

    L0 = args.start if args.start is not None else min_layer
    L1 = args.end   if args.end   is not None else max_layer

    total_all = 0
    matched_all = 0

    for L in range(L0, L1 + 1):
        # 逐层：加载 → 校验 → 释放
        tensors = load_one_layer_to_pinned_merged(
            manifest=manifest,
            layer_id=L,
            method=args.method,
            staging_mb=args.staging_mb,
            only_stream=args.only_stream,
        )

        if not tensors:
            print(f"[Layer {L}] (skipped: no entries under filter)")
            continue

        matched, total = sanity_check_whole_layer(
            tensors, manifest, layer_id=L, verbose=args.verbose
        )
        total_all += total
        matched_all += matched

        ok_rate = (matched / total * 100) if total else 100.0
        print(f"[Layer {L}] segment matches: {matched}/{total} ({ok_rate:.2f}%)")

        # 释放当前层内存
        del tensors
        gc.collect()

        if args.stop_on_fail and matched < total:
            print("[STOP] mismatch detected, stopping early.")
            break

    print(f"[SUMMARY] total segment matches: {matched_all}/{total_all} "
          f"({(matched_all/total_all*100 if total_all else 100):.2f}%)")

if __name__ == "__main__":
    main()
