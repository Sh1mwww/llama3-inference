# diagnose_device_issue.py
import argparse
import json
import ctypes
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

# 按你的工程路径导入（你说文件在 llama3/weights_io_ssd_dram.py）
from llama3.weights_io_ssd_dram import (
    DirectIOFile,
    alloc_pinned_aligned,
    DTYPE_MAP,  # {"float16": torch.float16, "bfloat16": torch.bfloat16, ...}
)

# numpy 能做 reinterpret 的 dtype（注意：不含 bfloat16）
NP_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.int8:    np.int8,
    torch.uint8:   np.uint8,
}

# -------------------- 低层工具 --------------------

def hexdump(b: bytes, n=32) -> str:
    return " ".join(f"{x:02x}" for x in b[:n])

def memcpy_bytes(dst_tensor: torch.Tensor, src_u8_tensor: torch.Tensor, nbytes: int):
    """
    将 src_u8_tensor 的前 nbytes 字节按原样拷贝到 dst_tensor 对应内存（CPU↔CPU）。
    src_u8_tensor 必须是 pinned 的 uint8；dst dtype/shape 任意（按字节覆盖）。
    """
    assert dst_tensor.device.type == "cpu"
    assert src_u8_tensor.device.type == "cpu"
    assert src_u8_tensor.dtype == torch.uint8
    ctypes.memmove(dst_tensor.data_ptr(), src_u8_tensor.data_ptr(), nbytes)

def read_bytes_odirect(dio: DirectIOFile, want_off: int, want_len: int, bsz: int) -> bytes:
    """
    从任意 offset（可不对齐）读取 want_len 字节：
    - 实际以 O_DIRECT 的对齐要求向下对齐到 aligned_off
    - 读取足够覆盖 [want_off, want_off+want_len) 的对齐长度
    - 返回中间切片
    """
    aligned_off = (want_off // bsz) * bsz
    delta = want_off - aligned_off
    need = delta + want_len
    need_aligned = ((need + bsz - 1) // bsz) * bsz
    buf = alloc_pinned_aligned(max(bsz, need_aligned), bsz)
    dio.pread_into_tensor(buf, buf.numel(), aligned_off)
    return bytes(buf[delta:delta + want_len].cpu().numpy())

# -------------------- 合并分片加载（核心） --------------------

def _detect_concat_axis(shapes: List[Tuple[int, ...]]) -> int:
    """
    自动判定拼接轴：若仅有一个轴长度不同，其余轴完全一致，则返回该轴索引；否则返回 -1。
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
    manifest_path: str,
    layer_id: int,
    method: str = "bytecopy",   # "bytecopy"（通用）或 "reinterpret"（fp16/fp32可用）
    staging_mb: int = 16,
    only_stream: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    从 raw block device 读取指定层的所有参数，并把“同名多分片”**字节级拼接**成完整 pinned 张量（仍在 CPU）。
    返回 {param_name: pinned_tensor}。
    """
    # 1) 读取 manifest
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    raw_dev = manifest["raw_device"]
    bsz     = int(manifest["block_size"])
    params  = manifest["params"]

    # 2) 过滤该层条目（可选只加载 stream 大矩阵）
    entries = [p for p in params if int(p["layer"]) == int(layer_id)]
    if only_stream:
        entries = [p for p in entries if p.get("policy", "resident") == "stream"]
    if not entries:
        raise ValueError(f"No params for layer {layer_id} (only_stream={only_stream})")

    # 3) 按 name 分组，保留 manifest 顺序（这就是 pack 顺序）
    groups: Dict[str, List[dict]] = defaultdict(list)
    for p in entries:
        groups[p["name"]].append(p)

    # 4) 打开原始设备 + staging
    dio = DirectIOFile(raw_dev, mode="r", block_size=bsz)
    staging_bytes = max(1, int(staging_mb)) * 1024 * 1024
    staging_bytes = (staging_bytes // bsz) * bsz
    staging = alloc_pinned_aligned(max(bsz, staging_bytes), bsz)  # uint8 pinned

    out: Dict[str, torch.Tensor] = {}

    for name, segs in groups.items():
        # 按 manifest 顺序拼接
        base_dtype = DTYPE_MAP[segs[0]["dtype"]]
        elem_size  = torch.tensor([], dtype=base_dtype).element_size()
        shapes     = [tuple(s["shape"]) for s in segs]
        concat_ax  = _detect_concat_axis(shapes)  # -1 表示不满足“仅一个轴不同”的简单情形

        total_nbytes = sum(int(s["nbytes"]) for s in segs)
        total_elems  = total_nbytes // elem_size

        if concat_ax == 0:
            # 典型 Llama 分片：在 dim=0 拼接
            base_shape = list(shapes[0])
            base_shape[0] = sum(s[0] for s in shapes)
            new_shape = tuple(base_shape)
            assert np.prod(new_shape) == total_elems, f"[{name}] shape/product mismatch after concat"
        elif concat_ax == -1:
            # fallback：无法确定拼接轴，使用 1D 形状（数据正确，形状退化）
            new_shape = (total_elems,)
        else:
            # 其它轴分片（较少见）：也退化为 1D，数据仍然正确
            new_shape = (total_elems,)

        # 目标 pinned 张量
        dst = torch.empty(new_shape, dtype=base_dtype, pin_memory=True)

        cursor_bytes = 0
        cursor_elems = 0

        for seg in segs:
            offset = int(seg["offset"])
            stride = int(seg["stride"])
            nbytes = int(seg["nbytes"])

            if stride > staging.numel():
                staging = alloc_pinned_aligned(((stride + bsz - 1)//bsz)*bsz, bsz)

            # SSD -> staging（对齐读）
            nread = dio.pread_into_tensor(staging, stride, offset)
            if nread != stride:
                raise RuntimeError(f"Short read on {name}: got {nread}, expect {stride}")

            seg_elems = nbytes // elem_size

            # 写入目标（优先尝试 reinterpret；bf16 或不支持时自动走 bytecopy）
            if method == "reinterpret" and base_dtype in NP_DTYPE and concat_ax == 0:
                # 在“源”staging 上做 numpy reinterpret（零拷贝视图），再一次 copy_ 到目标切片
                src_u8  = staging[:nbytes].cpu().numpy()
                np_view = src_u8.view(NP_DTYPE[base_dtype])[:seg_elems].reshape((seg["shape"][0],) + tuple(shapes[0][1:]))
                # 目标切片：dim=0 的 [cursor_elems : cursor_elems + seg_elems]，再按原形状 reshape
                dst_slice = dst[cursor_elems : cursor_elems + seg_elems].reshape(np_view.shape)
                dst_slice.copy_(torch.from_numpy(np_view), non_blocking=False)
            else:
                # 纯字节复制（dtype 无关，bf16 强烈推荐）
                ctypes.memmove(dst.data_ptr() + cursor_bytes, staging.data_ptr(), nbytes)

            cursor_bytes += nbytes
            cursor_elems += seg_elems

        out[name] = dst

    dio.close()
    return out

# -------------------- 逐片段 SANITY 校验 --------------------

def sanity_check_per_segment(pinned_dict: Dict[str, torch.Tensor], manifest_path: str, layer_id: int, sample_k: int = 3):
    """
    抽样若干参数；对每个参数的每个“分片”，比较 SSD 与 pinned 的“前 64 字节”。
    """
    m = json.load(open(manifest_path, "r"))
    raw, bsz = m["raw_device"], int(m["block_size"])
    entries = [p for p in m["params"] if int(p["layer"]) == int(layer_id)]

    groups: Dict[str, List[dict]] = defaultdict(list)
    for p in entries:
        groups[p["name"]].append(p)

    names = list(groups.keys())
    random.seed(0)
    picks = random.sample(names, k=min(sample_k, len(names)))

    dio = DirectIOFile(raw, mode="r", block_size=bsz)

    ok_total, seg_total = 0, 0
    for name in picks:
        if name not in pinned_dict:
            print(f"[SANITY] skip {name}: not loaded")
            continue

        dst = pinned_dict[name]
        elem_size = dst.element_size()
        total_nbytes = dst.numel() * elem_size

        # 顺序遍历每个分片，在目标张量里按“字节偏移”逐段对齐
        cursor_bytes = 0
        for seg in groups[name]:
            nbytes = int(seg["nbytes"])
            # 源头：分片的前 64 字节
            need = min(64, nbytes)
            src_head = read_bytes_odirect(dio, int(seg["offset"]), need, bsz)

            # 目标：该分片在 dst 中的前 64 字节
            if cursor_bytes + need > total_nbytes:
                print(f"[SANITY] {name} cursor beyond dst bytes"); break
            # 直接从 dst.data_ptr() + cursor_bytes 拿 need 字节
            dst_head = (ctypes.c_char * need).from_address(dst.data_ptr() + cursor_bytes).raw

            match = (src_head == dst_head)
            print(f"[SEG] {name} piece head match? {match}")
            ok_total += int(match); seg_total += 1

            cursor_bytes += nbytes

    dio.close()
    print(f"[SANITY] segment matches: {ok_total}/{seg_total}")
# === 新增：整层全量校验 ===
def sanity_check_whole_layer(pinned_dict: Dict[str, torch.Tensor], manifest_path: str, layer_id: int):
    """
    对该层所有参数的每个分片逐一校验：SSD 原始数据 vs pinned 目标张量（前 64 字节）。
    打印汇总统计和首个失败样例（若有）。
    """
    m = json.load(open(manifest_path, "r"))
    raw, bsz = m["raw_device"], int(m["block_size"])
    entries = [p for p in m["params"] if int(p["layer"]) == int(layer_id)]

    groups: Dict[str, List[dict]] = defaultdict(list)
    for p in entries:
        groups[p["name"]].append(p)

    dio = DirectIOFile(raw, mode="r", block_size=bsz)

    total_pieces = 0
    matched_pieces = 0
    failed_examples = []  # (name, seg_idx)

    for name, segs in groups.items():
        if name not in pinned_dict:
            # 没加载到（比如 --only-stream 时），跳过但记录
            continue

        dst = pinned_dict[name]
        elem_size = dst.element_size()
        total_nbytes = dst.numel() * elem_size

        cursor_bytes = 0
        for si, seg in enumerate(segs):
            nbytes = int(seg["nbytes"])
            need = min(64, nbytes)
            src_head = read_bytes_odirect(dio, int(seg["offset"]), need, bsz)

            if cursor_bytes + need > total_nbytes:
                failed_examples.append((name, si))
                total_pieces += 1
                cursor_bytes += nbytes
                continue

            dst_head = (ctypes.c_char * need).from_address(dst.data_ptr() + cursor_bytes).raw
            if src_head == dst_head:
                matched_pieces += 1
            else:
                failed_examples.append((name, si))
            total_pieces += 1
            cursor_bytes += nbytes

    dio.close()
    print(f"[ALL] segment matches: {matched_pieces}/{total_pieces} "
          f"({matched_pieces/total_pieces*100:.2f}% )")
    if failed_examples:
        name, si = failed_examples[0]
        print(f"[ALL] first mismatch: {name} (piece #{si})")

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="Load one layer from SSD to pinned memory (merge sharded pieces).")
    ap.add_argument("--manifest", required=True, help="runtime manifest json path")
    ap.add_argument("--layer", type=int, default=0, help="layer id to load")
    ap.add_argument("--method", choices=["bytecopy", "reinterpret"], default="bytecopy")
    ap.add_argument("--staging-mb", type=int, default=16)
    ap.add_argument("--only-stream", action="store_true")
    ap.add_argument("--samples", type=int, default=3, help="sample count for sanity (ignored if --all)")
    ap.add_argument("--all", action="store_true", help="sanity-check the whole layer (every param, every piece)")
    args = ap.parse_args()

    tensors = load_one_layer_to_pinned_merged(
        manifest_path=args.manifest,
        layer_id=args.layer,
        method=args.method,
        staging_mb=args.staging_mb,
        only_stream=args.only_stream,
    )

    print(f"[OK] loaded {len(tensors)} tensors for layer {args.layer} via {args.method}")
    any_name = next(iter(tensors))
    t = tensors[any_name]
    print("  sample:", any_name, "|", t.dtype, "|", tuple(t.shape), "| pinned:", t.is_pinned())

    if args.all:
        sanity_check_whole_layer(tensors, args.manifest, args.layer)
    else:
        sanity_check_per_segment(tensors, args.manifest, args.layer, sample_k=args.samples)


if __name__ == "__main__":
    main()
