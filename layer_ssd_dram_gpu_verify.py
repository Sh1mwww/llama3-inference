# layer_ssd_dram_gpu_verify.py
# 验证全部层（resident+stream 全部），一次只占一层的 DRAM：
# python layer_ssd_dram_gpu_verify.py --manifest /data1/llama-70b.runtime_manifest.json
# 只验证大矩阵（Q/K/V/O、W1/2/3）：
# python layer_ssd_dram_gpu_verify.py --manifest /data1/llama-70b.runtime_manifest.json --only-stream
# 指定层范围 & 出错即停：
# python layer_ssd_dram_gpu_verify.py --manifest /data1/llama-70b.runtime_manifest.json --start 0 --end 3 --stop-on-fail
# 打印每个分片的结果（冗长）：
# python layer_ssd_dram_gpu_verify.py --manifest /data1/llama-70b.runtime_manifest.json --verbose
# 顺带做 GPU 计算通路验证（每层挑最多 4 个 2D 权重做一次 matmul，容差自适应）：
# python layer_ssd_dram_gpu_verify.py --manifest /data1/llama-70b.runtime_manifest.json --gpu-check

import argparse
import ctypes
import gc
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

# 依赖你项目内的工具路径
from llama3.weights_io_ssd_dram import (
    DirectIOFile,
    alloc_pinned_aligned,
    DTYPE_MAP,  # {"float16": torch.float16, "bfloat16": torch.bfloat16, ...}
)

# ========== numpy 可 reinterpret 的 dtype（bf16 不支持，统一走 bytecopy） ==========
NP_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.int8:    np.int8,
    torch.uint8:   np.uint8,
}

# ========== O_DIRECT 任意偏移读取（内部自动对齐） ==========
def read_bytes_odirect(dio: DirectIOFile, want_off: int, want_len: int, bsz: int) -> bytes:
    aligned_off = (want_off // bsz) * bsz
    delta = want_off - aligned_off
    need = delta + want_len
    need_aligned = ((need + bsz - 1) // bsz) * bsz
    buf = alloc_pinned_aligned(max(bsz, need_aligned), bsz)
    dio.pread_into_tensor(buf, buf.numel(), aligned_off)
    return bytes(buf[delta:delta + want_len].cpu().numpy())

# ========== 分片分组 + 拼接轴探测 ==========
def _group_layer_segments(manifest: dict, layer_id: int, only_stream: bool) -> Dict[str, List[dict]]:
    params = manifest["params"]
    entries = [p for p in params if int(p["layer"]) == int(layer_id)]
    if only_stream:
        entries = [p for p in entries if p.get("policy", "resident") == "stream"]
    groups: Dict[str, List[dict]] = defaultdict(list)
    for p in entries:
        groups[p["name"]].append(p)  # 按 manifest 顺序自然追加（即 pack 顺序）
    return groups

def _detect_concat_axis(shapes: List[Tuple[int, ...]]) -> int:
    """若仅有一个轴长度不同，其余轴完全一致，返回该轴索引；否则返回 -1（退化为 1D 拼接）。"""
    base = shapes[0]
    for ax in range(len(base)):
        same = True
        for s in shapes:
            if len(s) != len(base):
                return -1
            for j in range(len(base)):
                if j == ax:
                    continue
                if s[j] != base[j]:
                    same = False
                    break
            if not same:
                break
        if same:
            return ax
    return -1

# ========== SSD -> DRAM (CPU pinned) ==========
def ssd2dram_layer(
    manifest: dict,
    layer_id: int,
    staging_mb: int = 16,
    only_stream: bool = False,
    method: str = "bytecopy",   # "bytecopy"（通用/bf16友好）或 "reinterpret"（fp16/fp32）
) -> Dict[str, torch.Tensor]:
    """
    从 raw block device 读取指定层，**按 name 合并多分片**到一个 CPU pinned 张量。
    返回 {name: pinned_tensor}；不做 H2D。
    """
    raw_dev = manifest["raw_device"]
    bsz     = int(manifest["block_size"])

    groups = _group_layer_segments(manifest, layer_id, only_stream)
    if not groups:
        return {}

    dio = DirectIOFile(raw_dev, mode="r", block_size=bsz)
    staging_bytes = max(1, int(staging_mb)) * 1024 * 1024
    staging_bytes = (staging_bytes // bsz) * bsz
    staging = alloc_pinned_aligned(max(bsz, staging_bytes), bsz)  # uint8 pinned

    out: Dict[str, torch.Tensor] = {}

    for name, segs in groups.items():
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
            new_shape = (total_elems,)  # 退化为一维（数据仍正确）

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
                # 源 staging 上 numpy reinterpret（零拷贝视图）+ 一次 copy_ 到目标切片
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

# ========== DRAM (CPU pinned) -> GPU(HBM) ==========
def dram2gpu_layer(
    pinned_dict: Dict[str, torch.Tensor],
    device: str | torch.device = "cuda:0",
    non_blocking: bool = True,
    stream: Optional[torch.cuda.Stream] = None,
    dtype_override: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """
    将 {name: CPU pinned tensor} 整层搬到 GPU（可选指定 stream / dtype 覆写）。
    返回 {name: gpu_tensor}。
    """
    device = torch.device(device)
    out: Dict[str, torch.Tensor] = {}

    if stream is None:
        # 无自定义 stream：直接走当前流
        for name, t in pinned_dict.items():
            tgt_dtype = dtype_override or t.dtype
            out[name] = t.to(device, dtype=tgt_dtype, non_blocking=non_blocking)
        return out

    # 自定义 stream：在该 stream 上提交 H2D
    with torch.cuda.stream(stream):
        for name, t in pinned_dict.items():
            tgt_dtype = dtype_override or t.dtype
            out[name] = t.to(device, dtype=tgt_dtype, non_blocking=non_blocking)

    # 这里不强制同步，交由调用方在 compute 前 wait_stream
    return out

# ========== 整层逐分片校验（前 64 字节） ==========
def sanity_check_whole_layer(
    pinned_dict: Dict[str, torch.Tensor],
    manifest: dict,
    layer_id: int,
    verbose: bool = False
) -> Tuple[int, int]:
    """
    对该层所有参数的每个分片逐一校验（前 64 字节）：返回 (matched_pieces, total_pieces)。
    仅依赖 manifest + pinned_dict（不关心是否已搬上 GPU）。
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

# ========== （可选）简单 GPU 计算可用性测试 ==========
@torch.no_grad()
def gpu_validate_matmul(
    weight_cpu: torch.Tensor,
    device: torch.device,
    batch: int = 4,
    tol: float = None,            # None -> 自适应容差
    accum_fp32: bool = False,     # True -> GPU 端 FP32 累加
) -> Tuple[bool, float, float]:
    """
    仅对二维权重进行 y = x @ W^T 校验。
    返回: (是否通过, max_abs_err, 使用的阈值)
    """
    if weight_cpu.dim() != 2:
        raise ValueError("Only 2D weights are supported")

    out_dim, in_dim = weight_cpu.shape

    # 自适应容差：~ k * sqrt(in_dim) * eps(dtype)
    if tol is None:
        if weight_cpu.dtype == torch.bfloat16:
            eps = 1.0 / 128.0
        elif weight_cpu.dtype == torch.float16:
            eps = 2.0 ** -10
        else:
            eps = 1e-7
        tol = 3.0 * (in_dim ** 0.5) * eps  # k = 3

    # 构造输入（CPU pinned）
    x_cpu = torch.randn(batch, in_dim, dtype=weight_cpu.dtype, pin_memory=True)

    # 参考（CPU float32）
    y_ref = (x_cpu.float() @ weight_cpu.float().t())

    # 权重与输入上 GPU
    if accum_fp32:
        x_gpu = x_cpu.to(device, dtype=torch.float32, non_blocking=True)
        W_gpu = weight_cpu.to(device, dtype=torch.float32, non_blocking=True)
        y_gpu = x_gpu @ W_gpu.t()
    else:
        x_gpu = x_cpu.to(device, non_blocking=True)
        W_gpu = weight_cpu.to(device, non_blocking=True)
        y_gpu = x_gpu @ W_gpu.t()

    # 对比
    max_abs_err = (y_gpu.float().cpu() - y_ref).abs().max().item()
    ok = max_abs_err <= tol
    return ok, max_abs_err, tol

# ========== CLI（顺序校验每个 layer，避免 OOM） ==========
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
    # 下面两个是可选 GPU 检查（不影响 DRAM 验证）
    ap.add_argument("--gpu-check", action="store_true", help="after DRAM verify, send a few 2D weights to GPU and run tiny matmul")
    ap.add_argument("--gpu-device", default="cuda:0", help="CUDA device for --gpu-check")
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
        # 逐层：SSD -> DRAM（合并分片）
        tensors = ssd2dram_layer(
            manifest=manifest,
            layer_id=L,
            staging_mb=args.staging_mb,
            only_stream=args.only_stream,
            method=args.method,
        )

        if not tensors:
            print(f"[Layer {L}] (skipped: no entries under filter)")
            continue

        # DRAM校验（按分片）
        matched, total = sanity_check_whole_layer(
            tensors, manifest, layer_id=L, verbose=args.verbose
        )
        total_all += total
        matched_all += matched

        ok_rate = (matched / total * 100) if total else 100.0
        print(f"[Layer {L}] segment matches: {matched}/{total} ({ok_rate:.2f}%)")

        # （可选）GPU 小计算验证
        if args.gpu_check:
            if not torch.cuda.is_available():
                print("CUDA not available; skip --gpu-check")
            else:
                device = torch.device(args.gpu_device)
                # 简单挑选最多 4 个二维权重做 matmul
                tested = 0
                for name, t in tensors.items():
                    if t.dim() != 2:
                        continue
                    ok, err, tol = gpu_validate_matmul(t, device=device, batch=4, tol=None, accum_fp32=False)
                    print(f"  [GPU] {name}: {'OK' if ok else 'FAIL'} | max_err={err:.4e} | tol={tol:.4e}")
                    tested += 1
                    if tested >= 4:
                        break
                if tested == 0:
                    print("  [GPU] no 2D weights to test.")

        # 释放当前层内存，避免 OOM
        del tensors
        gc.collect()

        if args.stop_on_fail and matched < total:
            print("[STOP] mismatch detected, stopping early.")
            break

    print(f"[SUMMARY] total segment matches: {matched_all}/{total_all} "
          f"({(matched_all/total_all*100 if total_all else 100):.2f}%)")

if __name__ == "__main__":
    main()
