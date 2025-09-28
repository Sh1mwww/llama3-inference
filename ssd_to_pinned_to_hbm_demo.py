# ssd_to_pinned_to_hbm_demo.py
import argparse
import ctypes
import json
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

# 你的项目工具（保持路径一致）
from llama3.weights_io_ssd_dram import (
    DirectIOFile,
    alloc_pinned_aligned,
    DTYPE_MAP,  # {"float16": torch.float16, "bfloat16": torch.bfloat16, ...}
)

# ---------- 小工具 ----------
def read_bytes_odirect(dio: DirectIOFile, want_off: int, want_len: int, bsz: int) -> bytes:
    """O_DIRECT 对齐读取任意 offset 的 want_len 字节，返回准确切片。"""
    aligned_off = (want_off // bsz) * bsz
    delta = want_off - aligned_off
    need = delta + want_len
    need_aligned = ((need + bsz - 1) // bsz) * bsz
    buf = alloc_pinned_aligned(max(bsz, need_aligned), bsz)
    dio.pread_into_tensor(buf, buf.numel(), aligned_off)
    return bytes(buf[delta:delta + want_len].cpu().numpy())

def hexdump(b: bytes, n=32) -> str:
    return " ".join(f"{x:02x}" for x in b[:n])

def _detect_concat_axis(shapes: List[Tuple[int, ...]]) -> int:
    """若仅有一个轴长度不同，其余轴完全一致，返回该轴索引；否则返回 -1。"""
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

# ---------- 从 SSD 读取一个 layer 并合并分片到 CPU pinned ----------
def load_one_layer_to_pinned_merged(
    manifest: dict,
    layer_id: int,
    staging_mb: int = 16,
    only_stream: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    按 name 合并多分片，返回 {name: CPU pinned tensor}（dtype/shape 正确）。
    使用字节 copy（bf16 友好）。
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
            assert np.prod(new_shape) == total_elems, f"[{name}] shape/product mismatch"
        else:
            new_shape = (total_elems,)  # 退化为一维形状（数据仍正确）

        dst = torch.empty(new_shape, dtype=base_dtype, pin_memory=True)

        cursor_bytes = 0
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

            # staging[:nbytes] -> dst（按字节放到目标内存）
            ctypes.memmove(dst.data_ptr() + cursor_bytes, staging.data_ptr(), nbytes)
            cursor_bytes += nbytes

        out[name] = dst

    dio.close()
    return out

# ---------- GPU 端 matmul 校验（支持自适应容差/FP32 累加） ----------
@torch.no_grad()
def gpu_validate_matmul(
    weight_cpu: torch.Tensor,
    device: torch.device,
    batch: int = 4,
    tol: float = None,            # None 则自适应容差
    accum_fp32: bool = False,     # True: GPU 侧用 FP32 做 matmul 验证
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

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser(description="Load one layer from SSD -> CPU pinned -> GPU and validate a small compute.")
    ap.add_argument("--manifest", required=True, help="runtime manifest json path")
    ap.add_argument("--layer", type=int, default=0, help="layer id to load")
    ap.add_argument("--device", default="cuda:0", help="CUDA device, e.g., cuda:0")
    ap.add_argument("--staging-mb", type=int, default=16, help="staging buffer MiB")
    ap.add_argument("--only-stream", action="store_true", help="only QKV/O and FFN W1/2/3 (big weights)")
    ap.add_argument("--max-tests", type=int, default=4, help="test up to N 2D weights in this layer")
    ap.add_argument("--batch", type=int, default=4, help="batch size for the tiny matmul")
    ap.add_argument("--tol", type=float, default=None, help="fixed tolerance (overrides adaptive if provided)")
    ap.add_argument("--adaptive-tol", action="store_true", help="use adaptive tolerance (ignored if --tol is set)")
    ap.add_argument("--accum-fp32", action="store_true", help="do GPU-side FP32 accumulation for validation")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device(args.device)
    m = json.load(open(args.manifest, "r"))

    # 1) SSD -> CPU pinned（合并分片）
    tensors = load_one_layer_to_pinned_merged(
        manifest=m,
        layer_id=args.layer,
        staging_mb=args.staging_mb,
        only_stream=args.only_stream,
    )
    print(f"[OK] loaded {len(tensors)} tensors for layer {args.layer} (CPU pinned)")

    # 2) 在 GPU 上做“计算可用性”验证（选择二维权重）
    tests = 0
    for name, t in tensors.items():
        if t.dim() != 2:
            continue
        print(f"  - testing: {name} | shape={tuple(t.shape)} | dtype={t.dtype} | pinned={t.is_pinned()}")

        # 决定容差策略
        used_tol = args.tol
        if used_tol is None and not args.adaptive_tol:
            # 默认开启自适应容差
            used_tol = None  # 交由函数内部计算

        ok, err, auto_tol = gpu_validate_matmul(
            weight_cpu=t,
            device=device,
            batch=args.batch,
            tol=used_tol,                 # None => 使用自适应容差
            accum_fp32=args.accum_fp32,   # True => GPU FP32 累加
        )
        final_tol = auto_tol if used_tol is None else used_tol
        mode = "fp32-accum" if args.accum_fp32 else "native"
        print(f"    -> {mode} matmul: {'OK' if ok else 'FAIL'} | max_err={err:.4e} | tol={final_tol:.4e}")

        tests += 1
        if tests >= args.max_tests:
            break

    if tests == 0:
        print("[WARN] no 2D weight found to test in this layer under current filters.")

if __name__ == "__main__":
    main()
