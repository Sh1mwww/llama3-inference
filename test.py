#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只测一个 EncoderBlock 的 MHA / FFN 时间（不跑整模型，不用 WSM/SSD）

场景：
  - prefill:  B=1, T=2048, start_pos=0
  - decode:   B=1, 每步 T=1，从 start_pos=2048 开始，共 32 步
  - KV window: 用 env KV_DECODE_WINDOW_TOKENS=512 控制
  - 输出：
      * prefill 阶段：MHA/FFN 总时间（该层）
      * decode 阶段：MHA/FFN 总时间 + 每 token 平均

依赖：
  - 直接用你项目里的 llama3/layers.py（EncoderBlock / PERF_TRACKER / cuda_timer）
"""

import os
import sys
import time
import argparse
from dataclasses import dataclass
from types import SimpleNamespace

os.environ.setdefault("LLM_PROFILE", "1")              # 打开 cuda_timer
os.environ.setdefault("KV_DECODE_WINDOW_TOKENS", "512")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Create a mock stream_mnt module that returns None for get_streams to avoid compute_stream path bug
mock_stream_mnt = SimpleNamespace(get_streams=lambda device: None)
sys.modules['llama3.stream_mnt'] = mock_stream_mnt

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from llama3.layers import (
    EncoderBlock,
    precompute_theta_pos_frequencies,
    reset_performance_stats,
    get_global_performance_stats,
)


@dataclass
class DummyArgs:
    # 只给 EncoderBlock / SelfAttention / FeedForward 用到的字段
    dim: int = 8192
    n_layers: int = 1
    n_heads: int = 64
    n_kv_heads: int = 8
    multiple_of: int = 4096
    ffn_dim_multiplier: float = 1.0
    norm_eps: float = 1e-5
    max_batch_size: int = 8
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    device: str = "cuda:0"
    topk_blk: int = 4
    use_stub_params: bool = False
    param_init_device: str = "cuda:0"   # 初始化权重所在 device


def build_block(args: DummyArgs, layer_id: int, device: torch.device) -> EncoderBlock:
    print(f"[BUILD] dim={args.dim}, n_heads={args.n_heads}, n_kv_heads={args.n_kv_heads}")
    block = EncoderBlock(args, layer_id=layer_id)
    # Convert to bfloat16 to match RMSNorm's default dtype
    block.to(device=device, dtype=torch.bfloat16)
    block.eval()
    for p in block.parameters():
        p.requires_grad_(False)
    return block


def run_prefill(block: EncoderBlock,
                args: DummyArgs,
                device: torch.device,
                prefill_len: int,
                layer_id: int,
                batch_size: int = 1):
    print(f"\n===== PREFILL (B={batch_size}, T={prefill_len}, start_pos=0) =====")
    reset_performance_stats()

    # 用权重的 dtype，避免 bfloat16 vs float32 冲突
    wq = block.attention.wq.weight
    dtype = wq.dtype

    x = torch.randn(batch_size, prefill_len, args.dim, device=device, dtype=dtype)
    head_dim = args.dim // args.n_heads
    freqs = precompute_theta_pos_frequencies(
        head_dim,
        args.max_seq_len,
        device=device,
        theta=args.rope_theta,
    )

    with torch.no_grad():
        # warmup
        for _ in range(3):
            _ = block(x, 0, freqs)
        torch.cuda.synchronize()
        reset_performance_stats()

        t0 = time.perf_counter()
        _ = block(x, 0, freqs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

    wall_ms = (t1 - t0) * 1000.0
    stats = get_global_performance_stats()
    layer_stats = stats.get("per_layer", {}).get(layer_id, {})

    attn_us = float(layer_stats.get("attn_us", 0.0))
    ffn_us = float(layer_stats.get("ffn_us", 0.0))
    kv_us = float(layer_stats.get("kv_fetch_us", 0.0))
    total_us = float(layer_stats.get("total_forward_us", attn_us + ffn_us))

    print(f"Wall time (prefill single call): {wall_ms:.3f} ms\n")
    print("Per-layer (PREFILL):")
    print(f"  attn_compute_ms  : {attn_us / 1000.0:.3f}")
    print(f"  ffn_compute_ms   : {ffn_us / 1000.0:.3f}")
    print(f"  kv_fetch_ms      : {kv_us / 1000.0:.3f}")
    print(f"  total_forward_ms : {total_us / 1000.0:.3f}")

    return {
        "wall_ms": wall_ms,
        "attn_ms": attn_us / 1000.0,
        "ffn_ms": ffn_us / 1000.0,
        "kv_ms": kv_us / 1000.0,
        "total_ms": total_us / 1000.0,
    }


def run_decode(block: EncoderBlock,
               args: DummyArgs,
               device: torch.device,
               prefill_len: int,
               decode_len: int,
               layer_id: int,
               batch_size: int = 1):
    print(f"\n===== DECODE (B={batch_size}, steps={decode_len}, start_pos={prefill_len}) =====")
    reset_performance_stats()

    wq = block.attention.wq.weight
    dtype = wq.dtype

    head_dim = args.dim // args.n_heads
    freqs = precompute_theta_pos_frequencies(
        head_dim,
        args.max_seq_len,
        device=device,
        theta=args.rope_theta,
    )

    with torch.no_grad():
        # warmup 几步
        for _ in range(4):
            x_warm = torch.randn(batch_size, 1, args.dim, device=device, dtype=dtype)
            _ = block(x_warm, prefill_len, freqs)
        torch.cuda.synchronize()
        reset_performance_stats()

        t0 = time.perf_counter()
        cur_pos = prefill_len
        for _ in range(decode_len):
            x_step = torch.randn(batch_size, 1, args.dim, device=device, dtype=dtype)
            _ = block(x_step, cur_pos, freqs)
            cur_pos += 1
        torch.cuda.synchronize()
        t1 = time.perf_counter()

    wall_ms = (t1 - t0) * 1000.0
    stats = get_global_performance_stats()
    layer_stats = stats.get("per_layer", {}).get(layer_id, {})

    attn_us = float(layer_stats.get("attn_us", 0.0))
    ffn_us = float(layer_stats.get("ffn_us", 0.0))
    kv_us = float(layer_stats.get("kv_fetch_us", 0.0))
    total_us = float(layer_stats.get("total_forward_us", attn_us + ffn_us))

    attn_ms_total = attn_us / 1000.0
    ffn_ms_total = ffn_us / 1000.0
    kv_ms_total = kv_us / 1000.0
    total_ms_total = total_us / 1000.0

    print(f"Wall time (decode {decode_len} steps): {wall_ms:.3f} ms")
    print(f"  wall per-token   : {wall_ms / decode_len:.4f} ms/token\n")

    print("Per-layer (DECODE, total over all tokens):")
    print(f"  attn_compute_ms_total  : {attn_ms_total:.3f}")
    print(f"  ffn_compute_ms_total   : {ffn_ms_total:.3f}")
    print(f"  kv_fetch_ms_total      : {kv_ms_total:.3f}")
    print(f"  total_forward_ms_total : {total_ms_total:.3f}")

    total_tokens = decode_len * batch_size  # 实际生成的 token 数

    print("\nPer-step (DECODE, averaged over all steps):")
    print(f"  attn_compute_ms_per_step : {attn_ms_total / decode_len:.4f}")
    print(f"  ffn_compute_ms_per_step  : {ffn_ms_total / decode_len:.4f}")
    print(f"  kv_fetch_ms_per_step     : {kv_ms_total / decode_len:.4f}")
    print(f"  total_forward_ms_per_step: {total_ms_total / decode_len:.4f}")

    print(f"\nPer-token (DECODE, total {total_tokens} tokens = {decode_len} steps × {batch_size} batch):")
    print(f"  attn_compute_ms_per_tok : {attn_ms_total / total_tokens:.4f}")
    print(f"  ffn_compute_ms_per_tok  : {ffn_ms_total / total_tokens:.4f}")
    print(f"  kv_fetch_ms_per_tok     : {kv_ms_total / total_tokens:.4f}")
    print(f"  total_forward_ms_per_tok: {total_ms_total / total_tokens:.4f}")
    print(f"  throughput (tokens/sec) : {total_tokens / (wall_ms / 1000.0):.2f}")

    return {
        "wall_ms_total": wall_ms,
        "wall_ms_per_step": wall_ms / decode_len,
        "wall_ms_per_tok": wall_ms / total_tokens,
        "attn_ms_total": attn_ms_total,
        "ffn_ms_total": ffn_ms_total,
        "kv_ms_total": kv_ms_total,
        "total_ms_total": total_ms_total,
        "total_tokens": total_tokens,
        "throughput": total_tokens / (wall_ms / 1000.0),
    }


def main():
    p = argparse.ArgumentParser("bench single layer MHA/FFN (prefill & decode)")
    p.add_argument("--dim", type=int, default=8192)
    p.add_argument("--n-heads", type=int, default=64)
    p.add_argument("--n-kv-heads", type=int, default=8)
    p.add_argument("--prefill-len", type=int, default=2048)
    p.add_argument("--decode-len", type=int, default=32)
    p.add_argument("--layer-id", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能跑这个脚本")

    device = torch.device("cuda:0")

    dummy = DummyArgs(
        dim=args.dim,
        n_layers=1,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_batch_size=args.batch_size,
        max_seq_len=args.prefill_len + args.decode_len + 16,
        device=str(device),
        param_init_device=str(device),
    )

    block = build_block(dummy, layer_id=args.layer_id, device=device)

    # 先 prefill，让 KV 里有 0..prefill_len 的历史
    prefill_res = run_prefill(
        block=block,
        args=dummy,
        device=device,
        prefill_len=args.prefill_len,
        layer_id=args.layer_id,
        batch_size=args.batch_size,
    )

    # 再 decode，测每 token 的 mha/ffn/kv 时间
    decode_res = run_decode(
        block=block,
        args=dummy,
        device=device,
        prefill_len=args.prefill_len,
        decode_len=args.decode_len,
        layer_id=args.layer_id,
        batch_size=args.batch_size,
    )

    print("\n===== SUMMARY =====")
    print(f"Prefill(B={args.batch_size},T={args.prefill_len}) layer {args.layer_id}:")
    print(f"  attn_compute_ms  : {prefill_res['attn_ms']:.3f}")
    print(f"  ffn_compute_ms   : {prefill_res['ffn_ms']:.3f}")
    print(f"  kv_fetch_ms      : {prefill_res['kv_ms']:.3f}")
    print(f"  total_forward_ms : {prefill_res['total_ms']:.3f}")

    total_tokens = decode_res['total_tokens']
    print(f"\nDecode(B={args.batch_size},steps={args.decode_len}) layer {args.layer_id}:")
    print(f"  Total tokens generated   : {total_tokens}")
    print(f"  Throughput (tokens/sec)  : {decode_res['throughput']:.2f}")
    print(f"  Time per step            : {decode_res['wall_ms_per_step']:.4f} ms")
    print(f"  Time per token           : {decode_res['wall_ms_per_tok']:.4f} ms")
    print(f"  attn_ms_per_tok          : {decode_res['attn_ms_total']/total_tokens:.4f}")
    print(f"  ffn_ms_per_tok           : {decode_res['ffn_ms_total']/total_tokens:.4f}")
    print(f"  kv_ms_per_tok            : {decode_res['kv_ms_total']/total_tokens:.4f}")


if __name__ == "__main__":
    main()
