# ================================
# scripts/profile_pipeline.py
# ================================
#!/usr/bin/env python
"""Comprehensive profiler for LLaMA-3 inference.
Measures:
  • Checkpoint load to CPU time & size
  • Weights transfer to GPU HBM time & size
  • KV-cache save (push) / load (fetch) time and size
  • MHA & FFN compute time
  • KV per-token size
Run:
    python scripts/profile_pipeline.py \
        --model-path /path/to/Llama3.2-3B \
        --device cuda \
        --prompt "Why is the sky blue?" \
        --max-gen-len 128
"""
import argparse, time, pathlib, builtins, types, math
from contextlib import contextmanager

import torch
from tqdm import tqdm
from llama3.generator import LLaMA

# ---------- global accumulators ----------
STAT = {
    "weights_cpu_ms": 0.0,
    "weights_hbm_ms": 0.0,
    "kv_push_ms": 0.0,
    "kv_fetch_ms": 0.0,
    "attn_ms": 0.0,
    "ffn_ms": 0.0,
}


# ---------- utility ----------
@contextmanager
def timer(key: str):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    yield
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    STAT[key] += (time.time() - t0) * 1e3  # ms

def sizeof_fmt(num):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024:
            return f"{num:.2f} {unit}"
        num /= 1024
    return f"{num:.2f} PB"

def param_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())

# ---------- monkey-patch KVOffloader ----------
import llama3.kv_offload as kvmod
if not hasattr(kvmod, "_patched"):
    orig_push, orig_fetch = kvmod.KVOffloader.push, kvmod.KVOffloader.fetch

    def push_patch(self, *args, **kwargs):
        with timer("kv_push_ms"):
            return orig_push(self, *args, **kwargs)

    def fetch_patch(self, *args, **kwargs):
        with timer("kv_fetch_ms"):
            return orig_fetch(self, *args, **kwargs)

    kvmod.KVOffloader.push = push_patch
    kvmod.KVOffloader.fetch = fetch_patch
    kvmod._patched = True

# ---------- monkey-patch FeedForward ----------
import llama3.layers as lyrs
if not hasattr(lyrs, "_ffn_patched"):
    orig_ffn_forward = lyrs.FeedForward.forward
    def ffn_patch(self, x):
        with timer("ffn_ms"):
            return orig_ffn_forward(self, x)
    lyrs.FeedForward.forward = ffn_patch
    lyrs._ffn_patched = True

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--max-gen-len", type=int, default=64)
    args = ap.parse_args()

    ckpt_dir = pathlib.Path(args.model_path)

    # ----- load checkpoint → CPU -----
    with timer("weights_cpu_ms"):
        llama = LLaMA.build(ckpt_dir, load_model=True, device="cpu")

    weights_file_bytes = sum(f.stat().st_size for f in ckpt_dir.glob("*.pth"))
    weights_tensor_bytes = param_bytes(llama.model)

    # ----- transfer weights → GPU -----
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        t0 = time.time()
        llama.model.to(args.device)
        torch.cuda.synchronize()
        STAT["weights_hbm_ms"] = (time.time() - t0) * 1e3
    else:
        llama.model.to(args.device)
        
    llama.args.device = args.device 
    for layer in llama.model.layers:
        off = layer.attention.offloader
        off.device = args.device              # 让 push()/fetch() 用 GPU
        off.copy_stream = torch.cuda.Stream(device=args.device)

    # ----- run inference (prefill + decode) -----
    prompts = [args.prompt]
    _, _ = llama.text_completion(prompts, max_gen_len=args.max_gen_len)

    # gather MHA time (already per layer)
    STAT["attn_ms"] = sum(llama.model.attn_times)

    # ----- KV size -----
    L = llama.args.n_layers
    Hk = llama.args.n_kv_heads or llama.args.n_heads
    D = llama.args.dim // llama.args.n_heads
    dtype_bytes = llama.model.embed_tokens.weight.element_size()
    seq_len = len(llama.tokenizer.encode(args.prompt)) + args.max_gen_len
    kv_total_bytes = L * 2 * Hk * D * seq_len * dtype_bytes
    kv_per_tok = kv_total_bytes / seq_len

    # ----- report -----
    print("\n===== Pipeline Profile =====")
    print(f"Weight files           : {sizeof_fmt(weights_file_bytes)}")
    print(f"Weight tensors         : {sizeof_fmt(weights_tensor_bytes)}")
    print(f"KV‑cache total         : {sizeof_fmt(kv_total_bytes)}")
    print(f"KV per token           : {sizeof_fmt(kv_per_tok)}")
    print("-------------------------------")
    print(f"Load weights → CPU     : {STAT['weights_cpu_ms']:.1f} ms")
    print(f"Transfer weights → HBM : {STAT['weights_hbm_ms']:.1f} ms")
    print(f"KV save (DRAM)         : {STAT['kv_push_ms']:.1f} ms")
    print(f"KV load (HBM)          : {STAT['kv_fetch_ms']:.1f} ms")
    print(f"MHA compute            : {STAT['attn_ms']:.1f} ms")
    print(f"FFN compute            : {STAT['ffn_ms']:.1f} ms")
    total_io = STAT['weights_cpu_ms'] + STAT['weights_hbm_ms'] + STAT['kv_push_ms'] + STAT['kv_fetch_ms']
    total_compute = STAT['attn_ms'] + STAT['ffn_ms']
    print("-------------------------------")
    print(f"Total I/O time         : {total_io:.1f} ms")
    print(f"Total compute time     : {total_compute:.1f} ms\n")

if __name__ == "__main__":
    main()