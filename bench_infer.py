
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bench_infer.py — Minimal‑overhead benchmark for your llama3 project.

Measures (without counting model load & warmup):
- TTFT (prefill start → first token ready)
- Prefill time
- First-token decode time
- Decode time (multi-token)
- End-to-end time (prefill+decode)
- Throughput (decode tokens / second)
- Optional I/O bandwidth from GlobalStateTracker (weight_h2d, kv_h2d, kv_d2h)

Design:
- No per-token synchronizations. We use 4 CUDA events total.
- Safe prompt trimming + chunked prefill to avoid OOM on 16GB GPUs.
- Tokens are created on the exact device used by embed_tokens.weight.
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch

# ===== Project imports =====
from llama3.generator import LLaMA
from llama3.global_state_tracker import init_global_tracker, get_global_tracker
from llama3.config import KVCacheArgs

# ------------------------------
# Env presets to avoid OOM and reduce stalls
# ------------------------------

def apply_env_presets(gpu_guard_mb: int = 1536, gpu_max_groups: int = 6, group_pf_depth: int = 2):
    # PyTorch allocator: reduce fragmentation impact
    os.environ.setdefault("OMP_NUM_THREADS",  "8")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")

    # （2）WSM 行为开关（滑窗 + 回环 + 组上限等）
    os.environ.setdefault("WSM_CPU_ROLLING_MODE",  "1")  # 滚动滑窗
    os.environ.setdefault("WSM_CPU_RING_OFFSET",  "0")  
    os.environ.setdefault("WSM_CPU_WRAP_AROUND",   "1")  # 窗口末尾后回环到 L0
    os.environ.setdefault("WSM_CPU_ROLL_STRIDE",   "1")
    os.environ.setdefault("WSM_CPU_ROLL_SYNC",     "1")  # 计算线程同步推进
    os.environ.setdefault("WSM_AGGRESSIVE_GPU_PREFETCH", "2")  # 当前层 ffn + 下一层 attn
    os.environ.setdefault("WSM_H2D_GROUP_BACKLOG_MAX",   "1")
    os.environ.setdefault("WSM_GPU_MAX_GROUPS",          "10")
    os.environ.setdefault("WSM_SKIP_PRELOAD_WAIT",       "1")  # 不卡在预热等待
    os.environ.setdefault("WSM_EVICT_FINISHED",        "1")   # 组算完即踢（释放预算）
    
    os.environ.setdefault("WSM_KV_THROTTLE_THRESHOLD",       "8")
    os.environ.setdefault("WSM_KV_THROTTLE_MS",       "16")
    
    os.environ.setdefault("WSM_BALANCE_PREFETCH", "1")
    os.environ.setdefault("WSM_BALANCE_TOL",      "1")   # attn/ffn 允许相差 ≤1
    os.environ.setdefault("WSM_PAIR_AHEAD",       "2")   # 就近择层范围：同层→i+1→i+2
    os.environ.setdefault("WSM_KIND_AHEAD_CAP",   "2")   # 单一类型最大前瞻距离
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 可选：减少初期碎片/线程数
    os.environ.setdefault("OMP_NUM_THREADS",  "8")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")


# ------------------------------
# Safe prompt trim (respect max_seq_len - max_new_tokens)
# ------------------------------

def safe_trim_prompt(text: str, tokenizer, max_seq_len: int, max_new_tokens: int) -> str:
    max_prompt_tokens = max_seq_len - max_new_tokens
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_prompt_tokens:
        ids = ids[-max_prompt_tokens:]
    return tokenizer.decode(ids)


# ------------------------------
# Build model (mixed/raw-SSD / stream / full)
# ------------------------------

def build_llama(args) -> LLaMA:
    mode_cfg = {}
    mode = args.mode
    device = args.device or (f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

    if mode in ("mixed", "ssd"):
        if not args.manifest or not args.raw_device:
            raise SystemExit("--mode mixed/ssd requires --manifest and --raw-device")
        mode_cfg.update({
            "raw_device": args.raw_device,
            "ssd_manifest_path": args.manifest,
            "prefetch_distance": args.prefetch_distance_layers,   # layer prefetch (WSM internal)
            "group_prefetch_depth": args.group_prefetch_depth,    # group prefetch (attn ahead)
            "max_cached_layers": args.gpu_cached_layers,
            "cpu_cache_layers": args.cpu_cache_layers,
            "warmup_layers": args.warmup_layers,
            "staging_mb": args.staging_mb,
            "verbose": True,
        })
        load_model = False  # critical: don't pull full 70B into CPU
    else:
        load_model = True

    llama = LLaMA.build(
        checkpoints_dir=args.checkpoints,
        load_model=load_model,
        device=device,
        mode=mode if mode != "ssd" else "mixed",
        mode_config=mode_cfg if mode in ("mixed", "ssd") else None,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.batch_size,
        topk_blk=args.topk_blk,
    )
    return llama


# ------------------------------
# Prefill (chunked) to reduce attention peak memory
# ------------------------------

def prefill_in_chunks(model, tokens: torch.Tensor, total_prefill: int, chunk: int):
    """
    Run prefill in smaller slices using start_pos, to reduce attention O(T^2) peak.
    """
    pos = 0
    while pos < total_prefill:
        step = min(chunk, total_prefill - pos)
        _ = model(tokens[:, pos:pos+step], start_pos=pos)
        pos += step


# ------------------------------
# Inference once + timing
# ------------------------------

@torch.inference_mode()
def run_once_measure(
    llama: LLaMA,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 0.9,
    prefill_chunk_tok: int = 768,
) -> Dict[str, Any]:
    tok = llama.tokenizer

    # Trim prompts safely
    prompts = [safe_trim_prompt(p, tok, llama.args.max_seq_len, max_new_tokens) for p in prompts]

    # Tokenize
    batch_tok = [tok.encode(p, add_special_tokens=False) for p in prompts]
    bsz = len(batch_tok)
    max_prompt = max(len(x) for x in batch_tok) if batch_tok else 0
    total_len = min(llama.args.max_seq_len, max_prompt + max_new_tokens)

    # Determine compute device from embed_tokens
    try:
        dev = llama.model.embed_tokens.weight.device
    except Exception:
        dev = torch.device(str(getattr(llama.model, "device", llama.args.device)))

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_id = tok.eos_token_id

    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=dev)
    for i, t in enumerate(batch_tok):
        tokens[i, :len(t)] = torch.tensor(t, device=dev)
    prompt_mask = tokens != pad_id
    eos_mask = torch.zeros(bsz, dtype=torch.bool, device=dev)

    use_cuda = torch.cuda.is_available() and str(dev).startswith("cuda")

    # Events / wallclock
    if use_cuda:
        ev_prefill_start = torch.cuda.Event(enable_timing=True)
        ev_prefill_end   = torch.cuda.Event(enable_timing=True)
        ev_first_end     = torch.cuda.Event(enable_timing=True)
        ev_decode_end    = torch.cuda.Event(enable_timing=True)
    else:
        t_prefill_start = t_prefill_end = t_first_end = t_decode_end = None

    # ---------- Prefill ----------
    prefill_len = max_prompt
    if use_cuda:
        ev_prefill_start.record()
    else:
        t_prefill_start = time.time()

    if prefill_len > 0:
        prefill_in_chunks(llama.model, tokens, prefill_len, prefill_chunk_tok)

    if use_cuda:
        ev_prefill_end.record()
    else:
        t_prefill_end = time.time()

    # ---------- Decode (no per-token sync) ----------
    first_token_marked = False
    gen_steps = 0
    cur_pos = prefill_len

    if use_cuda:
        # Mark first-token when available
        pass

    while cur_pos < total_len:
        logits = llama.model(tokens[:, cur_pos-1:cur_pos], cur_pos)
        last = logits[:, -1, :]
        if temperature > 0.0:
            probs = torch.softmax(last / temperature, dim=-1)
            # simple top-p approx: pick index at the cdf cutoff
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cdf <= top_p).sum(dim=-1, keepdim=True).clamp_min(1)
            gather_idx = torch.gather(sorted_idx, 1, cutoff - 1)
            next_tok = gather_idx.squeeze(1)
        else:
            next_tok = torch.argmax(last, dim=-1)

        # Respect original prompt (if still in prompt region)
        next_tok = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_tok)

        tokens[:, cur_pos] = next_tok
        gen_steps += 1

        if not first_token_marked:
            if use_cuda:
                ev_first_end.record()
            else:
                t_first_end = time.time()
            first_token_marked = True

        eos_mask |= (next_tok == eos_id)
        if torch.all(eos_mask):
            break

        cur_pos += 1

    if use_cuda:
        ev_decode_end.record()
        torch.cuda.synchronize(device=dev)
    else:
        t_decode_end = time.time()

    # ---------- Durations ----------
    if use_cuda:
        prefill_ms = ev_prefill_start.elapsed_time(ev_prefill_end)
        if first_token_marked:
            first_token_decode_ms = ev_prefill_end.elapsed_time(ev_first_end)
            ttft_ms = ev_prefill_start.elapsed_time(ev_first_end)
        else:
            first_token_decode_ms = 0.0
            ttft_ms = prefill_ms
        decode_ms = ev_prefill_end.elapsed_time(ev_decode_end)
        e2e_ms = ev_prefill_start.elapsed_time(ev_decode_end)
    else:
        prefill_ms = (t_prefill_end - t_prefill_start) * 1000.0 if prefill_len > 0 else 0.0
        if first_token_marked:
            first_token_decode_ms = (t_first_end - t_prefill_end) * 1000.0
            ttft_ms = (t_first_end - t_prefill_start) * 1000.0
        else:
            first_token_decode_ms = 0.0
            ttft_ms = prefill_ms
        decode_ms = (t_decode_end - t_prefill_end) * 1000.0
        e2e_ms = (t_decode_end - t_prefill_start) * 1000.0

    # ---------- Stats ----------
    decode_tokens_total = int(gen_steps * bsz)
    prefill_tokens_total = int(sum(len(x) for x in batch_tok))

    decode_s = max(1e-6, decode_ms / 1000.0)
    throughput_tok_s = decode_tokens_total / decode_s

    # Optional I/O bandwidth from tracker
    io_bw = {}
    try:
        tracker = get_global_tracker()
        if tracker is not None:
            io_bw = {
                "weight_h2d_MBps": tracker.last_bw("weight_h2d"),
                "kv_h2d_MBps": tracker.last_bw("kv_h2d"),
                "kv_d2h_MBps": tracker.last_bw("kv_d2h"),
            }
    except Exception:
        pass

    # GPU memory snapshot
    gpu_mem = {}
    try:
        if use_cuda:
            # derive index if dev like "cuda:0"
            if isinstance(dev, torch.device):
                idx = dev.index if dev.index is not None else torch.cuda.current_device()
            else:
                dev_str = str(dev)
                idx = int(dev_str.split(":")[1]) if ":" in dev_str else torch.cuda.current_device()
            st = torch.cuda.memory_stats(idx)
            gpu_mem = {
                "allocated_GB": st.get("allocated_bytes.all.current", 0)/(1<<30),
                "reserved_GB":  st.get("reserved_bytes.all.current", 0)/(1<<30),
            }
    except Exception:
        pass

    return {
        "batch_size": bsz,
        "max_new_tokens": max_new_tokens,
        "prompt_lens": [len(x) for x in batch_tok],
        "prefill_tokens_total": prefill_tokens_total,
        "decode_tokens_total": decode_tokens_total,
        "prefill_ms": round(prefill_ms, 3),
        "first_token_decode_ms": round(first_token_decode_ms, 3),
        "ttft_ms": round(ttft_ms, 3),
        "decode_ms": round(decode_ms, 3),
        "e2e_ms": round(e2e_ms, 3),
        "throughput_tok_s": round(throughput_tok_s, 2),
        "end_to_end_tok_s": round((prefill_tokens_total + decode_tokens_total) / max(1e-6, e2e_ms/1000.0), 2),
        "io_bw_MBps": io_bw,
        "gpu_mem_GB": gpu_mem,
    }


# ------------------------------
# Warmup
# ------------------------------

@torch.inference_mode()
def run_warmup(llama: LLaMA, rounds: int = 1, prompt_len: int = 64, gen_len: int = 4):
    tok = llama.tokenizer
    dev = llama.model.embed_tokens.weight.device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    dummy = "A" * prompt_len
    bsz = min(4, max(1, llama.args.max_batch_size))
    ids = tok.encode(dummy, add_special_tokens=False)
    total_len = min(llama.args.max_seq_len, prompt_len + gen_len)

    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=dev)
    for i in range(bsz):
        tokens[i, :len(ids)] = torch.tensor(ids, device=dev)

    for _ in range(max(0, rounds)):
        # prefill (chunked small)
        prefill_in_chunks(llama.model, tokens, prompt_len, chunk=min(prompt_len, 128))
        cur = prompt_len
        steps = 0
        while cur < total_len and steps < gen_len:
            logits = llama.model(tokens[:, cur-1:cur], cur)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1)
            tokens[:, cur] = next_tok
            cur += 1
            steps += 1


# ------------------------------
# CLI
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="OOM-safe, low-overhead inference benchmark for your llama3 project")
    p.add_argument("--checkpoints", type=str, required=True, help="Checkpoint dir (with tokenizer/params.json)")
    p.add_argument("--mode", type=str, default="mixed", choices=["mixed", "ssd", "stream", "preload", "full"])
    p.add_argument("--raw-device", dest="raw_device", type=str, default=None, help="Raw SSD device, e.g. /dev/nvme0n1p4")
    p.add_argument("--manifest", type=str, default=None, help="Runtime manifest path (for raw-SSD modes)")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--prompts", type=str, default=None, help="Text file with multiple lines; will repeat to fill batch")
    p.add_argument("--prompt", type=str, default=None, help="Single prompt; will be repeated to fill batch")
    p.add_argument("--warmup-rounds", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--out-json", type=str, default="bench_results.json")

    # streaming / WSM knobs
    p.add_argument("--prefetch-distance-layers", type=int, default=4)
    p.add_argument("--group-prefetch-depth", type=int, default=2)
    p.add_argument("--gpu-cached-layers", type=int, default=3)
    p.add_argument("--cpu-cache-layers", type=int, default=40)
    p.add_argument("--warmup-layers", type=int, default=1)
    p.add_argument("--staging-mb", type=int, default=64)
    p.add_argument("--topk-blk", type=int, default=8)

    # safety / performance
    p.add_argument("--prefill-chunk-tok", type=int, default=768, help="Prefill chunk size in tokens (512~1024 for 16GB)")
    p.add_argument("--env-presets", action="store_true", default=True, help="Apply WSM & allocator presets (on by default)")
    return p.parse_args()


def read_prompts(args) -> List[str]:
    if args.prompts and Path(args.prompts).exists():
        lines = [ln.strip() for ln in Path(args.prompts).read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            lines = ["Hello"]
    else:
        lines = [args.prompt or "Hello"]
    if len(lines) < args.batch_size:
        lines = (lines * ((args.batch_size + len(lines) - 1)//len(lines)))[:args.batch_size]
    else:
        lines = lines[:args.batch_size]
    return lines


def main():
    args = parse_args()

    if args.env_presets:
        apply_env_presets()

    llama = build_llama(args)

    # Global tracker (optional, but used for I/O stats)
    try:
        n_layers = getattr(llama.args, "n_layers", 80)
        tracker = init_global_tracker(max_batch=args.batch_size, layers=n_layers, n_blocks=128)
        # Don't write per-layer timings by default to avoid I/O overhead
        tracker.set_layer_timing_output(None)
    except Exception:
        tracker = None

    if args.warmup_rounds > 0:
        run_warmup(llama, rounds=args.warmup_rounds, prompt_len=64, gen_len=4)

    prompts = read_prompts(args)

    metrics = run_once_measure(
        llama=llama,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        prefill_chunk_tok=max(64, int(args.prefill_chunk_tok)),
    )

    # Decorate with mode/device info
    metrics["mode"] = args.mode
    metrics["device"] = str(llama.args.device)
    try:
        has_wsm = hasattr(llama, "weight_streaming_manager")
        metrics["wsm"] = {
            "enabled": bool(has_wsm),
            "ssd_enabled": bool(has_wsm and getattr(llama.weight_streaming_manager, "ssd_enabled", False)),
            "gpu_max_groups": int(has_wsm and getattr(llama.weight_streaming_manager, "gpu_max_groups", 0)),
        }
    except Exception:
        pass

    Path(args.out_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Benchmark Summary ===")
    for k in ("ttft_ms","prefill_ms","first_token_decode_ms","decode_ms","e2e_ms","throughput_tok_s","end_to_end_tok_s"):
        print(f"{k:>24}: {metrics.get(k)}")
    print(f"\nSaved JSON: {Path(args.out_json).resolve()}")


if __name__ == "__main__":
    main()
