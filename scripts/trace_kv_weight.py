#!/usr/bin/env python
"""
trace_kv_weight.py  ─  Chrome‑trace + CSV profiler for LLaMA‑3 runtime
====================================================================
Captures **four**类事件，并追加 I/O 字节与持续时间：
    ▸ WEIGHT_UPLOAD_Lx          权重 `.to(cuda)`
    ▸ KV_PUSH_Lx_Bb             KV‑cache GPU→CPU (push)
    ▸ KV_FETCH_Lx_Bb            KV‑cache CPU→GPU (prefetch)
    ▸ MHA_Lx  /  FFN_Lx         计算（Self‑Attention & FeedForward）

输出文件：
    • trace.json          —— Chrome://tracing 可视化时序
    • trace_summary.csv   —— event,cat,bytes,dur_us  汇总表

用法：
    python scripts/trace_kv_weight.py \
        --model-path /path/to/Llama3.2-3B \
        --prompt "Why is the sky blue?" \
        --max-gen-len 64 \
        --device cuda
"""
import argparse, json, os, csv, time, functools, pathlib
import torch
from llama3.generator import LLaMA
from llama3.kv_offload import KVOffloader

# ---------- Chrome trace容器 ----------
trace_events = []

def emit(ev_name, cat, start_host_ns, dur_us, bytes_=0):
    trace_events.append({
        "ph": "X",
        "name": ev_name,
        "cat": cat,
        "ts": start_host_ns / 1000,   # ns → µs
        "dur": dur_us,
        "pid": os.getpid(),
        "args": {"bytes": bytes_, "us": dur_us}
    })
    csv_writer.writerow([ev_name, cat, bytes_, f"{dur_us:.1f}"])

# ---------- Weight upload wrapper ----------

def wrap_weight_upload(param_module, tag: str):
    """Wrap `.to()` so第一次上传GPU时计时."""
    orig_to = param_module.to
    bytes_ = sum(p.numel()*p.element_size() for p in param_module.parameters(recurse=False))
    @functools.wraps(orig_to)
    def patched_to(*args, **kw):
        if all(p.is_cuda for p in param_module.parameters(recurse=False)):
            return orig_to(*args, **kw)
        t0 = time.perf_counter_ns()        # ★ host 起点(ns)
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        out = orig_to(*a, **kw)
        end.record(); torch.cuda.synchronize()
        dur_us = start.elapsed_time(end) * 1000
        emit(f"WEIGHT_UPLOAD_{tag}", "weight", t0, dur_us, bytes_)
        return out
    param_module.to = patched_to

# ---------- KV push / fetch wrappers ----------
orig_push  = KVOffloader.push
orig_fetch = KVOffloader.fetch

def kv_wrapper(orig_fn, tag_push: str, tag_fetch: str):
    @functools.wraps(orig_fn)
    def inner(self, layer_id, blocks, *a, **kw):
        # blocks:  int     for push
        #          tensor  for fetch
        s = torch.cuda.Event(True); s.record()
        host_ns = time.perf_counter_ns()

        out = orig_fn(self, layer_id, blocks, *a, **kw)

        e = torch.cuda.Event(True); e.record()
        torch.cuda.current_stream().synchronize()
        dur_us = s.elapsed_time(e) * 1000

        if isinstance(blocks, torch.Tensor):         # --- fetch ---
            uniq = blocks.unique().tolist()
            for b in uniq:
                k, v = self.hot[layer_id][b]          # 已在 hot
                bytes_ = (k.numel()+v.numel()) * k.element_size()
                emit(tag_fetch.format(layer_id, b), "kv",
                     host_ns, dur_us/len(uniq), bytes_)
        else:                                        # --- push ---
            k, v = a[0], a[1]                        # push(k,v,...)
            bytes_ = (k.numel()+v.numel()) * k.element_size()
            emit(tag_push.format(layer_id, blocks), "kv",
                 host_ns, dur_us, bytes_)
        return out
    return inner

KVOffloader.push  = kv_wrapper(KVOffloader.push ,
                               "KV_PUSH_L{}_B{}",
                               "KV_FETCH_L{}_B{}")  # second tag unused here
KVOffloader.fetch = kv_wrapper(KVOffloader.fetch,
                               "KV_PUSH_L{}_B{}",   # unused
                               "KV_FETCH_L{}_B{}")

# ---------- Compute wrappers ----------

def make_compute_wrapper(name: str, layer_id: int, orig_forward):
    @functools.wraps(orig_forward)
    def _wrapper(*a, **kw):
        t0 = time.perf_counter_ns()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        out = orig_forward(*a, **kw)
        e.record(); torch.cuda.current_stream().synchronize()
        dur_us = s.elapsed_time(e) * 1000
        emit(f"{name}_L{layer_id}", "compute", t0, dur_us)
        return out
    return _wrapper

# ---------- Main ----------

def main():
    global csv_writer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--max-gen-len", type=int, default=64)
    args = ap.parse_args()

    # ---------------- Load model on CPU ----------------
    llama = LLaMA.build(args.model_path, load_model=True, device="cpu")

    # 给所有 Linear/Embedding 包 weight-upload 计时
    for n, m in llama.model.named_modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
            tag = n.replace(".", "_")
            wrap_weight_upload(m, tag)

    # ---------------- Move to CUDA & update offloader device ----------------
    llama.model.to(args.device)
    llama.args.device = args.device
    for ly, blk in enumerate(llama.model.layers):
        off = blk.attention.offloader
        off.device = args.device
        if off.copy_stream is None:
            off.copy_stream = torch.cuda.Stream(device=args.device)

    # ---------------- Wrap compute fwd ----------------
    for idx, blk in enumerate(llama.model.layers):
        blk.attention.forward    = make_compute_wrapper("MHA", idx, blk.attention.forward)
        blk.feed_forward.forward = make_compute_wrapper("FFN", idx, blk.feed_forward.forward)

    # ---------------- Run one generation ----------------
    prompts = [args.prompt]
    csv_path = "trace_summary.csv"
    with open(csv_path, "w", newline="") as fcsv:
        csv_writer = csv.writer(fcsv)
        csv_writer.writerow(["event", "cat", "bytes", "dur_us"])
        _ = llama.text_completion(prompts, max_gen_len=args.max_gen_len)

    # ---------------- Dump trace ----------------
    with open("trace.json", "w") as f:
        json.dump(trace_events, f)
    print("[TRACE] trace.json")
    print(f"[CSV  ] {csv_path}")

if __name__ == "__main__":
    main()
