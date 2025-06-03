#!/usr/bin/env python
"""
trace_kv_weight.py  —  Chrome-trace + CSV profiler  (fixed v1.2)
---------------------------------------------------------------
✓ 权重上传：真实 bytes + 时间
✓ KV push / fetch：按 dim_head 计算 bytes
✓ MHA / FFN 计算：记录持续时间
输出：
    • trace.json           —  Chrome://tracing
    • trace_summary.csv    —  event,cat,bytes,dur_us
"""
import os, json, csv, time, argparse, functools, torch
from llama3.generator import LLaMA
from llama3.kv_offload import KVOffloader

# ========== Trace containers ==========
trace, csv_rows = [], []

def emit(name, cat, start_ns, dur_us, bytes_=0):
    trace.append({
        "ph": "X",
        "name": name,
        "cat": cat,
        "ts" : start_ns / 1_000,    # µs
        "dur": dur_us,
        "pid": os.getpid(),
        "args": {"bytes": bytes_, "us": dur_us}
    })
    csv_rows.append([name, cat, bytes_, f"{dur_us:.1f}"])

# ---------- 权重上传包装 ----------
def wrap_to_cuda(module, tag):
    orig_to = module.to
    bytes_  = sum(p.numel()*p.element_size() for p in module.parameters())

    def _to(*a, **kw):
        device = a[1] if len(a) > 1 else kw.get("device", None)
        if device is None or module.weight.is_cuda:
            return orig_to(*a, **kw)          # 已在 GPU 或仅 dtype 转换
        host_ns = time.perf_counter_ns()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        out = orig_to(*a, **kw)               # 真正搬运
        e.record(); torch.cuda.synchronize()
        emit(f"WEIGHT_UPLOAD_{tag}", "weight",
             host_ns, s.elapsed_time(e)*1000, bytes_)
        return out
    module.to = _to

# ---------- KV push / fetch 包装 ----------
orig_push,  orig_fetch = KVOffloader.push, KVOffloader.fetch

def push_wrapper(self, layer, blk, k, v):
    host_ns = time.perf_counter_ns()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    out = orig_push(self, layer, blk, k, v)
    e.record(); torch.cuda.current_stream().synchronize()
    dur_us = s.elapsed_time(e)*1000
    bytes_ = (k.numel()+v.numel())*k.element_size()
    emit(f"KV_PUSH_L{layer}_B{blk}", "kv", host_ns, dur_us, bytes_)
    return out

def fetch_wrapper(self, layer, blocks):
    host_ns = time.perf_counter_ns()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    out = orig_fetch(self, layer, blocks)
    e.record(); torch.cuda.current_stream().synchronize()
    dur_us = s.elapsed_time(e)*1000
    uniq = blocks.unique().tolist()
    per = dur_us/len(uniq)
    for b in uniq:
        k, v = self.hot[layer][b]
        bytes_ = (k.numel()+v.numel())*k.element_size()
        emit(f"KV_FETCH_L{layer}_B{b}", "kv", host_ns, per, bytes_)
    return out

KVOffloader.push  = push_wrapper
KVOffloader.fetch = fetch_wrapper

# ---------- MHA / FFN 计算包装 ----------
def make_comp(tag, idx, fwd):
    @functools.wraps(fwd)
    def _wrapper(*a, **kw):
        host_ns = time.perf_counter_ns()
        s,e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); out = fwd(*a, **kw); e.record(); torch.cuda.synchronize()
        emit(f"{tag}_L{idx}", "compute", host_ns, s.elapsed_time(e)*1000)
        return out
    return _wrapper

# ---------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--max-gen-len", type=int, default=64)
    args = ap.parse_args()

    llama = LLaMA.build(args.model_path, load_model=True, device="cpu")

    # ---- 对齐 dim_head → KVOffloader ----
    dim_head = llama.args.dim // llama.args.n_heads
    for blk in llama.model.layers:
        off = blk.attention.offloader
        off.dim         = dim_head
        off.dtype_bytes = 2                          # fp16
        off.max_batch   = llama.args.max_batch_size

    # ---- wrap weight modules ----
    for name, mod in llama.model.named_modules():
        if isinstance(mod, (torch.nn.Linear, torch.nn.Embedding)):
            wrap_to_cuda(mod, name)

    # ---- wrap compute ----
    for idx, blk in enumerate(llama.model.layers):
        blk.attention.forward    = make_comp("MHA", idx, blk.attention.forward)
        blk.feed_forward.forward = make_comp("FFN", idx, blk.feed_forward.forward)

    # ---- 迁移模型到 GPU（触发权重上传） ----
    llama.model.to(args.device)
    llama.args.device = args.device
    
    for blk in llama.model.layers:
        off = blk.attention.offloader
        off.device = args.device
        if off.copy_stream is None and args.device.startswith("cuda"):
            off.copy_stream = torch.cuda.Stream(device=args.device)
            
    # ---- 生成一次以触发 KV 推/取 & 计算 ----
    _ = llama.text_completion([args.prompt], max_gen_len=args.max_gen_len)

    # ---- 输出 ----
    with open("trace.json", "w") as f:
        json.dump(trace, f)
    with open("trace_summary.csv", "w", newline="") as f:
        csv.writer(f).writerows([["event","cat","bytes","dur_us"], *csv_rows])
    print("=> trace.json & trace_summary.csv 已生成")

if __name__ == "__main__":
    main()
