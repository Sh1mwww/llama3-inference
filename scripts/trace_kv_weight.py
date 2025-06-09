#!/usr/bin/env python
"""
trace_kv_weight.py v1.3
-----------------------
✓ 权重上传 bytes + dur
✓ dim_head 对齐 KV bytes
✓ KV push / fetch bytes + dur
✓ MHA / FFN dur
✓ 统计本次推理 DRAM→GPU 的 KV fetch 次数
输出:
  trace.json             (Chrome://tracing)
  trace_summary.csv      (event,cat,bytes,dur_us)
  终端打印 fetch 次数
"""
import os, json, csv, time, argparse, functools, torch, pathlib
from llama3.generator import LLaMA
from llama3.kv_offload import KVOffloader

# ---------- trace buffers ----------
trace, csv_rows = [], []
kv_fetch_counter = 0            

def emit(name, cat, start_ns, dur_us, bytes_=0):
    trace.append({
        "ph":"X","name":name,"cat":cat,
        "ts":start_ns/1_000, "dur":dur_us,
        "pid":os.getpid(),"tid":0,
        "args":{"bytes":bytes_, "us":dur_us}
    })
    csv_rows.append([name, cat, bytes_, f"{dur_us:.1f}"])

# ---------- WRAP weight -> cuda ----------
def wrap_to_cuda(module, tag):
    orig = module.to
    bytes_ = sum(p.numel()*p.element_size() for p in module.parameters())
    def _to(*a, **kw):
        device = a[1] if len(a)>1 else kw.get("device")
        if device is None or module.weight.is_cuda:
            return orig(*a, **kw)
        host_ns=time.perf_counter_ns(); s,e = torch.cuda.Event(True),torch.cuda.Event(True)
        s.record(); out=orig(*a, **kw); e.record(); torch.cuda.synchronize()
        emit(f"WEIGHT_UPLOAD_{tag}","weight",host_ns,s.elapsed_time(e)*1000,bytes_)
        return out
    module.to = _to

# ---------- WRAP KV push / fetch ----------
orig_push, orig_fetch = KVOffloader.push, KVOffloader.fetch
def push_wrapper(self, layer, blk, k, v):
    host_ns=time.perf_counter_ns(); s,e = torch.cuda.Event(True),torch.cuda.Event(True)
    s.record(); out=orig_push(self,layer,blk,k,v); e.record(); torch.cuda.current_stream().synchronize()
    emit(f"KV_PUSH_L{layer}_B{blk}","kv",host_ns,s.elapsed_time(e)*1000,
         (k.numel()+v.numel())*k.element_size())
    return out

def fetch_wrapper(self, layer, blocks):
    global kv_fetch_counter
    host_ns=time.perf_counter_ns(); s,e = torch.cuda.Event(True),torch.cuda.Event(True)
    s.record(); out=orig_fetch(self,layer,blocks); e.record(); torch.cuda.current_stream().synchronize()
    dur_us=s.elapsed_time(e)*1000
    uniq=blocks.unique().tolist(); per=dur_us/len(uniq)
    kv_fetch_counter += len(uniq)       # ★ 累加 DRAM fetch 次数
    for b in uniq:
        k,v=self.hot[layer][b]
        emit(f"KV_FETCH_L{layer}_B{b}","kv",host_ns,per,(k.numel()+v.numel())*k.element_size())
    return out

KVOffloader.push  = push_wrapper
KVOffloader.fetch = fetch_wrapper

# ---------- wrap compute ----------
def make_comp(tag, idx, fwd):
    @functools.wraps(fwd)
    def _w(*a, **kw):
        host_ns=time.perf_counter_ns(); s,e=torch.cuda.Event(True),torch.cuda.Event(True)
        s.record(); out=fwd(*a, **kw); e.record(); torch.cuda.synchronize()
        emit(f"{tag}_L{idx}","compute",host_ns,s.elapsed_time(e)*1000)
        return out
    return _w

# ---------------- MAIN ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-path",required=True)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--prompt",default="Hello")
    ap.add_argument("--max-gen-len",type=int,default=64)
    args=ap.parse_args()

    llama=LLaMA.build(args.model_path,load_model=True,device="cpu")

    # ------- dim_head 对齐 -------
    dim_head = llama.args.dim // llama.args.n_heads
    for blk in llama.model.layers:
        off=blk.attention.offloader
        off.dim = dim_head
        off.dtype_sz = 2
        off.heads = llama.args.n_kv_heads
        off.max_batch = llama.args.max_batch_size

    # ------- wrap weight modules -------
    for name,mod in llama.model.named_modules():
        if isinstance(mod,(torch.nn.Linear,torch.nn.Embedding)):
            wrap_to_cuda(mod,name)

    # ------- wrap compute -------
    for idx,blk in enumerate(llama.model.layers):
        blk.attention.forward    = make_comp("MHA",idx,blk.attention.forward)
        blk.feed_forward.forward = make_comp("FFN",idx,blk.feed_forward.forward)

    # ------- move to GPU -------
    llama.model.to(args.device)
    llama.args.device=args.device
    for blk in llama.model.layers:                # sync offloader device
        off=blk.attention.offloader
        off.device=args.device
        if off.copy_stream is None and args.device.startswith("cuda"):
            off.copy_stream=torch.cuda.Stream(device=args.device)

    # ------- run inference -------
    _=llama.text_completion([args.prompt],max_gen_len=args.max_gen_len)

    # ------- dump -------
    with open("trace.json","w") as f: json.dump(trace,f)
    with open("trace_summary.csv","w",newline="") as f:
        csv.writer(f).writerows([["event","cat","bytes","dur_us"],*csv_rows])

    print(f"KV DRAM fetch count: {kv_fetch_counter}")
    print("trace.json  &  trace_summary.csv  written")

if __name__=="__main__":
    main()
