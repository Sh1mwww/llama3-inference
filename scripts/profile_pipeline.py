"""Comprehensive profiler for LLaMA‑3 inference.
(保持原文件接口，只通过 **注释+插入** 修补计时逻辑)
Changes v2
===========
* 不再覆盖 layers.STAT 内的數值；所有計時以 layers.py 寫入為準。
* `weights_hbm_us` 現在在 `with timer()` 裡 **累加** 而非覆蓋。
* 移除對 `attn_us / ffn_us` 的負數覆蓋；仍打印 layers 中統計結果。
* 其餘函式接口與 CLI 參數保持不變。
"""
import math
import argparse, time, pathlib
from contextlib import contextmanager
import torch, csv, os
from llama3.generator import LLaMA
from llama3.layers import STAT     
from llama3.kv_offload import KVOffloader, BLOCK 

# ---------- utility ----------
@contextmanager
def timer(key: str):
    """累加 μs 到 layers.STAT (若 GPU 可用)"""
    if not torch.cuda.is_available():
        yield; return
    s,e = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    s.record(); yield; e.record(); torch.cuda.synchronize()
    STAT[key] = STAT.get(key,0) + int(s.elapsed_time(e)*1000)

def sizeof_fmt(num):
    for unit in ("B","KB","MB","GB","TB"):
        if num < 1024: return f"{num:.2f} {unit}"
        num /= 1024
    return f"{num:.2f} PB"

def param_bytes(model):
    return sum(p.numel()*p.element_size() for p in model.parameters())

# ---------- KVOffloader patch 加计时 ----------
import llama3.kv_offload as kvmod
if not hasattr(kvmod,"_patched_profile"):
    orig_push, orig_fetch = kvmod.KVOffloader.push, kvmod.KVOffloader.fetch
    def push_patch(self,*a,**k):
        with timer("kv_push_us"):
            return orig_push(self,*a,**k)
    def fetch_patch(self,*a,**k):
        with timer("kv_fetch_us"):
            return orig_fetch(self,*a,**k)
    kvmod.KVOffloader.push  = push_patch
    kvmod.KVOffloader.fetch = fetch_patch
    kvmod._patched_profile  = True

# ---------- FeedForward patch 加计时 ----------
import llama3.layers as lyrs
if not hasattr(lyrs,"_ffn_patch_profile"):
    orig_ffn = lyrs.FeedForward.forward
    def ffn_patch(self,x):
        with timer("ffn_us"):
            return orig_ffn(self,x)
    lyrs.FeedForward.forward = ffn_patch
    lyrs._ffn_patch_profile = True

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path",required=True)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prompt",default="Hello")
    ap.add_argument("--max-gen-len",type=int,default=64)
    ap.add_argument("--window-blk",type=int,default=8,help="滑窗宽度(block)")
    ap.add_argument("--csv",help="追加结果到 CSV")
    args = ap.parse_args()

    ckpt = pathlib.Path(args.model_path)

    # ----- checkpoint → CPU -----
    with timer("weights_cpu_us"):
        llama = LLaMA.build(ckpt,load_model=True,device="cpu")
        for layer in llama.model.layers:
            layer.attention.offloader.window_blk = args.window_blk

    weights_file_bytes   = sum(f.stat().st_size for f in ckpt.glob("*.pth"))
    weights_tensor_bytes = param_bytes(llama.model)

    # ----- weights → GPU -----
    if args.device.startswith("cuda"):
        with timer("weights_hbm_us"):
            llama.model.to(args.device)
    else:
        llama.model.to(args.device)

    llama.args.device = args.device
    for blk in llama.model.layers:
        off = blk.attention.offloader
        off.device = args.device
        if torch.cuda.is_available():
            off.copy_stream = torch.cuda.Stream(device=args.device)

    # ----- inference -----
    _ = llama.text_completion([args.prompt],max_gen_len=args.max_gen_len)

    # ----- KV size -----
    L   = llama.args.n_layers
    Hk  = llama.args.n_kv_heads or llama.args.n_heads
    D   = llama.args.dim // llama.args.n_heads
    block_sz = BLOCK
    dtype_bytes = llama.model.layers[0].attention.wk.weight.element_size()
    prompt_len = len(llama.tokenizer.encode(args.prompt))
    seq_len    = prompt_len + args.max_gen_len
    n_blocks   = (seq_len + block_sz - 1) // block_sz
    window_blk = args.window_blk

    print(f"L={L}, Hk={Hk}, D={D}, dtype_bytes={dtype_bytes}, block_sz={block_sz}, n_blocks={n_blocks}, window_blk={window_blk}, seq_len={seq_len}")

    kv_dram_bytes = L * 1 * n_blocks * block_sz * 2 * Hk * D * dtype_bytes
    kv_hbm_bytes  = L * 1 * window_blk * block_sz * 2 * Hk * D * dtype_bytes
    kv_push_bytes = L * 1 * 2 * Hk * D * seq_len * dtype_bytes
    total_tokens = 1 * seq_len
    kv_bytes_per_token = kv_push_bytes / total_tokens
    kv_bytes_per_layer_per_token = 2 * Hk * D * dtype_bytes

    # ---------- report ----------
    print("\n===== Pipeline Profile =====")
    print(f"Weight files           : {sizeof_fmt(weights_file_bytes)}")
    print(f"Weight tensors         : {sizeof_fmt(weights_tensor_bytes)}")
    print(f"Dtype_bytes            : {sizeof_fmt(dtype_bytes)}")
    print(f"KV-DRAM capacity       : {sizeof_fmt(kv_dram_bytes)}")              # 最大 DRAM 存储
    print(f"KV-HBM peak            : {sizeof_fmt(kv_hbm_bytes)}")               # GPU HBM 峰值
    print(f"KV pushed total        : {sizeof_fmt(kv_push_bytes)}")             # 累计 push 字节
    print(f"KV per token (total)   : {sizeof_fmt(kv_bytes_per_token)}")        # 平均每 token
    print(f"KV per layer per token : {sizeof_fmt(kv_bytes_per_layer_per_token)}")  # 单层单 token
    print("-------------------------------")

    us=STAT; ms=lambda k: us.get(k,0)/1000.0
    print(f"Load weights → CPU     : {ms('weights_cpu_us'):.1f} ms")
    print(f"Transfer weights → HBM : {ms('weights_hbm_us'):.1f} ms")
    print(f"KV save (DRAM)         : {ms('kv_push_us'):.1f} ms")
    print(f"KV load (HBM)          : {ms('kv_fetch_us'):.1f} ms")
    print(f"MHA compute            : {ms('attn_us'):.1f} ms")
    print(f"FFN compute            : {ms('ffn_us'):.1f} ms")
    total_io  = ms('weights_cpu_us')+ms('weights_hbm_us')+ms('kv_push_us')+ms('kv_fetch_us')
    total_cmp = ms('attn_us')+ms('ffn_us')
    print("-------------------------------")
    print(f"Total I/O time         : {total_io:.1f} ms")
    print(f"Total compute time     : {total_cmp:.1f} ms\n")

    # ----- CSV -----
    if args.csv:
        hdr=["model","prompt_tok","gen_tok","window_blk",*us.keys(),"weights_file_B","weights_tensor_B","kv_total_B"]
        row=[ ckpt.name, len(llama.tokenizer.encode(args.prompt)), args.max_gen_len, args.window_blk,
              *[us.get(k,0) for k in us], weights_file_bytes,weights_tensor_bytes,kv_total_bytes ]
        need=not os.path.exists(args.csv)
        with open(args.csv,"a",newline="") as f:
            wr=csv.writer(f);
            if need: wr.writerow(hdr)
            wr.writerow(row)
        print(f"[CSV] appended → {args.csv}")

if __name__=="__main__":
    main()
