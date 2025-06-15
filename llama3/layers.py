# llama3/layers.py  — async-weights + KV window (保留原逻辑，以注释方式标记)
# ============================================================
# 在原版基础上：
#   • 每层新增 weight_stream,权重按需异步拷贝 GPU
#   • KV Block 仍用 KVOffloader,但 fetch 在独立 stream 已计时
#   • 用 STAT & cuda_timer 统计 weights_hbm / kv_fetch / attn / ffn
# ============================================================

import math
from typing import Optional, Tuple, List
import torch, torch.nn as nn, torch.nn.functional as F

from .config import ModelArgs
from .kv_offload import KVOffloader, BLOCK

from contextlib import contextmanager

# ---------- timing util ----------
STAT = {
    "weights_hbm_us": 0,
    "kv_fetch_us": 0,
    "attn_us": 0,
    "ffn_us": 0,
}
@contextmanager
def cuda_timer(key:str):
    if not torch.cuda.is_available():
        yield; return
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); yield; e.record(); torch.cuda.synchronize()
    STAT[key] += int(s.elapsed_time(e) * 1000)

# ---------- helpers (RMSNorm / rotary / repeat_kv 保留) ----------
class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float=1e-6):
        super().__init__(); self.weight=nn.Parameter(torch.ones(dim)); self.eps=eps
    def forward(self,x:torch.Tensor):
        n=x.pow(2).mean(-1,keepdim=True).add(self.eps).rsqrt(); return self.weight*(x*n)

def precompute_theta_pos_frequencies(head_dim:int,seq_len:int,device:str,theta:float=10000.0):
    assert head_dim%2==0
    theta_i=1.0/(theta**(torch.arange(0,head_dim,2).float()/head_dim)).to(device)
    m=torch.arange(seq_len,device=device)
    freqs=torch.outer(m,theta_i)
    return torch.polar(torch.ones_like(freqs),freqs)

def apply_rotary_embeddings(x:torch.Tensor,freqs_complex:torch.Tensor)->torch.Tensor:
    b,l,h,d=x.shape
    x_=x.float().reshape(b,l,h,d//2,2)
    x_complex=torch.view_as_complex(x_)
    freqs_complex=freqs_complex.unsqueeze(0).unsqueeze(2)
    out=torch.view_as_real(x_complex*freqs_complex)
    return out.reshape(b,l,h,d).type_as(x)

def repeat_kv(x:torch.Tensor,n_rep:int)->torch.Tensor:
    if n_rep==1: return x
    b,t,h,d=x.shape
    return x[:,:,:,None,:].expand(b,t,h,n_rep,d).contiguous().view(b,t,h*n_rep,d)

# ---------- SelfAttention (modified) ----------
class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep      = self.n_heads_q // self.n_kv_heads
        self.head_dim   = args.dim // args.n_heads

        self.topk_blk   = args.window_blk          # 参数重用：含义改为 Top‑K 个 block
        self.device     = args.device
        self.is_cuda    = str(self.device).startswith("cuda") and torch.cuda.is_available()

        # Linear 权重仍初始化在 CPU
        self.wq = nn.Linear(args.dim, self.n_heads_q * self.head_dim, bias=False, device="cpu")
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False, device="cpu")
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False, device="cpu")
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False, device="cpu")
        self.weight_stream = torch.cuda.Stream(device=args.device) if self.is_cuda else None

        # KV offloader
        self.block_sz = BLOCK
        self.offloader = KVOffloader(layers=args.n_layers,
                                      heads=self.n_kv_heads,
                                      dim=self.head_dim,
                                      max_seq=args.max_seq_len,
                                      max_batch=args.max_batch_size,
                                      device=args.device,
                                      dtype_bytes=self.wq.weight.element_size())

        self.layer_id = -1   # 将由 EncoderBlock 设置
        self.kv_elapsed_time = -1.0
        self.attn_time = -1.0
    # ---------- utils ----------
    def _ensure_weights_cuda(self):
        if self.wq.weight.is_cuda: return
        with torch.cuda.stream(self.weight_stream):
            for l in (self.wq,self.wk,self.wv,self.wo):
                l.weight.data = l.weight.data.to("cuda", non_blocking=True)
        with cuda_timer("weights_hbm_us"):
            torch.cuda.current_stream().wait_stream(self.weight_stream)

    # ---------- forward ----------
    def forward(self, x:torch.Tensor, start_pos:int, freqs_complex:torch.Tensor)->torch.Tensor:
        bsz, seqlen, _ = x.shape
        self._ensure_weights_cuda()

        # QKV projection & rotary
        q = self.wq(x).view(bsz, seqlen, self.n_heads_q, self.head_dim)
        k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        q = apply_rotary_embeddings(q, freqs_complex)
        k = apply_rotary_embeddings(k, freqs_complex)

        # ---- push 当前 block ----
        blk_idx = start_pos // self.block_sz
        self.offloader.push(self.layer_id, blk_idx, k.squeeze(1), v.squeeze(1))

        # ---- Top‑K fetch ----
        blocks = self.offloader.topk_blocks(self.layer_id, self.topk_blk)
        if blk_idx not in blocks:
            blocks.append(blk_idx)
        blocks = sorted(blocks)
        needed = torch.tensor(blocks, device=x.device)

        with cuda_timer("kv_fetch_us"):
            k_full, v_full = self.offloader.fetch(self.layer_id, needed)

        # shape to (B, Hkv, T, D)
        if k_full.dim() == 3:
            k_full = k_full.permute(1,0,2).unsqueeze(0)
            v_full = v_full.permute(1,0,2).unsqueeze(0)
        k_full = k_full.to(q.dtype)
        v_full = v_full.to(q.dtype)
        if self.n_heads_q != self.n_kv_heads:
            k_full = k_full.repeat_interleave(self.n_rep, dim=1)
            v_full = v_full.repeat_interleave(self.n_rep, dim=1)

        # ---- attention compute ----
        q = q.transpose(1,2)  # (B,H,Tq,D)
        with cuda_timer("attn_us"):
            scores = torch.matmul(q, k_full.transpose(2,3)) / math.sqrt(self.head_dim)  # (B,H,Tq,Tkv)
            scores = torch.softmax(scores, dim=-1)
            out    = torch.matmul(scores, v_full)

        out = out.transpose(1,2).reshape(bsz, seqlen, -1)

        # ---- 更新 block importance (attention based) ----
        with torch.no_grad():
            token_imp = scores.mean(dim=(0,1,2))  # (Tkv,)
            block_scores: List[float] = []
            for i in range(len(blocks)):
                s = i * self.block_sz
                e = s + self.block_sz
                block_scores.append(float(token_imp[s:e].sum().item()))
            self.offloader.update_importances(self.layer_id, blocks, block_scores)

        return self.wo(out)

    
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = int(4 * args.dim * 2 / 3)
        if args.ffn_dim_multiplier:
            hidden_dim = int(hidden_dim * args.ffn_dim_multiplier)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        # linear 层仍 CPU 初始化
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False, device="cpu")
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False, device="cpu")
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False, device="cpu")
        self.device = args.device
        self.is_cuda = str(self.device).startswith("cuda") and torch.cuda.is_available()
        self.weight_stream = torch.cuda.Stream(device=args.device) if self.is_cuda else None

    def _ensure_weights_cuda(self):
        if self.w1.weight.is_cuda:
            return
        with torch.cuda.stream(self.weight_stream):
            for l in (self.w1, self.w2, self.w3):
                l.weight.data = l.weight.data.to("cuda", non_blocking=True)
        with cuda_timer("weights_hbm_us"):
            torch.cuda.current_stream().wait_stream(self.weight_stream)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_weights_cuda()
        with cuda_timer("ffn_us"):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        self.attention.layer_id = layer_id
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        h = x + self.attention(self.attn_norm(x), start_pos, freqs_complex)
        return h + self.feed_forward(self.ffn_norm(h))

# ============================================================
# 以上为重写后版本，以下原版关键逻辑已被替换，保留注释便于 diff
# ============================================================
### OLD: 原来自 SelfAttention.forward 的 weight.to() 逐行调用等均已合并到 _ensure_weights_cuda
### OLD: 原 KV cache 相关注释段落保留(见上原 file)
