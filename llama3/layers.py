import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelArgs
from .kv_offload import KVOffloader, BLOCK


# ---------- 基础工具 ----------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x * norm)


def precompute_theta_pos_frequencies(
    head_dim: int,
    seq_len: int,
    device: str,
    theta: float = 10000.0,
):
    """生成 RoPE 用到的复数频率表"""
    assert head_dim % 2 == 0, "head_dim 必须为偶数"
    theta_i = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    theta_i = theta_i.to(device)                                      # (D/2,)

    m = torch.arange(seq_len, device=device)                          # (L,)
    freqs = torch.outer(m, theta_i)                                   # (L, D/2)
    return torch.polar(torch.ones_like(freqs), freqs)                 # complex64


def apply_rotary_embeddings(
    x: torch.Tensor, freqs_complex: torch.Tensor
) -> torch.Tensor:
    """
    x: (B, L, H, D)
    freqs_complex: (L, D/2) complex
    """
    b, l, h, d = x.shape
    x_ = x.float().reshape(b, l, h, d // 2, 2)
    x_complex = torch.view_as_complex(x_)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)           # (1,L,1,D/2)
    out = torch.view_as_real(x_complex * freqs_complex)               # (B,L,H,D/2,2)
    return out.reshape(b, l, h, d).type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, t, h, d = x.shape
    return (
        x[:, :, :, None, :].expand(b, t, h, n_rep, d).contiguous()
        .view(b, t, h * n_rep, d)
    )


# ---------- 子层 ----------

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, self.n_heads_q * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # # KV cache 预分配
        # self.register_buffer(
        #     "cache_k",
        #     torch.zeros(
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_kv_heads,
        #         self.head_dim,
        #         device=args.device,
        #     ),
        # )
        # self.register_buffer(
        #     "cache_v",
        #     torch.zeros(
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_kv_heads,
        #         self.head_dim,
        #         device=args.device,
        #     ),
        # )
        
        
        # CPU-offload 管理器（每层独立）
        self.block_sz = BLOCK
        self.offloader = KVOffloader(
            layers=args.n_layers, 
            heads=self.n_kv_heads, 
            dim=self.head_dim,
            max_seq=args.max_seq_len, 
            max_batch=args.max_batch_size,
            hot_window=32,  # 保留最近 32 块 ≈ 2048 token
            device=args.device,
        )
        
        self.kv_elapsed_time = -1.0
        self.attn_time = -1.0

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        # ---- Q ----
        q = self.wq(x).view(bsz, seqlen, self.n_heads_q, self.head_dim)
        q = apply_rotary_embeddings(q, freqs_complex)

        # ---- KV & 计时 ----
        if torch.cuda.is_available():
            kv_start = torch.cuda.Event(enable_timing=True)
            kv_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            kv_start.record()

        k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        k = apply_rotary_embeddings(k, freqs_complex)

        # # 更新 cache
        # if bsz > self.cache_k.size(0):
        #     self.cache_k = torch.zeros(
        #     bsz, self.cache_k.size(1), self.cache_k.size(2), self.cache_k.size(3),
        #     device=self.cache_k.device, dtype=self.cache_k.dtype
        # )
        # self.cache_v = torch.zeros_like(self.cache_k)
        # self.cache_k[:bsz, start_pos : start_pos + seqlen] = k
        # self.cache_v[:bsz, start_pos : start_pos + seqlen] = v
        
        
        # 按 BLOCK 写入 offloader
        # ◆ 只支持 seqlen==1(decode 阶段)或 seqlen==block_sz(prefill)
        k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        k = apply_rotary_embeddings(k, freqs_complex)

        blk_idx = start_pos //self.block_sz
        k_block = k.squeeze(1)   # (H, D)  —— 或 k[0, 0]
        v_block = v.squeeze(1)
        self.offloader.push(self.layer_id, blk_idx, k_block, v_block)

        if torch.cuda.is_available():
            kv_end.record()
            torch.cuda.synchronize()
            self.kv_elapsed_time = kv_start.elapsed_time(kv_end)

        # ---- Attention ----
        if torch.cuda.is_available():
            attn_start = torch.cuda.Event(enable_timing=True)
            attn_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            attn_start.record()

        # k_full = self.cache_k[:bsz, : start_pos + seqlen]
        # v_full = self.cache_v[:bsz, : start_pos + seqlen]
        # k_full = repeat_kv(k_full, self.n_rep)
        # v_full = repeat_kv(v_full, self.n_rep)
        
                # ----- **On-Demand** 取齐所需块 -----
                
        needed = torch.arange(0, start_pos + seqlen,
                             device=x.device) // self.block_sz
        k_full, v_full = self.offloader.fetch(self.layer_id, needed)

        k_full = k_full.view(-1, self.n_kv_heads, self.head_dim)  # (Nblk*B,H,D)
        v_full = v_full.view_as(k_full)

        k_full = repeat_kv(k_full[None, ...], self.n_rep)         # (1,Htot,T,D)
        v_full = repeat_kv(v_full[None, ...], self.n_rep)
        
        

        q = q.transpose(1, 2)                    # (B,H,T,D)
        k_full = k_full.transpose(1, 2)
        v_full = v_full.transpose(1, 2)

        scores = torch.matmul(q, k_full.transpose(2, 3))
        scores = torch.softmax(scores / math.sqrt(self.head_dim), dim=-1)
        out = torch.matmul(scores, v_full)       # (B,H,T,D)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        out = self.wo(out)

        if torch.cuda.is_available():
            attn_end.record()
            torch.cuda.synchronize()
            self.attn_time = attn_start.elapsed_time(attn_end)

        return out


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = int(4 * args.dim * 2 / 3)
        if args.ffn_dim_multiplier:
            hidden_dim = int(hidden_dim * args.ffn_dim_multiplier)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor
    ) -> torch.Tensor:
        h = x + self.attention(self.attn_norm(x), start_pos, freqs_complex)
        return h + self.feed_forward(self.ffn_norm(h))
