from typing import List, Dict, Any   

import torch
import torch.nn as nn

from .config import ModelArgs,  LayerInfo
from .layers import (
    RMSNorm,
    EncoderBlock,
    precompute_theta_pos_frequencies,
)


class Transformer(nn.Module):
    """
    纯粹的 Forward 计算与 KV profiling，**不**负责权重载入 / 采样
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size

        self.embed_tokens = nn.Embedding(args.vocab_size, args.dim)
        # self.layers = nn.ModuleList(
        #     [EncoderBlock(args, i) for i in range(args.n_layers)]
        # )
        
        self.layers = nn.ModuleList()
        self.layer_infos: List[Dict[str, Any]] = []
        for i in range(args.n_layers):
            blk = EncoderBlock(args, i)
            self.layers.append(blk)
            self.layer_infos.append({
                "layer_id": i,
                "block": blk,          # 方便从 info 直接拿模块
                "extra": {}            # 留给后续任意字段
            })     
               
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(
            args.dim // args.n_heads,
            args.max_seq_len * 2,
            device=args.device,
            theta=args.rope_theta,
        )

        self.kv_times: List[float] = [0.0] * args.n_layers
        self.attn_times: List[float] = [0.0] * args.n_layers

    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        """
        tokens: (B, seqlen)  - 支持多batch和多token处理
        在生成阶段通常 seqlen=1，但支持更大的值用于预填充
        """
        bsz, seqlen = tokens.shape
        # 移除硬编码限制，支持seqlen > 1
        # assert seqlen == 1, "一次只能处理一个 token"

        h = self.embed_tokens(tokens)            # (B, seqlen, D)
        freqs = self.freqs_complex[start_pos : start_pos + seqlen].to(h.device)

        for idx, info in enumerate(self.layer_infos):
            blk = info["block"]                  # EncoderBlock
            h = blk(h, start_pos, freqs)
            self.kv_times[idx]  = blk.attention.kv_elapsed_time
            self.attn_times[idx] = blk.attention.attn_time

        h = self.norm(h)
        return self.output(h).float()
