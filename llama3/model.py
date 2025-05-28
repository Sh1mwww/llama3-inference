from typing import List

import torch
import torch.nn as nn

from .config import ModelArgs
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
        self.layers = nn.ModuleList(
            [EncoderBlock(args, i) for i in range(args.n_layers)]
        )
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(
            args.dim // args.n_heads,
            args.max_seq_len * 2,
            device=args.device,
            theta=args.rope_theta,
        )

        # 外部使用的 profiling 容器
        self.kv_times: List[float] = [0.0] * args.n_layers
        self.attn_times: List[float] = [0.0] * args.n_layers

    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        """
        tokens: (B,1)  - 每次只喂一个 token，保持与原实现一致
        """
        bsz, seqlen = tokens.shape
        assert seqlen == 1, "一次只能处理一个 token"

        h = self.embed_tokens(tokens)            # (B,1,D)
        freqs = self.freqs_complex[start_pos : start_pos + 1].to(h.device)

        for idx, layer in enumerate(self.layers):
            h = layer(h, start_pos, freqs)
            self.kv_times[idx] = layer.attention.kv_elapsed_time
            self.attn_times[idx] = layer.attention.attn_time

        h = self.norm(h)
        return self.output(h).float()
