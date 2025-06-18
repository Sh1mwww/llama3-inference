
from __future__ import annotations
from typing import List
import torch

BLOCK = 64  # tokens / block

class KVOffloader:
    """Off-GPU KV cache with attention-based Top-K fetch.

    * push():  save K/V block to pinned CPU DRAM.
    * fetch(): load a *set* of blocks to GPU HBM.
    * update_importances(): 由 SelfAttention 将每个 block 的注意力得分写入。
    * topk_blocks()
    """

    def __init__(self,
                 layers:int,
                 heads:int,
                 dim:int,
                 max_seq:int,
                 max_batch:int,
                 device:str,
                 dtype_bytes:int):
        n_blocks = (max_seq + BLOCK - 1) // BLOCK
        self.k_cpu = [[None for _ in range(n_blocks)] for _ in range(layers)]
        self.v_cpu = [[None for _ in range(n_blocks)] for _ in range(layers)]
        self.importance = [[0.0 for _ in range(n_blocks)] for _ in range(layers)]  # 累积注意力

        self.device = device
        self.dtype_bytes = dtype_bytes
        self.copy_stream = (torch.cuda.Stream(device=device)
                            if device.startswith("cuda") else None)
        self.heads = heads
        self.dim = dim
        self.max_batch = max_batch
        self.n_blocks = n_blocks

    # ---------------- internal -----------------
    def _alloc_block(self, layer:int, blk:int, batch_sz:int):
        if self.k_cpu[layer][blk] is None:
            shape = (batch_sz, self.heads, self.dim)
            self.k_cpu[layer][blk] = torch.empty(*shape, dtype=torch.float16,
                                                 pin_memory=True)
            self.v_cpu[layer][blk] = torch.empty_like(self.k_cpu[layer][blk])

    # ---------------- public API --------------
    def push(self, layer:int, blk:int, k:torch.Tensor, v:torch.Tensor):
        """Save K/V of *one* block from HBM → DRAM."""
        self._alloc_block(layer, blk, k.size(0))
        self.k_cpu[layer][blk][:k.size(0)].copy_(k, non_blocking=True)
        self.v_cpu[layer][blk][:v.size(0)].copy_(v, non_blocking=True)
        # 新 block 给一个极小初始权重，防止意外被忽略
        self.importance[layer][blk] = max(self.importance[layer][blk], 1e-6)

    def fetch(self, layer:int, blocks:torch.Tensor):
        """Return concat-K/V of given *unique* block indices (ascending)."""
        uniq = blocks.unique().tolist()
        k_parts, v_parts = [], []
        stream = self.copy_stream or torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            for blk in uniq:
                if self.k_cpu[layer][blk] is None:
                    raise RuntimeError(f"[KVOffloader] block {blk} not pushed (layer {layer})")
                k_parts.append(self.k_cpu[layer][blk].to(self.device, non_blocking=True))
                v_parts.append(self.v_cpu[layer][blk].to(self.device, non_blocking=True))
        if stream is not torch.cuda.current_stream():
            torch.cuda.current_stream().wait_stream(stream)
        return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)

    # ------------- attention importance -------------
    def update_importances(self,
                           layer:int,
                           block_indices:List[int],
                           block_scores:List[float],
                           momentum:float = 0.9):
        """EMA 累积注意力得分 (momentum 越大越偏向历史)。"""
        imp = self.importance[layer]
        for idx, score in zip(block_indices, block_scores):
            imp[idx] = momentum * imp[idx] + (1.0 - momentum) * score

    def topk_blocks(self, layer:int, k:int):
        """Return **ascending** indices of Top-k blocks by importance."""
        imp = self.importance[layer]
        ranked = sorted(range(self.n_blocks), key=lambda i: imp[i], reverse=True)
        chosen = [i for i in ranked if imp[i] > 0][:k]
        return sorted(chosen)
