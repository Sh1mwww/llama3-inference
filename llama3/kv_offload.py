"""
KV 逐块冷热分层管理：
-   GPU 端保留最近 hot_window 个 block
-   其它 block 存在 pinned CPU DRAM, 按需异步拷回
"""
from __future__ import annotations
from typing import Tuple, Dict, List
from .config import ModelArgs
import torch

BLOCK = 64                   # token/块；与 vllm block table 一致

class KVOffloader:
    def __init__(self, 
                 layers: int, 
                 heads: int, 
                 dim: int,
                 max_seq: int, 
                 max_batch: int,
                 hot_window: int, 
                 device: str,
                 dtype_bytes: int):
        
        self.layers = layers
        self.hot_window = hot_window
        self.device = device
        self.heads = heads
        self.dtype_bytes = dtype_bytes
        self.max_batch = max_batch
        args = ModelArgs
        
        max_batch = args.max_batch_size

        n_blocks = (max_seq + BLOCK - 1) // BLOCK
        shape = (n_blocks, max_batch, heads, dim)

        # CPU pinned cache: {layer_id: Tensor}
        self.k_cpu = [torch.empty(*shape, dtype=torch.float16, pin_memory=True)
                    for _ in range(layers)]
        self.v_cpu = [t.clone() for t in self.k_cpu]
        
        # record which blocks are hot(GPU)；列表比位图快许多
        self.hot: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = [
            {} for _ in range(layers)
        ]

        # 单独的拷贝 stream
        # self.copy_stream = torch.cuda.Stream(device=device)
        
        if device.startswith("cuda"):
            self.copy_stream = torch.cuda.Stream(device=device)
        else:
            self.copy_stream = None

    # ---------- 接口 ----------
    def push(self, 
             layer: int, 
             block_idx: int,
             k: torch.Tensor, 
             v: torch.Tensor):
        """把新写入的 KV (B=1,H,D) 写入 GPU & CPU"""
        self.k_cpu[layer][block_idx, :k.size(0)].copy_(k, non_blocking=True)
        self.v_cpu[layer][block_idx, :v.size(0)].copy_(v, non_blocking=True)
        self.hot[layer][block_idx] = (k.to(self.device, non_blocking=True),
                                    v.to(self.device, non_blocking=True))

        # 冷却旧块
        if len(self.hot[layer]) > self.hot_window:
            cold_block_idx = next(iter(self.hot[layer]))
            self.hot[layer].pop(cold_block_idx)       # 释放 GPU

    def fetch(self, layer: int, needed_blocks: torch.Tensor
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        保证 needed_blocks 在 GPU; 返回 (k_cat, v_cat)
        needed_blocks: LongTensor[B,T] → 按次序去重转一维
        """
        if self.copy_stream is None and needed_blocks.device.type == "cuda":
            self.copy_stream = torch.cuda.Stream(
                device=needed_blocks.device
            )
        uniq = needed_blocks.unique().tolist()
        misses = [b for b in uniq if b not in self.hot[layer]]
        if misses:
            with torch.cuda.stream(self.copy_stream):
                for b in misses:
                    k = self.k_cpu[layer][b].to(self.device, non_blocking=True)
                    v = self.v_cpu[layer][b].to(self.device, non_blocking=True)
                    self.hot[layer][b] = (k, v)
        torch.cuda.current_stream().wait_stream(self.copy_stream)

        # 拼接 (∑blocks, H, D) → 在调用方再 reshape
        k_cat = torch.cat([self.hot[layer][b][0] for b in uniq], dim=1)  # cat on T

        v_cat = torch.cat([self.hot[layer][b][1] for b in uniq], dim=1)
        return k_cat, v_cat
