
from __future__ import annotations
from typing import List
import torch
import os, mmap, numpy as np, math
from .config import KVCacheArgs  

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
        self.block_nbytes = (max_batch * heads * dim) * dtype_bytes * 2  # K+V
        
        self.ssd = SSDBackedKV(
            path=KVCacheArgs.ssd_path,
            n_layers=layers, blk_bytes=self.block_nbytes,
            blk_per_layer=n_blocks)
        self.on_ssd = [[False]*n_blocks for _ in range(layers)]
        self.dram_limit_blk = int(KVCacheArgs.dram_limit_gb * (1024**3) // self.block_nbytes)
        self.block_nbytes = (max_batch * heads * dim) * dtype_bytes * 2  # K+V

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
        self._maybe_evict()

    def fetch(self, layer:int, blocks:torch.Tensor):
        """Return concat-K/V of given *unique* block indices (ascending)."""

        uniq = blocks.unique().tolist() 
        need_load = [b for b in uniq if self.on_ssd[layer][b]]
        for b in need_load:               # 先把 SSD → DRAM
            self._load_from_ssd(layer, b)
            
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
    # ---------------- SSD spill / load --------------
    def _spill_to_ssd(self, L:int, B:int):
        self.ssd.write(L, B,
            torch.cat([self.k_cpu[L][B], self.v_cpu[L][B]], dim=-1))
        self.k_cpu[L][B] = self.v_cpu[L][B] = None
        self.on_ssd[L][B] = True

    def _maybe_evict(self):
        hot_cnt = sum(x is not None for lay in self.k_cpu for x in lay)
        if hot_cnt < self.dram_limit_blk:
            return
        # 找到最小 importance 的块
        cand = [(self.importance[L][B], L, B)
                for L in range(self.layers)
                for B in range(self.n_blocks)
                if self.k_cpu[L][B] is not None]
        _, L, B = min(cand)
        self._spill_to_ssd(L, B)

    def _load_from_ssd(self, L:int, B:int):
        shape = (self.max_batch, self.heads, self.dim*2)
        buf_gpu = torch.empty(shape, dtype=torch.float16,
                              device=self.device, non_blocking=True)
        self.ssd.read(L, B, buf_gpu)
        k_gpu, v_gpu = buf_gpu.split(self.dim, dim=-1)
        self._alloc_block(L, B, self.max_batch)
        self.k_cpu[L][B].copy_(k_gpu.cpu())
        self.v_cpu[L][B].copy_(v_gpu.cpu())
        self.on_ssd[L][B] = False
class SSDBackedKV:
    """把固定大小的 KV-Block 顺序写入单一 .bin 文件，并 mmap 回读。"""

    def __init__(self, path:str, 
                 n_layers:int, 
                 blk_bytes:int, 
                 blk_per_layer:int):
        self.path       = os.path.abspath(path)
        self.blk_bytes  = blk_bytes
        self.blk_pl     = blk_per_layer         # blocks per layer
        total_bytes     = n_layers * blk_per_layer * blk_bytes
        
        if not os.path.exists(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            os.system(f"fallocate -l {total_bytes} {self.path}")
        self.fd   = os.open(self.path, os.O_RDWR | os.O_DIRECT)
        self.mm   = mmap.mmap(self.fd, total_bytes, mmap.MAP_SHARED,
                              mmap.PROT_READ | mmap.PROT_WRITE)

    def _offset(self, layer:int, blk:int) -> int:
        return (layer * self.blk_pl + blk) * self.blk_bytes

    # --------- 对外 API ---------
    def write(self, layer:int, blk:int, t_cpu:torch.Tensor):
        """DRAM → SSD:t_cpu 必须是 pin_memory=True 的 float16 Tensor."""
        off   = self._offset(layer, blk)
        view  = np.ndarray(t_cpu.numel(),
                           dtype=np.float16,
                           buffer=self.mm, offset=off).reshape(t_cpu.shape)
        np.copyto(view, t_cpu.cpu().numpy(), casting='no')

    def read(self, layer:int, blk:int, t_gpu:torch.Tensor):
        """SSD → DRAM → HBM。t_gpu 已在 GPU/DRAM 分配好相同形状。"""
        off   = self._offset(layer, blk)
        view  = np.ndarray(t_gpu.numel(),
                           dtype=np.float16,
                           buffer=self.mm, offset=off).reshape(t_gpu.shape)
        h_tmp = torch.frombuffer(view, dtype=torch.float16).clone()  # DRAM
        t_gpu.copy_(h_tmp, non_blocking=True)                        # →HBM