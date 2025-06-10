"""
KVOffloader (global LRU, hot ≤ 32 blocks)
----------------------------------------
 • push(layer, block, k, v)   : 写入新 KV
 • fetch(layer, blocks_tensor): 确保所需块在 GPU, 返回拼接后的 (k,v)
"""
from __future__ import annotations
from collections import OrderedDict
from typing import OrderedDict as OD, Tuple
from .config import ModelArgs
import torch

BLOCK = 64       # token / block        
HOT_LIMIT = 32    # global hot-window ≤ 32 blocks

class KVOffloader:
    def __init__(self, 
                 layers: int, 
                 heads: int, 
                 dim: int,
                 max_seq: int, 
                 max_batch: int, 
                 device: str,
                 dtype_bytes: int):
        n_blocks = (max_seq + BLOCK - 1)   #num of blocks
        shape = (layers, n_blocks, max_batch, heads, dim)
        # self.k_cpu = torch.empty(*shape, dtype=torch.float16, pin_memory=True)
        # self.v_cpu = torch.empty_like(self.k_cpu)
        self.k_cpu = [[None for _ in range(n_blocks)] for _ in range(layers)]
        self.v_cpu = [[None for _ in range(n_blocks)] for _ in range(layers)]
        self.device = device
        self.dtype_bytes = dtype_bytes
        self.copy_stream = (torch.cuda.Stream(device=device)
                            if device.startswith("cuda") else None)
        
        # Global LRU: key=(layer, block)
        self.hot: "OD[Tuple[int,int], Tuple[torch.Tensor,torch.Tensor]]" = OrderedDict()
        self.heads = heads
        self.dim = dim
        self.max_batch = max_batch
        
    def _alloc_block(self, layer, blk, batch_sz):
        if self.k_cpu[layer][blk] is None:
            shape = (batch_sz, self.heads, self.dim)
            self.k_cpu[layer][blk] = torch.empty(*shape, dtype=torch.float16,
                                                pin_memory=True)
            self.v_cpu[layer][blk] = torch.empty_like(self.k_cpu[layer][blk])
    def push(self,
             layer:int,
             blk:int,
             k: torch.Tensor,
             v: torch.Tensor,):
        
        # saving from HBM2DRAM
        self._alloc_block(layer, blk, k.size(0))
        self.k_cpu[layer][blk][:k.size(0)].copy_(k, non_blocking=True)  
        self.v_cpu[layer][blk][:v.size(0)].copy_(v, non_blocking=True)

        # keeping hot block in HBM
        k_gpu, v_gpu = k.to(self.device, non_blocking=True), v.to(self.device, non_blocking=True)
        key = (layer, blk)
        self.hot[key] = (k_gpu, v_gpu)
        self.hot.move_to_end(key)
        
        if len(self.hot) > HOT_LIMIT:
            old_key, (old_k, old_v) = self.hot.popitem(last=False)
            
            
            
    def fetch(self,
              layer:int,
              blocks:torch.Tensor):
        """
        Identify the KV blocks required for this round of reasoning but not in HBM
        """
        uniq = blocks.unique().tolist()
        misses = [b for b in uniq if (layer, b) not in self.hot]
        
        # Use asynchronous stream to copy from DRAM back to HBM (avoid blocking main stream)
        if misses:
            if self.copy_stream:
                with torch.cuda.stream(self.copy_stream):
                    for b in misses:
                        k = self.k_cpu[layer][b][:1].to(self.device, non_blocking=True)
                        v = self.v_cpu[layer][b][:1].to(self.device, non_blocking=True)
                        self.hot[(layer, b)] = (k, v)
            torch.cuda.current_stream().wait_stream(self.copy_stream)     
              
        # mark all accessed blocks as recently used
        for b in uniq:
            self.hot.move_to_end((layer, b))
            
            
        k_list, v_list = [], []
        for b in uniq:
            k_list.append(self.hot[(layer, b)][0])
            v_list.append(self.hot[(layer, b)][1])
        k_cat = torch.cat(k_list, dim=1)   # (B, T*, H, D)
        v_cat = torch.cat(v_list, dim=1)
        return k_cat, v_cat