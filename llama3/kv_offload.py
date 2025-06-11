
from __future__ import annotations
from collections import OrderedDict
from typing import OrderedDict as OD, Tuple
import torch

BLOCK = 64       # token / block        


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
    
            
    def fetch(self,
              layer:int,
              blocks:torch.Tensor):
        """
        Identify the KV blocks required for this round of reasoning but not in HBM
        """
        uniq = blocks.unique().tolist()
        k_chunks, v_chunks = [], []
        # k_chunks = torch.cat([self.k_cpu[layer][b] for b in uniq], 0)
        # v_chunks = torch.cat([self.v_cpu[layer][b] for b in uniq], 0)
        stream = self.copy_stream or torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            for blk in uniq:
                k_gpu = self.k_cpu[layer][blk].to(self.device, non_blocking=True)
                v_gpu = self.v_cpu[layer][blk].to(self.device, non_blocking=True)
                k_chunks.append(k_gpu)
                v_chunks.append(v_gpu)
        if stream is not torch.cuda.current_stream():
            torch.cuda.current_stream().wait_stream(stream)
            
        k_cat = torch.cat(k_chunks, dim=0)   # (B, T*, H, D)
        v_cat = torch.cat(v_chunks, dim=0)
        return k_cat, v_cat