from __future__ import annotations
from typing import List, Optional, Set, Dict, Tuple, Union
import torch
import os, mmap, numpy as np, math
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from .config import KVCacheArgs  
from .SSDBacked import RawBlockKVBackend
from .global_state_tracker import GlobalStateTracker, StorageType, get_global_tracker, init_global_tracker

BLOCK = 256  # tokens / block

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
        
        # 改为三维：[max_batch, layers, n_blocks]
        self.importance = np.zeros((max_batch, layers, n_blocks), dtype=np.float32)
        self.access_count = np.zeros((max_batch, layers, n_blocks), dtype=np.int32)
        self.last_access = np.zeros((max_batch, layers, n_blocks), dtype=np.int32)
        self.global_time = np.zeros(max_batch, dtype=np.int32)
        
        # 全局重要性（用于跨batch分析和驱逐策略）
        self.global_importance = np.zeros((layers, n_blocks), dtype=np.float32)
        self.global_access_count = np.zeros((layers, n_blocks), dtype=np.int32)
        self.global_time_counter = 0

        self.device = device
        self.dtype_bytes = dtype_bytes
        self.copy_stream = (torch.cuda.Stream(device=device)
                            if device.startswith("cuda") else None)
        self.heads = heads
        self.dim = dim
        self.max_batch = max_batch
        self.layers = layers
        self.n_blocks = n_blocks
        self.block_nbytes = (max_batch * heads * dim) * dtype_bytes * 2  # K+V
        
        # 尝试初始化SSD后端，如果失败则禁用
        try:
            self.ssd = RawBlockKVBackend(
                dev_path="/dev/nvme0n1p3",     # 你的裸分区
                n_layers=layers,
                blk_bytes=self.block_nbytes,
                blk_per_layer=n_blocks,
                max_concurrent_io=getattr(KVCacheArgs, 'max_concurrent_io', 4)
            )
            # print("[INFO] SSD backend initialized successfully")
        except (PermissionError, FileNotFoundError, OSError) as e:
            print(f"[WARNING] Failed to initialize SSD backend: {e}")
            print("[INFO] Falling back to DRAM-only mode")
            self.ssd = None
        self.on_ssd = [[False]*n_blocks for _ in range(layers)]
        self.dram_limit_blk = int(KVCacheArgs.dram_limit_gb * (1024**3) // self.block_nbytes)
        
        # 初始化或获取全局状态跟踪器
        self.global_tracker = get_global_tracker()
        if self.global_tracker is None:
            self.global_tracker = init_global_tracker(max_batch, layers, n_blocks)
        
        # 设置存储容量限制
        if self.global_tracker:
            self.global_tracker.storage_stats[StorageType.DRAM]['capacity_limit'] = self.dram_limit_blk
            if self.ssd:
                # 估算SSD容量（基于裸分区大小）
                try:
                    ssd_capacity_blk = int(getattr(KVCacheArgs, 'ssd_capacity_gb', 100) * (1024**3) // self.block_nbytes)
                    self.global_tracker.storage_stats[StorageType.SSD]['capacity_limit'] = ssd_capacity_blk
                except:
                    pass

    # ---------------- internal -----------------
    def _alloc_block(self, layer:int, blk:int, batch_sz:int):
        shape = (batch_sz, self.heads, self.dim)        # 而不是 max_batch
        self.k_cpu[layer][blk] = torch.empty(*shape, dtype=torch.float16,
                                            pin_memory=True)
        self.v_cpu[layer][blk] = torch.empty_like(self.k_cpu[layer][blk])

    # ---------------- public API --------------
    def push(self, layer:int, blk:int, k:torch.Tensor, v:torch.Tensor, batch_idx:int = 0):
        """Save K/V of *one* block from HBM → DRAM."""
        self._alloc_block(layer, blk, k.size(0))
        self.k_cpu[layer][blk][:k.size(0)].copy_(k, non_blocking=True)
        self.v_cpu[layer][blk][:v.size(0)].copy_(v, non_blocking=True)
        
        # 新 block 给一个极小初始权重，防止意外被忽略
        if batch_idx < self.max_batch:
            self.importance[batch_idx, layer, blk] = max(self.importance[batch_idx, layer, blk], 1e-6)
        # 同时更新全局重要性
        self.global_importance[layer, blk] = max(self.global_importance[layer, blk], 1e-6)
        
        # 更新全局跟踪器: HBM → DRAM
        if self.global_tracker:
            # 从HBM移除
            self.global_tracker.update_hbm_storage(batch_idx, layer, [blk], 'remove')
            # 添加到DRAM
            importance_score = float(self.importance[batch_idx, layer, blk]) if batch_idx < self.max_batch else 1e-6
            self.global_tracker.update_dram_storage(batch_idx, layer, [blk], 'add', [importance_score])
        
        self._maybe_evict()

    def fetch(self, layer:int, blocks:torch.Tensor, batch_idx:int = 0):
        """Return concat-K/V of given *unique* block indices (ascending)."""

        uniq = blocks.unique().tolist() 
        need_load = [b for b in uniq if self.on_ssd[layer][b]]
        for b in need_load:               # 先把 SSD → DRAM
            self._load_from_ssd(layer, b)
        
        # 更新全局跟踪器: DRAM → HBM
        if self.global_tracker:
            for blk in uniq:
                if not self.on_ssd[layer][blk]:  # 如果不在SSD上，说明在DRAM中
                    # 从DRAM移除
                    self.global_tracker.update_dram_storage(batch_idx, layer, [blk], 'remove')
                    # 添加到HBM
                    importance_score = float(self.importance[batch_idx, layer, blk]) if batch_idx < self.max_batch else 1e-6
                    self.global_tracker.update_hbm_storage(batch_idx, layer, [blk], 'add', [importance_score])
            
        k_parts, v_parts = [], []
        stream = self.copy_stream or torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            for blk in uniq:
                if self.k_cpu[layer][blk] is None:
                    raise RuntimeError(f"[KVOffloader] block {blk} not pushed (layer {layer})")
                k_parts.append(self.k_cpu[layer][blk].to(self.device, non_blocking=True))
                v_parts.append(self.v_cpu[layer][blk].to(self.device, non_blocking=True))
        # 安全的流等待，避免死锁
        if stream is not torch.cuda.current_stream():
            try:
                current_stream = torch.cuda.current_stream()
                current_stream.wait_stream(stream)
                
                # 验证流状态
                if hasattr(stream, 'query') and not stream.query():
                    # 流还在运行，等待一小段时间
                    import time
                    time.sleep(0.001)  # 1ms
                    
            except RuntimeError as e:
                logger.warning(f"Stream wait failed in KV offloader: {e}")
                # 降级到同步
                torch.cuda.synchronize()
        # 检查k_parts的维度以决定concatenation策略
        if k_parts and k_parts[0].dim() == 3:
            # 单batch情况: 沿seq_len维度concat
            return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)
        else:
            # 多batch情况: 沿seq_len维度concat (dim=1)
            return torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1)

    # ------------- attention importance -------------
    def update_importances(self,
                           layer:int,
                           block_indices:List[int],
                           block_scores:List[float],
                           batch_idx:Union[int, List[int]] = None,
                           momentum:float = 0.9):
        """EMA 累积注意力得分 (momentum 越大越偏向历史)。
        
        Args:
            layer: 层索引
            block_indices: block索引列表
            block_scores: block分数列表
            batch_idx: batch索引，可以是单个int或int列表，None表示所有batch
            momentum: 动量系数
        """
        if batch_idx is None:
            # 更新所有batch
            batch_indices = list(range(self.max_batch))
        elif isinstance(batch_idx, int):
            # 单个batch
            batch_indices = [batch_idx]
        else:
            # 多个batch
            batch_indices = batch_idx
        
        for b_idx in batch_indices:
            if b_idx >= self.max_batch:
                continue
                
            # 更新batch特定的重要性
            for idx, score in zip(block_indices, block_scores):
                if idx < self.n_blocks:
                    self.importance[b_idx, layer, idx] = (
                        momentum * self.importance[b_idx, layer, idx] + 
                        (1.0 - momentum) * score
                    )
                    self.access_count[b_idx, layer, idx] += 1
                    self.last_access[b_idx, layer, idx] = self.global_time[b_idx]
            
            self.global_time[b_idx] += 1
        
        # 同时更新全局重要性（用于驱逐策略）
        for idx, score in zip(block_indices, block_scores):
            if idx < self.n_blocks:
                self.global_importance[layer, idx] = (
                    momentum * self.global_importance[layer, idx] + 
                    (1.0 - momentum) * score
                )
                self.global_access_count[layer, idx] += 1
        self.global_time_counter += 1

    def topk_blocks(self, 
                    layer:int, 
                    k:int, 
                    batch_idx:Union[int, List[int]] = None,
                    strategy:str = 'batch_specific',
                    access_weight:float = 0.1):
        """Return **ascending** indices of Top-k blocks by importance.
        
        Args:
            layer: 层索引
            k: 选择的block数量
            batch_idx: batch索引
            strategy: 选择策略
                - 'batch_specific': 使用batch特定的重要性
                - 'global': 使用全局重要性
                - 'hybrid': 结合batch和全局重要性
            access_weight: 访问频率权重
        """
        if batch_idx is None:
            # 使用全局重要性
            imp = self.global_importance[layer]
            access = self.global_access_count[layer]
            ranked = sorted(range(self.n_blocks), 
                          key=lambda i: imp[i] * (1 + access_weight * access[i]), 
                          reverse=True)
            chosen = [i for i in ranked if imp[i] > 0][:k]
            return sorted(chosen)
        
        elif isinstance(batch_idx, int):
            # 单个batch
            if strategy == 'batch_specific':
                imp = self.importance[batch_idx, layer]
                access = self.access_count[batch_idx, layer]
                ranked = sorted(range(self.n_blocks), 
                              key=lambda i: imp[i] * (1 + access_weight * access[i]), 
                              reverse=True)
                chosen = [i for i in ranked if imp[i] > 0][:k]
                return sorted(chosen)
            
            elif strategy == 'global':
                imp = self.global_importance[layer]
                access = self.global_access_count[layer]
                ranked = sorted(range(self.n_blocks), 
                              key=lambda i: imp[i] * (1 + access_weight * access[i]), 
                              reverse=True)
                chosen = [i for i in ranked if imp[i] > 0][:k]
                return sorted(chosen)
            
            elif strategy == 'hybrid':
                batch_imp = self.importance[batch_idx, layer]
                global_imp = self.global_importance[layer]
                batch_access = self.access_count[batch_idx, layer]
                global_access = self.global_access_count[layer]
                
                def hybrid_score(i):
                    combined_imp = 0.7 * batch_imp[i] + 0.3 * global_imp[i]
                    combined_access = batch_access[i] + global_access[i]
                    return combined_imp * (1 + access_weight * combined_access)
                
                ranked = sorted(range(self.n_blocks), key=hybrid_score, reverse=True)
                chosen = [i for i in ranked if hybrid_score(i) > 0][:k]
                return sorted(chosen)
        
        else:
            # 多个batch，返回字典
            result = {}
            for b_idx in batch_idx:
                if b_idx < self.max_batch:
                    result[b_idx] = self.topk_blocks(layer, k, b_idx, strategy, access_weight)
            return result

    def topk_blocks_aggregated(self, 
                              layer:int, 
                              k:int, 
                              batch_indices:List[int] = None,
                              aggregation:str = 'mean'):
        """跨batch聚合后选择top-k blocks
        
        Args:
            layer: 层索引
            k: 选择的block数量
            batch_indices: batch索引列表，None表示所有batch
            aggregation: 聚合方式 ('mean', 'max', 'min', 'sum')
        """
        if batch_indices is None:
            batch_indices = list(range(self.max_batch))
        
        # 聚合重要性分数
        if aggregation == 'mean':
            agg_imp = np.mean(self.importance[batch_indices, layer], axis=0)
            agg_access = np.mean(self.access_count[batch_indices, layer], axis=0)
        elif aggregation == 'max':
            agg_imp = np.max(self.importance[batch_indices, layer], axis=0)
            agg_access = np.max(self.access_count[batch_indices, layer], axis=0)
        elif aggregation == 'min':
            agg_imp = np.min(self.importance[batch_indices, layer], axis=0)
            agg_access = np.min(self.access_count[batch_indices, layer], axis=0)
        elif aggregation == 'sum':
            agg_imp = np.sum(self.importance[batch_indices, layer], axis=0)
            agg_access = np.sum(self.access_count[batch_indices, layer], axis=0)
        
        ranked = sorted(range(self.n_blocks), 
                       key=lambda i: agg_imp[i] * (1 + 0.1 * agg_access[i]), 
                       reverse=True)
        chosen = [i for i in ranked if agg_imp[i] > 0][:k]
        return sorted(chosen)

    def get_batch_statistics(self, batch_idx:int):
        """获取特定batch的统计信息"""
        if batch_idx >= self.max_batch:
            return None
            
        stats = {
            'batch_idx': batch_idx,
            'batch_time': int(self.global_time[batch_idx]),
            'total_importance': float(np.sum(self.importance[batch_idx])),
            'total_accesses': int(np.sum(self.access_count[batch_idx])),
            'layer_stats': []
        }
        
        for layer in range(self.layers):
            layer_stat = {
                'layer': layer,
                'active_blocks': int(np.sum(self.importance[batch_idx, layer] > 0)),
                'total_importance': float(np.sum(self.importance[batch_idx, layer])),
                'total_accesses': int(np.sum(self.access_count[batch_idx, layer])),
                'avg_importance': float(np.mean(self.importance[batch_idx, layer])),
                'max_importance': float(np.max(self.importance[batch_idx, layer]))
            }
            stats['layer_stats'].append(layer_stat)
        
        return stats

    def reset_batch(self, batch_idx:int):
        """重置特定batch的数据"""
        if batch_idx < self.max_batch:
            self.importance[batch_idx] = 0
            self.access_count[batch_idx] = 0
            self.last_access[batch_idx] = 0
            self.global_time[batch_idx] = 0

    # ---------------- SSD spill / load --------------
    def _spill_to_ssd(self, L:int, B:int):
        if self.ssd is None:
            # DRAM-only mode: just free the memory
            # 更新全局跟踪器: 从DRAM移除（数据丢失）
            if self.global_tracker:
                # 查找所有使用此block的batch
                for batch_idx in range(self.max_batch):
                    self.global_tracker.update_dram_storage(batch_idx, L, [B], 'remove')
            self.k_cpu[L][B] = self.v_cpu[L][B] = None
            return
        
        # 更新全局跟踪器: DRAM → SSD
        if self.global_tracker:
            # 查找所有使用此block的batch并移动到SSD
            for batch_idx in range(self.max_batch):
                if (batch_idx, L) in self.global_tracker.dram_storage and B in self.global_tracker.dram_storage[(batch_idx, L)]:
                    importance_score = float(self.importance[batch_idx, L, B]) if batch_idx < self.max_batch else 1e-6
                    self.global_tracker.update_dram_storage(batch_idx, L, [B], 'remove')
                    self.global_tracker.update_ssd_storage(batch_idx, L, [B], 'add', [importance_score])
        
        self.ssd.write(L, B,
            torch.cat([self.k_cpu[L][B], self.v_cpu[L][B]], dim=-1))
        self.k_cpu[L][B] = self.v_cpu[L][B] = None
        self.on_ssd[L][B] = True

    def _maybe_evict(self):
        hot_cnt = sum(x is not None for lay in self.k_cpu for x in lay)
        if hot_cnt < self.dram_limit_blk:
            return
        # 使用全局重要性找到最小 importance 的块
        cand = [(self.global_importance[L, B], L, B)
                for L in range(self.layers)
                for B in range(self.n_blocks)
                if self.k_cpu[L][B] is not None]
        _, L, B = min(cand)
        self._spill_to_ssd(L, B)

    def _load_from_ssd(self, L:int, B:int):
        shape = (self.max_batch, self.heads, self.dim*2)
        
        # 复用GPU缓冲区避免内存泄漏
        if not hasattr(self, '_ssd_buffer') or self._ssd_buffer is None:
            self._ssd_buffer = torch.empty(shape, dtype=torch.float16,
                                         device=self.device, non_blocking=True)
        elif self._ssd_buffer.shape != shape:
            # 形状不匹配时重新分配
            del self._ssd_buffer
            torch.cuda.empty_cache()
            self._ssd_buffer = torch.empty(shape, dtype=torch.float16,
                                         device=self.device, non_blocking=True)
        
        if self.ssd is None:
            # DRAM-only mode: data is lost, allocate empty
            self._alloc_block(L, B, self.max_batch)
            return
        
        # 更新全局跟踪器: SSD → DRAM
        if self.global_tracker:
            # 查找所有使用此block的batch并移动到DRAM
            for batch_idx in range(self.max_batch):
                if (batch_idx, L) in self.global_tracker.ssd_storage and B in self.global_tracker.ssd_storage[(batch_idx, L)]:
                    importance_score = float(self.importance[batch_idx, L, B]) if batch_idx < self.max_batch else 1e-6
                    self.global_tracker.update_ssd_storage(batch_idx, L, [B], 'remove')
                    self.global_tracker.update_dram_storage(batch_idx, L, [B], 'add', [importance_score])
        
        # 复用缓冲区
        buf_gpu = self._ssd_buffer
        self.ssd.read(L, B, buf_gpu)
        k_gpu, v_gpu = buf_gpu.split(self.dim, dim=-1)
        self._alloc_block(L, B, self.max_batch)
        self.k_cpu[L][B].copy_(k_gpu.cpu())
        self.v_cpu[L][B].copy_(v_gpu.cpu())
        self.on_ssd[L][B] = False

    def set_current_execution(self, batch_idx: int, layer_idx: int):
        """设置当前正在执行的batch和layer"""
        if self.global_tracker:
            self.global_tracker.set_current_execution(batch_idx, layer_idx)
    
    def get_global_state(self):
        """获取全局状态"""
        if self.global_tracker:
            return self.global_tracker.get_current_state()
        return None
    
    def print_global_state(self):
        """打印全局状态"""
        if self.global_tracker:
            self.global_tracker.print_current_state()
            self.global_tracker.print_storage_utilization()
        else:
            print("Global tracker not available")

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'prefetch_queue'):
            self.prefetch_queue.put((-1, -1))  # shutdown signal
        if hasattr(self, 'prefetch_executor'):
            self.prefetch_executor.shutdown(wait=True)


