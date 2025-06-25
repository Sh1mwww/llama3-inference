from __future__ import annotations
from typing import List, Optional, Set, Dict, Tuple
import torch
import os, mmap, numpy as np, math
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from .config import KVCacheArgs  

BLOCK = 256  # tokens / block

class KVOffloader:
    """优化的Off-GPU KV cache，支持异步SSD I/O和智能预取"""

    def __init__(self,
                 layers: int,
                 heads: int,
                 dim: int,
                 max_seq: int,
                 max_batch: int,
                 device: str,
                 dtype_bytes: int):
        
        n_blocks = (max_seq + BLOCK - 1) // BLOCK
        self.k_cpu = [[None for _ in range(n_blocks)] for _ in range(layers)]
        self.v_cpu = [[None for _ in range(n_blocks)] for _ in range(layers)]
        self.importance = [[0.0 for _ in range(n_blocks)] for _ in range(layers)]
        
        # 访问统计，用于预取决策
        self.access_count = [[0 for _ in range(n_blocks)] for _ in range(layers)]
        self.last_access = [[0 for _ in range(n_blocks)] for _ in range(layers)]
        self.global_time = 0
        
        self.layers = layers
        self.device = device
        self.dtype_bytes = dtype_bytes
        self.heads = heads
        self.dim = dim
        self.max_batch = max_batch
        self.n_blocks = n_blocks
        self.block_nbytes = (max_batch * heads * dim) * dtype_bytes * 2  # K+V
        
        # CUDA streams for async operations
        if device.startswith("cuda"):
            self.copy_stream = torch.cuda.Stream(device=device)
            self.prefetch_stream = torch.cuda.Stream(device=device)
        else:
            self.copy_stream = None
            self.prefetch_stream = None
        
        # SSD backend with optimizations
        self.ssd = OptimizedSSDBackedKV(
            path=KVCacheArgs.ssd_path,
            n_layers=layers,
            blk_bytes=self.block_nbytes,
            blk_per_layer=n_blocks,
            max_concurrent_io=getattr(KVCacheArgs, 'max_concurrent_io', 4)
        )
        
        self.on_ssd = [[False]*n_blocks for _ in range(layers)]
        self.loading_from_ssd = [[False]*n_blocks for _ in range(layers)]  # 防止重复加载
        self.dram_limit_blk = int(KVCacheArgs.dram_limit_gb * (1024**3) // self.block_nbytes)
        
        # LRU cache for DRAM management
        self.dram_lru = {}  # (layer, block) -> timestamp
        
        # Prefetch management
        self.prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kv_prefetch")
        self.prefetch_queue = Queue(maxsize=16)
        self.prefetching = set()  # Track blocks being prefetched
        
        # Start prefetch worker
        self._start_prefetch_worker()

    def _start_prefetch_worker(self):
        """启动预取工作线程"""
        def prefetch_worker():
            while True:
                try:
                    layer, block = self.prefetch_queue.get(timeout=1.0)
                    if (layer, block) == (-1, -1):  # shutdown signal
                        break
                    self._prefetch_block(layer, block)
                    self.prefetching.discard((layer, block))
                except Empty:
                    continue
                except Exception as e:
                    print(f"Prefetch error: {e}")
        
        self.prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self.prefetch_thread.start()

    def _prefetch_block(self, layer: int, block: int):
        """预取单个block"""
        if not self.on_ssd[layer][block] or self.k_cpu[layer][block] is not None:
            return
        
        try:
            self._load_from_ssd_sync(layer, block)
        except Exception as e:
            print(f"Prefetch failed for layer {layer}, block {block}: {e}")

    def _should_prefetch(self, layer: int, block: int) -> bool:
        """基于访问模式决定是否预取"""
        if (layer, block) in self.prefetching:
            return False
        
        access_freq = self.access_count[layer][block]
        recency = self.global_time - self.last_access[layer][block]
        
        # 高频访问且最近访问过的block优先预取
        return access_freq > 2 and recency < 100

    def _trigger_prefetch(self, layer: int, blocks: List[int]):
        """触发相邻block的预取"""
        for block in blocks:
            # 预取前后相邻的blocks
            candidates = [block - 1, block + 1]
            for candidate in candidates:
                if (0 <= candidate < self.n_blocks and 
                    self.on_ssd[layer][candidate] and
                    self._should_prefetch(layer, candidate)):
                    
                    try:
                        self.prefetch_queue.put_nowait((layer, candidate))
                        self.prefetching.add((layer, candidate))
                    except:
                        pass  # Queue full, skip

    def _alloc_block(self, layer: int, blk: int, batch_sz: int):
        """分配block内存，使用内存池避免频繁分配"""
        if self.k_cpu[layer][blk] is None:
            shape = (batch_sz, self.heads, self.dim)
            # 使用pin_memory提高传输效率
            self.k_cpu[layer][blk] = torch.empty(*shape, dtype=torch.float16, pin_memory=True)
            self.v_cpu[layer][blk] = torch.empty_like(self.k_cpu[layer][blk])
            
            # 更新LRU
            self.dram_lru[(layer, blk)] = self.global_time

    def push(self, layer: int, blk: int, k: torch.Tensor, v: torch.Tensor):
        """保存K/V block到DRAM，支持批量操作"""
        self._alloc_block(layer, blk, k.size(0))
        
        # 异步复制到CPU
        if self.copy_stream:
            with torch.cuda.stream(self.copy_stream):
                self.k_cpu[layer][blk][:k.size(0)].copy_(k, non_blocking=True)
                self.v_cpu[layer][blk][:v.size(0)].copy_(v, non_blocking=True)
            # 同步等待完成
            torch.cuda.current_stream().wait_stream(self.copy_stream)
        else:
            self.k_cpu[layer][blk][:k.size(0)].copy_(k)
            self.v_cpu[layer][blk][:v.size(0)].copy_(v)
        
        # 更新重要性和访问统计
        self.importance[layer][blk] = max(self.importance[layer][blk], 1e-6)
        self.access_count[layer][blk] += 1
        self.last_access[layer][blk] = self.global_time
        self.global_time += 1
        
        # 检查是否需要驱逐
        self._maybe_evict()

    def fetch(self, layer: int, blocks: torch.Tensor):
        """批量获取blocks，支持异步加载和预取"""
        uniq = blocks.unique().tolist()
        
        # 更新访问统计
        for block in uniq:
            self.access_count[layer][block] += 1
            self.last_access[layer][block] = self.global_time
        self.global_time += 1
        
        # 分离需要从SSD加载的blocks
        need_load = [b for b in uniq if self.on_ssd[layer][b] and not self.loading_from_ssd[layer][b]]
        
        # 批量从SSD加载
        if need_load:
            self._batch_load_from_ssd(layer, need_load)
        
        # 触发预取
        self._trigger_prefetch(layer, uniq)
        
        # 收集K/V tensors
        k_parts, v_parts = [], []
        stream = self.copy_stream or torch.cuda.current_stream()
        
        with torch.cuda.stream(stream):
            for blk in uniq:
                if self.k_cpu[layer][blk] is None:
                    raise RuntimeError(f"Block {blk} not available (layer {layer})")
                
                k_gpu = self.k_cpu[layer][blk].to(self.device, non_blocking=True)
                v_gpu = self.v_cpu[layer][blk].to(self.device, non_blocking=True)
                k_parts.append(k_gpu)
                v_parts.append(v_gpu)
        
        if stream != torch.cuda.current_stream():
            torch.cuda.current_stream().wait_stream(stream)
        
        return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)

    def _batch_load_from_ssd(self, layer: int, blocks: List[int]):
        """批量从SSD加载blocks"""
        if not blocks:
            return
        
        # 标记正在加载，防止重复
        for block in blocks:
            self.loading_from_ssd[layer][block] = True
        
        try:
            # 使用SSD的批量读取
            self.ssd.batch_read(layer, blocks, self._on_batch_loaded, layer)
        except Exception as e:
            # 失败时清除标记
            for block in blocks:
                self.loading_from_ssd[layer][block] = False
            raise e

    def _on_batch_loaded(self, layer: int, loaded_data: Dict[int, torch.Tensor]):
        """批量加载完成的回调"""
        for block, data in loaded_data.items():
            try:
                k_gpu, v_gpu = data.split(self.dim, dim=-1)
                self._alloc_block(layer, block, self.max_batch)
                self.k_cpu[layer][block].copy_(k_gpu.cpu(), non_blocking=True)
                self.v_cpu[layer][block].copy_(v_gpu.cpu(), non_blocking=True)
                self.on_ssd[layer][block] = False
            finally:
                self.loading_from_ssd[layer][block] = False

    def _load_from_ssd_sync(self, layer: int, block: int):
        """同步从SSD加载单个block"""
        if self.loading_from_ssd[layer][block]:
            return  # 已在加载中
        
        self.loading_from_ssd[layer][block] = True
        try:
            shape = (self.max_batch, self.heads, self.dim * 2)
            buf_gpu = torch.empty(shape, dtype=torch.float16, device=self.device)
            self.ssd.read(layer, block, buf_gpu)
            
            k_gpu, v_gpu = buf_gpu.split(self.dim, dim=-1)
            self._alloc_block(layer, block, self.max_batch)
            self.k_cpu[layer][block].copy_(k_gpu.cpu())
            self.v_cpu[layer][block].copy_(v_gpu.cpu())
            self.on_ssd[layer][block] = False
        finally:
            self.loading_from_ssd[layer][block] = False

    def update_importances(self, layer: int, block_indices: List[int], 
                          block_scores: List[float], momentum: float = 0.9):
        """更新重要性分数"""
        imp = self.importance[layer]
        for idx, score in zip(block_indices, block_scores):
            imp[idx] = momentum * imp[idx] + (1.0 - momentum) * score
            # 更新访问统计
            self.access_count[layer][idx] += 1
            self.last_access[layer][idx] = self.global_time
        self.global_time += 1

    def topk_blocks(self, layer: int, k: int):
        """返回Top-k重要blocks"""
        imp = self.importance[layer]
        # 结合重要性和访问频率
        scores = [(imp[i] * (1 + 0.1 * self.access_count[layer][i]), i) 
                 for i in range(self.n_blocks) if imp[i] > 0]
        scores.sort(reverse=True)
        chosen = [i for _, i in scores[:k]]
        return sorted(chosen)

    def _maybe_evict(self):
        """基于LRU策略驱逐blocks到SSD"""
        hot_cnt = sum(x is not None for lay in self.k_cpu for x in lay)
        if hot_cnt <= self.dram_limit_blk:
            return
        
        # 找到最久未访问的block
        candidates = [(self.dram_lru.get((L, B), 0), L, B)
                     for L in range(self.layers)
                     for B in range(self.n_blocks)
                     if self.k_cpu[L][B] is not None]
        
        if not candidates:
            return
        
        # 驱逐最久未访问的block
        _, L, B = min(candidates)
        self._spill_to_ssd(L, B)

    def _spill_to_ssd(self, layer: int, block: int):
        """异步写入到SSD"""
        kv_data = torch.cat([self.k_cpu[layer][block], self.v_cpu[layer][block]], dim=-1)
        
        # 异步写入
        self.ssd.write_async(layer, block, kv_data)
        
        # 清理内存
        self.k_cpu[layer][block] = None
        self.v_cpu[layer][block] = None
        self.on_ssd[layer][block] = True
        
        # 从LRU中移除
        self.dram_lru.pop((layer, block), None)

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'prefetch_queue'):
            self.prefetch_queue.put((-1, -1))  # shutdown signal
        if hasattr(self, 'prefetch_executor'):
            self.prefetch_executor.shutdown(wait=True)


class OptimizedSSDBackedKV:
    """优化的SSD存储后端，支持批量I/O和异步操作"""

    def __init__(self, path: str, n_layers: int, blk_bytes: int, 
                 blk_per_layer: int, max_concurrent_io: int = 4):
        self.path = os.path.abspath(path)
        self.blk_bytes = blk_bytes
        self.blk_pl = blk_per_layer
        self.n_layers = n_layers
        total_bytes = n_layers * blk_per_layer * blk_bytes
        
        # 创建文件
        if not os.path.exists(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            # 使用fallocate预分配空间
            os.system(f"fallocate -l {total_bytes} {self.path}")
        
        # 使用O_DIRECT提高性能
        self.fd = os.open(self.path, os.O_RDWR | os.O_DIRECT)
        self.mm = mmap.mmap(self.fd, total_bytes, mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE)
        
        # I/O线程池
        self.io_executor = ThreadPoolExecutor(max_workers=max_concurrent_io, 
                                            thread_name_prefix="ssd_io")
        
        # 写入缓冲区管理
        self.write_buffer = {}
        self.write_lock = threading.Lock()

    def _offset(self, layer: int, blk: int) -> int:
        return (layer * self.blk_pl + blk) * self.blk_bytes

    def write_async(self, layer: int, blk: int, t_cpu: torch.Tensor):
        """异步写入"""
        def _write():
            off = self._offset(layer, blk)
            data = t_cpu.cpu().numpy()
            view = np.ndarray(data.size, dtype=np.float16,
                            buffer=self.mm, offset=off).reshape(data.shape)
            np.copyto(view, data, casting='no')
            # 强制刷新到磁盘
            self.mm.flush(off, self.blk_bytes)
        
        return self.io_executor.submit(_write)

    def batch_read(self, layer: int, blocks: List[int], 
                   callback, callback_arg):
        """批量异步读取"""
        def _batch_read():
            loaded_data = {}
            for block in blocks:
                try:
                    off = self._offset(layer, block)
                    shape = (self.blk_bytes // 2,)  # float16
                    view = np.ndarray(shape[0], dtype=np.float16,
                                    buffer=self.mm, offset=off)
                    # 复制数据避免内存映射问题
                    data_copy = torch.from_numpy(view.copy())
                    loaded_data[block] = data_copy.reshape(-1)  # 需要根据实际shape调整
                except Exception as e:
                    print(f"Failed to read block {block}: {e}")
            
            if loaded_data:
                callback(callback_arg, loaded_data)
        
        return self.io_executor.submit(_batch_read)

    def read(self, layer: int, blk: int, t_gpu: torch.Tensor):
        """同步读取"""
        off = self._offset(layer, blk)
        view = np.ndarray(t_gpu.numel(), dtype=np.float16,
                         buffer=self.mm, offset=off).reshape(t_gpu.shape)
        h_tmp = torch.from_numpy(view.copy()).clone()
        t_gpu.copy_(h_tmp, non_blocking=True)

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'io_executor'):
            self.io_executor.shutdown(wait=True)
        if hasattr(self, 'mm'):
            self.mm.close()
        if hasattr(self, 'fd'):
            os.close(self.fd)