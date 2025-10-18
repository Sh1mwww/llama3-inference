from __future__ import annotations
import threading, time, collections
from queue import Empty, Queue, Full
from typing import List, Iterable, Tuple, Union
import numpy as np
import torch
from .SSDBacked import RawBlockKVBackend
from .config import KVCacheArgs
from .global_state_tracker import get_global_tracker, init_global_tracker, StorageType


BLOCK = 256  # tokens / block

class KVOffloader:
    # Class-level flag to print initialization info only once
    _init_printed = False
    """
    Off-GPU KV cache with attention-based Top-K fetch.

    Responsibilities:
    - push():  Save K/V block from GPU HBM → pinned CPU DRAM (and optionally mirror to SSD).
    - fetch(): Load a *set* of blocks from DRAM → GPU HBM (and SSD → DRAM if evicted).
    - update_importances(): Update per-block attention scores (written by SelfAttention).
    - topk_blocks(): Select top-k blocks by importance with multiple strategies.

    Notes:
    - Uses global tracker to record where (HBM / DRAM / SSD) blocks are stored and their importance.
    - DRAM capacity is enforced by `dram_limit_blk` and may trigger SSD spill.
    """
    def __init__(self, layers, heads, dim, max_seq, max_batch, device, dtype_bytes, streams=None):
        self.layers = layers
        self.heads = heads
        self.dim = dim
        self.max_batch = max_batch
        self.device = device
        self.dtype_bytes = dtype_bytes

        self.n_blocks = (max_seq + BLOCK - 1) // BLOCK

        # CPU pinned buffers per (layer, block): (max_batch, heads, BLOCK, dim)
        self.k_cpu = [[None for _ in range(self.n_blocks)] for _ in range(layers)]
        self.v_cpu = [[None for _ in range(self.n_blocks)] for _ in range(layers)]

        # importance / counts
        self.importance = np.zeros((max_batch, layers, self.n_blocks), dtype=np.float32)
        self.access_count = np.zeros((max_batch, layers, self.n_blocks), dtype=np.int32)
        self.last_access = np.zeros((max_batch, layers, self.n_blocks), dtype=np.int32)
        self.global_time = np.zeros(max_batch, dtype=np.int32)
        self.global_importance = np.zeros((layers, self.n_blocks), dtype=np.float32)
        self.global_access_count = np.zeros((layers, self.n_blocks), dtype=np.int32)
        self.global_time_counter = 0

        # Streams
        if streams is not None:
            self.h2d_stream = getattr(streams, "kv_h2d", None)
            self.d2h_stream = getattr(streams, "kv_d2h", None)
        else:
            self.h2d_stream = torch.cuda.Stream(device=device, priority=0) if device.startswith("cuda") else None
            self.d2h_stream = torch.cuda.Stream(device=device, priority=+1) if device.startswith("cuda") else None

        # bytes accounting (K+V) -- use a smaller sizing batch for DRAM quota estimation
        # 使用较小的 sizing batch 用于 DRAM 配额估算
        # 真实 decode 时 batch 常远小于 max_batch，用 max_batch 估算会过于悲观
        alloc_bsz = int(getattr(KVCacheArgs, "dram_sizing_batch", 8))
        alloc_bsz = max(1, min(max_batch, alloc_bsz))
        self.token_nbytes = (alloc_bsz * heads * dim) * dtype_bytes * 2
        self.block_nbytes = self.token_nbytes * BLOCK   # bytes per block (K+V)

        # 注意：数据结构仍使用 max_batch（self.k_cpu/v_cpu），这里只改配额计算

        # SSD backend
        try:
            ssd_device_path = getattr(KVCacheArgs, "ssd_device_path", "/dev/nvme0n1p4")
            self.ssd = RawBlockKVBackend(
                dev_path=ssd_device_path,
                n_layers=layers,
                blk_bytes=self.block_nbytes,
                blk_per_layer=self.n_blocks,
                max_concurrent_io=getattr(KVCacheArgs, "max_concurrent_io", 4),
            )
        except Exception as e:
            print(f"[WARNING] Failed to initialize SSD backend: {e}")
            print("[INFO] Falling back to DRAM-only mode")
            self.ssd = None

        self.on_ssd = [[False] * self.n_blocks for _ in range(layers)]

        # DRAM capacity (in blocks)
        # self.dram_limit_blk = int(KVCacheArgs.dram_limit_gb * (1024**3) // self.block_nbytes)
        # DRAM capacity in blocks (based on KVCacheArgs.dram_limit_gb), with a safety margin
        _dram_bytes = int(KVCacheArgs.dram_limit_gb * (1024**3))
        _safety = int(0.25 * _dram_bytes)  # 25% safety margin
        self.dram_limit_blk = max(0, (_dram_bytes - _safety) // self.block_nbytes)

        # 打印配额计算信息 - 只在第一次初始化时打印
        if not KVOffloader._init_printed:
            print(f"[KVOffloader] DRAM quota estimation:")
            print(f"  - Sizing batch: {alloc_bsz} (actual max_batch: {max_batch})")
            print(f"  - Block size: {self.block_nbytes / (1024**2):.2f} MB")
            print(f"  - DRAM limit: {self.dram_limit_blk} blocks ({self.dram_limit_blk * self.block_nbytes / (1024**3):.2f} GB)")

            if self.dram_limit_blk == 0:
                print("[KVOffloader][WARN] dram_limit_blk computed as 0; consider increasing KVCacheArgs.dram_limit_gb")

            KVOffloader._init_printed = True

        # --- SSD writer throttle (thread-safe flag) ---
        self._throttle_lock = threading.Lock()
        self._pause_write_until = 0.0  # monotonic seconds


        # global tracker
        self.global_tracker = get_global_tracker()
        if self.global_tracker is None:
            self.global_tracker = init_global_tracker(max_batch, layers, self.n_blocks)
        if self.global_tracker:
            self.global_tracker.storage_stats[StorageType.DRAM]["capacity_limit"] = self.dram_limit_blk

        # async writer (for SSD mirror)
        self._write_queue: "Queue[tuple[int,int,torch.Tensor]]" = Queue(
            maxsize=getattr(KVCacheArgs, "RAW_IO_QD_WRITE", 24)
        )
        self._writer_stop = threading.Event()   # ★ 统一命名：_writer_stop
        # self._pause_write_until: float = 0.0
        self._win_ms = getattr(KVCacheArgs, "IO_RAW_THROTTLE_MS", 30)
        self._write_target_bps = int(getattr(KVCacheArgs, "NVME_WRITE_TARGET_MBPS", 900) * 1024 * 1024)
        self._win_bytes = collections.deque()
        self._win_sum = 0

        if self.ssd is not None:
            t = threading.Thread(target=self._writer_loop, name="kv_writer", daemon=True)
            t.start()
            self._writer_thread = t
        else:
            self._writer_thread = None
            
        
    # 公开给 WSM 的写暂停 API（例如 PCIE 忙或 pinned 低水位时调用）
    
    # 提供供 WSM 调用的“暂停写”接口
    def throttle_writes_for(self, ms: int):
        self._pause_write_until = max(self._pause_write_until, time.time() + ms / 1000.0)
    
    def _writer_loop(self):
        batch: list[tuple[int,int,torch.Tensor]] = []
        last_flush = time.time()
        while not self._writer_stop.is_set():
            # 节流：写暂停期只拉队列不写（避免阻塞 push），但不超过 1 窗口累积
            now = time.time()
            if now < self._pause_write_until:
                time.sleep(0.001)
                continue
            try:
                item = self._write_queue.get(timeout=0.1)
                batch.append(item)
            except Empty:
                pass
            
            
            # 聚合：每 30ms 或积累到 ~1–2 MiB 的若干条后 flush
            flush_due = (time.time() - last_flush) * 1000.0 >= self._win_ms
            agg_bytes = sum(x[2].numel() * x[2].element_size() for x in batch)
            big_enough = agg_bytes >= (1 << 20)  # 1 MiB 起刷
            if not (flush_due or big_enough):
                continue
            
            # 限速：滑窗控制实际写速率 ≤ 目标
            self._drain_window()
            if self._win_sum >= self._write_target_bps * (self._win_ms/1000.0):
                # 轻睡，等窗口消化
                time.sleep(0.001)
                continue    
            
            # 真正执行写：串行写，避免过度竞争读
            for (L, B, kv_cpu) in batch:
                nbytes = kv_cpu.numel() * kv_cpu.element_size()
                try:
                    # 优先使用回你已有的 write_async；如不可用可退回 write()
                    if hasattr(self.ssd, "write_async"):
                        self.ssd.write_async(L, B, kv_cpu, sync=False)
                    else:
                        self.ssd.write(L, B, kv_cpu)
                    self._win_bytes.append((time.time(), nbytes))
                    self._win_sum += nbytes
                except Exception as e:
                    print(f"[WARN] SSD write failed @L{L} B{B}: {e}")
            batch.clear()
            last_flush = time.time()
    
    def _drain_window(self):
        """维护写限速滑窗（IO_RAW_THROTTLE_MS）"""
        cutoff = time.time() - self._win_ms/1000.0
        while self._win_bytes and self._win_bytes[0][0] < cutoff:
            _, nbytes = self._win_bytes.popleft()
            self._win_sum -= nbytes

    # ---------------- internal helpers -----------------
    def _alloc_block(self, layer: int, blk: int):
        """Pinned DRAM: (max_batch, heads, BLOCK, dim)"""
        if self.k_cpu[layer][blk] is None:
            shape = (self.max_batch, self.heads, BLOCK, self.dim)
            self.k_cpu[layer][blk] = torch.zeros(shape, dtype=torch.float16, pin_memory=True)
            self.v_cpu[layer][blk] = torch.zeros_like(self.k_cpu[layer][blk])

    # ---------------- public API -----------------
    def push(self, layer: int, blk: int, k: torch.Tensor, v: torch.Tensor,
             token_idx: int, batch_idx: int = 0, **kwargs):
        """
        Save K/V of *one* token from HBM → pinned DRAM.
        保存单个 token 的 K/V 从 HBM → 固定内存 DRAM。
        Optionally mirror the packed KV block to SSD (async) using block-aligned addressing.
        可选地将打包的 KV block 镜像到 SSD（异步），使用块对齐寻址。

        Args:
            layer: layer index / 层索引
            blk: block index (used for both DRAM and SSD addressing) / 块索引（用于 DRAM 和 SSD 寻址）
            k, v: tensors with shape (bsz, heads, dim) - NOTE: must keep 3D even if bsz=1
                  张量形状为 (bsz, heads, dim) - 注意：即使 bsz=1 也必须保持 3D
            token_idx: token index within the block / 块内的 token 索引
            batch_idx: batch index for importance update and tracker accounting / 批次索引用于重要性更新和跟踪器记账

        Note:
            Writes to specific token position (token_idx % BLOCK) in the block.
            写入块中的特定 token 位置 (token_idx % BLOCK)。
            SSD writes use block index (blk) for consistent addressing with _spill_to_ssd() and _load_from_ssd().
            SSD 写入使用块索引 (blk) 以与 _spill_to_ssd() 和 _load_from_ssd() 保持一致的寻址。
        """
        assert k.dim() == 3 and v.dim() == 3, "KV must be (bsz, heads, dim)"
        bsz = k.size(0)
        t_in_blk = int(token_idx) % BLOCK

        self._alloc_block(layer, blk)

        # Asynchronous GPU→CPU copy using dedicated D2H stream (if available)
        # 使用专用 D2H 流进行异步 GPU→CPU 拷贝（如果可用）
        stream = self.d2h_stream or torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            self.k_cpu[layer][blk][:bsz, :, t_in_blk, :].copy_(k, non_blocking=True)
            self.v_cpu[layer][blk][:bsz, :, t_in_blk, :].copy_(v, non_blocking=True)

        # Initialize a tiny positive importance to avoid accidental exclusion
        # 初始化一个很小的正重要性值以避免意外排除
        self.importance[min(batch_idx, self.max_batch-1), layer, blk] = max(
            self.importance[min(batch_idx, self.max_batch-1), layer, blk], 1e-6
        )
        self.global_importance[layer, blk] = max(self.global_importance[layer, blk], 1e-6)

        # Update global tracker: HBM → DRAM
        # 更新全局跟踪器：HBM → DRAM
        if self.global_tracker:
            self.global_tracker.update_hbm_storage(batch_idx, layer, [blk], "remove")
            self.global_tracker.update_dram_storage(batch_idx, layer, [blk], "add", [1e-6])

        self._maybe_evict()

        # Optional SSD mirror: pack whole block (max_batch, heads, BLOCK, 2*dim) and queue for async write
        # 可选 SSD 镜像：打包整个块 (max_batch, heads, BLOCK, 2*dim) 并排队异步写入
        if self.ssd is not None:
            # Wait for D2H transfer to complete before packing
            # 在打包前等待 D2H 传输完成
            if self.d2h_stream is not None:
                self.d2h_stream.synchronize()
            kv_pack_cpu = torch.cat(
                [self.k_cpu[layer][blk], self.v_cpu[layer][blk]], dim=-1
            ).contiguous()
            # 改用阻塞式 put，避免轻易丢块（保留超时告警）
            try:
                self._write_queue.put((layer, blk, kv_pack_cpu), timeout=1.0)
            except Full:
                print(f"[KV][WARN] write queue full for 1.0s, drop mirror @L{layer} B{blk}")
        
        
        

    def fetch(self, layer: int, blocks: torch.Tensor, batch_idx: int = 0, bsz: int | None = None):
        """
        Return concatenated K/V for the given **unique** block indices (ascending).
        返回给定唯一块索引（升序）的拼接 K/V。
        If blocks are on SSD, load them back to DRAM first; then transfer DRAM→HBM.
        如果块在 SSD 上，首先将它们加载回 DRAM；然后传输 DRAM→HBM。

        Returns:
            K, V of shape: (bsz, heads, sum(blocks)*BLOCK, dim) on GPU
            形状为 (bsz, heads, sum(blocks)*BLOCK, dim) 的 K, V 在 GPU 上
        """
        uniq = blocks.to(torch.long).unique(sorted=True).tolist()  # Ensure correct dtype / 确保正确的数据类型

        # SSD → DRAM for needed blocks / SSD → DRAM 加载需要的块
        need_load = [b for b in uniq if self.on_ssd[layer][b]]
        for b in need_load:
            self._load_from_ssd(layer, b)

        # DRAM → HBM copy (batched across blocks) on H2D stream
        # DRAM → HBM 拷贝（批量跨块）在 H2D 流上
        use_bsz = int(bsz) if bsz is not None else self.max_batch
        k_parts, v_parts = [], []
        stream = self.h2d_stream or torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            for b in uniq:
                if self.k_cpu[layer][b] is None:
                    raise RuntimeError(f"[KVOffloader] block {b} not pushed (layer {layer})")
                k_parts.append(self.k_cpu[layer][b][:use_bsz].to(self.device, non_blocking=True))
                v_parts.append(self.v_cpu[layer][b][:use_bsz].to(self.device, non_blocking=True))

        # Wait for transfer to complete on current stream
        # 等待当前流上的传输完成
        if stream is not torch.cuda.current_stream():
            torch.cuda.current_stream().wait_stream(stream)

        # Concatenate along token dimension (dim=2)
        # 在 token 维度（dim=2）上拼接
        k_full = torch.cat(k_parts, dim=2)
        v_full = torch.cat(v_parts, dim=2)
        return k_full, v_full

    # ------------- attention importance -------------
    def update_importances(
        self,
        layer: int,
        block_indices: List[int],
        block_scores: List[float],
        batch_idx: Union[int, List[int]] = None,
        momentum: float = 0.9,
    ):
        """
        EMA-based accumulation of attention scores (larger momentum = more historical bias).

        Args:
            layer: layer index
            block_indices: block indices to update
            block_scores: corresponding scores
            batch_idx: None = update all batches; int = single batch; List[int] = multiple batches
            momentum: EMA coefficient (0..1)
        """
        if batch_idx is None:
            batch_indices = list(range(self.max_batch))
        elif isinstance(batch_idx, int):
            batch_indices = [batch_idx]
        else:
            batch_indices = batch_idx

        # Per-batch updates (importance, access counts, last access time)
        for b_idx in batch_indices:
            if b_idx >= self.max_batch:
                continue

            for idx, score in zip(block_indices, block_scores):
                if idx < self.n_blocks:
                    self.importance[b_idx, layer, idx] = (
                        momentum * self.importance[b_idx, layer, idx] + (1.0 - momentum) * score
                    )
                    self.access_count[b_idx, layer, idx] += 1
                    self.last_access[b_idx, layer, idx] = self.global_time[b_idx]

            self.global_time[b_idx] += 1

        # Global updates (batch-agnostic)
        for idx, score in zip(block_indices, block_scores):
            if idx < self.n_blocks:
                self.global_importance[layer, idx] = (
                    momentum * self.global_importance[layer, idx] + (1.0 - momentum) * score
                )
                self.global_access_count[layer, idx] += 1

        self.global_time_counter += 1

    def topk_blocks(
        self,
        layer: int,
        k: int,
        batch_idx: Union[int, List[int]] = None,
        strategy: str = "batch_specific",
        access_weight: float = 0.1,
    ):
        """
        Return **ascending** indices of top-k blocks by importance.

        Strategies:
            - 'batch_specific': Use per-batch importance.
            - 'global': Use global importance across all batches.
            - 'hybrid': Blend batch and global (0.7/0.3) and weigh by accesses.
        """
        if batch_idx is None:
            imp = self.global_importance[layer]
            access = self.global_access_count[layer]
            ranked = sorted(
                range(self.n_blocks),
                key=lambda i: imp[i] * (1 + access_weight * access[i]),
                reverse=True,
            )
            chosen = [i for i in ranked if imp[i] > 0][:k]
            return sorted(chosen)

        elif isinstance(batch_idx, int):
            if strategy == "batch_specific":
                imp = self.importance[batch_idx, layer]
                access = self.access_count[batch_idx, layer]
                ranked = sorted(
                    range(self.n_blocks),
                    key=lambda i: imp[i] * (1 + access_weight * access[i]),
                    reverse=True,
                )
                chosen = [i for i in ranked if imp[i] > 0][:k]
                return sorted(chosen)

            elif strategy == "global":
                imp = self.global_importance[layer]
                access = self.global_access_count[layer]
                ranked = sorted(
                    range(self.n_blocks),
                    key=lambda i: imp[i] * (1 + access_weight * access[i]),
                    reverse=True,
                )
                chosen = [i for i in ranked if imp[i] > 0][:k]
                return sorted(chosen)

            elif strategy == "hybrid":
                batch_imp = self.importance[batch_idx, layer]
                global_imp = self.global_importance[layer]
                batch_access = self.access_count[batch_idx, layer]
                global_access = self.global_access_count[layer]

                def hybrid_score(i: int):
                    combined_imp = 0.7 * batch_imp[i] + 0.3 * global_imp[i]
                    combined_access = batch_access[i] + global_access[i]
                    return combined_imp * (1 + access_weight * combined_access)

                ranked = sorted(range(self.n_blocks), key=hybrid_score, reverse=True)
                chosen = [i for i in ranked if hybrid_score(i) > 0][:k]
                return sorted(chosen)

        else:
            # Multiple batches: return a dict {batch_idx: [block_ids]}
            result = {}
            for b_idx in batch_idx:
                if b_idx < self.max_batch:
                    result[b_idx] = self.topk_blocks(layer, k, b_idx, strategy, access_weight)
            return result

    def topk_blocks_aggregated(
        self,
        layer: int,
        k: int,
        batch_indices: List[int] = None,
        aggregation: str = "mean",
    ):
        """
        Aggregate importance across batches and then select top-k blocks.

        Args:
            layer: layer index
            k: number of blocks to select
            batch_indices: list of batch indices; None = all batches
            aggregation: 'mean' | 'max' | 'min' | 'sum'
        """
        if batch_indices is None:
            batch_indices = list(range(self.max_batch))

        if aggregation == "mean":
            agg_imp = np.mean(self.importance[batch_indices, layer], axis=0)
            agg_access = np.mean(self.access_count[batch_indices, layer], axis=0)
        elif aggregation == "max":
            agg_imp = np.max(self.importance[batch_indices, layer], axis=0)
            agg_access = np.max(self.access_count[batch_indices, layer], axis=0)
        elif aggregation == "min":
            agg_imp = np.min(self.importance[batch_indices, layer], axis=0)
            agg_access = np.min(self.access_count[batch_indices, layer], axis=0)
        elif aggregation == "sum":
            agg_imp = np.sum(self.importance[batch_indices, layer], axis=0)
            agg_access = np.sum(self.access_count[batch_indices, layer], axis=0)
        else:
            # Preserve original behavior (no explicit else in your code): fall back to mean-like path
            agg_imp = np.mean(self.importance[batch_indices, layer], axis=0)
            agg_access = np.mean(self.access_count[batch_indices, layer], axis=0)

        ranked = sorted(
            range(self.n_blocks),
            key=lambda i: agg_imp[i] * (1 + 0.1 * agg_access[i]),
            reverse=True,
        )
        chosen = [i for i in ranked if agg_imp[i] > 0][:k]
        return sorted(chosen)

    def get_batch_statistics(self, batch_idx: int):
        """
        Return summary statistics for a particular batch: totals and per-layer stats.
        """
        if batch_idx >= self.max_batch:
            return None

        stats = {
            "batch_idx": batch_idx,
            "batch_time": int(self.global_time[batch_idx]),
            "total_importance": float(np.sum(self.importance[batch_idx])),
            "total_accesses": int(np.sum(self.access_count[batch_idx])),
            "layer_stats": [],
        }

        for layer in range(self.layers):
            layer_stat = {
                "layer": layer,
                "active_blocks": int(np.sum(self.importance[batch_idx, layer] > 0)),
                "total_importance": float(np.sum(self.importance[batch_idx, layer])),
                "total_accesses": int(np.sum(self.access_count[batch_idx, layer])),
                "avg_importance": float(np.mean(self.importance[batch_idx, layer])),
                "max_importance": float(np.max(self.importance[batch_idx, layer])),
            }
            stats["layer_stats"].append(layer_stat)

        return stats

    def reset_batch(self, batch_idx: int):
        """
        Reset importance and counters for a particular batch.
        """
        if batch_idx < self.max_batch:
            self.importance[batch_idx] = 0
            self.access_count[batch_idx] = 0
            self.last_access[batch_idx] = 0
            self.global_time[batch_idx] = 0

    # ---------------- SSD spill / load -----------------
    def _spill_to_ssd(self, L: int, B: int):
        """
        Evict a DRAM block to SSD (if available). If no SSD, free DRAM storage.
        Also update global tracker transitions accordingly.
        """
        if self.ssd is None:
            # DRAM-only mode: free the memory and remove from DRAM in tracker
            if self.global_tracker:
                for batch_idx in range(self.max_batch):
                    self.global_tracker.update_dram_storage(batch_idx, L, [B], "remove")
            self.k_cpu[L][B] = self.v_cpu[L][B] = None
            return

        # DRAM → SSD in tracker per batch
        if self.global_tracker:
            for batch_idx in range(self.max_batch):
                if (batch_idx, L) in self.global_tracker.dram_storage and B in self.global_tracker.dram_storage[(batch_idx, L)]:
                    importance_score = (
                        float(self.importance[batch_idx, L, B]) if batch_idx < self.max_batch else 1e-6
                    )
                    self.global_tracker.update_dram_storage(batch_idx, L, [B], "remove")
                    self.global_tracker.update_ssd_storage(batch_idx, L, [B], "add", [importance_score])

        # Actual write: concat K and V on the last dim
        self.ssd.write(L, B, torch.cat([self.k_cpu[L][B], self.v_cpu[L][B]], dim=-1))
        self.k_cpu[L][B] = self.v_cpu[L][B] = None
        self.on_ssd[L][B] = True

    def _maybe_evict(self):
        """
        If DRAM usage (hot blocks) exceeds limit, evict the globally least-important block.
        """
        hot_cnt = sum(x is not None for lay in self.k_cpu for x in lay)
        if hot_cnt < self.dram_limit_blk:
            return

        # Candidate with minimal global importance
        cand = [
            (self.global_importance[L, B], L, B)
            for L in range(self.layers)
            for B in range(self.n_blocks)
            if self.k_cpu[L][B] is not None
        ]
        _, L, B = min(cand)
        self._spill_to_ssd(L, B)

    def _load_from_ssd(self, L: int, B: int):
        """
        Load a spilled block from SSD → DRAM.
        从 SSD 加载已溢出的块 → DRAM。
        Reuses a GPU buffer for read I/O and then copies back to pinned CPU.
        重用 GPU 缓冲区进行读取 I/O，然后复制回固定 CPU。
        """
        shape = (self.max_batch, self.heads, BLOCK, self.dim * 2)

        # Reuse GPU buffer to avoid churn / 重用 GPU 缓冲区以避免频繁分配
        if not hasattr(self, "_ssd_buffer") or self._ssd_buffer is None or tuple(self._ssd_buffer.shape) != shape:
            self._ssd_buffer = torch.empty(shape, dtype=torch.float16, device=self.device, non_blocking=True)

        if self.ssd is None:
            # DRAM-only mode: data is lost → allocate empty block
            # 仅 DRAM 模式：数据丢失 → 分配空块
            self._alloc_block(L, B)
            return

        # SSD read into GPU buffer → split → copy back to pinned CPU
        # SSD 读取到 GPU 缓冲区 → 分割 → 复制回固定 CPU
        self.ssd.read(L, B, self._ssd_buffer)
        k_gpu, v_gpu = self._ssd_buffer.split(self.dim, dim=-1)

        self._alloc_block(L, B)
        self.k_cpu[L][B].copy_(k_gpu.cpu())
        self.v_cpu[L][B].copy_(v_gpu.cpu())
        self.on_ssd[L][B] = False


    def throttle_writes_for(self, ms: int):
        """Externally pause SSD writes for 'ms' milliseconds (thread-safe)."""
        until = time.monotonic() + (ms / 1000.0)
        with self._throttle_lock:
            self._pause_write_until = max(self._pause_write_until, until)

    def _should_pause_writes(self) -> bool:
        with self._throttle_lock:
            return time.monotonic() < self._pause_write_until


    # ---------------- global tracker helpers -----------------
    def set_current_execution(self, batch_idx: int, layer_idx: int):
        """Set the currently executing (batch, layer) for global tracker (for visualization/monitoring)."""
        if self.global_tracker:
            self.global_tracker.set_current_execution(batch_idx, layer_idx)

    def get_global_state(self):
        """Return a snapshot of current global tracker state (or None if disabled)."""
        if self.global_tracker:
            return self.global_tracker.get_current_state()
        return None

    def print_global_state(self):
        """Print current global state and storage utilization from tracker."""
        if self.global_tracker:
            self.global_tracker.print_current_state()
            self.global_tracker.print_storage_utilization()
        else:
            print("Global tracker not available")

    def __del__(self):
        """
        Cleanup background prefetch resources (if present).
        NOTE: Prefetch members are only cleaned if they exist.
        """
        if hasattr(self, "prefetch_queue"):
            self.prefetch_queue.put((-1, -1))  # shutdown signal
        if hasattr(self, "prefetch_executor"):
            self.prefetch_executor.shutdown(wait=True)
        if hasattr(self, "_writer_stop"):
            self._writer_stop.set()
