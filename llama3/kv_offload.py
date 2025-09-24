from __future__ import annotations
import math
import mmap
import os
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
from .SSDBacked import RawBlockKVBackend
from .config import KVCacheArgs
from .global_state_tracker import (
    GlobalStateTracker,
    StorageType,
    get_global_tracker,
    init_global_tracker,
)


BLOCK = 256  # tokens / block

class KVOffloader:
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
    def __init__(
        self,
        layers: int,
        heads: int,
        dim: int,
        max_seq: int,
        max_batch: int,
        device: str,
        dtype_bytes: int,
        streams=None,
    ):
        n_blocks = (max_seq + BLOCK - 1) // BLOCK

        # Per-layer, per-block CPU storage (pinned). Each entry becomes a torch tensor after allocation.
        self.k_cpu = [[None for _ in range(n_blocks)] for _ in range(layers)]
        self.v_cpu = [[None for _ in range(n_blocks)] for _ in range(layers)]

        # Importance/Access tracking: 3D arrays with shape [max_batch, layers, n_blocks]
        self.importance = np.zeros((max_batch, layers, n_blocks), dtype=np.float32)
        self.access_count = np.zeros((max_batch, layers, n_blocks), dtype=np.int32)
        self.last_access = np.zeros((max_batch, layers, n_blocks), dtype=np.int32)
        self.global_time = np.zeros(max_batch, dtype=np.int32)

        # Global (batch-agnostic) importance & counters for eviction policy
        self.global_importance = np.zeros((layers, n_blocks), dtype=np.float32)
        self.global_access_count = np.zeros((layers, n_blocks), dtype=np.int32)
        self.global_time_counter = 0

        self.device = device
        self.dtype_bytes = dtype_bytes

        # Stream management: prefer external streams (e.g., from get_streams), fallback to internal streams
        if streams is not None:
            self.h2d_stream = getattr(streams, "kv_h2d", None)
            self.d2h_stream = getattr(streams, "kv_d2h", None)
            self.copy_stream = self.h2d_stream
        else:
            self.h2d_stream = torch.cuda.Stream(device=device, priority=0) if device.startswith("cuda") else None
            self.d2h_stream = torch.cuda.Stream(device=device, priority=+1) if device.startswith("cuda") else None
            self.copy_stream = self.h2d_stream

        self.heads = heads
        self.dim = dim
        self.max_batch = max_batch
        self.layers = layers
        self.n_blocks = n_blocks

        # Bytes per token (K + V). NOTE: original logic multiplies by max_batch.
        self.token_nbytes = (max_batch * heads * dim) * dtype_bytes * 2
        self.block_nbytes = (max_batch * heads * dim) * dtype_bytes * 2  # K+V

        # Initialize SSD backend if available; otherwise fallback to DRAM-only mode
        try:
            self.ssd = RawBlockKVBackend(
                dev_path="/dev/nvme0n1p4",
                n_layers=layers,
                blk_bytes=self.block_nbytes,
                blk_per_layer=n_blocks,
                max_concurrent_io=getattr(KVCacheArgs, "max_concurrent_io", 4),
            )
            # print("[INFO] SSD backend initialized successfully")
        except (PermissionError, FileNotFoundError, OSError) as e:
            print(f"[WARNING] Failed to initialize SSD backend: {e}")
            print("[INFO] Falling back to DRAM-only mode")
            self.ssd = None

        # on_ssd[L][B] indicates whether block (L,B) has been spilled to SSD
        self.on_ssd = [[False] * n_blocks for _ in range(layers)]

        # DRAM capacity in blocks (based on KVCacheArgs.dram_limit_gb)
        self.dram_limit_blk = int(KVCacheArgs.dram_limit_gb * (1024**3) // self.block_nbytes)

        # Initialize or get the global state tracker
        self.global_tracker = get_global_tracker()
        if self.global_tracker is None:
            self.global_tracker = init_global_tracker(max_batch, layers, n_blocks)

        # Configure storage capacity limits in tracker
        if self.global_tracker:
            self.global_tracker.storage_stats[StorageType.DRAM]["capacity_limit"] = self.dram_limit_blk
            if self.ssd:
                try:
                    ssd_capacity_blk = int(
                        getattr(KVCacheArgs, "ssd_capacity_gb", 100) * (1024**3) // self.block_nbytes
                    )
                    self.global_tracker.storage_stats[StorageType.SSD]["capacity_limit"] = ssd_capacity_blk
                except Exception:
                    pass

    # ---------------- internal helpers -----------------
    def _alloc_block(self, layer: int, blk: int, batch_sz: int):
        """
        Allocate pinned CPU buffers for a specific (layer, block) pair.
        Shape uses current batch size (not max_batch): (batch_sz, heads, dim).
        """
        shape = (batch_sz, self.heads, self.dim)
        self.k_cpu[layer][blk] = torch.empty(*shape, dtype=torch.float16, pin_memory=True)
        self.v_cpu[layer][blk] = torch.empty_like(self.k_cpu[layer][blk])

    # ---------------- public API -----------------
    def push(
        self,
        layer: int,
        blk: int,
        k: torch.Tensor,
        v: torch.Tensor,
        token_idx: int = None,
        batch_idx: int = 0,
        **kwargs,
    ):
        """
        Save K/V of *one* block from HBM → pinned DRAM.
        Optionally mirror the packed KV to SSD (async) when token_idx is provided.

        Args:
            layer: layer index
            blk: block index
            k, v: tensors with shape (bsz, heads, dim) or (heads, dim)
            token_idx: optional token-aligned index for SSD layout
            batch_idx: batch index for importance update and tracker accounting
        """
        self._alloc_block(layer, blk, k.size(0))

        # Asynchronous GPU→CPU copy using dedicated D2H stream (if available)
        stream = self.d2h_stream or torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            self.k_cpu[layer][blk][: k.size(0)].copy_(k, non_blocking=True)
            self.v_cpu[layer][blk][: v.size(0)].copy_(v, non_blocking=True)

        # Initialize a tiny positive importance to avoid accidental exclusion
        if batch_idx < self.max_batch:
            self.importance[batch_idx, layer, blk] = max(self.importance[batch_idx, layer, blk], 1e-6)
        self.global_importance[layer, blk] = max(self.global_importance[layer, blk], 1e-6)

        # Update global tracker: HBM → DRAM
        if self.global_tracker:
            self.global_tracker.update_hbm_storage(batch_idx, layer, [blk], "remove")
            importance_score = (
                float(self.importance[batch_idx, layer, blk]) if batch_idx < self.max_batch else 1e-6
            )
            self.global_tracker.update_dram_storage(batch_idx, layer, [blk], "add", [importance_score])

        self._maybe_evict()

        # Optional SSD mirror: pack to (max_batch, heads, 2*dim) fp16 and write async
        if token_idx is not None and self.ssd is not None:
            if k.dim() == 2:  # (heads, dim) → (1, heads, dim)
                k = k.unsqueeze(0)
                v = v.unsqueeze(0)
            bsz = k.size(0)

            kv_pack = torch.zeros(
                (self.max_batch, self.heads, self.dim * 2),
                dtype=torch.float16,
                device=k.device,
            )
            kv_pack[:bsz, :, : self.dim].copy_(k)
            kv_pack[:bsz, :, self.dim :].copy_(v)

            try:
                self.ssd.write_async(layer, token_idx, kv_pack)
            except Exception as e:
                print(f"[WARN] SSD mirror write failed @L{layer} T{token_idx}: {e}")

    def fetch(self, layer: int, blocks: torch.Tensor, batch_idx: int = 0):
        """
        Return concatenated K/V for the given **unique** block indices (ascending).
        If blocks are on SSD, load them back to DRAM first; then transfer DRAM→HBM.
        """
        uniq = blocks.unique().tolist()

        # SSD → DRAM for needed blocks
        need_load = [b for b in uniq if self.on_ssd[layer][b]]
        for b in need_load:
            self._load_from_ssd(layer, b)

        # Tracker update: DRAM → HBM for the fetched blocks (only if not on SSD)
        if self.global_tracker:
            for blk in uniq:
                if not self.on_ssd[layer][blk]:
                    self.global_tracker.update_dram_storage(batch_idx, layer, [blk], "remove")
                    importance_score = (
                        float(self.importance[batch_idx, layer, blk]) if batch_idx < self.max_batch else 1e-6
                    )
                    self.global_tracker.update_hbm_storage(batch_idx, layer, [blk], "add", [importance_score])

        # DRAM → HBM copy (batched across blocks) on H2D stream
        k_parts, v_parts = [], []
        stream = self.h2d_stream or torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            for blk in uniq:
                if self.k_cpu[layer][blk] is None:
                    raise RuntimeError(f"[KVOffloader] block {blk} not pushed (layer {layer})")
                k_parts.append(self.k_cpu[layer][blk].to(self.device, non_blocking=True))
                v_parts.append(self.v_cpu[layer][blk].to(self.device, non_blocking=True))

        # Defensive stream wait to avoid deadlock; fall back to synchronize on error
        if stream is not torch.cuda.current_stream():
            try:
                current_stream = torch.cuda.current_stream()
                current_stream.wait_stream(stream)

                # If stream has 'query' and is still running, sleep briefly (1ms)
                if hasattr(stream, "query") and not stream.query():
                    import time as _time  # alias to avoid masking stdlib 'time' elsewhere
                    _time.sleep(0.001)
            except RuntimeError as e:
                # NOTE: 'logger' is not defined in original code; kept as-is per your request.
                # logger.warning(f"Stream wait failed in KV offloader: {e}")
                torch.cuda.synchronize()

        # Concatenation policy preserved from original:
        # - If tensors are 3D (bsz, heads, dim), concat along dim=0 (sequence-like axis per block)
        # - Else (multi-batch layout), concat along dim=1
        if k_parts and k_parts[0].dim() == 3:
            return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)
        else:
            return torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1)

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
        Reuses a GPU buffer for read I/O and then copies back to pinned CPU.
        """
        shape = (self.max_batch, self.heads, self.dim * 2)

        # Reuse GPU buffer to avoid churn
        if not hasattr(self, "_ssd_buffer") or self._ssd_buffer is None:
            self._ssd_buffer = torch.empty(shape, dtype=torch.float16, device=self.device, non_blocking=True)
        elif self._ssd_buffer.shape != shape:
            del self._ssd_buffer
            torch.cuda.empty_cache()
            self._ssd_buffer = torch.empty(shape, dtype=torch.float16, device=self.device, non_blocking=True)

        if self.ssd is None:
            # DRAM-only mode: data is lost → allocate empty block
            self._alloc_block(L, B, self.max_batch)
            return

        # Tracker: SSD → DRAM for all batches that reference this block
        if self.global_tracker:
            for batch_idx in range(self.max_batch):
                if (batch_idx, L) in self.global_tracker.ssd_storage and B in self.global_tracker.ssd_storage[(batch_idx, L)]:
                    importance_score = (
                        float(self.importance[batch_idx, L, B]) if batch_idx < self.max_batch else 1e-6
                    )
                    self.global_tracker.update_ssd_storage(batch_idx, L, [B], "remove")
                    self.global_tracker.update_dram_storage(batch_idx, L, [B], "add", [importance_score])

        # SSD read into GPU buffer → split → copy back to pinned CPU
        buf_gpu = self._ssd_buffer
        self.ssd.read(L, B, buf_gpu)
        k_gpu, v_gpu = buf_gpu.split(self.dim, dim=-1)

        self._alloc_block(L, B, self.max_batch)
        self.k_cpu[L][B].copy_(k_gpu.cpu())
        self.v_cpu[L][B].copy_(v_gpu.cpu())
        self.on_ssd[L][B] = False

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
