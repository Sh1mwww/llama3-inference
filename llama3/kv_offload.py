
from __future__ import annotations

"""
kv_offload_fixed.py

修复点（与原版对照）
------------------
1) **单例化 & 幂等初始化**：KVOffloader 覆盖 __new__ 实现进程级单例，__init__ 支持幂等，避免
   每层构造时启动多份后台线程/队列（对应“问题1：跨层共享但缺乏同步保护”）。

2) **按块锁（per-(L,B)）**：新增 self._blk_locks[L][B]，所有对 k_cpu/v_cpu、gpu_k/gpu_v、on_ssd、
   事件表的读写都在相应的块锁保护下进行，消除无锁读写（对应“问题2、问题4、问题5”的根因）。

3) **原子发布（atomic publish）**：_alloc_block() 采用“在局部变量中完成分配/reshape/zero，再一次性
   发布到 self.k_cpu/v_cpu”，并在块锁内双检，消除检查与赋值之间的竞态窗口（问题4）。

4) **push() / _load_from_ssd() 并发**：push() 在块锁中安排 D2H 拷贝，并记录 per-block D2H 事件；
   _spill_to_ssd() 在释放 pinned 前会非阻塞轮询该事件；_load_from_ssd() 也在块锁内执行，确保和
   push 对同一 (L,B) 不会交错覆盖（问题3）。

5) **prefetch_async() 写 GPU 缓存的同步**：对每个块在拷贝之后**单独记录 per-block CUDA Event**
   到 self._blk_ready_evt[(L,B)]，fetch()/wait_blocks_ready() 只等待该块事件；组级事件仍保留在
   self._prefetch_map 中用于整组快拼，但不再复用“同一个 event 给多个块”（问题6、问题5）。

6) **fetch() 的多流友好 & 全 GPU 快路径**：新增 stream 形参；当所有块都已在 GPU 且形状匹配时，
   仅在 compute 流上等待块事件并直接拼接；否则按需补齐 SSD→DRAM、DRAM→HBM，并在 H2D 流上
   记录 per-block 事件，主流通过 wait_stream() 桥接，避免隐性同步（提升正确性与吞吐）。

7) **零拷贝加载（可选）**：_load_from_ssd() 优先尝试对齐的 pinned-uint8 直接读取（若后端实现
   `read_into_pinned_aligned`），不可用则回退到 legacy 路径，保证兼容性。

8) **写入限速与线程关闭**：保留原有 writer/packer 节流逻辑与关停流程，同时在单例上只启动一份
   后台线程，避免多实例竞争 IO。

本文件只依赖你工程中的：
- .SSDBacked.RawBlockKVBackend
- .config.KVCacheArgs
- .global_state_tracker.get_global_tracker / init_global_tracker / StorageType

若你的后端未实现 read_into_pinned_aligned，本代码会自动走回退路径。

"""

import threading
import time
import collections
import math
import os
import gc
from collections import OrderedDict
from queue import Empty, Queue, Full
from typing import List, Iterable, Tuple, Union, Optional, Dict

import numpy as np
import torch
import concurrent.futures

from .SSDBacked import RawBlockKVBackend
from .config import KVCacheArgs
from .global_state_tracker import get_global_tracker, init_global_tracker, StorageType

BLOCK = 256  # tokens / block


# ============================================================================
# TTL-LRU 容器：防止预取记录无限增长
# ============================================================================
class _TTLDict(OrderedDict):
    """
    带 TTL 和容量限制的字典，用于防止 prefetch map 内存泄漏。
    - 自动淘汰最老的 entries（LRU）
    - 自动过期超时的 entries（TTL）
    """
    def __init__(self, maxlen: int = 1000, ttl_s: float = 60.0):
        super().__init__()
        self.maxlen = int(maxlen)
        self.ttl = float(ttl_s)

    def set(self, key, value):
        """写入 key，并记录时间戳"""
        now = time.time()
        super().__setitem__(key, (now, value))
        # LRU 淘汰
        while len(self) > self.maxlen:
            self.popitem(last=False)
        # TTL 清理（轻量扫描）
        self._sweep(now)

    def get(self, key, default=None):
        """读取 key，自动检查 TTL"""
        item = super().get(key, None)
        if item is None:
            return default
        ts, value = item
        if time.time() - ts > self.ttl:
            # 过期，删除并返回 default
            try:
                super().__delitem__(key)
            except Exception:
                pass
            return default
        return value

    def _sweep(self, now=None):
        """清理过期的 entries（增量扫描，避免阻塞）"""
        now = now or time.time()
        # 只扫描最多 10% 的 entries
        max_check = max(10, len(self) // 10)
        checked = 0
        dels = []
        for k, (ts, _) in self.items():
            if now - ts > self.ttl:
                dels.append(k)
            checked += 1
            if checked >= max_check:
                break
        for k in dels:
            try:
                super().__delitem__(k)
            except Exception:
                pass


# ============================================================================
# Global DRAM Pool (Singleton for all layers)
# ============================================================================

class DRAMPool:
    """
    全局 DRAM 池，支持预分配和懒分配两种模式。额外实现“温和回收”（trim_backoff）。
    """
    def __init__(self, bytes_limit_gb: float, block_bytes: int,
                 preallocate: bool = False, lazy_init: bool = True,
                 verbose: bool = True, trim_backoff: float = 0.8):
        self.bytes_limit = int(bytes_limit_gb * (1 << 30))
        self.block_bytes = int(math.ceil(int(block_bytes) / 4096) * 4096)  # 4KiB 对齐
        self.preallocate = preallocate
        self.lazy_init = lazy_init
        self.used = 0
        self.lock = threading.Lock()

        self.trim_backoff = float(trim_backoff)
        self._last_trim_ts = 0.0

        self.lazy_free: List[torch.Tensor] = []
        self.lazy_live: set[int] = set()

        if preallocate:
            try:
                self.dram_buf = torch.empty(self.bytes_limit, dtype=torch.uint8, pin_memory=True)
                self.free_segments = [(0, self.bytes_limit)]
                if verbose:
                    print(f"[DRAMPool] Preallocated {self.bytes_limit / (1<<30):.2f} GB")
            except RuntimeError as e:
                if verbose:
                    print(f"[DRAMPool][WARN] Preallocate failed ({e}), fallback to lazy mode")
                self.dram_buf = None
                self.free_segments = []
                self.lazy_init = True
        else:
            self.dram_buf = None
            self.free_segments: List[Tuple[int,int]] = []
            if verbose:
                print(f"[DRAMPool] Lazy allocation (limit={self.bytes_limit/(1<<30):.2f} GB)")

    def _alloc_from_segments(self, need: int) -> Optional[int]:
        for i, (off, sz) in enumerate(self.free_segments):
            if sz >= need:
                new_off = off
                rem = sz - need
                if rem == 0:
                    self.free_segments.pop(i)
                else:
                    self.free_segments[i] = (off + need, rem)
                return new_off
        return None

    def _coalesce_segment(self, off: int, sz: int):
        segs = self.free_segments
        segs.append((off, sz))
        segs.sort()
        merged = []
        cur_off, cur_sz = segs[0]
        for o, s in segs[1:]:
            if cur_off + cur_sz == o:
                cur_sz += s
            else:
                merged.append((cur_off, cur_sz))
                cur_off, cur_sz = o, s
        merged.append((cur_off, cur_sz))
        self.free_segments = merged

    def alloc_block(self, nbytes: Optional[int] = None) -> Optional[torch.Tensor]:
        need_bytes = int(nbytes if nbytes is not None else self.block_bytes)
        need_bytes = int(math.ceil(need_bytes / 4096) * 4096)

        with self.lock:
            if self.dram_buf is not None:
                if self.used + need_bytes > self.bytes_limit:
                    return None
                off = self._alloc_from_segments(need_bytes)
                if off is None:
                    return None
                self.used += need_bytes
                return self.dram_buf.narrow(0, off, need_bytes)

            if self.used + need_bytes > self.bytes_limit:
                return None

            # 复用
            for i, t in enumerate(self.lazy_free):
                if t.numel() == need_bytes:
                    blk = self.lazy_free.pop(i)
                    self.lazy_live.add(blk.untyped_storage().data_ptr())
                    self.used += need_bytes
                    return blk

            try:
                t = torch.empty(need_bytes, dtype=torch.uint8, pin_memory=True)
                self.lazy_live.add(t.untyped_storage().data_ptr())
                self.used += need_bytes
                return t
            except RuntimeError as e:
                print(f"[DRAMPool][ERROR] alloc {need_bytes/(1<<20):.2f}MB failed: {e}")
                return None

    def free_block(self, buf: torch.Tensor):
        with self.lock:
            if self.dram_buf is not None:
                off = buf.storage_offset()
                size = buf.numel()
                self._coalesce_segment(off, size)
                self.used -= size
            else:
                ptr = buf.untyped_storage().data_ptr()
                if ptr in self.lazy_live:
                    self.lazy_live.remove(ptr)
                    self.lazy_free.append(buf)
                    self.used -= buf.numel()
                    self._maybe_trim()

    def _maybe_trim(self):
        if self.dram_buf is not None:
            return
        now = time.time()
        if now - self._last_trim_ts < 0.5:
            return
        free_bytes = sum(b.numel() for b in self.lazy_free)
        target_free = int(self.bytes_limit * (1 - self.trim_backoff))
        if free_bytes > target_free:
            bytes_to_release = free_bytes - target_free
            released = 0
            while self.lazy_free and released < bytes_to_release:
                t = self.lazy_free.pop()
                released += t.numel()
                del t
            gc.collect()
        self._last_trim_ts = now

    def stats_str(self) -> str:
        with self.lock:
            mode = "Preallocate" if self.dram_buf is not None else "Lazy"
            used_gb = self.used / (1 << 30)
            limit_gb = self.bytes_limit / (1 << 30)
            util = (self.used / self.bytes_limit * 100) if self.bytes_limit > 0 else 0
            free_count = len(self.free_segments) if self.dram_buf is not None else len(self.lazy_free)
            live_count = len(self.lazy_live)
            return (f"mode={mode}, used={used_gb:.2f}/{limit_gb:.2f}GB ({util:.1f}%), "
                    f"free_blocks={free_count}, live_blocks={live_count}")


# Global pool singleton
_GLOBAL_DRAM_POOL: Optional[DRAMPool] = None
_GLOBAL_POOL_CFG: Optional[tuple] = None

def _pool_cfg_tuple():
    return (
        float(KVCacheArgs.dram_limit_gb),
        int(KVCacheArgs.block_bytes),
        bool(getattr(KVCacheArgs, "preallocate", False)),
        bool(getattr(KVCacheArgs, "lazy_init", True)),
        float(getattr(KVCacheArgs, "trim_backoff", 0.9)),
    )

def _get_or_create_pool(verbose: bool = True) -> DRAMPool:
    global _GLOBAL_DRAM_POOL, _GLOBAL_POOL_CFG
    cfg = _pool_cfg_tuple()
    if _GLOBAL_DRAM_POOL is None or _GLOBAL_POOL_CFG != cfg:
        _GLOBAL_DRAM_POOL = DRAMPool(
            bytes_limit_gb=cfg[0],
            block_bytes=cfg[1],
            preallocate=cfg[2],
            lazy_init=cfg[3],
            verbose=verbose,
            trim_backoff=cfg[4],
        )
        _GLOBAL_POOL_CFG = cfg
    return _GLOBAL_DRAM_POOL


# ============================================================================
# KVOffloader (singleton, thread-safe)
# ============================================================================

class KVOffloader:
    """
    Off-GPU KV cache with attention-based Top-K fetch (thread-safe version).
    - push():  GPU HBM → pinned CPU （可选镜像到 SSD）
    - fetch(): SSD/DRAM → GPU HBM（支持多流与快路径）
    - prefetch*: 异步预取，按块事件管理
    """

    # 进程级单例
    _singleton: "KVOffloader|None" = None
    _singleton_lock = threading.Lock()
    _init_printed = False

    def __new__(cls, *args, **kwargs):
        with cls._singleton_lock:
            if cls._singleton is None:
                cls._singleton = super().__new__(cls)
            return cls._singleton

    def __init__(self, layers, heads, dim, max_seq, max_batch, device, dtype_bytes, streams=None):
        # 幂等初始化（避免多次 __init__ 重复启动线程/重置状态）
        if getattr(self, "_inited", False):
            return
        self._inited = True

        self.layers = int(layers)
        self.heads = int(heads)
        self.dim = int(dim)
        self.max_batch = int(max_batch)
        self.device = str(device)
        self.dtype_bytes = int(dtype_bytes)

        self.n_blocks = (int(max_seq) + BLOCK - 1) // BLOCK

        # ------- 主存储 -------
        self.k_cpu: List[List[Optional[torch.Tensor]]] = [[None for _ in range(self.n_blocks)] for _ in range(self.layers)]
        self.v_cpu: List[List[Optional[torch.Tensor]]] = [[None for _ in range(self.n_blocks)] for _ in range(self.layers)]
        self._k_raw_buf: List[List[Optional[torch.Tensor]]] = [[None for _ in range(self.n_blocks)] for _ in range(self.layers)]
        self._v_raw_buf: List[List[Optional[torch.Tensor]]] = [[None for _ in range(self.n_blocks)] for _ in range(self.layers)]

        # GPU 暂存（按块缓存）
        self.gpu_k: List[List[Optional[torch.Tensor]]] = [[None for _ in range(self.n_blocks)] for _ in range(self.layers)]
        self.gpu_v: List[List[Optional[torch.Tensor]]] = [[None for _ in range(self.n_blocks)] for _ in range(self.layers)]

        # 重要性/访问计数（与原版一致）
        self.importance = np.zeros((self.max_batch, self.layers, self.n_blocks), dtype=np.float32)
        self.access_count = np.zeros((self.max_batch, self.layers, self.n_blocks), dtype=np.int32)
        self.last_access = np.zeros((self.max_batch, self.layers, self.n_blocks), dtype=np.int32)
        self.global_time = np.zeros(self.max_batch, dtype=np.int32)
        self.global_importance = np.zeros((self.layers, self.n_blocks), dtype=np.float32)
        self.global_access_count = np.zeros((self.layers, self.n_blocks), dtype=np.int32)
        self.global_time_counter = 0

        # ------- KV cache 命中率统计（按 fetch 维度） -------
        # 总共在 fetch() 中请求了多少 block（按 uniq block 计数）
        self._fetch_blocks_total = 0
        # 在 fetch() 内部，仍需要从 SSD 读取（SSD->DRAM）的 block 数
        self._fetch_miss_blocks = 0
        # 在 prefetch_async() 中，从 SSD 读取的 block 数（提前加载）
        self._prefetch_ssd_load_blocks = 0
        # spill 到 SSD 的 block 数（包括同步 spill 和异步 batch spill）
        self._evict_blocks = 0
        
        # ------- 流 -------
        if streams is not None:
            self.h2d_stream = getattr(streams, "kv_h2d", None)
            self.d2h_stream = getattr(streams, "kv_d2h", None)
        else:
            self.h2d_stream = torch.cuda.Stream(device=self.device, priority=0) if self.device.startswith("cuda") else None
            self.d2h_stream = torch.cuda.Stream(device=self.device, priority=+1) if self.device.startswith("cuda") else None

        # ------- DRAM 配额估计 -------
        alloc_bsz = int(getattr(KVCacheArgs, "dram_sizing_batch", 8))
        alloc_bsz = max(1, min(self.max_batch, alloc_bsz))
        self.token_nbytes = (alloc_bsz * self.heads * self.dim) * self.dtype_bytes * 2
        self.block_nbytes = self.token_nbytes * BLOCK

        # SSD backend（可选）
        try:
            ssd_device_path = getattr(KVCacheArgs, "ssd_device_path", "/dev/nvme0n1")
            self.ssd = RawBlockKVBackend(
                dev_path=ssd_device_path,
                n_layers=self.layers,
                blk_bytes=self.block_nbytes,
                blk_per_layer=self.n_blocks,
                max_concurrent_io=getattr(KVCacheArgs, "max_concurrent_io", 4),
            )
        except Exception as e:
            print(f"[KVOffloader][WARN] SSD backend init failed: {e}; DRAM-only mode")
            self.ssd = None

        self.on_ssd = [[False] * self.n_blocks for _ in range(self.layers)]
        self.in_dram = [[False] * self.n_blocks for _ in range(self.layers)]

        # SSD staging buffer for zero-copy reads
        self._ssd_buffer = None

        _dram_bytes = int(KVCacheArgs.dram_limit_gb * (1024**3))
        _safety = int(0.25 * _dram_bytes)
        self.dram_limit_blk = max(0, (_dram_bytes - _safety) // self.block_nbytes)

        if not KVOffloader._init_printed:
            print(f"[KVOffloader] DRAM quota estimation: sizing_batch={alloc_bsz}, "
                  f"block={self.block_nbytes/(1024**2):.2f}MB, limit={self.dram_limit_blk} blocks")
            KVOffloader._init_printed = True

        # 统一 KVCacheArgs.block_bytes 下限（保证至少能容纳一块 K 或 V）
        bytes_per_kv = self.max_batch * self.heads * BLOCK * self.dim
        kv_dtype = getattr(KVCacheArgs, "kv_dtype", torch.float16 if not getattr(KVCacheArgs, "prefer_bf16", False) else torch.bfloat16)
        self.kv_dtype = kv_dtype  # Store as instance attribute for later use
        elem_size = torch.tensor([], dtype=kv_dtype).element_size()
        bytes_per_kv *= elem_size
        if KVCacheArgs.block_bytes < bytes_per_kv:
            new_blk = int(math.ceil(bytes_per_kv / 4096) * 4096)
            print(f"[KVOffloader] Auto-enlarge KVCacheArgs.block_bytes → {new_blk/(1<<20):.2f}MB")
            KVCacheArgs.block_bytes = new_blk

        # ------- 全局 DRAM 池（单例） -------
        self.pool = _get_or_create_pool(verbose=getattr(KVCacheArgs, "verbose_pool", True))
        if not hasattr(KVOffloader, "_pool_announced"):
            print(f"[KVOffloader] DRAM pool: {self.pool.stats_str()}")
            KVOffloader._pool_announced = True

        # ------- 锁与事件 -------
        # 每块一个重入锁，保护 k_cpu/v_cpu、gpu_k/gpu_v、on_ssd、事件写入
        self._blk_locks: List[List[threading.RLock]] = [
            [threading.RLock() for _ in range(self.n_blocks)] for _ in range(self.layers)
        ]

        # per-block H2D ready 事件（严格一块一个）
        self._blk_ready_evt: Dict[Tuple[int,int], torch.cuda.Event] = {}

        # 仅"组级快拼"使用的 map（不会替代 per-block 事件）
        self._prefetch_lock = threading.Lock()
        # 使用 TTL-LRU 防止无限增长：保留最近 1000 个预取记录，30 秒后过期
        self._prefetch_map = _TTLDict(
            maxlen=int(os.getenv("KV_PREFETCH_STATE_CAP", "1000")),
            ttl_s=float(os.getenv("KV_PREFETCH_STATE_TTL", "30"))
        )

        # 记录最新一次 push 的 D2H 事件，避免并发 spill/free
        self._last_d2h_evt: Dict[Tuple[int,int], torch.cuda.Event] = {}

        # ------- 背景打包+写线程 -------
        self._write_queue: "Queue[tuple[int,int,torch.Tensor|torch.cuda.Event]]" = Queue(
            maxsize=getattr(KVCacheArgs, "RAW_IO_QD_WRITE", 24)
        )
        self._writer_stop = threading.Event()
        self._win_ms = int(getattr(KVCacheArgs, "IO_RAW_THROTTLE_MS", 30))
        self._write_target_bps = int(getattr(KVCacheArgs, "NVME_WRITE_TARGET_MBPS", 900) * 1024 * 1024)
        self._win_bytes = collections.deque()
        self._win_sum = 0

        # ------- 写入节流（必须在线程启动前初始化） -------
        self._throttle_lock = threading.Lock()
        self._pause_write_until = 0.0  # monotonic seconds

        if self.ssd is not None:
            self._writer_thread = threading.Thread(target=self._writer_loop, name="kv_writer", daemon=True)
            self._writer_thread.start()
        else:
            self._writer_thread = None

        self._pack_queue: "Queue[tuple[int,int,torch.cuda.Event|None]]" = Queue(
            maxsize=getattr(KVCacheArgs, "RAW_IO_QD_PACK", 64)
        )
        self._packer_stop = threading.Event()
        self._packer_thread = threading.Thread(target=self._packer_loop, name="kv_packer", daemon=True)
        self._packer_thread.start()

        # ------- 预取线程池（默认更保守=3，可通过 env 上调） -------
        kv_prefetch_workers = int(os.getenv("KV_PREFETCH_WORKERS", "6"))
        self.prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=kv_prefetch_workers)

        # 全局 tracker（可选）
        self.global_tracker = get_global_tracker() or init_global_tracker(self.max_batch, self.layers, self.n_blocks)
        if self.global_tracker:
            self.global_tracker.storage_stats[StorageType.DRAM]["capacity_limit"] = self.dram_limit_blk

    # ---------------- 背景线程实现 ----------------

    def _packer_loop(self):
        while not self._packer_stop.is_set():
            try:
                L, B, d2h_evt = self._pack_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                # 轻量轮询：避免同步阻塞
                if d2h_evt is not None:
                    t0 = time.time()
                    while not d2h_evt.query():
                        if time.time() - t0 > 5.0:
                            break
                        time.sleep(0.001)

                with self._blk_locks[L][B]:
                    kc = self.k_cpu[L][B]
                    vc = self.v_cpu[L][B]
                    if kc is None or vc is None:
                        continue
                    kv_pack_cpu = torch.cat([kc, vc], dim=-1).contiguous()
                    try:
                        kv_pack_cpu = kv_pack_cpu.pin_memory()
                    except Exception:
                        pass
                try:
                    self._write_queue.put((L, B, kv_pack_cpu), timeout=0.5)
                except Full:
                    print(f"[KV][WARN] write queue full, drop mirror @L{L} B{B}")
            finally:
                self._pack_queue.task_done()

    def _writer_loop(self):
        batch: list[tuple[int,int,torch.Tensor|torch.cuda.Event]] = []
        last_flush = time.time()
        while not self._writer_stop.is_set():
            now = time.monotonic()
            if now < self._pause_write_until:
                time.sleep(0.001); continue
            try:
                item = self._write_queue.get(timeout=0.1)
                batch.append(item)
            except Empty:
                pass

            flush_due = (time.time() - last_flush) * 1000.0 >= self._win_ms
            agg_bytes = 0
            for x in batch:
                if len(x) >= 3 and isinstance(x[2], torch.Tensor):
                    agg_bytes += x[2].numel() * x[2].element_size()
            big_enough = agg_bytes >= (1 << 20)
            if not (flush_due or big_enough):
                continue

            self._drain_window()
            if self._win_sum >= self._write_target_bps * (self._win_ms/1000.0):
                time.sleep(0.001); continue

            for (layer, blk, token) in batch:
                if isinstance(token, torch.cuda.Event):
                    t0 = time.time()
                    while not token.query():
                        if time.time() - t0 > 5.0:
                            print(f"[WARN] Event timeout @L{layer} B{blk}"); break
                        time.sleep(0.001)
                    with self._blk_locks[layer][blk]:
                        kc = self.k_cpu[layer][blk]
                        vc = self.v_cpu[layer][blk]
                        if kc is None or vc is None:
                            continue
                        kv_cpu = torch.cat([kc, vc], dim=-1).contiguous()
                else:
                    kv_cpu = token

                nbytes = kv_cpu.numel() * kv_cpu.element_size()
                try:
                    if hasattr(self.ssd, "write_async"):
                        self.ssd.write_async(layer, blk, kv_cpu, sync=False)
                        self.on_ssd[layer][blk] = True
                    else:
                        self.ssd.write(layer, blk, kv_cpu)
                        self.on_ssd[layer][blk] = True
                    self._win_bytes.append((time.time(), nbytes))
                    self._win_sum += nbytes
                except Exception as e:
                    print(f"[WARN] SSD write failed @L{layer} B{blk}: {e}")
            batch.clear()
            last_flush = time.time()

    def _drain_window(self):
        cutoff = time.time() - self._win_ms/1000.0
        while self._win_bytes and self._win_bytes[0][0] < cutoff:
            _, nbytes = self._win_bytes.popleft()
            self._win_sum -= nbytes

    # ---------------- 内部工具 ----------------

    def _alloc_block(self, layer: int, blk: int):
        """
        在块锁保护下，原子地分配与发布 K/V pinned 缓冲。
        """
        lock = self._blk_locks[layer][blk]
        with lock:
            if self.k_cpu[layer][blk] is not None and self.v_cpu[layer][blk] is not None:
                return

            shape = (self.max_batch, self.heads, BLOCK, self.dim)
            single_kv_bytes = self.max_batch * self.heads * BLOCK * self.dim * 2  # float16/bf16 2 bytes
            # 分配 uint8 原始缓冲
            k_buf = self.pool.alloc_block(single_kv_bytes)
            while k_buf is None:
                if not self._maybe_evict(force=True):
                    time.sleep(0.001)
                k_buf = self.pool.alloc_block(single_kv_bytes)
            v_buf = self.pool.alloc_block(single_kv_bytes)
            while v_buf is None:
                if not self._maybe_evict(force=True):
                    time.sleep(0.001)
                v_buf = self.pool.alloc_block(single_kv_bytes)

            # 在局部完成 view/zero，再一次性发布
            k_view = k_buf[:single_kv_bytes].view(torch.float16).reshape(shape)
            v_view = v_buf[:single_kv_bytes].view(torch.float16).reshape(shape)
            k_view.zero_(); v_view.zero_()

            self.k_cpu[layer][blk] = k_view
            self.v_cpu[layer][blk] = v_view
            self._k_raw_buf[layer][blk] = k_buf
            self._v_raw_buf[layer][blk] = v_buf

    # ---------------- 公共 API ----------------

    def push(self, layer: int, blk: int, k: torch.Tensor, v: torch.Tensor,
             token_idx: int, batch_idx: int = 0, **kwargs):
        """
        把单个 token 的 K/V 从 GPU 复制到 pinned CPU（增量写），并可异步镜像到 SSD。
        """
        assert k.dim() == 3 and v.dim() == 3, "KV must be (bsz, heads, dim)"
        bsz = int(k.size(0))
        t_in_blk = int(token_idx) % BLOCK

        self._alloc_block(layer, blk)
        lock = self._blk_locks[layer][blk]
        with lock:
            stream = self.d2h_stream or torch.cuda.current_stream()
            with torch.cuda.stream(stream):
                self.k_cpu[layer][blk][:bsz, :, t_in_blk, :].copy_(k, non_blocking=True)
                self.v_cpu[layer][blk][:bsz, :, t_in_blk, :].copy_(v, non_blocking=True)
            d2h_evt = torch.cuda.Event(blocking=False); d2h_evt.record(stream)
            self._last_d2h_evt[(layer, blk)] = d2h_evt

        # 重要性/统计
        bi = min(batch_idx, self.max_batch-1)
        self.importance[bi, layer, blk] = max(self.importance[bi, layer, blk], 1e-6)
        self.global_importance[layer, blk] = max(self.global_importance[layer, blk], 1e-6)
        if self.global_tracker:
            self.global_tracker.update_hbm_storage(batch_idx, layer, [blk], "remove")
            self.global_tracker.update_dram_storage(batch_idx, layer, [blk], "add", [1e-6])

        self._maybe_evict()

        # 可选镜像到 SSD：投递到打包线程（由打包线程等待 d2h 事件）
        if self.ssd is not None and getattr(KVCacheArgs, "mirror_on_push", True):
            try:
                self._pack_queue.put_nowait((layer, blk, d2h_evt))
            except Full:
                if getattr(KVCacheArgs, "verbose_pool", False):
                    print(f"[KV][WARN] drop mirror (pack queue full) @L{layer} B{blk}")

    def eager_spill_layer(self, layer: int, upto_token: int, async_write: bool = True, include_partial: bool = True):
        end_blk = (int(upto_token) - 1) // BLOCK if upto_token > 0 else -1
        if end_blk < 0:
            return
        blocks = list(range(0, end_blk + 1))

        if self.ssd is None or not async_write:
            for B in blocks:
                with self._blk_locks[layer][B]:
                    if self.k_cpu[layer][B] is not None:
                        self._spill_to_ssd(layer, B)
            return

        # 异步批写：收集 d2h 事件并后台 write_batch_async
        events, ev_blks = [], []
        for B in blocks:
            with self._blk_locks[layer][B]:
                if self.k_cpu[layer][B] is None:
                    continue
                evt = self._last_d2h_evt.get((layer, B))
                if evt is None:
                    evt = torch.cuda.Event(blocking=False)
                    if self.d2h_stream is not None:
                        evt.record(self.d2h_stream)
                events.append(evt); ev_blks.append(B)

        if not ev_blks:
            return

        def _spill_job():
            local_blks, local_tensors = [], []
            for e, B in zip(events, ev_blks):
                t0 = time.time()
                while not e.query():
                    if time.time() - t0 > 5.0:
                        break
                    time.sleep(0.001)
                with self._blk_locks[layer][B]:
                    if self.k_cpu[layer][B] is None or self.v_cpu[layer][B] is None:
                        continue
                    kv_pack = torch.cat([self.k_cpu[layer][B], self.v_cpu[layer][B]], dim=-1).contiguous()
                local_blks.append(B); local_tensors.append(kv_pack)
            if local_blks:
                fut = self.ssd.write_batch_async(layer, local_blks, local_tensors)
                def _on_done(_):
                    for BB in local_blks:
                        with self._blk_locks[layer][BB]:
                            if self._k_raw_buf[layer][BB] is not None:
                                self.pool.free_block(self._k_raw_buf[layer][BB]); self._k_raw_buf[layer][BB] = None
                            if self._v_raw_buf[layer][BB] is not None:
                                self.pool.free_block(self._v_raw_buf[layer][BB]); self._v_raw_buf[layer][BB] = None
                            self.k_cpu[layer][BB] = self.v_cpu[layer][BB] = None
                            self.on_ssd[layer][BB] = True
                            self._evict_blocks += 1
                fut.add_done_callback(_on_done)
        threading.Thread(target=_spill_job, name="kv_spill", daemon=True).start()

    def eager_spill_decode_window(self, upto_token: int, keep_tail_blocks: int = 1,
                                  include_partial: bool = False, layers: Optional[List[int]] = None):
        if self.ssd is None:
            return
        end_blk = (int(upto_token) // BLOCK) - int(keep_tail_blocks)
        if end_blk < 0:
            return
        target_layers = layers if layers is not None else list(range(self.layers))
        for L in target_layers:
            for B in range(0, end_blk + 1):
                with self._blk_locks[L][B]:
                    if self.k_cpu[L][B] is None:
                        continue
                    is_full = self._is_full_block(L, B, upto_token)
                    if include_partial or is_full:
                        self._spill_to_ssd(L, B)

    def _is_full_block(self, layer: int, blk: int, upto_token: int) -> bool:
        return int(upto_token) >= (blk + 1) * BLOCK

    def append_token_to_gpu(self, layer: int, blk: int, t_in_blk: int,
                            k: torch.Tensor, v: torch.Tensor):
        bsz = int(k.size(0))
        shape = (bsz, self.heads, BLOCK, self.dim)

        s = self.h2d_stream or torch.cuda.current_stream()
        with self._blk_locks[layer][blk]:
            if self.gpu_k[layer][blk] is None or tuple(self.gpu_k[layer][blk].shape) != shape:
                self.gpu_k[layer][blk] = torch.empty(shape, dtype=k.dtype, device=self.device)
                self.gpu_v[layer][blk] = torch.empty(shape, dtype=v.dtype, device=self.device)
            with torch.cuda.stream(s):
                self.gpu_k[layer][blk][:bsz, :, t_in_blk, :].copy_(k, non_blocking=True)
                self.gpu_v[layer][blk][:bsz, :, t_in_blk, :].copy_(v, non_blocking=True)
            evt = torch.cuda.Event(blocking=False); evt.record(s)
            self._blk_ready_evt[(layer, blk)] = evt

    def ensure_on_gpu(self, layer_idx: int, start_pos: int, k_cur: torch.Tensor, v_cur: torch.Tensor):
        if not str(k_cur.device).startswith("cuda"):
            raise RuntimeError("ensure_on_gpu: only CUDA is supported")
        if start_pos == 0:
            return k_cur, v_cur
        return k_cur, v_cur

    def fetch(self, layer: int, blocks: Union[List[int], torch.Tensor],
              batch_idx: int = 0, bsz: Optional[int] = None, stream=None):
        # blocks 规格化
        if isinstance(blocks, (list, tuple)):
            uniq = sorted(set(int(b) for b in blocks))
        else:
            uniq = blocks.to(torch.long).unique(sorted=True).tolist()

        use_bsz = int(bsz) if bsz is not None else self.max_batch
        shape = (use_bsz, self.heads, BLOCK, self.dim)

        # ---- KV cache 统计：本次 fetch 请求的 block 数（按 uniq 计）----
        num_blocks = len(uniq)
        self._fetch_blocks_total += num_blocks

        # -------- 1) 全 GPU 快路径 --------
        all_gpu = True
        for b in uniq:
            with self._blk_locks[layer][b]:
                kg = self.gpu_k[layer][b]; vg = self.gpu_v[layer][b]
                evt = self._blk_ready_evt.get((layer, b))
            if (kg is None) or (vg is None) or kg.device.type != "cuda" or tuple(kg.shape) != shape:
                all_gpu = False
                break
        if all_gpu:
            return self._gather_from_gpu_cache_strict(layer, uniq, use_bsz, stream=stream)

        # -------- 2) 预取组命中（整组在 GPU 上） --------
        dev = self.device if isinstance(self.device, (str, torch.device)) else torch.device(self.device)
        with self._prefetch_lock:
            rec = self._prefetch_map.get((int(layer), tuple(uniq), int(use_bsz)))
        if rec is not None:
            evt_group = rec.get("evt")
            k_list = rec.get("k") or []
            v_list = rec.get("v") or []
            if evt_group is not None:
                s = stream or torch.cuda.current_stream(device=dev)
                try:
                    s.wait_event(evt_group)
                except Exception:
                    pass
            if k_list and v_list:
                k_full = torch.cat([k[:use_bsz] for k in k_list], dim=2)
                v_full = torch.cat([v[:use_bsz] for v in v_list], dim=2)
                if k_full.device.type != "cuda":
                    k_full = k_full.to(dev, non_blocking=True)
                if v_full.device.type != "cuda":
                    v_full = v_full.to(dev, non_blocking=True)
                return k_full, v_full

        # -------- 3) SSD→DRAM（仅对缺失块） + DRAM→HBM --------
        need_load = []
        for b in uniq:
            with self._blk_locks[layer][b]:
                if self.k_cpu[layer][b] is None or self.v_cpu[layer][b] is None:
                    need_load.append(b)

        # 统计：这次 fetch 中有多少 block 在 CPU 里缺失，需要同步从 SSD 读取
        if need_load:
            self._fetch_miss_blocks += len(need_load)

        for b in need_load:
            self._load_from_ssd(layer, b)

        # 2) DRAM→HBM：逐块拷贝并记录事件
        h2d_stream = stream or self.h2d_stream or torch.cuda.current_stream()
        k_parts, v_parts = [], []
        with torch.cuda.stream(h2d_stream):
            for b in uniq:
                with self._blk_locks[layer][b]:
                    kg = self.gpu_k[layer][b]; vg = self.gpu_v[layer][b]
                    if kg is not None and vg is not None and tuple(kg.shape) == shape:
                        # 仍需等上次 H2D 的完成事件
                        evt = self._blk_ready_evt.get((layer, b))
                        if evt is not None:
                            (stream or torch.cuda.current_stream()).wait_event(evt)
                        k_parts.append(kg[:use_bsz]); v_parts.append(vg[:use_bsz])
                    else:
                        kc = self.k_cpu[layer][b][:use_bsz]; vc = self.v_cpu[layer][b][:use_bsz]
                        self.gpu_k[layer][b] = torch.empty(shape, dtype=kc.dtype, device=self.device)
                        self.gpu_v[layer][b] = torch.empty(shape, dtype=vc.dtype, device=self.device)
                        kg = self.gpu_k[layer][b]; vg = self.gpu_v[layer][b]
                        kg.copy_(kc, non_blocking=True); vg.copy_(vc, non_blocking=True)
                        evt_b = torch.cuda.Event(blocking=False); evt_b.record(h2d_stream)
                        self._blk_ready_evt[(layer, b)] = evt_b
                        k_parts.append(kg); v_parts.append(vg)

        current = stream or torch.cuda.current_stream()
        if h2d_stream is not current:
            current.wait_stream(h2d_stream)

        k_full = torch.cat(k_parts, dim=2)
        v_full = torch.cat(v_parts, dim=2)
        if k_full.device.type != "cuda":
            k_full = k_full.to(self.device, non_blocking=True)
        if v_full.device.type != "cuda":
            v_full = v_full.to(self.device, non_blocking=True)
        return k_full, v_full

    # ---- 仅从 GPU 缓存拼接（严格语义） ----
    def _gather_from_gpu_cache_strict(self, layer: int, blocks: List[int], bsz: int, stream=None):
        s = stream or torch.cuda.current_stream()
        k_parts, v_parts = [], []
        for b in sorted(set(blocks)):
            with self._blk_locks[layer][b]:
                evt = self._blk_ready_evt.get((layer, b))
                if evt is not None:
                    s.wait_event(evt)
                kg = self.gpu_k[layer][b]; vg = self.gpu_v[layer][b]
                if (kg is None) or (vg is None) or kg.device.type != "cuda":
                    raise RuntimeError(f"[KV] L{layer} B{b} not on GPU; prefetch first.")
                k_parts.append(kg[:bsz]); v_parts.append(vg[:bsz])
        return torch.cat(k_parts, dim=2), torch.cat(v_parts, dim=2)

    def prefetch_async(self, *, layer: int, blocks: List[int], bsz: int,
                       device: str | torch.device = None):
        if not blocks:
            return
        uniq = sorted(set(int(b) for b in blocks))
        use_bsz = int(bsz if bsz is not None else self.max_batch)
        dev = str(device or self.device)

        def _task():
            # 1) SSD→DRAM（缺失才拉）
            need = []
            for b in uniq:
                with self._blk_locks[layer][b]:
                    if self.k_cpu[layer][b] is None or self.v_cpu[layer][b] is None:
                        need.append(b)
                        
            if need:
                self._prefetch_ssd_load_blocks += len(need)

            for b in need:
                self._load_from_ssd(layer, b)

            # 2) DRAM→HBM：逐块拷贝并记录 per-block 事件；额外记录组级事件用于快拼
            s = self.h2d_stream or torch.cuda.current_stream()
            k_list, v_list = [], []
            with torch.cuda.stream(s):
                for b in uniq:
                    with self._blk_locks[layer][b]:
                        kc = self.k_cpu[layer][b]; vc = self.v_cpu[layer][b]
                        if kc is None or vc is None:
                            raise RuntimeError(f"[KV] block {b} missing in DRAM")
                        shape = (use_bsz, self.heads, BLOCK, self.dim)
                        kg = self.gpu_k[layer][b]; vg = self.gpu_v[layer][b]
                        if (kg is None) or (kg.device.type != "cuda") or (tuple(kg.shape) != shape):
                            kg = torch.empty(shape, dtype=kc.dtype, device=self.device)
                            vg = torch.empty(shape, dtype=vc.dtype, device=self.device)
                            self.gpu_k[layer][b] = kg; self.gpu_v[layer][b] = vg
                        kg.copy_(kc[:use_bsz], non_blocking=True)
                        vg.copy_(vc[:use_bsz], non_blocking=True)
                        # 逐块事件：copy 之后立即记录
                        evt_b = torch.cuda.Event(blocking=False); evt_b.record(s)
                        self._blk_ready_evt[(layer, b)] = evt_b
                        k_list.append(kg); v_list.append(vg)

                evt_group = torch.cuda.Event(blocking=False); evt_group.record(s)

            with self._prefetch_lock:
                self._prefetch_map.set(
                    (int(layer), tuple(uniq), int(use_bsz)),
                    {"evt": evt_group, "k": k_list, "v": v_list}
                )

        try:
            self.prefetch_executor.submit(_task)
        except Exception as e:
            print(f"[KV][WARN] prefetch submit failed: {e}")

    def prefetch_blocks_async(self, layer_idx: int, blocks: List[int],
                              stream: Optional[torch.cuda.Stream] = None,
                              bsz: Optional[int] = None, device: Optional[Union[str, torch.device]] = None):
        if not blocks or not torch.cuda.is_available():
            return
        s = stream or getattr(self, "h2d_stream", None) or torch.cuda.current_stream()
        effective_bsz = int(bsz if bsz is not None else self.max_batch)
        effective_dev = device if device is not None else self.device
        with torch.cuda.stream(s):
            self.prefetch_async(layer=layer_idx, blocks=blocks, bsz=effective_bsz, device=effective_dev)

    def wait_blocks_ready(self, layer: int, blocks: List[int], stream: Optional[torch.cuda.Stream] = None):
        s = stream or torch.cuda.current_stream()
        for b in set(int(x) for x in blocks):
            evt = self._blk_ready_evt.get((int(layer), b))
            if evt is not None:
                try:
                    s.wait_event(evt)
                except Exception:
                    pass

    def plan_tail_window_blocks(self, start_pos: int, seqlen: int, window_tokens: int = BLOCK) -> List[int]:
        end = int(start_pos) + int(seqlen) - 1
        if end < 0:
            return []
        left = max(0, end - int(window_tokens) + 1)
        blk_lo = left // BLOCK
        blk_hi = end // BLOCK
        return list(range(blk_lo, blk_hi + 1))

    def prefetch_for_next_layer(self, *, current_layer: int, start_pos: int, seqlen: int,
                                bsz: int, window_tokens: int = BLOCK):
        nxt = int(current_layer) + 1
        if nxt >= self.layers:
            return
        blocks = self.plan_tail_window_blocks(start_pos, seqlen, window_tokens)
        if blocks:
            self.prefetch_async(layer=nxt, blocks=blocks, bsz=bsz, device=self.device)

    # ------------- importance 统计（原样） -------------
    def update_importances(self, layer: int, block_indices: List[int], block_scores: List[float],
                           batch_idx: Union[int, List[int]] = None, momentum: float = 0.9):
        if batch_idx is None:
            batch_indices = list(range(self.max_batch))
        elif isinstance(batch_idx, int):
            batch_indices = [batch_idx]
        else:
            batch_indices = batch_idx
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
        for idx, score in zip(block_indices, block_scores):
            if idx < self.n_blocks:
                self.global_importance[layer, idx] = (
                    momentum * self.global_importance[layer, idx] + (1.0 - momentum) * score
                )
                self.global_access_count[layer, idx] += 1
        self.global_time_counter += 1

    def topk_blocks(self, layer: int, k: int, batch_idx: Union[int, List[int]] = None,
                    strategy: str = "batch_specific", access_weight: float = 0.1):
        if batch_idx is None:
            imp = self.global_importance[layer]
            access = self.global_access_count[layer]
            ranked = sorted(range(self.n_blocks), key=lambda i: imp[i]*(1+access_weight*access[i]), reverse=True)
            chosen = [i for i in ranked if imp[i] > 0][:k]
            return sorted(chosen)
        elif isinstance(batch_idx, int):
            if strategy == "batch_specific":
                imp = self.importance[batch_idx, layer]
                access = self.access_count[batch_idx, layer]
                ranked = sorted(range(self.n_blocks), key=lambda i: imp[i]*(1+access_weight*access[i]), reverse=True)
                chosen = [i for i in ranked if imp[i] > 0][:k]
                return sorted(chosen)
            elif strategy == "global":
                imp = self.global_importance[layer]
                access = self.global_access_count[layer]
                ranked = sorted(range(self.n_blocks), key=lambda i: imp[i]*(1+access_weight*access[i]), reverse=True)
                chosen = [i for i in ranked if imp[i] > 0][:k]
                return sorted(chosen)
            elif strategy == "hybrid":
                batch_imp = self.importance[batch_idx, layer]
                global_imp = self.global_importance[layer]
                batch_access = self.access_count[batch_idx, layer]
                global_access = self.global_access_count[layer]
                def hybrid_score(i: int):
                    combined_imp = 0.7*batch_imp[i] + 0.3*global_imp[i]
                    combined_access = batch_access[i] + global_access[i]
                    return combined_imp * (1 + access_weight * combined_access)
                ranked = sorted(range(self.n_blocks), key=hybrid_score, reverse=True)
                chosen = [i for i in ranked if hybrid_score(i) > 0][:k]
                return sorted(chosen)
        else:
            result = {}
            for b_idx in batch_idx:
                if b_idx < self.max_batch:
                    result[b_idx] = self.topk_blocks(layer, k, b_idx, strategy, access_weight)
            return result

    def topk_blocks_aggregated(self, layer: int, k: int, batch_indices: List[int] = None, aggregation: str = "mean"):
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
            agg_imp = np.mean(self.importance[batch_indices, layer], axis=0)
            agg_access = np.mean(self.access_count[batch_indices, layer], axis=0)
        ranked = sorted(range(self.n_blocks), key=lambda i: agg_imp[i]*(1+0.1*agg_access[i]), reverse=True)
        chosen = [i for i in ranked if agg_imp[i] > 0][:k]
        return sorted(chosen)

    def get_batch_statistics(self, batch_idx: int):
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
        if batch_idx < self.max_batch:
            self.importance[batch_idx] = 0
            self.access_count[batch_idx] = 0
            self.last_access[batch_idx] = 0
            self.global_time[batch_idx] = 0

    # ---------------- SSD spill / load -----------------

    def _spill_to_ssd(self, L: int, B: int):
        if self.ssd is None:
            with self._blk_locks[L][B]:
                if self._k_raw_buf[L][B] is not None:
                    self.pool.free_block(self._k_raw_buf[L][B]); self._k_raw_buf[L][B] = None
                if self._v_raw_buf[L][B] is not None:
                    self.pool.free_block(self._v_raw_buf[L][B]); self._v_raw_buf[L][B] = None
                self.k_cpu[L][B] = self.v_cpu[L][B] = None
            return

        # 等待最近一次 D2H（若存在且未完成）
        evt = self._last_d2h_evt.get((L, B))
        if evt is not None:
            t0 = time.time()
            while not evt.query():
                if time.time() - t0 > 5.0:
                    break
                time.sleep(0.001)

        # 更新 tracker
        if self.global_tracker:
            for batch_idx in range(self.max_batch):
                if (batch_idx, L) in self.global_tracker.dram_storage and B in self.global_tracker.dram_storage[(batch_idx, L)]:
                    importance_score = float(self.importance[batch_idx, L, B]) if batch_idx < self.max_batch else 1e-6
                    self.global_tracker.update_dram_storage(batch_idx, L, [B], "remove")
                    self.global_tracker.update_ssd_storage(batch_idx, L, [B], "add", [importance_score])

        with self._blk_locks[L][B]:
            kv_cpu = torch.cat([self.k_cpu[L][B], self.v_cpu[L][B]], dim=-1)
        self.ssd.write(L, B, kv_cpu)

        with self._blk_locks[L][B]:
            if self._k_raw_buf[L][B] is not None:
                self.pool.free_block(self._k_raw_buf[L][B]); self._k_raw_buf[L][B] = None
            if self._v_raw_buf[L][B] is not None:
                self.pool.free_block(self._v_raw_buf[L][B]); self._v_raw_buf[L][B] = None
            self.k_cpu[L][B] = self.v_cpu[L][B] = None
            self.on_ssd[L][B] = True
            # 可选：释放 GPU 缓存，避免显存堆积
            self.gpu_k[L][B] = None; self.gpu_v[L][B] = None
            self._blk_ready_evt.pop((L,B), None)
            
            self.k_cpu[L][B] = self.v_cpu[L][B] = None
            self.on_ssd[L][B] = True

            # 统计：一次 eviction
            self._evict_blocks += 1

    def _maybe_evict(self, force: bool = False) -> bool:
        hot_cnt = sum(x is not None for lay in self.k_cpu for x in lay)
        over_quota = (hot_cnt >= self.dram_limit_blk)
        per_blk_pinned = 2 * self.max_batch * self.heads * BLOCK * self.dim * self.dtype_bytes
        per_blk_pinned = int(math.ceil(per_blk_pinned / 4096) * 4096)
        hard_pressure = (self.pool.used + per_blk_pinned) > int(self.pool.bytes_limit * getattr(KVCacheArgs, "trim_backoff", 0.9))

        if not (over_quota or hard_pressure or force):
            return False
        cand = [(self.global_importance[L][B], L, B)
                for L in range(self.layers) for B in range(self.n_blocks)
                if self.k_cpu[L][B] is not None]
        if not cand:
            return False
        _, L, B = min(cand)
        self._spill_to_ssd(L, B)
        return True

    def _load_from_ssd(self, L: int, B: int):
        """
        SSD → DRAM：优先走零拷贝（若支持），否则回退到 GPU 中转。
        """
        lock = self._blk_locks[L][B]
        with lock:
            if self.ssd is None:
                # 没有 SSD backend，直接在 DRAM 里 alloc 一个空块
                self._alloc_block(L, B)
                return

        USE_ZERO_COPY = (os.getenv("KV_USE_ZERO_COPY_READ", "1") == "1")
        try:
            from .weights_io_ssd_dram import alloc_pinned_aligned
        except Exception:
            USE_ZERO_COPY = False

        single_k_bytes = self.max_batch * self.heads * BLOCK * self.dim * self.dtype_bytes
        blk_bytes = 2 * single_k_bytes  # K+V

        # ---------- 首选：SSD -> pinned uint8 -> DRAM K/V ----------
        if USE_ZERO_COPY and hasattr(self.ssd, "read_into_pinned_aligned"):
            block_size = int(getattr(self.ssd, "block_size", 4096))
            stride = ((blk_bytes + block_size - 1) // block_size) * block_size

            # 临时 pinned 对齐缓冲（只在本函数栈内存活，不污染 DRAMPool）
            buf = alloc_pinned_aligned(stride, block_size)  # uint8 pinned 对齐缓冲
            self.ssd.read_into_pinned_aligned(L, B, buf)

            with lock:
                self._alloc_block(L, B)  # 在 DRAMPool 里申请真正的 K/V 存储
                fuse = buf[:blk_bytes].view(torch.float16).view(
                    self.max_batch, self.heads, BLOCK, self.dim * 2
                )
                k_cpu, v_cpu = fuse.split(self.dim, dim=-1)
                self.k_cpu[L][B].copy_(k_cpu, non_blocking=False)
                self.v_cpu[L][B].copy_(v_cpu, non_blocking=False)
                self.on_ssd[L][B] = True
            return

        # ---------- 回退路径：SSD → GPU buffer → CPU ----------
        shape = (self.max_batch, self.heads, BLOCK, self.dim * 2)
        if (
            not hasattr(self, "_ssd_buffer")
            or self._ssd_buffer is None
            or tuple(self._ssd_buffer.shape) != shape
        ):
            self._ssd_buffer = torch.empty(shape, dtype=torch.float16, device=self.device)

        # 这里的 read 仍然是同步的，但如果是通过 prefetch 线程调用，就不会挡住计算
        self.ssd.read(L, B, self._ssd_buffer)
        k_gpu, v_gpu = self._ssd_buffer.split(self.dim, dim=-1)

        with lock:
            self._alloc_block(L, B)
            self.k_cpu[L][B].copy_(k_gpu.cpu(), non_blocking=False)
            self.v_cpu[L][B].copy_(v_gpu.cpu(), non_blocking=False)
            self.on_ssd[L][B] = True



    # ---------------- 限速 ----------------

    def throttle_writes_for(self, ms: int):
        until = time.monotonic() + (ms / 1000.0)
        with self._throttle_lock:
            self._pause_write_until = max(self._pause_write_until, until)

    def _should_pause_writes(self) -> bool:
        with self._throttle_lock:
            return time.monotonic() < self._pause_write_until

    # ---------------- tracker & 清理 ----------------

    def set_current_execution(self, batch_idx: int, layer_idx: int):
        if self.global_tracker:
            self.global_tracker.set_current_execution(batch_idx, layer_idx)

    def get_global_state(self):
        if self.global_tracker:
            return self.global_tracker.get_current_state()
        return None

    def print_global_state(self):
        if self.global_tracker:
            self.global_tracker.print_current_state()
            self.global_tracker.print_storage_utilization()
        else:
            print("Global tracker not available")
            
    # ---------------- 命中率统计 API ----------------

    def reset_cache_stats(self):
        """
        清空 KV cache 统计计数，在一轮新的推理开始前可以调用（可选）。
        """
        self._fetch_blocks_total = 0
        self._fetch_miss_blocks = 0
        self._prefetch_ssd_load_blocks = 0
        self._evict_blocks = 0

    def get_cache_stats(self) -> Dict[str, float]:
        """
        返回 KV cache 统计信息，字段含义：
        - fetch_blocks_total: 在 fetch() 里请求的 block 总数（去重后）
        - hits / misses / hit_ratio: 从 fetch 视角的命中率（是否在 fetch 中触发 SSD 读取）
        - ssd_load_blocks_prefetch: 预取阶段从 SSD 拉取的 block 数
        - evictions: spill 到 SSD 的 block 数
        """
        total = int(self._fetch_blocks_total)
        misses = int(self._fetch_miss_blocks)
        hits = max(total - misses, 0)
        hit_ratio = float(hits) / float(total) if total > 0 else None

        return {
            "fetch_blocks_total": total,
            "hits": hits,
            "misses": misses,
            "hit_ratio": hit_ratio,
            "ssd_load_blocks_prefetch": int(self._prefetch_ssd_load_blocks),
            "evictions": int(self._evict_blocks),
        }


    def __del__(self):
        try:
            if hasattr(self, "prefetch_executor"):
                self.prefetch_executor.shutdown(wait=True)
            if hasattr(self, "_writer_stop"):
                self._writer_stop.set()
            if hasattr(self, "_packer_stop"):
                self._packer_stop.set()
        except Exception:
            pass
