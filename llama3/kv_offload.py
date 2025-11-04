from __future__ import annotations
import threading, time, collections, math
from queue import Empty, Queue, Full
from typing import List, Iterable, Tuple, Union, Optional
import numpy as np
import torch
import concurrent.futures

from .SSDBacked import RawBlockKVBackend
from .config import KVCacheArgs
from .global_state_tracker import get_global_tracker, init_global_tracker, StorageType


BLOCK = 256  # tokens / block


# ============================================================================
# Global DRAM Pool (Singleton for all layers)
# ============================================================================

class DRAMPool:
    """
    全局 DRAM 池，支持预分配和懒分配两种模式
    Global DRAM pool supporting both preallocate and lazy allocation modes
    """
    def __init__(self, bytes_limit_gb: float, block_bytes: int,
                 preallocate: bool = False, lazy_init: bool = True,
                 verbose: bool = True):
        self.bytes_limit = int(bytes_limit_gb * (1 << 30))
        self.block_bytes = int(block_bytes)
        # ★ 向上对齐到 4KiB（与后续可能的 O_DIRECT 对接安全）
        self.block_bytes = int(math.ceil(self.block_bytes / 4096) * 4096)

        self.preallocate = preallocate
        self.lazy_init = lazy_init
        self.used = 0
        self.lock = threading.Lock()

        # 懒分配池：可复用 pinned 块的栈 & 活跃集合（用 data_ptr 跟踪）
        self.lazy_free = []
        self.lazy_live = set()

        if preallocate:
            # Preallocate mode: allocate entire buffer upfront (旧逻辑)
            try:
                self.dram_buf = torch.empty(self.bytes_limit, dtype=torch.uint8, pin_memory=True)
                self.free_segments = [(0, self.bytes_limit)]  # List of (offset, size) tuples
                if verbose:
                    print(f"[DRAMPool] Preallocated {self.bytes_limit / (1<<30):.2f} GB")
            except RuntimeError as e:
                if verbose:
                    print(f"[DRAMPool][WARN] Preallocate failed ({e}), falling back to lazy mode")
                self.dram_buf = None
                self.free_segments = []
                self.lazy_init = True
        else:
            # Lazy allocation mode: allocate blocks on-demand (新的懒分配模式)
            self.dram_buf = None
            self.free_segments = []
            if verbose:
                print(f"[DRAMPool] Lazy allocation mode (on-demand, limit={self.bytes_limit / (1<<30):.2f} GB)")

    def _alloc_from_segments(self, need):
        """从预分配的大缓冲 carve 出一段，返回 offset；若失败返回 None"""
        # 最简单的 first-fit 实现
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

    def _coalesce_segment(self, off, sz):
        """把 [off, off+sz) 插回 free list，并与邻接段合并"""
        segs = self.free_segments
        segs.append((off, sz))
        segs.sort()  # 按 offset
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
        """
        分配一个 pinned DRAM 块，返回 torch.uint8 张量视图；失败返回 None
        Allocate a pinned DRAM block, return torch.uint8 tensor view; None on failure
        """
        need_bytes = nbytes if nbytes is not None else self.block_bytes
        # 向上对齐到 4KiB
        need_bytes = int(math.ceil(need_bytes / 4096) * 4096)

        with self.lock:
            # 预分配模式：从大缓冲上 carve
            if self.dram_buf is not None:
                if self.used + need_bytes > self.bytes_limit:
                    return None
                off = self._alloc_from_segments(need_bytes)
                if off is None:
                    return None
                self.used += need_bytes
                # narrow 返回 pinned 的子张量（不拷贝）
                return self.dram_buf.narrow(0, off, need_bytes)

            # 懒分配模式：逐块分配或复用
            if self.used + need_bytes > self.bytes_limit:
                return None

            # 查找匹配大小的空闲块（简单的精确匹配）
            for i, t in enumerate(self.lazy_free):
                if t.numel() == need_bytes:
                    blk = self.lazy_free.pop(i)
                    self.lazy_live.add(blk.storage().data_ptr())
                    self.used += need_bytes
                    return blk

            # 没有匹配的，分配新块
            try:
                t = torch.empty(need_bytes, dtype=torch.uint8, pin_memory=True)
                # self.lazy_live.add(t.storage().data_ptr())
                self.lazy_live.add(t.untyped_storage().data_ptr())
                self.used += need_bytes
                return t
            except RuntimeError as e:
                print(f"[DRAMPool][ERROR] Failed to allocate {need_bytes / (1<<20):.2f} MB: {e}")
                return None

    def free_block(self, buf: torch.Tensor):
        """
        释放一个由 alloc_block 产生的块（支持两种模式）
        Free a block produced by alloc_block (supports both modes)
        """
        with self.lock:
            if self.dram_buf is not None:
                # 预分配模式：把子视图的 offset 封回 free list
                off = buf.storage_offset()
                size = buf.numel()
                self._coalesce_segment(off, size)
                self.used -= size
            else:
                # 懒分配：回收到复用栈，避免频繁 CUDA pinned alloc/free
                # ptr = buf.storage().data_ptr()
                ptr = buf.untyped_storage().data_ptr()
                if ptr in self.lazy_live:
                    self.lazy_live.remove(ptr)
                    self.lazy_free.append(buf)
                    self.used -= buf.numel()

    def stats_str(self) -> str:
        """返回池的统计信息字符串"""
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
    """获取当前配置的元组（用于检测配置变化）"""
    return (
        float(KVCacheArgs.dram_limit_gb),
        int(KVCacheArgs.block_bytes),
        bool(getattr(KVCacheArgs, "preallocate", False)),
        bool(getattr(KVCacheArgs, "lazy_init", True)),
    )


def _get_or_create_pool(verbose: bool = True) -> DRAMPool:
    """
    获取或创建全局 DRAM 池（单例模式）
    Get or create global DRAM pool (singleton pattern)
    """
    global _GLOBAL_DRAM_POOL, _GLOBAL_POOL_CFG
    cfg = _pool_cfg_tuple()
    if _GLOBAL_DRAM_POOL is None or _GLOBAL_POOL_CFG != cfg:
        _GLOBAL_DRAM_POOL = DRAMPool(
            bytes_limit_gb=cfg[0],
            block_bytes=cfg[1],
            preallocate=cfg[2],
            lazy_init=cfg[3],
            verbose=verbose,
        )
        _GLOBAL_POOL_CFG = cfg
    return _GLOBAL_DRAM_POOL


# ============================================================================

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

        # 跟踪底层 uint8 buffer (用于 free_block)
        # Track underlying uint8 buffers (for free_block)
        self._k_raw_buf = [[None for _ in range(self.n_blocks)] for _ in range(layers)]
        self._v_raw_buf = [[None for _ in range(self.n_blocks)] for _ in range(layers)]

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

        # --- Auto-adjust KVCacheArgs.block_bytes if needed ---
        # 计算单个 KV 块所需字节数（用于确保池块足够大）
        bytes_per_kv = self.max_batch * self.heads * BLOCK * self.dim
        kv_dtype = getattr(KVCacheArgs, "kv_dtype", None)
        if kv_dtype is None:
            # 如果 prefer_bf16=True，使用 bfloat16；否则使用 float16
            if getattr(KVCacheArgs, "prefer_bf16", False):
                kv_dtype = torch.bfloat16
            else:
                kv_dtype = torch.float16  # 默认使用 float16
        elem_size = torch.tensor([], dtype=kv_dtype).element_size()
        bytes_per_kv *= elem_size  # 通常是 2B for fp16/bf16

        # 如果配置的块大小太小，自动提升到最近的 4KiB 对齐值
        if KVCacheArgs.block_bytes < bytes_per_kv:
            new_blk = int(math.ceil(bytes_per_kv / 4096) * 4096)
            if getattr(KVCacheArgs, "verbose_pool", True):
                print(f"[KVOffloader] Auto-enlarge block_bytes: {KVCacheArgs.block_bytes / (1<<20):.2f} MB → "
                      f"{new_blk / (1<<20):.2f} MB (≥ one KV block {bytes_per_kv / (1<<20):.2f} MB)")
            KVCacheArgs.block_bytes = new_blk

        # --- DRAM Pool (Global Singleton) ---
        self.pool = _get_or_create_pool(verbose=getattr(KVCacheArgs, "verbose_pool", True))
        # 只第一次打印池信息，后续安静
        if not hasattr(KVOffloader, "_pool_announced"):
            print(f"[KVOffloader] DRAM pool: {self.pool.stats_str()}")
            KVOffloader._pool_announced = True

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
            
            
        mirror_on_push: bool = True
        self._prefetch_lock = threading.Lock()
        self._prefetch_map = {}  # key: (layer, tuple(blocks), bsz) -> {'evt': evt, 'k': [..], 'v': [..]}
        self.prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # 轻量 GPU 暂存（仅保存“当前窗口”的块，跨调用可复用）
        self.gpu_k = [[None for _ in range(self.n_blocks)] for _ in range(self.layers)]
        self.gpu_v = [[None for _ in range(self.n_blocks)] for _ in range(self.layers)]
        
        
        
        # --- KV H2D 就绪事件表（按 block）+ 打包→写盘后台队列 ---
        self._kv_ready_events: dict[tuple[int,int], torch.cuda.Event] = {}

        # 背景"打包→入写队列"线程：等 D2H 事件完成后在 CPU 拼包，再投递给 SSD writer
        self._pack_queue: "Queue[tuple[int,int,torch.cuda.Event|None]]" = Queue(
            maxsize=getattr(KVCacheArgs, "RAW_IO_QD_PACK", 64)
        )
        self._packer_stop = threading.Event()
        def _packer_loop():
            while not self._packer_stop.is_set():
                try:
                    L, B, d2h_evt = self._pack_queue.get(timeout=0.1)
                except Empty:
                    continue
                try:
                    if d2h_evt is not None:
                        # 等待本块 D2H 完成（只等这个事件，不同步整条流）
                        d2h_evt.synchronize()
                    kc = self.k_cpu[L][B]
                    vc = self.v_cpu[L][B]
                    if kc is None or vc is None:
                        continue  # 可能被逐出
                    kv_pack_cpu = torch.cat([kc, vc], dim=-1).contiguous()
                    # 可选：为后端 DMA 对齐，尽量置为 pinned
                    try:
                        kv_pack_cpu = kv_pack_cpu.pin_memory()
                    except Exception:
                        pass
                    # 投给 SSD 写线程（已存在的 _writer_loop）
                    try:
                        self._write_queue.put((L, B, kv_pack_cpu), timeout=0.5)
                    except Full:
                        print(f"[KV][WARN] write queue full, drop mirror @L{L} B{B}")
                finally:
                    self._pack_queue.task_done()
        # 启动打包线程
        self._packer_thread = threading.Thread(target=_packer_loop, name="kv_packer", daemon=True)
        self._packer_thread.start()

        
    # 公开给 WSM 的写暂停 API（例如 PCIE 忙或 pinned 低水位时调用）
    
    # 提供供 WSM 调用的“暂停写”接口
    def throttle_writes_for(self, ms: int):
        self._pause_write_until = max(self._pause_write_until, time.time() + ms / 1000.0)
    
    def _writer_loop(self):
        batch: list[tuple[int,int,Union[torch.Tensor,torch.cuda.Event]]] = []
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
            # 计算字节数时需要处理 Event 类型
            agg_bytes = 0
            for x in batch:
                if len(x) >= 3 and isinstance(x[2], torch.Tensor):
                    agg_bytes += x[2].numel() * x[2].element_size()
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
            for item in batch:
                layer, blk, token = item

                # 检查是否是 CUDA Event，如果是则等待完成后再打包
                if isinstance(token, torch.cuda.Event):
                    try:
                        token.synchronize()  # 背景线程等待，不阻塞主线程
                    except Exception as e:
                        print(f"[WARN] Event sync failed @L{layer} B{blk}: {e}")
                        continue
                    # 事件完成后，从 CPU pinned 内存打包 KV
                    if self.k_cpu[layer][blk] is None or self.v_cpu[layer][blk] is None:
                        print(f"[WARN] KV not available @L{layer} B{blk}, skip write")
                        continue
                    kv_cpu = torch.cat([self.k_cpu[layer][blk], self.v_cpu[layer][blk]], dim=-1).contiguous()
                else:
                    # 兼容旧路径：直接使用已打包的 tensor
                    kv_cpu = token

                nbytes = kv_cpu.numel() * kv_cpu.element_size()
                try:
                    # 优先使用回你已有的 write_async；如不可用可退回 write()
                    if hasattr(self.ssd, "write_async"):
                        self.ssd.write_async(layer, blk, kv_cpu, sync=False)
                    else:
                        self.ssd.write(layer, blk, kv_cpu)
                    self._win_bytes.append((time.time(), nbytes))
                    self._win_sum += nbytes
                except Exception as e:
                    print(f"[WARN] SSD write failed @L{layer} B{blk}: {e}")
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
        """
        Allocate KV cache block using pool management: (max_batch, heads, BLOCK, dim)
        使用池管理分配 KV cache 块：(max_batch, heads, BLOCK, dim)
        """
        if self.k_cpu[layer][blk] is None:
            shape = (self.max_batch, self.heads, BLOCK, self.dim)
            # 计算每个 K/V 块所需字节数 (float16 = 2 bytes)
            single_kv_bytes = self.max_batch * self.heads * BLOCK * self.dim * 2  # 2 for float16

            # 从全局池中分配 K 块
            k_buf = self.pool.alloc_block(single_kv_bytes)
            if k_buf is None:
                # 达到 DRAM 配额，回退到直接分配或报错
                print(f"[KVOffloader][WARN] DRAM pool exhausted, cannot allocate K block for L{layer} B{blk}")
                # 回退：直接分配（不受池管理）
                self.k_cpu[layer][blk] = torch.zeros(shape, dtype=torch.float16, pin_memory=True)
                self.v_cpu[layer][blk] = torch.zeros_like(self.k_cpu[layer][blk])
                return

            # 从全局池中分配 V 块
            v_buf = self.pool.alloc_block(single_kv_bytes)
            if v_buf is None:
                # V 分配失败，释放已分配的 K 块
                self.pool.free_block(k_buf)
                print(f"[KVOffloader][WARN] DRAM pool exhausted, cannot allocate V block for L{layer} B{blk}")
                # 回退：直接分配
                self.k_cpu[layer][blk] = torch.zeros(shape, dtype=torch.float16, pin_memory=True)
                self.v_cpu[layer][blk] = torch.zeros_like(self.k_cpu[layer][blk])
                return

            # 将 uint8 缓冲区 view 为 float16 并 reshape 到所需形状
            k_view = k_buf[:single_kv_bytes].view(torch.float16).reshape(shape)
            v_view = v_buf[:single_kv_bytes].view(torch.float16).reshape(shape)

            # 初始化为零
            k_view.zero_()
            v_view.zero_()

            self.k_cpu[layer][blk] = k_view
            self.v_cpu[layer][blk] = v_view

            # 保存底层 buffer 引用以便后续释放
            self._k_raw_buf[layer][blk] = k_buf
            self._v_raw_buf[layer][blk] = v_buf
    

    # ---------------- public API -----------------
    def push(self, layer: int, blk: int, k: torch.Tensor, v: torch.Tensor,
            token_idx: int, batch_idx: int = 0, **kwargs):
        assert k.dim() == 3 and v.dim() == 3, "KV must be (bsz, heads, dim)"
        bsz = k.size(0)
        t_in_blk = int(token_idx) % BLOCK

        self._alloc_block(layer, blk)

        # D2H：入 kv_d2h stream，非阻塞
        stream = self.d2h_stream or torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            self.k_cpu[layer][blk][:bsz, :, t_in_blk, :].copy_(k, non_blocking=True)
            self.v_cpu[layer][blk][:bsz, :, t_in_blk, :].copy_(v, non_blocking=True)

        # 记账：重要性极小正数
        self.importance[min(batch_idx, self.max_batch-1), layer, blk] = max(
            self.importance[min(batch_idx, self.max_batch-1), layer, blk], 1e-6
        )
        self.global_importance[layer, blk] = max(self.global_importance[layer, blk], 1e-6)

        if self.global_tracker:
            self.global_tracker.update_hbm_storage(batch_idx, layer, [blk], "remove")
            self.global_tracker.update_dram_storage(batch_idx, layer, [blk], "add", [1e-6])

        self._maybe_evict()

        # 可选镜像到 SSD：不再同步；改为记录 D2H 完成事件并排队后台打包
        if self.ssd is not None and getattr(KVCacheArgs, "mirror_on_push", True):
            d2h_evt = torch.cuda.Event(blocking=False)
            d2h_evt.record(stream)
            try:
                self._pack_queue.put_nowait((layer, blk, d2h_evt))
            except Full:
                # 丢弃镜像，不要阻塞前向线程
                if getattr(KVCacheArgs, "verbose_pool", False):
                    print(f"[KV][WARN] drop mirror (pack queue full) @L{layer} B{blk}")


        
    # kv_offload.py 内（类 KVOffloader）
    def eager_spill_layer(self, layer: int, upto_token: int, async_write: bool = True, include_partial: bool = True):
        """
        在 prefill 层尾调用：把 [0 .. upto_token) 覆盖到的块下放到 SSD，并释放对应 DRAM。
        async_write=True → 用 SSD.write_batch_async；False → 逐块 _spill_to_ssd 同步写。
        """
        end_blk = (int(upto_token) - 1) // BLOCK if upto_token > 0 else -1
        if end_blk < 0:
            return

        # 需处理的块：0..end_blk；include_partial=True 表示末尾半块也一并写
        blocks = list(range(0, end_blk + 1))

        if self.ssd is None or not async_write:
            # 最简单：同步逐块写 + 释放
            for B in blocks:
                if self.k_cpu[layer][B] is not None:
                    self._spill_to_ssd(layer, B)   # 已有方法：写→释放→on_ssd=True
            return

        # 异步批写：收集事件
        events, ev_blks = [], []
        for B in blocks:
            if self.k_cpu[layer][B] is None:
                continue
            evt = torch.cuda.Event(blocking=False)
            if self.d2h_stream is not None:
                evt.record(self.d2h_stream)
            events.append(evt)
            ev_blks.append(B)

        if not ev_blks:
            return

        # 后台线程：等事件→打包→批写
        def _spill_job():
            local_blks, local_tensors = [], []
            for e, B in zip(events, ev_blks):
                try:
                    e.synchronize()
                except Exception:
                    pass
                kv_pack = torch.cat([self.k_cpu[layer][B], self.v_cpu[layer][B]], dim=-1).contiguous()
                local_blks.append(B)
                local_tensors.append(kv_pack)
            if local_blks:
                fut = self.ssd.write_batch_async(layer, local_blks, local_tensors)
                def _on_done(_):
                    for BB in local_blks:
                        if self._k_raw_buf[layer][BB] is not None:
                            self.pool.free_block(self._k_raw_buf[layer][BB])
                            self._k_raw_buf[layer][BB] = None
                        if self._v_raw_buf[layer][BB] is not None:
                            self.pool.free_block(self._v_raw_buf[layer][BB])
                            self._v_raw_buf[layer][BB] = None
                        self.k_cpu[layer][BB] = self.v_cpu[layer][BB] = None
                        self.on_ssd[layer][BB] = True
                try:
                    fut.add_done_callback(_on_done)
                except Exception:
                    fut.result(timeout=None)
                    _on_done(None)
        threading.Thread(target=_spill_job, daemon=True).start()

        # 在 KVOffloader 类中追加
    def prefetch_for_next_layer(self, *, current_layer: int, start_pos: int, seqlen: int,
                                bsz: int, window_tokens: int = BLOCK):
        """
        为“下一层”的解码提前拉起最近窗口的历史 KV（SSD->DRAM->HBM），纯异步。
        """
        if self.layers <= 0:
            return
        nxt = min(int(current_layer) + 1, int(self.layers) - 1)
        # 下一层会访问到的“尾窗”块（以当前已写入的末尾 token 为界）
        end_pos = int(start_pos) + int(seqlen)  # 已完成到的最后 token 的下一位置
        blocks = self.plan_tail_window_blocks(end_pos - 1, 1, window_tokens=window_tokens)
        if blocks:
            self.prefetch_async(layer=nxt, blocks=blocks, bsz=int(bsz), device=self.device)


    def ensure_on_gpu(self, layer_idx: int, start_pos: int, k_cur: torch.Tensor, v_cur: torch.Tensor):
        """
        确保本层在计算时用于注意力的 K/V 都在 CUDA。
        Ensure that K/V tensors for attention computation are on CUDA.

        对于 prefill：通常只用 k_cur/v_cur；
        对于 decode：需要把历史 KV 拉上来组成 k_full/v_full。

        For prefill: typically only uses k_cur/v_cur;
        For decode: needs to fetch historical KV to form k_full/v_full.

        Args:
            layer_idx: layer index
            start_pos: current start position in sequence
            k_cur: current K tensor from projection (B, seq_len, n_kv_heads, head_dim)
            v_cur: current V tensor from projection (B, seq_len, n_kv_heads, head_dim)

        Returns:
            k_full, v_full: Complete K/V tensors on CUDA, ready for attention computation
        """
        # 确保输入在 CUDA（防止 q 在 CUDA、k_cur/v_cur 在 CPU 的 bmm 报错）
        # Ensure inputs are on CUDA (prevent bmm error when q is on CUDA but k_cur/v_cur on CPU)
        target_device = k_cur.device
        if not str(target_device).startswith("cuda"):
            raise RuntimeError(f"[KVOffloader] ensure_on_gpu: k_cur is on {target_device}, but only CUDA is supported")

        # 对于 prefill（start_pos == 0）或单 token decode，直接返回当前 K/V
        # For prefill (start_pos == 0) or single token decode, directly return current K/V
        if start_pos == 0:
            # Prefill: no historical KV, just use current
            return k_cur, v_cur

        # 对于 decode：需要组装历史 KV + 当前 KV
        # For decode: need to assemble historical KV + current KV
        # 这里的实现取决于你的 KV cache 架构
        # Implementation depends on your KV cache architecture

        # 如果你已经有类似 _gather_kv 或 fetch 的逻辑，可以复用
        # If you already have logic like _gather_kv or fetch, reuse it
        # 这里提供一个简化版本：直接返回当前 K/V（适用于 streaming decode）
        # Here's a simplified version: directly return current K/V (for streaming decode)

        # 注意：这个简化版本假设你在 SelfAttention.forward() 中已经通过
        # offloader.fetch() 获取了完整的 k_full/v_full
        # Note: This simplified version assumes you've already fetched complete k_full/v_full
        # via offloader.fetch() in SelfAttention.forward()

        return k_cur, v_cur

   
    def fetch(self, layer: int, blocks: torch.Tensor, batch_idx: int = 0, bsz: int | None = None):
        uniq = blocks.to(torch.long).unique(sorted=True).tolist()
        try:
            self.wait_blocks_ready(layer, uniq)
        except Exception:
            pass
        use_bsz = int(bsz) if bsz is not None else self.max_batch

        # 先看是否有“整组命中”的预取记录（有事件）
        key = (int(layer), tuple(uniq), int(use_bsz))
        rec = None
        with self._prefetch_lock:
            rec = self._prefetch_map.pop(key, None)

        if rec is not None:
            # 等待预取组的 H2D 完成
            torch.cuda.current_stream().wait_event(rec["evt"])
            k_full = torch.cat(rec["k"], dim=2)  # dim=2 为 token 维（每块 BLOCK）
            v_full = torch.cat(rec["v"], dim=2)
            return k_full, v_full

        # 部分命中：逐块若有 GPU 暂存则直接用；否则补齐 SSD->DRAM + H2D
        k_parts, v_parts = [], []

        # 1) 先补齐还在 SSD 的块到 DRAM
        need_load = [b for b in uniq if self.on_ssd[layer][b]]
        for b in need_load:
            self._load_from_ssd(layer, b)

        # 2) H2D（优先复用 GPU 暂存；缺失才 .to()）
        stream = self.h2d_stream or torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            for b in uniq:
                kg = self.gpu_k[layer][b]
                vg = self.gpu_v[layer][b]
                if kg is not None and vg is not None and kg.size(0) >= use_bsz:
                    k_parts.append(kg[:use_bsz])
                    v_parts.append(vg[:use_bsz])
                else:
                    kc = self.k_cpu[layer][b][:use_bsz]
                    vc = self.v_cpu[layer][b][:use_bsz]
                    k_parts.append(kc.to(self.device, non_blocking=True))
                    v_parts.append(vc.to(self.device, non_blocking=True))

        if stream is not torch.cuda.current_stream():
            torch.cuda.current_stream().wait_stream(stream)

        k_full = torch.cat(k_parts, dim=2)
        v_full = torch.cat(v_parts, dim=2)
        if k_full.device.type != "cuda":
            k_full = k_full.to(self.device, non_blocking=True)
        if v_full.device.type != "cuda":
            v_full = v_full.to(self.device, non_blocking=True)
        return k_full, v_full

    # 新增：根据“最近 N token”计算需要的 block（默认 N=BLOCK=256）
    def plan_tail_window_blocks(self, start_pos: int, seqlen: int, window_tokens: int = BLOCK) -> list[int]:
        end = int(start_pos) + int(seqlen) - 1
        if end < 0:
            return []
        left = max(0, end - int(window_tokens) + 1)
        blk_lo = left // BLOCK
        blk_hi = end  // BLOCK
        return list(range(blk_lo, blk_hi + 1))

    # 新增：异步预取（SSD->DRAM + DRAM->HBM）并记录事件
    def prefetch_async(self, *, layer: int, blocks: list[int], bsz: int,
                    device: str | torch.device = None):
        if not blocks:
            return
        uniq = sorted(set(int(b) for b in blocks))
        use_bsz = int(bsz) if bsz is not None else self.max_batch
        dev = str(device or self.device)

        # 1) SSD->DRAM 走后台线程；完成后在同线程里继续排 H2D
        def _task():
            # 1.1 拉回仍在 SSD 的块到 DRAM
            need = [b for b in uniq if self.on_ssd[layer][b]]
            for b in need:
                try:
                    self._load_from_ssd(layer, b)
                except Exception as e:
                    print(f"[KV][WARN] SSD load L{layer} B{b} failed: {e}")

            # 2) DRAM->HBM：在 kv_h2d stream 上排队；为整组与逐块都记录 ready 事件
            stream = self.h2d_stream or torch.cuda.current_stream()
            k_list, v_list = [], []
            with torch.cuda.stream(stream):
                for b in uniq:
                    kc = self.k_cpu[layer][b]
                    vc = self.v_cpu[layer][b]
                    if kc is None or vc is None:
                        raise RuntimeError(f"[KV] block {b} not present in DRAM for layer {layer}")
                    shape = (use_bsz, self.heads, BLOCK, self.dim)
                    kg = self.gpu_k[layer][b]
                    vg = self.gpu_v[layer][b]
                    if (kg is None) or (kg.device.type != "cuda") or (tuple(kg.shape) != shape):
                        kg = torch.empty(shape, dtype=kc.dtype, device=self.device)
                        vg = torch.empty(shape, dtype=vc.dtype, device=self.device)
                        self.gpu_k[layer][b] = kg
                        self.gpu_v[layer][b] = vg
                    kg.copy_(kc[:use_bsz], non_blocking=True)
                    vg.copy_(vc[:use_bsz], non_blocking=True)
                    k_list.append(kg); v_list.append(vg)

            evt = torch.cuda.Event(blocking=False)
            evt.record(stream)

            # 组级就绪：供 fast-path 直接拼接
            with self._prefetch_lock:
                self._prefetch_map[(int(layer), tuple(uniq), int(use_bsz))] = {
                    "evt": evt, "k": k_list, "v": v_list
                }
            # 块级就绪：供任意 fetch/wait 安全等待
            for b in uniq:
                self._kv_ready_events[(int(layer), int(b))] = evt

        # 真正异步：不再等待 future；只投递任务
        try:
            self.prefetch_executor.submit(_task)
        except Exception as e:
            print(f"[KV][WARN] prefetch submit failed: {e}")
            
    def prefetch_blocks_async(
        self,
        layer_idx: int,
        blocks: "list[int]",
        stream: "torch.cuda.Stream | None" = None,
        bsz: "int | None" = None,
        device: "str | torch.device | None" = None,
    ):
        """
        触发给定 block 集合的异步预取（SSD/DRAM→GPU），并在 KV H2D 流上记录
        block 级 CUDA 事件；供 compute 流通过 wait_blocks_ready() 做非阻塞依赖。

        Args:
            layer_idx: 层索引
            blocks: 需要预取的 block 索引列表
            stream: 可选的 CUDA 流（优先使用；否则用 offloader 的 kv_h2d）
            bsz: 批大小（用于分配 GPU 缓冲区）
            device: 目标设备（默认为 self.device）
        """
        if not blocks:
            return
        if not torch.cuda.is_available():
            return

        # 选流：优先调用方传入；否则用 offloader 的 kv_h2d（如有）
        s = stream or getattr(self.streams, "kv_h2d", None) if hasattr(self, 'streams') else None
        if s is None:
            # 退化到当前流也可工作，但建议总是传入 kv_h2d
            s = torch.cuda.current_stream()

        # 兼容已有 prefetch_async()（它内部会创建/记录 block 事件到 self._kv_ready_events）
        try:
            effective_bsz = bsz if bsz is not None else getattr(self, "max_batch", 1)
            effective_dev = device if device is not None else getattr(self, "device", "cuda")
            # 进入 KV H2D 流上下文，让后续记录的事件与数据入队在同一条流
            with torch.cuda.stream(s):
                self.prefetch_async(layer=layer_idx, blocks=blocks, bsz=effective_bsz, device=effective_dev)
        except AttributeError:
            # 没有 prefetch_async 的老实现：可在这里补上你自己的 SSD/DRAM→GPU copy，并手动记录事件
            for bid in blocks:
                e = torch.cuda.Event(blocking=False)
                e.record(s)
                self._kv_ready_events[(int(layer_idx), int(bid))] = e

    def wait_blocks_ready(self, layer: int, blocks: list[int], stream: "torch.cuda.Stream|None" = None):
        """在给定 stream 上等待若干块的 H2D ready 事件（若存在则等待；没有则直接返回）。"""
        s = stream or torch.cuda.current_stream()
        for b in set(int(x) for x in blocks):
            evt = self._kv_ready_events.get((int(layer), b))
            if evt is not None:
                try:
                    s.wait_event(evt)
                except Exception:
                    # 容错：若事件已被 GC 或已完成，忽略即可
                    pass

    # 新增：供上层简单调用——在 L 的 MHA 完成后，FFN(L) 期间预取 L+1
    def prefetch_for_next_layer(self, *, current_layer: int, start_pos: int, seqlen: int, bsz: int, window_tokens: int = BLOCK):
        nxt = int(current_layer) + 1
        if nxt >= self.layers:
            return
        blocks = self.plan_tail_window_blocks(start_pos, seqlen, window_tokens)
        if blocks:
            self.prefetch_async(layer=nxt, blocks=blocks, bsz=bsz, device=self.device)


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

            # 释放底层 buffer 回全局池
            if self._k_raw_buf[L][B] is not None:
                self.pool.free_block(self._k_raw_buf[L][B])
                self._k_raw_buf[L][B] = None
            if self._v_raw_buf[L][B] is not None:
                self.pool.free_block(self._v_raw_buf[L][B])
                self._v_raw_buf[L][B] = None

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

        # 释放底层 buffer 回全局池
        if self._k_raw_buf[L][B] is not None:
            self.pool.free_block(self._k_raw_buf[L][B])
            self._k_raw_buf[L][B] = None
        if self._v_raw_buf[L][B] is not None:
            self.pool.free_block(self._v_raw_buf[L][B])
            self._v_raw_buf[L][B] = None

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
            self._ssd_buffer = torch.empty(shape, dtype=torch.float16, device=self.device)

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
