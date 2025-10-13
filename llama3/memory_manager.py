"""
HBM内存限制和管理模块
提供GPU显存使用限制、监控和智能分配功能
"""

import math
import torch
import threading
import time
import logging
from typing import Optional, Dict, Any, Callable, List, Tuple
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """内存配置"""
    max_hbm_gb: float = 14.0         
    reserved_hbm_gb: float = 1.0     
    enable_monitoring: bool = True     
    cleanup_threshold: float = 0.9     
    oom_retry_count: int = 3          
    monitor_interval: float = 1.0    


class HBMMemoryManager:
    """HBM内存管理器"""
    
    def __init__(self, config: MemoryConfig, device: str = "cuda:0"):
        self.config = config
        self.device = device
        self.device_id = int(device.split(":")[1]) if ":" in device else 0
        
        # 检查设备可用性
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
            
        if self.device_id >= torch.cuda.device_count():
            raise RuntimeError(f"Device {device} not found")
        
        # 获取设备总内存
        self.total_memory = torch.cuda.get_device_properties(self.device_id).total_memory
        self.max_memory_bytes = int(self.config.max_hbm_gb * 1024**3)
        self.reserved_bytes = int(self.config.reserved_hbm_gb * 1024**3)
        
        # 验证配置
        if self.max_memory_bytes > self.total_memory:
            logger.warning(f"Requested {self.config.max_hbm_gb}GB > total {self.total_memory/(1024**3):.1f}GB")
            self.max_memory_bytes = self.total_memory - self.reserved_bytes
            
        self.available_bytes = self.max_memory_bytes - self.reserved_bytes
        
        # 状态跟踪
        self.allocated_tensors: Dict[int, Dict[str, Any]] = {}  # tensor_id -> info
        self.allocation_lock = threading.Lock()
        self.current_allocated = 0
        self.peak_allocated = 0
        
        # 监控线程
        self.monitoring = False
        self.monitor_thread = None
        
        logger.info(f"HBM Manager initialized: {self.available_bytes/(1024**3):.2f}GB available")
        
        if config.enable_monitoring:
            self.start_monitoring()
    
    def start_monitoring(self):
        """启动内存监控"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._memory_monitor_loop,
            daemon=True,
            name="hbm_monitor"
        )
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """停止内存监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _memory_monitor_loop(self):
        """内存监控循环"""
        while self.monitoring:
            try:
                self._check_memory_status()
                time.sleep(self.config.monitor_interval)
            except Exception as e:
                logger.error(f"Memory monitor error: {e}")
                time.sleep(self.config.monitor_interval)
    
    def _check_memory_status(self):
        """检查内存状态"""
        if not torch.cuda.is_available():
            return
            
        # 避免频繁同步，使用缓存的内存统计
        if not hasattr(self, '_last_sync_time'):
            self._last_sync_time = 0
        
        current_time = time.time()
        # 每秒最多同步一次
        if current_time - self._last_sync_time > 1.0:
            torch.cuda.synchronize(self.device_id)
            self._last_sync_time = current_time
            
        memory_stats = torch.cuda.memory_stats(self.device_id)
        
        allocated = memory_stats.get("allocated_bytes.all.current", 0)
        
        usage_ratio = allocated / self.max_memory_bytes
        
        # 更新统计
        with self.allocation_lock:
            self.current_allocated = allocated
            self.peak_allocated = max(self.peak_allocated, allocated)
        
        # 检查是否需要清理
        if usage_ratio > self.config.cleanup_threshold:
            logger.warning(f"High memory usage: {usage_ratio:.1%}, triggering cleanup")
            self._emergency_cleanup()
    
    def _emergency_cleanup(self):
        """紧急内存清理"""
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device_id)
            logger.info("Emergency memory cleanup completed")
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        torch.cuda.synchronize(self.device_id)
        memory_stats = torch.cuda.memory_stats(self.device_id)
        
        allocated = memory_stats.get("allocated_bytes.all.current", 0)
        reserved = memory_stats.get("reserved_bytes.all.current", 0)
        
        return {
            "total_gb": self.total_memory / (1024**3),
            "max_allowed_gb": self.max_memory_bytes / (1024**3),
            "available_gb": self.available_bytes / (1024**3),
            "current_allocated_gb": allocated / (1024**3),
            "reserved_gb": reserved / (1024**3),
            "usage_ratio": allocated / self.max_memory_bytes,
            "peak_allocated_gb": self.peak_allocated / (1024**3),
            "can_allocate_gb": max(0, (self.max_memory_bytes - allocated)) / (1024**3)
        }
    
    def can_allocate(self, size_bytes: int) -> bool:
        """检查是否可以分配指定大小的内存"""
        if not torch.cuda.is_available():
            return False
        
        try:
            torch.cuda.synchronize(self.device_id)
            memory_stats = torch.cuda.memory_stats(self.device_id)
            allocated = memory_stats.get("allocated_bytes.all.current", 0)
            
            return (allocated + size_bytes) <= self.max_memory_bytes
        except Exception:
            return False
    
    def estimate_tensor_size(self, shape: tuple, dtype: torch.dtype = torch.float32) -> int:
        """估算张量内存大小"""
        element_size = torch.tensor([], dtype=dtype).element_size()
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        return total_elements * element_size
    
    @contextmanager
    def allocate_tensor(self, shape: tuple, dtype: torch.dtype = torch.float32, **kwargs):
        """安全分配张量"""
        size_bytes = self.estimate_tensor_size(shape, dtype)
        
        if not self.can_allocate(size_bytes):
            # 尝试清理内存
            self._emergency_cleanup()
            
            if not self.can_allocate(size_bytes):
                raise RuntimeError(
                    f"Cannot allocate {size_bytes/(1024**3):.2f}GB, "
                    f"would exceed limit {self.max_memory_bytes/(1024**3):.2f}GB"
                )
        
        tensor = None
        tensor_id = None
        
        try:
            # 分配张量
            tensor = torch.empty(shape, dtype=dtype, device=self.device, **kwargs)
            tensor_id = id(tensor)
            
            # 记录分配
            with self.allocation_lock:
                self.allocated_tensors[tensor_id] = {
                    "shape": shape,
                    "dtype": dtype,
                    "size_bytes": size_bytes,
                    "allocated_at": time.time()
                }
            
            yield tensor
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM during tensor allocation: {e}")
            self._emergency_cleanup()
            raise RuntimeError(f"GPU OOM: failed to allocate {size_bytes/(1024**3):.2f}GB") from e
            
        finally:
            # 清理记录
            if tensor_id and tensor_id in self.allocated_tensors:
                with self.allocation_lock:
                    del self.allocated_tensors[tensor_id]
            
            # 删除张量
            if tensor is not None:
                del tensor
    
    @contextmanager
    def memory_limit_context(self, temp_limit_gb: Optional[float] = None):
        """临时内存限制上下文"""
        if temp_limit_gb is None:
            yield
            return
        
        # 保存原始限制
        original_limit = self.max_memory_bytes
        original_available = self.available_bytes
        
        try:
            # 设置临时限制
            temp_limit_bytes = int(temp_limit_gb * 1024**3)
            self.max_memory_bytes = min(temp_limit_bytes, original_limit)
            self.available_bytes = self.max_memory_bytes - self.reserved_bytes
            
            logger.info(f"Temporary memory limit: {temp_limit_gb}GB")
            yield
            
        finally:
            # 恢复原始限制
            self.max_memory_bytes = original_limit
            self.available_bytes = original_available


class GlobalMemoryManager:
    """全局内存管理器单例"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config: Optional[MemoryConfig] = None, device: str = "cuda:0"):
        with cls._lock:
            if cls._instance is None:
                if config is None:
                    config = MemoryConfig()
                cls._instance = HBMMemoryManager(config, device)
            return cls._instance
    
    @classmethod
    def get_instance(cls) -> Optional[HBMMemoryManager]:
        """获取全局实例"""
        return cls._instance
    
    @classmethod
    def reset(cls):
        """重置全局实例"""
        with cls._lock:
            if cls._instance:
                cls._instance.stop_monitoring()
            cls._instance = None


def safe_tensor_allocation(shape: tuple, dtype: torch.dtype = torch.float32, 
                         device: str = "cuda:0", **kwargs):
    """安全的张量分配装饰器工厂"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **func_kwargs):
            manager = GlobalMemoryManager.get_instance()
            if manager is None:
                # 如果没有全局管理器，直接分配
                return func(*args, **func_kwargs)
            
            with manager.allocate_tensor(shape, dtype, **kwargs) as tensor:
                return func(tensor, *args, **func_kwargs)
        return wrapper
    return decorator


@contextmanager
def hbm_memory_limit(limit_gb: float, device: str = "cuda:0", 
                    reserved_gb: float = 1.0, enable_monitoring: bool = True):
    """HBM内存限制上下文管理器"""
    config = MemoryConfig(
        max_hbm_gb=limit_gb,
        reserved_hbm_gb=reserved_gb,
        enable_monitoring=enable_monitoring
    )
    
    manager = HBMMemoryManager(config, device)
    
    try:
        yield manager
    finally:
        manager.stop_monitoring()


def set_global_memory_limit(limit_gb: float = None, device: str = "cuda:0",
                          reserved_gb: float = None, config: MemoryConfig = None):
    """设置全局内存限制"""
    if config is not None:
        # 直接使用传入的 MemoryConfig
        memory_config = config
    else:
        # 向后兼容：从参数构建 MemoryConfig
        memory_config = MemoryConfig(
            max_hbm_gb=limit_gb or 12.0,
            reserved_hbm_gb=reserved_gb or 1.0,
            enable_monitoring=True
        )

    # 重置并创建新的全局管理器
    GlobalMemoryManager.reset()
    GlobalMemoryManager(memory_config, device)
    
    logger.info(f"Global memory limit set to {memory_config.max_hbm_gb}GB on {device}")


def get_memory_info(device: str = "cuda:0") -> Dict[str, Any]:
    """获取内存信息"""
    manager = GlobalMemoryManager.get_instance()
    if manager:
        return manager.get_memory_info()
    else:
        # 返回基本信息
        if torch.cuda.is_available():
            device_id = int(device.split(":")[1]) if ":" in device else 0
            props = torch.cuda.get_device_properties(device_id)
            memory_stats = torch.cuda.memory_stats(device_id)
            allocated = memory_stats.get("allocated_bytes.all.current", 0)
            
            return {
                "total_gb": props.total_memory / (1024**3),
                "current_allocated_gb": allocated / (1024**3),
                "no_limit": True
            }
        else:
            return {"error": "CUDA not available"}
        
# ===============================
# Host Pinned Extent Pool 
# ===============================   

    
@dataclass
class PinnedPoolParans:
    """来自 RUNTIME.pinned 与 RUNTIME.io 的最小必要参数"""
    WEIGHT_PINNED_BYTES: int         # 例如 32<<30
    EXTENT_BYTES: int                # 例如 2<<20
    PINNED_REGISTER_CHUNK: int       # 例如 512<<20
    PINNED_REGISTER_N: int           # 若传入 None，外部应已换算为 ceil
    RAW_IO_ALIGN_BYTES: int          # 例如 4096
    name: str = "weights"            # 标识用途
    
@dataclass
class ExtentMeta:
    """extent 元数据（供策略/驱逐/监控使用）"""
    chunk_id: int
    offset: int                      # 相对于 chunk 张量起点的 byte 偏移（已 4KiB 对齐）
    bytes: int                       # extent 大小，通常=EXTENT_BYTES
    refcnt: int = 0
    inflight_read: bool = False
    inflight_write: bool = False
    prefetched: bool = False
    consumed: bool = False
    layer: int = -1
    group: str = ""                  # "mha" / "ffn" / "kv" / ...
    prefetch_ts_ms: int = 0
    
class HostPinnedExtentPool:
    """
    主机锁页内存池，按 extent 分配与管理。
    - 大块注册 + extent 子分配 的 Host pinned 池
    - 预先分配 N 个大 pinned 张量（每个 ~PINNED_REGISTER_CHUNK）
    - 计算每个张量的 4KiB 对齐起点（head_pad），只在对齐后的可用区间内切 EXTENT_BYTES 的 extent
    - 提供 extent 级分配/释放/引用计数/在飞保护
    - 暴露 chunk 与 extent 的张量视图，供 I/O 层注册/读写
    """
    
    def __init__(self, params: PinnedPoolParans):
        self.params = params
        self.lock = threading.Lock()
        
        # compute block nums
        total = self.params.WEIGHT_PINNED_BYTES
        chunk = self.params.PINNED_REGISTER_CHUNK
        n_chunks = self.params.PINNED_REGISTER_N or math.ceil(total / chunk)
        self._chunks: List[torch.Tensor] = []
        self._chunk_head_pad: List[int] = []   # 每块对齐到 RAW_IO_ALIGN_BYTES 的头部偏移
        self._chunk_usable_bytes: List[int] = []
        
        # ---- 分配大块 pinned 张量（+对齐余量）----
        align = self.params.RAW_IO_ALIGN_BYTES
        if align <= 0 or (align & (align - 1)) != 0:
            raise ValueError(f"RAW_IO_ALIGN_BYTES must be positive power of two, got {align}")
        
        # 为了保证可对齐切片，额外 +align 字节的缓冲，避免头部浪费不足
        alloc_bytes = chunk + align
        for i in range(n_chunks):
            t = torch.empty(alloc_bytes, dtype =torch.uint8, pin_memory=True)
            base = t.data_ptr()
            head_pad = (align - (base & (align - 1))) & align
            usable = chunk # 真正用于注册/切 extent 的长度
            self._chunks.append(t)
            self._chunk_head_pad.append(head_pad)
            self._chunk_usable_bytes.append(usable)
            
        # ---- extent 元数据管理 ----
        extent = self.params.EXTENT_BYTES
        if extent % align != 0:
            raise ValueError(f"EXTENT_BYTES({extent}) 必须是 RAW_IO_ALIGN_BYTES({align}) 的整数倍")
        self._extent: List[ExtentMeta] = []
        self._free_eids: List[int] = []            # 空闲 extent 索引列表
        eids = 0
        for cid in range(n_chunks):
            head = self._chunk_head_pad[cid]
            usable = self._chunk_usable_bytes[cid]
            n_extents = usable // extent
            for i in range(n_extents):
                off = head + i * extent
                em = ExtentMeta(chunk_id=cid, offset=off, bytes=extent)
                self._extent.append(em)
                self._free_eids.append(eids)
                eids += 1
                
        self._total_extents = len(self._extents)
        self._free_list_watermark = self._total_extents  # 便于监控最低水位
            
            
    # ---------- 公开信息 ----------
    @property
    def total_bytes(self) -> int:
        return self._total_extents * self.params.EXTENT_BYTES
    
    @property
    def free_extents(self) -> int:
        with self.lock:
            return len(self._free_eids)
        
    @property
    def free_ratio(self) -> float:
        return self.free_extents / max(1, self._total_extents)
    
    def stats(self) -> Dict[str, Any]:
        with self.lock:
            used = self._total_extents - len(self._free_eids)
            self._free_list_watermark = min(self._free_list_watermark, len(self._free_eids))
            return {
                "name": self.params.name,
                "chunks": len(self._chunks),
                "extent_bytes": self.params.EXTENT_BYTES,
                "total_extents": self._total_extents,
                "used_extents": used,
                "free_extents": len(self._free_eids),
                "free_ratio": len(self._free_eids) / max(1, self._total_extents),
                "min_free_extents": self._free_list_watermark,
                "align_bytes": self.params.RAW_IO_ALIGN_BYTES,
            }         
            
    # ---------- 分配/引用/释放 ----------
    def allocate_extent(self) -> int:   
        """获取一个空闲 extent 的 EID （refcnt=1）"""
        with self.lock:
            if not self._free_eids:
                raise RuntimeError("PinnedExtentPool: out of pinned extents")     
            eid = self._free_eids.pop()
            meta = self._extent[eid]
            meta.refcnt = 1
            return eid
        
    def allocate_extents(self, n: int) -> List[int]:
        eids = []
        for _ in range(n):
            eids.append(self.allocate_extent())
        return eids
    
    def acquire(self, eid:int):
        with self.lock:
            meta = self._extent[eid]
            if meta.refcnt <= 0:
                raise RuntimeError(f"acquire extent {eid} with refcnt={meta.refcnt}")
            meta.refcnt += 1
            
    def release(self, eid: int):
        with self.lock:
            m = self._extents[eid]
            if m.refcnt <= 0:
                raise RuntimeError(f"release on zero-ref extent {eid}")
            m.refcnt -= 1
            if m.refcnt == 0 and not (m.inflight_read or m.inflight_write):
                # 清理元数据（保留 layer/group 供调试可选）
                m.prefetched = False
                m.consumed = False
                m.layer = -1
                m.group = ""
                m.prefetch_ts_ms = 0
                self._free_eids.append(eid)        
    
    def set_inflight(self, eid:int, read:bool=False, write:bool=False):
        with self.lock:
            m = self._extent[eid]
            if read:
                if m.inflight_read:
                    raise RuntimeError(f"extent {eid} already inflight read")
                m.inflight_read = bool(read)
            if write:
                if m.inflight_write:
                    raise RuntimeError(f"extent {eid} already inflight write")
                m.inflight_write = bool(write)
    
    def defer_release(self, eid:int, evt: "torch.cuda.Event"):
        """
        事件完成后再 release（避免 UAF）。调用方可在合适时机轮询 evt.query() 并触发释放；
        这里提供一个最小工具函数：若 evt 已完成则直接 release，否则由上层统一轮询释放。
        """
        if evt is None or evt.query():
            self.release(eid)
        else:
            # 由上层事件回收器统一处理；这里不做线程/轮询
            pass        
        
        
    # ---------- 视图/注册 ----------
    def _chunk_and_offset(self, eid:int) -> Tuple[int,int]:
        """返回 (chunk_id, offset)"""
        if eid < 0 or eid >= self._total_extents:
            raise ValueError(f"invalid extent id {eid}")
        m = self._extent[eid]
        return m.chunk_id, m.offset
    
    def extent_tensor(self, eid:int) -> torch.tensor:
        """
        返回 uint8 pinned tensor 视图（长度=EXTENT_BYTES，4KiB 对齐）
        """
        cid, off = self._chunk_and_offset(eid)
        return cid.narrow(0, off, self._extent[eid].bytes)    
        
        
        