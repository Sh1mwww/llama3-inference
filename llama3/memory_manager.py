"""
HBM内存限制和管理模块
提供GPU显存使用限制、监控和智能分配功能
"""

import torch
import threading
import time
import logging
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
from dataclasses import dataclass

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