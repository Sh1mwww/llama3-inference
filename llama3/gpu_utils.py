"""
GPU错误处理和内存管理工具类
提供GPU状态检查、错误恢复、内存管理等功能
"""
import torch
import psutil
import time
import logging
from typing import Optional, Callable, Any, Dict
from contextlib import contextmanager
from functools import wraps

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUMemoryError(Exception):
    """GPU内存相关错误"""
    pass


class GPUDeviceError(Exception):
    """GPU设备相关错误"""
    pass


class GPUHealthMonitor:
    """GPU健康状态监控器"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.gpu_available else 0
        self.min_free_memory_gb = 1.0  # 最小保留显存(GB)
        
    def check_gpu_health(self, device_id: int = 0) -> Dict[str, Any]:
        """检查GPU健康状态"""
        if not self.gpu_available:
            return {
                "status": "unavailable",
                "message": "CUDA not available"
            }
        
        try:
            # 检查设备有效性
            if device_id >= self.device_count:
                return {
                    "status": "invalid_device",
                    "message": f"Device {device_id} not found. Available: {self.device_count}"
                }
            
            # 检查内存状态
            memory_info = self.get_memory_info(device_id)
            free_gb = memory_info["free"] / (1024**3)
            
            if free_gb < self.min_free_memory_gb:
                return {
                    "status": "low_memory",
                    "message": f"Low GPU memory: {free_gb:.2f}GB free",
                    "memory_info": memory_info
                }
            
            # 检查GPU温度和利用率(如果支持)
            try:
                torch.cuda.synchronize(device_id)
                # 简单的GPU功能测试
                test_tensor = torch.randn(100, 100, device=device_id)
                _ = test_tensor @ test_tensor.T
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                return {
                    "status": "gpu_error",
                    "message": f"GPU function test failed: {str(e)}"
                }
            
            return {
                "status": "healthy",
                "message": "GPU is healthy",
                "memory_info": memory_info
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Health check failed: {str(e)}"
            }
    
    def get_memory_info(self, device_id: int = 0) -> Dict[str, int]:
        """获取GPU内存信息"""
        if not self.gpu_available:
            return {"total": 0, "used": 0, "free": 0}
        
        torch.cuda.synchronize(device_id)
        memory_stats = torch.cuda.memory_stats(device_id)
        
        allocated = memory_stats.get("allocated_bytes.all.current", 0)
        reserved = memory_stats.get("reserved_bytes.all.current", 0)
        
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        
        return {
            "total": total_memory,
            "allocated": allocated,
            "reserved": reserved,
            "free": total_memory - reserved
        }


class SafeGPUManager:
    """安全的GPU操作管理器"""
    
    def __init__(self, device: str = "cuda", auto_fallback: bool = True):
        self.monitor = GPUHealthMonitor()
        self.auto_fallback = auto_fallback
        self.original_device = device
        self.current_device = self._validate_device(device)
        
    def _validate_device(self, device: str) -> str:
        """验证并返回可用的设备"""
        if not device.startswith("cuda"):
            return device
            
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return "cpu" if self.auto_fallback else device
        
        # 提取设备ID
        try:
            if ":" in device:
                device_id = int(device.split(":")[1])
            else:
                device_id = 0
        except (ValueError, IndexError):
            device_id = 0
        
        health = self.monitor.check_gpu_health(device_id)
        
        if health["status"] == "healthy":
            return device
        elif health["status"] == "low_memory":
            logger.warning(f"GPU memory low: {health['message']}")
            self._try_cleanup_memory(device_id)
            return device
        else:
            logger.error(f"GPU health check failed: {health['message']}")
            if self.auto_fallback:
                logger.info("Falling back to CPU")
                return "cpu"
            else:
                raise GPUDeviceError(f"GPU device unusable: {health['message']}")
    
    def _try_cleanup_memory(self, device_id: int):
        """尝试清理GPU内存"""
        try:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'reset_max_memory_allocated'):
                torch.cuda.reset_max_memory_allocated(device_id)
            logger.info(f"GPU memory cleanup completed for device {device_id}")
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
    
    @contextmanager
    def safe_cuda_context(self, device_id: int = 0):
        """安全的CUDA上下文管理器"""
        if not self.current_device.startswith("cuda"):
            yield self.current_device
            return
        
        try:
            # 检查内存状态
            memory_before = self.monitor.get_memory_info(device_id)
            yield self.current_device
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM Error: {e}")
            self._try_cleanup_memory(device_id)
            raise GPUMemoryError(f"GPU out of memory: {e}")
            
        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.error(f"CUDA Runtime Error: {e}")
                if self.auto_fallback:
                    logger.info("Attempting fallback to CPU")
                    self.current_device = "cpu"
                raise GPUDeviceError(f"CUDA runtime error: {e}")
            else:
                raise
                
        finally:
            # 可选的内存状态检查
            if self.current_device.startswith("cuda"):
                try:
                    memory_after = self.monitor.get_memory_info(device_id)
                    memory_diff = memory_after["allocated"] - memory_before["allocated"]
                    if memory_diff > 100 * 1024 * 1024:  # 超过100MB增长
                        logger.info(f"Memory usage increased by {memory_diff / 1024**2:.1f}MB")
                except Exception:
                    pass


def gpu_safe_operation(retry_count: int = 2, cleanup_on_error: bool = True):
    """GPU安全操作装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retry_count + 1):
                try:
                    return func(*args, **kwargs)
                except torch.cuda.OutOfMemoryError as e:
                    logger.warning(f"GPU OOM on attempt {attempt + 1}: {e}")
                    if cleanup_on_error:
                        torch.cuda.empty_cache()
                    if attempt == retry_count:
                        raise GPUMemoryError(f"GPU OOM after {retry_count + 1} attempts: {e}")
                    time.sleep(0.1)  # 短暂延迟
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        logger.error(f"CUDA error on attempt {attempt + 1}: {e}")
                        if attempt == retry_count:
                            raise GPUDeviceError(f"CUDA error after {retry_count + 1} attempts: {e}")
                        time.sleep(0.5)
                    else:
                        raise
        return wrapper
    return decorator


@contextmanager
def gpu_memory_guard(device: str = "cuda", threshold_gb: float = 0.5):
    """GPU内存保护上下文管理器"""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        yield
        return
    
    device_id = int(device.split(":")[1]) if ":" in device else 0
    monitor = GPUHealthMonitor()
    
    # 检查进入时的内存状态
    memory_before = monitor.get_memory_info(device_id)
    free_gb_before = memory_before["free"] / (1024**3)
    
    if free_gb_before < threshold_gb:
        logger.warning(f"Low GPU memory before operation: {free_gb_before:.2f}GB")
        torch.cuda.empty_cache()
    
    try:
        yield
    finally:
        # 检查退出时的内存状态
        memory_after = monitor.get_memory_info(device_id)
        free_gb_after = memory_after["free"] / (1024**3)
        
        if free_gb_after < threshold_gb:
            logger.warning(f"Low GPU memory after operation: {free_gb_after:.2f}GB")
            torch.cuda.empty_cache()


def get_optimal_device(prefer_cuda: bool = True, min_memory_gb: float = 1.0) -> str:
    """获取最优设备"""
    monitor = GPUHealthMonitor()
    
    if not prefer_cuda or not torch.cuda.is_available():
        return "cpu"
    
    # 检查所有可用GPU
    best_device = None
    max_free_memory = 0
    
    for device_id in range(torch.cuda.device_count()):
        health = monitor.check_gpu_health(device_id)
        if health["status"] == "healthy":
            memory_info = health.get("memory_info", {})
            free_gb = memory_info.get("free", 0) / (1024**3)
            
            if free_gb >= min_memory_gb and free_gb > max_free_memory:
                max_free_memory = free_gb
                best_device = f"cuda:{device_id}"
    
    if best_device:
        logger.info(f"Selected {best_device} with {max_free_memory:.2f}GB free memory")
        return best_device
    else:
        logger.warning("No suitable GPU found, falling back to CPU")
        return "cpu"