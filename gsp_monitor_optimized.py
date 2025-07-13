#!/usr/bin/env python3
"""
优化的GSP错误监控器 - 适合长期运行
解决内存泄漏、功耗和性能问题
"""

import torch
import time
import threading
import logging
import logging.handlers
import subprocess
import sys
import gc
import os
from typing import Optional, Dict
from contextlib import contextmanager

# 可选导入psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# 配置轻量级日志，避免日志文件过大
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 创建旋转文件处理器
rotating_handler = logging.handlers.RotatingFileHandler(
    'gsp_monitor.log', 
    maxBytes=10*1024*1024,  # 10MB
    backupCount=2
)
rotating_handler.setFormatter(log_formatter)

# 创建控制台处理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

# 配置根日志器
logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, rotating_handler]
)
logger = logging.getLogger(__name__)

class OptimizedGSPMonitor:
    """优化的GSP错误监控器 - 适合长期运行"""
    
    def __init__(self, device: str = "cuda:0", base_interval: float = 60.0):
        self.device = device
        self.base_interval = base_interval  # 基础检查间隔
        self.current_interval = base_interval
        self.monitoring = False
        self.monitor_thread = None
        self.last_error_time = 0
        self.error_count = 0
        self.health_streak = 0  # 连续健康检查次数
        
        # 智能模式
        self.smart_mode = True  # 启用智能监控
        self.min_interval = 30.0   # 最小间隔
        self.max_interval = 300.0  # 最大间隔（5分钟）
        
        # 资源管理
        self.keepalive_tensor = None
        self.last_cleanup = time.time()
        self.cleanup_interval = 3600  # 每小时清理一次
        
        # 检查GPU是否可用
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
            
        self.device_id = int(device.split(":")[1]) if ":" in device else 0
        
        # 延迟初始化keepalive张量
        self._keepalive_active = False
        
    def _init_keepalive(self):
        """按需初始化保活张量"""
        if self._keepalive_active and self.keepalive_tensor is not None:
            return True
            
        try:
            # 创建最小的张量保持GPU活跃
            self.keepalive_tensor = torch.ones(10, 10, device=self.device, dtype=torch.float16)
            self._keepalive_active = True
            logger.info(f"Minimal keepalive tensor created (400 bytes)")
            return True
        except Exception as e:
            logger.error(f"Failed to create keepalive tensor: {e}")
            self._keepalive_active = False
            return False
            
    def _cleanup_keepalive(self):
        """清理保活张量"""
        if self.keepalive_tensor is not None:
            del self.keepalive_tensor
            self.keepalive_tensor = None
            self._keepalive_active = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug("Keepalive tensor cleaned up")
    
    @contextmanager
    def _temporary_keepalive(self):
        """临时keepalive上下文管理器"""
        was_active = self._keepalive_active
        if not was_active:
            self._init_keepalive()
        try:
            yield
        finally:
            if not was_active:
                self._cleanup_keepalive()
    
    def check_gpu_health(self, lightweight: bool = False) -> Dict:
        """检查GPU健康状态
        
        Args:
            lightweight: 是否使用轻量级检查（不创建张量）
        """
        try:
            # 基本CUDA检查
            if not torch.cuda.is_available():
                return {"status": "cuda_unavailable", "error": "CUDA not available"}
            
            # 检查内存状态（轻量级）
            memory_info = torch.cuda.memory_stats(self.device_id)
            allocated = memory_info.get("allocated_bytes.all.current", 0)
            
            health_data = {
                "status": "healthy",
                "allocated_mb": allocated / (1024**2),
                "timestamp": time.time()
            }
            
            # 非轻量级检查：进行简单计算测试
            if not lightweight:
                with self._temporary_keepalive():
                    if self.keepalive_tensor is not None:
                        # 最小的计算测试
                        result = self.keepalive_tensor.sum()
                        del result
            
            # 获取nvidia-smi信息（降低频率）
            if not lightweight or self.health_streak % 5 == 0:
                gpu_info = self._check_nvidia_smi()
                health_data["gpu_info"] = gpu_info
                
                # 检查温度和功耗
                if gpu_info:
                    temp = gpu_info.get("temperature", 0)
                    power = gpu_info.get("power_draw", 0)
                    if temp > 85:  # 高温警告
                        health_data["warning"] = f"High temperature: {temp}°C"
                    if power > 300:  # 高功耗警告
                        health_data["warning"] = f"High power draw: {power}W"
            
            return health_data
            
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "gsp" in error_msg or "gpu system processor" in error_msg:
                logger.error(f"GSP error detected: {e}")
                self._handle_gsp_error()
                return {"status": "gsp_error", "error": str(e), "timestamp": time.time()}
            else:
                logger.warning(f"GPU health check failed: {e}")
                return {"status": "gpu_error", "error": str(e), "timestamp": time.time()}
                
        except Exception as e:
            logger.error(f"Unexpected error in GPU health check: {e}")
            return {"status": "unknown_error", "error": str(e), "timestamp": time.time()}
    
    def _check_nvidia_smi(self) -> Optional[Dict]:
        """检查nvidia-smi状态（带缓存）"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used', 
                 '--format=csv,noheader,nounits'], 
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                return {
                    "temperature": float(parts[0]),
                    "power_draw": float(parts[1]),
                    "utilization": float(parts[2]),
                    "memory_used_mb": float(parts[3])
                }
        except Exception as e:
            logger.debug(f"nvidia-smi check failed: {e}")
        return None
    
    def _handle_gsp_error(self):
        """处理GSP错误"""
        current_time = time.time()
        
        # 记录错误频率
        if current_time - self.last_error_time < 300:  # 5分钟内重复错误
            self.error_count += 1
        else:
            self.error_count = 1
            
        self.last_error_time = current_time
        self.health_streak = 0  # 重置健康streak
        
        logger.warning(f"GSP error #{self.error_count} detected")
        
        # 尝试恢复
        self._attempt_recovery()
        
        # 调整监控策略
        if self.smart_mode:
            # GSP错误后提高监控频率
            self.current_interval = max(self.min_interval, self.current_interval * 0.5)
            logger.info(f"Increased monitoring frequency to {self.current_interval}s")
        
        # 如果错误过于频繁，建议重启
        if self.error_count >= 3:
            logger.error("Multiple GSP errors detected! Consider rebooting or disabling GSP firmware.")
            
    def _attempt_recovery(self):
        """轻量级GSP错误恢复"""
        try:
            # 1. 清理所有GPU资源
            self._cleanup_keepalive()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # 避免同步，除非必要
                # torch.cuda.synchronize(self.device_id)
                
            # 2. 强制Python垃圾回收
            gc.collect()
            
            # 3. 短暂等待让GPU稳定
            time.sleep(2)
            
            logger.info("Lightweight GSP recovery completed")
            
        except Exception as e:
            logger.error(f"GSP recovery failed: {e}")
    
    def _adaptive_interval(self):
        """智能调整监控间隔"""
        if not self.smart_mode:
            return
            
        # 连续健康时降低频率
        if self.health_streak > 10:
            self.current_interval = min(self.max_interval, self.current_interval * 1.2)
        elif self.health_streak > 5:
            self.current_interval = min(self.max_interval, self.current_interval * 1.1)
        elif self.error_count > 0:
            # 有错误历史时保持较高频率
            self.current_interval = max(self.min_interval, self.current_interval * 0.9)
    
    def _periodic_cleanup(self):
        """定期清理资源"""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            logger.info("Performing periodic cleanup...")
            
            # 清理GPU资源
            was_active = self._keepalive_active
            self._cleanup_keepalive()
            
            # 强制垃圾回收
            gc.collect()
            
            # 记录内存使用（如果psutil可用）
            if PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process(os.getpid())
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    logger.info(f"Process memory usage: {memory_mb:.1f}MB")
                except Exception as e:
                    logger.debug(f"Memory check failed: {e}")
            else:
                logger.debug("psutil not available for memory monitoring")
            
            # 如果keepalive之前是活跃的，重新创建
            if was_active:
                self._init_keepalive()
                
            self.last_cleanup = current_time
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Optimized GSP monitoring started (base interval: {self.base_interval}s, smart mode: {self.smart_mode})")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        self._cleanup_keepalive()
        logger.info("GSP monitoring stopped and resources cleaned")
    
    def _monitor_loop(self):
        """优化的监控循环"""
        consecutive_errors = 0
        
        while self.monitoring:
            try:
                # 根据健康状态选择检查类型
                lightweight = self.health_streak > 5 and self.error_count == 0
                health = self.check_gpu_health(lightweight=lightweight)
                
                if health["status"] == "healthy":
                    self.health_streak += 1
                    consecutive_errors = 0
                    
                    # 记录健康状态（降低频率）
                    if self.health_streak % 10 == 0:
                        logger.info(f"GPU healthy (streak: {self.health_streak})")
                        
                elif health["status"] == "gsp_error":
                    consecutive_errors += 1
                    self.health_streak = 0
                    
                    # 连续错误时增加等待时间
                    if consecutive_errors > 2:
                        logger.warning("Multiple consecutive GSP errors, extended wait...")
                        time.sleep(30)
                        
                else:
                    logger.warning(f"GPU health issue: {health['status']}")
                    consecutive_errors += 1
                    self.health_streak = 0
                
                # 智能调整间隔
                self._adaptive_interval()
                
                # 定期清理
                self._periodic_cleanup()
                
                # 等待下次检查
                time.sleep(self.current_interval)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                consecutive_errors += 1
                time.sleep(min(self.current_interval, 60))  # 错误时最多等待1分钟
    
    def get_status(self) -> Dict:
        """获取监控器状态"""
        return {
            "monitoring": self.monitoring,
            "current_interval": self.current_interval,
            "health_streak": self.health_streak,
            "error_count": self.error_count,
            "keepalive_active": self._keepalive_active,
            "uptime": time.time() - (self.last_cleanup - self.cleanup_interval) if hasattr(self, 'start_time') else 0
        }
    
    def __del__(self):
        """清理资源"""
        self.stop_monitoring()


def main():
    """主函数"""
    print("🛡️ Optimized GSP Monitor for Long-term Use")
    print("==========================================")
    
    try:
        # 创建优化的监控器
        monitor = OptimizedGSPMonitor(device="cuda:0", base_interval=60.0)
        
        # 运行初始健康检查
        print("\n📊 Initial GPU Health Check:")
        health = monitor.check_gpu_health(lightweight=False)
        print(f"Status: {health['status']}")
        if health.get('gpu_info'):
            info = health['gpu_info']
            print(f"Temperature: {info.get('temperature', 'N/A')}°C")
            print(f"Power: {info.get('power_draw', 'N/A')}W")
            print(f"Memory Used: {info.get('memory_used_mb', 'N/A')}MB")
        
        # 开始监控
        monitor.start_monitoring()
        
        print(f"\n🔍 Optimized monitoring started")
        print("Features:")
        print("- Smart interval adjustment (60s - 300s)")
        print("- Lightweight health checks")
        print("- Automatic resource cleanup")
        print("- Memory leak prevention")
        print("\nPress Ctrl+C to stop...")
        
        try:
            while True:
                time.sleep(10)
                # 每10秒显示一次状态
                status = monitor.get_status()
                print(f"Interval: {status['current_interval']:.0f}s, Health streak: {status['health_streak']}, Errors: {status['error_count']}")
        except KeyboardInterrupt:
            print("\n🛑 Stopping monitor...")
            monitor.stop_monitoring()
            print("✅ Monitor stopped and cleaned up")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())