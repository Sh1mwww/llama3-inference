"""
流管理模块 - 统一管理 CUDA 流
"""
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class Streams:
    """流集合"""
    # 权重流式传输相关流
    weight_h2d: Optional[torch.cuda.Stream] = None    # 权重从 CPU 到 GPU
    # weight_d2h: Optional[torch.cuda.Stream] = None    # 权重从 GPU 到 CPU
    weight_compute: Optional[torch.cuda.Stream] = None # 权重计算流
    
    # KV cache 相关流
    kv_h2d: Optional[torch.cuda.Stream] = None       # KV cache 从 DRAM 到 GPU
    kv_d2h: Optional[torch.cuda.Stream] = None       # KV cache 从 GPU 到 DRAM
    
    def wait_weight_ready_on_current(self, device: Optional[str] = None):
        """在“当前计算流”上等待“权重H2D流”，不阻塞主机线程"""
        if self.weight_h2d is None:
            return
        dev = device if device is not None else torch.cuda.current_device()
        torch.cuda.current_stream(dev).wait_stream(self.weight_h2d)


_streams_cache = {}


def get_streams(device: str) -> Streams:
    """获取指定设备的流集合，使用缓存避免重复创建"""
    global _streams_cache
    
    if device in _streams_cache:
        return _streams_cache[device]
    
    if not device.startswith("cuda"):
        # CPU 设备不需要流
        streams = Streams()
    else:
        # 创建 CUDA 流
        try:
            streams = Streams(
                weight_h2d=torch.cuda.Stream(device=device, priority=0),
                # weight_d2h=torch.cuda.Stream(device=device, priority=1),
                weight_compute=torch.cuda.Stream(device=device, priority=-1),
                kv_h2d=torch.cuda.Stream(device=device, priority=0),
                kv_d2h=torch.cuda.Stream(device=device, priority=1),
            )
        except Exception:
            # 回退到无流模式
            streams = Streams()
    
    _streams_cache[device] = streams
    return streams


def clear_streams_cache():
    """清除流缓存"""
    global _streams_cache
    _streams_cache.clear()