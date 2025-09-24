import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class Streams:
    weight_h2d: Optional[torch.cuda.Stream] = None    # 权重从 CPU 到 GPU
    weight_compute: Optional[torch.cuda.Stream] = None # 权重计算流
    
    kv_h2d: Optional[torch.cuda.Stream] = None       # KVcache 从 DRAM 到 GPU
    kv_d2h: Optional[torch.cuda.Stream] = None       # KVcache 从 GPU 到 DRAM
    
    def wait_weight_ready_on_current(self, device: Optional[str] = None):
        if self.weight_h2d is None:
            return
        dev = device if device is not None else torch.cuda.current_device()
        torch.cuda.current_stream(dev).wait_stream(self.weight_h2d)

_streams_cache = {}

def get_streams(device: str) -> Streams:
    global _streams_cache
    
    if device in _streams_cache:
        return _streams_cache[device]
    
    if not device.startswith("cuda"):
        streams = Streams()
    else:
        try:
            streams = Streams(
                weight_h2d=torch.cuda.Stream(device=device, priority=0),
                # weight_d2h=torch.cuda.Stream(device=device, priority=1),
                weight_compute=torch.cuda.Stream(device=device, priority=-1),
                kv_h2d=torch.cuda.Stream(device=device, priority=0),
                kv_d2h=torch.cuda.Stream(device=device, priority=1),
            )
        except Exception:
            streams = Streams()
    
    _streams_cache[device] = streams
    return streams


def clear_streams_cache():
    global _streams_cache
    _streams_cache.clear()