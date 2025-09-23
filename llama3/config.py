import json
import torch.nn as nn
from dataclasses import dataclass, fields, field
from typing import Optional, List, Dict, Any

@dataclass
class LayerInfo:
    layer_id:int
    block: Optional[nn.Module] = None   #encoderblock
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KVCacheArgs:
    ssd_path: str = "/mnt/kv_cache/kv_cache.bin"
    ssd_size_gb: int = 500
    dram_limit_gb: float = 0.1


"""
HBM内存限制配置
"""
@dataclass
class MemoryLimitArgs:
    max_hbm_gb: float = 12.0          # 最大HBM使用量(GB)
    reserved_hbm_gb: float = 1.0      # 预留HBM(GB)  
    enable_monitoring: bool = True     # 启用内存监控
    cleanup_threshold: float = 0.9     # 清理阈值
    auto_limit: bool = True           # 自动设置限制
    
@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: Optional[int]
    vocab_size: int
    multiple_of: int
    ffn_dim_multiplier: Optional[float]
    norm_eps: float
    rope_theta: float
    use_scaled_rope: bool
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    max_batch_size: int = 512  
    max_seq_len: int = 2048
    device: str = "cuda"
    topk_blk: int = 8
    layer_infos: List[LayerInfo] = field(default_factory=list)
    memory_limit: MemoryLimitArgs = field(default_factory=MemoryLimitArgs) 
    
    @staticmethod
    def from_json(params_path: str,
                  max_seq_len: int,
                  max_batch_size: int,
                  device: str = None,
                  memory_limit_gb: float = None):
        # 导入GPU工具
        try:
            from .gpu_utils import get_optimal_device, GPUHealthMonitor
            from .memory_manager import set_global_memory_limit, get_memory_info
        except ImportError:
            # 如果GPU工具不可用，使用基本的设备检查
            import torch
            def get_optimal_device(prefer_cuda=True, min_memory_gb=1.0):
                if prefer_cuda and torch.cuda.is_available():
                    return "cuda"
                return "cpu"
            def set_global_memory_limit(limit_gb, device, reserved_gb=1.0):
                pass
            def get_memory_info(device):
                return {"no_limit": True}
        
        # 智能设备选择
        if device is None:
            device = get_optimal_device(prefer_cuda=True, min_memory_gb=1.0)
            print(f"Auto-selected device: {device}")
        else:
            # 验证指定的设备
            if device.startswith("cuda"):
                try:
                    monitor = GPUHealthMonitor()
                    device_id = int(device.split(":")[1]) if ":" in device else 0
                    health = monitor.check_gpu_health(device_id)
                    
                    if health["status"] != "healthy":
                        print(f"Warning: GPU device {device} health check failed: {health['message']}")
                        fallback_device = get_optimal_device(prefer_cuda=True, min_memory_gb=0.5)
                        if fallback_device != device:
                            print(f"Falling back to {fallback_device}")
                            device = fallback_device
                except Exception as e:
                    print(f"Warning: GPU health check failed: {e}")
                    if not torch.cuda.is_available():
                        print("CUDA not available, using CPU")
                        device = "cpu"
        
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        allowed = {f.name for f in fields(ModelArgs)}
        filtered = {k: v for k, v in params.items() if k in allowed}

        # 设置内存限制
        memory_limit_config = MemoryLimitArgs()
        if memory_limit_gb is not None:
            memory_limit_config.max_hbm_gb = memory_limit_gb
        elif device and device.startswith("cuda"):
            # 自动设置内存限制
            try:
                memory_info = get_memory_info(device)
                if "total_gb" in memory_info:
                    # 默认使用80%的显存，预留20%
                    auto_limit = memory_info["total_gb"] * 0.8
                    memory_limit_config.max_hbm_gb = auto_limit
                    memory_limit_config.reserved_hbm_gb = memory_info["total_gb"] * 0.2
                    print(f"Auto-set memory limit: {auto_limit:.1f}GB (reserved: {memory_limit_config.reserved_hbm_gb:.1f}GB)")
            except Exception as e:
                print(f"Failed to auto-set memory limit: {e}")

        args =  ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            memory_limit=memory_limit_config,
            **filtered
        )

        # 应用全局内存限制
        if device and device.startswith("cuda"):
            try:
                set_global_memory_limit(
                    limit_gb=args.memory_limit.max_hbm_gb,
                    device=device,
                    reserved_gb=args.memory_limit.reserved_hbm_gb
                )
            except Exception as e:
                print(f"Warning: Failed to set global memory limit: {e}")

        args.layer_infos = [LayerInfo(layer_id=i) for i in range(args.n_layers)]
        return args