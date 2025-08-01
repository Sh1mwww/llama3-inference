"""
llama3 - Minimal LLaMA-3 inference package with GPU optimization
"""
from .config import ModelArgs, MemoryLimitArgs
from .model import Transformer
from .generator import LLaMA

# GPU optimization modules (optional imports)
try:
    from .gpu_utils import GPUHealthMonitor, SafeGPUManager, get_optimal_device
    from .memory_manager import set_global_memory_limit, get_memory_info
    __all__ = ["ModelArgs", "MemoryLimitArgs", "Transformer", "LLaMA", 
               "GPUHealthMonitor", "SafeGPUManager", "get_optimal_device",
               "set_global_memory_limit", "get_memory_info"]
except ImportError:
    __all__ = ["ModelArgs", "MemoryLimitArgs", "Transformer", "LLaMA"]
