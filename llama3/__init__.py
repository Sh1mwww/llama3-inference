"""
This __init__.py serves as the unified entry point for the llama3 package.

Main purposes:
1. Re-export core classes (ModelArgs, Transformer, LLaMA) so users can import
   them directly from the top-level package:
       from llama3 import LLaMA, ModelArgs, Transformer

2. Optionally expose GPU utilities (GPUHealthMonitor, SafeGPUManager, etc.)
   if the environment supports CUDA and the required dependencies are available.
   If not, the package gracefully falls back to a CPU-only mode without raising
   ImportError.

3. Define __all__ dynamically so that `from llama3 import *` adapts to the
   runtime environment:
       - On GPU-enabled systems → include GPU/memory utilities
       - On CPU-only systems   → only expose the core model components
"""

from .config import ModelArgs
from .model import Transformer
from .generator import LLaMA

try:
    from .gpu_utils import GPUHealthMonitor, SafeGPUManager, get_optimal_device
    from .memory_manager import set_global_memory_limit, get_memory_info
    __all__ = ["ModelArgs", "Transformer", "LLaMA",
               "GPUHealthMonitor", "SafeGPUManager", "get_optimal_device",
               "set_global_memory_limit", "get_memory_info"]
except ImportError:
    __all__ = ["ModelArgs", "Transformer", "LLaMA"]
