import time
import json
import threading
import psutil
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn

from .stream_mnt import get_streams
from .config import load_runtime_config

# NVTX profiling support (no-op fallback if unavailable)
try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False

    class nvtx:
        @staticmethod
        def range_push(name): ...
        @staticmethod
        def range_pop(): ...


def _pinned_clone_cpu(t: torch.Tensor) -> torch.Tensor:
    """Return a contiguous, (best-effort) pinned CPU copy of tensor t."""
    x = t.detach().cpu().contiguous()
    try:
        x = x.pin_memory()
    except Exception:
        pass
    return x


def _collect_block_modules(block: nn.Module) -> Dict[str, nn.Module]:
    """
    Collect submodules within a transformer block that should be streamed:
      - Prefer fine-grained modules exposed by attention/feed_forward via `_get_modules_dict`.
      - Collect common norm layers (small, suitable for GPU residency).
      - Fallback to the entire block if nothing else is found.
    The returned dict maps a descriptive name -> submodule.
    """
    mods: Dict[str, nn.Module] = {}

    # Attention submodules
    if hasattr(block, "attention") and hasattr(block.attention, "_get_modules_dict"):
        try:
            mods.update(block.attention._get_modules_dict())
        except Exception:
            pass

    # FFN submodules
    if hasattr(block, "feed_forward") and hasattr(block.feed_forward, "_get_modules_dict"):
        try:
            mods.update(block.feed_forward._get_modules_dict())
        except Exception:
            pass

    # Norm layers (small; good GPU residents)
    norm_modules: Dict[str, nn.Module] = {}
    for norm_name in ["attn_norm", "ffn_norm", "attention_norm", "ffn_norm_pre", "ffn_norm_post"]:
        if hasattr(block, norm_name):
            norm_module = getattr(block, norm_name)
            if isinstance(norm_module, nn.Module):
                norm_modules[f"norm_{norm_name}"] = norm_module

    if norm_modules:
        mods.update(norm_modules)

    # Fallback: stream the whole block
    if not mods:
        mods["__block__"] = block

    return mods


class WeightStreamingManager:
    """
    Weight Streaming Manager (minimally invasive):

    - Keep a pinned-CPU master copy of parameters; stage to GPU just-in-time.
    - Before entering layer i: ensure_on_gpu(i) (optionally wait for readiness).
    - Prefetch layers i+1..i+prefetch_distance asynchronously on a dedicated H2D stream.
    - Simple LRU keeps up to `max_cached_layers` layers resident on GPU.
      Eviction does not do D2H for parameters—switches pointers back to CPU master copy.
      (Buffers may copy back to CPU if needed.)
    - Norm modules are moved to GPU once and kept resident.

    Typical usage:
        wsm = WeightStreamingManager(model, device="cuda",
                                     prefetch_distance=1, max_cached_layers=4)
        # Hooks are installed automatically on each layer.
    """

    def __init__(self,
                 model: nn.Module,
                 device: str = "cuda",
                 prefetch_distance: int = 1,
                 max_cached_layers: int = 4,
                 layers_attr: str = "layers",
                 warmup_layers: int = 1,
                 verbose: bool = False,
                 monitor_fragmentation: bool = False,
                 ssd_manifest_path: Optional[str] = None,
                 cpu_cache_layers: int = 50,
                 staging_mb: int = 64):
        assert torch.cuda.is_available(), "CUDA not available"
        self.device = device
        self.verbose = verbose
        self.streams = get_streams(device)
        self.prefetch_distance = int(prefetch_distance)
        self.max_cached_layers = int(max_cached_layers)
        self.cpu_cache_layers = int(cpu_cache_layers)

        # SSD backend configuration
        self.ssd_enabled = ssd_manifest_path is not None
        self.ssd_manifest = None
        self.ssd_dio = None
        self.staging_buffer = None
        self.layers_params = {}

        # CPU cache for SSD->CPU->GPU pipeline
        self.cpu_cache: OrderedDict[int, Dict[str, torch.Tensor]] = OrderedDict()
        self.cpu_cache_lock = threading.Lock()

        # Background prefetch management
        self.prefetch_thread: Optional[threading.Thread] = None
        self.prefetch_queue: List[int] = []
        self.stop_prefetch = threading.Event()
        self.prefetch_lock = threading.Lock()

        # Preload completion tracking
        self.gpu_preload_complete = threading.Event()
        self.cpu_preload_complete = threading.Event()
        self.target_gpu_layers = warmup_layers
        self.target_cpu_layers = cpu_cache_layers

        # Optional fragmentation monitoring
        self.monitor_fragmentation = monitor_fragmentation
        self.memory_stats: List[Dict] = []
        self.fragmentation_threshold = 0.3
        # PD 自适应参数（滞回 + EMA）
        
        
        _rcfg = {}
        try:
            _rcfg = load_runtime_config()
            _rcfg = getattr(_rcfg, "io", {}) if hasattr(_rcfg, "io") else {}
        except Exception:
            _rcfg = {}
        self.pd_cap = int(getattr(_rcfg, "PD_CAP", 3) or 3)
        self.pcie_hi = float(getattr(_rcfg, "PCIE_BUSY_UTIL_TH_HI", 0.70))
        self.pcie_lo = float(getattr(_rcfg, "PCIE_BUSY_UTIL_TH_LO", 0.60))
        self.pin_lo  = float(getattr(_rcfg, "PINNED_LOW_WATERMARK", 0.20))
        self.pin_hi  = float(getattr(_rcfg, "PINNED_HIGH_WATERMARK", 0.30))
        self.throttle_ms = int(getattr(_rcfg, "IO_RAW_THROTTLE_MS", 30))
        self._pd_current = max(1, self.prefetch_distance)
        self._pcie_ema = 0.0
        self._ema_alpha = 0.2
        self._last_h2d_ms = 0.0
        # 可选：外部 KV Offloader（若主程序传入，可用于触发“暂停写”）
        self.kv_offloader = None

        # Resolve transformer blocks (layers)
        blocks: Optional[List[nn.Module]] = None
        if hasattr(model, "layer_infos"):
            try:
                blocks = [info["block"] for info in getattr(model, "layer_infos")]
            except Exception:
                blocks = None
        if blocks is None and hasattr(model, layers_attr):
            blocks = list(getattr(model, layers_attr))
        if not blocks:
            candidates = []
            for _name, child in model.named_children():
                if isinstance(child, (nn.Sequential, nn.ModuleList)):
                    candidates.append(list(child))
            blocks = max(candidates, key=len) if candidates else []
        if not blocks:
            raise RuntimeError("Cannot locate transformer blocks (layers) on model.")

        self.blocks: List[nn.Module] = blocks

        # Per-layer streaming units
        self.block_mods: Dict[int, Dict[str, nn.Module]] = {
            i: _collect_block_modules(b) for i, b in enumerate(self.blocks)
        }

        # Update target layers based on actual block count
        self.target_cpu_layers = min(self.target_cpu_layers, len(self.blocks))
        self.target_gpu_layers = min(self.target_gpu_layers, len(self.blocks))

        # CPU master copy: Parameter object id -> pinned CPU tensor
        self.cpu_stash: Dict[int, torch.Tensor] = {}
        # GPU LRU cache: layer index -> None
        self.gpu_cache = OrderedDict()
        # Per-layer "ready" events recorded on H2D stream
        self.layer_events: Dict[int, torch.cuda.Event] = {}

        # Install pre-forward hooks: ensure current layer + prefetch next ones
        for i, blk in enumerate(self.blocks):
            blk.register_forward_pre_hook(self._pre_hook_factory(i))

        # Move norm modules to GPU once and keep resident
        self._setup_resident_norms()

        # Initialize SSD backend if enabled
        if self.ssd_enabled:
            self._initialize_ssd_backend(ssd_manifest_path, staging_mb)

        # (Optional) Warm up target GPU layers to reduce initial latency
        if self.target_gpu_layers > 0:
            warm = list(range(min(self.target_gpu_layers, len(self.blocks))))
            self.prefetch(warm)
            if self.verbose:
                print(f"[WSM] GPU warmup prefetch: {warm} (target: {self.target_gpu_layers} layers)")

    # -------- SSD backend initialization --------

    def _initialize_ssd_backend(self, manifest_path: str, staging_mb: int):
        """Initialize SSD raw device backend for weight streaming"""
        try:
            from .weights_io_ssd_dram import (
                DirectIOFile, load_resident_to_gpu,
                alloc_pinned_aligned, DTYPE_MAP
            )

            # Load manifest
            self.ssd_manifest = json.loads(Path(manifest_path).read_text())
            raw_device = self.ssd_manifest["raw_device"]
            block_size = self.ssd_manifest["block_size"]

            # Create DirectIO file handle
            self.ssd_dio = DirectIOFile(raw_device, mode="r", block_size=block_size)

            # Create staging buffer for SSD->CPU transfers
            staging_bytes = staging_mb * 1024 * 1024
            staging_bytes = (staging_bytes // block_size) * block_size
            self.staging_buffer = alloc_pinned_aligned(staging_bytes, block_size)

            # Organize parameters by layer
            self._organize_params_by_layer()

            # Check DRAM capacity before proceeding
            self._check_dram_capacity()

            # Load resident weights to GPU
            # Note: This will handle uninitialized parameters by loading from SSD
            load_resident_to_gpu(self.model, self.ssd_manifest, device=self.device)

            # Start background prefetch worker if prefetch is enabled
            if self.prefetch_distance > 0:
                self._start_prefetch_worker()
            else:
                print("⚠️  Prefetch disabled - checking if all weights fit in DRAM...")
                self._validate_no_prefetch_mode()

            if self.verbose:
                print(f"[WSM] SSD backend initialized: {raw_device}")
                print(f"[WSM] Staging buffer: {staging_mb}MB, CPU cache: {self.cpu_cache_layers} layers")

        except Exception as e:
            if self.verbose:
                print(f"[WSM] Failed to initialize SSD backend: {e}")
            self.ssd_enabled = False

    def _organize_params_by_layer(self):
        """Organize manifest parameters by layer for efficient access"""
        self.layers_params = {}
        for param in self.ssd_manifest["params"]:
            layer_id = param["layer"]
            if layer_id >= 0:  # Skip non-layer params
                if layer_id not in self.layers_params:
                    self.layers_params[layer_id] = []
                self.layers_params[layer_id].append(param)

    def _start_prefetch_worker(self):
        """Start background thread for SSD->CPU prefetching"""
        def prefetch_worker():
            # Initial preload: schedule first target_cpu_layers for CPU cache
            if self.ssd_enabled:
                initial_layers = list(range(min(self.target_cpu_layers, len(self.blocks))))
                with self.prefetch_lock:
                    self.prefetch_queue.extend(initial_layers)
                if self.verbose:
                    print(f"[WSM] Scheduled initial CPU preload for layers: {initial_layers}")

            while not self.stop_prefetch.is_set():
                layer_to_prefetch = None

                with self.prefetch_lock:
                    if self.prefetch_queue:
                        layer_to_prefetch = self.prefetch_queue.pop(0)

                if layer_to_prefetch is not None:
                    try:
                        self._load_layer_to_cpu(layer_to_prefetch)
                    except Exception as e:
                        if self.verbose:
                            print(f"[WSM] CPU prefetch failed for layer {layer_to_prefetch}: {e}")

                time.sleep(0.001)  # Small delay to avoid CPU spinning

        self.prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self.prefetch_thread.start()

    def _load_layer_to_cpu(self, layer_idx: int):
        """Load layer weights from SSD to CPU cache"""
        if not self.ssd_enabled or layer_idx in self.cpu_cache:
            return

        nvtx.range_push(f"ssd_to_cpu_layer_{layer_idx}")

        with self.cpu_cache_lock:
            # Double-check after acquiring lock
            if layer_idx in self.cpu_cache:
                nvtx.range_pop()
                return

            # Manage CPU cache size
            while len(self.cpu_cache) >= self.cpu_cache_layers:
                old_layer, _ = self.cpu_cache.popitem(last=False)
                if self.verbose:
                    print(f"[WSM] Evicted layer {old_layer} from CPU cache")

            if layer_idx not in self.layers_params:
                nvtx.range_pop()
                return

            layer_weights = {}

            # Load stream weights for this layer
            for param_info in self.layers_params[layer_idx]:
                if param_info["policy"] != "stream":
                    continue

                try:
                    # Import here to avoid circular imports
                    from .weights_io_ssd_dram import DTYPE_MAP

                    param_name = param_info["name"]
                    stride = param_info["stride"]
                    offset = param_info["offset"]
                    nbytes = param_info["nbytes"]

                    # Ensure staging buffer is large enough
                    if stride > len(self.staging_buffer):
                        from .weights_io_ssd_dram import alloc_pinned_aligned
                        block_size = self.ssd_manifest["block_size"]
                        new_size = ((stride + block_size - 1) // block_size) * block_size
                        self.staging_buffer = alloc_pinned_aligned(new_size, block_size)
                        if self.verbose:
                            print(f"[WSM] Expanded staging buffer to {new_size} bytes")

                    # Read from SSD to staging buffer
                    self.ssd_dio.pread_into_tensor(self.staging_buffer, stride, offset)

                    # Convert to proper tensor format
                    param_tensor = torch.empty(
                        param_info["shape"],
                        dtype=DTYPE_MAP[param_info["dtype"]],
                        pin_memory=True
                    )
                    param_tensor.view(-1).view(torch.uint8)[:nbytes].copy_(
                        self.staging_buffer[:nbytes]
                    )

                    layer_weights[param_name] = param_tensor

                except Exception as e:
                    if self.verbose:
                        print(f"[WSM] Failed to load {param_name}: {e}")
                    continue

            if layer_weights:
                self.cpu_cache[layer_idx] = layer_weights
                if self.verbose:
                    print(f"[WSM] Loaded layer {layer_idx} to CPU cache ({len(layer_weights)} params)")

        nvtx.range_pop()

    def _check_dram_capacity(self):
        """Check if there's enough DRAM to cache the required weights"""
        if not self.ssd_enabled:
            return

        # Calculate total weight size for stream parameters
        total_stream_bytes = 0
        stream_layers_count = 0

        for layer_id, params in self.layers_params.items():
            layer_bytes = 0
            for param_info in params:
                if param_info["policy"] == "stream":
                    layer_bytes += param_info["nbytes"]

            if layer_bytes > 0:
                total_stream_bytes += layer_bytes
                stream_layers_count += 1

        # Calculate required DRAM for CPU cache
        if stream_layers_count > 0:
            avg_layer_size = total_stream_bytes / stream_layers_count
            required_cache_bytes = avg_layer_size * self.cpu_cache_layers
        else:
            required_cache_bytes = 0

        # Get current system memory info
        memory = psutil.virtual_memory()
        available_bytes = memory.available
        total_bytes = memory.total

        # Convert to human readable
        required_gb = required_cache_bytes / (1024**3)
        available_gb = available_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)

        print(f"💾 DRAM Capacity Check:")
        print(f"   Total system DRAM: {total_gb:.1f} GB")
        print(f"   Available DRAM: {available_gb:.1f} GB")
        print(f"   Required for {self.cpu_cache_layers} layer cache: {required_gb:.1f} GB")
        print(f"   Total stream layers: {stream_layers_count}")
        print(f"   Average layer size: {avg_layer_size/(1024**3):.2f} GB")

        # Check if we have enough memory with safety margin
        safety_margin = 0.1  # 10% safety margin
        required_with_margin = required_cache_bytes * (1 + safety_margin)

        if required_with_margin > available_bytes:
            deficit_gb = (required_with_margin - available_bytes) / (1024**3)
            print(f"❌ INSUFFICIENT DRAM!")
            print(f"   Deficit: {deficit_gb:.1f} GB (including 10% safety margin)")
            print(f"   Suggestion: Reduce cpu_cache_layers from {self.cpu_cache_layers} to {int(self.cpu_cache_layers * available_bytes / required_with_margin)}")
            raise RuntimeError(f"Insufficient DRAM: need {required_gb:.1f}GB but only {available_gb:.1f}GB available")

        print(f"✅ DRAM capacity sufficient (margin: {(available_bytes - required_cache_bytes)/(1024**3):.1f} GB)")

    def _validate_no_prefetch_mode(self):
        """Validate that all weights can fit in DRAM when prefetch is disabled"""
        if not self.ssd_enabled:
            return

        # Calculate total size of all stream weights
        total_stream_bytes = 0
        for layer_id, params in self.layers_params.items():
            for param_info in params:
                if param_info["policy"] == "stream":
                    total_stream_bytes += param_info["nbytes"]

        # Get available memory
        memory = psutil.virtual_memory()
        available_bytes = memory.available

        total_gb = total_stream_bytes / (1024**3)
        available_gb = available_bytes / (1024**3)

        print(f"🔍 No-Prefetch Mode Validation:")
        print(f"   Total stream weights: {total_gb:.1f} GB")
        print(f"   Available DRAM: {available_gb:.1f} GB")

        # Check if all weights fit with safety margin
        safety_margin = 0.15  # 15% safety margin for no-prefetch mode
        required_with_margin = total_stream_bytes * (1 + safety_margin)

        if required_with_margin > available_bytes:
            deficit_gb = (required_with_margin - available_bytes) / (1024**3)
            print(f"❌ CANNOT RUN WITHOUT PREFETCH!")
            print(f"   All weights ({total_gb:.1f} GB) cannot fit in available DRAM ({available_gb:.1f} GB)")
            print(f"   Deficit: {deficit_gb:.1f} GB (including 15% safety margin)")
            print(f"💡 Solutions:")
            print(f"   1. Enable prefetch mode: set prefetch_distance > 0")
            print(f"   2. Reduce cpu_cache_layers to enable streaming")
            print(f"   3. Add more DRAM to your system")
            raise RuntimeError(f"Cannot run without prefetch: need {total_gb:.1f}GB but only {available_gb:.1f}GB available")

        print(f"✅ All weights fit in DRAM - no prefetch mode validated")
        # Set CPU cache to hold all layers since we're not using prefetch
        self.cpu_cache_layers = len(self.layers_params)
        print(f"📝 Updated CPU cache to hold all {self.cpu_cache_layers} layers")

    def _schedule_cpu_prefetch(self, current_layer: int):
        """Schedule CPU prefetch for upcoming layers"""
        if not self.ssd_enabled:
            return

        with self.prefetch_lock:
            for offset in range(1, self.prefetch_distance + 1):
                next_layer = current_layer + offset
                if (next_layer < len(self.blocks) and
                    next_layer not in self.cpu_cache and
                    next_layer not in self.prefetch_queue):
                    self.prefetch_queue.append(next_layer)

    def wait_for_preload_ready(self, timeout: float = 300.0) -> bool:
        """
        等待预加载完成：GPU有target_gpu_layers层，CPU有target_cpu_layers层

        Args:
            timeout: 最大等待时间（秒）

        Returns:
            bool: 是否在超时前完成预加载
        """
        import time

        if self.verbose:
            print(f"[WSM] Waiting for preload: target GPU={self.target_gpu_layers}, target CPU={self.target_cpu_layers}")

        start_time = time.time()

        while time.time() - start_time < timeout:
            gpu_ready = len(self.gpu_cache) >= self.target_gpu_layers
            cpu_ready = len(self.cpu_cache) >= self.target_cpu_layers if self.ssd_enabled else True

            if gpu_ready and cpu_ready:
                if self.verbose:
                    print(f"[WSM] Preload completed: {len(self.gpu_cache)} GPU layers + {len(self.cpu_cache)} CPU layers ready")
                self.gpu_preload_complete.set()
                self.cpu_preload_complete.set()
                return True

            if self.verbose and int(time.time() - start_time) % 5 == 0:  # Progress update every 5 seconds
                print(f"[WSM] Preload progress: GPU {len(self.gpu_cache)}/{self.target_gpu_layers}, CPU {len(self.cpu_cache)}/{self.target_cpu_layers}")

            time.sleep(0.1)

        print(f"[WSM] ⚠️  Preload timeout after {timeout}s: GPU {len(self.gpu_cache)}/{self.target_gpu_layers}, CPU {len(self.cpu_cache)}/{self.target_cpu_layers}")
        return False

    # -------- CPU/GPU movement primitives --------

    def _setup_resident_norms(self):
        """Move norm modules to GPU and exclude them from streaming/eviction."""
        if self.verbose:
            print("[WSM] Setting up resident norm layers...")
        norm_count = 0
        for layer_id, modules_dict in self.block_mods.items():
            for module_name, module in modules_dict.items():
                if module_name.startswith("norm_"):
                    try:
                        module.to(self.device)
                        norm_count += 1
                        if self.verbose:
                            print(f"[WSM] Layer {layer_id}: {module_name} -> {self.device} (resident)")
                    except Exception as e:
                        if self.verbose:
                            print(f"[WSM] Warning: failed to move {module_name} to {self.device}: {e}")
        if self.verbose:
            print(f"[WSM] {norm_count} norm modules set as GPU resident")

    def _record_layer_ready_event(self, idx: int):
        """Record an event on the weight H2D stream marking all enqueued copies for this layer."""
        if not torch.cuda.is_available() or getattr(self.streams, "weight_h2d", None) is None:
            return
        evt = self.layer_events.get(idx)
        if evt is None:
            evt = torch.cuda.Event(blocking=False)
            self.layer_events[idx] = evt
        evt.record(self.streams.weight_h2d)
        
        # 记录一次 H2D 活动时长，用于 PCIE 近似占用度估计（EMA）
        try:
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record(self.streams.weight_h2d)
            end.record(self.streams.weight_h2d)
            end.synchronize()
            dt_ms = start.elapsed_time(end)  # ~0，但可以作为一次采样
            self._last_h2d_ms = max(self._last_h2d_ms * 0.9, dt_ms)
        except Exception:
            pass


    def _wait_layer_ready(self, idx: int):
        """Wait on the layer's ready event on the current stream (fallback: wait for H2D stream)."""
        evt = self.layer_events.get(idx)
        if evt is not None:
            torch.cuda.current_stream(self.device).wait_event(evt)
        else:
            self.streams.wait_weight_ready_on_current(self.device)

    def _ensure_param_cpu_stash_inplace(self, p: torch.nn.Parameter):
        """
        Replace p.data with a pinned CPU master copy and stash it by Parameter id.
        This avoids holding duplicate CPU tensors.
        """
        pid = id(p)
        if pid in self.cpu_stash:
            return
        pinned = _pinned_clone_cpu(p.data)
        p.data = pinned
        self.cpu_stash[pid] = p.data

    def _ensure_param_on_gpu(self, p: torch.nn.Parameter, layer_idx: Optional[int] = None, param_name: Optional[str] = None):
        """Stage parameter to GPU on the weight H2D stream and switch p.data to the GPU tensor."""
        nvtx.range_push("param_h2d")
        if p.device.type != "cpu":
            nvtx.range_pop()
            return

        # Try to get from CPU cache first if SSD backend is enabled
        if (self.ssd_enabled and layer_idx is not None and param_name is not None and
            layer_idx in self.cpu_cache and param_name in self.cpu_cache[layer_idx]):

            nvtx.range_push("cpu_cache_to_gpu")
            with torch.cuda.stream(self.streams.weight_h2d):
                cached_tensor = self.cpu_cache[layer_idx][param_name]
                p_gpu = cached_tensor.to(self.device, non_blocking=True)
            p.data = p_gpu
            nvtx.range_pop()
            nvtx.range_pop()
            return

        # Fallback to original behavior
        pid = id(p)
        nvtx.range_push("ensure_cpu_stash")
        self._ensure_param_cpu_stash_inplace(p)
        nvtx.range_pop()

        nvtx.range_push("weight_h2d_stream")
        with torch.cuda.stream(self.streams.weight_h2d):
            nvtx.range_push("cpu_to_gpu_transfer")
            p_gpu = self.cpu_stash[pid].to(self.device, non_blocking=True)
            nvtx.range_pop()
        nvtx.range_pop()

        p.data = p_gpu
        nvtx.range_pop()

    def _evict_param_to_cpu(self, p: torch.nn.Parameter):
        """Evict by re-pointing p.data back to its CPU master copy (no D2H)."""
        pid = id(p)
        if pid in self.cpu_stash:
            p.data = self.cpu_stash[pid]

    def _ensure_module_on_gpu(self, m: nn.Module, layer_idx: Optional[int] = None, module_name: Optional[str] = None):
        """Ensure all params/buffers of module m are on GPU."""
        for param_name, p in m.named_parameters(recurse=True):
            # Build full parameter name for CPU cache lookup
            full_param_name = None
            if layer_idx is not None and module_name is not None:
                full_param_name = f"layers.{layer_idx}.{module_name}.{param_name}"
            self._ensure_param_on_gpu(p, layer_idx, full_param_name)

        for b in m.buffers(recurse=True):
            if b.device.type == "cpu":
                with torch.cuda.stream(self.streams.weight_h2d):
                    b_gpu = b.detach().to(self.device, non_blocking=True)
                try:
                    b.data = b_gpu
                except Exception:
                    pass

    def _evict_module_to_cpu(self, m: nn.Module):
        """Evict module m: params point back to CPU master; buffers copied back to CPU."""
        for p in m.parameters(recurse=True):
            self._evict_param_to_cpu(p)
        for b in m.buffers(recurse=True):
            if b.device.type != "cpu":
                try:
                    b_cpu = b.detach().cpu()
                    try:
                        b_cpu = b_cpu.pin_memory()
                    except Exception:
                        pass
                    b.data = b_cpu
                except Exception:
                    pass

    # -------- Layer-level ensure / prefetch / evict --------

    def _pre_hook_factory(self, idx: int):
        def _pre_hook(_module, _inputs):
            # Schedule CPU prefetch for upcoming layers
            self._schedule_cpu_prefetch(idx)

            # Ensure current layer ready on GPU, then prefetch next ones (fire-and-forget).
            self.ensure_on_gpu(idx, wait=True)
            if self.prefetch_distance > 0:
                nxt = [j for j in range(idx + 1, min(idx + 1 + self.prefetch_distance, len(self.blocks)))]
                self.prefetch(nxt)
        return _pre_hook

    def ensure_on_gpu(self, idx: int, wait: bool):
        """Ensure layer idx is present on GPU (respecting LRU); optionally wait for readiness."""
        # 在正式 H2D 前评估一次 PD
        self._decide_pd()
        
        nvtx.range_push(f"ensure_layer_{idx}")
        self._record_memory_stats("ensure_start", idx)

        if idx in self.gpu_cache:
            nvtx.range_push(f"cache_hit_layer_{idx}")
            self.gpu_cache.move_to_end(idx)
            if wait:
                nvtx.range_push(f"wait_layer_{idx}")
                self._wait_layer_ready(idx)
                nvtx.range_pop()
            nvtx.range_pop()
        else:
            nvtx.range_push(f"cache_miss_layer_{idx}")

            # Evict until space is available
            while len(self.gpu_cache) >= self.max_cached_layers:
                old, _ = self.gpu_cache.popitem(last=False)
                nvtx.range_push(f"evict_layer_{old}")
                self._record_memory_stats("evict_start", old)
                self._evict_layer_to_cpu(old)
                self._record_memory_stats("evict_end", old)
                nvtx.range_pop()

            # H2D for current layer (skip resident norms)
            nvtx.range_push(f"h2d_transfer_layer_{idx}")
            self._record_memory_stats("h2d_start", idx)
            for module_name, mod in self.block_mods[idx].items():
                if not module_name.startswith("norm_"):
                    nvtx.range_push(f"h2d_{module_name}")
                    self._ensure_module_on_gpu(mod, idx, module_name)
                    nvtx.range_pop()
            self._record_memory_stats("h2d_end", idx)
            nvtx.range_pop()

            self.gpu_cache[idx] = None

            # Record "layer ready" event on H2D stream after enqueuing all copies
            nvtx.range_push(f"record_event_layer_{idx}")
            self._record_layer_ready_event(idx)
            nvtx.range_pop()

            if self.verbose:
                print(f"[WSM] ->GPU layer={idx}")
            if wait:
                nvtx.range_push(f"wait_layer_{idx}")
                self._wait_layer_ready(idx)
                nvtx.range_pop()

            nvtx.range_pop()  # cache_miss

        self._record_memory_stats("ensure_end", idx)
        nvtx.range_pop()  # ensure_layer

    def prefetch(self, ids: List[int]):
        """Asynchronously prefetch a list of layer indices, respecting the LRU budget."""
        if not ids:
            return
        ids = ids[: max(1, min(self._pd_current, self.pd_cap))]
        nvtx.range_push(f"prefetch_layers_{ids}")

        for idx in ids:
            nvtx.range_push(f"prefetch_layer_{idx}")

            if idx in self.gpu_cache:
                self.gpu_cache.move_to_end(idx)
                nvtx.range_pop()
                continue

            # Evict until space is available
            while len(self.gpu_cache) >= self.max_cached_layers:
                old, _ = self.gpu_cache.popitem(last=False)
                nvtx.range_push(f"prefetch_evict_{old}")
                self._evict_layer_to_cpu(old)
                nvtx.range_pop()

            # H2D for this prefetched layer (skip resident norms)
            nvtx.range_push(f"prefetch_h2d_layer_{idx}")
            for module_name, mod in self.block_mods[idx].items():
                if not module_name.startswith("norm_"):
                    nvtx.range_push(f"prefetch_h2d_{module_name}")
                    self._ensure_module_on_gpu(mod, idx, module_name)
                    nvtx.range_pop()
            nvtx.range_pop()

            self.gpu_cache[idx] = None

            # Record ready event (do not wait)
            nvtx.range_push(f"prefetch_record_event_{idx}")
            self._record_layer_ready_event(idx)
            nvtx.range_pop()

            if self.verbose:
                print(f"[WSM] prefetch layer={idx}")

            nvtx.range_pop()  # prefetch_layer

        nvtx.range_pop()  # prefetch_layers
        
        
    # --------- PD 自适应（滞回+EMA） ---------
    def _decide_pd(self):
        """
        估计 PCIE 忙闲（EMA），并结合 pinned 水位（若可得）做滞回调整：
        - 忙：PCIE>hi 或 pinned<low -> PD=1，暂停 KV 写 throttle_ms
        - 闲：PCIE<lo 且 pinned>high -> PD=min(PD+1, cap)
        - 中：保持
        """
        # 近似估计 PCIE 利用率：以 H2D stream 的 backlog 与最近 H2D 触发为 proxy
        busy_proxy = 1.0
        try:
            # 如果 H2D stream 上还有工作未完成，则趋向 1，否则趋向 0
            busy_proxy = 0.9 if (not self.streams.weight_h2d.query()) else 0.1
        except Exception:
            pass
        # EMA
        self._pcie_ema = self._ema_alpha * busy_proxy + (1.0 - self._ema_alpha) * self._pcie_ema

        # pinned 水位：此处无法读取系统 pinned 池，采用保守估计 0.5；如果你有 HostPinnedExtentPool，可在外层注入
        pinned_free_ratio = 0.5

        # 忙态：降 PD、暂停写
        if (self._pcie_ema >= self.pcie_hi) or (pinned_free_ratio <= self.pin_lo):
            self._pd_current = 1
            if self.kv_offloader is not None:
                try:
                    self.kv_offloader.throttle_writes_for(self.throttle_ms)
                except Exception:
                    pass
            return

        # 闲态：升 PD
        if (self._pcie_ema <= self.pcie_lo) and (pinned_free_ratio >= self.pin_hi):
            self._pd_current = min(self._pd_current + 1, self.pd_cap)
            return
        # 中态：不变


    def _evict_layer_to_cpu(self, idx: int):
        """Evict a layer back to CPU (parameters: pointer switch; buffers: copied)."""
        for name, mod in self.block_mods[idx].items():
            # Keep norm modules resident on GPU.
            if name.startswith("norm_"):
                continue
            self._evict_module_to_cpu(mod)
        # Clear any stale ready event (will be recreated next time)
        self.layer_events.pop(idx, None)
        if self.verbose:
            print(f"[WSM]   evict layer={idx}")

    # -------- Compatibility shims for layers.py --------

    def ensure_weights_cuda(self, layer_id: int, modules: Dict[str, nn.Module], priority: bool = False):
        """
        Compatibility for layers.py: ensure provided layer's modules are on GPU.
        `priority=True` implies 'wait' for readiness.
        """
        if layer_id >= len(self.blocks):
            return
        self.ensure_on_gpu(layer_id, wait=priority)
        # Extra safety: if caller passes a different module dict, ensure those too.
        for name, module in modules.items():
            try:
                self._ensure_module_on_gpu(module, layer_id, name)
            except Exception as e:
                if self.verbose:
                    print(f"[WSM] Warning: ensure {name} on GPU failed (layer {layer_id}): {e}")

    def prefetch_weights(self, layer_ids: List[int], modules_dict: Dict[int, Dict[str, nn.Module]] = None):
        """Compatibility for layers.py: prefetch by layer ids."""
        self.prefetch(layer_ids)

    # -------- Memory stats / fragmentation reporting --------

    def _get_memory_info(self):
        """Return current CUDA allocator stats for the configured device."""
        if not torch.cuda.is_available():
            return None

        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        max_allocated = torch.cuda.max_memory_allocated(self.device)
        max_reserved = torch.cuda.max_memory_reserved(self.device)
        memory_stats = torch.cuda.memory_stats(self.device)

        return {
            "timestamp": time.time(),
            "allocated_mb": allocated / 1024**2,
            "reserved_mb": reserved / 1024**2,
            "max_allocated_mb": max_allocated / 1024**2,
            "max_reserved_mb": max_reserved / 1024**2,
            "fragmentation_ratio": (reserved - allocated) / reserved if reserved > 0 else 0.0,
            "allocation_count": memory_stats.get("allocation.all.current", 0),
            "segment_count": memory_stats.get("segment.all.current", 0),
            "large_pool_allocated": memory_stats.get("allocated_bytes.large_pool.current", 0) / 1024**2,
            "small_pool_allocated": memory_stats.get("allocated_bytes.small_pool.current", 0) / 1024**2,
        }

    def _record_memory_stats(self, operation: str, layer_id: int = -1):
        """Append a snapshot of allocator stats if monitoring is enabled."""
        if not self.monitor_fragmentation:
            return
        info = self._get_memory_info()
        if info:
            info["operation"] = operation
            info["layer_id"] = layer_id
            self.memory_stats.append(info)
            if info["fragmentation_ratio"] > self.fragmentation_threshold:
                print(
                    f"⚠️  High fragmentation: {info['fragmentation_ratio']:.3f} "
                    f"during {operation} (layer {layer_id})"
                )

    def get_fragmentation_report(self):
        """Summarize fragmentation/allocator information collected so far."""
        if not self.memory_stats:
            return {"error": "No memory statistics collected"}

        fragmentation_ratios = [s["fragmentation_ratio"] for s in self.memory_stats]
        segment_counts = [s["segment_count"] for s in self.memory_stats]

        report = {
            "total_operations": len(self.memory_stats),
            "max_fragmentation": max(fragmentation_ratios),
            "avg_fragmentation": sum(fragmentation_ratios) / len(fragmentation_ratios),
            "min_fragmentation": min(fragmentation_ratios),
            "max_segments": max(segment_counts),
            "avg_segments": sum(segment_counts) / len(segment_counts),
            "high_fragmentation_count": sum(1 for r in fragmentation_ratios if r > self.fragmentation_threshold),
            "peak_allocated_mb": max(s["allocated_mb"] for s in self.memory_stats),
            "peak_reserved_mb": max(s["reserved_mb"] for s in self.memory_stats),
        }

        if report["max_fragmentation"] > 0.4:
            report["severity"] = "CRITICAL"
        elif report["max_fragmentation"] > 0.3:
            report["severity"] = "HIGH"
        elif report["max_fragmentation"] > 0.15:
            report["severity"] = "MEDIUM"
        else:
            report["severity"] = "LOW"

        return report

    def save_memory_stats(self, filename: str):
        """Persist memory statistics + report to a JSON file."""
        if not self.memory_stats:
            print("⚠️  No memory statistics to save")
            return

        with open(filename, "w") as f:
            json.dump(
                {
                    "memory_stats": self.memory_stats,
                    "fragmentation_report": self.get_fragmentation_report(),
                    "config": {
                        "max_cached_layers": self.max_cached_layers,
                        "prefetch_distance": self.prefetch_distance,
                        "device": self.device,
                        "fragmentation_threshold": self.fragmentation_threshold,
                    },
                },
                f,
                indent=2,
            )
        print(f"💾 Memory statistics saved to {filename}")

    def clear_memory_stats(self):
        """Clear collected stats and reset CUDA peak tracking."""
        self.memory_stats.clear()
        torch.cuda.reset_peak_memory_stats(self.device)

    # -------- Cleanup and resource management --------

    def cleanup(self):
        """Clean up SSD resources and background threads"""
        if self.verbose:
            print("[WSM] Cleaning up resources...")

        # Stop prefetch thread
        if self.ssd_enabled:
            self.stop_prefetch.set()
            if self.prefetch_thread and self.prefetch_thread.is_alive():
                self.prefetch_thread.join(timeout=1.0)

        # Close SSD DirectIO handle
        if hasattr(self, 'ssd_dio') and self.ssd_dio:
            try:
                self.ssd_dio.close()
            except Exception:
                pass

        # Clear caches
        with self.cpu_cache_lock:
            self.cpu_cache.clear()
        self.gpu_cache.clear()

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except Exception:
            pass

    def get_ssd_stats(self) -> Dict[str, Any]:
        """Get SSD backend statistics"""
        if not self.ssd_enabled:
            return {"ssd_enabled": False}

        return {
            "ssd_enabled": True,
            "cpu_cache_size": len(self.cpu_cache),
            "cpu_cached_layers": list(self.cpu_cache.keys()),
            "cpu_cache_max": self.cpu_cache_layers,
            "prefetch_queue_size": len(self.prefetch_queue),
            "staging_buffer_size": len(self.staging_buffer) if self.staging_buffer is not None else 0,
            "total_layers": len(self.layers_params),
        }
