import time
import json
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .stream_mnt import get_streams

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
                 monitor_fragmentation: bool = False):
        assert torch.cuda.is_available(), "CUDA not available"
        self.device = device
        self.verbose = verbose
        self.streams = get_streams(device)
        self.prefetch_distance = int(prefetch_distance)
        self.max_cached_layers = int(max_cached_layers)

        # Optional fragmentation monitoring
        self.monitor_fragmentation = monitor_fragmentation
        self.memory_stats: List[Dict] = []
        self.fragmentation_threshold = 0.3

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

        # (Optional) Warm up a few layers to reduce initial latency
        if warmup_layers > 0:
            warm = list(range(min(warmup_layers, len(self.blocks))))
            self.prefetch(warm)
            if self.verbose:
                print(f"[WSM] warmup prefetch: {warm}")

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

    def _ensure_param_on_gpu(self, p: torch.nn.Parameter):
        """Stage parameter to GPU on the weight H2D stream and switch p.data to the GPU tensor."""
        nvtx.range_push("param_h2d")
        if p.device.type != "cpu":
            nvtx.range_pop()
            return

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

    def _ensure_module_on_gpu(self, m: nn.Module):
        """Ensure all params/buffers of module m are on GPU."""
        for p in m.parameters(recurse=True):
            self._ensure_param_on_gpu(p)
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
            # Ensure current layer ready on GPU, then prefetch next ones (fire-and-forget).
            self.ensure_on_gpu(idx, wait=True)
            if self.prefetch_distance > 0:
                nxt = [j for j in range(idx + 1, min(idx + 1 + self.prefetch_distance, len(self.blocks)))]
                self.prefetch(nxt)
        return _pre_hook

    def ensure_on_gpu(self, idx: int, wait: bool):
        """Ensure layer idx is present on GPU (respecting LRU); optionally wait for readiness."""
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
                    self._ensure_module_on_gpu(mod)
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
                    self._ensure_module_on_gpu(mod)
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
                self._ensure_module_on_gpu(module)
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
