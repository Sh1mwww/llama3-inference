import os
import queue
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
        
ATTN_GROUP = (
    "attention.wq.weight", "attention.wk.weight",
    "attention.wv.weight", "attention.wo.weight",
)
FFN_GROUP  = (
    "feed_forward.w1.weight", "feed_forward.w2.weight",
    "feed_forward.w3.weight",
)
GROUPS = {"attn": ATTN_GROUP, "ffn": FFN_GROUP}

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
                 prefetch_distance: int = 4,
                 max_cached_layers: int = 4,
                 layers_attr: str = "layers",
                 warmup_layers: int = 3,
                 verbose: bool = False,
                 monitor_fragmentation: bool = False,
                 ssd_manifest_path: Optional[str] = None,
                 cpu_cache_layers: int = 50,
                 staging_mb: int = 64):
        assert torch.cuda.is_available(), "CUDA not available"
        self.model = model
        self.device = device
        self.verbose = verbose

        # 诊断：打印传入的 device 参数
        print(f"[WSM] Initialized with device={device} (type={type(device)})")
        
        self.streams = get_streams(device)
        self.prefetch_distance = int(prefetch_distance)
        # self.max_cached_layers = int(max_cached_layers)
        # self.cpu_cache_layers = int(cpu_cache_layers)
        self.gpu_cache_max = max(1, int(max_cached_layers))
        self.cpu_cache_max = max(1, int(cpu_cache_layers))
        self.max_cached_layers = self.gpu_cache_max
        self.cpu_cache_layers  = self.cpu_cache_max
        self.warmup_layers = max(0, int(warmup_layers))

        # —— 运行状态 ——
        self.n_layers = len(getattr(model, "layers", []))
        self.grouped_mode = True  # 供 SelfAttention/FFN 走组级 API
        self._anchor = 0          # 计算锚点（EncoderBlock.forward 会持续更新）
        self._anchor_lock = threading.Lock()

        
        # retain policy to avoid evicting "now" or "soon" layers
        self._retain_set = set()               # layers to avoid evicting
        self._last_touch = {}                  # layer_id -> monotonic seconds
        self._min_retain_ms = 50               # default 50ms; tune via runtime cfg if needed


        # SSD backend configuration
        self.ssd_enabled = ssd_manifest_path is not None
        self.ssd_manifest = None
        self.ssd_dio = None
        self.staging_buffer = None
        self.layers_params = {}

        # CPU cache for SSD->CPU->GPU pipeline
        self.cpu_cache: OrderedDict[int, Dict[str, torch.Tensor]] = OrderedDict()
        self.cpu_cache_lock = threading.Lock()

        # # CPU LRU tracking: 维护已缓存层集合 + LRU 队列
        # self._cpu_cached_layers = set()
        # self._cpu_lru = []  # 存 layer_idx，最近使用移到末尾
        # GPU/CPU 缓存结构（LRU）
        self._gpu_layers_lru = OrderedDict()   # {layer_idx: timestamp}
        self._cpu_layers_lru = OrderedDict()   # {layer_idx: timestamp}
        self._cpu_cached_layers = set()        # membership 快速判断
        self._cpu_lru: list[int] = []  # LRU: oldest at front, newest at end


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
        
        
        
        
        # --- thread & queue state ---
        self._stopped: bool = False
        self._stop_event: threading.Event = threading.Event()
        self._threads: list[threading.Thread] = []
                
        self._cpu_lock = threading.RLock()
        self._epoch = 0
        self._cpu_pf_q: "queue.Queue[tuple[int, int|None]]" = queue.Queue(maxsize=2048)
        self._inflight_cpu_layers = set()
        
        # ------------- GPU group tracking -------------
        self._gpu_group_inflight: dict[tuple[int, str], threading.Event] = {}
        self._gpu_group_in_use: set[tuple[int,str]] = set()   # 可选，已有则复用
        self._group_ready_events: dict[tuple[int,str], torch.cuda.Event] = {}
        
        # ------------- H2D 并发闸门（组级）-------------
        self._h2d_groups_max: int = int(os.getenv("WSM_H2D_GROUP_BACKLOG_MAX", "1"))  # 强烈建议=1
        self._h2d_sem = threading.Semaphore(self._h2d_groups_max)

        # ------------- GPU 内存余量守卫 -------------
        self._gpu_free_guard_mb: int = int(os.getenv("WSM_GPU_FREE_GUARD_MB", "1024"))  # 1GB 保护
        self._gpu_max_groups: int = int(os.getenv("WSM_GPU_MAX_GROUPS", "3"))           # 总组上限（强烈建议≤3）


        t = threading.Thread(target=self._cpu_prefetch_worker, name="wsm_cpu_pf", daemon=True)
        self._threads.append(t)
        t.start()

        
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
        
        self.grouped_mode = True  # 开启"组级"模式
        self.gpu_max_groups = int(os.getenv("WSM_GPU_MAX_GROUPS", "4"))  # 建议 3~5；16GB 卡优先设 3~4 更稳
        self.cpu_prefetch_distance = int(os.getenv("WSM_CPU_PREFETCH_DISTANCE", "50"))  # CPU 端预取窗口
        self.cpu_cache_cap_layers  = int(os.getenv("WSM_CPU_CACHE_CAP_LAYERS",  "50"))  # 硬上限
        self.cpu_cache_hwm_layers  = int(os.getenv("WSM_CPU_CACHE_HWM_LAYERS",  "55"))  # 高水位
        self.gpu_free_guard_mb = int(os.getenv("WSM_GPU_FREE_GUARD_MB", "1024"))

        # 滑动窗口 + 回滞参数
        # ★ 关键修复: cpu_cache_cap 应该使用 cpu_cache_layers 参数
        env_cap = os.getenv("WSM_CPU_CACHE_CAP_LAYERS")
        if env_cap is not None:
            self.cpu_cache_cap = int(env_cap)
            print(f"[WSM] Using WSM_CPU_CACHE_CAP_LAYERS from env: {self.cpu_cache_cap}")
        else:
            self.cpu_cache_cap = cpu_cache_layers
            print(f"[WSM] Using cpu_cache_layers parameter: {self.cpu_cache_cap}")

        self.cpu_hwm       = int(os.getenv("WSM_CPU_CACHE_HWM_LAYERS", str(self.cpu_cache_cap + 5)))
        self.cpu_lwm       = int(os.getenv("WSM_CPU_CACHE_LWM_LAYERS", str(max(2, self.cpu_cache_cap - 5))))
        self.cpu_back_margin = int(os.getenv("WSM_CPU_BACK_MARGIN", "4"))  # 留一点历史
        self.cpu_win_base  = 0  # 滑动窗口起点（层号）
        self._warm_done = False  # 预热幂等标志

        print(f"[WSM] CPU cache config: cap={self.cpu_cache_cap}, hwm={self.cpu_hwm}, lwm={self.cpu_lwm}")

        # 激进预取模式：在 hook 中自动预取下 N 层（与计算重叠，提升 GPU 利用率）
        self.aggressive_gpu_prefetch = int(os.getenv("WSM_AGGRESSIVE_GPU_PREFETCH", "2"))  # 默认预取后 2 层

        # 保存原始的prefetch_distance，根据模式选择使用
        self._original_prefetch_distance = self.prefetch_distance
        if self.grouped_mode:
            # 组级模式：关闭整层预取，改用组级预取
            self._layer_prefetch_distance = 0
            self._group_prefetch_distance = 2  # 预取2组（当前层ffn + 下一层attn）
        else:
            # 层级模式：保持原有逻辑
            self._layer_prefetch_distance = self.prefetch_distance
            self._group_prefetch_distance = 0

        self._gpu_group_lru = []    # [(layer_idx, 'attn'/'ffn'), ...] 维护在卡上的组
        self._gpu_groups_in_use = set()  # 正在计算的组，防止被踢掉

        # ★ 修复 5: 去重 - 防止重复加载同一层/组
        self._inflight_cpu_layers = set()       # 正在加载到 CPU 的层
        self._inflight_gpu_groups = set()       # 正在加载到 GPU 的组 (layer, kind)
        self._inflight_lock = threading.Lock()  # 保护 inflight 集合

        # ★ 修复 6: 窗口驱动的 Prefetch Cursor（有序加载）
        self._cpu_pf_cursor = 0                 # CPU 预取游标
        self._gpu_pf_cursor_attn = 0            # GPU attn 组预取游标
        self._gpu_pf_cursor_ffn = 0             # GPU ffn 组预取游标

        # ★ 修复 7: Resident 模块预算（防止碎片和 OOM）
        self.resident_budget_gb = float(os.getenv("WSM_RESIDENT_BUDGET_GB", "3.0"))  # 默认 3GB
        self.resident_max_modules = int(os.getenv("WSM_RESIDENT_MAX_MODULES", "200"))  # 默认最多 200 个（足够 80 层 * 2 norm）
        self._resident_bytes_used = 0  # 已使用的 resident 预算

        # ★ 修复 8: KV I/O 带宽仲裁
        self.kv_throttle_threshold = int(os.getenv("WSM_KV_THROTTLE_THRESHOLD", "3"))  # H2D backlog 阈值
        self.kv_throttle_ms = int(os.getenv("WSM_KV_THROTTLE_MS", "50"))  # throttle 时长（毫秒）
        self._h2d_pending_count = 0  # weight_h2d stream 待处理事件数
        self._last_kv_throttle_time = 0  # 上次 throttle KV 的时间
        
        
        # 在 __init__ 结尾附近、其它配置旁边加：
        self.cpu_rolling_mode   = (os.getenv("WSM_CPU_ROLLING_MODE",  "1") == "1")   # 开启“层层滚动”
        self.cpu_wrap_around    = (os.getenv("WSM_CPU_WRAP_AROUND",   "1") == "1")   # 支持下一轮回到 L0
        self.cpu_roll_stride    = int(os.getenv("WSM_CPU_ROLL_STRIDE","1"))          # 每次右移几层，默认 1
        self.cpu_roll_sync      = (os.getenv("WSM_CPU_ROLL_SYNC",     "1") == "1")   # 触发后同步确保窗口（简单可靠）

        # 独立 H2D stream（仅在CUDA设备上创建）
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            self._copy_stream = torch.cuda.Stream(device=self.device)
        else:
            self._copy_stream = None  # CPU模式下不需要独立stream
        

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
        # self.n_layers = len(self.blocks)  # 添加n_layers属性

        # Per-layer streaming units
        self.block_mods: Dict[int, Dict[str, nn.Module]] = {
            i: _collect_block_modules(b) for i, b in enumerate(self.blocks)
        }

        # Update target layers based on actual block count
        self.target_cpu_layers = min(self.target_cpu_layers, len(self.blocks))
        self.target_gpu_layers = min(self.target_gpu_layers, len(self.blocks))

        # 构建参数名到Parameter对象的映射（用于组级预取）
        self.name_to_param: Dict[str, nn.Parameter] = {}
        # 构建参数归属模块映射：name -> (module_ref, attr_name)
        self.param_owner: Dict[str, tuple] = {}

        for layer_idx, block in enumerate(self.blocks):
            for param_name, param in block.named_parameters():
                full_name = f"layers.{layer_idx}.{param_name}"
                self.name_to_param[full_name] = param

        # 构建 param_owner 映射
        for module_name, module in self.model.named_modules():
            for attr, p in module.named_parameters(recurse=False):
                full = f"{module_name}.{attr}" if module_name else attr
                self.param_owner[full] = (module, attr)

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

    # ---- Retention helpers --------------------------------------------------
    def _refresh_retain_window(self, current_idx: int):
        now = time.monotonic()
        self._last_touch[current_idx] = now
        hi = min(current_idx + max(1, self._pd_current) + 1, len(self.blocks))
        self._retain_set = set(range(current_idx, hi))

    def _should_retain(self, layer_id: int) -> bool:
        if layer_id in self._retain_set:
            return True
        last = self._last_touch.get(layer_id)
        if last is None:
            return False
        return (time.monotonic() - last) < (self._min_retain_ms / 1000.0)
    
    
    
    def _free_gpu_mem_bytes(self) -> int:
        # 兜底：如果 device 不是 CUDA，返回一个很大的值（避免 OOM 检查失败）
        if not str(self.device).startswith("cuda"):
            if self.verbose:
                print(f"[WSM] Warning: _free_gpu_mem_bytes called with non-CUDA device: {self.device}")
            return 100 * 1024**3  # 返回 100GB 作为占位
        free, total = torch.cuda.mem_get_info(self.device)
        return int(free)

    def _ensure_gpu_headroom(self, required_bytes: int, exclude: set[tuple[int,str]] | None = None):
        """确保有 enough_free ≥ required + guard，不够则逐出（排除 in_use 与 exclude）。"""
        guard = self._gpu_free_guard_mb * 1024 * 1024
        exclude = exclude or set()
        tries = 0
        while True:
            free_now = self._free_gpu_mem_bytes()
            if free_now >= required_bytes + guard:
                return
            # 逐出一个 LRU 组（跳过 in_use 与 exclude）
            if not self._evict_one_group_from_gpu(exclude=exclude):
                # 再清一次缓存；仍不够就抛
                torch.cuda.empty_cache()
                free_now = self._free_gpu_mem_bytes()
                if free_now >= required_bytes + guard:
                    return
                raise torch.cuda.OutOfMemoryError(
                    f"insufficient headroom: need={required_bytes/2**20:.2f}MB "
                    f"free={free_now/2**20:.2f}MB guard={guard/2**20:.2f}MB")
            torch.cuda.empty_cache()
            tries += 1
            if tries > 64:
                raise torch.cuda.OutOfMemoryError("eviction loop exceeded")

    
    

    def _evict_one_not_retained(self):
        # Try oldest→newest; evict the first not "retained recently"
        for old in list(self.gpu_cache.keys()):
            if not self._should_retain(old):
                self.gpu_cache.pop(old, None)
                self._evict_layer_to_cpu(old)
                if self.verbose:
                    print(f"[WSM]   evict layer={old}")
                return old
        # Fallback: all are retained; evict LRU anyway (warn)
        old, _ = self.gpu_cache.popitem(last=False)
        if self.verbose:
            print(f"[WSM][WARN] retain window full; evict LRU layer={old}")
        self._evict_layer_to_cpu(old)
        return old

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

    def _advance_cpu_window(self, cur_layer: int):
        """
        只前移，不后退；确保当前层在窗口内
        ★ 修复: 窗口应该包含当前层，而不是滞后
        """
        # 计算窗口基准：确保 cur_layer 在窗口内，且尽量靠前
        # window = [base, base+cap-1]
        # 我们希望 cur_layer 在窗口前部（留出预取空间）
        target_base = max(0, cur_layer - self.cpu_back_margin)

        # 但如果当前层已经超出窗口右端，必须推进窗口
        L0, L1 = self._target_cpu_window()
        if cur_layer > L1:
            # 当前层超出窗口，强制推进
            target_base = max(target_base, cur_layer - self.cpu_cache_cap + 1)
            print(f"[WSM DEBUG] Current layer {cur_layer} > window end {L1}, forcing window advance")

        old_base = self.cpu_win_base
        if target_base > self.cpu_win_base:
            self.cpu_win_base = target_base
            print(f"[WSM DEBUG] _advance_cpu_window: cur_layer={cur_layer}, "
                  f"window_base {old_base} -> {self.cpu_win_base} (target={target_base})")
        else:
            print(f"[WSM DEBUG] _advance_cpu_window: cur_layer={cur_layer}, "
                  f"window_base={self.cpu_win_base} unchanged (target={target_base})")

    def _target_cpu_window(self):
        """
        返回当前滑动窗口的范围 [L0, L1]
        """
        L0 = self.cpu_win_base
        L1 = min(self.n_layers - 1, self.cpu_win_base + self.cpu_cache_cap - 1)
        return L0, L1

    def _ensure_cpu_window(self):
        """
        确保滑动窗口内的层都已加载到 CPU cache
        ★ 关键修复: 先逐出窗口外的层，再加载缺失层，保持容量恒定
        """
        L0, L1 = self._target_cpu_window()

        # ★ DEBUG: 打印窗口和游标状态
        print(f"[WSM DEBUG] _ensure_cpu_window: window=[{L0}, {L1}], cursor={self._cpu_pf_cursor}, "
              f"win_base={self.cpu_win_base}, cache_size={len(self.cpu_cache)}")

        # ★ 关键修复: 先清理窗口外的层（主动逐出）
        layers_to_evict = []
        for layer_id in list(self.cpu_cache.keys()):
            if layer_id < L0 or layer_id > L1:
                layers_to_evict.append(layer_id)

        if layers_to_evict:
            print(f"[WSM DEBUG] Evicting {len(layers_to_evict)} layers outside window [{L0}, {L1}]")
            print(f"[WSM DEBUG] Layers to evict: {sorted(layers_to_evict)}")
            print(f"[WSM DEBUG] Current cache keys: {sorted(list(self.cpu_cache.keys()))}")
            for layer_id in layers_to_evict:
                with self.cpu_cache_lock:
                    self.cpu_cache.pop(layer_id, None)
                if layer_id in self._cpu_lru:
                    self._cpu_lru.remove(layer_id)
                self._cpu_cached_layers.discard(layer_id)

        # ★ 修复: 游标只能在窗口内移动
        if self._cpu_pf_cursor > L1:
            print(f"[WSM DEBUG] Cursor {self._cpu_pf_cursor} > window end {L1}, resetting to {L0}")
            self._cpu_pf_cursor = L0
        else:
            old_cursor = self._cpu_pf_cursor
            self._cpu_pf_cursor = max(self._cpu_pf_cursor, L0)
            if old_cursor != self._cpu_pf_cursor:
                print(f"[WSM DEBUG] Advanced cursor from {old_cursor} to {self._cpu_pf_cursor}")

        # 统计需要加载的层
        missing_layers = []
        for L in range(L0, L1 + 1):
            if L not in self.cpu_cache:
                missing_layers.append(L)

        if missing_layers:
            print(f"[WSM DEBUG] Need to load {len(missing_layers)} missing layers: {missing_layers[:5]}...")

        # 按序加载缺失层（从游标位置开始）
        for L in range(self._cpu_pf_cursor, L1 + 1):
            if L not in self.cpu_cache:
                # ★ 关键: 加载前检查容量，必要时先腾出空间
                while len(self.cpu_cache) >= self.cpu_cache_cap:
                    # 容量已满，踢掉一个最老的层（LRU）
                    if self._cpu_lru:
                        evict_layer = self._cpu_lru[0]
                        print(f"[WSM DEBUG] Cache full ({len(self.cpu_cache)}/{self.cpu_cache_cap}), evicting layer {evict_layer}")
                        with self.cpu_cache_lock:
                            self.cpu_cache.pop(evict_layer, None)
                        self._cpu_lru.pop(0)
                        self._cpu_cached_layers.discard(evict_layer)
                    else:
                        break

                self._load_layer_to_cpu(L)
                self._cpu_pf_cursor = L + 1

        # 如果窗口内所有层都已加载，游标推进到窗口末尾+1
        if self._cpu_pf_cursor <= L1:
            self._cpu_pf_cursor = L1 + 1

        print(f"[WSM DEBUG] _ensure_cpu_window done: cache_size={len(self.cpu_cache)}, cursor={self._cpu_pf_cursor}")

    def _evict_cpu_layers(self, k: int):
        """
        优先踢窗口外层；若不得不踢窗口内，则同步平移窗口并列出右端 must-fetch
        """
        L0, L1 = self._target_cpu_window()
        evicted = 0
        must_fetch = 0
        i = 0

        if self.verbose:
            print(f"[WSM] Evicting {k} CPU layers, window=[{L0}, {L1}], cache_size={len(self.cpu_cache)}")

        # Phase 1: 优先踢窗口外的层（LRU 顺序）
        while evicted < k and i < len(self._cpu_lru):
            L = self._cpu_lru[i]
            # 窗口外 → 可以踢
            if L < L0 or L > L1:
                with self.cpu_cache_lock:
                    self.cpu_cache.pop(L, None)
                self._cpu_lru.pop(i)
                self._cpu_cached_layers.discard(L)
                evicted += 1
                if self.verbose:
                    print(f"[WSM] CPU cache evict (out of window): layer {L}")
                continue
            # 窗口内 → 暂跳过（避免 thrash）
            i += 1

        # Phase 2: 如果还不够，平移窗口（右移 d 层），并把右端 d 层标记为 must-fetch
        if evicted < k:
            d = min(k - evicted, self.cpu_cache_cap)  # 最多平移一个窗口宽
            self.cpu_win_base += d
            must_fetch = d
            if self.verbose:
                print(f"[WSM] Window shift: base {self.cpu_win_base - d} -> {self.cpu_win_base}, must_fetch={must_fetch}")

        # 平移后确保窗口（含 must-fetch）
        if must_fetch > 0:
            self._ensure_cpu_window()

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

    # NOTE: _shrink_cpu_cache_if_needed 已被 _evict_cpu_layers 替代（滑动窗口模式）

    def _load_layer_to_cpu(self, layer_idx: int):
        """Load layer weights from SSD to CPU cache"""
        print(f"[WSM DEBUG] _load_layer_to_cpu called for layer {layer_idx}, ssd_enabled={self.ssd_enabled}, in_cache={layer_idx in self.cpu_cache}")
        if not self.ssd_enabled or layer_idx in self.cpu_cache:
            print(f"[WSM DEBUG] _load_layer_to_cpu skipping layer {layer_idx}: ssd_enabled={self.ssd_enabled}, in_cache={layer_idx in self.cpu_cache}")
            return

        # ★ 修复 5: 去重 - 检查是否已在加载中
        with self._inflight_lock:
            if layer_idx in self._inflight_cpu_layers:
                if self.verbose:
                    print(f"[WSM] Layer {layer_idx} already inflight to CPU, skipping duplicate load")
                return
            # 标记为加载中
            self._inflight_cpu_layers.add(layer_idx)

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
                print(f"[WSM ERROR] Layer {layer_idx} not found in layers_params! Available layers: {list(self.layers_params.keys())[:10]}")
                nvtx.range_pop()
                return

            print(f"[WSM DEBUG] Loading layer {layer_idx} from SSD: {len(self.layers_params[layer_idx])} params in manifest")
            layer_weights = {}

            # Load stream weights for this layer
            # Use a set to track loaded params to avoid duplicates
            loaded_params = set()

            stream_count = sum(1 for p in self.layers_params[layer_idx] if p["policy"] == "stream")
            print(f"[WSM DEBUG] Layer {layer_idx}: {stream_count} stream params to load")

            for param_info in self.layers_params[layer_idx]:
                if param_info["policy"] != "stream":
                    continue

                try:
                    # Import here to avoid circular imports
                    from .weights_io_ssd_dram import DTYPE_MAP

                    param_name = param_info["name"]

                    # Skip if already loaded (manifest may have duplicates)
                    if param_name in loaded_params:
                        continue
                    loaded_params.add(param_name)

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
                    print(f"[WSM DEBUG] ✓ Loaded {param_name}: {param_tensor.shape} {param_tensor.dtype}")

                except Exception as e:
                    print(f"[WSM ERROR] ✗ Failed to load {param_info.get('name', 'unknown')}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            if layer_weights:
                self.cpu_cache[layer_idx] = layer_weights
                # 更新 LRU 跟踪
                self._cpu_cached_layers.add(layer_idx)
                if layer_idx in self._cpu_lru:
                    self._cpu_lru.remove(layer_idx)
                self._cpu_lru.append(layer_idx)
                print(f"[WSM] ✅ Loaded layer {layer_idx} to CPU cache ({len(layer_weights)} params)")

                # NOTE: 收缩检查已移至 _ensure_cpu_window，此处不再需要
            else:
                print(f"[WSM ERROR] ❌ Failed to load ANY weights for layer {layer_idx}!")
                print(f"[WSM ERROR]    Total params in manifest: {len(self.layers_params[layer_idx])}")
                print(f"[WSM ERROR]    Stream params: {stream_count}")
                print(f"[WSM ERROR]    Successfully loaded: 0")

        nvtx.range_pop()

        # ★ 修复 5: 移除 inflight 标记
        with self._inflight_lock:
            self._inflight_cpu_layers.discard(layer_idx)

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
        safety_margin = 0.25  # align with KVOffloader
        # peak concurrent: current H2D + one next prefetch layer
        peak_concurrent_bytes = avg_layer_size * 2 if stream_layers_count > 0 else 0
        required_with_margin = (required_cache_bytes + peak_concurrent_bytes) * (1 + safety_margin)

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


    def note_compute_advance(self, cur_layer: int):
        """
        由 EncoderBlock.forward 在每层入口调用。
        滚动模式下：把 CPU 窗口右移（默认 1 层），从而：
        - 逐出窗口左端（例如刚刚进入计算的层）
        - 把窗口右端的新层从 SSD→DRAM 预取进来
        """
        # 没开 SSD 后端或没启用滚动，就不做事，仍更新保留窗口用于 GPU LRU
        self._refresh_retain_window(cur_layer)

        if not self.ssd_enabled or not self.cpu_rolling_mode:
            return

        # 允许的 window 基准范围：[0 .. max_base]
        max_base = max(0, self.n_layers - self.cpu_cache_cap)
        # 目标基准：当前层后一格（或 stride）
        next_base = cur_layer + self.cpu_roll_stride

        if self.cpu_wrap_around:
            # 到了最右侧窗口 [max_base, n-1] 的最右端，再“新的一轮”会回到 0
            if next_base > max_base:
                next_base = 0
        else:
            next_base = min(next_base, max_base)

        if next_base == self.cpu_win_base:
            return  # 窗口没变化，避免无意义工作

        old_base = self.cpu_win_base
        self.cpu_win_base = next_base
        if self.verbose:
            print(f"[WSM DEBUG] note_compute_advance: base {old_base} -> {self.cpu_win_base} (cur={cur_layer})")

        if self.cpu_roll_sync:
            # 简单可靠：立刻确保新窗口，**同步**触发“左端逐出 + 右端加载”
            self._ensure_cpu_window()
        else:
            # 低阻塞版本：只推进 epoch + 投递后台加载，由线程 _cpu_prefetch_worker 负责 I/O
            self._advance_cpu_window_by_compute(cur_layer)



    def _advance_cpu_window_by_compute(self, cur_layer: int):
        """
        仅由“计算线程”调用：推进 CPU 预取窗口并把缺失层入队。
        不做任何同步 IO（不直接 _load_layer_to_cpu）。
        """
        with self._cpu_lock:
            # 只前移，不后退；保留一点历史可由你现有逻辑决定
            # 这里简单 bump epoch，让过期的预取任务自动被丢弃
            self._epoch += 1

            
            L0 = self.cpu_win_base
            L1 = min(self.n_layers - 1, L0 + self.cpu_cache_cap - 1)
            missing = [L for L in range(L0, L1 + 1)
                    if 0 <= L < self.n_layers and (L not in self.cpu_cache) and (L not in self._inflight_cpu_layers)]

            for L in missing:
                self._inflight_cpu_layers.add(L)
                self._cpu_pf_q.put((self._epoch, L))



    def _schedule_cpu_prefetch(self, current_layer: int):
        """
        滑动窗口预取：只在当前层接近窗口末尾时推进窗口
        ★ 关键修复: 窗口应该平滑滑动，而不是跳跃式推进
        """
        if not self.ssd_enabled:
            return

        L0, L1 = self._target_cpu_window()

        # ★ 修复: 只在当前层超出窗口或接近末尾时，推进窗口 1 层
        # 这样窗口会平滑滑动：[0,49] → [1,50] → [2,51] → ...
        if current_layer > L1 or (current_layer >= L1 - 5):
            # 计算新的窗口基准：确保当前层在窗口内，但只推进必要的量
            if current_layer > L1:
                # 当前层已超出窗口，推进到刚好包含当前层
                new_base = current_layer - self.cpu_cache_cap + 1
            elif current_layer >= L1 - 5:
                # 当前层接近窗口末尾，推进 1 层
                new_base = self.cpu_win_base + 1
            else:
                new_base = self.cpu_win_base

            new_base = max(0, new_base)

            if new_base > self.cpu_win_base:
                print(f"[WSM DEBUG] Layer {current_layer} near/beyond window end {L1}, advancing base {self.cpu_win_base} -> {new_base}")
                self.cpu_win_base = new_base

        # 确保窗口内的层都已加载
        self._ensure_cpu_window()

    def _touch_cpu_layer(self, layer_idx: int):
        """
        标记某层最近被使用，更新 LRU 队列
        应在层的 forward 开始时调用（MHA 或 FFN 入口）
        """
        if layer_idx in self._cpu_lru:
            self._cpu_lru.remove(layer_idx)
            self._cpu_lru.append(layer_idx)

    def wait_for_preload_ready(self, timeout: float = 300.0) -> bool:
        """
        等待预加载完成：GPU有target_gpu_layers层，CPU有target_cpu_layers层
        ★ 修复 9: 支持跳过等待，允许边跑边滚动预取

        Args:
            timeout: 最大等待时间（秒）

        Returns:
            bool: 是否在超时前完成预加载
        """
        import time

        # ★ 修复 9: 检查环境变量，允许跳过预加载等待
        skip_wait = os.getenv("WSM_SKIP_PRELOAD_WAIT", "0") == "1"
        if skip_wait:
            if self.verbose:
                print("[WSM] ⚡ WSM_SKIP_PRELOAD_WAIT=1: Skipping preload wait, will prefetch on-the-fly")
            return True

        # 从环境变量读取 timeout（允许缩短）
        timeout_override = os.getenv("WSM_PRELOAD_TIMEOUT")
        if timeout_override is not None:
            try:
                timeout = float(timeout_override)
                if self.verbose:
                    print(f"[WSM] Using custom timeout from WSM_PRELOAD_TIMEOUT: {timeout}s")
            except ValueError:
                pass

        if self.verbose:
            print(f"[WSM] Waiting for preload: target GPU={self.target_gpu_layers}, target CPU={self.target_cpu_layers}, timeout={timeout}s")

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

        # ★ 修复 9: 超时时给出建议
        print(f"[WSM] ⚠️  Preload timeout after {timeout}s: GPU {len(self.gpu_cache)}/{self.target_gpu_layers}, CPU {len(self.cpu_cache)}/{self.target_cpu_layers}")
        print(f"[WSM] 💡 Tip: Set WSM_SKIP_PRELOAD_WAIT=1 to skip waiting and prefetch on-the-fly")
        print(f"[WSM] 💡 Or set WSM_PRELOAD_TIMEOUT=<seconds> to adjust timeout")
        return False

    # -------- CPU/GPU movement primitives --------

    def _setup_resident_norms(self):
        """
        Move norm modules to GPU and exclude them from streaming/eviction.
        ★ 修复 7: 遵守预算上限，防止碎片和 OOM
        """
        if self.verbose:
            print("[WSM] Setting up resident norm layers...")
            print(f"[WSM] Resident budget: {self.resident_budget_gb:.1f} GB, max {self.resident_max_modules} modules")

        norm_count = 0
        skipped_count = 0
        budget_bytes = int(self.resident_budget_gb * 1024**3)

        for layer_id, modules_dict in self.block_mods.items():
            for module_name, module in modules_dict.items():
                if module_name.startswith("norm_"):
                    # 检查预算
                    if norm_count >= self.resident_max_modules:
                        skipped_count += 1
                        if self.verbose and skipped_count <= 5:
                            print(f"[WSM] Skipping {module_name} (layer {layer_id}): max modules reached")
                        continue

                    # 计算模块大小
                    module_bytes = sum(
                        p.numel() * p.element_size()
                        for p in module.parameters()
                    )

                    if self._resident_bytes_used + module_bytes > budget_bytes:
                        skipped_count += 1
                        if self.verbose and skipped_count <= 5:
                            print(f"[WSM] Skipping {module_name} (layer {layer_id}): budget exhausted "
                                  f"({self._resident_bytes_used/(1024**3):.2f}/{self.resident_budget_gb:.1f} GB)")
                        continue

                    try:
                        module.to(self.device)
                        self._resident_bytes_used += module_bytes
                        norm_count += 1
                        if self.verbose and norm_count <= 10:
                            print(f"[WSM] Layer {layer_id}: {module_name} -> {self.device} "
                                  f"({module_bytes/(1024**2):.1f} MB, resident)")
                    except Exception as e:
                        if self.verbose:
                            print(f"[WSM] Warning: failed to move {module_name} to {self.device}: {e}")

        if self.verbose:
            print(f"[WSM] {norm_count} norm modules set as GPU resident "
                  f"({self._resident_bytes_used/(1024**3):.2f} GB used)")
            if skipped_count > 0:
                print(f"[WSM] {skipped_count} norm modules kept in DRAM (budget limit)")

    def _check_and_throttle_kv(self):
        """
        ★ 修复 8: 检查 weight_h2d backlog，必要时 throttle KV I/O
        避免 KV 抢带宽导致权重迟迟不上来
        """
        # 检查 weight_h2d stream 是否繁忙
        h2d_stream = getattr(self.streams, "weight_h2d", None) if hasattr(self, "streams") else None
        if h2d_stream is None:
            return

        try:
            # 检查 stream 是否完成（False = 还在忙）
            is_busy = not h2d_stream.query()
        except Exception:
            is_busy = False

        if is_busy:
            self._h2d_pending_count += 1
        else:
            self._h2d_pending_count = max(0, self._h2d_pending_count - 1)

        # 如果 backlog 超过阈值，throttle KV
        if self._h2d_pending_count >= self.kv_throttle_threshold:
            current_time = time.time()
            # 避免过于频繁的 throttle（至少间隔 100ms）
            if current_time - self._last_kv_throttle_time > 0.1:
                if self.kv_offloader is not None and hasattr(self.kv_offloader, "throttle_writes_for"):
                    try:
                        self.kv_offloader.throttle_writes_for(self.kv_throttle_ms)
                        if self.verbose:
                            print(f"[WSM] Throttled KV writes for {self.kv_throttle_ms}ms "
                                  f"(H2D backlog: {self._h2d_pending_count})")
                        self._last_kv_throttle_time = current_time
                    except Exception as e:
                        if self.verbose:
                            print(f"[WSM] KV throttle failed: {e}")

    def _record_layer_ready_event(self, idx: int):
        """Record an event on the weight H2D stream marking all enqueued copies for this layer."""
        if not torch.cuda.is_available() or getattr(self.streams, "weight_h2d", None) is None:
            return
        evt = self.layer_events.get(idx)
        if evt is None:
            evt = torch.cuda.Event(blocking=False)
            self.layer_events[idx] = evt
        evt.record(self.streams.weight_h2d)

        # ★ 修复 8: 检查并 throttle KV
        self._check_and_throttle_kv()

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

        # Light-weight GC of event pool to avoid pending growth
        try:
            from . import stream_mnt
            stream_mnt._get_event_pool(self.device).gc(limit=512)
        except Exception:
            pass


    def _wait_layer_ready(self, idx: int):
        """Wait on the layer's ready event on the current stream (fallback: wait for H2D stream)."""
        evt = self.layer_events.get(idx)
        if self.verbose:
            print(f"[WSM] _wait_layer_ready layer={idx} event={'exists' if evt else 'None'}")
        if evt is not None:
            if self.verbose:
                print(f"[WSM] _wait_layer_ready layer={idx} waiting on event...")
            torch.cuda.current_stream(self.device).wait_event(evt)
            if self.verbose:
                print(f"[WSM] _wait_layer_ready layer={idx} event wait done")
        else:
            if self.verbose:
                print(f"[WSM] _wait_layer_ready layer={idx} fallback to wait_weight_ready_on_current...")
            self.streams.wait_weight_ready_on_current(self.device)
            if self.verbose:
                print(f"[WSM] _wait_layer_ready layer={idx} fallback done")

    def _ensure_param_cpu_stash_inplace(self, p: torch.nn.Parameter):
        """
        Replace p.data with a pinned CPU master copy and stash it by Parameter id.
        This avoids holding duplicate CPU tensors.

        WARNING: This should NOT be called on meta tensors! They should be handled by SSD backend.
        """
        pid = id(p)
        if pid in self.cpu_stash:
            return

        # CPU stash 已废弃，不再使用（CPU stub 方案不需要 stash）
        # 保留空实现，避免破坏现有调用
        pass

    def _ensure_param_on_gpu(self, p: torch.nn.Parameter, layer_idx: Optional[int] = None, param_name: Optional[str] = None):
        """
        确保参数在 GPU 上（从 CPU cache 加载）
        移除了 meta device 支持，参数要么是 0-size CPU stub，要么已在 GPU
        """
        nvtx.range_push("param_h2d")

        # 已经在 GPU 上且非 stub：直接返回
        if p.is_cuda and p.numel() > 0:
             nvtx.range_pop()
             return

        # CPU stub 或 CPU 参数：从 SSD/CPU cache 加载
        if self.ssd_enabled:
            # 确保该层已加载到 CPU cache
            if layer_idx is not None and layer_idx not in self.cpu_cache:
                 # 检查是否正在加载中
                with self._inflight_lock:
                    is_inflight = layer_idx in self._inflight_cpu_layers

                if is_inflight:
                    # 正在被后台线程加载，等待完成
                    try:
                        self._wait_cpu_ready(layer_idx, timeout=10.0)
                    except Exception as e:
                        if self.verbose:
                            print(f"[WSM] Timeout waiting for layer {layer_idx} to load: {e}")
                        nvtx.range_pop()
                        return
                else:
                    # 不在加载中，也不在缓存中，需要立即加载
                    try:
                        self._load_layer_to_cpu(layer_idx)
                    except Exception as e:
                        if self.verbose:
                            print(f"[WSM] Failed to load layer {layer_idx} to CPU cache: {e}")
                        nvtx.range_pop()
                        return

            # 从 CPU cache 加载到 GPU
            if (layer_idx is not None and param_name is not None and
                layer_idx in self.cpu_cache and param_name in self.cpu_cache[layer_idx]):

                nvtx.range_push("cpu_cache_to_gpu")
                with torch.cuda.stream(self.streams.weight_h2d):
                    cached_tensor = self.cpu_cache[layer_idx][param_name]
                    p_gpu = cached_tensor.to(self.device, non_blocking=True)

                # 直接替换 param.data（无需 meta 检查）
                p.data = p_gpu

                nvtx.range_pop()
                nvtx.range_pop()
                return
            else:
                # 无法从 CPU cache 获取，说明这是个 resident 参数，应该已经被加载
                if self.verbose:
                    print(f"[WSM] Warning: param {param_name} not in CPU cache (layer {layer_idx}), skipping")
                nvtx.range_pop()
                return

        # CPU 参数：从 CPU cache（SSD 模式）或 CPU stash（传统模式）加载到 GPU
        if p.device.type == "cpu":
            # SSD 模式：优先使用 CPU cache
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

            # 传统模式：直接将 CPU 参数移动到 GPU
            # 注意：参数已经在 CPU 上且有完整数据，直接 transfer
            if p.numel() > 0:  # 确保不是 stub
                nvtx.range_push("weight_h2d_stream")
                with torch.cuda.stream(self.streams.weight_h2d):
                    nvtx.range_push("cpu_to_gpu_transfer")
                    p_gpu = p.data.to(self.device, non_blocking=True)
                    nvtx.range_pop()
                nvtx.range_pop()
                p.data = p_gpu
            else:
                # 0-size stub：不应该出现在传统模式
                if self.verbose:
                    print(f"[WSM] Warning: encountered 0-size CPU stub in traditional streaming mode")
                nvtx.range_pop()
                return

        nvtx.range_pop()

    def _evict_param_to_cpu(self, p: torch.nn.Parameter):
        """
        驱逐参数出GPU：将 param.data 设为 0-size CPU tensor（stub）
        不再使用 CPU stash，因为我们在构建期就创建了 stub
        """
        # 创建 0-size CPU stub（保持 dtype）
        stub = torch.empty(0, dtype=p.dtype, device="cpu")
        p.data = stub

    def _ensure_module_on_gpu(self, m: nn.Module, layer_idx: Optional[int] = None, module_name: Optional[str] = None):
        """Ensure all params/buffers of module m are on GPU."""
        # For modules, we need to handle parameter replacement differently
        # because meta parameters cannot be assigned via .data =

        params_to_replace = {}

        for local_param_name, p in m.named_parameters(recurse=False):  # Non-recursive to get direct params
            # Build full parameter name for CPU cache lookup
            # module_name is like "wq", "wk", etc. from _get_modules_dict()
            # param_name from named_parameters() is just "weight" or "bias"
            # We need to determine the parent module path (attention or feed_forward)
            full_param_name = None
            if layer_idx is not None and module_name is not None:
                # Determine parent module based on module_name
                if module_name in ["wq", "wk", "wv", "wo"]:
                    parent_module = "attention"
                elif module_name in ["w1", "w2", "w3"]:
                    parent_module = "feed_forward"
                else:
                    parent_module = None

                if parent_module:
                    full_param_name = f"layers.{layer_idx}.{parent_module}.{module_name}.{local_param_name}"
                else:
                    # Fallback for unknown modules
                    full_param_name = f"layers.{layer_idx}.{module_name}.{local_param_name}"

            # Check if this is a meta parameter that needs loading from SSD
            is_meta = (p.device.type == "meta" or getattr(p, "is_meta", False))

            if is_meta and self.ssd_enabled and full_param_name:
                # Try to load from CPU cache
                if layer_idx in self.cpu_cache and full_param_name in self.cpu_cache[layer_idx]:
                    cached_tensor = self.cpu_cache[layer_idx][full_param_name]
                    expected = tuple(getattr(getattr(m, local_param_name), "shape", ()))
                    chosen_name = full_param_name
                    chosen_tensor = cached_tensor

                    if expected and tuple(cached_tensor.shape) != expected:
                        # 在同层同族里找形状能对上的备选
                        sibling_candidates = []
                        def _add(name):
                            t = self.cpu_cache[layer_idx].get(name)
                            if t is not None:
                                sibling_candidates.append((name, t))

                        if module_name in ("wq", "wk", "wv"):
                            for alt in ("wq", "wk", "wv"):
                                _add(f"layers.{layer_idx}.attention.{alt}.{local_param_name}")
                        elif module_name in ("w1", "w2", "w3"):
                            for alt in ("w1", "w2", "w3"):
                                _add(f"layers.{layer_idx}.feed_forward.{alt}.{local_param_name}")

                        for alt_name, t in sibling_candidates:
                            if tuple(t.shape) == expected:
                                chosen_name, chosen_tensor = alt_name, t
                                break
                        else:
                            raise RuntimeError(
                                f"[WSM] SSD manifest shape mismatch for {full_param_name}: "
                                f"expected {expected}, got {tuple(cached_tensor.shape)}. "
                                f"Check MANIFEST <-> params.json/model size."
                            )
                    with torch.cuda.stream(self.streams.weight_h2d):
                        p_gpu = chosen_tensor.to(self.device, non_blocking=True)
                    params_to_replace[local_param_name] = nn.Parameter(p_gpu, requires_grad=p.requires_grad)
                    if self.verbose:
                        print(f"[WSM DEBUG] ✓ Loaded meta param {full_param_name} to GPU: {p_gpu.shape}")
                    if chosen_name != full_param_name and self.verbose:
                        print(f"[WSM] ⚠️ shape-fix: remapped {full_param_name} -> {chosen_name}")
                else:
                    # CPU cache miss - 这是正常的预加载流程，不是错误
                    if self.verbose >= 2:  # 只在详细模式下显示
                        print(f"[WSM DEBUG] Cache miss: {full_param_name} not in CPU cache, loading layer {layer_idx}...")
                    # 尝试立即加载该层
                    if layer_idx not in self.cpu_cache:
                        try:
                            if self.verbose >= 2:  # 只在详细模式下显示
                                print(f"[WSM DEBUG] Triggering on-demand load for layer {layer_idx}...")
                            self._load_layer_to_cpu(layer_idx)
                            # 重试一次
                            if layer_idx in self.cpu_cache and full_param_name in self.cpu_cache[layer_idx]:
                                cached_tensor = self.cpu_cache[layer_idx][full_param_name]
                                with torch.cuda.stream(self.streams.weight_h2d):
                                    p_gpu = cached_tensor.to(self.device, non_blocking=True)
                                params_to_replace[local_param_name] = nn.Parameter(p_gpu, requires_grad=p.requires_grad)
                                if self.verbose:
                                    print(f"[WSM DEBUG] ✓ Loaded meta param {full_param_name} to GPU after retry: {p_gpu.shape}")
                        except Exception as e:
                            if self.verbose:
                                print(f"[WSM ERROR] Failed to load layer {layer_idx}: {e}")
            else:
                # Regular parameter - use standard method
                self._ensure_param_on_gpu(p, layer_idx, full_param_name)

        # Replace meta parameters
        for param_name, new_param in params_to_replace.items():
            # 使用 _parameters 字典直接替换，这是 PyTorch 的内部机制
            m._parameters[param_name] = new_param

        # for b in m.buffers(recurse=True):
        #     if b.device.type == "cpu":
        #         with torch.cuda.stream(self.streams.weight_h2d):
        #             b_gpu = b.detach().to(self.device, non_blocking=True)
        #         try:
        #             b.data = b_gpu
        #         except Exception:
        #             pass
        for b in m.buffers(recurse=True):
        # 对于 meta buffer，先 to_empty(materialize) 再填充；对于已有CPU/GPU buffer，保持原逻辑
            if getattr(b, "is_meta", False):
                try:
                    b = b.to_empty(device=self.device)  # 关键字参数！
                except Exception:
                    pass
            elif b.device.type == "cpu":
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
            # 标记当前层被使用（更新 CPU LRU）
            self._touch_cpu_layer(idx)

            # Schedule CPU prefetch for upcoming layers (滚动窗口)
            self._schedule_cpu_prefetch(idx)

            # 异步预取后续层到 GPU（与当前层计算重叠，提升 GPU 利用率）
            # aggressive_gpu_prefetch 控制预取的层数（默认 2 层）
            if self.aggressive_gpu_prefetch > 0:
                prefetch_targets = []
                for offset in range(1, self.aggressive_gpu_prefetch + 1):
                    next_idx = idx + offset
                    if next_idx < self.n_layers:
                        prefetch_targets.append(next_idx)

                if prefetch_targets:
                    try:
                        self.prefetch(prefetch_targets)  # 异步，不等待
                        if self.verbose:
                            print(f"[WSM] Async GPU prefetch: layers {prefetch_targets}")
                    except Exception as e:
                        if self.verbose:
                            print(f"[WSM] Async prefetch failed for layers {prefetch_targets}: {e}")

            if self.grouped_mode:
                # 组级预取模式
                # 注意：当前层的attn已在forward前由layer调用ensure_group_on_gpu确保
                # 这里主要负责预取当前层的ffn和下一层的attn
                try:
                    # 异步预取当前层的ffn（与attn计算重叠）
                    if hasattr(self, "prefetch_group_async"):
                        self.prefetch_group_async(idx, "ffn")
                        # 异步预取下一层的attn
                        if idx + 1 < self.n_layers:
                            self.prefetch_group_async(idx + 1, "attn")
                    else:
                        # 回退到同步加载
                        if self.verbose:
                            print(f"[WSM] prefetch_group_async not available, using synchronous load")
                        if hasattr(self, "ensure_group_on_gpu"):
                            self.ensure_group_on_gpu(idx, "ffn")
                except Exception as e:
                    if self.verbose:
                        print(f"[WSM] Group prefetch failed in hook for layer {idx}: {e}")
                    # 回退到层级预取
                    self.ensure_on_gpu(idx, wait=True)
            else:
                # 原有的层级预取逻辑
                self.ensure_on_gpu(idx, wait=True)
                if self._layer_prefetch_distance > 0:
                    nxt = [j for j in range(idx + 1, min(idx + 1 + self._layer_prefetch_distance, len(self.blocks)))]
                    self.prefetch(nxt)
        return _pre_hook

    def ensure_on_gpu(self, idx: int, wait: bool):
        """Ensure layer idx is present on GPU (respecting LRU); optionally wait for readiness."""
        # 在正式 H2D 前评估一次 PD

        self._decide_pd()

        nvtx.range_push(f"ensure_layer_{idx}")
        
        # 在 meta 初始化+SSD 模式下，确保有 CPU 层缓存可用（按需从 SSD 拉起）
        if self.ssd_enabled and (idx not in self.cpu_cache):
                try:
                    self._load_layer_to_cpu(idx)
                except Exception:
                    pass
                
        self._record_memory_stats("ensure_start", idx)

        # prepare retention window
        self._refresh_retain_window(idx)
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
                self._record_memory_stats("evict_start", -1)
                self._evict_one_not_retained()
                self._record_memory_stats("evict_end", -1)

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
                # 判断权重从哪里加载
                source = "SSD→CPU→GPU" if (self.ssd_enabled and idx in self.cpu_cache) else \
                         "SSD→GPU" if self.ssd_enabled else \
                         "CPU→GPU"
                print(f"[WSM] ->GPU layer={idx} ({source})")
            if wait:
                nvtx.range_push(f"wait_layer_{idx}")
                self._wait_layer_ready(idx)
                nvtx.range_pop()

            nvtx.range_pop()  # cache_miss

        self._record_memory_stats("ensure_end", idx)
        nvtx.range_pop()  # ensure_layer
        
        
    def _bytes_of(self, t):
        itemsize = {torch.float16:2, torch.bfloat16:2, torch.float32:4, torch.int8:1,
                    torch.uint8:1, torch.int32:4, torch.int64:8, torch.float64:8}.get(t.dtype, 2)
        return t.numel() * itemsize

    def _mark_group_in_use(self, layer_idx: int, group: str):
        """标记组为正在使用，防止在计算期间被踢掉"""
        key = (layer_idx, group)
        self._gpu_groups_in_use.add(key)
        if self.verbose:
            print(f"[WSM] Marked group {key} as IN_USE")

    def _unmark_group_in_use(self, layer_idx: int, group: str):
        """取消组的使用标记"""
        key = (layer_idx, group)
        self._gpu_groups_in_use.discard(key)
        if self.verbose:
            print(f"[WSM] Unmarked group {key} from IN_USE")

    def _evict_one_group_from_gpu(self, exclude=()):
        """
        LRU 淘汰一个组，避开 exclude 和 in_use（当前正在用的组）
        修复：防止踢掉正在计算的组，否则会导致 meta device 错误
        """
        for i, (lyr, grp) in enumerate(self._gpu_group_lru):
            key = (lyr, grp)
            # 跳过 exclude 列表中的组
            if key in exclude:
                continue
            # 跳过正在使用的组（关键修复！）
            if key in self._gpu_groups_in_use:
                if self.verbose:
                    print(f"[WSM] Skipping eviction of IN_USE group {key}")
                continue
            if self._should_retain(lyr):
                continue
            # 释放该组的显存占用
            for suf in GROUPS[grp]:
                name = f"layers.{lyr}.{suf}"
                p = self.name_to_param.get(name)
                if p is None:
                    continue
                # 判定是否在GPU：device 是 cuda 且 numel > 0（非 stub）
                if p.is_cuda and p.numel() > 0:
                    # 驱逐到 CPU（替换为 0-size stub）
                    self._evict_param_to_cpu(p)
            self._gpu_group_lru.pop(i)
            torch.cuda.empty_cache()
            if self.verbose:
                print(f"[WSM] Evicted group {key} from GPU")
            return True
        if self.verbose:
            print(f"[WSM] No groups available for eviction (all in_use or excluded)")
        return False

    def _ensure_gpu_room(self, need_bytes, exclude=()):
        guard = self.gpu_free_guard_mb * 1024 * 1024
        while True:
            free, _ = torch.cuda.mem_get_info(self.device.index)
            if free >= need_bytes + guard and len(self._gpu_group_lru) < self.gpu_max_groups:
                return
            if not self._evict_one_group_from_gpu(exclude=exclude):
                break
        # 再次检查，如果还不够，让上层处理 OOM

    def _install_param_tensor(self, pname: str, dst_gpu_tensor: torch.Tensor):
        """
        把 dst_gpu_tensor 安装到模型的参数 pname 上：
        直接替换 param.data（CPU stub → GPU tensor）
        """
        param = self.name_to_param.get(pname)
        if param is None:
            # 参数不存在，可能是新参数或映射未更新
            if self.verbose:
                print(f"[WSM] Warning: param {pname} not in name_to_param, skipping")
            return

        # 判断是否需要替换参数对象（dtype/shape 不匹配时）
        # 注意：0-size CPU stub 的 shape 是 (0,)，与真实权重不同，所以会触发替换
        need_replace = (
            param.dtype != dst_gpu_tensor.dtype
            or param.shape != dst_gpu_tensor.shape
        )

        if need_replace:
            # 用真正的数据创建一个新的 Parameter 并替换到模块上
            if pname in self.param_owner:
                mod, attr = self.param_owner[pname]
                new_p = nn.Parameter(dst_gpu_tensor, requires_grad=False)
                setattr(mod, attr, new_p)
                self.name_to_param[pname] = new_p

                if self.verbose:
                    old_info = f"{param.device}/{param.dtype}/{param.shape}"
                    new_info = f"{new_p.device}/{new_p.dtype}/{new_p.shape}"
                    print(f"[WSM] Replaced param {pname}: {old_info} -> {new_info}")
            else:
                # Fallback: 直接赋值 data
                param.data = dst_gpu_tensor
                self.name_to_param[pname] = param
        else:
            # 同 dtype/device/shape，走 copy_ 覆盖（常规路径）
            param.data.copy_(dst_gpu_tensor)

    def _move_to_gpu(self, pname: str, src_cpu_tensor: torch.Tensor, exclude: set[tuple[int,str]] | None = None):
        """
        CPU→GPU 搬运 + 安装（使用拷贝 + 替换 param.data 方式）
        移除了 meta device 物化逻辑，改用简单的参数数据替换
        """

        # 1) 计算所需空间并确保 GPU headroom
        need_bytes = src_cpu_tensor.numel() * src_cpu_tensor.element_size()
        self._ensure_gpu_headroom(need_bytes, exclude=exclude)

        # 2) 获取 H2D stream
        h2d_stream = getattr(self.streams, "weight_h2d", None) if hasattr(self, "streams") else None
        if h2d_stream is None:
            h2d_stream = self._copy_stream  # fallback

        # 3) H2D 传输（在 weight_h2d 流中进行）
        try:
            if h2d_stream is not None:
                # CUDA设备：使用 weight_h2d stream（异步，不阻塞）
                with torch.cuda.stream(h2d_stream):
                    dst = src_cpu_tensor.to(self.device, non_blocking=True)
            else:
                # CPU设备或无stream：直接传输
                dst = src_cpu_tensor.to(self.device, non_blocking=False)
        except torch.cuda.OutOfMemoryError:
            # 最后兜底：再逐出一个组、清缓存、重试一次
            if self._evict_one_group_from_gpu(exclude=exclude or set()):
                torch.cuda.empty_cache()
                if h2d_stream is not None:
                    with torch.cuda.stream(h2d_stream):
                        dst = src_cpu_tensor.to(self.device, non_blocking=True)
                else:
                    dst = src_cpu_tensor.to(self.device, non_blocking=False)
            else:
                # 无法逐出任何东西，抛出 OOM
                raise

        # 4) 安装参数到模型（使用保险丝机制）
        self._install_param_tensor(pname, dst)
    
        return dst

    def ensure_group_on_gpu(self, layer_idx: int, group: str):
        """阻塞式：确保 (layer_idx, group) 在 GPU；若后台任务在飞则等待，超时则同步兜底。"""
        wanted = GROUPS[group]

        # 先看是否已有后台任务在飞
        key = (layer_idx, group)
        inflight_evt = self._gpu_group_inflight.get(key)

        if inflight_evt is not None:
            # 等一小段时间，给后台一个完成的机会（避免重复拷贝）
            if not inflight_evt.wait(timeout=3.0):
                if self.verbose:
                    print(f"[WSM] Group {key} inflight timeout; will fallback to sync load")

        # 确保 CPU 层已经在缓存（后台没完成就立刻兜底，但这在确保函数里，接受阻塞）
        try:
            self._wait_cpu_ready(layer_idx, timeout=2.0)
        except Exception:
            # 兜底同步拉层
            self._load_layer_to_cpu(layer_idx)

        layer_cache = self.cpu_cache.get(layer_idx, {})

        # 逐 param 检查/复制到 GPU（走 H2D stream；此处可阻塞）
        for suf in wanted:
            name = f"layers.{layer_idx}.{suf}"
            src = layer_cache.get(name)
            if src is None:
                # 尝试从模型本身获取参数（传统 streaming 模式）
                if not self.ssd_enabled:
                    src = self._get_param_from_model(layer_idx, suf)
                if src is None:
                    # 最后才尝试从 SSD 加载
                    src = self._load_param_from_ssd(name)
            self._move_to_gpu(name, src, exclude={key})

        # 记录 ready 事件
        self._record_layer_ready_event(layer_idx)

        # 更新组 LRU
        if key in self._gpu_group_lru:
            self._gpu_group_lru.remove(key)
        self._gpu_group_lru.append(key)
        while len(self._gpu_group_lru) > self.gpu_max_groups:
            self._evict_one_group_from_gpu(exclude={key})

    # --- inside class WeightStreamingManager ---
    def _read_layer_from_ssd(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Read one full layer's streamable params from SSD into pinned-CPU tensors
        and return {param_name: tensor} without mutating cpu_cache.
        """
        if not self.ssd_enabled:
            raise RuntimeError("SSD backend not enabled")
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise IndexError(f"layer {layer_idx} out of range 0..{self.n_layers-1}")
        if layer_idx not in self.layers_params:
            raise KeyError(f"Layer {layer_idx} not in manifest")

        from .weights_io_ssd_dram import DTYPE_MAP, alloc_pinned_aligned

        layer_weights: Dict[str, torch.Tensor] = {}
        params = self.layers_params[layer_idx]

        for pinfo in params:
            if pinfo.get("policy") != "stream":
                continue

            stride = int(pinfo["stride"])
            offset = int(pinfo["offset"])
            nbytes = int(pinfo["nbytes"])

            # ensure staging buffer big enough
            if stride > len(self.staging_buffer):
                bs = int(self.ssd_manifest["block_size"])
                new_sz = ((stride + bs - 1) // bs) * bs
                self.staging_buffer = alloc_pinned_aligned(new_sz, bs)

            # SSD → staging (pinned)
            self.ssd_dio.pread_into_tensor(self.staging_buffer, stride, offset)

            # materialize pinned CPU tensor with correct shape/dtype
            t = torch.empty(
                tuple(pinfo["shape"]),
                dtype=DTYPE_MAP[pinfo["dtype"]],
                pin_memory=True
            )
            # copy exact bytes
            t.view(-1).view(torch.uint8)[:nbytes].copy_(self.staging_buffer[:nbytes])

            layer_weights[pinfo["name"]] = t

        if not layer_weights:
            raise RuntimeError(f"no stream params loaded for layer {layer_idx}")
        return layer_weights

    # def _cpu_prefetch_worker(self):
    #     while not self._stopped:
    #         try:
    #             epoch, L = self._cpu_pf_q.get(timeout=0.1)
    #         except queue.Empty:
    #             continue

    #         # 读 SSD → 构建临时字典（不持锁）
    #         tmp = self._read_layer_from_ssd(L)

    #         with self._cpu_lock:
    #             if epoch != self._epoch:
    #                 # 窗口已前移，丢弃过期结果
    #                 self._inflight_cpu_layers.discard(L)
    #                 self._cpu_pf_q.task_done()
    #                 continue
    #             self._evict_if_over_hwm_locked(incoming=1)  # 只踢窗口外（见下）
    #             self.cpu_cache[L] = tmp
    #             self._touch_cpu_lru_locked(L)

    #         self._inflight_cpu_layers.discard(L)
    #         self._cpu_pf_q.task_done()
    #         print(f"[WSM] ✅ Loaded layer {L} to CPU cache ({len(tmp)} params)")
    def _cpu_prefetch_worker(self):
        while not (self._stopped or self._stop_event.is_set()):
            try:
                item = self._cpu_pf_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # 支持关停的哨兵
            if item is None or (isinstance(item, tuple) and item[1] is None):
                continue

            epoch, layer_idx = item

            # 读 SSD → 临时字典（不加锁）
            try:
                tmp = self._read_layer_from_ssd(layer_idx)
            except Exception as e:
                # 读取失败，清理 inflight 并继续
                with self._cpu_lock:
                    self._inflight_cpu_layers.discard(layer_idx)
                print(f"[WSM][cpu_pf] SSD read failed for layer {layer_idx}: {e}", flush=True)
                self._cpu_pf_q.task_done()
                continue

            # 落地到 CPU cache（加锁），过期任务丢弃
            with self._cpu_lock:
                if epoch != self._epoch:
                    # 窗口已前移，丢弃过期结果
                    self._inflight_cpu_layers.discard(layer_idx)
                    self._cpu_pf_q.task_done()
                    continue

                # 回滞式收缩：仅踢窗口外（避免抖动）
                self._evict_if_over_hwm_locked(incoming=1)

                self.cpu_cache[layer_idx] = tmp
                # 维护 LRU：最近使用移到末尾
                if layer_idx in self._cpu_lru:
                    self._cpu_lru.remove(layer_idx)
                self._cpu_lru.append(layer_idx)

                self._inflight_cpu_layers.discard(layer_idx)

            self._cpu_pf_q.task_done()
            print(f"[WSM] ✅ Loaded layer {layer_idx} to CPU cache ({len(tmp)} params)")


    def prefetch_group_async(self, layer_idx: int, group: str):
        if layer_idx < 0 or layer_idx >= self.n_layers:
            return

        # 推进窗口 & 入队（不做同步 IO）
        try:
            self._advance_cpu_window_by_compute(layer_idx)
        except Exception:
            pass

        key = (layer_idx, group)
        # 已在飞则返回
        if key in self._gpu_group_inflight:
            return

        evt = threading.Event()
        self._gpu_group_inflight[key] = evt

        def _task():
            try:
                # 等 CPU 层 ready（给个短超时）
                self._wait_cpu_ready(layer_idx, timeout=5.0)

                # 正确命中 DRAM：先按层取，再按名
                layer_cache = self.cpu_cache.get(layer_idx, {})

                for suf in GROUPS[group]:
                    name = f"layers.{layer_idx}.{suf}"
                    src = layer_cache.get(name)
                    if src is None:
                        src = self._load_param_from_ssd(name)  # 兜底
                    self._move_to_gpu(name, src)  # 走 weight_h2d 流

                # 记层级 ready 事件（所有 H2D 入队之后）
                self._record_layer_ready_event(layer_idx)

                # 更新组 LRU，并按上限逐出（保护当前 key）
                if key in self._gpu_group_lru:
                    self._gpu_group_lru.remove(key)
                self._gpu_group_lru.append(key)
                while len(self._gpu_group_lru) > self.gpu_max_groups:
                    self._evict_one_group_from_gpu(exclude={key})

            except Exception as e:
                print(f"[WSM][prefetch_group_async] {layer_idx}/{group} failed: {e}", flush=True)
            finally:
                evt.set()
                self._gpu_group_inflight.pop(key, None)

        # 后台执行
        t = threading.Thread(target=_task, name=f"wsm_gpf_{layer_idx}_{group}", daemon=True)
        self._threads.append(t)
        t.start()

    # def prefetch_group_async(self, layer_idx: int, group: str):
    #     """非阻塞：后台把 (layer_idx, group) 从 SSD→CPU→GPU。失败时要能自愈，不阻塞主线程。"""
    #     if layer_idx < 0 or layer_idx >= self.n_layers:
    #         return

    #     # 先推进窗口并入队 CPU 预取（绝不阻塞）
    #     try:
    #         self._advance_cpu_window_by_compute(layer_idx)
    #     except Exception:
    #         # 兼容旧分支：直接调 schedule（不抛出）
    #         self._schedule_cpu_prefetch(layer_idx)

    #     key = (layer_idx, group)
    #     # 若已有同组任务在飞，直接返回（避免重复）
    #     if key in self._gpu_group_inflight:
    #         return

    #     evt = threading.Event()
    #     self._gpu_group_inflight[key] = evt

    #     def _task():
    #         try:
    #             # 等 CPU 层就绪（内部有超时后“立即加载”的兜底，但那是后台线程，不会堵前向）
    #             self._wait_cpu_ready(layer_idx, timeout=5.0)

    #             # 从 DRAM 取到这一层的参数字典
    #             layer_cache = self.cpu_cache.get(layer_idx, {})

    #             # 逐 param 发起 H2D（走独立 weight_h2d stream，非阻塞）
    #             for suf in GROUPS[group]:
    #                 name = f"layers.{layer_idx}.{suf}"
    #                 src = layer_cache.get(name)
    #                 if src is None:
    #                     # DRAM miss：兜底从 SSD 读“单参数”
    #                     src = self._load_param_from_ssd(name)
    #                 self._move_to_gpu(name, src)  # 内部已用 weight_h2d stream

    #             # 记录这一层的 ready 事件（所有 H2D 入队之后）
    #             self._record_layer_ready_event(layer_idx)

    #             # 更新组 LRU（受限于 gpu_max_groups）
    #             if key in self._gpu_group_lru:
    #                 self._gpu_group_lru.remove(key)
    #             self._gpu_group_lru.append(key)
    #             while len(self._gpu_group_lru) > self.gpu_max_groups:
    #                 self._evict_one_group_from_gpu(exclude={key})

    #         except Exception as e:
    #             # 出错时打印但不要影响前向；最重要的是把 inflight 事件置位，避免主线程死等
    #             print(f"[WSM][prefetch_group_async] {layer_idx}/{group} failed: {e}", flush=True)
    #         finally:
    #             evt.set()
    #             # 清理 inflight 标记
    #             self._gpu_group_inflight.pop(key, None)

    #     # 后台执行
    #     self._bg_submit(_task)


    def _maybe_schedule_cpu_prefetch(self, cur_layer: int):
        # 目标窗口 [cur_layer+1, cur_layer+self.cpu_prefetch_distance]
        # 你已有的 CPU 预取线程可在此更新目标；没有则简单地把该窗口依次 _load_layer_to_cpu()
        target = range(cur_layer+1, min(self.n_layers, cur_layer + 1 + self.cpu_prefetch_distance))
        for L in target:
            with self._cpu_lock:
                # 只允许“计算线程”推进窗口（详见下一节）
                self._advance_cpu_window_by_compute(cur_layer)
                epoch = self._epoch
                L0 = self.cpu_win_base
                L1 = self.cpu_win_base + self.cpu_cache_cap - 1
                for L in range(L0, L1 + 1):
                    if L in self.cpu_cache or L in self._inflight_cpu_layers:
                        continue
                    self._inflight_cpu_layers.add(L)
                    self._cpu_pf_q.put((epoch, L)) 
        # 同时逐步淘汰较早的 CPU 缓存（比如 < cur_layer-2）
        self._evict_cpu_layers_older_than(cur_layer-2)

    def _wait_cpu_ready(self, layer_idx: int, timeout: float = 5.0):
        """等待CPU缓存准备好指定层"""
        import time
        start = time.time()
        while layer_idx not in self.cpu_cache:
            if time.time() - start > timeout:
                # 超时时尝试立即加载
                if self.verbose:
                    print(f"[WSM] CPU cache layer {layer_idx} timeout, loading immediately...")
                try:
                    self._load_layer_to_cpu(layer_idx)
                    return
                except Exception as e:
                    raise TimeoutError(f"CPU cache layer {layer_idx} not ready after {timeout}s: {e}")
            time.sleep(0.01)

    def _get_param_from_model(self, layer_idx: int, param_suffix: str) -> Optional[torch.Tensor]:
        """
        从模型本身获取参数（传统 streaming 模式）
        param_suffix: 例如 "attention.wq.weight"
        返回: CPU tensor 或 None
        """
        try:
            if not hasattr(self.model, "layers") or layer_idx >= len(self.model.layers):
                return None

            layer = self.model.layers[layer_idx]

            # 解析参数路径：例如 "attention.wq.weight" -> ["attention", "wq", "weight"]
            parts = param_suffix.split('.')
            obj = layer
            for part in parts:
                if not hasattr(obj, part):
                    return None
                obj = getattr(obj, part)

            # obj 应该是一个 Parameter 或 Tensor
            if isinstance(obj, torch.nn.Parameter):
                tensor = obj.data
            elif isinstance(obj, torch.Tensor):
                tensor = obj
            else:
                return None

            # 确保是 CPU tensor（如果已经在 GPU 上，返回 None 让其他逻辑处理）
            if tensor.device.type == "cpu" and tensor.numel() > 0:
                return tensor

            return None

        except Exception as e:
            if self.verbose:
                print(f"[WSM] Error getting param from model: layer={layer_idx}, suffix={param_suffix}, error={e}")
            return None

    def _load_param_from_ssd(self, param_name: str) -> torch.Tensor:
        """从SSD加载单个参数"""
        if not self.ssd_enabled:
            raise RuntimeError("SSD backend not enabled")

        # 解析参数名获取层号
        # 例如: "layers.5.attention.wq.weight"
        parts = param_name.split('.')
        if len(parts) < 3 or parts[0] != "layers":
            raise ValueError(f"Invalid param name: {param_name}")

        layer_idx = int(parts[1])
        if layer_idx not in self.layers_params:
            raise KeyError(f"Layer {layer_idx} not in manifest")

        # 在manifest中查找参数
        for param_info in self.layers_params[layer_idx]:
            if param_info["name"] == param_name:
                # 使用现有的SSD加载逻辑
                from .weights_io_ssd_dram import DTYPE_MAP, alloc_pinned_aligned
                stride = param_info["stride"]
                offset = param_info["offset"]
                nbytes = param_info["nbytes"]

                # 确保staging buffer足够大
                if stride > len(self.staging_buffer):
                    block_size = self.ssd_manifest["block_size"]
                    new_size = ((stride + block_size - 1) // block_size) * block_size
                    self.staging_buffer = alloc_pinned_aligned(new_size, block_size)
                    if self.verbose:
                        print(f"[WSM] Expanded staging buffer to {new_size} bytes for {param_name}")

                # 从SSD读取
                self.ssd_dio.pread_into_tensor(self.staging_buffer, stride, offset)

                # 转换为proper tensor
                param_tensor = torch.empty(
                    param_info["shape"],
                    dtype=DTYPE_MAP[param_info["dtype"]],
                    pin_memory=True
                )
                param_tensor.view(-1).view(torch.uint8)[:nbytes].copy_(
                    self.staging_buffer[:nbytes]
                )
                return param_tensor

        raise KeyError(f"Param {param_name} not found in layer {layer_idx} manifest")

    def _bg_submit(self, task):
        """提交后台任务到线程池"""
        import threading
        t = threading.Thread(target=task, daemon=True)
        t.start()

    def _cpu_layer_ready(self, layer_idx: int) -> bool:
        """检查CPU层是否就绪"""
        return layer_idx in self.cpu_cache

    def _evict_cpu_layers_older_than(self, layer_idx: int):
        """淘汰早于指定层的CPU缓存"""
        if not self.ssd_enabled:
            return

        with self.cpu_cache_lock:
            to_evict = [L for L in list(self.cpu_cache.keys()) if L < layer_idx]
            for L in to_evict:
                self.cpu_cache.pop(L, None)
                if self.verbose:
                    print(f"[WSM] Evicted CPU cache layer {L}")

    

    def warmup_cpu_cache(self):
        """
        预热 CPU cache：幂等式加载初始窗口
        可被多次调用，只在首次执行实际加载
        """
        if self._warm_done:
            if self.verbose:
                print("[WSM] CPU warmup already done, skipping")
            return

        if self.verbose:
            print("[WSM] Starting CPU cache warmup...")

        # 使用滑动窗口机制加载初始窗口
        self._ensure_cpu_window()
        self._warm_done = True

        if self.verbose:
            print(f"[WSM] CPU warmup complete: {len(self.cpu_cache)} layers in cache")

    def prefetch(self, ids: List[int], warm: bool = False):
        """
        Asynchronously prefetch a list of layer indices, respecting the LRU budget.

        Args:
            ids: Layer indices to prefetch
            warm: If True, treat as warmup (idempotent, uses sliding window)
        """
        if warm:
            # 预热模式：幂等式加载
            self.warmup_cpu_cache()
            return

        if not ids:
            return
        ids = ids[: max(1, min(self._pd_current, self.pd_cap))]
        nvtx.range_push(f"prefetch_layers_{ids}")

        for idx in ids:
            nvtx.range_push(f"prefetch_layer_{idx}")
            self._refresh_retain_window(idx)

            if idx in self.gpu_cache:
                self.gpu_cache.move_to_end(idx)
                nvtx.range_pop()
                continue

            # Evict until space is available
            while len(self.gpu_cache) >= self.max_cached_layers:
                self._evict_one_not_retained()

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
                # 判断权重从哪里加载
                source = "SSD→CPU→GPU" if (self.ssd_enabled and idx in self.cpu_cache) else \
                         "SSD→GPU" if self.ssd_enabled else \
                         "CPU→GPU"
                print(f"[WSM] prefetch layer={idx} ({source})")

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
            # step-down, not cliff-drop
            self._pd_current = max(1, self._pd_current - 1)
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

    def shutdown(self, wait: bool = True, timeout: float = 2.0):
        self._stopped = True
        self._stop_event.set()
        try:
            self._cpu_pf_q.put_nowait((self._epoch, None))  # 唤醒 worker
        except Exception:
            pass
        if wait:
            for t in list(self._threads):
                try:
                    t.join(timeout=timeout)
                except Exception:
                    pass

    def __del__(self):
        try:
            self.shutdown(wait=False)
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
