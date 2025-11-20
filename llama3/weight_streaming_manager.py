import os
import queue
import time
import json
import threading
import psutil
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn

from .stream_mnt import get_streams
from .config import load_runtime_config
from .weight_lbt import classify_group  # 用于从参数名推断 attn/ffn

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
    - GPU residency is tracked purely at the (layer, group) level via a small ring of attn/ffn groups.
      Layer-level caches (max_cached_layers) are deprecated; eviction operates on groups only.
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
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError("WeightStreamingManager requires a CUDA device")
        self.model = model
        self.device = device
        self.device_index = device.index if device.index is not None else torch.cuda.current_device()
        self.verbose = verbose

        # 诊断：打印传入的 device 参数
        print(f"[WSM] Initialized with device={device} (type={type(device)})")
        
        self.streams = get_streams(device)
        
        self._phase: str = "prefill"          # 当前阶段：prefill -> decoder
        self._decoder_prime_done: bool = False
        self._last_executed_layer: int = -1    # 最近一次 note_compute_advance 上报的层
        
        base_prefetch_distance = max(0, int(prefetch_distance))
        decode_env = os.getenv("DECODE_PREFETCH_DISTANCE")
        self.prefetch_distance_decode = max(0, int(decode_env)) if decode_env is not None else base_prefetch_distance
        prefill_env = os.getenv("PREFILL_PREFETCH_DISTANCE")
        default_prefill = max(10, base_prefetch_distance)
        self.prefetch_distance_prefill = max(0, int(prefill_env)) if prefill_env is not None else default_prefill
        self.prefetch_distance = self.prefetch_distance_prefill

        self.prefill_cpu_layers = max(0, int(os.getenv("PREFILL_CPU_LAYERS", "40")))
        self.prefill_gpu_layers = max(0, int(os.getenv("PREFILL_GPU_LAYERS", "5")))

        self._cpu_protect_set: set[int] = set()
        self.decoder_protect_layers: int = int(os.getenv("WSM_DECODER_PROTECT_LAYERS", "12"))  # 預設保護前 4 層

        # ⭐ 统一在初始化早期解析环境变量，避免后续覆盖导致配置漂移
        # 优先级：环境变量 WSM_CPU_CACHE_LAYERS > 构造参数 cpu_cache_layers
        _env_cpu_cache = os.getenv("WSM_CPU_CACHE_LAYERS")
        if _env_cpu_cache is not None:
            self.cpu_cache_layers = max(1, int(_env_cpu_cache))
            if self.verbose:
                print(f"[WSM] cpu_cache_layers from env: {self.cpu_cache_layers}")
        else:
            self.cpu_cache_layers = max(1, int(cpu_cache_layers))
            if self.verbose:
                print(f"[WSM] cpu_cache_layers from param: {self.cpu_cache_layers}")

        self.cpu_cache_max = self.cpu_cache_layers  # 保持向后兼容
        self.max_cached_layers = 0  # legacy field (layer-level cache disabled)
        self.warmup_layers = max(0, int(warmup_layers))

        # —— 运行状态 ——
        self.n_layers = len(getattr(model, "layers", []))
        self.grouped_mode = True  # 供 SelfAttention/FFN 走组级 API
        self._anchor = 0          # 计算锚点（EncoderBlock.forward 会持续更新）
        self._anchor_lock = threading.Lock()

        
        # SSD backend configuration
        self.ssd_enabled = ssd_manifest_path is not None
        self.ssd_manifest = None
        self.ssd_dio = None
        self.staging_buffer = None
        self.layers_params = {}

        # CPU cache for SSD->CPU->GPU pipeline
        self.cpu_cache: OrderedDict[int, Dict[str, torch.Tensor]] = OrderedDict()
        self.cpu_cache_lock = threading.Lock()

        # CPU 缓存结构
        self._cpu_cached_layers = set()        # membership 快速判断

        # ==== add to class WeightStreamingManager (init: set knobs) ====
        self.evict_after_use: bool = (os.getenv("WSM_EVICT_AFTER_USE", "1") == "1")
        self.evict_delay_ms: int = int(os.getenv("WSM_EVICT_DELAY_MS", "0"))
        self.cpu_evict_after_use: bool = (os.getenv("WSM_CPU_EVICT_AFTER_USE", "0") == "1")

        # Background prefetch management
        self.prefetch_thread: Optional[threading.Thread] = None
        self.prefetch_queue: List[int] = []
        self.stop_prefetch = threading.Event()
        self.prefetch_lock = threading.Lock()

        # Preload completion tracking
        self.gpu_preload_complete = threading.Event()
        self.cpu_preload_complete = threading.Event()
        self.target_gpu_layers = max(self.prefill_gpu_layers, warmup_layers)
        self.target_cpu_layers = min(self.prefill_cpu_layers, self.cpu_cache_layers)

        # Optional fragmentation monitoring
        self.monitor_fragmentation = monitor_fragmentation
        self.memory_stats: List[Dict] = []
        self.fragmentation_threshold = 0.3
        # PD 自适应参数（滞回 + EMA）
        
        
        self.max_pinned_groups   = int(os.getenv("WSM_GROUP_PIN_BUDGET", "4"))
        
        # 统一的组状态机：(L, kind) -> {"CPU","INFLIGHT","RESIDENT","EVICTING"}
        self._group_state: dict[tuple[int,str], str] = {}

        # 参数版本号：每次成功“安装”或“逐出”都会自增，供 CAS 语义使用
        self._param_version: defaultdict[str, int] = defaultdict(int)

        
        # --- thread & queue state ---
        self._stopped: bool = False
        self._stop_event: threading.Event = threading.Event()
        self._threads: list[threading.Thread] = []
                
        self._cpu_lock = threading.RLock()
        self._epoch = 0
        self._cpu_pf_q: "queue.Queue[tuple[int, int|None]]" = queue.Queue(maxsize=2048)
        self._inflight_cpu_layers = set()
        self._cpu_pf_thread: Optional[threading.Thread] = None

        # ===== 线程池化 CPU 读取配置 =====
        self._use_pooled_cpu_read = (os.getenv("WSM_POOLED_CPU_READ", "0") == "1")
        self._cpu_pf_workers = int(os.getenv("WSM_CPU_PF_WORKERS", "8"))
        self._cpu_executor: Optional[ThreadPoolExecutor] = None
        self._cpu_tls = threading.local()  # 每线程私有 staging buffer
        self.staging_mb = staging_mb  # 保存供线程私有 buffer 使用

        if self.ssd_enabled:
            if self._use_pooled_cpu_read:
                # 线程池化模式：FIFO 调度 + 并行读取
                self._cpu_executor = ThreadPoolExecutor(
                    max_workers=self._cpu_pf_workers,
                    thread_name_prefix="wsm_cpu"
                )
                self._cpu_pf_thread = threading.Thread(
                    target=self._cpu_dispatch_loop,
                    name="wsm_cpu_dispatch",
                    daemon=True,
                )
            else:
                # 传统单线程模式
                self._cpu_pf_thread = threading.Thread(
                    target=self._cpu_prefetch_worker,
                    name="wsm_cpu_pf",
                    daemon=True,
                )
            self._threads.append(self._cpu_pf_thread)
            self._cpu_pf_thread.start()
        
        # ------------- GPU group tracking -------------
        # 事件表：任何组（inflight/resident）的最新 ready event
        self._group_events: dict[tuple[int, str], torch.cuda.Event] = {}
        # ✨ 新增：Host-side 事件标记，记录 CUDA 事件是否已被 record
        self._group_recorded_host: dict[tuple[int, str], threading.Event] = {}
        # 正在进行 H2D 的组键集合（用于并发闸门 + 跳过重复）
        self._gpu_group_inflight: set[tuple[int, str]] = set()
        # 修改：_gpu_group_in_use 改为引用计数，支持嵌套使用
        self._gpu_group_in_use: dict[tuple[int,str], int] = {}   # refcount

        self._placeholder_keys: set[tuple[int, str]] = set()
        
        # CPU->GPU 流水线优化：等候 CPU 就绪后才启动 H2D 的组集合
        self._gpu_need_on_cpu_ready: set[tuple[int, str]] = set()  # {(L, 'attn'/'ffn'), ...}

        # ------------- Balanced group scheduler extensions -------------
        # 预取后但未计算、需要保护的组（引用计数）
        self._pinned_groups: dict[tuple[int, str], int] = {}  # refcount
        # 当前处理的层索引（用于窗口保护）
        self._current_layer: Optional[int] = None
        # 窗口大小：限制预取只在 [i..i+window_size-1] 内
        self._window_size: int = 0  # 将在后面根据 gpu_max_groups 设置

        # ------------- GPU 内存余量守卫 -------------
        self._gpu_free_guard_mb: int = int(os.getenv("WSM_GPU_FREE_GUARD_MB", "512"))  # 1GB 保护
        # self._gpu_max_groups: int = int(os.getenv("WSM_GPU_MAX_GROUPS", "3"))  # [已废弃] 使用下方的 self.gpu_max_groups

        # ⭐ 替换守护线程为共享线程池（避免每次 _bg_submit 创建新线程导致调度瓶颈）
        _bg_workers = int(os.getenv("WSM_BG_WORKERS", "8"))
        self._bg_executor = ThreadPoolExecutor(max_workers=_bg_workers, thread_name_prefix="wsm_bg")

        # Debug toggle: when enabled, emit detailed prints for prefetch/evict decisions
        self.debug_prefetch = (os.getenv("WSM_DEBUG_PREFETCH", "0") == "1")

        # Verbose mismatch: when enabled, emit detailed logs when event ready but group not resident
        self.verbose_mismatch = (os.getenv("WSM_VERBOSE_MISMATCH", "0") == "1")

        # Event wait timeout in seconds (default 0.5s for faster failure detection)
        self._evt_wait_timeout_s = float(os.getenv("WSM_EVENT_TIMEOUT", "0.5"))

        # SDPA workspace headroom in MB (default 64MB)
        self.attn_workspace_headroom_mb = int(os.getenv("ATTN_WORKSPACE_HEADROOM_MB", "64"))

        
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
        # 可选：外部 KV Offloader（若主程序传入，可用于触发"暂停写"）
        self.kv_offloader = None

        # ⭐ 信用/配额机制：限制 inflight 组数，避免 H2D 队列过载（FlexGen credit-based）
        # 典型值：2 * prefetch_distance，即允许预取窗口内的组都处于 inflight
        _max_inflight_env = int(os.getenv("WSM_MAX_INFLIGHT_GROUPS", "0"))
        self.max_inflight_groups = _max_inflight_env or (2 * max(1, self.prefetch_distance))
        if getattr(self, "verbose", False):
            print(f"[WSM] 信用配额: max_inflight_groups={self.max_inflight_groups} (2 * prefetch_distance={self.prefetch_distance})")
        
        self.grouped_mode = True  # 开启"组级"模式

        # —— 统一 GPU 组预算，仅设置一次 ——
        # 组预算计算：当前层(attn=1) + 当前层预取(ffn=1) + 下一层预取(attn=1) + 安全余量(1) + 弹性缓冲(1~2) = 5~7
        # 对于 SSD 模式或更激进的并发预取，建议 6~9；显存充足时可以设更高
        # 自动确保至少满足 gpu_ahead_layers 的需求：3 + gpu_ahead_layers
        _env_max = int(os.getenv("WSM_GPU_MAX_GROUPS", "0"))
        _gpu_ahead = int(os.getenv("WSM_GPU_AHEAD", os.getenv("WSM_GPU_AHEAD_LAYERS", "4")))
        _required_min = 3 + _gpu_ahead  # 当前组(2) + 预取组(_gpu_ahead) + 余量(1)
        self.gpu_max_groups = max(_env_max or 10, _required_min)
        print(f"[WSM] GPU组预算: max_groups={self.gpu_max_groups} (env={_env_max or 'default'}, required_min={_required_min})")

        # 设置窗口大小（用于平衡组调度器）
        self._window_size = self.gpu_max_groups
        
        self.cpu_prefetch_distance = int(os.getenv("WSM_CPU_PREFETCH_DISTANCE", "50"))  # CPU 端预取窗口
        self.cpu_cache_cap_layers  = int(os.getenv("WSM_CPU_CACHE_CAP_LAYERS",  "50"))  # 硬上限
        self.cpu_cache_hwm_layers  = int(os.getenv("WSM_CPU_CACHE_HWM_LAYERS",  "55"))  # 高水位
        

        # 滑动窗口 + 回滞参数
        # ★ 关键修复: cpu_cache_cap 应该使用已解析的 self.cpu_cache_layers（而非参数）
        env_cap = os.getenv("WSM_CPU_CACHE_CAP_LAYERS")
        if env_cap is not None:
            self.cpu_cache_cap = int(env_cap)
            print(f"[WSM] Using WSM_CPU_CACHE_CAP_LAYERS from env: {self.cpu_cache_cap}")
        else:
            # ✅ 使用已经从环境变量更新后的 self.cpu_cache_layers
            self.cpu_cache_cap = self.cpu_cache_layers
            print(f"[WSM] Using cpu_cache_layers (final value): {self.cpu_cache_cap}")

        # ⭐ 关键配置：HWM/LWM 水位线
        # HWM (High Water Mark): 触发 emergency cleanup 的阈值
        # LWM (Low Water Mark): emergency cleanup 的目标清理后容量
        #
        # ⚠️ 推荐设置：
        # - HWM = cap + 10 到 cap + 20（给窗口前移和 inflight 层留缓冲）
        # - LWM = cap - 3 到 cap（清理后保持接近容量）
        #
        # 如果 emergency cleanup 频繁触发（日志中看到大量 "CPU cache evict (HWM)"），
        # 说明窗口清理不够及时，应该：
        # 1. 增加 HWM 缓冲（避免频繁触发）
        # 2. 减少 cpu_cache_cap（更严格的容量控制）
        # 3. 检查保护层是否过多（_cpu_protect_set）
        self.cpu_hwm       = int(os.getenv("WSM_CPU_CACHE_HWM_LAYERS", str(self.cpu_cache_cap + 15)))
        self.cpu_lwm       = int(os.getenv("WSM_CPU_CACHE_LWM_LAYERS", str(max(2, self.cpu_cache_cap - 3))))
        self.cpu_back_margin = int(os.getenv("WSM_CPU_BACK_MARGIN", "4"))  # 留一点历史
        self.cpu_lookahead_distance = int(os.getenv("WSM_CPU_LOOKAHEAD_DISTANCE", "10"))  # 提前多少层开始预取
        self.cpu_front_margin = max(0, self.cpu_cache_layers - self.cpu_back_margin)  # 在此处一次性计算

        # CPU 窗口起点（层号）- 初始值 0，推理时会动态推进
        self.cpu_win_base  = 0
        self._warm_done = False  # 预热幂等标志
        self._last_cpu_advance_layer = -1  # 记录最后一次推进窗口的层号（去重）

        print(f"[WSM] CPU cache config: cap={self.cpu_cache_cap}, hwm={self.cpu_hwm}, lwm={self.cpu_lwm}, lookahead={self.cpu_lookahead_distance}")

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

        self._gpu_group_ring = []    # [(layer_idx, 'attn'/'ffn'), ...] 维护在卡上的组
        # 注意：_gpu_group_in_use 已在上面定义为 dict（引用计数），这里不再重复定义
        self._group_lock = threading.RLock()

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
        
        # 在 __init__ 里新增一个开关（默认 False；由环境变量控制）
        self.evict_finished_group = (os.getenv("WSM_EVICT_FINISHED", "1") == "1")
        
        # 在 __init__ 结尾附近、其它配置旁边加：
        # self.cpu_rolling_mode   = (os.getenv("WSM_CPU_ROLLING_MODE",  "1") == "1")   # 开启“层层滚动”
        # self.cpu_wrap_around    = (os.getenv("WSM_CPU_WRAP_AROUND",   "1") == "1")   # 支持下一轮回到 L0
        # self.cpu_roll_stride    = int(os.getenv("WSM_CPU_ROLL_STRIDE","1"))          # 每次右移几层，默认 1
        # self.cpu_roll_sync      = (os.getenv("WSM_CPU_ROLL_SYNC",     "1") == "1")   # 触发后同步确保窗口（简单可靠）

        # # ---- Group-window policy (GPU) ----
        # ======== [已废弃的激进预取策略 - 保留供参考] ========
        # # 按旧策略：当前(i) attn+ffn；i+1..i+3 的 attn+ffn；以及 i+4 的 attn
        # self.future_both_layers = int(os.getenv("WSM_FUTURE_BOTH_LAYERS", "3"))   # i+1..i+3 两组
        # self.buffer_attn_ahead  = int(os.getenv("WSM_BUFFER_ATTN_AHEAD",  "4"))   # i+4 的 attn
        # self.group_wrap_around  = (os.getenv("WSM_GROUP_WRAP_AROUND", "0") == "1")
        #
        # # 需要的组上限 = 2(当前) + 2*future_both + 1(buffer attn) = 9
        # _required_groups = 2 + 2*self.future_both_layers + 1
        # # 若外部没显式设 WSM_GPU_MAX_GROUPS，则采用所需上限；若设了，就取更大的那个
        # try:
        #     env_max = int(os.getenv("WSM_GPU_MAX_GROUPS", "0"))
        # except ValueError:
        #     env_max = 0
        # self.gpu_max_groups = max(self.gpu_max_groups, _required_groups, env_max or 0)
        # print(f"[WSM] Group-window policy: future_both={self.future_both_layers}, "
        #     f"buffer_attn={self.buffer_attn_ahead}, gpu_max_groups={self.gpu_max_groups}")
        #
        # 注：当前实现采用更保守的预取策略（见 _pre_hook_factory 的组级预取逻辑）：
        #     - 当前层 attn (执行中) = 1
        #     - 当前层 ffn (异步预取) = 1
        #     - 下一层 attn (异步预取) = 1
        #     - 安全余量 = 1
        #     - 弹性缓冲 = 1~2
        #     → 总需求 = 5~7 组，默认设为 6，建议范围 6~9
        # ======================================================
        self.gpu_ahead_layers   = _gpu_ahead  # 重用前面已解析的值
        self.gpu_behind_layers  = max(1, int(os.getenv("WSM_GPU_BEHIND", "3")))  # 默认保留刚用过的 3 层
        self.cpu_ring_mode   = (os.getenv("WSM_CPU_RING_MODE",  "1") == "1")  # 开：环形窗口
        self.cpu_ring_offset = int(os.getenv("WSM_CPU_RING_OFFSET", str(self.gpu_ahead_layers)))  # 从环境变量读取，默认=gpu_ahead_layers
        # ❌ 删除重复赋值：gpu_max_groups 已在前面统一设置
        
        # --- group-level retention (add in __init__) ---

        
        
        
        # 独立 H2D stream（仅在CUDA设备上创建）
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            self._copy_stream = torch.cuda.Stream(device=self.device_index)
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
        # GPU group ring (debug only; eviction decisions rely on window logic)
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

        # # (Optional) Warm up target GPU layers to reduce initial latency
        # if self.target_gpu_layers > 0:
        #     warm = list(range(min(self.target_gpu_layers, len(self.blocks))))
        #     self.prefetch(warm)
        #     if self.verbose:
        #         print(f"[WSM] GPU warmup prefetch: {warm} (target: {self.target_gpu_layers} layers)")

        # ---- H2D 并发闸门（必须在 warmup 之前初始化）----
        # ✅ P0-2: 自适应并发控制（使用令牌持有器而非固定 Semaphore）
        self._h2d_base_concurrency = int(os.getenv("WSM_H2D_BASE_CONCURRENCY", "2"))
        self._h2d_prefill_multiplier = float(os.getenv("WSM_H2D_PREFILL_MULT", "2.0"))
        self._h2d_decode_multiplier = float(os.getenv("WSM_H2D_DECODE_MULT", "1.0"))
        self._h2d_groups_max: int = self._h2d_base_concurrency
        self._h2d_max_possible = 8  # 最大可能并发数
        self._h2d_sem = threading.Semaphore(self._h2d_max_possible)
        self._h2d_active_tokens = 0
        self._h2d_tokens_lock = threading.Lock()
        if self.verbose:
            print(f"[WSM] H2D concurrency limit: {self._h2d_groups_max} (adaptive: base={self._h2d_base_concurrency})")
            
        # ===== P0-1: Wrap-around 性能平滑 =====
        self._wraparound_warmup_layers = int(os.getenv("WSM_WRAPAROUND_WARMUP", "6"))
        self._last_layer_idx = -1
        self._is_wraparound_warmup = False
        if self.n_layers >= 15:
            self._wraparound_detect_threshold = self.n_layers - 10
            self._wraparound_enabled = True
        else:
            self._wraparound_detect_threshold = 0
            self._wraparound_enabled = False
        self._wraparound_warmup_sem = threading.Semaphore(2)    
        # ===== P0-3: H2D 传输超时与重试 =====
        self._h2d_timeout_sec = float(os.getenv("WSM_H2D_TIMEOUT_SEC", "10.0"))
        self._h2d_max_retries = int(os.getenv("WSM_H2D_MAX_RETRIES", "3"))
        self._h2d_retry_base_delay = float(os.getenv("WSM_H2D_RETRY_BASE_DELAY", "0.2"))
        self._h2d_timeout_count = 0
        self._h2d_retry_count = 0
        self._h2d_stats_lock = threading.Lock()
        self._h2d_retry_streams = [
            torch.cuda.Stream(device=self.device)
            for _ in range(self._h2d_max_retries + 1)
        ]
        self._pin_buffer_pool = {}

        if self.verbose:
            print(f"[WSM][P0] Patches enabled:")
            print(f"  - Wrap-around: warmup={self._wraparound_warmup_layers}, enabled={self._wraparound_enabled}")
            print(f"  - H2D timeout: {self._h2d_timeout_sec}s, retries={self._h2d_max_retries}")


        # 组级预取 backlog（拿不到信号量时排队，不静默丢弃）
        self._gpf_backlog_max = int(os.getenv("WSM_H2D_GROUP_BACKLOG_MAX", "64"))
        self._gpf_q: "queue.Queue[tuple[int, tuple[int,str], bool, str, torch.cuda.Stream|None]]" = \
            queue.Queue(maxsize=self._gpf_backlog_max)
        self._gpf_seen: set[tuple[int,str]] = set()   # 去重

        # ===== 异步逐出队列 =====
        self._evict_queue_size = int(os.getenv("WSM_EVICT_QUEUE_SIZE", "100"))
        self._evict_q: "queue.Queue[tuple[int, str, torch.cuda.Event|None]|None]" = \
            queue.Queue(maxsize=self._evict_queue_size)
        self._evict_seen: set[tuple[int, str]] = set()  # 去重

        # 组级 lookahead
                # Accept multiple env aliases for compatibility
        _gpd_env = os.getenv("WSM_GROUP_PREFETCH_DISTANCE")
        if _gpd_env is None:
            _gpd_env = os.getenv("WSM_GROUP_PREFETCH_DEPTH")
        if _gpd_env is None:
            _gpd_env = os.getenv("WSM_PAIR_AHEAD")
        if _gpd_env is None:
            _gpd_env = os.getenv("WSM_GPU_AHEAD")
        self.group_prefetch_depth: int = int(_gpd_env or "3")

        # 启动组级 backlog 调度线程
        t = threading.Thread(target=self._h2d_dispatch_loop, name="wsm_h2d_dispatch", daemon=True)
        self._threads.append(t)
        t.start()

        # 启动异步逐出工作线程
        evict_t = threading.Thread(target=self._async_eviction_worker, name="wsm_evict", daemon=True)
        self._threads.append(evict_t)
        evict_t.start()



        # (Optional) Warm up at start
        if self.target_gpu_layers > 0:
            if getattr(self, "grouped_mode", False):
                # 组级 warmup（默认 doublet：0.attn,0.ffn,1.attn,1.ffn,...）
                scheme = os.getenv("WSM_WARMUP_GROUP_SCHEME", "doublet")
                self.warmup_groups_prefetch(layers=self.target_gpu_layers, scheme=scheme)
            else:
                # 旧的整层 warmup 路径
                warm = list(range(min(self.target_gpu_layers, len(self.blocks))))
                self.prefetch(warm)
                if self.verbose:
                    print(f"[WSM] GPU warmup prefetch: {warm} (target: {self.target_gpu_layers} layers)")

        self.balance_prefetch = (os.getenv("WSM_BALANCE_PREFETCH", "1") == "1")
        self.balance_tolerance = int(os.getenv("WSM_BALANCE_TOL", "1"))          # 允许 attn/ffn 驻留差值
        self.pair_ahead_layers = int(os.getenv("WSM_PAIR_AHEAD", "2"))           # 就近层数，优先同层→i+1→i+2
        self.kind_ahead_cap = int(os.getenv("WSM_KIND_AHEAD_CAP", "2"))          # 单一类型最多前瞻距离


    
        # --- within WeightStreamingManager.__init__ ---
        # ❌ 删除第三次重复赋值：gpu_max_groups 已在前面统一设置
        self.target_gpu_groups   = int(os.getenv("WSM_TARGET_GPU_GROUPS", self.gpu_max_groups / 2))
        # ✅ 已在初始化早期统一解析 cpu_cache_layers，此处不再覆盖
        # 基于最终值重新计算依赖配置（防止 blocks 数量变化导致的不一致）
        self.target_cpu_layers = min(self.prefill_cpu_layers, self.cpu_cache_layers, len(self.blocks))
        # 是否允许在溢出第二轮收缩时强制踢掉"非 IN_USE 的 pinned 组"
        self.allow_force_unpin   = (os.getenv("WSM_ALLOW_FORCE_UNPIN", "1") == "1")
        # 限制同时被 pin 的组数（软上限，超限则后续的 prefetch 不再 pin）
        
        # 是否在前向路径同步 topoff（默认关闭，走异步回调）
        self.rebalance_sync      = (os.getenv("WSM_REBALANCE_SYNC", "0") == "1")

        self.evict_finished_group = (os.getenv("WSM_EVICT_FINISHED","1") == "1")
        self.balance_prefetch    = (os.getenv("WSM_BALANCE_PREFETCH", "1") == "1")
        self.balance_tolerance   = int(os.getenv("WSM_BALANCE_TOL", "1"))
        self.pair_ahead_layers   = int(os.getenv("WSM_PAIR_AHEAD", "2"))

        self.warmup_layers_gpu = int(os.getenv("WSM_WARMUP_LAYERS_GPU", "2"))
        self._bootstrap_done = False
        
         # --- inflight 僵尸清理策略 ---
        self.recent_layers_protect = int(os.getenv("WSM_RECENT_LAYERS_PROTECT", "2"))
        self._pending_drop_after_ready: set[tuple[int, str]] = set()


            # GPU 窗口配置（Group 级）
        self.gpu_ahead_groups = int(os.getenv("WSM_GPU_AHEAD_GROUPS", "8"))  # 预取 8 个组
    # ---- Balanced group scheduler: key normalization ----------
    @staticmethod
    def _key(L: int, kind: str) -> tuple[int, str]:
        """
        规范化组键，确保一致性。

        Args:
            L: 层索引
            kind: 组类型 ('attn' 或 'ffn')

        Returns:
            (layer_idx, normalized_kind) 元组
        """
        return (int(L), 'attn' if kind == 'attn' else 'ffn')

    def _resident_kind_counts(self) -> dict:
        """统计当前 GPU 上（含inflight）的 attn/ffn 组数。"""
        c = {"attn": 0, "ffn": 0}
        with self._group_lock:
            for lyr, grp in list(self._gpu_group_ring):
                if grp in c: c[grp] += 1
        return c

    def _nearest_candidates(self, cur_idx: int, kind: str, max_dist: int):
        """
        产生“就近”的候选 (layer, kind)：ffn 从 0 距离(同层)开始，attn 从 1 距离开始。
        要求：不在 GPU、且不在 inflight。
        """
        start = 0 if kind == "ffn" else 1
        inflight = set(self._gpu_group_inflight)
        for d in range(start, max_dist + 1):
            j = (cur_idx + d) % self.n_layers
            key = (j, kind)
            if self._group_is_resident(j, kind):   # 已在卡
                continue
            if key in inflight:                    # 正在搬
                continue
            yield key


    def _async_eviction_worker(self):
        """
        后台线程：处理异步逐出请求
        从 _evict_q 取逐出任务，等待事件完成后执行逐出
        """
        debug = getattr(self, "debug_prefetch", False)
        while not getattr(self, "_stopped", False):
            try:
                # 从队列取逐出任务（带超时，避免阻塞退出）
                task = self._evict_q.get(timeout=0.5)
                if task is None:  # 停止信号
                    break

                layer_idx, group, wait_event = task
                key = (layer_idx, group)

                # ✅ 事件驱动：只在必要时等待（在后台线程，不阻塞主线程）
                if wait_event is not None:
                    try:
                        # synchronize() 会阻塞但只阻塞后台线程
                        wait_event.synchronize()
                    except Exception as e:
                        if debug:
                            print(f"[WSM ASYNC EVICT] Event sync failed for {key}: {e}")

                # 执行逐出（D2H 在后台完成）
                try:
                    self._evict_group_immediately(layer_idx, group, skip_prefetch=True)
                    if debug:
                        print(f"[WSM ASYNC EVICT] Evicted {key} in background")
                except Exception as e:
                    if self.verbose:
                        print(f"[WSM ASYNC EVICT] Evict {key} failed: {e}")
                finally:
                    # 清理去重标记
                    self._evict_seen.discard(key)

            except queue.Empty:
                continue
            except Exception as e:
                if self.verbose:
                    print(f"[WSM ASYNC EVICT] Worker error: {e}")


    def _h2d_dispatch_loop(self):
        """
        后台线程：消费 _gpf_q，完成 CPU→GPU 安装，并保证以下顺序：
        - 真实事件在 H2D 流上 record()
        - 再 set host_evt（host 侧“record 完成”）
        - 再替换占位（若存在）
        兼容 5 元组/3 元组两种载荷格式。
        """
        import torch, threading, queue
        while not self._stop_event.is_set():
            try:
                item = self._gpf_q.get(timeout=0.05)
            except queue.Empty:
                continue

            # --- 统一解析队列载荷 ---
            epoch = getattr(self, "_epoch", 0)
            pin = False; reason = "queued"; h2d_override = None
            if isinstance(item, tuple) and len(item) >= 5:
                ep, key, pin, reason, h2d_override = item[:5]
                if ep != epoch:
                    # 过期任务直接丢弃
                    continue
                layer_idx, group = key
            elif isinstance(item, tuple) and len(item) == 3:
                # 兼容旧格式
                layer_idx, group, h2d_override = item
            else:
                # 未知格式，忽略
                continue

            key = (int(layer_idx), 'attn' if group == 'attn' else 'ffn')

            # CPU 未就绪则回投
            if not self._cpu_group_ready(layer_idx, key[1]):
                self._requeue_gpf(key, h2d_override, pin, reason="cpu_wait")
                continue

            # 选择 H2D 流
            if key[1] == "attn":
                h2d_stream = h2d_override or getattr(self.streams, "weight_h2d_mha", None)
            else:
                h2d_stream = h2d_override or getattr(self.streams, "weight_h2d_ffn", None)
            if h2d_stream is None and torch.cuda.is_available():
                h2d_stream = torch.cuda.current_stream()

            # 确保有事件与 host 标志（占位可能已存在）
            with self._group_lock:
                evt = self._group_events.get(key)
                if evt is None:
                    evt = torch.cuda.Event(blocking=False)
                    self._group_events[key] = evt
                    getattr(self, "_placeholder_keys", set()).add(key)
                host_evt = self._group_recorded_host.get(key)
                if host_evt is None:
                    host_evt = threading.Event()
                    self._group_recorded_host[key] = host_evt
                self._set_state(key, "INFLIGHT")
                self._gpu_group_inflight.add(key)

            # 真正 H2D + 在 H2D 流上记录“同一个”事件
            with torch.cuda.stream(h2d_stream):
                self._install_group_on_gpu(layer_idx, key[1], h2d_override=h2d_stream)
                evt.record(h2d_stream)


            
            with self._group_lock:
                getattr(self, "_placeholder_keys", set()).discard(key)
                # 进入 ring（保持 INFLIGHT；由使用时升级为 RESIDENT）
                try:
                    if key in self._gpu_group_ring:
                        self._gpu_group_ring.remove(key)
                    self._gpu_group_ring.append(key)
                except ValueError:
                    self._gpu_group_ring.append(key)
                    
            host_evt.set()


    def _plan_balanced_groups(self, cur_idx: int, budget: int) -> list[tuple[int, str]]:
        """
        等水位规划：先保证“同层FFN”，再在预算内按“谁少补谁”的原则，就近补到差值≤tolerance。
        """
        if budget <= 0:
            return []

        plan: list[tuple[int, str]] = []
        counts = self._resident_kind_counts()

        # (A) 刚需：同层 FFN（和当前 MHA 计算强重叠）
        for key in self._nearest_candidates(cur_idx, "ffn", max_dist=0):
            plan.append(key); budget -= 1; counts["ffn"] += 1
            break

        # (B) 就近填平：先看差值，谁少补谁；找“就近层的该组”
        tol = max(0, self.balance_tolerance)
        dist_cap = max(1, self.pair_ahead_layers)

        def _try_fill(kind: str):
            nonlocal budget
            for key in self._nearest_candidates(cur_idx, kind, max_dist=min(dist_cap, self.kind_ahead_cap)):
                if budget <= 0: break
                plan.append(key); budget -= 1; counts[kind] += 1
                # 每补一次就重新比较一次，避免一口气补太多
                if abs(counts["attn"] - counts["ffn"]) <= tol:
                    break

        # 循环填平直到预算或差值满足
        while budget > 0:
            if counts["attn"] + counts["ffn"] >= self.gpu_max_groups - 1:
                break
            if (counts["attn"] - counts["ffn"]) > tol:
                _try_fill("ffn")
            elif (counts["ffn"] - counts["attn"]) > tol:
                _try_fill("attn")
            else:
                # 已在容差内：再补“下一层的 attn”（就近一格），增强流水线稳定性
                for key in self._nearest_candidates(cur_idx, "attn", max_dist=1):
                    if budget <= 0: break
                    plan.append(key); budget -= 1; counts["attn"] += 1
                    break
                break  # 不再无限加长

        return plan

    
    def _free_gpu_mem_bytes(self) -> int:
        # 兜底：如果 device 不是 CUDA，返回一个很大的值（避免 OOM 检查失败）
        if not str(self.device).startswith("cuda"):
            if self.verbose:
                print(f"[WSM] Warning: _free_gpu_mem_bytes called with non-CUDA device: {self.device}")
            return 100 * 1024**3  # 返回 100GB 作为占位
        free, total = torch.cuda.mem_get_info(self.device_index)
        return int(free)

    # --- helper: 环绕索引 ---
    def _wrap(self, idx: int) -> int:
        return int(idx % self.n_layers)
    
    def _ring_contains(self, head: int, L: int, window: int) -> bool:
        # L 是否落在 [head, head+window) 的环上
        return ((int(L) - int(head)) % self.n_layers) < int(window)

    # ===== P0-1: Wrap-around 检测与预热 =====
    # ===== P0-1: Wrap-around 检测与预热 =====
    def _detect_and_handle_wraparound(self, current_layer: int) -> bool:
        """
        检测并处理 wrap-around：
        - prefill 接近末尾时触发
        - 重置 CPU 窗口基准到 0
        - 预热前 N 层到 CPU / GPU
        - decoder 阶段加保护窗口
        """
        if not getattr(self, "_wraparound_enabled", False):
            return False

        if self.n_layers <= 0 or self._last_layer_idx < 0:
            self._last_layer_idx = current_layer
            return False

        wraparound_happened = (
            self._last_layer_idx >= self._wraparound_detect_threshold
            and current_layer < 10
        )

        if wraparound_happened and not self._is_wraparound_warmup:
            if self.verbose:
                print(f"[WSM][P0-1] Wrap-around detected: L{self._last_layer_idx} -> L{current_layer}")

            # 1) 重置 CPU 窗口基准 & 游标，让 decode 从头开始滚动
            try:
                with self._cpu_lock:
                    # 线性窗口从 0 开始
                    self.cpu_win_base = 0
                    if hasattr(self, "_cpu_pf_cursor"):
                        self._cpu_pf_cursor = 0
                    # bump epoch，清理旧的预取任务
                    if hasattr(self, "_epoch"):
                        self._epoch += 1
            except Exception as e:
                if self.verbose:
                    print(f"[WSM][P0-1] wrap-around reset cpu window failed: {e}")

            # 2) 预热前 N 层（SSD->CPU, CPU->GPU）
            self._warmup_first_layers(self._wraparound_warmup_layers)
            self._is_wraparound_warmup = True

            # 3) decoder 阶段的额外保护窗口
            if getattr(self, "_phase", "") == "decoder":
                protect_n = max(
                    getattr(self, "decoder_protect_layers", 0),
                    self._wraparound_warmup_layers,
                )
                if hasattr(self, "_ensure_decoder_protect_window"):
                    self._ensure_decoder_protect_window(protect_n, reason="wraparound")

            self._last_layer_idx = current_layer
            return True

        if self._is_wraparound_warmup and current_layer > self._wraparound_warmup_layers:
            self._is_wraparound_warmup = False

        self._last_layer_idx = current_layer
        return False



    def _warmup_first_layers(self, n_layers: int) -> None:
        """预热前 N 层，使用限流避免洪泛"""
        n = min(n_layers, self.n_layers)
        if n <= 0:
            return

        debug = getattr(self, "debug_prefetch", False)
        if debug:
            print(f"[WSM][P0-1] Warming up first {n} layers (throttled)")

        if getattr(self, "ssd_enabled", False):
            for lid in range(n):
                if lid not in getattr(self, "cpu_cache", {}):
                    try:
                        if hasattr(self, "_schedule_cpu_load"):
                            self._schedule_cpu_load(lid)
                        elif hasattr(self, "_load_layer_to_cpu"):
                            try:
                                self._load_layer_to_cpu(lid)
                            except Exception:
                                pass
                    except Exception as e:
                        if self.verbose:
                            print(f"[WSM][P0-1] Failed to warmup L{lid}: {e}")

        def _throttled_prefetch(layer_id: int):
            sem = getattr(self, "_wraparound_warmup_sem", None)
            if sem is None:
                return

            if not sem.acquire(blocking=False):
                if debug:
                    print(f"[WSM][P0-1] Skipped warmup L{layer_id} (throttled)")
                return

            try:
                if hasattr(self, "prefetch_group_async"):
                    self.prefetch_group_async(layer_id, "attn", reason="wraparound-warmup")
            except Exception:
                pass
            finally:
                sem.release()

        for lid in range(n):
            try:
                if hasattr(self, "_bg_executor"):
                    self._bg_executor.submit(_throttled_prefetch, lid)
                else:
                    _throttled_prefetch(lid)
            except Exception:
                pass

    def _clear_group_ready_event(self, key: tuple[int,str]):
        with self._group_lock:
            self._group_events.pop(key, None)

    def _get_pin_budget(self) -> int:
        return int(getattr(self, "max_pinned_groups", 2))
    
    # --- inside class WeightStreamingManager ---
    def _clear_stale_group_events(self):
        """
        安全清理：不得删除“占位但尚未 record”的事件（_placeholder_keys）。
        - 仅清理：不在 ring 且不在 placeholder 集；或状态为 EVICTING 的事件。
        - 若发现某 key 非 RESIDENT 且既无事件也不在 placeholder 集，则把它从 ring 移除并置 CPU。
        """
        ph = getattr(self, "_placeholder_keys", set())
        with self._group_lock:
            # 1) 清理真正的“游离事件”
            for key, evt in list(self._group_events.items()):
                if key in ph:
                    continue  # 占位期，不可删
                st = self._get_state(key)
                if st == "INFLIGHT":
                    continue
                in_ring = key in self._gpu_group_ring
                if st == "INFLIGHT":
                    continue
                if (not in_ring) or (st == "EVICTING"):
                    self._group_events.pop(key, None)

            # 2) 悬空保护：非 RESIDENT 且“没有事件也不在占位集”的直接撤出 ring
            for key in list(self._gpu_group_ring):
                if key in ph:
                    continue
                st = self._get_state(key)
                if st != "RESIDENT" and key not in self._group_events:
                    try:
                        self._gpu_group_ring.remove(key)
                    except (ValueError, KeyError):
                        pass
                    self._set_state(key, "CPU")


    def _set_prefetch_distance_for_phase(self, phase: Optional[str] = None):
        """
        调整不同阶段的预取距离，便于在 prefill/decoder 间切换不同 lookahead。
        """
        phase = phase or getattr(self, "_phase", "prefill")
        target = self.prefetch_distance_prefill if phase == "prefill" else self.prefetch_distance_decode
        prev = getattr(self, "prefetch_distance", None)
        if prev == target:
            return
        self.prefetch_distance = target
        if hasattr(self, "_pd_current"):
            self._pd_current = max(1, target) if target > 0 else 1
        if hasattr(self, "group_prefetch_depth"):
            self.group_prefetch_depth = max(1, target) if target > 0 else 1
        if not getattr(self, "grouped_mode", True):
            self._layer_prefetch_distance = target
        if self.verbose:
            print(f"[WSM] Prefetch distance switched to {target} (phase={phase})")

        # ✅ P0-2: 同时调整 H2D 并发
        self._adjust_h2d_concurrency_for_phase(phase)

    # ===== P0-2: 自适应并发控制方法 =====
    def _h2d_acquire_token(self, timeout: float = None) -> bool:
        """
        获取 H2D 传输令牌（受动态并发限制）

        timeout 语义：
        - None  : 阻塞等待
        - > 0   : 阻塞等待，超时返回 False
        - <= 0  : 非阻塞立即返回（不传 timeout 参数）
        """
        # 先看动态并发上限（只是软限制，真正的控制在令牌计数）
        with self._h2d_tokens_lock:
            if self._h2d_active_tokens >= self._h2d_groups_max:
                # 这里不提前 return，让底层 sem 自己做流控
                pass

        # 正确调用 Semaphore.acquire，避免 "can't specify timeout for non-blocking acquire"
        if timeout is None:
            acquired = self._h2d_sem.acquire()  # 阻塞
        elif timeout <= 0:
            acquired = self._h2d_sem.acquire(blocking=False)  # 非阻塞，不传 timeout 参数
        else:
            acquired = self._h2d_sem.acquire(timeout=timeout)  # 阻塞 + 超时

        if not acquired:
            return False

        # 拿到令牌，更新活跃计数
        with self._h2d_tokens_lock:
            self._h2d_active_tokens += 1

        return True


    def _h2d_release_token(self) -> None:
        """释放 H2D 传输令牌"""
        with self._h2d_tokens_lock:
            self._h2d_active_tokens = max(0, self._h2d_active_tokens - 1)

        self._h2d_sem.release()

    def _adjust_h2d_concurrency_for_phase(self, phase: str) -> None:
        """根据阶段动态调整 H2D 并发限制"""
        if phase == "prefill":
            target = int(self._h2d_base_concurrency * self._h2d_prefill_multiplier)
        elif phase == "decoder":
            target = int(self._h2d_base_concurrency * self._h2d_decode_multiplier)
        else:
            target = self._h2d_base_concurrency

        target = max(1, min(target, self._h2d_max_possible))

        with self._h2d_tokens_lock:
            if target == self._h2d_groups_max:
                return

            old_limit = self._h2d_groups_max
            self._h2d_groups_max = target

            if target < old_limit:
                excess = old_limit - target
                if self.verbose:
                    print(f"[WSM][P0-2] Reducing H2D concurrency: {old_limit} -> {target} (will absorb {excess} tokens)")
            else:
                if self.verbose:
                    print(f"[WSM][P0-2] Increasing H2D concurrency: {old_limit} -> {target} (phase={phase})")

    # ===== P0-3: H2D 传输超时与重试方法 =====
    def _ensure_pinned(self, cpu_tensor: torch.Tensor) -> torch.Tensor:
        """确保张量是 pinned memory，保证真正的非阻塞传输"""
        if cpu_tensor.is_pinned():
            return cpu_tensor

        try:
            return cpu_tensor.pin_memory()
        except Exception:
            if self.verbose:
                print(f"[WSM][P0-3] Warning: Failed to pin tensor (numel={cpu_tensor.numel()}), non_blocking may block")
            return cpu_tensor

    def _h2d_transfer_with_retry(self,
                                 cpu_tensor: torch.Tensor,
                                 param_name: str,
                                 base_stream: torch.cuda.Stream) -> torch.Tensor:
        """带超时和重试的 H2D 传输（默认非轮询，提升并发度）"""
        import random
        import os

        retry_count = 0
        last_error = None
        pinned_tensor = self._ensure_pinned(cpu_tensor)

        # ✅ 默认: 非轮询模式（异步，立即返回）
        # 环境变量 WSM_H2D_STRICT_TIMEOUT=1 可启用严格模式（带轮询）
        strict_timeout = os.getenv("WSM_H2D_STRICT_TIMEOUT", "0") == "1"

        if not strict_timeout:
            # 非严格模式: 不轮询，直接返回
            use_stream = base_stream
            with torch.cuda.stream(use_stream):
                gpu_tensor = pinned_tensor.to(device=self.device, non_blocking=True)
            # ✅ 立即返回，调用者负责 record 事件
            return gpu_tensor

        # ✅ 严格模式（仅调试）: 带轮询和重试
        while retry_count <= self._h2d_max_retries:
            try:
                if retry_count == 0:
                    use_stream = base_stream
                else:
                    stream_idx = min(retry_count - 1, len(self._h2d_retry_streams) - 1)
                    use_stream = self._h2d_retry_streams[stream_idx]

                with torch.cuda.stream(use_stream):
                    gpu_tensor = pinned_tensor.to(device=self.device, non_blocking=True)

                    timeout_event = torch.cuda.Event()
                    timeout_event.record(use_stream)

                    start_time = time.time()
                    poll_interval = 0.001
                    while not timeout_event.query():
                        elapsed = time.time() - start_time
                        if elapsed > self._h2d_timeout_sec:
                            raise TimeoutError(
                                f"H2D transfer timeout after {elapsed:.2f}s for {param_name}"
                            )
                        time.sleep(poll_interval)

                    return gpu_tensor

            except TimeoutError as e:
                last_error = e
                with self._h2d_stats_lock:
                    self._h2d_timeout_count += 1

                if retry_count < self._h2d_max_retries:
                    with self._h2d_stats_lock:
                        self._h2d_retry_count += 1

                    backoff = self._h2d_retry_base_delay * (2 ** retry_count)
                    jitter = random.uniform(0, backoff * 0.1)
                    delay = backoff + jitter

                    if self.verbose:
                        print(f"[WSM][P0-3] H2D timeout for {param_name}, retry {retry_count + 1}/{self._h2d_max_retries} after {delay:.3f}s")

                    time.sleep(delay)
                    retry_count += 1
                    continue
                else:
                    raise RuntimeError(
                        f"H2D transfer failed after {self._h2d_max_retries} retries: {param_name}"
                    ) from last_error

            except torch.cuda.OutOfMemoryError as e:
                # OOM 通常不适合盲目重试，这里直接抛出，上层会触发紧急逐出等措施
                raise

            except Exception as e:
                raise RuntimeError(f"H2D transfer error for {param_name}: {e}") from e

        raise RuntimeError(f"H2D transfer failed unexpectedly: {param_name}")

    def get_h2d_timeout_stats(self) -> dict:
        """获取 H2D 超时统计信息"""
        with self._h2d_stats_lock:
            return {
                "timeout_count": self._h2d_timeout_count,
                "retry_count": self._h2d_retry_count,
            }

    def report_h2d_stats(self) -> None:
        """打印 H2D 超时统计信息"""
        stats = self.get_h2d_timeout_stats()
        if stats["timeout_count"] > 0 or stats["retry_count"] > 0:
            print(f"[WSM][P0-3] H2D Stats - Timeouts: {stats['timeout_count']}, Retries: {stats['retry_count']}")

    # === 类内新增：进入 decoder 前，预热前 N 层（默认 4，安全上限 8） ===
    def _ensure_decoder_protect_window(self, first_n: int, reason: str = "decoder"):
        """
        确保 decoder 阶段的前 N 层长期保留在 CPU/GPU 环窗内，避免 wrap-around 时
        layer0/首层被立刻逐出。
        """
        if self.n_layers <= 0:
            return
        # n = max(1, min(int(first_n), self.n_layers))
        n = min(max(int(first_n), 0), self.n_layers)
        if n <= 0:
            self._cpu_protect_set.clear()
            return
        protect = set(range(n))

        # 1) 仅保留 0..n-1 的保护层，防止集合无限膨胀
        self._cpu_protect_set.intersection_update(protect)
        self._cpu_protect_set.update(protect)

        # 2) 确保这些层在 CPU cache 中
        if getattr(self, "ssd_enabled", False):
            for lid in protect:
                if lid not in self.cpu_cache:
                    try:
                        self._load_layer_to_cpu(lid)
                    except Exception:
                        if self.verbose:
                            print(f"[WSM][protect] failed to load layer {lid} into CPU cache")

        # 3) 触发异步预取，确保它们有事件可等待
        for lid in protect:
            self.prefetch_group_async(lid, "attn", reason=reason)
            self.prefetch_group_async(lid, "ffn", reason=reason)

    def _prime_decoder_window(self, first_n: int = 4):
        first_n = max(2, min(int(first_n), 8))
        # 1) 清伪就绪事件，避免“有事件但不在 GPU”
        self._clear_stale_group_events()
        try:
            self.pin_group(0, "ffn", reason="prime-decoder")
        except Exception:
            pass
        # 2) 保证首层窗口常驻（CPU + 异步预取）
        self._ensure_decoder_protect_window(first_n, reason="prime-decoder")
            
    def gpu_frontier(self) -> int:
        """
        返回"GPU 已驻留或在飞"组集合的环上最大层号，用于驱动 CPU 环窗。
        ★ 修复：加锁访问 _gpu_group_ring 和 _gpu_group_inflight
        """
        with self._group_lock:
            allL = [L for (L, _G) in list(self._gpu_group_ring)]
            allL += [L for (L, _G) in list(self._gpu_group_inflight)]

        if not allL:
            return int(getattr(self, "_last_executed_layer", 0))
        head = int(getattr(self, "_ring_head", getattr(self, "_current_layer", 0)))
        n = max(1, int(self.n_layers))
        def dist(L): return (int(L) - head) % n
        return max(allL, key=dist)

    # -------- GPU window: i.ffn(pin) + (i+1..i+4).attn --------
    
    # ========== 修改 5: 删除或简化 rebalance_and_topoff ==========
    def rebalance_and_topoff(self, current_layer: int) -> None:
        """
        ⚠️ 已废弃：GPU 窗口管理已迁移到 _slide_window_forward。
        如需，可仅用于环形回绕的 CPU 预热处理。
        """
        if not getattr(self, "ssd_enabled", False):
            return
        # 当接近尾部且下一轮将 wrap-around 时，提前保护并预热前若干层
        near_end = (current_layer >= self.n_layers - 5)
        if near_end and hasattr(self, "_detect_and_handle_wraparound"):
            self._detect_and_handle_wraparound(current_layer)
            
    def pump_gpu_window_prefetch(self, current_layer: int) -> None:
        """Strict GPU window: pin (i,'ffn') + prefetch (i+1..i+gpu_ahead,'attn')."""
        with self._group_lock:
            used = len(self._gpu_group_ring) + len(self._gpu_group_inflight)
        if used >= self.gpu_max_groups:
            # 超额时先收缩，再考虑补对 / 前瞻
            self._shrink_gpu_groups_now(exclude={(current_layer, 'attn'), (current_layer, 'ffn')})
            with self._group_lock:
                used = len(self._gpu_group_ring) + len(self._gpu_group_inflight)
            if used >= self.gpu_max_groups:
                return
        D = max(1, int(self.gpu_ahead_layers))
        # pair first (pin)
        self.prefetch_group_async(current_layer, "ffn", pin=True, reason="pair")
        # i+1..i+D ATTn
        for d in range(1, D+1):
            nxt = self._wrap(current_layer + d)
            self.prefetch_group_async(nxt, "attn", pin=False, reason=f"ring i+{d}.attn")
        # 轻量顶补（不改变窗口语义，仅补齐预算空位）
        try:
            if hasattr(self, "rebalance_and_topoff"):
                self.rebalance_and_topoff(current_layer)
        except Exception:
            pass

    # -------- CPU DRAM ring window: anchor = i+4, size = 40 --------
    def _schedule_cpu_ring_async(self, current_layer: int) -> None:
        """Async schedule a DRAM ring window [i+offset .. i+offset+cap-1] (mod n)."""
        if not getattr(self, "ssd_enabled", False):
            if self.verbose:
                print(f"[WSM DEBUG] _schedule_cpu_ring_async skipped: ssd_enabled=False")
            return
        if not self.cpu_ring_mode:
            if self.verbose:
                print(f"[WSM DEBUG] _schedule_cpu_ring_async skipped: cpu_ring_mode=False")
            return
        nL = int(self.n_layers)
        if nL <= 0:
            return
        # anchor = (int(current_layer) + int(self.cpu_ring_offset)) % nL
        i    = int(current_layer)
        offs = int(self.cpu_ring_offset)
        cap  = int(self.cpu_cache_cap)

        # ⭐ 关键修复：确保当前层总是在窗口内
        # 策略1: 如果窗口容量 ≥ 总层数，包含所有层（无需环形）
        if cap >= nL:
            anchor = 0
            target = set(range(nL))
        else:
            # 策略2: 窗口从当前层开始，覆盖 cap 层（环形）
            # ✅ 修复：anchor 使用 (i + offs)，但强制包含当前层
            anchor = (i + offs) % nL
            target = set(self._ring_range(anchor, cap))
            # ✅ 双重保险：无论 offset 如何，强制包含当前层和前后几层
            # 确保 GPU 需要的层（i-back_margin 到 i+gpu_ahead）都在窗口内
            safety_margin = max(int(getattr(self, 'cpu_back_margin', 4)), 2)
            gpu_ahead = max(int(getattr(self, 'gpu_ahead_layers', 4)), 2)
            for delta in range(-safety_margin, gpu_ahead + 1):
                target.add((i + delta) % nL)

        # ⭐ 关键修复：更新 cpu_win_base，确保 _layer_in_cpu_window 使用最新窗口
        self.cpu_win_base = anchor

        # ⭐ 关键修复：将 target 中的所有层添加到保护集，避免 worker 拒绝它们
        with self._cpu_lock:
            if not hasattr(self, '_cpu_protect_set'):
                self._cpu_protect_set = set()
            # 清理旧的保护集（只保留当前 target）
            self._cpu_protect_set = set(target)

        if self.verbose:
            print(f"[WSM DEBUG] _schedule_cpu_ring_async(L{current_layer}): anchor={anchor}, offset={self.cpu_ring_offset}, target={sorted(list(target))[:10]}...{sorted(list(target))[-3:]}")
            print(f"[WSM DEBUG] _schedule_cpu_ring_async: cpu_win_base={self.cpu_win_base}, protect_set size={len(self._cpu_protect_set)}")
        # 入队缺层 SSD->DRAM（避免重复/inflight）
        with self.cpu_cache_lock:
            present = set(self.cpu_cache.keys())
        missing = [L for L in target if (L not in present)]
        for L in missing:
            with self._cpu_lock:
                if L in self._inflight_cpu_layers:
                    continue
                with self._cpu_lock:  # 整个操作在锁内
                    self._inflight_cpu_layers.add(L)
                    epoch = self._epoch  # 同时读取 epoch
            try:
                 self._cpu_pf_q.put_nowait((epoch, int(L)))
            except Exception:
                with self._cpu_lock:
                    self._inflight_cpu_layers.discard(L)
                break
        # 淘汰环外层，保持 DRAM 环窗
        # ⭐ 修复：收集GPU上resident的层，避免驱逐它们
        gpu_resident_layers = set()
        for (layer, grp), state in self._group_state.items():
            if state in ("RESIDENT", "INFLIGHT"):
                gpu_resident_layers.add(layer)

        with self.cpu_cache_lock:
            for L in list(self.cpu_cache.keys()):
                if (L not in target) and (L not in self._cpu_protect_set) and (L not in gpu_resident_layers):
                    self.cpu_cache.pop(L, None)
                    if self.verbose:
                        print(f"[WSM] Evicted CPU cache layer {L} (ring)")



    def _ensure_gpu_headroom(self, required_bytes: int, exclude: set[tuple[int,str]] | None = None):
        guard = self._gpu_free_guard_mb * 1024 * 1024
        exclude = exclude or set()
        tries = 0
        while True:
            free_now = self._free_gpu_mem_bytes()
            if free_now >= required_bytes + guard:
                return
            # 第一轮：尊重留存
            if self._evict_one_group_from_gpu(exclude=exclude, ignore_retain=False):
                torch.cuda.empty_cache()
                tries += 1
                if tries > 64:
                    raise torch.cuda.OutOfMemoryError("eviction loop exceeded(pass0)")
                continue
            # 第二轮：忽略留存
            if self._evict_one_group_from_gpu(exclude=exclude, ignore_retain=True):
                torch.cuda.empty_cache()
                tries += 1
                if tries > 96:
                    raise torch.cuda.OutOfMemoryError("eviction loop exceeded(pass1)")
                continue
            # 仍不够：彻底 OOM
            torch.cuda.empty_cache()
            free_now = self._free_gpu_mem_bytes()
            if free_now >= required_bytes + guard:
                return
            raise torch.cuda.OutOfMemoryError(
                f"insufficient headroom: need={required_bytes/2**20:.2f}MB "
                f"free={free_now/2**20:.2f}MB guard={guard/2**20:.2f}MB")
    
    # ---- Public: ensure extra free space for upcoming compute ----------------
    def ensure_headroom_mb(self, mb: int, exclude: "set[tuple[int,str]]|None" = None):
        """
        计算前的“临时显存工作区”预留（额外 headroom）。
        会在不碰当前 in-use 组（以及 exclude 指定的组）的前提下，
        通过逐出其它组 + empty_cache 来腾挪出 mb MB 的额外空闲。
        """
        try:
            need = int(max(0, mb)) * 1024 * 1024
        except Exception:
            need = 0
        self._ensure_gpu_headroom(need, exclude=exclude or set())

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

    def _async_advance_cpu_window(self, current_layer: int):
        """
        后台线程：在 _slide_window_forward() 中被提交。
        目标：推进 CPU 侧的缓存窗口，使其与 GPU 前沿对齐。
        """
        debug = getattr(self, "debug_prefetch", False)
        try:
            # ✅ 在环形模式下，直接用环形的调度/确保，避免线性 [L0,L1] 语义干扰 wrap 行为
            if getattr(self, "cpu_ring_mode", False):
                if debug:
                    print(f"[WSM WINDOW ASYNC] (ring) ensure cpu window around L{current_layer}")
                self._schedule_cpu_ring_async(current_layer)  # 内部调用 _ensure_cpu_ring_window()
            else:
                # 仅非环形模式保留线性推进 + 线性 ensure
                self._advance_cpu_window(current_layer)
                self._ensure_cpu_window()

        except Exception as e:
            if getattr(self, "verbose", False):
                print(f"[WSM WINDOW ASYNC] async advance failed at L{current_layer}: {e}")
        finally:
            if debug:
                print(f"[WSM WINDOW ASYNC] ========== Slide Forward Done (async) ==========\n")

    def _target_cpu_window(self):
        """
        返回当前滑动窗口的范围 [L0, L1]
        """
        L0 = self.cpu_win_base
        L1 = min(self.n_layers - 1, self.cpu_win_base + self.cpu_cache_cap - 1)
        return L0, L1

    def _layer_in_cpu_window(self, layer_idx: int) -> bool:
        """
        环形或线性 CPU 窗口判定：
        - cpu_ring_mode=True 时，用环形区间 [base, base+cap) 判定；
        - 否则退回到线性 [L0, L1] 判定。
        """
        if getattr(self, "cpu_ring_mode", False):
            base = int(getattr(self, "cpu_win_base", 0))
            cap  = int(getattr(self, "cpu_cache_cap", getattr(self, "cpu_cache_layers", 40)))
            return self._ring_contains(base, int(layer_idx), cap)  # 利用 ((L-base)%n)<cap【turn22file16†weight_streaming_manager.py†L5-L10】
        L0, L1 = self._target_cpu_window()  # 线性窗口仍保留【turn23file5†weight_streaming_manager.py†L1-L18】
        return (int(layer_idx) >= L0) and (int(layer_idx) <= L1)

    def _ensure_cpu_window(self):
        """
        确保滑动窗口内的层都已加载到 CPU cache
        ★ 关键修复: 先逐出窗口外的层(跳过保护层)，再加载缺失层，保持容量恒定
        """
        L0, L1 = self._target_cpu_window()

        if getattr(self, "verbose", False):
            print(f"[WSM DEBUG] _ensure_cpu_window: window=[{L0}, {L1}], cursor={self._cpu_pf_cursor}, "
                  f"win_base={self.cpu_win_base}, cache_size={len(self.cpu_cache)}")

        # ★ 先清理窗口外的层（主动逐出，跳过保护）
        # ⭐ 修复：使用环形窗口判断，并保护GPU resident层
        base = int(getattr(self, "cpu_win_base", 0))
        cap = int(getattr(self, "cpu_cache_cap", 40))

        # ⭐ 修复：收集GPU上resident的层，避免驱逐它们
        gpu_resident_layers = set()
        for (layer, grp), state in self._group_state.items():
            if state in ("RESIDENT", "INFLIGHT"):
                gpu_resident_layers.add(layer)

        layers_to_evict = []
        for layer_id in list(self.cpu_cache.keys()):
            # 使用环形判断
            in_window = self._ring_contains(base, layer_id, cap)
            if not in_window:
                if layer_id in self._cpu_protect_set:
                    continue
                if layer_id in gpu_resident_layers:
                    continue
                layers_to_evict.append(layer_id)

        if layers_to_evict and getattr(self, "verbose", False):
            print(f"[WSM DEBUG] Evicting {len(layers_to_evict)} layers outside window [{L0}, {L1}]")
            print(f"[WSM DEBUG] Layers to evict: {sorted(layers_to_evict)}")
            print(f"[WSM DEBUG] Current cache keys: {sorted(list(self.cpu_cache.keys()))}")

        for layer_id in layers_to_evict:
            with self.cpu_cache_lock:
                self.cpu_cache.pop(layer_id, None)
            self._cpu_cached_layers.discard(layer_id)

        # ★ 修复: 游标只能在窗口内移动
        if self._cpu_pf_cursor > L1:
            if getattr(self, "verbose", False):
                print(f"[WSM DEBUG] Cursor {self._cpu_pf_cursor} > window end {L1}, resetting to {L0}")
            self._cpu_pf_cursor = L0
        else:
            old_cursor = self._cpu_pf_cursor
            self._cpu_pf_cursor = max(self._cpu_pf_cursor, L0)
            if old_cursor != self._cpu_pf_cursor and getattr(self, "verbose", False):
                print(f"[WSM DEBUG] Advanced cursor from {old_cursor} to {self._cpu_pf_cursor}")

        # 统计需要加载的层
        missing_layers = [L for L in range(L0, L1 + 1) if L not in self.cpu_cache]
        if missing_layers and getattr(self, "verbose", False):
            print(f"[WSM DEBUG] Need to load {len(missing_layers)} missing layers: {missing_layers[:5]}...")

        # 按序加载缺失层（从游标位置开始）
        for L in range(self._cpu_pf_cursor, L1 + 1):
            if L not in self.cpu_cache:
                # ★ 关键: 加载前检查容量，必要时先腾出空间(跳过保护层)
                while len(self.cpu_cache) >= self.cpu_cache_cap:
                    evict_layer = None
                    # 找到第一个非保护层
                    for cand in list(self.cpu_cache.keys()):
                        if cand not in self._cpu_protect_set:
                            evict_layer = cand
                            break
                    if evict_layer is None:
                        # 全是保护层，放弃逐出
                        break
                    if getattr(self, "verbose", False):
                        print(f"[WSM DEBUG] Cache full ({len(self.cpu_cache)}/{self.cpu_cache_cap}), evicting layer {evict_layer}")
                    with self.cpu_cache_lock:
                        self.cpu_cache.pop(evict_layer, None)
                    self._cpu_cached_layers.discard(evict_layer)

                self._load_layer_to_cpu(L)
                self._cpu_pf_cursor = L + 1

        # 如果窗口内所有层都已加载，游标推进到窗口末尾+1
        if self._cpu_pf_cursor <= L1:
            self._cpu_pf_cursor = L1 + 1

        if getattr(self, "verbose", False):
            print(f"[WSM DEBUG] _ensure_cpu_window done: cache_size={len(self.cpu_cache)}, cursor={self._cpu_pf_cursor}")
    def _ensure_cpu_ring_window(self, cur_layer: int):
        if not self.ssd_enabled:
            return
        # 锚点：i + offset
        anchor = (cur_layer + self.cpu_ring_offset) % self.n_layers
        target = self._ring_range(anchor, self.cpu_cache_cap)
        # 打印更清楚的日志
        if getattr(self, "verbose", False):
            preview = target[:min(len(target), 30)]
            print(f"[WSM][CPU-RING] cur={cur_layer} need={preview} ... | cap={self.cpu_cache_cap}, n={self.n_layers}")

        # 逐出不在 target 里的旧层(跳过保护层)
        for L in list(self.cpu_cache.keys()):
            if (L not in target) and (L not in self._cpu_protect_set):
                with self.cpu_cache_lock:
                    self.cpu_cache.pop(L, None)
                self._cpu_cached_layers.discard(L)

        # 按环形顺序补齐缺失
        for L in target:
            if L in self.cpu_cache:
                continue
            # 容量满则踢最老(跳过保护层)
            while len(self.cpu_cache) >= self.cpu_cache_cap:
                ev = None
                for cand in list(self.cpu_cache.keys()):
                    if cand not in self._cpu_protect_set:
                        ev = cand
                        break
                if ev is None:
                    break
                with self.cpu_cache_lock:
                    self.cpu_cache.pop(ev, None)
                self._cpu_cached_layers.discard(ev)
                if getattr(self, "verbose", False):
                    print(f"[WSM] CPU cache evict (ring): layer {ev}")
            # 只加载流式参数（SSD→CPU）
            self._load_layer_to_cpu(L)
    def _evict_if_over_hwm_locked(self, incoming: int = 0) -> None:
        """
        在持有 self._cpu_lock 的前提下调用：
        当 (当前已缓存层数 + incoming) 超过 CPU 高水位(HWM) 时，
        仅逐出“窗口外”的 LRU 层(且非保护层)，直到容量回到 LWM 或无可逐出项。
        """
        if not getattr(self, "ssd_enabled", False):
            return

        L0, L1 = self._target_cpu_window()

        cur = len(self.cpu_cache)
        if (cur + incoming) <= self.cpu_hwm:
            return  # 低于 HWM，无需处理

        target_max = max(self.cpu_lwm, 0)
        evicted = 0
        # 扫描缓存，逐出窗口外的层
        layers_to_evict = []

        # ⭐ 修复：使用环形窗口判断
        base = int(getattr(self, "cpu_win_base", 0))
        cap = int(getattr(self, "cpu_cache_cap", 40))

        # ⭐ 修复：收集GPU上resident的层，避免驱逐它们
        gpu_resident_layers = set()
        for (layer, grp), state in self._group_state.items():
            if state in ("RESIDENT", "INFLIGHT"):
                gpu_resident_layers.add(layer)

        for lyr in list(self.cpu_cache.keys()):
            if (len(self.cpu_cache) + incoming - len(layers_to_evict)) <= target_max:
                break
            # ⭐ 只踢窗口外 + 非保护层 + 非GPU resident层（使用环形判断）
            in_window = self._ring_contains(base, lyr, cap)
            if (not in_window) and (lyr not in self._cpu_protect_set) and (lyr not in gpu_resident_layers):
                layers_to_evict.append(lyr)

        for lyr in layers_to_evict:
            self.cpu_cache.pop(lyr, None)
            self._cpu_cached_layers.discard(lyr)
            evicted += 1

        if evicted and getattr(self, "verbose", False):
            print(f"[WSM] CPU cache evict (HWM): evicted={evicted}, after={len(self.cpu_cache)} "
                  f"(win=[{L0},{L1}], lwm={self.cpu_lwm}, hwm={self.cpu_hwm})")
    def _evict_cpu_layers(self, k: int):
        """
        优先踢窗口外层(且非保护层)；若不得不踢窗口内，则同步平移窗口并列出右端 must-fetch
        """
        L0, L1 = self._target_cpu_window()
        evicted = 0
        must_fetch = 0

        if getattr(self, "verbose", False):
            print(f"[WSM] Evicting {k} CPU layers, window=[{L0}, {L1}], cache_size={len(self.cpu_cache)}")

        # Phase 1: 优先踢窗口外的层（跳过保护层）
        # ⭐ 修复：使用环形窗口判断
        base = int(getattr(self, "cpu_win_base", 0))
        cap = int(getattr(self, "cpu_cache_cap", 40))

        # ⭐ 修复：收集GPU上resident的层，避免驱逐它们
        gpu_resident_layers = set()
        for (layer, grp), state in self._group_state.items():
            if state in ("RESIDENT", "INFLIGHT"):
                gpu_resident_layers.add(layer)

        layers_to_evict = []
        for L in list(self.cpu_cache.keys()):
            if evicted >= k:
                break
            # ⭐ 使用环形判断 + 保护GPU resident层
            in_window = self._ring_contains(base, L, cap)
            if (not in_window) and (L not in self._cpu_protect_set) and (L not in gpu_resident_layers):
                layers_to_evict.append(L)
                evicted += 1

        for L in layers_to_evict:
            with self.cpu_cache_lock:
                self.cpu_cache.pop(L, None)
            self._cpu_cached_layers.discard(L)
            if getattr(self, "verbose", False):
                print(f"[WSM] CPU cache evict (out of window): layer {L}")

        # Phase 2: 如果还不够，平移窗口（右移 d 层），并把右端 d 层标记为 must-fetch
        if evicted < k:
            d = min(k - evicted, self.cpu_cache_cap)  # 最多平移一个窗口宽
            self.cpu_win_base += d
            must_fetch = d
            if getattr(self, "verbose", False):
                print(f"[WSM] Window shift: base {self.cpu_win_base - d} -> {self.cpu_win_base}, must_fetch={must_fetch}")

        # 平移后确保窗口（含 must-fetch）
        if must_fetch > 0:
            self._ensure_cpu_window()
            
    # --- in weight_streaming_manager.py (inside class WeightStreamingManager) ---

    def _evict_cpu_layer_immediately(self, layer_idx: int, ignore_protect: bool = True) -> int:
        """
        立刻从 CPU cache 移除指定层（可忽略保护集），返回是否成功移除(0/1)。
        """
        if not getattr(self, "ssd_enabled", False):
            return 0
        removed = 0
        with self.cpu_cache_lock:
            if (not ignore_protect) and (layer_idx in getattr(self, "_cpu_protect_set", set())):
                return 0
            if layer_idx in self.cpu_cache:
                self.cpu_cache.pop(layer_idx, None)
                removed = 1
        # 同步 membership 集
        try:
            self._cpu_cached_layers.discard(layer_idx)
        except Exception:
            pass
        if removed and getattr(self, "verbose", False):
            print(f"[WSM] CPU evict-on-finish: layer {layer_idx}")

        # ⭐ Sliding Window: CPU驱逐后prefetch下一层到DRAM
        # ⚠️ 已禁用：rebalance_and_topoff 会统一管理CPU cache的prefetch
        # 这段逻辑在prefill阶段会导致错误的prefetch行为（基于max层号而不是当前计算位置）
        # if removed:
        #     try:
        #         max_cpu_layer = self._get_max_loaded_cpu_layer()
        #         nextL = max_cpu_layer + 1
        #
        #         if nextL < self.n_layers:
        #             # 检查是否已在cache或inflight
        #             need_prefetch = False
        #             with self.cpu_cache_lock:
        #                 if nextL not in self.cpu_cache:
        #                     with self._cpu_lock:
        #                         if nextL not in self._inflight_cpu_layers:
        #                             need_prefetch = True
        #
        #             if need_prefetch:
        #                 if self.verbose:
        #                     print(f"[WSM][CPU-EVICT→PREFETCH] Evicted L{layer_idx} from DRAM, "
        #                           f"prefetch L{nextL} (max_cpu_layer={max_cpu_layer})")
        #                 self._cpu_try_enqueue(nextL, reason="cpu-evict->ssd-prefetch")
        #         else:
        #             if self.verbose:
        #                 print(f"[WSM][CPU-EVICT→PREFETCH] Evicted L{layer_idx} from DRAM, "
        #                       f"reached end (max_cpu_layer={max_cpu_layer}, n_layers={self.n_layers})")
        #     except Exception as e:
        #         if self.verbose:
        #             print(f"[WSM][CPU-EVICT→PREFETCH] Failed: {e}")

        return removed
        

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

        #  去重 - 检查是否已在加载中
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
                if old_layer in self._cpu_protect_set:
                    # 放回去，找下一个可逐出的
                    self.cpu_cache[old_layer] = _
                    # 线性扫描找一个不在保护集的
                    victim = next((L for L in self.cpu_cache.keys() if L not in self._cpu_protect_set), None)
                    if victim is None: break
                    self.cpu_cache.pop(victim, None)
                    old_layer = victim

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
                self._cpu_cached_layers.add(layer_idx)
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

        # ⭐ 检查是否使用环形窗口模式（CPU cache < 总层数）
        # 环形窗口模式下，prefetch_distance=0 表示"使用组级预取，不使用整层预取"
        # 而不是"把所有权重都放在DRAM"
        ring_mode = os.getenv("WSM_CPU_RING_MODE", "0") == "1"
        using_cpu_window = self.cpu_cache_layers < len(self.layers_params)

        if ring_mode or using_cpu_window:
            print(f"🔄 Ring Window / CPU Streaming Mode:")
            print(f"   CPU cache: {self.cpu_cache_layers} layers (out of {len(self.layers_params)})")
            print(f"   Ring mode: {ring_mode}")
            print(f"   Using group-level prefetch (prefetch_distance=0 for layer-level)")
            print(f"✅ Streaming mode validated - weights will be loaded on-demand")
            return

        # 原有的no-prefetch模式验证逻辑（全量DRAM检查）
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


    def note_compute_advance(self, layer_idx: int) -> None:
        """
        在 layer 计算刚要开始时调用（由 layers.py 触发）：
        - 更新锚点（供窗口与回绕逻辑使用）
        - 立即泵一次 GPU 窗口：pin(i.ffn) + 预取(i+1..i+D).attn
        - 推进 CPU 环形窗口：anchor = i + cpu_ring_offset
        """
        layer_idx = int(layer_idx)
        with self._anchor_lock:
            self._current_layer = layer_idx
            self._last_executed_layer = layer_idx

        # 1) GPU：成对 + 前瞻
        try:
            self.pump_gpu_window_prefetch(layer_idx)  # i.ffn pin + (i+1..i+D).attn 预取
        except Exception as e:
            if self.verbose:
                print(f"[WSM] note_compute_advance: pump failed at L{layer_idx}: {e}")

        # 2) CPU：以 i+offset 为锚推进 DRAM 环窗
        try:
            self._schedule_cpu_ring_async(layer_idx)  # [i+off .. i+off+cap-1]
        except Exception as e:
            if self.verbose:
                print(f"[WSM] note_compute_advance: cpu-ring failed at L{layer_idx}: {e}")




    def _advance_cpu_window_by_compute(self, cur_layer: int):
        """
        仅由“计算线程”调用：推进 CPU 预取窗口并把缺失层入队。
        不做任何同步 IO（不直接 _load_layer_to_cpu）。
        """
        with self._cpu_lock:
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

        if getattr(self, "cpu_ring_mode", False):
                # 环形窗口：在 cur+offset 起的环上取 cpu_cache_cap 个层
                self._ensure_cpu_ring_window(current_layer)
                
        else:

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
                    with self._cpu_lock:  # 添加锁保护
                        if new_base > self.cpu_win_base:  # 双重检查
                            print(f"[WSM DEBUG] Layer {current_layer} near/beyond window end {L1}, advancing base {self.cpu_win_base} -> {new_base}")
                            self.cpu_win_base = new_base

            # # 确保窗口内的层都已加载
            # self._ensure_cpu_window()
            
            # 不在前向线程里做 SSD 同步读！
            # 仅推进窗口基准与“缺失层排队”，由 _cpu_prefetch_worker 后台加载
            self._advance_cpu_window_by_compute(current_layer)


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

        resident_layers = 0
        while time.time() - start_time < timeout:
            resident_layers = 0
            if getattr(self, "grouped_mode", False):
                layer_groups: Dict[int, set[str]] = defaultdict(set)
                for lyr, grp in list(self._gpu_group_ring):
                    layer_groups[int(lyr)].add(str(grp))
                resident_layers = sum(1 for groups in layer_groups.values()
                                      if ("attn" in groups and "ffn" in groups))
            gpu_ready = resident_layers >= self.target_gpu_layers
            
            if getattr(self, "grouped_mode", False):
                # 目标组数：每层 2 组（attn+ffn），受组预算上限约束；至少需要 1 组即可开跑
                group_target = min(max(1, 2 * max(1, self.target_gpu_layers)), self.gpu_max_groups)
                group_ready  = len(self._gpu_group_ring) >= group_target or ((0, "attn") in self._gpu_group_ring)
                gpu_ready = gpu_ready or group_ready
            
            
            cpu_ready = len(self.cpu_cache) >= self.target_cpu_layers if self.ssd_enabled else True

            if gpu_ready and cpu_ready:
                if self.verbose:
                    print(f"[WSM] Preload completed: {resident_layers} GPU layers (groups={len(self._gpu_group_ring)}) + {len(self.cpu_cache)} CPU layers ready")
                self.gpu_preload_complete.set()
                self.cpu_preload_complete.set()
                return True

            if self.verbose and int(time.time() - start_time) % 5 == 0:  # Progress update every 5 seconds
                print(f"[WSM] Preload progress: GPU {resident_layers}/{self.target_gpu_layers} (groups={len(self._gpu_group_ring)}), CPU {len(self.cpu_cache)}/{self.target_cpu_layers}")

            time.sleep(0.1)

        # ★ 修复 9: 超时时给出建议
        print(f"[WSM] ⚠️  Preload timeout after {timeout}s: GPU {resident_layers}/{self.target_gpu_layers}, CPU {len(self.cpu_cache)}/{self.target_cpu_layers}")
        print(f"[WSM] 💡 Tip: Set WSM_SKIP_PRELOAD_WAIT=1 to skip waiting and prefetch on-the-fly")
        print(f"[WSM] 💡 Or set WSM_PRELOAD_TIMEOUT=<seconds> to adjust timeout")
        return False

    # -------- CPU/GPU movement primitives --------

    def _setup_resident_norms(self):
        """
        Move norm modules to GPU and exclude them from streaming/eviction.
        ★ 修复 7: 遵守预算上限，防止碎片和 OOM
        ★ 关键改进：
          1. 使用 copy_stream + non_blocking=True 异步上卡，降低初始化延迟
          2. 更新 name_to_param / param_owner 映射，确保参数引用一致性
          3. norm 层永不被 _evict_group_immediately 驱逐（仅处理 attn/ffn 组）
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
                        # ★ 使用异步拷贝流 + non_blocking 上卡（减少阻塞）
                        if self._copy_stream is not None:
                            with torch.cuda.stream(self._copy_stream):
                                for pname, p in module.named_parameters(recurse=True):
                                    if p.device.type != "cuda":
                                        # 异步拷贝到 GPU
                                        p_gpu = p.detach().to(self.device, non_blocking=True)
                                        # 替换模块内的参数引用
                                        module._parameters[pname.split('.')[-1]] = torch.nn.Parameter(
                                            p_gpu, requires_grad=p.requires_grad
                                        )

                                        # ★ 更新全局映射：确保后续查询拿到的是 GPU 版本
                                        full_name = f"layers.{layer_id}.{module_name.replace('norm_', '')}.{pname}"
                                        self.name_to_param[full_name] = module._parameters[pname.split('.')[-1]]
                                        self.param_owner[full_name] = (module, pname.split('.')[-1])
                        else:
                            # 回退：同步拷贝（CPU 模式或无 CUDA）
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
            print("[WSM] ✓ Norm layers are protected from eviction (only attn/ffn groups are evicted)")

    def _check_and_throttle_kv(self):
        """
        ★ 修复 8: 检查 weight_h2d backlog，必要时 throttle KV I/O
        避免 KV 抢带宽导致权重迟迟不上来
        """
        # 同时观察两条 H2D 流
        s_mha = getattr(self.streams, "weight_h2d_mha", None)
        s_ffn = getattr(self.streams, "weight_h2d_ffn", None)
        def _busy(s):
            try:
                return (s is not None) and (not s.query())
            except Exception:
                return False
        is_busy = _busy(s_mha) or _busy(s_ffn)

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
        """
        在两条权重 H2D 流（MHA/FFN）上各记一次“层就绪”事件。
        计算侧需要同时等待这两个事件，避免半层缺权重导致隐式同步。
        """
        if not torch.cuda.is_available():
            return
        evts = []
        mha = getattr(self.streams, "weight_h2d_mha", None)
        ffn = getattr(self.streams, "weight_h2d_ffn", None)
        for s in (mha, ffn):
            if s is None:
                continue
            e = torch.cuda.Event(blocking=False)
            e.record(s)
            evts.append(e)
        if evts:
            self.layer_events[idx] = tuple(evts) if len(evts) > 1 else evts[0]

        # PCIE 背压：如 H2D backlog 偏大，柔性节流 KV 写
        self._check_and_throttle_kv()  # 已在项目中实现
        try:
            # 轻量 GC 事件池，避免 pending 增长
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
            with torch.cuda.device(self.device_index):
                st = torch.cuda.current_stream()
                if isinstance(evt, tuple):
                    for e in evt:
                        st.wait_event(e)
                else:
                    st.wait_event(evt)
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

    def _cpu_cache_tensor(self, layer_idx: int, param_name: str) -> Optional[torch.Tensor]:
        with self.cpu_cache_lock:
            layer_entry = self.cpu_cache.get(layer_idx)
            if layer_entry:
                tensor = layer_entry.get(param_name)
                if tensor is not None:
                    return tensor
        return None

    def _ensure_param_on_gpu(
        self,
        p: torch.nn.Parameter,
        layer_idx: Optional[int] = None,
        param_name: Optional[str] = None,
    ):
        """
        确保参数在 GPU 上（从 CPU cache 加载）。
        ★ 约定：这里只做“最佳努力”的非阻塞 H2D，不再在这里等待 CPU/SSD I/O。
        真正的分层预取由 sliding window + CPU/H2D worker 负责。
        """
        nvtx.range_push("param_h2d")

        # 已经在 GPU 上且非 stub：直接返回
        if p.is_cuda and p.numel() > 0:
            nvtx.range_pop()
            return

        # meta/stub 参数：只在有 CPU cache 命中的情况下才尝试搬运
        if self.ssd_enabled and layer_idx is not None and param_name is not None:
            try:
                layer_entry = self.cpu_cache.get(layer_idx, None)
            except Exception:
                layer_entry = None

            if layer_entry is not None:
                cached_tensor = layer_entry.get(param_name)
                if cached_tensor is not None:
                    # 非阻塞 H2D：挂在对应 H2D 流上，让事件和窗口去同步
                    nvtx.range_push("cpu_cache_to_gpu")
                    h2d_stream = self._select_h2d_stream_for(name=param_name)
                    with torch.cuda.stream(h2d_stream):
                        p_gpu = cached_tensor.to(self.device, non_blocking=True)
                    p.data = p_gpu
                    nvtx.range_pop()
                    nvtx.range_pop()
                    return

            # 没在 CPU cache 里：不要在这里触发 SSD 读或等待 inflight
            if self.verbose:
                print(
                    f"[WSM] _ensure_param_on_gpu: CPU cache miss for "
                    f"{param_name} (layer {layer_idx}); "
                    f"will rely on sliding-window prefetch."
                )
            nvtx.range_pop()
            return

        # 非 SSD 模式：从 CPU 参数直接异步 H2D
        if p.device.type == "cpu":
            nvtx.range_push("cpu_param_to_gpu")
            try:
                h2d_stream = self._select_h2d_stream_for(name=param_name)
                with torch.cuda.stream(h2d_stream):
                    p_gpu = p.to(self.device, non_blocking=True)
                p.data = p_gpu
            finally:
                nvtx.range_pop()
                nvtx.range_pop()
            return

        # 其它情况：不做任何事（保持原状）
        nvtx.range_pop()


    def _evict_param_to_cpu(self, p: torch.nn.Parameter, pname: str | None = None):
        """
        驱逐参数出 GPU：设为 0-size CPU stub，并 bump 版本（A2）。
        """
        try:
            if pname is None:
                pname = self._guess_param_name(p)
            stub = torch.empty(0, dtype=p.dtype, device="cpu")
            with self._group_lock:
                # 逐出视为一次“写”：防止随后落地的老版本覆盖
                if pname is not None:
                    self._param_version[pname] = int(self._param_version.get(pname, 0)) + 1
                p.data = stub
        except Exception:
            # 兜底
            p.data = torch.empty(0, dtype=p.dtype, device="cpu")

    def _ensure_module_on_gpu(self, m: nn.Module, layer_idx: Optional[int] = None, module_name: Optional[str] = None):
        """Ensure all params/buffers of module m are on GPU."""
        # For modules, we need to handle parameter replacement differently
        # because meta parameters cannot be assigned via .data =

        params_to_replace = {}  # {local_param_name: new_param}
        params_full_names = {}  # {local_param_name: full_param_name}

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
            is_cpu_stub = (p.device.type == "cpu") and (p.numel() == 0)
            if (is_meta or is_cpu_stub) and self.ssd_enabled and full_param_name:
                cached_tensor = None
                if layer_idx is not None:
                    cached_tensor = self._cpu_cache_tensor(layer_idx, full_param_name)
                    if cached_tensor is None:
                        try:
                            self._wait_cpu_ready(layer_idx, timeout=5.0)
                        except Exception:
                            pass
                        cached_tensor = self._cpu_cache_tensor(layer_idx, full_param_name)
                    if cached_tensor is None:
                        try:
                            cached_tensor = self._load_param_from_ssd(full_param_name)
                            if cached_tensor is not None:
                                with self.cpu_cache_lock:
                                    self.cpu_cache.setdefault(layer_idx, {})[full_param_name] = cached_tensor
                        except Exception:
                            cached_tensor = None

                if cached_tensor is not None:
                    # expected = tuple(getattr(getattr(m, local_param_name), "shape", ()))
                    param_obj = getattr(m, local_param_name)
                    expected = getattr(param_obj, "_shape_hint", None)
                    if expected is None:
                        expected = tuple(getattr(param_obj, "shape", ()))
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
                    with torch.cuda.stream(self._select_h2d_stream_for(module_name=module_name)):
                        p_gpu = chosen_tensor.to(self.device, non_blocking=True)
                        
                    params_to_replace[local_param_name] = nn.Parameter(p_gpu, requires_grad=p.requires_grad)
                    # params_full_names[local_param_name] = full_param_name
                    params_full_names[local_param_name] = chosen_name if 'chosen_name' in locals() else full_param_name
                    
                    if self.verbose:
                        print(f"[WSM DEBUG] ✓ Loaded meta param {full_param_name} to GPU: {p_gpu.shape}")
                    if chosen_name != full_param_name and self.verbose:
                        print(f"[WSM] ⚠️ shape-fix: remapped {full_param_name} -> {chosen_name}")
                else:
                    if self.verbose:
                        print(f"[WSM WARN] CPU cache miss for {full_param_name} (layer {layer_idx}); will retry later")
            else:
                # Regular parameter - use standard method
                self._ensure_param_on_gpu(p, layer_idx, full_param_name)

        # Replace meta parameters
        # for param_name, new_param in params_to_replace.items():
        #     # 使用 _parameters 字典直接替换，这是 PyTorch 的内部机制
        #     m._parameters[param_name] = new_param

        #     # 更新全名映射（影响驱逐正确性）
        #     full_param_name = params_full_names.get(param_name)
        #     if full_param_name:
        #         # 更新 name_to_param：全名 -> Parameter 对象
        #         self.name_to_param[full_param_name] = getattr(m, param_name)
        #         # 更新 param_owner：全名 -> (module, attr_name)
        #         self.param_owner[full_param_name] = (m, param_name)
        
        for param_name, new_param in params_to_replace.items():
            # 1) 替换到模块
            m._parameters[param_name] = new_param
            # 2) 更新全名映射
            full_param_name = params_full_names.get(param_name)
            if full_param_name:
                # name -> Parameter 对象
                try:
                    pobj = getattr(m, param_name)
                except Exception:
                    pobj = new_param
                self.name_to_param[full_param_name] = pobj
                # name -> (module, attr)
                self.param_owner[full_param_name] = (m, param_name)
                
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
                with torch.cuda.stream(self._select_h2d_stream_for(module_name=module_name)):
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

    def _ring_range(self, start: int, count: int) -> list[int]:
        n = self.n_layers
        return [ (start + k) % n for k in range(count) ]


    def _pre_hook_factory(self, idx: int):
        def _pre_hook(_module, _inputs):
            
            if getattr(self, "debug_prefetch", False):
                print(f"[WSM DEBUG] pre_hook fired: layer={idx}, grouped_mode={getattr(self, 'grouped_mode', False)}")

            if not getattr(self, "_bootstrap_done", False) and idx == 0:
                # 1) CPU 首窗：只会做一次，内部幂等
                self.warmup_cpu_cache()   # 见: warmup_cpu_cache()

                # 2) GPU 组级灌窗：把 L0.. 的 (attn, ffn) 直接丢到异步预取
                layers = min(self.warmup_layers_gpu, self.n_layers)
                self.warmup_groups_prefetch(layers=layers, scheme="doublet")  # 直接用已有的 warmup 组预取

                # 3) 保护 decoder 首 N 层（避免 wrap 时把 layer0 立刻踢掉）
                self._ensure_decoder_protect_window(first_n=min(8, layers), reason="prefill-bootstrap")  #

                self._bootstrap_done = True

            # 1) 轻量维护：清理非 resident 的组事件（避免 warmup 遗留）
            try:
                self._clear_stale_group_events()
            except Exception:
                pass

            # 3) 轻量前置清理：窗口外的旧组做一轮"主动驱逐"，为 (idx,*) 腾位
            try:
                self._proactive_cleanup_old_groups(idx)
            except Exception:
                pass

            # 4) CPU：以 i+offset 为锚的 40 层环窗，异步入队（SSD→DRAM）
            if self.ssd_enabled:
                if getattr(self, "cpu_ring_mode", False):
                    self._schedule_cpu_ring_async(idx)
                else:
                    self._advance_cpu_window_by_compute(idx)

            # 5) GPU：在 ATTN→FFN 缝隙保护同层 FFN
            try:
                self.pin_group(idx, "ffn", reason="pair")
            except Exception:
                pass

            # 6) GPU 侧：成对 + 最近优先的统一泵
            depth = int(getattr(self, "group_prefetch_depth", 2))
            plan = self._plan_pairwise_nearest(idx, depth)
            for (L, G) in plan:
                try:
                    self.prefetch_group_async(L, G, reason="prehook_pair")
                except Exception:
                    pass

            # 7) ★ 用 GPU frontier 驱动 CPU 环窗
            try:
                front = self.gpu_frontier()
                self._schedule_cpu_ring_async(int(front))
            except Exception:
                pass
        return _pre_hook

    def ensure_on_gpu(self, idx: int, wait: bool):
        """Compatibility shim: ensure both attn/ffn groups are queued/resident."""
        if not getattr(self, "grouped_mode", False):
            return

        reason = "ensure_layer_block" if wait else "ensure_layer_prefetch"
        for grp in ("attn", "ffn"):
            try:
                self.prefetch_group_async(idx, grp, pin=True, reason=reason)
            except Exception:
                pass
            if wait:
                try:
                    self.wait_group_ready(idx, grp, compute_stream=None)
                except Exception:
                    pass
        
        
    def _bytes_of(self, t):
        itemsize = {torch.float16:2, torch.bfloat16:2, torch.float32:4, torch.int8:1,
                    torch.uint8:1, torch.int32:4, torch.int64:8, torch.float64:8}.get(t.dtype, 2)
        return t.numel() * itemsize

    # ---------------- Group residency introspection & printing ----------------
    
    def _name_to_group_key(self, pname: str) -> tuple[int, str] | None:
        """从参数全名推断 (L, 'attn'/'ffn') 组键；非分组权重返回 None。"""
        try:
            if not pname.startswith("layers."): 
                return None
            parts = pname.split(".")
            L = int(parts[1])
            if ".attention." in pname:
                return (L, "attn")
            if ".feed_forward." in pname:
                return (L, "ffn")
            return None
        except Exception:
            return None

    def _get_state(self, key: tuple[int,str]) -> str:
        return self._group_state.get((int(key[0]), str(key[1])), "CPU")

    def _set_state(self, key: tuple[int,str], new_state: str):
        self._group_state[(int(key[0]), str(key[1]))] = new_state

    def _pick_h2d_stream(self, group: str):
        """
        按组选择 H2D 流（FlexGen 风格：MHA/FFN 分路）。
        Returns the appropriate H2D stream for the given group, or None for default stream.
        """
        s = self.streams
        if group == "attn" and getattr(s, "weight_h2d_mha", None):
            return s.weight_h2d_mha
        if group == "ffn" and getattr(s, "weight_h2d_ffn", None):
            return s.weight_h2d_ffn
        return None  # 回退到默认流/当前流

    def _guess_param_name(self, p: torch.nn.Parameter) -> str | None:
        """从 name_to_param 反查名字（O(N)，仅逐出兜底时使用）。"""
        try:
            for n, obj in self.name_to_param.items():
                if obj is p:
                    return n
        except Exception:
            pass
        return None

    
    def _snapshot_gpu_groups(self):
        """
        返回当前在 GPU 的组快照：[(layer, 'attn'/'ffn', in_use_bool, inflight_bool), ...]
        顺序按照内部环列当前排列（仅用于调试，不再表示 LRU）。
        ★ 修复：统一加锁复制快照，避免并发访问导致的数据不一致
        """
        with self._group_lock:
            # 一次性复制所有需要的数据结构快照
            ring_snapshot = list(self._gpu_group_ring)
            inflight_snapshot = set(self._gpu_group_inflight)
            in_use_snapshot = dict(self._gpu_group_in_use)

        # 释放锁后进行格式化处理
        lst = []
        for (lyr, grp) in ring_snapshot:
            key = (int(lyr), str(grp))
            in_use = in_use_snapshot.get(key, 0) > 0
            inflight = key in inflight_snapshot
            lst.append((key[0], key[1], in_use, inflight))
        return lst


    def _snapshot_cpu_groups(self):
        """
        返回当前在 CPU cache 中的组快照：
        [(layer, 'attn'/'ffn', present_cnt, total_cnt), ...]
        只要该组至少有一个参数在 CPU cache 中就会展示；便于观察"部分命中"。
        ★ 修复：加锁访问 cpu_cache，避免并发修改导致的数据不一致
        """
        summary = []
        with self.cpu_cache_lock:
            # 一次性复制 cpu_cache 的快照
            cache_snapshot = {k: dict(v) if isinstance(v, dict) else v
                            for k, v in getattr(self, "cpu_cache", {}).items()}

        # 释放锁后进行分析
        for lyr in sorted(cache_snapshot.keys()):
            layer_dict = cache_snapshot.get(lyr, {})
            if not isinstance(layer_dict, dict):
                continue
            for grp_name, suffixes in GROUPS.items():
                tot = len(suffixes)
                hit = 0
                for suf in suffixes:
                    if f"layers.{lyr}.{suf}" in layer_dict:
                        hit += 1
                if hit > 0:
                    summary.append((int(lyr), grp_name, hit, tot))
        return summary

    def print_group_residency(self, current: tuple[int, str] | None = None, header: str | None = None):
        """
        打印“当前 GPU/CPU 上有哪些组”的一行摘要。
        - current: 可选，用来高亮当前正在计算的 (layer, group)
        - header:  可选，自定义前缀；默认 '[WSM][groups]'
        """
        try:
            gpu = self._snapshot_gpu_groups()
            cpu = self._snapshot_cpu_groups()
        except Exception as e:
            print(f"[WSM][groups] snapshot failed: {e}")
            return

        def _fmt_gpu(item):
            # ★ 修复：移除直接访问 self._gpu_group_inflight（无锁访问）
            # _snapshot_gpu_groups() 已经返回包含 inflight 的4元组
            if len(item) == 3:
                # 兼容旧版本快照（无 inflight 信息）
                L, G, in_use = item
                inflight = False  # 旧快照无此信息，默认 False
            else:
                L, G, in_use, inflight = item
            star = "*" if in_use else ""
            dot  = "…" if (not in_use and inflight) else ""
            cur  = "⚑" if (current is not None and (L, G) == current) else ""
            return f"L{L}.{G}{star}{cur}{dot}"

        def _fmt_cpu(item):
            L, G, hit, tot = item
            cur  = "⚑" if (current is not None and (L, G) == current) else ""
            suffix = "" if hit == tot else f"({hit}/{tot})"
            return f"L{L}.{G}{suffix}{cur}"

        gpu_txt = ", ".join(_fmt_gpu(x) for x in gpu) if gpu else "—"
        cpu_txt = ", ".join(_fmt_cpu(x) for x in cpu) if cpu else "—"
        pfx = header or "[WSM][groups]"
        print(f"{pfx} GPU: {gpu_txt} | CPU: {cpu_txt}")

    
    def _mark_group_in_use(self, layer_idx: int, group: str):
        """标记组为使用中（引用计数），防止被淘汰）
        - 变更：移除‘标记前主动清理’；清理改到 wait_group_ready 的 ready 之后进行。
        """
        key = self._key(layer_idx, group)
        with self._group_lock:
            self._gpu_group_in_use[key] = self._gpu_group_in_use.get(key, 0) + 1
        self._touch_group(layer_idx, group)
        if self.verbose:
            print(f"[WSM] Marked group {key} as IN_USE (refcount={self._gpu_group_in_use[key]})")
        try:
            if os.getenv("WSM_PRINT_GROUPS", "1") == "1":
                self.print_group_residency(current=key)
        except Exception:
            pass


    def _unmark_group_in_use(self, layer_idx: int, group: str):
        """解除组的使用标记（减少引用计数）；真正的逐出由滑动窗口与 headroom 逻辑统一管理。"""
        key = self._key(layer_idx, group)

        # 引用计数递减
        with self._group_lock:
            c = self._gpu_group_in_use.get(key, 0)
            if c <= 1:
                self._gpu_group_in_use.pop(key, None)
            else:
                self._gpu_group_in_use[key] = c - 1
                self._touch_group(layer_idx, group)

        # 组计算刚结束再 touch 一次，避免刚算完就被当成“最老”踢掉
        self._touch_group(layer_idx, group)

        if self.verbose:
            print(f"[WSM] Unmarked group {key} from IN_USE (refcount={self._gpu_group_in_use.get(key, 0)})")

        # 这里不再直接驱逐 group，只做轻量 shrink（保持 GPU 预算）
        try:
            pair_key = self._pair_key(layer_idx, group)
            pair_protect = {pair_key} if pair_key is not None else set()
        except Exception:
            pair_protect = set()

        try:
            self._shrink_gpu_groups_now(exclude={key} | pair_protect)
        except Exception as e:
            if self.verbose:
                print(f"[WSM] _shrink_gpu_groups_now failed after unmark {key}: {e}")

        # CPU 侧仅做 older-than 修剪（可选），不强制 evict 当前层
        if group == "ffn" and self.ssd_enabled:
            try:
                if getattr(self, "cpu_evict_after_use", False):
                    self._evict_cpu_layer_immediately(layer_idx, ignore_protect=True)
                else:
                    self._cpu_trim_to_window_if_needed(force=False, reason="unmark")
            except Exception as e:
                if self.verbose:
                    print(f"[WSM] CPU evict-on-finish failed: {e}")



    def _evict_group_immediately(self, layer_idx: int, group: str, skip_prefetch: bool = False):
        """
        立即驱逐指定组到 CPU。

        Args:
            layer_idx: 层索引
            group: 组名 ('attn' 或 'ffn')
            skip_prefetch: 如果为 True，驱逐后不触发下一层的 prefetch（用于批量清理时避免内存爆炸）
        """
        key = (layer_idx, group)
        if self.verbose or True:  # ⭐ 临时强制打印，用于调试
            print(f"[WSM][EVICT] Evicting {key} (skip_prefetch={skip_prefetch})")

        # 1) 驱逐参数到CPU
        for suf in GROUPS[group]:  # 例如 'attn' -> wq, wk, wv, wo
            name = f"layers.{layer_idx}.{suf}"
            p = self.name_to_param.get(name)
            if p is not None and p.is_cuda and p.numel() > 0:
                self._evict_param_to_cpu(p)

        with self._group_lock:
            # 2) 清理状态（无论是否在ring中都要执行）
            # ⭐ 修复：无条件更新状态和清理inflight，避免"僵尸"组
            self._set_state(key, "CPU")
            self._gpu_group_inflight.discard(key)
            self._group_events.pop(key, None)
            self._group_recorded_host.pop(key, None)  # ✅ 避免残留 host 事件

            # 3) 从ring中移除
            if key in self._gpu_group_ring:
                self._gpu_group_ring.remove(key)

            # ⭐ Sliding Window策略已禁用
            # ⚠️ 原设计：驱逐后prefetch max_loaded_layer + 1
            # 问题：在prefill阶段会导致错误的prefetch（基于GPU最大层号而不是当前计算位置）
            # 解决：由 rebalance_and_topoff 统一管理prefetch
            #
            # 原来的逻辑：
            # 4) 尝试prefetch下一层（除非 skip_prefetch=True）
            # - 获取GPU上已加载的最大层号 max_gpu_layer
            # - prefetch nextL = max_gpu_layer + 1
            #
            # 现已完全禁用，prefetch由 _unmark_group_in_use → rebalance_and_topoff 负责

            # ⭐ 修复：驱逐总是成功的，返回True表示驱逐完成
            return True


    def _group_is_resident(self, layer_idx: int, group: str, wait_for_event: bool = False) -> bool:
        """
        该组(attn/ffn)的所有参数是否已在 GPU 且为非空张量。

        核心改动：延迟 RESIDENT 提交 —— 只有当事件真正完成才算 resident。
        - RESIDENT 状态：若仍有事件且未完成，则视作未完成（避免提前"已就绪"）
        - INFLIGHT 状态：只有当事件真正完成才算 resident，并自动升级状态为 RESIDENT
        - 其它状态（CPU/NONE）：按旧逻辑兜底检查参数实际存在性
        """
        suffixes = GROUPS.get(group)
        if suffixes is None:
            raise ValueError(f"unknown group '{group}', expected one of {tuple(GROUPS.keys())}")

        key = (layer_idx, group)
        state = self._group_state.get(key)
        evt = self._group_events.get(key)  # 统一事件源（唯一真相）

        # 已标记 RESIDENT，但若仍有事件且未完成，则视作未完成（避免提前"已就绪"）
        if state == "RESIDENT":
            if evt is None or evt.query():
                return True
            if not wait_for_event:
                return False
            # 如需要，允许在此同步（很少走到这步；正常应由 compute 流等待）
            evt.synchronize()
            return True

        # INFLIGHT：只有当事件真正完成才算 resident
        if state == "INFLIGHT":
            if evt is not None and evt.query():
                # ✨ 升级为 RESIDENT 并清理 inflight 标记
                with self._group_lock:
                    self._group_state[key] = "RESIDENT"
                    self._gpu_group_inflight.discard(key)
                return True
            if wait_for_event and evt is not None:
                evt.synchronize()
                # ✨ 升级为 RESIDENT 并清理 inflight 标记
                with self._group_lock:
                    self._group_state[key] = "RESIDENT"
                    self._gpu_group_inflight.discard(key)
                return True
            return False

        # 其它状态（CPU/NONE/EVICTING）：按旧逻辑检查参数实际存在性（兜底）
        missing = []
        for suf in suffixes:
            pname = f"layers.{layer_idx}.{suf}"
            p = self.name_to_param.get(pname)
            if (p is None) or (not getattr(p, "is_cuda", False)) or (p.numel() == 0):
                missing.append(pname)

        if missing:
            # 降噪：若事件仍在进行中，不打印"missing"
            silent_when_inflight = (evt is not None) and (not evt.query())
            if (not silent_when_inflight) and (getattr(self, "verbose_mismatch", False) or getattr(self, "verbose", False)):
                preview = ", ".join(missing[:4])
                more = " ..." if len(missing) > 4 else ""
                print(f"[WSM][resident] {layer_idx}.{group} missing tensors: {preview}{more}")
            return False

        return True



    def _record_group_ready_event(self, layer_idx: int, group: str, stream: "torch.cuda.Stream|None" = None) -> None:
        """
        在权重 H2D 流上记录**组级** ready 事件。
        需在把本组所有参数的 H2D 入队后调用（见 ensure_group_on_gpu/prefetch_group_async）。

        注意：此函数必须在对应的 H2D stream context 内调用（with torch.cuda.stream(h2d_stream):）
        这样 event.record() 会在所有 .to() 操作入队后才记录。
        """
        if not torch.cuda.is_available():
            return
        
         # 1) 优先使用调用方传来的实际拷贝流（可能是 compute_stream 的 override）
        h2d = stream
        # 2) 无 override 再按组名分流到固定 H2D stream
        if h2d is None:
            if group == "attn":
                h2d = getattr(self.streams, "weight_h2d_mha", None)
            elif group == "ffn":
                h2d = getattr(self.streams, "weight_h2d_ffn", None)

        if h2d is None:
            return

        key = (layer_idx, group)
        with self._group_lock:
            evt = self._group_events.get(key)
            if evt is None:
                evt = torch.cuda.Event(blocking=False)
                self._group_events[key] = evt
        # ⭐ 关键修复: 显式在指定的 h2d stream 上记录 event
        # 即使在 with torch.cuda.stream() 上下文中，也要明确指定 stream
        # 这确保 event 会在该 stream 的所有前序操作（包括所有 .to() 传输）完成后触发
        evt.record(h2d)

        # 与层级事件一样，做一次 KV 流量仲裁的轻量检查
        try:
            self._check_and_throttle_kv()
        except Exception:
            pass
        
    # --- inside class WeightStreamingManager ---

    def enter_decode_mode(self, *, protect_layers: int = 6, prime_window: int = 8):
        """
        进入 decode 模式：
        - 清理 prefill 遗留的占位事件/半途 inflight
        - 设置解码期保护窗口
        - 预热 prime_window 个 (attn, ffn) group
        """
        self._decode_mode = True

        # 1) 清理所有占位与未记录事件，清空 inflight，避免 prefill 垃圾状态影响 decode
        self._clear_stale_group_events()     # 清空 _placeholder_keys, 移除没有 host_record 的 event
        self._gpu_group_inflight.clear()     # 彻底清空 inflight 标记
        self._rebalance_and_topoff()         # 把 GPU ring 清到合理容量

        # 2) 建立 decode 保护窗口（避免刚切换就被回收）
        self._ensure_decoder_protect_window(protect_layers)

        # 3) 一次性 prime 后续 prime_window 层（attn/ffn 成对）
        self._prime_decoder_window(prime_window)

        
    def _cpu_trim_to_window_if_needed(
        self,
        *,
        force: bool = False,
        max_evict: int | None = None,
        reason: str = "win"
    ) -> int:
        """
        按“目标 CPU 窗口”主动修剪 CPU 缓存中的过期层（轻量、无 I/O）。

        - 仅逐出窗口外的层（<L0 或 >L1）
        - 跳过受保护层（_cpu_protect_set）
        - 跳过仍在 CPU 加载中的层（_inflight_cpu_layers）
        - 默认在 cache 大于 LWM 时生效；传入 force=True 可强制按窗口清理
        - max_evict 可限制本次最多清理数量

        Returns:
            实际逐出的层数
        """
        # 若未启用 SSD 流式，CPU cache 通常很小/非关键，直接返回
        if not getattr(self, "ssd_enabled", False):
            return 0

        # 目标窗口 [L0, L1]
        try:
            L0, L1 = self._target_cpu_window()  # 你的版本里用它/或等价逻辑来推导窗口
        except Exception:
            # 兜底：根据 cpu_win_base/cpu_cache_cap 计算
            base = int(getattr(self, "cpu_win_base", 0))
            cap  = int(getattr(self, "cpu_cache_cap", getattr(self, "cpu_cache_layers", 40)))
            L0 = base
            L1 = min(int(getattr(self, "n_layers", 0)) - 1, base + cap - 1)

        protect = set(getattr(self, "_cpu_protect_set", set()))
        inflight = set(getattr(self, "_inflight_cpu_layers", set()))

        evicted = 0
        with self._cpu_lock:
            # ⭐ 修复：force=True 时不检查 LWM，强制清理窗口外的层
            # 这样可以确保窗口前移时真正清理掉旧层，而不是累积到 emergency cleanup
            lwm = int(getattr(self, "cpu_lwm", max(2, getattr(self, "cpu_cache_cap", 40) - 3)))
            current_size = len(self.cpu_cache)

            if force:
                # 强制模式：必须清理窗口外的层
                need_trim = True
            else:
                # 正常模式：只有超过 LWM 才清理
                need_trim = (current_size > lwm)

            if not need_trim:
                return 0

            # 计算目标窗口集合（支持 wrap-around）
            nL = int(getattr(self, "n_layers", 0))
            base = int(getattr(self, "cpu_win_base", 0))
            cap  = int(getattr(self, "cpu_cache_cap", getattr(self, "cpu_cache_layers", 40)))
            win_set = set(self._ring_range(base % max(1, nL), cap if cap > 0 else 0))

            # ⭐ 修复：收集GPU上resident的层，避免驱逐它们
            gpu_resident_layers = set()
            for (layer, grp), state in self._group_state.items():
                if state in ("RESIDENT", "INFLIGHT"):
                    gpu_resident_layers.add(layer)

            layers_to_evict = []
            for lyr in list(self.cpu_cache.keys()):
                # ⭐ 修复：force=True 时清理所有窗口外的层
                # 正常模式下，如果 cache 大小已经回到 cap 以内，停止清理
                if not force and (current_size - len(layers_to_evict) <= cap):
                    break

                if (max_evict is not None) and (len(layers_to_evict) >= max_evict):
                    break

                # 仅处理窗口外 + 非受保护 + 非在途 + 非GPU resident的层
                if (lyr not in win_set) and (lyr not in protect) and (lyr not in inflight) and (lyr not in gpu_resident_layers):
                    layers_to_evict.append(lyr)

            # 执行逐出
            for lyr in layers_to_evict:
                if (max_evict is not None) and (evicted >= max_evict):
                    break
                with self.cpu_cache_lock:
                    self.cpu_cache.pop(lyr, None)
                self._cpu_cached_layers.discard(lyr)
                evicted += 1

                if getattr(self, "verbose", False):
                    print(f"[WSM] CPU trim({reason}) evict layer {lyr}, after={len(self.cpu_cache)}, win=[{L0},{L1}]")

        return evicted

    def _requeue_gpf(self, key: tuple[int, str], h2d_override=None, pin: bool = False, reason: str = "requeue"):
        """
        把组级预取任务按统一 5 元组格式回投到 _gpf_q（避免旧格式导致的 worker 解包错误）
        """
        try:
            # 允许重复回投（会被 _gpf_seen 限制或被 worker 过滤）
            self._gpf_q.put_nowait((self._epoch, key, bool(pin), reason, h2d_override))
        except Exception:
            pass
        try:
            self._check_and_throttle_kv()
        except Exception:
            pass

    def get_group_ready_event(self, layer_idx: int, group: str) -> "torch.cuda.Event|None":
        """
        非阻塞地获取(layer, group)的权重就绪事件，不会停下计算流。

        返回：
        - torch.cuda.Event: 权重就绪事件（可能尚未完成）
        - None: 权重已驻留且无待处理事件，或无需等待

        调用方可以：
        1. 获取事件后立即启动下一层预取
        2. 在真正需要使用权重前，通过 stream.wait_event() 等待
        """
        import torch
        if not torch.cuda.is_available():
            return None

        key = (int(layer_idx), 'attn' if group == 'attn' else 'ffn')
        st = self._get_state(key)

        # Fast path: 已驻留且无待处理事件
        evt = self._group_events.get(key)
        if st == "RESIDENT" and (evt is None or evt.query()):
            return None

        # 触发预取（如果尚未在途/驻留）
        if st not in ("INFLIGHT", "RESIDENT"):
            self.prefetch_group_async(layer_idx, group, pin=False, reason="get_event")
            # 重新获取事件（预取可能创建了新事件）
            evt = self._group_events.get(key)

        return evt

    def _is_group_physically_resident(self, layer_idx: int, group: str) -> bool:
        suffixes = GROUPS.get(group)
        if suffixes is None:
            return False
        for suf in suffixes:
            pname = f"layers.{layer_idx}.{suf}"
            p = self.name_to_param.get(pname)
            if (p is None) or (not getattr(p, "is_cuda", False)) or (p.numel() == 0):
                return False
        return True

    def wait_group_ready(self, layer_idx: int, group: str, compute_stream=None):
        """
        严格的“先预取→后等待”：
        1) 若未调度，触发预取（固定 H2D 流，避免随机化）
        2) 先等 host 侧确认“record() 已完成”
        3) 确保占位已被真实事件替换
        4) 在 compute 流上 wait_event（非阻塞 CPU）
        """
        import time, torch
        key = (int(layer_idx), 'attn' if group == 'attn' else 'ffn')

        # 1) 若无事件，触发一次预取（选取与组匹配的 H2D 流）
        if key not in self._group_events:
            if group == "attn":
                h2d0 = getattr(self.streams, "weight_h2d_mha", None)
            else:
                h2d0 = getattr(self.streams, "weight_h2d_ffn", None)
            self.prefetch_group_async(layer_idx, group, h2d_override=h2d0, reason="wait_on_demand")

        # 2) host 侧 record() 完成确认（避免 GPU 等"假事件"）
        timeout_s = float(getattr(self, "_evt_wait_timeout_s", 2.0))
        host_evt  = self._group_recorded_host.get(key, None)
        if host_evt is not None:
            if not host_evt.wait(timeout=timeout_s):
                # 再触发一次预取并重试（处理抢占失败或被驱逐）
                if self.verbose:
                    print(f"[WSM WAIT] Timeout waiting for {key}, retrying prefetch...")
                self.prefetch_group_async(layer_idx, group, reason="wait_retry")
                if not host_evt.wait(timeout=timeout_s):
                    if self.verbose:
                        print(f"[WSM WAIT] Second timeout for {key}, checking CPU readiness...")
                        cpu_ready = self._cpu_group_ready(layer_idx, group)
                        print(f"[WSM WAIT] CPU ready for {key}: {cpu_ready}")
                    raise RuntimeError(f"WSM: host record not observed for {key}")

        # 3) 等占位被真实事件替换（轻量自旋，带上限）
        spin_max = int(getattr(self, "_evt_spin_max", 10000))
        spins = 0
        while key in getattr(self, "_placeholder_keys", set()):
            if spins >= spin_max:
                raise RuntimeError(f"WSM: placeholder not replaced for {key}")
            time.sleep(1e-4); spins += 1

        # 4) 在 compute 流上建立事件依赖（非阻塞 CPU）
        evt = self._group_events.get(key, None)
        if evt is None:
            # 极端兜底：重触发 + 等 host_evt，再拿事件
            self.prefetch_group_async(layer_idx, group, reason="wait_missing_evt")
            host_evt = self._group_recorded_host.get(key, None)
            if host_evt is not None:
                host_evt.wait(timeout=timeout_s)
            evt = self._group_events.get(key, None)
            if evt is None:
                raise RuntimeError(f"WSM: missing CUDA event after prefetch for {key}")

        if compute_stream is None:
            compute_stream = torch.cuda.current_stream()
        compute_stream.wait_event(evt)  # 非阻塞：仅 GPU 侧依赖



    def num_gpu_groups(self) -> int:
            with self._group_lock:
                return len(self._gpu_group_ring) + len(self._gpu_group_inflight)
    def _gpu_budget_allows_new_group(self, key: tuple[int, str], reason: str | None) -> bool:
        """
        轻量预算门：
        - 机会型预取（window / pair / ring 等）在 GPU groups 已满时直接跳过；
        - 硬需求预取（当前层马上要算）永远放行。
        """
        used = self.num_gpu_groups()
        if used < self.gpu_max_groups:
            return True  # 预算没打满

        r = (reason or "").lower()

        # === 关键：这些 reason 一律放行 ===
        # 1) 没填 reason（旧调用路径） -> 视为必须
        if reason is None or r == "":
            return True
        # 2) wait_group_ready / wait_cpu_retry_* -> 当前层要算
        if r.startswith("wait_"):
            return True
        # 3) get_event / get_group_event 等 -> 也是 wait path
        if r.startswith("get_event"):
            return True
        if r.startswith("window_"):
            return True
        if r.startswith("pair"):
            return True
        if "prime-decoder" in r:
            return True

        # 其它 reason 视为“机会型”预取，预算打满就直接略过
        if self.debug_prefetch:
            print(f"[WSM] prefetch_group_async: skip {key} (GPU groups {used}/{self.gpu_max_groups}, reason={r})")
        return False


    # --- inside class WeightStreamingManager ---
    def prime_decode_window(self, first_layer: int = 0, window: int = 6):
        """
        在解码开始时预热 CPU 缓存窗口：[first_layer, first_layer+window)（考虑回绕）。
        """
        if not getattr(self, "ssd_enabled", False):
            return
        n = int(getattr(self, "n_layers", 0))
        if n <= 0:
            return
        first_layer %= n
        window = max(1, int(window))
        # 重置 CPU 窗口基准与游标，使随后 _ensure_cpu_window() 正确"从 0"补齐
        self.cpu_win_base = first_layer
        self._cpu_pf_cursor = first_layer
        # ⭐ 修改：不调用同步 _ensure_cpu_window，依赖异步 rebalance_and_topoff 填充
        # _ensure_cpu_window() 会因为layers已在inflight而跳过，导致CPU cache无法达到40层
        # self._ensure_cpu_window()


    def _proactive_cleanup_old_groups(self, current_layer: int):
        """
        主动清理不在窗口内的旧组，防止 GPU group ring 累积过多旧组。
        在每次标记新组为 IN_USE 之前调用。
        """
        # ⭐ FIX: 窗口应该包含"向前预取"和"向后保留已用"两部分
        # 向前: i+1 到 i+gpu_ahead (预取窗口)
        # 向后: i-1 到 i-2 (避免刚用完就驱逐)
        ahead = self.gpu_ahead_layers  # 4
        behind = max(1, int(getattr(self, "gpu_behind_layers", 3)))  # 环窗内保留刚用过的若干层
        W_total = 1 + ahead + behind  # 当前层 + 前面4层 + 后面3层 = 8

        if W_total >= self.n_layers:
            W_total = max(1, self.n_layers - 1)

        debug = getattr(self, "debug_prefetch", False)
        evicted_count = 0
        max_evict = 10  # 每次最多驱逐 10 个旧组

        # 找出所有不在窗口内、引用计数为 0 的旧组
        candidates = []
        with self._group_lock:
            for lyr, grp in list(self._gpu_group_ring):
                key = (int(lyr), str(grp))
                # 跳过当前层的 FFN（必须保留）
                if grp == "ffn" and int(lyr) == current_layer:
                    continue

                # ⭐ FIX: 新的窗口逻辑 - 向前和向后都保留
                # 保留区间: [current-behind, current+ahead+1)
                # 例如 current=75, behind=3, ahead=4: 保留 [72, 80)
                lyr_int = int(lyr)
                cur_int = int(current_layer)

                # 计算相对位置 (处理环形)
                rel_pos = (lyr_int - cur_int) % self.n_layers
                if rel_pos > self.n_layers // 2:  # 负数情况
                    rel_pos -= self.n_layers

                # 在 [-behind, ahead+1) 区间内则保留
                if -behind <= rel_pos <= ahead:
                    continue

                # 检查引用计数
                if self._gpu_group_in_use.get(key, 0) > 0:
                    continue
                # 强制 unpin 旧组（如果被 pin）
                if self._is_pinned(lyr, grp):
                    self._pinned_groups.pop(key, None)
                    if debug:
                        print(f"[WSM DEBUG][cleanup] force-unpin old group {key}")

                candidates.append(key)
                if len(candidates) >= max_evict:
                    break

        # 直接驱逐候选组（不调用 _evict_one_group_from_gpu 避免重复扫描）
        for key in candidates:
            lyr, grp = key
            suffixes = ("attention.wq.weight","attention.wk.weight","attention.wv.weight","attention.wo.weight") if grp=="attn" \
                    else ("feed_forward.w1.weight","feed_forward.w3.weight","feed_forward.w2.weight")

            # 驱逐参数到 CPU
            for suf in suffixes:
                pname = f"layers.{lyr}.{suf}"
                p = self.name_to_param.get(pname)
                if p is None:
                    continue
                if p.is_cuda and p.numel() > 0:
                    try:
                        self._evict_param_to_cpu(p, pname=pname)
                    except Exception:
                        pass

            # 更新状态
            with self._group_lock:
                self._set_state(key, "CPU")
                try:
                    if key in self._gpu_group_ring:
                        self._gpu_group_ring.remove(key)
                except Exception:
                    pass

            evicted_count += 1
            if debug:
                print(f"[WSM DEBUG][cleanup] evicted old group {key}")

        if evicted_count > 0:
            torch.cuda.empty_cache()
            if debug:
                print(f"[WSM DEBUG][cleanup] proactively evicted {evicted_count} old groups before L{current_layer}")

    def _touch_group(self, layer_idx: int, group: str):
        # Legacy no-op: ring eviction no longer depends on timestamps.
        return

    def _should_retain_group(self, layer_idx: int, group: str) -> bool:
        key = (int(layer_idx), str(group))
        if self._gpu_group_in_use.get(key, 0) > 0:
            return True
        if self._is_pinned(layer_idx, group):
            return True
        return False

    # ---------- Balanced group scheduler: pin/unpin methods ----------
    def _is_pinned(self, L: int, kind: str) -> bool:
        """检查组是否被 pin（受保护不被淘汰）"""
        k = self._key(L, kind)
        return self._pinned_groups.get(k, 0) > 0

    def pin_group(self, L: int, kind: str, reason: str = "pair") -> None:
        """
        Pin 一个组，防止其被 LRU 淘汰（引用计数）。

        Args:
            L: 层索引
            kind: 组类型 ('attn' 或 'ffn')
            reason: pin 的原因（用于调试日志）
        """
        with self._group_lock:
            k = self._key(L, kind)
            self._pinned_groups[k] = self._pinned_groups.get(k, 0) + 1
            if self.verbose:
                print(f"[WSM][pin] {k} ({reason}) refcount={self._pinned_groups[k]}")

    def unpin_group(self, L: int, kind: str) -> None:
        """
        Unpin 一个组（减少引用计数）。

        Args:
            L: 层索引
            kind: 组类型 ('attn' 或 'ffn')
        """
        with self._group_lock:
            k = self._key(L, kind)
            c = self._pinned_groups.get(k, 0)
            if c <= 1:
                self._pinned_groups.pop(k, None)
            else:
                self._pinned_groups[k] = c - 1
            if self.verbose:
                print(f"[WSM][unpin] {k} refcount={self._pinned_groups.get(k, 0)}")

    def _pick_victim(self, exclude: set[tuple[int, str]]) -> Optional[tuple[int, str]]:
        """
        选择一个 victim 组进行淘汰（LRU 策略 + 窗口保护）。

        Args:
            exclude: 需要排除的组集合（正在使用或其他原因）

        Returns:
            (layer_idx, group_type) 或 None（如果没有可淘汰的）
        """
        # 保护近用窗口：当前层与下一层的关键组优先跳过
        cur = self._current_layer
        protected = set()
        if cur is not None:
            protected.add((cur, 'ffn'))
            if cur + 1 < self.n_layers:
                protected.add((cur + 1, 'attn'))

        # 合并所有需要保护的组
        protected_keys = exclude | protected
        # 添加所有 pinned 和 in_use 的组
        protected_keys |= set(k for k, cnt in self._pinned_groups.items() if cnt > 0)
        protected_keys |= set(k for k, cnt in self._gpu_group_in_use.items() if cnt > 0)

        # 遍历 GPU group ring（从最老的开始）
        for k in list(self._gpu_group_ring):
            if k in protected_keys:
                continue
            # 新增：最近触达过的跳过
            if self._should_retain_group(k[0], k[1]):
                continue
            return k

        return None


    def notify_group_compute_done(self, layer_idx: int, group: str, evt: "torch.cuda.Event|None"):
        """
        纯滑动窗口：唯一的窗口推进入口
        - 每完成一个 group（attn/ffn），整体推进窗口：
        1) 逐出窗口外组（GPU→CPU/SSD）
        2) 预取窗口内缺失组（CPU/SSD→GPU）
        3) 在整层完成时推进 CPU 窗口

        ⚠️ 不再在这里等待 CUDA 事件完成，避免阻塞前向线程。
        计算正确性由:
        - 引用计数 in_use（pin_group/unmark_group）
        - 计算流上的 stream.wait_event()
        来保证；窗口只根据状态和窗口位置做"决策 + IO 入队"。
        """
        # key = (layer_idx, group)
        # # 记录 compute 完成事件，用于优化逐出/抢占策略
        # if evt is not None:
        #     with self._group_lock:
        #         self._last_compute_evt[key] = evt

        # Step 1: 更新执行锚点（仅在 ffn 完成时推进到下一层）
        with self._anchor_lock:
            if group == "ffn":
                # ffn 完成后，认为这一层已经"用完"，锚到下一层
                self._last_executed_layer = (layer_idx + 1) % self.n_layers

        # Step 2: 不再等待 evt.query()，完全交给 CUDA 流自己保证顺序

        # Step 3: 推进窗口（核心，内部只做决策和异步 IO 入队）
        try:
            self._slide_window_forward(layer_idx, group)
        except Exception as e:
            if self.verbose:
                print(f"[WSM] _slide_window_forward failed for L{layer_idx}.{group}: {e}")
                import traceback
                traceback.print_exc()




    # ========== 修改 2: 异步滑动窗口（纯决策函数，所有 I/O 后台执行） ==========
    def _slide_window_forward(self, current_layer: int, current_group: str):
        """
        ✅ 异步窗口滑动：只做决策，所有 I/O 提交到后台线程
        主线程耗时 <1ms，不阻塞计算

        核心改进：
          1) GPU 窗口逐出：提交到 _evict_q（异步后台处理）
          2) GPU 窗口预取：乐观入队到 _gpf_q（即使令牌不足）
          3) CPU 窗口推进：提交到 _bg_executor（异步执行）
        """
        debug = getattr(self, "debug_prefetch", False)
        if debug:
            print(f"\n[WSM WINDOW ASYNC] ========== Slide Forward: L{current_layer}.{current_group} ==========")

        # ---------- Part 1: GPU 窗口决策（快速路径） ----------
        gpu_window = self._build_gpu_window(current_layer, current_group)
        if debug:
            try:
                msg = ", ".join(f"L{L}.{g}" for (L, g) in sorted(gpu_window))
            except Exception:
                msg = str(gpu_window)
            print(f"[WSM WINDOW ASYNC] GPU window: [{msg}]")

        # 1.1 ✅ 异步逐出窗口外组（提交到后台队列，不阻塞主线程）
        evict_submitted = 0
        with self._group_lock:
            ring_snapshot = list(self._gpu_group_ring)

        for (L, grp) in ring_snapshot:
            key = (L, grp)
            if key in gpu_window:
                continue

            # 硬保护（快速检查，不等待事件）
            if self._gpu_group_in_use.get(key, 0) > 0:
                if debug:
                    print(f"[WSM WINDOW ASYNC] Skip evict {key} (in_use={self._gpu_group_in_use[key]})")
                continue
            if key in self._gpu_group_inflight:
                if debug:
                    print(f"[WSM WINDOW ASYNC] Skip evict {key} (inflight)")
                continue

            # ✅ 去重：避免重复提交
            if key in self._evict_seen:
                continue

            # ✅ 异步提交逐出（不阻塞主线程，事件等待在后台线程）
            evt = self._group_events.get(key)
            try:
                self._evict_q.put_nowait((L, grp, evt))
                self._evict_seen.add(key)
                evict_submitted += 1
                if debug:
                    print(f"[WSM WINDOW ASYNC] Submitted evict {key}")
            except queue.Full:
                if debug:
                    print(f"[WSM WINDOW ASYNC] Evict queue full, skip {key}")

        if debug and evict_submitted > 0:
            print(f"[WSM WINDOW ASYNC] Submitted {evict_submitted} evictions to background")

        # 1.2 ✅ 异步预取窗口内缺失组（批量提交，不等待配额）
        prefetch_submitted = 0
        for (L, grp) in gpu_window:
            key = (L, grp)
            if self._group_is_resident(L, grp):
                continue
            if key in self._gpu_group_inflight:
                continue

            # ✅ 乐观提交预取（即使令牌不足也会占位 + 入队）
            try:
                pin = (L == current_layer and grp == "ffn" and current_group == "attn")
                success = self.prefetch_group_async(L, grp, pin=pin, reason=f"window_L{L}_{grp}")
                # 只要成功占位（包括入队）就算提交成功
                if success or key in self._placeholder_keys:
                    prefetch_submitted += 1
                    if debug:
                        print(f"[WSM WINDOW ASYNC] Submitted prefetch {key}")
            except Exception as e:
                if debug:
                    print(f"[WSM WINDOW ASYNC] Prefetch {key} failed: {e}")

        if debug and prefetch_submitted > 0:
            print(f"[WSM WINDOW ASYNC] Submitted {prefetch_submitted} prefetches")

        # ---------- Part 2: ✅ 异步推进 DRAM 窗口（后台线程） ----------
        if current_group == "ffn" and getattr(self, "ssd_enabled", False):
            if debug:
                print(f"[WSM WINDOW ASYNC] Submitting CPU window advance (layer {current_layer} completed)")
            # ✅ 提交到后台线程池异步执行（不阻塞主线程）
            try:
                self._bg_executor.submit(self._async_advance_cpu_window, current_layer)
            except Exception as e:
                if debug:
                    print(f"[WSM WINDOW ASYNC] CPU window submit failed: {e}")

        if debug:
            print(f"[WSM WINDOW ASYNC] ========== Slide Forward Done (async) ==========\n")
        # ✅ 主线程立即返回（<1ms）



    # ========== 修改 3: 新增 _build_gpu_window 及映射辅助函数 ==========
    def _build_gpu_window(self, current_layer: int, current_group: str) -> set[tuple[int, str]]:
        """
        计算 GPU 窗口内的组集合：
        - 执行 L.attn 时：窗口 = [L.attn, L.ffn] + 后续 gpu_ahead_groups 个组
        - 执行 L.ffn 时：窗口 = [L.ffn] + 后续 gpu_ahead_groups 个组
        """
        window: set[tuple[int, str]] = set()
        # 当前组必在
        window.add((current_layer, current_group))
        # attn 执行时成对保护本层 ffn
        if current_group == "attn":
            window.add((current_layer, "ffn"))
        # 向前预取
        ahead_groups = int(getattr(self, "gpu_ahead_groups", 4))
        ahead_groups = max(0, min(ahead_groups, max(0, self.gpu_max_groups - 2)))
        cur_pos = self._group_position(current_layer, current_group)
        start_pos = cur_pos + (2 if current_group == "attn" else 1)
        for i in range(ahead_groups):
            pos = start_pos + i
            L, grp = self._position_to_group(pos)
            window.add((L, grp))
        return window


    def _group_position(self, layer: int, group: str) -> int:
        """(layer, group) → 全局组序列位置：L.attn=even, L.ffn=odd"""
        return int(layer) * 2 + (0 if group == "attn" else 1)

    def _position_to_group(self, pos: int) -> tuple[int, str]:
        """反向映射：全局位置 → (layer, group)，支持环形。"""
        n_groups = self.n_layers * 2
        pos = int(pos) % n_groups  # 环形支持
        layer = pos // 2
        group = "attn" if pos % 2 == 0 else "ffn"
        return layer, group

    def _resident(self, L: int, kind: str) -> bool:
        """辅助方法：检查组是否已驻留（用于 rebalance_and_topoff）"""
        return self._group_is_resident(L, kind)

    def _inflight(self, L: int, kind: str) -> bool:
        """辅助方法：检查组是否在 inflight 中（用于 rebalance_and_topoff）"""
        return self._key(L, kind) in self._gpu_group_inflight

    def _shrink_gpu_groups_now(self, exclude: set[tuple[int, str]] = frozenset()):
        # 分两轮：第一轮尊重"最近触达"；第二轮忽略之
        for pass_idx in (0, 1):
            while True:
                with self._group_lock:
                    if len(self._gpu_group_ring) <= self.gpu_max_groups:
                        return
                ok = self._evict_one_group_from_gpu(
                    exclude=exclude,
                    ignore_retain=(pass_idx == 1)
                )
                if not ok:
                    # 第二轮仍失败：尝试强制解 pin + 淘汰"非 IN_USE 的 pinned 组"
                    if pass_idx == 1 and self.allow_force_unpin:
                        ok = self._evict_one_group_from_gpu(
                            exclude=exclude, ignore_retain=True, allow_unpin=True
                        )
                        if ok:
                            continue
                    if self.verbose:
                        print(f"[WSM] shrink pass{pass_idx}: no candidate; "
                            f"overflow={len(self._gpu_group_ring)} > cap={self.gpu_max_groups}")
                    break  # 进入下一轮（若还有）


    # ========== 修改 4: 简化 _evict_one_group_from_gpu ==========
    def _evict_one_group_from_gpu(self, exclude=(), ignore_retain=False, allow_unpin: bool = True) -> bool:
        """
        紧急逐出：仅在 GPU 容量超限时调用（兜底保护）。
        在纯滑动窗口模式下，逐出应由 _slide_window_forward 统一管理。
        """
        debug = getattr(self, "debug_prefetch", False)
        exclude_set = set(exclude) if exclude else set()

        if debug:
            print(f"[WSM EMERGENCY] _evict_one_group_from_gpu called (emergency eviction)")

        with self._group_lock:
            ring_list = list(getattr(self, "_gpu_group_ring", []))
        for idx, (L, grp) in enumerate(ring_list):
            key = (L, grp)

            # 硬保护
            if key in exclude_set:
                continue

            # ⭐ 关键修复：先调用 _group_is_resident 来触发 INFLIGHT->RESIDENT 自动升级
            # 这会清理已完成事件的 inflight 标记，避免所有组都被锁定无法驱逐
            try:
                self._group_is_resident(L, grp, wait_for_event=False)
            except Exception:
                pass

            # 现在再检查 inflight（已完成的会被上面的调用清理掉）
            if key in self._gpu_group_inflight:
                continue
            if self._gpu_group_in_use.get(key, 0) > 0:
                continue
            evt = self._group_events.get(key)
            if evt is not None:
                try:
                    if not evt.query():
                        continue
                except Exception:
                    pass
            if getattr(self, "_is_pinned", lambda *_: False)(L, grp) and not allow_unpin:
                continue

            if debug:
                print(f"[WSM EMERGENCY] Evicting {key} (ring index={idx})")
            try:
                self._evict_group_immediately(L, grp, skip_prefetch=True)
                return True
            except Exception as e:
                if debug:
                    print(f"[WSM EMERGENCY] Evict {key} failed: {e}")
                continue

        if debug:
            print(f"[WSM EMERGENCY] No evictable group found (ring size={len(getattr(self, '_gpu_group_ring', []))})")
        return False


    def _ensure_gpu_room(self, need_bytes, exclude=()):
        guard = self._gpu_free_guard_mb * 1024 * 1024
        debug = getattr(self, "debug_prefetch", False) or self.verbose
        evict_attempts = 0
        max_evict_attempts = 20  # 防止无限循环
        while evict_attempts < max_evict_attempts:
            free, _ = torch.cuda.mem_get_info(self.device_index)
            if free >= need_bytes + guard and len(self._gpu_group_ring) < self.gpu_max_groups:
                return

            # ⭐ CRITICAL FIX: 多轮驱逐策略，逐步放宽限制
            ok = False
            if evict_attempts < 10:
                # 第一阶段：尊重 retain 时间戳，允许 force-unpin
                ok = self._evict_one_group_from_gpu(exclude=exclude, ignore_retain=False, allow_unpin=True)
            else:
                # 第二阶段：忽略 retain，强制 unpin
                ok = self._evict_one_group_from_gpu(exclude=exclude, ignore_retain=True, allow_unpin=True)

            if not ok:
                # 驱逐失败，打印诊断信息
                if debug:
                    with self._group_lock:
                        in_use_list = [(k, v) for k, v in self._gpu_group_in_use.items() if v > 0]
                        pinned_list = [(k, v) for k, v in self._pinned_groups.items() if v > 0]
                        print(f"[WSM ERROR] _ensure_gpu_room: 驱逐失败 after {evict_attempts} attempts")
                        print(f"  GPU group ring ({len(self._gpu_group_ring)} groups): {self._gpu_group_ring[:10]}")
                        print(f"  In-use groups ({len(in_use_list)}): {in_use_list}")
                        print(f"  Pinned groups ({len(pinned_list)}): {pinned_list}")
                        print(f"  Free GPU: {free/(1<<20):.1f} MiB, need: {(need_bytes+guard)/(1<<20):.1f} MiB")
                        print(f"  Current layer: {getattr(self, '_last_executed_layer', 0)}")
                break
            evict_attempts += 1
        # 再次检查，如果还不够，让上层处理 OOM

    def _install_param_tensor(self, pname: str, dst_gpu_tensor: torch.Tensor, version_hint: int | None = None):
        param = self.name_to_param.get(pname)
        if param is None:
            return

        need_replace = (param.dtype != dst_gpu_tensor.dtype) or (param.shape != dst_gpu_tensor.shape)
        key = self._name_to_group_key(pname)

        with self._group_lock:
            if key is not None and self._get_state(key) == "EVICTING":
                return
            if version_hint is not None and int(self._param_version.get(pname, 0)) != int(version_hint):
                return

            if need_replace:
                if pname in self.param_owner:
                    mod, attr = self.param_owner[pname]
                    new_p = nn.Parameter(dst_gpu_tensor, requires_grad=False)
                    setattr(mod, attr, new_p)
                    self.name_to_param[pname] = new_p
                else:
                    param.data = dst_gpu_tensor
                    self.name_to_param[pname] = param
            else:
                # 需要真正把张量迁移到 GPU，而不是仅做数据拷贝
                param.data = dst_gpu_tensor

            self._param_version[pname] = int(self._param_version.get(pname, 0)) + 1

    def _install_group_on_gpu(self, layer_idx: int, group: str, *, h2d_override=None):
        """
        将 (layer, group) 的 CPU pinned 张量拷到 GPU（统一的 H2D 安装路径）：
        - 选择 H2D 流（attn/ffn 分流，或使用 override）
        - 逐张量 non_blocking H2D
        - 记录 inflight 事件到 self._group_events
        - 仅设为 INFLIGHT；RESIDENT 延迟到事件完成时再升级

        这是多流 + 事件记录的核心入口，避免在各处重复 H2D 逻辑。
        """
        key = (int(layer_idx), group)
        suffixes = GROUPS.get(group)
        if suffixes is None:
            raise ValueError(f"Unknown group '{group}', expected one of {tuple(GROUPS.keys())}")

        # 1) 选择 H2D 流
        stream = h2d_override or self._pick_h2d_stream(group)
        if stream is None and torch.cuda.is_available():
            stream = torch.cuda.current_stream()

        # 2) 获取 CPU 张量（优先从 cpu_cache，兜底从模型）
        with self.cpu_cache_lock:
            layer_cache = dict(self.cpu_cache.get(layer_idx, {}))

        # 3) 逐张量 non_blocking H2D
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            for suf in suffixes:
                pname = f"layers.{layer_idx}.{suf}"
                cpu_t = layer_cache.get(pname)
                if cpu_t is None:
                    # 兜底：从模型中获取 CPU 参数
                    cpu_t = self._get_param_from_model(layer_idx, suf)
                if cpu_t is None or cpu_t.numel() == 0:
                    continue

                # 执行 H2D 传输
                gpu_t = cpu_t.to(device=self.device, non_blocking=True)
                # 安装到模块参数
                self._install_param_tensor(pname, gpu_t)

            # 4) 记录 inflight 事件
            with self._group_lock:
                evt = self._group_events.get(key)
                if evt is None:
                    evt = torch.cuda.Event(blocking=False, interprocess=False)
                    self._group_events[key] = evt

            # 确保在实际的 H2D stream 上 record
            if stream is not None:
                evt.record(stream)
            else:
                # 兜底：当前流
                evt.record(torch.cuda.current_stream())
                
        # 5) 统一事件表 + INFLIGHT 状态
        # with self._group_lock:
        #     self._group_events[key] = evt
        #     self._set_state(key, "INFLIGHT")
        #     self._gpu_group_inflight.add(key)
        with self._group_lock:
            self._set_state(key, "INFLIGHT")
            self._gpu_group_inflight.add(key)
            # 注意：不在这里设 RESIDENT（延迟到事件完成）


# ---------- H2D stream 选择：按组分流 ----------
    def _select_h2d_stream_for(self, name: Optional[str] = None, module_name: Optional[str] = None):
        """
        按参数/模块路径将 H2D 路由到对应流：
          - *.attention.* -> weight_h2d_mha
          - *.feed_forward.* -> weight_h2d_ffn
        """
        s = None
        try:
            n = (name or "").lower()
            m = (module_name or "").lower()
            if (".attention." in n) or ("attent" in m):
                s = getattr(self.streams, "weight_h2d_mha", None)
            elif (".feed_forward." in n) or ("feed_forward" in m) or ("ffn" in m):
                s = getattr(self.streams, "weight_h2d_ffn", None)
        except Exception:
            s = None
        # fallback：尽量选一条可用的新流
        return s or getattr(self.streams, "weight_h2d_mha", None) or getattr(self.streams, "weight_h2d_ffn", None)

    def _move_to_gpu(self, pname: str, src_cpu_tensor: torch.Tensor, exclude: set[tuple[int,str]] | None = None):
        need_bytes = src_cpu_tensor.numel() * src_cpu_tensor.element_size()
        self._ensure_gpu_headroom(need_bytes, exclude=exclude)

        h2d_stream = self._select_h2d_stream_for(name=pname) or self._copy_stream
        ver0 = int(self._param_version.get(pname, 0))

        # ✅ P0-3: 使用带超时和重试的 H2D 传输
        try:
            if h2d_stream is not None:
                dst = self._h2d_transfer_with_retry(src_cpu_tensor, pname, h2d_stream)
            else:
                dst = src_cpu_tensor.to(self.device, non_blocking=False)
        except torch.cuda.OutOfMemoryError:
            if self._evict_one_group_from_gpu(exclude=exclude or set()):
                torch.cuda.empty_cache()
                if h2d_stream is not None:
                    dst = self._h2d_transfer_with_retry(src_cpu_tensor, pname, h2d_stream)
                else:
                    dst = src_cpu_tensor.to(self.device, non_blocking=False)
            else:
                raise

        self._install_param_tensor(pname, dst, version_hint=ver0)
        return dst


    # def ensure_group_on_gpu(self, layer_idx: int, group: str):
    #     raise NotImplementedError("Blocking ensure is removed; use prefetch_group_async + wait_group_ready.")
        
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


    def _read_layer_from_ssd_threadsafe(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        线程安全版本的 SSD 读取：使用线程私有 staging buffer。
        与 _read_layer_from_ssd 功能相同，但每个线程使用独立的 staging buffer，
        避免并发冲突。
        """
        if not self.ssd_enabled:
            raise RuntimeError("SSD backend not enabled")
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise IndexError(f"layer {layer_idx} out of range 0..{self.n_layers-1}")
        if layer_idx not in self.layers_params:
            raise KeyError(f"Layer {layer_idx} not in manifest")

        from .weights_io_ssd_dram import DTYPE_MAP, alloc_pinned_aligned

        # 初始化线程私有 staging buffer
        if not hasattr(self._cpu_tls, "staging") or self._cpu_tls.staging is None:
            bs = self.ssd_manifest["block_size"]
            staging_bytes = self.staging_mb * 1024 * 1024
            staging_bytes = (staging_bytes // bs) * bs
            self._cpu_tls.staging = alloc_pinned_aligned(staging_bytes, bs)

        layer_weights: Dict[str, torch.Tensor] = {}
        params = self.layers_params[layer_idx]

        for pinfo in params:
            if pinfo.get("policy") != "stream":
                continue

            stride = int(pinfo["stride"])
            offset = int(pinfo["offset"])
            nbytes = int(pinfo["nbytes"])

            # 确保线程私有 staging buffer 足够大
            if stride > len(self._cpu_tls.staging):
                bs = int(self.ssd_manifest["block_size"])
                new_sz = ((stride + bs - 1) // bs) * bs
                self._cpu_tls.staging = alloc_pinned_aligned(new_sz, bs)

            # 直接读到线程私有 staging
            self.ssd_dio.pread_into_tensor(self._cpu_tls.staging, stride, offset)

            # 组装 pinned tensor
            t = torch.empty(
                tuple(pinfo["shape"]),
                dtype=DTYPE_MAP[pinfo["dtype"]],
                pin_memory=True
            )
            # copy exact bytes
            t.view(-1).view(torch.uint8)[:nbytes].copy_(self._cpu_tls.staging[:nbytes])

            layer_weights[pinfo["name"]] = t

        if not layer_weights:
            raise RuntimeError(f"no stream params loaded for layer {layer_idx}")
        return layer_weights

    def _cpu_next_to_fill(self, anchor: int | None = None) -> int | None:
        """
        返回“从 anchor 起环绕意义下，当前 CPU/在飞集合里的最大层 + 1（取模）”。
        如果一个都没有，返回 anchor 本身。
        """
        nL = int(getattr(self, "n_layers", 0))
        if nL <= 0:
            return None

        if anchor is None:
            anchor = int(getattr(self, "_last_executed_layer", 0))

        with self.cpu_cache_lock:
            present = set(self.cpu_cache.keys())
        inflight = set(getattr(self, "_inflight_cpu_layers", set()))
        have = present | inflight
        if not have:
            return int(anchor % nL)

        # 找到离 anchor 最远的那个层（环形距离最大）
        best = None
        best_dist = -1
        for L in have:
            d = (int(L) - int(anchor)) % nL
            if d >= best_dist:
                best = int(L)
                best_dist = d

        return int((best + 1) % nL)


    def _cpu_try_enqueue(self, layer_idx: int, reason: str = "cpu-pump") -> bool:
        """
        尝试把 layer_idx 入队到 SSD→CPU 预取，如果已经在 cache 或 inflight 则跳过。
        """
        layer_idx = int(layer_idx)
        with self.cpu_cache_lock:
            if layer_idx in self.cpu_cache:
                return False
        if layer_idx in getattr(self, "_inflight_cpu_layers", set()):
            return False

        # 标记 inflight 并入队
        with self._cpu_lock:
            self._inflight_cpu_layers.add(layer_idx)
            try:
                self._cpu_pf_q.put_nowait((self._epoch, layer_idx))
                if getattr(self, "verbose", False):
                    print(f"[WSM][cpu-pump] enqueue L{layer_idx} ({reason})")
                return True
            except queue.Full:
                # 放不进去就撤销 inflight 标记
                try:
                    self._inflight_cpu_layers.discard(layer_idx)
                except Exception:
                    pass
                if getattr(self, "verbose", False):
                    print(f"[WSM][cpu-pump] queue full, drop L{layer_idx}")
                return False


    def _cpu_pump_fill(self, reason: str = "cpu-pump", max_new: int = 2) -> int:
        """
        把 CPU cache 填满到 target_cpu_layers（或 cpu_hwm），每次最多新拉 max_new 个。
        策略：以 _last_executed_layer 为锚，计算“当前最大层 + 1（取模）”，持续推进。
        """
        if not getattr(self, "ssd_enabled", False):
            return 0

        target = int(getattr(self, "target_cpu_layers", getattr(self, "cpu_hwm", 0)))
        if target <= 0:
            return 0

        made = 0
        # 估算当前填充度：cache + inflight
        with self.cpu_cache_lock:
            cur_cached = len(self.cpu_cache)
        cur_inflight = len(getattr(self, "_inflight_cpu_layers", set()))
        while (cur_cached + cur_inflight) < target and made < int(max_new):
            cand = self._cpu_next_to_fill(getattr(self, "_last_executed_layer", 0))
            if cand is None:
                break
            if self._cpu_try_enqueue(cand, reason=reason):
                made += 1
                cur_inflight += 1
            else:
                # 候选已经在 cache 或 inflight，推进 anchor 再试一次
                try:
                    setattr(self, "_last_executed_layer", int((getattr(self, "_last_executed_layer", 0) + 1) % self.n_layers))
                except Exception:
                    break
        return made



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
            
            # === 窗口检查：若该层已不在当前窗口，直接跳过 ===
            with self._cpu_lock:
                in_window = self._layer_in_cpu_window(layer_idx)
                in_protect = (int(layer_idx) in getattr(self, "_cpu_protect_set", set()))
                if not (in_window or in_protect):
                    if self.verbose:
                        print(f"[WSM DEBUG] _cpu_prefetch_worker: L{layer_idx} not in window/protect (cpu_win_base={self.cpu_win_base}, cpu_cache_cap={self.cpu_cache_cap}, cpu_ring_mode={self.cpu_ring_mode})")
                    self._inflight_cpu_layers.discard(layer_idx)
                    self._cpu_pf_q.task_done()
                    continue

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
                in_win = self._layer_in_cpu_window(layer_idx)
                in_protect = (int(layer_idx) in getattr(self, "_cpu_protect_set", set()))
                if not (in_win or in_protect):
                    # 窗口已前移，丢弃过期结果
                    if self.verbose:
                        print(f"[WSM DEBUG] _cpu_prefetch_worker (after read): L{layer_idx} not in window/protect, discard")
                    self._inflight_cpu_layers.discard(layer_idx)
                    self._cpu_pf_q.task_done()
                    continue

                # ⭐ 修复：如果已经在 CPU cache 中，跳过（避免重复加载）
                if layer_idx in self.cpu_cache:
                    self._inflight_cpu_layers.discard(layer_idx)
                    self._cpu_pf_q.task_done()
                    if getattr(self, "verbose", False):
                        print(f"[WSM DEBUG] Layer {layer_idx} already in CPU cache, skip redundant load")
                    continue

                # 回滞式收缩：仅踢窗口外（避免抖动）
                self._evict_if_over_hwm_locked(incoming=1)

                self.cpu_cache[layer_idx] = tmp
                self._inflight_cpu_layers.discard(layer_idx)

            self._cpu_pf_q.task_done()
            print(f"[WSM] ✅ Loaded layer {layer_idx} to CPU cache ({len(tmp)} params)")


    def _cpu_dispatch_loop(self):
        """
        单线程调度器：从队列按 FIFO 顺序取任务，提交到线程池并行执行。
        这样既保证了 FIFO 入队顺序，又实现了并行 SSD 读取。
        """
        while not (self._stopped or self._stop_event.is_set()):
            try:
                item = self._cpu_pf_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # 支持关停的哨兵
            if item is None or (isinstance(item, tuple) and item[1] is None):
                self._cpu_pf_q.task_done()
                continue

            epoch, layer_idx = item

            # 快速检查：如果不在窗口内，直接跳过
            with self._cpu_lock:
                in_win = self._layer_in_cpu_window(layer_idx)
                in_protect = (int(layer_idx) in getattr(self, "_cpu_protect_set", set()))
                if not (in_win or in_protect):
                    if self.verbose:
                        print(f"[WSM DEBUG] _cpu_dispatch_loop: L{layer_idx} not in window (cpu_win_base={self.cpu_win_base}, cpu_cache_cap={self.cpu_cache_cap}, cpu_ring_mode={self.cpu_ring_mode})")
                    self._inflight_cpu_layers.discard(layer_idx)
                    self._cpu_pf_q.task_done()
                    continue

            # 提交到线程池并行执行
            if self._cpu_executor is not None:
                self._cpu_executor.submit(self._cpu_pf_task, epoch, layer_idx)
            else:
                # 降级到同步执行（不应该发生）
                self._cpu_pf_task(epoch, layer_idx)


    def _cpu_pf_task(self, epoch: int, layer_idx: int):
        """
        并行 CPU 预取任务（在线程池中执行）。
        使用线程私有的 staging buffer 避免数据竞争。
        """
        # 过期保护
        layer_idx = self._wrap(int(layer_idx))     # 统一环形层号
        with self._cpu_lock:
            if epoch != self._epoch:
                self._inflight_cpu_layers.discard(layer_idx)
                self._cpu_pf_q.task_done()
                return
            in_win = self._layer_in_cpu_window(layer_idx)
            in_protect = (int(layer_idx) in getattr(self, "_cpu_protect_set", set()))
            if not (in_win or in_protect):
                self._inflight_cpu_layers.discard(layer_idx)
                self._cpu_pf_q.task_done()
                return

        # 并行读：使用线程私有 staging buffer
        try:
            tmp = self._read_layer_from_ssd_threadsafe(layer_idx)
        except Exception as e:
            if self.verbose:
                print(f"[WSM] SSD read failed L{layer_idx}: {e}")
            with self._cpu_lock:
                self._inflight_cpu_layers.discard(layer_idx)
            self._cpu_pf_q.task_done()
            return

        # 落地到 CPU cache（加锁）
        with self._cpu_lock:
            if epoch != self._epoch:
                self._cpu_pf_q.task_done()
                return
            # ⭐ 关键修复：加载后也要检查保护集，避免wraparound层被丢弃
            in_win_after = self._layer_in_cpu_window(layer_idx)
            in_protect_after = (int(layer_idx) in getattr(self, "_cpu_protect_set", set()))
            if not (in_win_after or in_protect_after):
                self._inflight_cpu_layers.discard(layer_idx)
                self._cpu_pf_q.task_done()
                return

            # HWM/LWM 驱逐
            self._evict_if_over_hwm_locked(incoming=1)

            # 安装到 CPU cache
            self.cpu_cache[layer_idx] = tmp
            self._inflight_cpu_layers.discard(layer_idx)

        self._cpu_pf_q.task_done()

        if self.verbose:
            print(f"[WSM] ✅ Loaded layer {layer_idx} to CPU cache ({len(tmp)} params)")

        # CPU 完成回调：如果有人等待这个层的组，立即触发 GPU 预取
        self._on_cpu_layer_ready(layer_idx)


    def _on_cpu_layer_ready(self, layer_idx: int):
        """
        CPU 层加载完成回调：主动推送等待中的 GPU 预取。

        当一个 CPU 层加载完成后，检查是否有等待该层的 GPU 组预取请求，
        如果有则立即触发 H2D 传输。这样可以避免：
        1. 过早占用 H2D 信号量/GPU 预算
        2. 事件和实际数据的不一致
        3. 无效的等待和重试

        Args:
            layer_idx: 刚完成加载的 CPU 层索引
        """
        # 快速取出并移除等候项
        wants = []
        with self._cpu_lock:
            for g in ("attn", "ffn"):
                key = (layer_idx, g)
                if key in self._gpu_need_on_cpu_ready:
                    wants.append(g)
            # 移除等待标记
            for g in wants:
                self._gpu_need_on_cpu_ready.discard((layer_idx, g))

        # 立即发起 H2D（仍会受 H2D 信号量和 GPU 预算约束）
        for g in wants:
            try:
                success = self.prefetch_group_async(layer_idx, g, pin=False, reason="cpu-ready")
                if self.verbose and success:
                    print(f"[WSM] CPU ready triggered GPU prefetch: layer={layer_idx} group={g}")
            except Exception as e:
                if self.verbose:
                    print(f"[WSM] Failed to trigger GPU prefetch for ({layer_idx}, {g}): {e}")

        # 可选：顺带触发窗口顶补，填实 GPU 预算
        # if wants:
        #     try:
        #         # 使用最近执行的层作为锚点，或者使用刚就绪的层
        #         anchor = getattr(self, "_last_executed_layer", layer_idx)
        #         self.rebalance_and_topoff(anchor)
        #     except Exception as e:
        #         if self.verbose:
        #             print(f"[WSM] rebalance_and_topoff failed in cpu_ready callback: {e}")
                    
        # 回调末尾顺手把 CPU 填到目标
        # try:
        #     self._cpu_pump_fill(reason="cpu-ready", max_new=2)
        # except Exception as e:
        #     if getattr(self, "verbose", False):
        #         print(f"[WSM] cpu-pump on ready failed: {e}")


    def prefetch_group_async(self, layer_idx: int, group: str,
                            pin: bool = False, reason: str | None = None,
                            h2d_override=None) -> bool:
        """
        统一的组级预取入口：
        - CPU 未就绪：登记等待 + 放占位事件 + 入队后台线程
        - 无 H2D 配额：放占位事件 + 入队后台线程
        - 有 H2D 配额：直接在 H2D 流上安装 + 记录真实事件 + set host_evt
        始终把 5 元组 (epoch, key, pin, reason, h2d_override) 放入 _gpf_q。
        """
        import torch, threading
        key = (int(layer_idx), 'attn' if group == 'attn' else 'ffn')

        # --- 先保证占位事件 & host 事件存在（无论 CPU 是否就绪）---
        with self._group_lock:
            evt = self._group_events.get(key)
            if evt is None:
                evt = torch.cuda.Event(blocking=False)
                self._group_events[key] = evt
                getattr(self, "_placeholder_keys", set()).add(key)
            host_evt = self._group_recorded_host.get(key)
            if host_evt is None:
                host_evt = threading.Event()
                self._group_recorded_host[key] = host_evt

        # --- 再判定 CPU 是否就绪 ---
        if not self._cpu_group_ready(layer_idx, key[1]):
            # ⭐ 主动触发 CPU 加载（避免无限等待）
            if getattr(self, "ssd_enabled", False):
                try:
                    self._cpu_try_enqueue(layer_idx, reason=f"prefetch_{key[1]}")
                except Exception:
                    pass
            # 回投 CPU 等待队列，但我们已经有占位 & host_evt 了
            self._requeue_gpf(key, h2d_override, pin, reason="cpu_wait")
            return True  # 表示已提交（占位），让上层能够等待
        # 状态快速检查与失配修正
        st = self._get_state(key)
        if st == "RESIDENT" and (not self._is_group_physically_resident(layer_idx, key[1])):
            if self.verbose:
                print(f"[WSM] stale RESIDENT -> CPU for {key}")
            with self._group_lock:
                self._set_state(key, "CPU")
            st = "CPU"
        elif st == "INFLIGHT":
            evt = self._group_events.get(key)
            # host = self._group_recorded_host.get(key)
            if (evt is None) :
                if self.verbose:
                    print(f"[WSM] stale INFLIGHT(no evt) -> CPU for {key}")
                with self._group_lock:
                    self._set_state(key, "CPU")
                    self._gpu_group_inflight.discard(key)
                    self._group_recorded_host.pop(key, None)
                st = "CPU"

        # 已在途/常驻：直接返回
        if st in ("RESIDENT", "INFLIGHT") or key in self._gpu_group_inflight:
            return True

        # 预算门：仅拦“机会型”预取；必需型（get_event/wait）放行
        if not self._gpu_budget_allows_new_group(key, reason):
            return False

        # ---- 情况 A：CPU 未就绪 → 先占位，入队后台线程 ----
        if getattr(self, "ssd_enabled", False) and not self._cpu_group_ready(layer_idx, key[1]):
            with self._cpu_lock:
                self._gpu_need_on_cpu_ready.add(key)
            enq_ok = False
            try:
                enq_ok = self._cpu_try_enqueue(layer_idx, reason=reason or "prefetch")
            except Exception:
                enq_ok = False
            if not enq_ok:
                with self._cpu_lock:
                    self._gpu_need_on_cpu_ready.discard(key)
                return False

            placeholder_evt = torch.cuda.Event(blocking=False)
            host_evt        = threading.Event()
            with self._group_lock:
                self._group_events[key]        = placeholder_evt
                self._group_recorded_host[key] = host_evt
                self._placeholder_keys.add(key)
                self._set_state(key, "INFLIGHT")
                self._gpu_group_inflight.add(key)
                if pin:
                    cnt = self._pinned_groups.get(key, 0)
                    if cnt < self.max_pinned_groups:
                        self._pinned_groups[key] = cnt + 1

            # 入队（5 元组）
            try:
                self._gpf_q.put_nowait((self._epoch, key, bool(pin), reason or "cpu_wait", h2d_override))
            except Exception:
                # 回滚占位（极少发生）
                with self._group_lock:
                    self._group_events.pop(key, None)
                    self._group_recorded_host.pop(key, None)
                    self._placeholder_keys.discard(key)
                    self._gpu_group_inflight.discard(key)
                    if self._get_state(key) == "INFLIGHT":
                        self._set_state(key, "CPU")
                with self._cpu_lock:
                    self._gpu_need_on_cpu_ready.discard(key)
                return False
            return True

        # ---- 情况 B：无 H2D 配额 → 先占位，入队后台线程 ----
        if not self._h2d_acquire_token(timeout=0):
            placeholder_evt = torch.cuda.Event(blocking=False)
            host_evt        = threading.Event()
            with self._group_lock:
                self._group_events[key]        = placeholder_evt
                self._group_recorded_host[key] = host_evt
                self._placeholder_keys.add(key)
                self._set_state(key, "INFLIGHT")
                self._gpu_group_inflight.add(key)
                if pin:
                    cnt = self._pinned_groups.get(key, 0)
                    if cnt < self.max_pinned_groups:
                        self._pinned_groups[key] = cnt + 1
            try:
                self._gpf_q.put_nowait((self._epoch, key, bool(pin), reason or "queued", h2d_override))
                return True
            except Exception:
                with self._group_lock:
                    self._group_events.pop(key, None)
                    self._group_recorded_host.pop(key, None)
                    self._placeholder_keys.discard(key)
                    self._gpu_group_inflight.discard(key)
                    if self._get_state(key) == "INFLIGHT":
                        self._set_state(key, "CPU")
                return False

        # ---- 情况 C：立即 H2D（有配额） → 直接安装并记录真实事件 ----
        try:
            if group == "attn":
                h2d_stream = h2d_override or getattr(self.streams, "weight_h2d_mha", None)
            else:
                h2d_stream = h2d_override or getattr(self.streams, "weight_h2d_ffn", None)
            if h2d_stream is None and torch.cuda.is_available():
                h2d_stream = torch.cuda.current_stream()

            inflight_evt = torch.cuda.Event(blocking=False)
            recorded_host = self._group_recorded_host.get(key) or threading.Event()
            with self._group_lock:
                self._set_state(key, "INFLIGHT")
                self._gpu_group_inflight.add(key)
                self._group_events[key]        = inflight_evt
                self._group_recorded_host[key] = recorded_host
                self._placeholder_keys.add(key)
                if pin:
                    cnt = self._pinned_groups.get(key, 0)
                    if cnt < self.max_pinned_groups:
                        self._pinned_groups[key] = cnt + 1

            # 真正 H2D + 在 H2D 流上 record 同一个事件
            with torch.cuda.stream(h2d_stream):
                self._install_group_on_gpu(layer_idx, group, h2d_override=h2d_stream)
                inflight_evt.record(h2d_stream)

            # host 侧 OK → 通知 wait_group_ready 可以安全 wait_event
            

            with self._group_lock:
                # 进入 ring，保持 INFLIGHT；RESIDENT 升级由查询+使用时机判定
                self._placeholder_keys.discard(key)
                try:
                    if key in self._gpu_group_ring:
                        self._gpu_group_ring.remove(key)
                    self._gpu_group_ring.append(key)
                except ValueError:
                    self._gpu_group_ring.append(key)
                self._placeholder_keys.discard(key)
            recorded_host.set()
            return True
        finally:
            self._h2d_release_token()




    def _do_prefetch_once(self, layer_idx: int, group: str,
                        inflight_evt: "torch.cuda.Event",
                        h2d_override: "torch.cuda.Stream|None" = None,
                        pin: bool = False) -> None:
        """
        核心的一次组级 H2D 预取：
        1) 确保本层 CPU cache 已就绪（必要时触发 _load_layer_to_cpu）
        2) 选择 H2D stream（attn / ffn 分流；若调用方传入 h2d_override 则使用之）
        2.5) ✨ 设置 INFLIGHT 状态（仅在确认即将执行 H2D 时）
        3) 将该组所有参数从 CPU → GPU（非阻塞 .to(self.device)）
        4) 在该 H2D stream 上 record 组级 ready 事件（供 wait_group_ready 挂依赖）
        5) 状态切换 INFLIGHT → RESIDENT；更新 ring；释放 _h2d_sem
        """
        key = (int(layer_idx), 'attn' if group == 'attn' else 'ffn')
        debug = getattr(self, "debug_prefetch", False)

        # 组内权重后缀集合
        suffixes = ("attention.wq.weight", "attention.wk.weight",
                    "attention.wv.weight", "attention.wo.weight") if group == "attn" else \
                ("feed_forward.w1.weight", "feed_forward.w3.weight", "feed_forward.w2.weight")

        # ---------- 1) 确保 CPU cache ----------
        try:
            if self.ssd_enabled and (layer_idx not in self.cpu_cache):
                # 尽量不阻塞：等待一下；还不在就同步加载一遍（只此一层）
                # ⭐ 修复：增加超时时间从 0.5s → 3.0s，给 CPU prefetch 线程更多时间
                try:
                    self._wait_cpu_ready(layer_idx, timeout=3.0)
                except Exception:
                    self._load_layer_to_cpu(layer_idx)
        except Exception as e:
            # CPU cache 失败不应崩溃；让下面的"从 model 直接取"做兜底
            if self.verbose:
                print(f"[WSM] CPU cache ensure failed for L{layer_idx}: {e}")

        with self.cpu_cache_lock:
            layer_cache = dict(self.cpu_cache.get(layer_idx, {}))

        # ---------- 2) 选择 H2D stream ----------
        # 优先调用方 override（例如 wait_on_demand 场景）；否则用固定分流
        h2d_stream = None
        if h2d_override is not None:
            h2d_stream = h2d_override
        else:
            # 按组分流：attn→weight_h2d_mha，ffn→weight_h2d_ffn（已有方法）
            h2d_stream = self._select_h2d_stream_for(module_name=("attention" if group=="attn" else "feed_forward"))

        # ---------- 2.5) ✨ 设置 INFLIGHT 状态 ----------
        recorded_host = self._group_recorded_host.get(key)
        if recorded_host is None:
            recorded_host = threading.Event()
        # 统一源：让 wait_group_ready 拿到的是“同一个”对象
        self._group_recorded_host[key] = recorded_host
        # 仅在确认即将执行 H2D 时才设置状态，避免留下假 INFLIGHT
        with self._group_lock:
            self._set_state(key, "INFLIGHT")
            self._gpu_group_inflight.add(key)
            self._group_events[key] = inflight_evt
            self._group_recorded_host[key] = recorded_host
            # 处理 pin 计数
            if pin:
                cnt = self._pinned_groups.get(key, 0)
                if cnt < self.max_pinned_groups:
                    self._pinned_groups[key] = cnt + 1

        # ---------- 3-5) H2D 主流程（异常安全） ----------
        h2d_success = False
        try:
            # ---------- 3) CPU→GPU 拷贝（非阻塞） ----------
            # 估算显存并适度逐出，为该组合理出空间（排除自己）
            need_bytes = 0
            for suf in suffixes:
                pname = f"layers.{layer_idx}.{suf}"
                t = layer_cache.get(pname)
                if t is None:  # 兜底：从模型已存在的 CPU 参数拿一份
                    t = self._get_param_from_model(layer_idx, suf)
                if t is not None:
                    need_bytes += int(t.numel()) * int(t.element_size())

            try:
                self._ensure_gpu_headroom(need_bytes, exclude={key})
            except Exception:
                # 尝试小范围收缩再继续
                self._shrink_gpu_groups_now(exclude={key})

            # 把该组的每个参数搬到 GPU，并"安装"到对应的 Parameter 上
            for suf in suffixes:
                pname = f"layers.{layer_idx}.{suf}"
                cpu_t = layer_cache.get(pname)
                if cpu_t is None:
                    # 二级兜底：显式从 SSD 读取单个参数（若 manifest 存在）
                    try:
                        cpu_t = self._load_param_from_ssd(pname) if self.ssd_enabled else None
                    except Exception:
                        cpu_t = None
                if cpu_t is None:
                    # 三级兜底：从模型里找一份 CPU 张量（不应频繁走到）
                    cpu_t = self._get_param_from_model(layer_idx, suf)
                if cpu_t is None:
                    # 到此仍为空，说明该权重确实不存在（不应该发生），跳过
                    if self.verbose:
                        print(f"[WSM WARN] missing CPU tensor for {pname}; skip")
                    continue

                # ✅ P0-3: 使用带超时和重试的 H2D 传输
                try:
                    if h2d_stream is not None:
                        dst = self._h2d_transfer_with_retry(cpu_t, pname, h2d_stream)
                    else:
                        # 阻塞模式保持原样（通常不应走到这里）
                        dst = cpu_t.to(self.device, non_blocking=False)
                except torch.cuda.OutOfMemoryError:
                    # 再退一步：主动收缩再重试一次
                    self._shrink_gpu_groups_now(exclude={key})
                    if h2d_stream is not None:
                        dst = self._h2d_transfer_with_retry(cpu_t, pname, h2d_stream)
                    else:
                        dst = cpu_t.to(self.device, non_blocking=False)

                # 安装到 Parameter（覆盖 param.data or 更新 owner）
                self._install_param_tensor(pname, dst)

            # ---------- 4) 记录组级 ready 事件 ----------
            # 重要：event 要在"对应 H2D stream"上 record，这样 wait_group_ready 才能把 compute stream 挂依赖
            # 注意：inflight_evt 已在前面（step 2.5）占位到 _group_events[key]
            # _record_group_ready_event 会复用它并在 h2d_stream 上记录
            try:
                self._record_group_ready_event(layer_idx, group, stream=h2d_stream)
            except Exception as e:
                if self.verbose:
                    print(f"[WSM] record ready event failed for {key}: {e}")

            # ✨ 修复：set host-side 事件（通知 wait_group_ready 可以安全 wait_event）
            recorded_host.set()

            # ---------- 5) 状态切换 / ring 更新 ----------
            # ⭐ 核心改动：保持 INFLIGHT，直到确认事件完成后再升级为 RESIDENT
            # 不要在 H2D 异步发起后立刻把状态置为 RESIDENT
            # wait_group_ready 或 _group_is_resident 会在事件完成后升级状态
            with self._group_lock:
                # 保持 INFLIGHT 状态，不提前置为 RESIDENT
                # self._set_state(key, "RESIDENT")  # ❌ 删除此行
                try:
                    # 去重插入 ring
                    if key in self._gpu_group_ring:
                        self._gpu_group_ring.remove(key)
                    self._gpu_group_ring.append(key)
                except Exception:
                    pass
                # 注意：不清理 inflight 标记，保持 INFLIGHT 直到事件完成
                # self._gpu_group_inflight.discard(key)  # ❌ 删除此行

            h2d_success = True

        finally:
            # ✅ P0-2: 无论成功失败，都要释放令牌
            try:
                self._h2d_release_token()
            except Exception:
                pass

            # 如果失败，清理可能的不一致状态
            if not h2d_success:
                with self._group_lock:
                    # 回退到 CPU 状态
                    self._set_state(key, "CPU")
                    # 清理事件
                    self._group_events.pop(key, None)
                    # 清理 inflight 标记
                    self._gpu_group_inflight.discard(key)
                    # 清理 pin 计数
                    if key in self._pinned_groups:
                        self._pinned_groups.pop(key, None)

        if debug:
            print(f"[WSM DEBUG][prefetch] H2D enqueued (INFLIGHT) for {key}, ring_sz={len(self._gpu_group_ring)}")


    


    def _prefetch_kv_for_layer(self, layer_idx: int):
        """
        在预取 attention 权重时同步预取该层需要的 KV cache。
        Prefetch KV cache for the layer while prefetching attention weights.

        调用 KVOffloader.prefetch_async() 触发 SSD→DRAM + DRAM→GPU 的异步传输。
        Calls KVOffloader.prefetch_async() to trigger async SSD→DRAM + DRAM→GPU transfer.
        """
        # 检查是否有 KV offloader
        if self.kv_offloader is None:
            # 尝试从模型中获取
            try:
                first_layer = self.model.layers[0]
                if hasattr(first_layer, "attention") and hasattr(first_layer.attention, "offloader"):
                    self.kv_offloader = first_layer.attention.offloader
            except Exception:
                return

        if self.kv_offloader is None:
            return

        # 调用 offloader 的现有方法（已经实现了 SSD→DRAM + DRAM→GPU）
        try:
            # 获取推理状态（从全局跟踪器或简化假设）
            from .global_state_tracker import get_global_tracker
            tracker = get_global_tracker()
            batch_idx = getattr(tracker, "current_batch", 0) if tracker else 0

            # 预取该层重要的 KV 块（top-k）
            k = 4  # 预取 4 个最重要的块
            blocks = self.kv_offloader.topk_blocks(
                layer=layer_idx,
                k=k,
                batch_idx=batch_idx,
                strategy="hybrid"
            )

            if blocks:
                # 使用现有的异步预取方法
                self.kv_offloader.prefetch_async(
                    layer=layer_idx,
                    blocks=blocks,
                    bsz=1,  # 简化假设
                    device=self.device
                )
                if self.verbose:
                    print(f"[WSM] Prefetched KV for L{layer_idx}: blocks {blocks}")

        except Exception as e:
            if self.verbose:
                print(f"[WSM] KV prefetch failed for L{layer_idx}: {e}")

    

    def _maybe_schedule_cpu_prefetch(self, cur_layer: int):
        # 目标窗口 [cur_layer+1, cur_layer+self.cpu_prefetch_distance]
        # 使用环形范围以支持 wrap-around
        distance = getattr(self, 'cpu_prefetch_distance', 4)
        target = self._ring_range(self._wrap(cur_layer + 1), distance)
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
        """提交后台任务到共享线程池（避免创建大量线程导致调度瓶颈）"""
        self._bg_executor.submit(task)

    def _cpu_layer_ready(self, layer_idx: int) -> bool:
        """检查CPU层是否就绪"""
        return layer_idx in self.cpu_cache

    def _cpu_group_ready(self, layer_idx: int, group: str) -> bool:
        """
        检查指定组 (layer_idx, group) 是否已在 CPU cache 中完整可用。
        用于判断是否可以启动 H2D 传输。

        Args:
            layer_idx: 层索引
            group: "attn" 或 "ffn"

        Returns:
            True 如果该组的所有参数都在 CPU cache 中，否则 False
        """
        if group == "attn":
            suffixes = (
                "attention.wq.weight",
                "attention.wk.weight",
                "attention.wv.weight",
                "attention.wo.weight",
            )
        elif group == "ffn":
            suffixes = (
                "feed_forward.w1.weight",
                "feed_forward.w2.weight",
                "feed_forward.w3.weight",
            )
        else:
            return False

        with self.cpu_cache_lock:
            layer_data = self.cpu_cache.get(layer_idx)
            if not layer_data:
                return False

            # 检查所有参数是否都存在
            for suffix in suffixes:
                param_name = f"layers.{layer_idx}.{suffix}"
                if param_name not in layer_data:
                    return False

        return True

    def _evict_cpu_layers_older_than(self, layer_idx: int):
        """淘汰早于指定层的CPU缓存"""
        if not self.ssd_enabled:
            return

        evicted = 0
        with self.cpu_cache_lock:
            to_evict = [L for L in list(self.cpu_cache.keys()) if (L < layer_idx) and (L not in self._cpu_protect_set)]
            for L in to_evict:
                self.cpu_cache.pop(L, None)
                evicted += 1
                if self.verbose:
                    print(f"[WSM] Evicted CPU cache layer {L}")
                    
        if evicted > 0 and self.verbose:
            print(f"[WSM] CPU older-than evict: evicted={evicted} (< {layer_idx})")

    def warmup_groups_prefetch(self, layers: Optional[int] = None,
                            scheme: str = "doublet",
                            blocking_first: bool = True) -> None:
        """
        Warmup on GPU at *group* granularity (attn/ffn).

        Args:
            layers: 计划热身的前若干层（默认使用 warmup_layers）
            scheme: "doublet" => 0.attn,0.ffn,1.attn,1.ffn,...
                    "attn-first" => 先铺一列 attn，再铺 ffn
                    "pairwise-nearest" => 使用 _plan_pairwise_nearest (0.ffn, 1.attn, 1.ffn, 2.attn, 2.ffn, ...)
            blocking_first: 第一个组用阻塞式 ensure，确保至少一组就绪
        """
        if layers is None:
            layers = max(1, int(getattr(self, "warmup_layers", 1)))
        layers = min(layers, self.n_layers)

        # SSD 模式先把 CPU 窗口暖起来，避免首次组装载 miss
        if self.ssd_enabled:
            try:
                self.warmup_cpu_cache()
            except Exception:
                self._ensure_cpu_window()

        # 生成 warmup 计划（受 GPU 组预算约束）
        plan: list[tuple[int, str]] = []
        if scheme == "pairwise-nearest":
            # 使用新的成对最近优先规划器
            # 从第 0 层开始，深度设为 layers-1
            depth = max(0, layers - 1)
            plan = self._plan_pairwise_nearest(0, depth)
            # 限制在 GPU 预算内
            plan = plan[:self.gpu_max_groups]
        elif scheme == "attn-first":
            for i in range(layers):
                if len(plan) >= self.gpu_max_groups: break
                plan.append((i, "attn"))
            for i in range(layers):
                if len(plan) >= self.gpu_max_groups: break
                plan.append((i, "ffn"))
        else:  # "doublet"
            for i in range(layers):
                if len(plan) >= self.gpu_max_groups: break
                plan.append((i, "attn"))
                if len(plan) >= self.gpu_max_groups: break
                plan.append((i, "ffn"))

        # 去重（理论上不应重复，这里稳妥处理一下）
        uniq, seen = [], set()
        for item in plan:
            if item not in seen:
                seen.add(item)
                uniq.append(item)
        plan = uniq

        # 执行：纯异步 warmup（不阻塞主线程）
        for (lid, grp) in plan:
            self.prefetch_group_async(lid, grp, pin=False, reason="warmup")

        if self.verbose:
            msg = ", ".join(f"L{l}.{g}" for (l, g) in plan)
            print(f"[WSM] Warmup group prefetch: {msg} | cap={self.gpu_max_groups}")
            try:
                self.print_group_residency(current=plan[0] if plan else None,
                                        header="[WSM][groups][warmup]")
            except Exception:
                pass
    

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
        Asynchronously prefetch a list of layer indices, respecting the GPU group budget.

        Args:
            ids: Layer indices to prefetch
            warm: If True, treat as warmup (idempotent, uses sliding window)
        """
        if warm:
            self.warmup_cpu_cache()
            return

        if not ids:
            return
        ids = ids[: max(1, min(self._pd_current, self.pd_cap))]
        nvtx.range_push(f"prefetch_layers_{ids}")

        for idx in ids:
            nvtx.range_push(f"prefetch_layer_{idx}")
            self.prefetch_group_async(idx, "attn", pin=False, reason="layer-prefetch")
            self.prefetch_group_async(idx, "ffn", pin=False, reason="layer-prefetch")
            nvtx.range_pop()

        nvtx.range_pop()
        
        
    # --------- PD 自适应（滞回+EMA） ---------
    def _decide_pd(self):
        """
        估计 PCIE 忙闲（EMA），并结合 pinned 水位 + inflight 组数做滞回调整：
        - 忙：PCIE>hi 或 pinned<low 或 inflight>上限 -> PD=1，暂停 KV 写 throttle_ms
        - 闲：PCIE<lo 且 pinned>high 且 inflight<下限 -> PD=min(PD+1, cap)
        - 中：保持

        ⭐ 新增：使用 inflight 组数作为背压信号（credit-based backpressure）
        """
        # 以两条 H2D 流的 backlog 作为 proxy
        def _busy(s):
            try:
                return (s is not None) and (not s.query())
            except Exception:
                return False
        s_mha = getattr(self.streams, "weight_h2d_mha", None)
        s_ffn = getattr(self.streams, "weight_h2d_ffn", None)
        busy_proxy = 0.9 if (_busy(s_mha) or _busy(s_ffn)) else 0.1
        # EMA
        self._pcie_ema = self._ema_alpha * busy_proxy + (1.0 - self._ema_alpha) * self._pcie_ema

        # pinned 水位：此处无法读取系统 pinned 池，采用保守估计 0.5；如果你有 HostPinnedExtentPool，可在外层注入
        pinned_free_ratio = 0.5

        # ⭐ inflight 组数作为背压信号
        with self._group_lock:
            inflight_count = len(self._gpu_group_inflight)
        inflight_ratio = inflight_count / max(1, self.max_inflight_groups)

        # 忙态：降 PD、暂停写
        # 条件：PCIE 忙 或 pinned 低 或 inflight 超过 80% 上限
        if (self._pcie_ema >= self.pcie_hi) or (pinned_free_ratio <= self.pin_lo) or (inflight_ratio >= 0.8):
            # step-down, not cliff-drop
            self._pd_current = max(1, self._pd_current - 1)
            if self.kv_offloader is not None:
                try:
                    self.kv_offloader.throttle_writes_for(self.throttle_ms)
                except Exception:
                    pass
            return

        # 闲态：升 PD
        # 条件：PCIE 闲 且 pinned 高 且 inflight 低于 50% 下限
        if (self._pcie_ema <= self.pcie_lo) and (pinned_free_ratio >= self.pin_hi) and (inflight_ratio <= 0.5):
            self._pd_current = min(self._pd_current + 1, self.pd_cap)
            return
        # 中态：不变


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
                        "gpu_max_groups": self.gpu_max_groups,
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

    def shutdown(self, wait: bool = True, timeout: float = 2.0):
        self._stopped = True
        self._stop_event.set()
        try:
            self._cpu_pf_q.put_nowait((self._epoch, None))  # 唤醒 worker
        except Exception:
            pass

        # 关闭线程池
        if self._cpu_executor is not None:
            try:
                self._cpu_executor.shutdown(wait=wait, cancel_futures=True)
            except Exception as e:
                if self.verbose:
                    print(f"[WSM] Error shutting down CPU executor: {e}")

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
