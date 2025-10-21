import json
import torch.nn as nn
from typing import Optional, List, Dict, Any
import math
from dataclasses import dataclass, field, fields, is_dataclass

@dataclass
class LayerInfo:
    layer_id:int
    block: Optional[nn.Module] = None   #encoderblock
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KVCacheArgs:
    # ssd_path: str = "/mnt/kv_cache/kv_cache.bin"
    ssd_size_gb: int = 500
    dram_limit_gb: float = 8
    ssd_device_path: str = "/dev/nvme0n1p4"  # Raw block device path for KV cache
    max_concurrent_io: int = 4
    ssd_capacity_gb: int = 100
    block_bytes       = 256 * 1024
    preallocate       = False
    lazy_init         = True
    # KV cache data type (default: float16)
    kv_dtype          = None  # Will default to torch.float16 if None
    prefer_bf16       = False # If True, use bfloat16 for KV cache (overrides kv_dtype if None)
    # Verbose pool initialization messages
    verbose_pool      = True 

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
    # 新增：参数初始化所用的设备（"meta" | "cpu" | "cuda"）。70B 走 "meta"
    param_init_device: str = "cpu"
    # 新增：权重来源（"checkpoint" | "raw-ssd"），70B 走 "raw-ssd"
    weight_source: str = "checkpoint"
    
    
    @staticmethod
    def from_json(params_path: str,
                  max_seq_len: int,
                  max_batch_size: int,
                  device: str = None,
                  memory_limit_gb: float = None):
        try:
            from .gpu_utils import get_optimal_device, GPUHealthMonitor
            from .memory_manager import MemoryConfig, set_global_memory_limit, get_memory_info
        except ImportError:
            import torch
            
            def get_optimal_device(prefer_cuda=True, min_memory_gb=1.0):
                if prefer_cuda and torch.cuda.is_available():
                    return "cuda"
                return "cpu"
            def set_global_memory_limit(limit_gb, device, reserved_gb=1.0):
                pass
            def get_memory_info(device):
                return {"no_limit": True}

            class MemoryConfig:
                def __init__(self, max_hbm_gb=14.0, reserved_hbm_gb=1.0,
                           enable_monitoring=True, cleanup_threshold=0.9,
                           oom_retry_count=3, monitor_interval=1.0):
                    self.max_hbm_gb = max_hbm_gb
                    self.reserved_hbm_gb = reserved_hbm_gb
                    self.enable_monitoring = enable_monitoring
                    self.cleanup_threshold = cleanup_threshold
                    self.oom_retry_count = oom_retry_count
                    self.monitor_interval = monitor_interval
        
        if device is None:
            device = get_optimal_device(prefer_cuda=True, min_memory_gb=1.0)
            print(f"Auto-selected device: {device}")
        else:
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
        
        # 处理 memory_limit 参数
        if memory_limit_gb is not None:
            memory_limit_config = MemoryConfig(max_hbm_gb=memory_limit_gb)
        elif device and device.startswith("cuda"):
            try:
                memory_info = get_memory_info(device)
                if "total_gb" in memory_info:
                    # 控制默认最大使用HBM为总量的95%
                    auto_limit = memory_info["total_gb"] * 0.95
                    reserved_limit = memory_info["total_gb"] * 0.05  
                    memory_limit_config = MemoryConfig(
                        max_hbm_gb=auto_limit,
                        reserved_hbm_gb=reserved_limit
                    )
                    print(f"Auto-set memory limit: {auto_limit:.1f}GB (reserved: {reserved_limit:.1f}GB)")
                else:
                    memory_limit_config = MemoryConfig()  
            except Exception as e:
                print(f"Failed to auto-set memory limit: {e}")
                memory_limit_config = MemoryConfig()  
        else:
            memory_limit_config = MemoryConfig()  

        args =  ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **filtered
        )

        # 动态添加 memory_limit 属性
        args.memory_limit = memory_limit_config

        if device and device.startswith("cuda"):
            try:
                set_global_memory_limit(config=args.memory_limit, device=device)
            except Exception as e:
                print(f"Warning: Failed to set global memory limit: {e}")

        args.layer_infos = [LayerInfo(layer_id=i) for i in range(args.n_layers)]
        return args
    
    
# ---------- 运行期总开关 ----------
@dataclass
class RuntimeFlags:
    FINE_GRAINED_PIPELINE: bool = True
    INFER_MODE_DEFAULT: str = "prefill"        # "prefill" or "decode"

# ---------- HBM 窗口与预取 ----------
@dataclass
class WindowPrefetchConfig:
    HBM_WINDOW_PREFILL: int = 3
    HBM_WINDOW_DECODE: int = 5
    PREFETCH_DIST_PREFILL_DEFAULT: int = 2
    PREFETCH_DIST_DECODE_DEFAULT: int = 2
    PD_CAP: int = 3

# ---------- pinned 池与注册 ----------
@dataclass
class PinnedPoolConfig:
    WEIGHT_PINNED_BYTES: int = 64 << 30        # 64 GiB
    KV_PINNED_BYTES: int     = 12 << 30        # 12 GiB
    EXTENT_BYTES: int        = 2  << 20        # 2 MiB 逻辑粒度
    PINNED_REGISTER_CHUNK: int = 512 << 20     # 512 MiB 注册一块
    PINNED_REGISTER_N: Optional[int] = None    # 若为 None 自动按 ceil 计算

    def __post_init__(self):
        # 基础对齐检查（与 IO.RAW_IO_ALIGN_BYTES 的跨项校验在 validate_runtime_config 里做）
        if self.EXTENT_BYTES % 4096 != 0:
            raise ValueError("EXTENT_BYTES 必须是 4KiB 的整数倍")
        if self.PINNED_REGISTER_N is None:
            self.PINNED_REGISTER_N = math.ceil(self.WEIGHT_PINNED_BYTES / self.PINNED_REGISTER_CHUNK)

# ---------- DRAM 注册传送带 ----------
@dataclass
class RegisteredPoolConfig:
    REG_POOL_N_BUFFERS: int = 12
    REG_POOL_BUF_BYTES: int = 32 << 20        # 单缓冲 ~32 MiB，总 ~384 MiB

# ---------- RAW I/O 与节流 ----------
@dataclass
class IOConfig:
    RAW_IO_USE_ODIRECT: bool = True
    RAW_IO_ALIGN_BYTES: int = 4096
    RAW_IO_QD_READ: int = 32                 # 先保守，压测稳定后可调到 48
    RAW_IO_QD_WRITE: int = 8                 # 先保守，压测稳定后可调到 24
    RAW_PREFETCH_ON_COMPUTE: bool = True
    PINNED_LOW_WATERMARK: float = 0.20       # 低水位：强制 PD=1
    PINNED_HIGH_WATERMARK: float = 0.30      # 高水位：滞回上限
    PCIE_BUSY_UTIL_TH_HI: float = 0.70       # 繁忙阈值
    PCIE_BUSY_UTIL_TH_LO: float = 0.60       # 恢复阈值（滞回）
    IO_RAW_THROTTLE_MS: int = 30             # 预取节流窗口

# ---------- Warmup / Bootstrap ----------
@dataclass
class WarmupBootstrapConfig:
    WARMUP_HOTSET_LAYERS: int = 10           # L0..L9 MHA promote→pinned
    DECODE_BOOTSTRAP_LAYERS: int = 5         # Prefill 末把 0..4 MHA 推 HBM
    BOOTSTRAP_PROTECT: bool = True

# ---------- 策略修饰 ----------
@dataclass
class StreamingPolicyConfig:
    NEXT_USE_TIE_BREAKER: str = "size"       # or "cost"
    EVICT_PREFETCH_FIRST: bool = True
    PREFETCH_MIN_RETAIN_MS: int = 60         # 建议= 2 * IO.IO_RAW_THROTTLE_MS
    PREFETCH_EVICT_DIST_GUARD: int = 1       # 仅当 dist > (W+PD)//2 才优先踢

# ---------- 系统稳定性 ----------
@dataclass
class SystemStabilityConfig:
    REQUIRE_MEMLOCK_UNLIMITED: bool = True
    SWAPPINESS_LOW: bool = True
    PERSISTENT_MODE: bool = True             # nvidia-smi -pm 1

# ---------- 监控 ----------
@dataclass
class MonitoringConfig:
    enable_monitoring: bool = True
    monitor_interval_s: float = 1.0

# ---------- 汇总 ----------
@dataclass
class RuntimeConfig:
    flags: RuntimeFlags = field(default_factory=RuntimeFlags)
    window: WindowPrefetchConfig = field(default_factory=WindowPrefetchConfig)
    pinned: PinnedPoolConfig = field(default_factory=PinnedPoolConfig)
    regpool: RegisteredPoolConfig = field(default_factory=RegisteredPoolConfig)
    io: IOConfig = field(default_factory=IOConfig)
    warmup: WarmupBootstrapConfig = field(default_factory=WarmupBootstrapConfig)
    policy: StreamingPolicyConfig = field(default_factory=StreamingPolicyConfig)
    system: SystemStabilityConfig = field(default_factory=SystemStabilityConfig)
    monitor: MonitoringConfig = field(default_factory=MonitoringConfig)
    # 复用你已有的 KV 配置；若需要 KV 独立 QD，可在这里扩展字段或在 KV 层覆写
    kv_cache: 'KVCacheArgs' = field(default_factory=lambda: KVCacheArgs())

def _update_dataclass(dc_obj, updates: Dict[str, Any]):
    """递归更新 dataclass 字段（只更新已存在字段），支持子结构字典。"""
    for k, v in (updates or {}).items():
        if not hasattr(dc_obj, k):
            continue
        cur = getattr(dc_obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _update_dataclass(cur, v)
        else:
            setattr(dc_obj, k, v)

def validate_runtime_config(cfg: RuntimeConfig) -> RuntimeConfig:
    """做跨项校验并在需要时做安全收敛（避免直接抛异常导致进程退出）"""
    errs = []
    # 对齐关系：extent 必须是 RAW 对齐的整数倍
    if cfg.pinned.EXTENT_BYTES % cfg.io.RAW_IO_ALIGN_BYTES != 0:
        errs.append(f"EXTENT_BYTES({cfg.pinned.EXTENT_BYTES}) 不是 RAW_IO_ALIGN_BYTES({cfg.io.RAW_IO_ALIGN_BYTES}) 的整数倍")

    # 注册块覆盖能力：注册块总和必须 ≥ WEIGHT_PINNED_BYTES
    covered = cfg.pinned.PINNED_REGISTER_N * cfg.pinned.PINNED_REGISTER_CHUNK
    if covered < cfg.pinned.WEIGHT_PINNED_BYTES:
        # 自动上调块数到 ceil
        cfg.pinned.PINNED_REGISTER_N = math.ceil(cfg.pinned.WEIGHT_PINNED_BYTES / cfg.pinned.PINNED_REGISTER_CHUNK)

    # 水位与滞回区间
    if not (0.0 < cfg.io.PINNED_LOW_WATERMARK < cfg.io.PINNED_HIGH_WATERMARK < 1.0):
        errs.append("PINNED_[LOW,HIGH]_WATERMARK 需满足 0 < LOW < HIGH < 1")

    # QD 的安全范围
    if cfg.io.RAW_IO_QD_READ <= 0 or cfg.io.RAW_IO_QD_WRITE < 0:
        errs.append("RAW_IO_QD_[READ/WRITE] 非法（READ>0, WRITE>=0）")

    # 预取最短保留建议值：若未显式给出，自动取 2 * throttle
    if cfg.policy.PREFETCH_MIN_RETAIN_MS <= 0:
        cfg.policy.PREFETCH_MIN_RETAIN_MS = 2 * cfg.io.IO_RAW_THROTTLE_MS

    if errs:
        # 打印警告但不要强退（让上层可以继续跑压测）
        for e in errs:
            print(f"[RuntimeConfig][WARN] {e}")
    return cfg

def runtime_config_to_dict(cfg: RuntimeConfig) -> Dict[str, Any]:
    """把多层 dataclass 展平为 dict（便于日志/导出）"""
    def to_obj(x):
        if is_dataclass(x):
            return {f.name: to_obj(getattr(x, f.name)) for f in fields(x)}
        if isinstance(x, list):
            return [to_obj(i) for i in x]
        return x
    return to_obj(cfg)

def load_runtime_config(overrides: Optional[Dict[str, Any]] = None) -> RuntimeConfig:
    """
    生成带默认值的运行期配置，并应用可选的 dict 覆盖。
    用法：
        RUNTIME = load_runtime_config({
            "io": {"RAW_IO_QD_READ": 48},
            "pinned": {"WEIGHT_PINNED_BYTES": 28<<30}
        })
    """
    cfg = RuntimeConfig()
    if overrides:
        _update_dataclass(cfg, overrides)
    return validate_runtime_config(cfg)

# 可选导出（便于 from config import *）
__all__ = [
    "RuntimeFlags", "WindowPrefetchConfig", "PinnedPoolConfig", "RegisteredPoolConfig",
    "IOConfig", "WarmupBootstrapConfig", "StreamingPolicyConfig", "SystemStabilityConfig",
    "MonitoringConfig", "RuntimeConfig", "load_runtime_config", "validate_runtime_config",
    "runtime_config_to_dict",
    # 你已有的类
    "LayerInfo", "KVCacheArgs", "ModelArgs"
]