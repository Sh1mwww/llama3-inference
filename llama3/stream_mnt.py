import torch
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

# 说明：
# - 设备常见仅支持两级优先级（-1=高，0=普通）。原文件里使用 priority=1 可能越界导致创建失败，
#   这里统一做“安全优先级折叠”，并提供有序关系：MHA>=KV(H2D) > FFN(H2D) > KV(D2H)。
# - 提供事件池：record_event_on / wait_event_on。事件只在完成后回收到池，不复用未完成事件。
# - 提供 on_stream(stream) 语法糖，避免默认流提交 kernel。

# ---------- 辅助：安全优先级映射 ----------
PRIO_HIGH = -1
PRIO_NORM = 0

def _safe_priority(requested: int) -> int:
    # 绝大多数设备仅支持 [-1, 0]，将请求值安全折叠
    return PRIO_HIGH if requested <= PRIO_HIGH else PRIO_NORM

def _make_stream(device: str, priority: int) -> Optional[torch.cuda.Stream]:
    try:
        return torch.cuda.Stream(device=device, priority=_safe_priority(priority))
    except Exception:
        return None


class _EventPool:
    def __init__(self, device: str):
        self.device = device
        self._free: list[torch.cuda.Event] = []
        self._pending: list[torch.cuda.Event] = []

    def _new_event(self) -> torch.cuda.Event:
        with torch.cuda.device(self.device):
            return torch.cuda.Event(enable_timing=False, blocking=False, interprocess=False)
    
    def acquire(self) -> torch.cuda.Event:
        if self._free:
            return self._free.pop()
        return self._new_event()
    
    def record_on(self, stream: torch.cuda.Stream) -> torch.cuda.Event:
        event = self.acquire()
        with torch.cuda.device(self.device):
            event.record(stream)
        self._pending.append(event)
        return event

    def release(self, event: torch.cuda.Event):
        if event.query():
            self._free.append(event)
        else:
            self._pending.append(event)

    def gc(self, limit: int = 64) -> int :
        """尝试将已完成的 pending 事件回收到 free；limit 控制每次检查数量"""
        if not self._pending:
            return 0
        count = 0
        remaining = []
        for event in self._pending[:limit]:
            if event.query():
                self._free.append(event)
                count += 1
            else:
                remaining.append(event)
        remaining.extend(self._pending[limit:])
        self._pending = remaining
        return count


@dataclass
class Streams:
    # 计算
    compute_mha: Optional[torch.cuda.Stream] = None
    compute_ffn: Optional[torch.cuda.Stream] = None
    # 权重传输
    weight_h2d_mha: Optional[torch.cuda.Stream] = None
    weight_h2d_ffn: Optional[torch.cuda.Stream] = None
    # KV 传输
    kv_h2d: Optional[torch.cuda.Stream] = None
    kv_d2h: Optional[torch.cuda.Stream] = None
    # 事件池
    _event_pool: Optional[_EventPool] = None
    # ---- 向后兼容（旧字段别名）----
    # 旧代码可能使用 weight_h2d / weight_compute
    weight_h2d: Optional[torch.cuda.Stream] = None
    weight_compute: Optional[torch.cuda.Stream] = None

    def __post_init__(self):
        # 为旧字段提供别名：默认映射到 MHA 主路径
        if self.weight_h2d is None:
            self.weight_h2d = self.weight_h2d_mha
        if self.weight_compute is None:
            self.weight_compute = self.compute_mha
            
    def wait_weight_ready_on_current(self, device: Optional[str] = None):
        # ★ 修正：进入 device 上下文再取 current_stream
        dev = device if device is not None else f"cuda:{torch.cuda.current_device()}"
        if self.weight_h2d_mha is None and self.weight_h2d_ffn is None and self.weight_h2d is None:
            return
        with torch.cuda.device(dev):
            cur = torch.cuda.current_stream()
            for s in (self.weight_h2d_mha, self.weight_h2d_ffn, self.weight_h2d):
                if s is not None:
                    cur.wait_stream(s)
                
                
# 设备缓存
_streams_cache: Dict[str, Streams] = {}
_event_pools: Dict[str, _EventPool] = {}

def _get_event_pool(device: str) -> _EventPool:
    pool = _event_pools.get(device)
    if pool is None:
        pool = _EventPool(device)
        _event_pools[device] = pool
    return pool


def _normalize_device(dev_in) -> tuple[bool, Optional[str]]:
    """
    返回 (is_cuda, device_str)。支持输入：
      - "cuda" / "cuda:0" / "cuda:N"
      - int 索引（0,1,...）
      - torch.device("cuda:0")
    非 CUDA 则返回 (False, None)。
    """
    import torch
    # 无 CUDA 可用
    if not torch.cuda.is_available():
        return (False, None)
    # torch.device
    try:
        import torch as _t
        if isinstance(dev_in, _t.device):
            if dev_in.type != "cuda":
                return (False, None)
            return (True, str(dev_in))
    except Exception:
        pass
    # int 索引
    if isinstance(dev_in, int):
        return (True, f"cuda:{dev_in}")
    # 字符串
    if isinstance(dev_in, str):
        if dev_in.startswith("cuda"):
            return (True, dev_in)
        return (False, None)
    # 其他类型一律视为非 CUDA
    return (False, None)

def get_streams(device: str) -> Streams:
    """
    获取（并缓存）设备上的 6 条流与事件池。
    - 参数 runtime 可传入 config.load_runtime_config() 的返回，用于未来扩展基于配置的优先级细化；
      当前实现采用“安全两级折叠”的优先级。
    - 若传入非 CUDA 设备，返回空 Streams（全部为 None）。
    """
    
    global _streams_cache
    if device in _streams_cache:
        return _streams_cache[device]
    
    is_cuda, dev_str = _normalize_device(device)
    if not is_cuda:
        streams = Streams()
    else:
        # 优先级关系：compute_mha/weight_h2d_mha/kv_h2d 用高，其余用普通；kv_d2h 为最低（折叠为普通）
        try:
            streams = Streams(
                compute_mha    = _make_stream(device, PRIO_HIGH),
                compute_ffn    = _make_stream(device, PRIO_NORM),
                weight_h2d_mha = _make_stream(device, PRIO_HIGH),
                weight_h2d_ffn = _make_stream(device, PRIO_NORM),
                kv_h2d        = _make_stream(device, PRIO_HIGH),
                kv_d2h        = _make_stream(device, PRIO_NORM),
                _event_pool   = _get_event_pool(device)
            )
        except Exception:
            streams = Streams()
    
    _streams_cache[str(device)] = streams
    return streams


def clear_streams_cache():
    global _streams_cache , _event_pools
    _streams_cache.clear()
    _event_pools.clear()
    
# ---------- 事件工具（供各层/WSM 使用） ----------
def record_event_on(store: Dict[Any, torch.cuda.Event],
                    key : Any,
                    stream: torch.cuda.Stream,
                    device: Optional[str] = None) -> Optional[torch.cuda.Event]:
    """
    在指定 stream 上记录事件，并保存到 store[key]。
    - device 未提供时，使用当前设备。
    - 事件记录后加入 pending 集，待 gc() 完成再回收。
    """
    dev = device if device is not None else f"cuda:{torch.cuda.current_device()}"
    pool = _get_event_pool(dev)
    evt = pool.record_on(stream)
    if store is not None:
        if not isinstance(store, dict):
            raise ValueError("store must be a dict or None")
        store[key] = evt
    return evt

def wait_event_on(stream: torch.cuda.Stream,
                  evt: Optional[torch.cuda.Event]):
    """在 stream 上等待 evt；若 evt 为 None 则忽略。"""
    if evt is None:
        return
    stream.wait_event(evt)
    
# ---------- 默认流禁用：提供显式 on_stream 语法糖 ----------
class on_stream:
    """
    用法：
        s = get_streams("cuda:0")
        with on_stream(s.compute_mha):
            # 这里提交的 CUDA kernel/拷贝不会落到默认流
            ...
    """
    def __init__(self, stream: Optional[torch.cuda.Stream], device: Optional[str] = None):
        self.stream = stream
        self.device = device
        self._ctx_stream = None
        self._ctx_device = None

    def __enter__(self):
        if self.stream is None:
            return self
        # 绑定目标设备 + 目标流
        self._ctx_device = torch.cuda.device(self.device) if self.device is not None else None
        if self._ctx_device is not None:
            self._ctx_device.__enter__()
        self._ctx_stream = torch.cuda.stream(self.stream)
        self._ctx_stream.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._ctx_stream is not None:
            self._ctx_stream.__exit__(exc_type, exc_val, exc_tb)
        if self._ctx_device is not None:
            self._ctx_device.__exit__(exc_type, exc_val, exc_tb)