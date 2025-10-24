import math
from typing import Optional, List, Dict
import torch, torch.nn as nn, torch.nn.functional as F
import threading
import logging
from contextlib import contextmanager
import time

# 配置日志
logger = logging.getLogger(__name__)

# NVTX profiling support
try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False
    # Fallback no-op functions
    class nvtx:
        @staticmethod
        def range_push(name): pass
        @staticmethod
        def range_pop(): pass

from .config import ModelArgs
from .kv_offload import KVOffloader, BLOCK
from .global_state_tracker import GlobalStateTracker, get_global_tracker, init_global_tracker

# ---------- Stub Parameter Helper ----------
def make_stub_linear(in_features, out_features, bias=False, dtype=torch.bfloat16, device="cpu"):
    """创建一个空骨架 nn.Linear，权重为 0-size stub，避免内存分配"""
    # 先用 meta device 创建 Linear（避免初始化）
    with torch.device("meta"):
        linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    # 替换权重为 CPU 上的 0-size stub
    stub_weight = torch.nn.Parameter(
        torch.empty(0, dtype=dtype, device=device),
        requires_grad=False
    )
    stub_weight._shape_hint = (out_features, in_features)  # 保留形状信息
    linear.weight = stub_weight

    if bias:
        stub_bias = torch.nn.Parameter(
            torch.empty(0, dtype=dtype, device=device),
            requires_grad=False
        )
        stub_bias._shape_hint = (out_features,)
        linear.bias = stub_bias

    return linear
# ---------- Enhanced timing util ----------
class PerformanceTracker:
    def __init__(self):
        self.stats = {
            "weights_hbm_us": 0,
            "kv_fetch_us": 0, 
            "attn_us": 0,
            "ffn_us": 0,
            "total_forward_us": 0,
            "memory_alloc_us": 0,
        }
        self.layer_stats = {}  # per-layer statistics
        self.lock = threading.Lock()
    
    def reset(self):
        with self.lock:
            for key in self.stats:
                self.stats[key] = 0
            self.layer_stats.clear()
    
    def get_stats(self) -> Dict:
        with self.lock:
            return {
                "global": self.stats.copy(),
                "per_layer": self.layer_stats.copy()
            }
    
    def add_layer_stat(self, layer_id: int, stat_name: str, value: float):
        with self.lock:
            if layer_id not in self.layer_stats:
                self.layer_stats[layer_id] = {}
            if stat_name not in self.layer_stats[layer_id]:
                self.layer_stats[layer_id][stat_name] = 0
            self.layer_stats[layer_id][stat_name] += value

PERF_TRACKER = PerformanceTracker()

@contextmanager
def cuda_timer(key: str, layer_id: Optional[int] = None):
    if not torch.cuda.is_available():
        yield; return
    
    start_event = None
    end_event = None
    cuda_error_occurred = False
    
    try:
        # Check CUDA context health before creating events
        try:
            torch.cuda.current_device()
        except RuntimeError:
            logger.warning(f"CUDA context unhealthy for {key}, skipping timing")
            yield
            return
            
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        yield
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA OOM in timer for {key}: {e}")
        cuda_error_occurred = True
        torch.cuda.empty_cache()
        raise
    except RuntimeError as e:
        if "CUDA" in str(e):
            logger.error(f"CUDA error in timer for {key}: {e}")
            cuda_error_occurred = True
            raise
        else:
            logger.error(f"Runtime error in timer for {key}: {e}")
            raise
    finally:
        # Only attempt cleanup if no CUDA error occurred and events were created
        if not cuda_error_occurred and start_event is not None and end_event is not None:
            try:
                # Check if CUDA context is still valid
                torch.cuda.current_device()
                
                end_event.record()
                torch.cuda.synchronize()
                
                elapsed_us = int(start_event.elapsed_time(end_event) * 1000)

                with PERF_TRACKER.lock:
                    PERF_TRACKER.stats[key] += elapsed_us
                    if layer_id is not None:
                        # 直接更新，避免嵌套锁
                        if layer_id not in PERF_TRACKER.layer_stats:
                            PERF_TRACKER.layer_stats[layer_id] = {}
                        if key not in PERF_TRACKER.layer_stats[layer_id]:
                            PERF_TRACKER.layer_stats[layer_id][key] = 0
                        PERF_TRACKER.layer_stats[layer_id][key] += elapsed_us
            except Exception as e:
                logger.warning(f"Error in cuda_timer cleanup for {key}: {e}")
                # Don't re-raise exceptions in cleanup

def set_weight_manager(manager):
    global WEIGHT_MANAGER
    WEIGHT_MANAGER = manager

def get_weight_manager(device: str):
    global WEIGHT_MANAGER
    return WEIGHT_MANAGER

# class RMSNorm(nn.Module):
#     def __init__(self, dim: int, eps: float = 1e-6):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(dim))
#         self.eps = eps
    
#     def forward(self, x: torch.Tensor):
#         with cuda_timer("memory_alloc_us"):
#             norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
#         # 防呆：把權重的 device / dtype 對齊輸入
#         w = self.weight
#         if(w.device != x.device):
#             w = w.to(device=x.device, dtype=x.dtype)
#         return w * (x * norm)
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, *, dtype=torch.bfloat16, device=None, requires_grad=False):
        super().__init__()
        self.eps = float(eps)
        # 权重直接用目标 dtype/device 创建，推理默认不需要梯度
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=device), requires_grad=requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 计算用 fp32 更稳定
        y = x.to(torch.float32)
        # 用乘法代替 pow，少一次 kernel
        inv_rms = torch.rsqrt(y.mul(y).mean(dim=-1, keepdim=True).add_(self.eps))
        y = y.mul(inv_rms)                 # fp32

        # 2) 输出严格回到输入 dtype（例如 bfloat16）
        out = y.to(dtype=x.dtype)          # 与下游 Linear.weight 的 dtype 一致

        # 3) 仅在必要时把权重对齐到输入的 device/dtype（尽量避免每步 .to）
        w = self.weight
        if w.device != x.device:
            w = w.to(device=x.device, non_blocking=True)
        if w.dtype != x.dtype:
            w = w.to(dtype=x.dtype)

        return out * w


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0
    theta_i = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim)).to(device)
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, theta_i)
    return torch.polar(torch.ones_like(freqs), freqs)

# def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor) -> torch.Tensor:
#     b, l, h, d = x.shape
#     x_ = x.float().reshape(b, l, h, d // 2, 2)
#     x_complex = torch.view_as_complex(x_)
#     freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
#     out = torch.view_as_real(x_complex * freqs_complex)
#     return out.reshape(b, l, h, d).type_as(x)
def apply_rotary_embeddings(x: torch.Tensor,
                            freqs_complex: torch.Tensor,
                            start_pos: int = 0) -> torch.Tensor:
    """
    x: (B, L, H, D)
    freqs_complex: 支持以下任一形状：
        - (L, D/2)                           # 最常见
        - (1, L, 1, D/2)                     # 已经被扩成4D
        - (L, 1, D/2) / (1, L, D/2) / (L, D/2, 1)   # 其他带1维的变体
    行为：
        1) 先把 freqs 规范成 (L, D/2)
        2) 切片到 [start_pos : start_pos+Lx] 其中 Lx = x.shape[1]
        3) 广播乘到 x 的前一半维度（视作复数）
    """
    import torch
    B, Lx, H, D = x.shape
    if D % 2 != 0:
        raise RuntimeError(f"apply_rotary_embeddings: head_dim {D} must be even")

    # ---- 规范 freqs 到 (L, D/2) ----
    fc = freqs_complex
    # 常见输入：(1, L, 1, D/2)
    if fc.dim() == 4 and fc.size(0) == 1 and fc.size(2) == 1:
        fc = fc.squeeze(0).squeeze(1) if fc.size(1) == 1 else fc.squeeze(0).squeeze(2)  # -> (L, D/2)
    # 其他带 1 的三维
    if fc.dim() == 3:
        # 尝试去掉单例维，优先去掉中间的 1 维
        if fc.size(1) == 1:
            fc = fc.squeeze(1)  # -> (L, D/2)
        elif fc.size(0) == 1:
            fc = fc.squeeze(0)  # -> (L, D/2)
        elif fc.size(2) == 1:
            fc = fc.squeeze(2)  # -> (L, D/2)
    # 两维就不用动
    if fc.dim() != 2:
        # 最后兜底：如果第0维正好等于 L 或 Lx，就 reshape 成 (L, D/2)
        if fc.size(0) in (Lx, fc.size(0)) and fc.numel() % fc.size(0) == 0:
            fc = fc.reshape(fc.size(0), -1)
        else:
            raise RuntimeError(f"apply_rotary_embeddings: unexpected freqs_complex shape {freqs_complex.shape}, "
                               f"cannot normalize to (L, D/2)")

    # ---- 切片到当前窗口 [start_pos : start_pos+Lx] ----
    if fc.size(0) < start_pos + Lx:
        raise RuntimeError(f"apply_rotary_embeddings: freqs length {fc.size(0)} < needed {start_pos+Lx}")
    fc = fc[start_pos: start_pos + Lx, :]   # (Lx, D/2)

    # ---- 设备与 dtype 对齐 ----
    # x_被转为 float 做复数视图，最终再转换回 x.dtype
    x_ = x.to(torch.float32).reshape(B, Lx, H, D // 2, 2)       # (B,L,H,D/2,2)
    x_complex = torch.view_as_complex(x_)                       # (B,L,H,D/2)
    fc = fc.to(dtype=x_complex.dtype, device=x.device)          # (Lx,D/2)

    # ---- 广播到 (1,Lx,1,D/2) 与 x_complex 相乘 ----
    fc = fc.unsqueeze(0).unsqueeze(2)                           # (1,Lx,1,D/2)
    out = torch.view_as_real(x_complex * fc)                    # (B,L,H,D/2,2)
    out = out.reshape(B, Lx, H, D).to(dtype=x.dtype)
    return out


# def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
#     if n_rep == 1: 
#         return x
#     b, t, h, d = x.shape
#     return x[:, :, :, None, :].expand(b, t, h, n_rep, d).contiguous().view(b, t, h * n_rep, d)

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.kv_elapsed_time = 0.0
        self.attn_time = 0.0

        self.topk_blk = args.topk_blk
        self.device = args.device
        self.is_cuda = str(self.device).startswith("cuda") and torch.cuda.is_available()

        # Linear权重初始化 - 使用 stub 避免大内存分配
        use_stub = getattr(args, "use_stub_params", False)
        if use_stub:
            # SSD streaming 模式：使用 0-size stub
            self.wq = make_stub_linear(args.dim, self.n_heads_q * self.head_dim, bias=False)
            self.wk = make_stub_linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
            self.wv = make_stub_linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
            self.wo = make_stub_linear(args.n_heads * self.head_dim, args.dim, bias=False)
        else:
            # 传统模式：正常初始化
            _dev = getattr(args, "param_init_device", None)
            kw = ({"device": _dev} if _dev is not None else {})
            self.wq = nn.Linear(args.dim, self.n_heads_q * self.head_dim, bias=False, **kw)
            self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False, **kw)
            self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False, **kw)
            self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False, **kw)
        self.block_sz = BLOCK

        self.apply_causal_mask = True

        streams = None
        try:
            import llama3.stream_mnt as stream_mnt
            streams = stream_mnt.get_streams(args.device)
        except Exception:
            pass  # 回退到内部流创建

        self.streams = streams
        self.compute_stream = getattr(streams, "compute_mha", None) or getattr(streams, "weight_compute", None)
        self.weight_h2d_stream = getattr(streams, "weight_h2d_mha", None) or getattr(streams, "weight_h2d", None)
        self.weight_manager = None

        # 组级预取支持标记
        self.supports_group_prefetch = False
        
        self.offloader = None   
        self.enable_profiling = False
        
        self.offloader = KVOffloader(
            layers=args.n_layers,
            heads=self.n_kv_heads,
            dim=self.head_dim,
            max_seq=args.max_seq_len,
            max_batch=args.max_batch_size,
            device=args.device,
            dtype_bytes=2,  # float16
            streams=streams
        )
        
        self.layer_id = -1
        self.attention_history = []  # 用于分析注意力模式
        self.qkv_buffer = None
        self.scores_buffer = None
        # self.streams = streams  # 保存streams引用用于compute
        
    def _get_causal_mask(self, t: int, device):
        # 若缓存不存在 / 太小 / 设备不一致，就重建
        cm = getattr(self, "_causal_mask", None)
        if (cm is None) or (cm.device != device) or (cm.size(-1) < t):
            cm = torch.ones((1, 1, t, t), dtype=torch.bool, device=device).triu(1)
            try:
                self.register_buffer("_causal_mask", cm, persistent=False)
            except Exception:
                self._causal_mask = cm
        return cm[..., :t, :t]    
    
    @staticmethod
    def _safe_item_sum_1d(t: torch.Tensor, s: int, e: int) -> float:
        # t 可能是 meta；也可能为空/None；都给出 0.0
        if t is None:
            return 0.0
        if getattr(t, "is_meta", False) or (hasattr(t, "device") and t.device.type == "meta"):
            return 0.0
        if s >= t.size(0):
            return 0.0
        # 用 detach().cpu().item() 保证不是 CUDA/Event 等特殊张量
        return float(t[s:e].sum().detach().cpu().item())
        
    
    def _get_modules_dict(self):
        return {
            "wq": self.wq,
            "wk": self.wk, 
            "wv": self.wv,
            "wo": self.wo
        }
    
    def _sync_event_safely(self, evt: torch.cuda.Event, timeout_ms: int = 5000):
        if not getattr(self, 'enable_profiling', False):
            return
        # 輕量輪詢 + timeout：避免某些環境下事件沒有被正確記錄導致永久等待
        import time
        start = time.time()
        while not evt.query():
            if (time.time() - start) * 1000.0 > timeout_ms:
                # 不丟擲異常，打印警告並退出等待；避免測試被卡死
                print(f"[WARN][L{self.layer_id}] event sync timeout after {timeout_ms} ms")
                return
            time.sleep(0.001)
        # 事件已完成，無需再同步
        return
    
    def _ensure_weights_cuda(self):
        wm = getattr(self, "weight_manager", None)
        if wm is None:
            # 没有 weight_manager：权重应该已经在正确的设备上
            # No weight_manager: weights should already be on the correct device
            # Forward 不做设备迁移，只由 WSM 管理
            # Forward does NOT do device migration, only managed by WSM
            return

        # 组级预取模式：优先使用新的组级API
        if getattr(wm, "grouped_mode", False) and hasattr(wm, "ensure_group_on_gpu"):
            try:
                # 确保attn组在GPU上（阻塞式）
                wm.ensure_group_on_gpu(self.layer_id, "attn")
                # 异步预取下一层的attn（与当前MHA计算重叠）
                if hasattr(wm, "n_layers") and self.layer_id + 1 < wm.n_layers:
                    if hasattr(wm, "prefetch_group_async"):
                        wm.prefetch_group_async(self.layer_id + 1, "attn")
                self.supports_group_prefetch = True
                return
            except (AttributeError, NotImplementedError, KeyError) as e:
                # 组级API不可用或实现不完整，回退到层级API
                logger.warning(f"Layer {self.layer_id} SelfAttention: group-level prefetch failed ({e}), fallback to layer-level")
                self.supports_group_prefetch = False

        # 回退到原有的层级预取
        modules = self._get_modules_dict()
        wm.ensure_weights_cuda(self.layer_id, modules, priority=True)

    def _allocate_buffers(self, batch_size: int, seq_len: int, max_kv_len: int):
        if (self.qkv_buffer is None or 
            self.qkv_buffer[0].size(0) < batch_size or 
            self.qkv_buffer[0].size(1) < seq_len):
            
            with cuda_timer("memory_alloc_us", self.layer_id):
                MAX_BUFFER_ELEMENTS = 50_000_000  # 约100MB for float16
                q_elements = batch_size * seq_len * self.n_heads_q * self.head_dim
                kv_elements = batch_size * seq_len * self.n_kv_heads * self.head_dim
                scores_elements = batch_size * self.n_heads_q * seq_len * max_kv_len
                if scores_elements > MAX_BUFFER_ELEMENTS:
                    logger.warning(f"Large attention buffer requested ({scores_elements} elements), limiting to prevent OOM")
                    safe_kv_len = MAX_BUFFER_ELEMENTS // (batch_size * self.n_heads_q * seq_len)
                    max_kv_len = min(max_kv_len, max(safe_kv_len, 1024))  # 至少保留1024
                    scores_elements = batch_size * self.n_heads_q * seq_len * max_kv_len
                    self.use_chunked_attention = True
                else:
                    self.use_chunked_attention = False
                
                try:
                    from .memory_manager import GlobalMemoryManager
                    memory_manager = GlobalMemoryManager.get_instance()
                    if memory_manager:
                        total_bytes = (q_elements + 2 * kv_elements + scores_elements) * 2  # float16
                        if not memory_manager.can_allocate(total_bytes):
                            # 尝试清理内存
                            if hasattr(self, 'qkv_buffer') and self.qkv_buffer:
                                del self.qkv_buffer
                            if hasattr(self, 'scores_buffer') and self.scores_buffer:
                                del self.scores_buffer
                            torch.cuda.empty_cache()
                            
                            if not memory_manager.can_allocate(total_bytes):
                                raise RuntimeError(f"Insufficient GPU memory: need {total_bytes/(1024**3):.2f}GB")
                except ImportError:
                    pass  # memory_manager not available
                
                try:
                    # QKV buffer
                    q_shape = (batch_size, seq_len, self.n_heads_q, self.head_dim)
                    kv_shape = (batch_size, seq_len, self.n_kv_heads, self.head_dim)
                    
                    self.qkv_buffer = (
                        torch.empty(q_shape, dtype=torch.float16, device=self.device),
                        torch.empty(kv_shape, dtype=torch.float16, device=self.device),
                        torch.empty(kv_shape, dtype=torch.float16, device=self.device)
                    )
                    # Attention scores buffer
                    scores_shape = (batch_size, self.n_heads_q, seq_len, max_kv_len)
                    self.scores_buffer = torch.empty(scores_shape, dtype=torch.float16, device=self.device)
                    
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"GPU OOM during buffer allocation: batch={batch_size}, seq={seq_len}, kv_len={max_kv_len}")
                    torch.cuda.empty_cache()
                    raise RuntimeError(f"GPU OOM: Cannot allocate attention buffers. Try reducing batch_size (current: {batch_size}) or max sequence length.") from e
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        # Check CUDA context health
        if x.is_cuda:
            try:
                torch.cuda.current_device()
            except RuntimeError as e:
                logger.error(f"CUDA context error in attention forward: {e}")
                raise RuntimeError("CUDA context is corrupted") from e
        
        start_time = time.time()
        assert x.dim()==3, f"x dim={x.dim()}, shape={x.shape}"
        bsz, seqlen, _ = x.shape


        def _ensure_cpu_scalar_attr(mod, name: str):
            if hasattr(mod, name):
                t = getattr(mod, name)
                if isinstance(t, torch.Tensor) and (getattr(t, "is_meta", False)
                                                    or (hasattr(t, "device") and t.device.type == "meta")):
                    setattr(mod, name, torch.zeros((), dtype=torch.int64, device="cpu"))
            else:
                # 首次建立
                setattr(mod, name, torch.zeros((), dtype=torch.int64, device="cpu"))

        _ensure_cpu_scalar_attr(self, "attn_us")
        _ensure_cpu_scalar_attr(self, "total_forward_us")

        print(f"[ATTN] Layer {self.layer_id} forward starting...")

        # ============================================================
        # 1) 确保 attn 组在 GPU，并等待该组 H2D 完成（在 compute_mha 流上等待）
        # ============================================================
        wm = getattr(self, "weight_manager", None)
        in_use = False
        if wm and hasattr(wm, "_make_group_in_use"):
            wm._make_group_in_use(self.layer_id, "attn")
            in_use = True
        try:
            if wm is not None:
                # 确保 attn 组在 GPU（阻塞式，直到权重加载完成）
                if hasattr(wm, "ensure_group_on_gpu"):
                    wm.ensure_group_on_gpu(self.layer_id, "attn")
                else:
                    # 回退到层级加载
                    modules = self._get_modules_dict()
                    wm.ensure_weights_cuda(self.layer_id, modules, priority=True)

                # 在 compute stream 上等待 attn 组 H2D 传输完成（禁止降级到 CPU）
                if hasattr(wm, "wait_group_ready"):
                    wm.wait_group_ready(self.layer_id, "attn", compute_stream=self.compute_stream)
        finally:
            if in_use and hasattr(wm, "_unmark_group_in_use"):
                wm._unmake_group_in_use(self.layer_id, "attn")

        print(f"[ATTN] Layer {self.layer_id} weights ensured and ready")

        # ============================================================
        # 2) 验证：所有权重必须已经在 CUDA 上（由 WSM 负责，forward 不做设备迁移）
        # Validation: All weights must already be on CUDA (managed by WSM, forward does NOT do device migration)
        # ============================================================
        target_device = x.device

        # 确保 target_device 是 CUDA（不接受 CPU）
        if not str(target_device).startswith("cuda"):
            raise RuntimeError(f"Layer {self.layer_id} SelfAttention: input x is on {target_device}, but only CUDA is supported")

        for mod in (self.wq, self.wk, self.wv, self.wo):
            # 权重必须在 CUDA，不允许 meta 或 CPU
            if str(mod.weight.device) == "meta" or getattr(mod.weight, "is_meta", False):
                raise RuntimeError(f"Layer {self.layer_id} SelfAttention: {mod.__class__.__name__}.weight still on meta device after ensure_group_on_gpu()")

            if not str(mod.weight.device).startswith("cuda"):
                raise RuntimeError(f"Layer {self.layer_id} SelfAttention: {mod.__class__.__name__}.weight on {mod.weight.device}, must be on CUDA (managed by WSM)")

            # 验证设备一致性（不做迁移，只检查）
            # Validate device consistency (no migration, only check)
            if mod.weight.device != target_device:
                raise RuntimeError(f"Layer {self.layer_id} SelfAttention: {mod.__class__.__name__}.weight on {mod.weight.device} but input on {target_device}. "
                                 "Device mismatch must be fixed by WSM, not by forward().")

        print(f"[ATTN] Layer {self.layer_id} device validation passed (all on CUDA)")

        # 更新全局状态跟踪器
        tracker = get_global_tracker()
        if tracker:
            batch_idx = tracker.current_batch
        else:
            batch_idx = 0

        print(f"[ATTN] Layer {self.layer_id} starting computation...")

        # 预期形状
        exp_q = (self.n_heads_q * self.head_dim, x.size(-1))
        exp_kv = (self.n_kv_heads * self.head_dim, x.size(-1))

        def _shape(t): return tuple(t.shape)

        if _shape(self.wq.weight) != exp_q:
            raise RuntimeError(f"[Q shape] { _shape(self.wq.weight) } != {exp_q} "
                            f"(likely manifest q/k/v mapping issue)")
        if _shape(self.wk.weight) != exp_kv:
            raise RuntimeError(f"[K shape] { _shape(self.wk.weight) } != {exp_kv} "
                            f"(likely manifest q/k/v mapping issue)")
        if _shape(self.wv.weight) != exp_kv:
            raise RuntimeError(f"[V shape] { _shape(self.wv.weight) } != {exp_kv} "
                            f"(likely manifest q/k/v mapping issue)")    

        # QKV投影 - 使用专门的compute stream
        # compute_stream = self.streams.weight_compute if self.streams else None
        compute_stream = self.compute_stream
        if compute_stream:
            with torch.cuda.stream(compute_stream):
                with cuda_timer("attn_us", self.layer_id):
                    # print("wq.weight.device =", self.wq.weight.device)
                    q = self.wq(x).view(bsz, seqlen, self.n_heads_q, self.head_dim)
                    k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                    v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

                    # 应用旋转位置编码
                    # q = apply_rotary_embeddings(q, freqs_complex)
                    # k = apply_rotary_embeddings(k, freqs_complex)
                    q = apply_rotary_embeddings(q, freqs_complex, start_pos=start_pos)
                    k = apply_rotary_embeddings(k, freqs_complex, start_pos=start_pos)

        else:
            # 回退到默认stream
            with cuda_timer("attn_us", self.layer_id):
                # print("wq.weight.device =", self.wq.weight.device)
                q = self.wq(x).view(bsz, seqlen, self.n_heads_q, self.head_dim)
                k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

                # 应用旋转位置编码
                # q = apply_rotary_embeddings(q, freqs_complex)
                # k = apply_rotary_embeddings(k, freqs_complex)
                q = apply_rotary_embeddings(q, freqs_complex, start_pos=start_pos)
                k = apply_rotary_embeddings(k, freqs_complex, start_pos=start_pos)
        
        # ------------------------- (A) 推入 KV 到 offloader -------------------------
        bsz, seqlen, n_heads, head_dim = k.shape

        # 对于每个token位置, 计算对应的block并push
        if getattr(self, "offloader", None) is not None:
            for seq_idx in range(seqlen):
                blk_idx   = (start_pos + seq_idx) // self.block_sz
                token_idx =  start_pos + seq_idx

                # 保持 (bsz, heads, dim) —— 切勿 squeeze，否则 heads 会被误当 batch
                k_curr = k[:, seq_idx, :, :]    # (bsz, n_kv_heads, head_dim)
                v_curr = v[:, seq_idx, :, :]

                # 将该 token 写入所属 block 的正确 token 槽位
                self.offloader.push(
                    layer=self.layer_id,
                    blk=blk_idx,
                    k=k_curr,
                    v=v_curr,
                    token_idx=token_idx,
                    batch_idx=batch_idx,
                )

        # ------------------------- (B) 选择并取回需要的 blocks -------------------------
        blk_idx = start_pos // self.block_sz 
        
        # 获取Top-K blocks (如果有 offloader)
        nvtx.range_push(f"layer_{self.layer_id}_kv_fetch")
        with cuda_timer("kv_fetch_us", self.layer_id):
            do_profile_gpu = bool(self.enable_profiling and x.is_cuda)
            if do_profile_gpu:
                fetch_evt_start = torch.cuda.Event(enable_timing=True)
                fetch_evt_end = torch.cuda.Event(enable_timing=True)
                fetch_evt_start.record()

            if getattr(self, "offloader", None) is not None:
                blocks = self.offloader.topk_blocks(self.layer_id, self.topk_blk, batch_idx=batch_idx)
                # 保证当前 block 在列表内
                if blk_idx not in blocks:
                    blocks = sorted(set(blocks + [blk_idx]))
                    # 简单截断到 topk，避免过度复杂的距离计算
                    if len(blocks) > self.topk_blk:
                        blocks = blocks[:self.topk_blk]
                else:
                    blocks = sorted(blocks)
                needed = torch.tensor(blocks, device=x.device, dtype=torch.long)
                k_full, v_full = self.offloader.fetch(
                    self.layer_id, needed, batch_idx=batch_idx, bsz=bsz
                )
            else:
                # 无 offloader：直接使用当前序列窗口（转为 (B,H,T,D)）
                k_full = k.transpose(1, 2).contiguous()
                v_full = v.transpose(1, 2).contiguous()
                blocks = [blk_idx]  # 为后续 update_importances 提供 blocks 列表

            if do_profile_gpu:
                fetch_evt_end.record()
                if not fetch_evt_end.query():
                    fetch_evt_end.synchronize()
                self.kv_elapsed_time = fetch_evt_start.elapsed_time(fetch_evt_end) * 1000
                PERF_TRACKER.add_layer_stat(self.layer_id, "kv_fetch_us", self.kv_elapsed_time)
            else:
                self.kv_elapsed_time = 0
        nvtx.range_pop()  # kv_fetch

        # ============================================================
        # 3) 确保 KV Cache（历史 K/V）在 CUDA（防止 q 在 CUDA、k_full 在 CPU 的 bmm 报错）
        # Ensure KV cache (historical K/V) is on CUDA (prevent bmm error when q is on CUDA but k_full on CPU)
        # ============================================================
        if k_full.device.type != "cuda":
            raise RuntimeError(f"Layer {self.layer_id} SelfAttention: k_full is on {k_full.device}, but q is on {q.device}. "
                             "This would cause 'mat2 is on cpu' error in bmm. KV cache must be on CUDA.")
        if v_full.device.type != "cuda":
            raise RuntimeError(f"Layer {self.layer_id} SelfAttention: v_full is on {v_full.device}, but q is on {q.device}. "
                             "This would cause 'mat2 is on cpu' error in bmm. KV cache must be on CUDA.")

        # 如果设备不一致（例如不同的 CUDA 设备），强制对齐到 q 的设备
        # If devices mismatch (e.g., different CUDA devices), force align to q's device
        if k_full.device != q.device:
            k_full = k_full.to(q.device, non_blocking=True)
        if v_full.device != q.device:
            v_full = v_full.to(q.device, non_blocking=True)

        #  batch
        if k_full.dim() == 3:
            # (seq_len, n_heads, head_dim) -> (1, n_heads, seq_len, head_dim)
            k_full = k_full.permute(1, 0, 2).unsqueeze(0)
            v_full = v_full.permute(1, 0, 2).unsqueeze(0)
        elif k_full.dim() == 4:
            # 检查是否已经是正确的格式 (bsz, n_heads, seq_len, head_dim)
            # 通过比较 dimension sizes：heads 应该比 head_dim 小
            if k_full.size(1) == self.n_kv_heads:
                # 已经是 (bsz, n_heads, seq_len, head_dim)，不需要 transpose
                pass
            else:
                # 假设是 (bsz, seq_len, n_heads, head_dim)，需要 transpose
                k_full = k_full.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
                v_full = v_full.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)

        # 确保k_full和v_full与q的batch维度一致
        if k_full.size(0) == 1 and bsz > 1:
            # 单batch的KV需要扩展到多batch
            k_full = k_full.expand(bsz, -1, -1, -1)
            v_full = v_full.expand(bsz, -1, -1, -1)

        k_full = k_full.to(q.dtype)
        v_full = v_full.to(q.dtype)
        # 重复KV头以匹配查询头数
        # if self.n_heads_q != self.n_kv_heads:
        if (self.n_heads_q != self.n_kv_heads) and (k_full.size(1) != self.n_heads_q):
            k_full = k_full.repeat_interleave(self.n_rep, dim=1)
            v_full = v_full.repeat_interleave(self.n_rep, dim=1)
        
        # 确保缓冲区足够大
        q = q.transpose(1, 2)  # (B, H, Tq, D)

        # Attention计算 - 使用compute stream
        nvtx.range_push(f"layer_{self.layer_id}_attention_compute")
        do_profile_gpu = bool(self.enable_profiling and x.is_cuda)
        if compute_stream:
            with torch.cuda.stream(compute_stream):
                with cuda_timer("attn_us", self.layer_id):
                    # mha
                    if do_profile_gpu:
                        attn_evt_start = torch.cuda.Event(enable_timing=True)
                        attn_evt_end = torch.cuda.Event(enable_timing=True)
                        attn_evt_start.record()

                    scores = torch.matmul(q, k_full.transpose(2, 3))
                    scores = scores / math.sqrt(self.head_dim)

                    # causal mask
                    if hasattr(self, 'apply_causal_mask') and self.apply_causal_mask:
                        seq_len_q = q.size(2)
                        seq_len_k = k_full.size(2)
                        mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=q.device), diagonal=1)
                        scores = scores.masked_fill(mask.bool(), float('-inf'))

                    # Softmax和输出计算
                    attn_weights = torch.softmax(scores, dim=-1)
                    out = torch.matmul(attn_weights, v_full)

                    if do_profile_gpu:
                        attn_evt_end.record()
                        if not attn_evt_end.query():
                            attn_evt_end.synchronize()
                        self.attn_time = attn_evt_start.elapsed_time(attn_evt_end) * 1000
                        PERF_TRACKER.add_layer_stat(self.layer_id, "attn_us", self.attn_time)
                    else:
                        self.attn_time = 0
        else:
            with cuda_timer("attn_us", self.layer_id):
                # mha
                if do_profile_gpu:
                    attn_evt_start = torch.cuda.Event(enable_timing=True)
                    attn_evt_end = torch.cuda.Event(enable_timing=True)
                    attn_evt_start.record()

                scores = torch.matmul(q, k_full.transpose(2, 3))
                scores = scores / math.sqrt(self.head_dim)

                # causal mask
                if hasattr(self, 'apply_causal_mask') and self.apply_causal_mask:
                    seq_len_q = q.size(2)
                    seq_len_k = k_full.size(2)
                    # mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=q.device), diagonal=1)
                    # # scores = scores.masked_fill(mask.bool(), float('-inf'))
                    
                    # mask = self._get_causal_mask(scores.size(-1), device=scores.device)
                    seq_len_q = q.size(2)
                    seq_len_k = k_full.size(2)
                    delta = max(0, seq_len_k - seq_len_q)
                    mask = torch.triu(torch.ones((seq_len_q, seq_len_k), dtype=torch.bool, device=q.device),
                                    diagonal=1 + delta)
                    scores = scores.masked_fill(mask, float('-inf'))
                    


                # Softmax和输出计算
                attn_weights = torch.softmax(scores, dim=-1)
                out = torch.matmul(attn_weights, v_full)

                if do_profile_gpu:
                    attn_evt_end.record()
                    if not attn_evt_end.query():
                        attn_evt_end.synchronize()
                    self.attn_time = attn_evt_start.elapsed_time(attn_evt_end) * 1000
                    PERF_TRACKER.add_layer_stat(self.layer_id, "attn_us", self.attn_time)
                else:
                    self.attn_time = 0
        nvtx.range_pop()  # attention_compute
        # out = out.transpose(1, 2).reshape(bsz, seqlen, -1)
        
        # stats = PERF_TRACKER.layer_stats.get(self.layer_id, {})
        # self.kv_elapsed_time = stats.get("kv_fetch_us", 0)
        # self.attn_time       = stats.get("attn_us",     0)
        
        # # Update block importances (only if offloader exists)
        # if getattr(self, "offloader", None) is not None:
        #     with torch.no_grad():
        #         token_imp = attn_weights.mean(dim=(0, 1, 2))  # (Tkv,)
        #         block_scores = []
        #         for i, _ in enumerate(blocks):
        #             s = i * self.block_sz
        #             e = min(s + self.block_sz, token_imp.size(0))
        #             score = self._safe_item_sum_1d(token_imp, s, e)
        #             block_scores.append(score)

        #         self.offloader.update_importances(self.layer_id, blocks, block_scores, batch_idx=batch_idx)

        # feat = self.n_heads_q * self.head_dim  # = dim
        # if out.dim() == 2:
        #     # 误展平成 [B, seqlen*dim] 的情形：还原成 [B, seqlen, dim]
        #     out = out.view(bsz, seqlen, self.n_heads_q, self.head_dim).reshape(bsz, seqlen, feat)
        # elif out.dim() == 3 and out.size(-1) != feat:
        #     # 例如被写成 view(bsz, -1) 合并了最后两维
        #     out = out.reshape(bsz, seqlen, feat)

        # # 输出投影 - 使用compute stream
        # if compute_stream:
        #     with torch.cuda.stream(compute_stream):
        #         result = self.wo(out)
        # else:
        #     result = self.wo(out)

        # total_time = (time.time() - start_time) * 1000000  # 转换为微秒
        # PERF_TRACKER.add_layer_stat(self.layer_id, "total_forward_us", total_time)
        # print(f"[ATTN] Layer {self.layer_id} computation done")
        # return result
        
        out = out.transpose(1, 2).contiguous()

        # --- 统计信息（若有） ---
        stats = PERF_TRACKER.layer_stats.get(self.layer_id, {})
        self.kv_elapsed_time = stats.get("kv_fetch_us", 0)
        self.attn_time       = stats.get("attn_us",     0)


        feat = self.n_heads_q * self.head_dim  # = dim
        B, Tq = bsz, seqlen
        w = self.wo.weight
        
        # 重要度（attn_weights 已实体化时才计算；blocks 缺失则按 block_sz 推导）
        if getattr(self, "offloader", None) is not None:
            with torch.no_grad():
                if ('attn_weights' in locals()
                    and attn_weights is not None
                    and not getattr(attn_weights, "is_meta", False)):
                    # [B, Hq, Tq, Tkv] -> (Tkv,)
                    token_imp = attn_weights.detach().mean(dim=(0, 1, 2))
                    if 'blocks' not in locals() or blocks is None:
                        Tkv = int(token_imp.size(0))
                        nblk = (Tkv + self.block_sz - 1) // self.block_sz
                        blocks = [(i*self.block_sz, min((i+1)*self.block_sz, Tkv)) for i in range(nblk)]
                    block_scores = []
                    for i, _ in enumerate(blocks):
                        s = i * self.block_sz
                        e = min(s + self.block_sz, token_imp.size(0))
                        if hasattr(self, "_safe_item_sum_1d"):
                            score = self._safe_item_sum_1d(token_imp, s, e)  # 建议已定义为 @staticmethod
                        else:
                            score = float(token_imp[s:e].sum().detach().cpu().item())
                        block_scores.append(score)
                    self.offloader.update_importances(self.layer_id, blocks, block_scores,
                                                    batch_idx=locals().get("batch_idx", 0))
                    
                    
        # --- 形状护栏：确保送入 wo 前为 [B, seqlen, dim] ---

        def _as_btD_take_last(_out, B, Tq, feat):
            """把 out 统一成 [B, Tq, feat]；若含累计 T_total，则仅取最后 Tq。"""
            ne = _out.numel()
            if _out.dim() == 3:
                # e.g. [B, T?, D?]：矫正最后一维，再裁剪 Tq
                T_found = _out.size(1)
                if _out.size(-1) != feat:
                    # 用元素数反推 T_found
                    T_found = ne // (B * feat)
                    _out = _out.reshape(B, T_found, feat)
                return _out[:, -Tq:, :]  # decode 场景：只取最后 Tq
            elif _out.dim() == 2:
                # e.g. [B, T_total*feat] 或 [B, feat]
                C = _out.size(1)
                if C == feat:
                    return _out.view(B, 1, feat)[:, -Tq:, :]
                if C % feat == 0:
                    T_found = C // feat
                    return _out.view(B, T_found, feat)[:, -Tq:, :]
                # 兜底：按元素数反推
                T_found = ne // (B * feat)
                return _out.view(B, T_found, feat)[:, -Tq:, :]
            else:
                # 其它异常：用元素数反推
                T_found = ne // (B * feat)
                return _out.reshape(B, T_found, feat)[:, -Tq:, :]

        # 如果不是期望形状/维度，做一次统一矫正
        if out.dim() != 3 or out.size(0) != B or out.size(1) != Tq or out.size(2) != feat:
            out = _as_btD_take_last(out, B, Tq, feat)
        else:
            # 标准路径：已是 [B, Tq, *]，但最后一维可能不是 feat
            if out.size(-1) != feat:
                # 先尝试按元素数恢复再裁剪
                T_found = out.numel() // (B * feat)
                out = out.reshape(B, T_found, feat)[:, -Tq:, :]
                
        assert out.shape == (B, Tq, feat), f"out={out.shape}, B={B}, Tq={Tq}, feat={feat}"
        # --- 线性层稳定计算：统一 2D → Linear → 3D ---
        # 对齐 dtype / device 到 wo.weight
        out2d = out.reshape(-1, feat)
        w = self.wo.weight
        if getattr(out2d, "is_meta", False) or (hasattr(out2d, "device") and out2d.device.type == "meta"):
            out2d = torch.zeros((B*Tq, feat), dtype=w.dtype, device=w.device)
        else:
            if out2d.dtype != w.dtype:
                out2d = out2d.to(w.dtype)
            if out2d.device != w.device:
                out2d = out2d.to(w.device, non_blocking=True)

        assert out.shape == (bsz, seqlen, feat), f"out={out.shape}, bsz={bsz}, seqlen={seqlen}, feat={feat}"

        # 输出投影（可用 compute_stream）
        if compute_stream:
            with torch.cuda.stream(compute_stream):
                res2d = self.wo(out2d)
        else:
            res2d = self.wo(out2d)

        result = res2d.view(B, Tq, -1).contiguous()

        # --- 统计收尾 ---
        total_time = (time.time() - start_time) * 1e6  # μs
        PERF_TRACKER.add_layer_stat(self.layer_id, "total_forward_us", total_time)
        # print(f"[ATTN] Layer {self.layer_id} computation done")

        # IN_USE 标记会在下一轮evict时自动清理，这里不需要手动unmark
        return result

# ---------- Optimized FeedForward ----------
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = int(4 * args.dim * 2 / 3)
        if args.ffn_dim_multiplier:
            hidden_dim = int(hidden_dim * args.ffn_dim_multiplier)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        # 使用 stub 避免大内存分配
        use_stub = getattr(args, "use_stub_params", False)
        if use_stub:
            # SSD streaming 模式：使用 0-size stub
            self.w1 = make_stub_linear(args.dim, hidden_dim, bias=False)
            self.w2 = make_stub_linear(hidden_dim, args.dim, bias=False)
            self.w3 = make_stub_linear(args.dim, hidden_dim, bias=False)
        else:
            # 传统模式：正常初始化
            _dev = getattr(args, "param_init_device", None)
            kw = ({"device": _dev} if _dev is not None else {})
            self.w1 = nn.Linear(args.dim, hidden_dim, bias=False, **kw)
            self.w2 = nn.Linear(hidden_dim, args.dim, bias=False, **kw)
            self.w3 = nn.Linear(args.dim, hidden_dim, bias=False, **kw)

        self.device = args.device
        self.layer_id = -1

        self.activation_buffer = None

        # 获取streams引用
        streams = None
        try:
            import llama3.stream_mnt as stream_mnt
            streams = stream_mnt.get_streams(args.device)
        except Exception:
            pass
        self.streams = streams
       
    
    def _get_modules_dict(self):
        return {
            "w1": self.w1,
            "w2": self.w2,
            "w3": self.w3
        }
    
    def _ensure_weights_cuda(self):
        wm = self.weight_manager
        modules = self._get_modules_dict()

        if wm is None:
            # 没有 weight_manager：权重应该已经在正确的设备上
            # No weight_manager: weights should already be on the correct device
            # Forward 不做设备迁移，只由 WSM 管理
            # Forward does NOT do device migration, only managed by WSM
            return

        # 组级预取模式：优先使用新的组级API
        if getattr(wm, "grouped_mode", False) and hasattr(wm, "ensure_group_on_gpu"):
            try:
                # 确保ffn组在GPU上（阻塞式）
                wm.ensure_group_on_gpu(self.layer_id, "ffn")
                # 异步预取下一层的ffn（与当前FFN计算重叠）
                if hasattr(wm, "n_layers") and self.layer_id + 1 < wm.n_layers:
                    if hasattr(wm, "prefetch_group_async"):
                        wm.prefetch_group_async(self.layer_id + 1, "ffn")
                return
            except (AttributeError, NotImplementedError, KeyError) as e:
                # 组级API不可用或实现不完整，回退到层级API
                logger.warning(f"Layer {self.layer_id} FeedForward: group-level prefetch failed ({e}), fallback to layer-level")

        # 回退到原有的层级预取
        wm.ensure_weights_cuda(self.layer_id, modules, priority=True)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # Check CUDA context health
    #     if x.is_cuda:
    #         try:
    #             torch.cuda.current_device()
    #         except RuntimeError as e:
    #             logger.error(f"CUDA context error in feedforward: {e}")
    #             raise RuntimeError("CUDA context is corrupted") from e

    #     print(f"[FFN] Layer {self.layer_id} forward starting...")
    #     self._ensure_weights_cuda()
    #     print(f"[FFN] Layer {self.layer_id} weights ensured")

    #     # 确保权重H2D传输完成后再开始计算（使用统一的事件化等待）
    #     if self.streams and x.is_cuda:
    #         try:
    #             self.streams.wait_weight_ready_on_current(self.device)
    #         except Exception as e:
    #             logger.warning(f"Layer {self.layer_id} FeedForward: weight sync failed: {e}")

    #     print(f"[FFN] Layer {self.layer_id} starting computation...")
    #     # FFN计算 - 使用compute stream
    #     # compute_stream = self.streams.weight_compute if self.streams else None
    #     compute_stream = getattr(self.streams, "compute_ffn", None) or getattr(self.streams, "weight_compute", None)
    #     if compute_stream:
    #         with torch.cuda.stream(compute_stream):
    #             with cuda_timer("ffn_us", self.layer_id):
    #                 # SwiGLU
    #                 gate = self.w1(x)
    #                 up = self.w3(x)
    #                 hidden = F.silu(gate) * up
    #                 result = self.w2(hidden)
    #         print(f"[FFN] Layer {self.layer_id} computation done")
    #         return result
    #     else:
    #         with cuda_timer("ffn_us", self.layer_id):
    #             # SwiGLU
    #             gate = self.w1(x)
    #             up = self.w3(x)
    #             hidden = F.silu(gate) * up
    #             result = self.w2(hidden)
    #         print(f"[FFN] Layer {self.layer_id} computation done")
    #         return result
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import time
        import torch
        import torch.nn.functional as F

        # ---- CUDA 上下文健康检查（保持原样）----
        if x.is_cuda:
            try:
                torch.cuda.current_device()
            except RuntimeError as e:
                logger.error(f"CUDA context error in feedforward: {e}")
                raise RuntimeError("CUDA context is corrupted") from e

        print(f"[FFN] Layer {self.layer_id} forward starting...")

        # ---- 统一把计时器从 meta 拉成 CPU 标量，避免 .item()/加法在 meta/dtype 冲突 ----
        def _ensure_cpu_scalar_attr(mod, name: str):
            t = getattr(mod, name, None)
            if not isinstance(t, torch.Tensor) or getattr(t, "is_meta", False) \
            or (hasattr(t, "device") and t.device.type == "meta"):
                setattr(mod, name, torch.zeros((), dtype=torch.int64, device="cpu"))
        _ensure_cpu_scalar_attr(self, "ffn_us")
        _ensure_cpu_scalar_attr(self, "total_forward_us")

        # ============================================================
        # 1) 确保 ffn 组在 GPU，并等待该组 H2D 完成（在 compute_ffn 流上等待）
        # ============================================================
        wm = getattr(self, "weight_manager", None)
        in_use = False
        compute_stream = getattr(self.streams, "compute_ffn", None) or getattr(self.streams, "weight_compute", None)
        if wm and hasattr(wm, "_make_group_in_use"):
            wm._make_group_in_use(self.layer_id, "ffn")
            in_use = True
        try:
            if wm is not None:
                # 确保 ffn 组在 GPU（阻塞式，直到权重加载完成）
                if hasattr(wm, "ensure_group_on_gpu"):
                    wm.ensure_group_on_gpu(self.layer_id, "ffn")
                else:
                    # 回退到层级加载
                    modules = self._get_modules_dict()
                    wm.ensure_weights_cuda(self.layer_id, modules, priority=True)

                # 在 compute stream 上等待 ffn 组 H2D 传输完成（禁止降级到 CPU）
                if hasattr(wm, "wait_group_ready"):
                    wm.wait_group_ready(self.layer_id, "ffn", compute_stream=compute_stream)
        finally:
            if in_use and hasattr(wm, "_unmark_group_in_use"):
                wm._unmake_group_in_use(self.layer_id, "ffn")

        print(f"[FFN] Layer {self.layer_id} weights ensured and ready")

        # ============================================================
        # 2) 验证：所有权重必须已经在 CUDA 上（由 WSM 负责，forward 不做设备迁移）
        # Validation: All weights must already be on CUDA (managed by WSM, forward does NOT do device migration)
        # ============================================================
        target_dev = x.device

        # 确保 target_dev 是 CUDA（不接受 CPU）
        if not str(target_dev).startswith("cuda"):
            raise RuntimeError(f"Layer {self.layer_id} FeedForward: input x is on {target_dev}, but only CUDA is supported")

        # 验证所有 FFN 权重在 CUDA（不做迁移，只检查）
        for mod in (self.w1, self.w2, self.w3):
            # 权重必须在 CUDA，不允许 meta 或 CPU
            if str(mod.weight.device) == "meta" or getattr(mod.weight, "is_meta", False):
                raise RuntimeError(f"Layer {self.layer_id} FeedForward: {mod.__class__.__name__}.weight still on meta device after ensure_group_on_gpu()")

            if not str(mod.weight.device).startswith("cuda"):
                raise RuntimeError(f"Layer {self.layer_id} FeedForward: {mod.__class__.__name__}.weight on {mod.weight.device}, must be on CUDA (managed by WSM)")

            # 验证设备一致性（不做迁移，只检查）
            # Validate device consistency (no migration, only check)
            if mod.weight.device != target_dev:
                raise RuntimeError(f"Layer {self.layer_id} FeedForward: {mod.__class__.__name__}.weight on {mod.weight.device} but input on {target_dev}. "
                                 "Device mismatch must be fixed by WSM, not by forward().")

        print(f"[FFN] Layer {self.layer_id} device validation passed (all on CUDA)")
        print(f"[FFN] Layer {self.layer_id} starting computation...")

        # ---- 处理输入 x 的维度 ----
        if x.dim() == 2:
            # 允许上游给 2D；推断 D，稍后还原形状时按 B=1
            B, D = 1, x.size(-1)
            T = x.size(0)
            need_view_back = True
        else:
            assert x.dim() == 3, f"FFN expects 3D or 2D, got {x.shape}"
            B, T, D = x.shape
            need_view_back = False

        # 推断目标 dtype（从权重或输入）
        target_dtype = self.w1.weight.dtype if hasattr(self.w1, 'weight') else torch.bfloat16

        if getattr(x, "is_meta", False) or (hasattr(x, "device") and x.device.type == "meta"):
            x = torch.zeros((B, T, D), dtype=target_dtype, device=target_dev)

        # ---- 统一用 2D 计算（稳定 matmul 语义），再还原 ----
        x2 = x.reshape(-1, D)  # [B*T, D]


        # compute_stream 已在上面等待时获取，直接使用
        def _core_ffn(x2_: torch.Tensor) -> torch.Tensor:
            # SwiGLU：gate(w1) * up(w3)
            gate = self.w1(x2_)                 # [B*T, H]
            up   = self.w3(x2_)                 # [B*T, H]
            if up.dtype != gate.dtype:
                up = up.to(gate.dtype)
            act = F.silu(gate) * up

            # 送入 w2 前与 w2.weight 对齐（有些实现 w2 与 w1/bf16 混精度）
            if act.dtype != self.w2.weight.dtype or act.device != self.w2.weight.device:
                act = act.to(device=self.w2.weight.device, dtype=self.w2.weight.dtype, non_blocking=True)
            y2 = self.w2(act)                   # [B*T, D]
            return y2
        
        # protect
        in_use = False
        try:
            if wm is not None and hasattr(wm, "_mark_group_in_use"):
                wm._mark_group_in_use(self.layer_id, "ffn")
                in_use = True        
            start_t = time.time()
            if compute_stream:
                with torch.cuda.stream(compute_stream):
                    with cuda_timer("ffn_us", self.layer_id):
                        y2 = _core_ffn(x2)
            else:
                with cuda_timer("ffn_us", self.layer_id):
                    y2 = _core_ffn(x2)
        finally:
            if in_use and hasattr(wm, "_unmark_group_in_use"):
                wm._unmark_group_in_use(self.layer_id, "ffn")

        # 还原形状
        if need_view_back:
            result = y2.view(T, -1).unsqueeze(0).contiguous()  # [1, T, D]
        else:
            result = y2.view(B, T, -1).contiguous()

        # 统计收尾（用 Python float，避免与 bf16 张量混算）
        try:
            us = float((time.time() - start_t) * 1e6)
            PERF_TRACKER.add_layer_stat(self.layer_id, "total_forward_us", us)
        except Exception:
            pass

        print(f"[FFN] Layer {self.layer_id} computation done")

        # IN_USE 标记会在下一轮evict时自动清理，这里不需要手动unmark
        return result

# ---------- Optimized EncoderBlock ----------
class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = args.n_layers

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        self.attention.layer_id = layer_id
        self.feed_forward.layer_id = layer_id

        if layer_id < args.n_layers - 1:
            self.attention._next_layer_modules = self.feed_forward._get_modules_dict()

        self.forward_count = 0
        self.total_forward_time = 0.0

        self.weight_manager = None

        # 获取streams引用用于同步
        self.device = args.device
        streams = None
        try:
            import llama3.stream_mnt as stream_mnt
            streams = stream_mnt.get_streams(args.device)
        except Exception:
            pass
        self.streams = streams

        # 用于事件池定期清理的计数器
        self._gc_counter = 0

    def _get_modules_dict(self):
        """收集所有需要管理的模块（attention + feedforward）"""
        mods = {}
        if hasattr(self.attention, '_get_modules_dict'):
            mods.update(self.attention._get_modules_dict())
        if hasattr(self.feed_forward, '_get_modules_dict'):
            mods.update(self.feed_forward._get_modules_dict())
        return mods

    # def _ensure_weights_cuda(self):
    #     wm = getattr(self, "weight_manager", None)
    #     if wm is None:
    #         # 没有 weight_manager：权重应该已经在正确的设备上
    #         # No weight_manager: weights should already be on the correct device
    #         # Forward 不做设备迁移，只由 WSM 管理
    #         # Forward does NOT do device migration, only managed by WSM
    #         return
    #     modules = self._get_modules_dict()
    #     wm.ensure_weights_cuda(self.layer_id, modules, priority=True)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        forward_start = time.time()
        
        # dev   = self.device
        dev = str(self.device)
        if not dev.startswith("cuda"):
            try:
                dev = str(self.attention.wq.weight.device)
                self.device = dev  # 缓存，避免每次判断
            except Exception:
                pass
        dtype = getattr(self, "param_dtype", torch.bfloat16)

        if x.device != dev or x.dtype != dtype:
            x = x.to(device=dev, dtype=dtype, non_blocking=True)
        if freqs_complex.device != dev or freqs_complex.dtype != dtype:
            freqs_complex = freqs_complex.to(device=dev,  non_blocking=True)

        # 告知 WSM：计算前移（驱动 CPU/SSD 滑窗）
        wm = getattr(self, "weight_manager", None)
        if wm is not None and hasattr(wm, "note_compute_advance"):
            wm.note_compute_advance(self.layer_id)

        # 入口防呆：输入与归一化权重必须在 CUDA
        if x.device.type != "cuda":
            raise RuntimeError(f"Layer {self.layer_id} EncoderBlock: input x is on {x.device}, but only CUDA is supported")
        if self.attention_norm.weight.device.type != "cuda":
            raise RuntimeError(f"Layer {self.layer_id} EncoderBlock: attention_norm on {self.attention_norm.weight.device}, must be on CUDA")
        if self.ffn_norm.weight.device.type != "cuda":
            raise RuntimeError(f"Layer {self.layer_id} EncoderBlock: ffn_norm on {self.ffn_norm.weight.device}, must be on CUDA")

        dev = x.device
        streams = self.streams

        nvtx.range_push(f"layer_{self.layer_id}_forward")
        with cuda_timer("total_forward_us", self.layer_id):

            # -------- MHA 阶段：在 compute_mha 流上排队，先等组就绪事件 --------
            if wm is not None:
                wm.ensure_group_on_gpu(self.layer_id, "attn")
                # 让 compute_mha 流等待“attn 组 H2D 完成”的事件
                if streams and streams.compute_mha and hasattr(wm, "wait_group_ready"):
                    wm.wait_group_ready(self.layer_id, "attn", compute_stream=streams.compute_mha)
                elif hasattr(wm, "wait_group_ready"):
                    wm.wait_group_ready(self.layer_id, "attn", compute_stream=None)

            nvtx.range_push(f"layer_{self.layer_id}_attention")
            if streams and streams.compute_mha:
                torch.cuda.current_stream(dev).wait_stream(streams.compute_mha)  # default 等 MHA 流（安全）
                with torch.cuda.stream(streams.compute_mha):
                    attn_in  = self.attention_norm(x)
                    attn_out = self.attention(attn_in, start_pos, freqs_complex)  # 在 compute_mha 上排队
                # 在 MHA 流记录一个事件，供 FFN 流等待
                mha_eid, mha_evt = None, None
                try:
                    from llama3 import stream_mnt
                    mha_eid, mha_evt = stream_mnt.record_event_on(streams.compute_mha, device=dev)
                except Exception:
                    mha_evt = torch.cuda.Event()
                    mha_evt.record(streams.compute_mha)
            else:
                # 回退到默认流（不推荐，但保证可运行）
                attn_out = self.attention(self.attention_norm(x), start_pos, freqs_complex)

            # 在 MHA 完成之前不要在默认流上消费 attn_out；先做残差也放到 MHA 流里
            if streams and streams.compute_mha:
                with torch.cuda.stream(streams.compute_mha):
                    h = x + attn_out
            else:
                h = x + attn_out
            nvtx.range_pop()  # attention

            # -------- FFN 阶段：在 compute_ffn 流上排队，并等待 MHA 事件 --------
            if wm is not None:
                wm.ensure_group_on_gpu(self.layer_id, "ffn")
                if streams and streams.compute_ffn and hasattr(wm, "wait_group_ready"):
                    wm.wait_group_ready(self.layer_id, "ffn", compute_stream=streams.compute_ffn)
                elif hasattr(wm, "wait_group_ready"):
                    wm.wait_group_ready(self.layer_id, "ffn", compute_stream=None)

            if streams and streams.compute_ffn:
                # 让 FFN 流等待 MHA 结束（用上面记录的事件）
                if 'mha_evt' in locals() and mha_evt is not None:
                    streams.compute_ffn.wait_event(mha_evt)
                else:
                    # 最坏情况：FFN 流直接等 MHA 流
                    streams.compute_ffn.wait_stream(streams.compute_mha)

            nvtx.range_push(f"layer_{self.layer_id}_ffn")
            if streams and streams.compute_ffn:
                with torch.cuda.stream(streams.compute_ffn):
                    ffn_in  = self.ffn_norm(h)
                    ffn_out = self.feed_forward(ffn_in)   # 在 compute_ffn 上排队
                    out     = h + ffn_out
                # 记录 FFN 完成事件，便于上层/下一层需要
                try:
                    from llama3 import stream_mnt
                    ffn_eid, ffn_evt = stream_mnt.record_event_on(streams.compute_ffn, device=dev)
                    # 返回前让默认流等待 FFN 事件（保证 out 可安全被默认流使用）
                    stream_mnt.wait_event_on(torch.cuda.current_stream(dev), ffn_evt)
                    stream_mnt.release_event(ffn_eid, device=dev)
                except Exception:
                    ffn_evt = torch.cuda.Event()
                    ffn_evt.record(streams.compute_ffn)
                    torch.cuda.current_stream(dev).wait_event(ffn_evt)
            else:
                out = h + self.feed_forward(self.ffn_norm(h))
            nvtx.range_pop()  # ffn

            # 清理 MHA 事件
            if streams and streams.compute_mha and 'mha_eid' in locals() and mha_eid is not None:
                try:
                    from llama3 import stream_mnt
                    stream_mnt.release_event(mha_eid, device=dev)
                except Exception:
                    pass

        nvtx.range_pop()  # layer_forward

        self.forward_count += 1
        self.total_forward_time += time.time() - forward_start

        # 周期性 GC 事件池
        self._gc_counter += 1
        if self._gc_counter % 10 == 0:
            try:
                from llama3 import stream_mnt
                stream_mnt.gc_event_pool(device=dev, force=False)
            except Exception:
                pass

        return out

    
    def get_performance_stats(self) -> Dict:
        avg_time = self.total_forward_time / max(self.forward_count, 1) * 1000  # ms
        return {
            "layer_id": self.layer_id,
            "forward_count": self.forward_count,
            "total_time_ms": self.total_forward_time * 1000,
            "avg_time_ms": avg_time,
            "detailed_stats": PERF_TRACKER.layer_stats.get(self.layer_id, {})
        }

# ---------- Utility functions ----------
def get_global_performance_stats() -> Dict:
    return PERF_TRACKER.get_stats()

def reset_performance_stats():
    PERF_TRACKER.reset()

def optimize_layer_execution_order(layers: List[EncoderBlock]) -> List[int]:
    layer_stats = [(i, layer.get_performance_stats()["avg_time_ms"]) 
                   for i, layer in enumerate(layers)]
    
    layer_stats.sort(key=lambda x: x[1])
    return [layer_id for layer_id, _ in layer_stats]