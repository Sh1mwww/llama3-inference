import math, os
from typing import Optional, List, Dict
import torch, torch.nn as nn, torch.nn.functional as F
import threading
import logging
from contextlib import contextmanager
import time
from torch.backends.cuda import sdp_kernel as sdpa_kernel
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

# ---------- Global Thread Pool for Async Forward ----------
from concurrent.futures import ThreadPoolExecutor

_EXECUTOR_SINGLETON = None
_EXECUTOR_LOCK = threading.Lock()

def _get_executor():
    """
    获取全局线程池单例，用于 forward_async 的异步收尾。
    每个进程共享一个轻量线程池（也可放到 Transformer 里全局持有）。
    前向只有极少的"收尾回调"，2~4 个线程足矣。
    """
    global _EXECUTOR_SINGLETON
    if _EXECUTOR_SINGLETON is None:
        with _EXECUTOR_LOCK:
            if _EXECUTOR_SINGLETON is None:
                _EXECUTOR_SINGLETON = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fwd_async")
    return _EXECUTOR_SINGLETON

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

# Profiling control via environment variable
PROFILE = os.getenv("LLM_PROFILE", "0") == "1"

@contextmanager
def cuda_timer(key: str, layer_id: Optional[int] = None):
    # No-op when profiling is disabled (避免任何同步开销)
    if not PROFILE:
        yield
        return

    if not torch.cuda.is_available():
        yield
        return

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
                # 只同步当前流的事件，避免全局阻塞其它流（尤其是H2D）
                end_event.synchronize()

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
        self.compute_stream = getattr(streams, "compute_mha", None) 
        self.weight_h2d_stream = getattr(streams, "weight_h2d_mha", None) 
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
        # scores_buffer 已移除 - Flash Attention 不需要预分配 [B,H,T,T] 矩阵
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
            return
        if hasattr(wm, "wait_group_ready"):
            compute_stream = getattr(self.streams, "compute_attn", None)
            wm.wait_group_ready(self.layer_id, "attn", compute_stream=compute_stream)

    def _allocate_buffers(self, batch_size: int, seq_len: int, max_kv_len: int):
        """
        ⚠️ 此方法目前未被使用（已改用 Flash Attention）
        使用 scaled_dot_product_attention 后，不再需要预分配 attention scores buffer
        Flash Attention 内部使用 kernel fusion，避免物化 [B,H,T,T] 矩阵

        保留此方法仅用于向后兼容，如果需要回退到手写 attention 可以参考
        """
        if (self.qkv_buffer is None or
            self.qkv_buffer[0].size(0) < batch_size or
            self.qkv_buffer[0].size(1) < seq_len):

            with cuda_timer("memory_alloc_us", self.layer_id):
                # 注意：使用 Flash Attention 后，不再需要 scores_buffer
                # 以下代码仅分配 QKV buffers（如果需要）
                q_elements = batch_size * seq_len * self.n_heads_q * self.head_dim
                kv_elements = batch_size * seq_len * self.n_kv_heads * self.head_dim

                try:
                    from .memory_manager import GlobalMemoryManager
                    memory_manager = GlobalMemoryManager.get_instance()
                    if memory_manager:
                        # 只计算 QKV 的内存需求（不包括 scores）
                        total_bytes = (q_elements + 2 * kv_elements) * 2  # float16
                        if not memory_manager.can_allocate(total_bytes):
                            # 尝试清理内存
                            if hasattr(self, 'qkv_buffer') and self.qkv_buffer:
                                del self.qkv_buffer
                            torch.cuda.empty_cache()

                            if not memory_manager.can_allocate(total_bytes):
                                raise RuntimeError(f"Insufficient GPU memory: need {total_bytes/(1024**3):.2f}GB")
                except ImportError:
                    pass  # memory_manager not available

                try:
                    # QKV buffer (如果需要)
                    q_shape = (batch_size, seq_len, self.n_heads_q, self.head_dim)
                    kv_shape = (batch_size, seq_len, self.n_kv_heads, self.head_dim)

                    self.qkv_buffer = (
                        torch.empty(q_shape, dtype=torch.float16, device=self.device),
                        torch.empty(kv_shape, dtype=torch.float16, device=self.device),
                        torch.empty(kv_shape, dtype=torch.float16, device=self.device)
                    )
                    # ✅ scores_buffer 已移除 - Flash Attention 不需要显式分配

                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"GPU OOM during buffer allocation: batch={batch_size}, seq={seq_len}")
                    torch.cuda.empty_cache()
                    raise RuntimeError(f"GPU OOM: Cannot allocate attention buffers. Try reducing batch_size (current: {batch_size}) or max sequence length.") from e
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        # ============================================================
        # ⭐ 回绕通知 + 事件等待（彻底不兜底）
        # ============================================================
        wm = getattr(self, "weight_manager", None)
 
        # 这不会改变 WSM 的权重流式行为，只是防止激活回到 CPU
        if x.device.type != "cuda":
            target_device = getattr(self, "device", "cuda:0")
            logger.warning(
                f"[SelfAttention L{self.layer_id}] Input x on {x.device}, moving to {target_device}. "
                f"This should not happen in normal flow - investigate upstream."
            )
            x = x.to(target_device, non_blocking=True)

        # ⭐⭐⭐ 防御式检查：激活必须在 CUDA 上（早失败，避免后续隐式同步）
        if not x.is_cuda:
            raise RuntimeError(
                f"[SelfAttention L{self.layer_id}] Input activation is on {x.device}, expected CUDA. "
                f"This indicates activation was incorrectly moved to CPU. Shape: {x.shape}, dtype: {x.dtype}"
            )

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

        # ⭐ 调试日志：仅当环境变量启用时输出（避免 prefill 阶段 CPU 瓶颈）
        _verbose = os.getenv("ATTN_VERBOSE_LOG", "0") == "1"
        if _verbose:
            print(f"[ATTN] Layer {self.layer_id} forward starting...")

        # ============================================================
        # 1) 只用事件、不阻塞：标记组使用 + 等待组 ready 事件
        # ============================================================
        wm = getattr(self, "weight_manager", None)
        in_use = False
        try:
            if wm and hasattr(wm, "_mark_group_in_use"):
                wm._mark_group_in_use(self.layer_id, "attn")
                in_use = True

            # ⭐ 只用事件等待，不做同步阻塞
            # 在 compute_mha 流上等待 attn 组的 ready 事件（非阻塞式，只让流依赖事件）
            if wm is not None and hasattr(wm, "wait_group_ready"):
                #  额外保护：在 ATTN 开始时预 pin 同层 FFN，避免缝隙被逐出
                if hasattr(wm, "pin_group"):
                    try: wm.pin_group(self.layer_id, "ffn", reason="pair")
                    except Exception: pass
                wm.wait_group_ready(self.layer_id, "attn", compute_stream=self.compute_stream)

                # ✨ Stub 兜底检查：确保权重已真正加载（非空 stub）
                if self.wq.weight.numel() == 0:
                    # WSM 承诺的权重未到位，强制同步回退一次
                    print(f"[ATTN][L{self.layer_id}][ERROR] wq.weight is still stub after wait_group_ready!")
                    # 尝试强制同步加载（阻塞）
                    if hasattr(wm, "_group_is_resident"):
                        if not wm._group_is_resident(self.layer_id, "attn", wait_for_event=True):
                            raise RuntimeError(f"[ATTN][L{self.layer_id}] Cannot load weights: still stub after sync wait")
                    else:
                        raise RuntimeError(f"[ATTN][L{self.layer_id}] Cannot load weights: stub detected (no resident check)")

            # ⭐ 可选：等待 KV 块 ready 事件（如果有预取）
            # 在 decode 阶段（start_pos > 0），等待本层所需的 KV 块 H2D 完成
            if start_pos > 0 and self.offloader is not None and hasattr(self.offloader, "wait_blocks_ready"):
                # 计算本层需要的块：最近窗口 tokens
                blocks = self.offloader.plan_tail_window_blocks(start_pos, seqlen, window_tokens=BLOCK)
                if blocks:
                    self.offloader.wait_blocks_ready(self.layer_id, blocks, stream=self.compute_stream)

            if _verbose:
                print(f"[ATTN] Layer {self.layer_id} weights event wait done (non-blocking)")

            # 更新全局状态跟踪器
            tracker = get_global_tracker()
            if tracker:
                batch_idx = tracker.current_batch
            else:
                batch_idx = 0

            if _verbose:
                print(f"[ATTN] Layer {self.layer_id} starting computation...")
            
            # ⭐⭐⭐ Background prefetch (compute-overlapped)
            try:
                wm = getattr(self, "weight_manager", None)

                # --- 组级权重预取：本层 FFN + 后续 D 层 ATTn ---
                # 说明：prefetch_group_async 幂等；若该组已在 GPU 或正在 H2D，会被 WSM 跳过
                if wm is not None and hasattr(wm, "prefetch_group_async"):
                    # 1) ⭐ 先把"本层 FFN"挂起并 PIN（防止在 MHA→FFN 缝隙中被淘汰）
                    wm.prefetch_group_async(self.layer_id, "ffn", pin=True, reason="pair")
                    # 2) 顶补延后：由 compute-done 回调触发（避免前向路径同步收缩/补齐）
                    #    见：wm.notify_group_compute_done() 的后台线程

                # --- KV：为“下一层注意力”预拉最近窗口的历史 KV 到 HBM（仅 decode 阶段） ---
                # 注意：start_pos==0 为 prefill，此时下一层当前块可能尚不存在，故跳过
                if (start_pos > 0) and getattr(self, "offloader", None) is not None:
                    # 该方法会：必要时先 SSD->DRAM，再在 kv_h2d 流发起 DRAM->HBM；并记录事件，fetch() 将命中
                    self.offloader.prefetch_for_next_layer(
                        current_layer=self.layer_id,
                        start_pos=int(start_pos),
                        seqlen=int(seqlen),
                        bsz=int(bsz),
                        window_tokens=BLOCK,
                    )
            except Exception as e:
                # 非致命：任何预取异常都不影响主计算路径
                if getattr(wm, "verbose", False):
                    print(f"[ATTN][L{self.layer_id}] background prefetch skipped: {e}")

            # 预期形状
            exp_q = (self.n_heads_q * self.head_dim, x.size(-1))
            exp_kv = (self.n_kv_heads * self.head_dim, x.size(-1))

            def _shape(p: torch.nn.Parameter):
                # 真实形状优先；stub 或 meta 时读 _shape_hint
                if getattr(p, "is_meta", False) or (hasattr(p, "device") and p.device.type == "meta"):
                    return getattr(p, "_shape_hint", tuple(p.shape))
                if p.numel() == 0 and hasattr(p, "_shape_hint"):
                    return getattr(p, "_shape_hint")
                return tuple(p.shape)
            
            def _check_or_defer(p: torch.nn.Parameter, exp, name: str):
                shp = _shape(p)
                if shp != exp:
                    # 允许在 WSM 管理下的 stub 先“通过”，真正的 weight 会由 WSM 在 H2D 完成后安装
                    if p.numel() == 0 and getattr(self, "weight_manager", None) is not None:
                        if os.getenv("WSM_VERBOSE_MISMATCH", "0") == "1":
                            print(f"[ATTN][L{self.layer_id}] defer {name} shape check: stub {shp} -> expect {exp}")
                        return
                    raise RuntimeError(f"[{name} shape] {shp} != {exp} (dim/head config mismatch?)")

            # if _shape(self.wq.weight) != exp_q:
            #     raise RuntimeError(f"[Q shape] { _shape(self.wq.weight) } != {exp_q} "
            #                     f"(likely manifest q/k/v mapping issue)")
            # if _shape(self.wk.weight) != exp_kv:
            #     raise RuntimeError(f"[K shape] { _shape(self.wk.weight) } != {exp_kv} "
            #                     f"(likely manifest q/k/v mapping issue)")
            # if _shape(self.wv.weight) != exp_kv:
            #     raise RuntimeError(f"[V shape] { _shape(self.wv.weight) } != {exp_kv} "
            #                     f"(likely manifest q/k/v mapping issue)")   
            
            _check_or_defer(self.wq.weight, exp_q,  "Q")
            _check_or_defer(self.wk.weight, exp_kv, "K")
            _check_or_defer(self.wv.weight, exp_kv, "V")

            # ---- Device alignment (no synchronous fallback) ----
            dev = self.wq.weight.device
            if x.device != dev:
                x = x.to(dev, non_blocking=True)
            if freqs_complex.device != dev:
                freqs_complex = freqs_complex.to(dev, non_blocking=True)

            # QKV投影 - 使用专门的compute stream
            # compute_stream = self.streams.weight_compute if self.streams else None
            compute_stream = self.compute_stream
            qkv_start = time.time()
            if compute_stream:
                with torch.cuda.stream(compute_stream):
                    with cuda_timer("attn_us", self.layer_id):
                        # print("wq.weight.device =", self.wq.weight.device)
                        q = self.wq(x).view(bsz, seqlen, self.n_heads_q, self.head_dim)
                        if _verbose:
                            print(f"[ATTN] Layer {self.layer_id} Q projection done ({(time.time()-qkv_start)*1000:.2f}ms)")
                        k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                        if _verbose:
                            print(f"[ATTN] Layer {self.layer_id} K projection done ({(time.time()-qkv_start)*1000:.2f}ms)")
                        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                        if _verbose:
                            print(f"[ATTN] Layer {self.layer_id} V projection done ({(time.time()-qkv_start)*1000:.2f}ms)")

                        # 应用旋转位置编码
                        # q = apply_rotary_embeddings(q, freqs_complex)
                        # k = apply_rotary_embeddings(k, freqs_complex)
                        q = apply_rotary_embeddings(q, freqs_complex, start_pos=start_pos)
                        k = apply_rotary_embeddings(k, freqs_complex, start_pos=start_pos)
                        if _verbose:
                            print(f"[ATTN] Layer {self.layer_id} RoPE done ({(time.time()-qkv_start)*1000:.2f}ms)")

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
            # 重复KV头以匹配查询头数（使用零拷贝视图扩展，避免物理复制）
            # if self.n_heads_q != self.n_kv_heads:
            if (self.n_heads_q != self.n_kv_heads) and (k_full.size(1) != self.n_heads_q):
                # 旧方式（物理复制）：
                # k_full = k_full.repeat_interleave(self.n_rep, dim=1)
                # v_full = v_full.repeat_interleave(self.n_rep, dim=1)

                # 新方式（零拷贝）：(B,Hkv,Tk,D) -> (B,Hkv,1,Tk,D) -> (B,Hkv,n_rep,Tk,D) -> (B,Hq,Tk,D)
                k_full = k_full.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1)\
                               .reshape(bsz, self.n_heads_q, k_full.size(2), self.head_dim)
                v_full = v_full.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1)\
                               .reshape(bsz, self.n_heads_q, v_full.size(2), self.head_dim)
        
            # 确保缓冲区足够大
            q = q.transpose(1, 2)  # (B, H, Tq, D)

            # 在进入注意力计算前为 workspace 预留显存余量
            # 注意：使用 Flash Attention 后，workspace 需求大幅降低（无需物化 [B,H,T,T]）
            wm = getattr(self, "weight_manager", None)
            if wm is not None and hasattr(wm, "ensure_headroom_mb"):
                try:
                    # 默认 64 MB（Flash Attention 只需少量 workspace）
                    # 优先使用 WSM 初始化时读取的值，兼容运行时环境变量修改
                    extra_headroom_mb = getattr(wm, "attn_workspace_headroom_mb", 64)
                except Exception:
                    extra_headroom_mb = 64
                # 避免误逐出当前层 attn 组
                excl = {(self.layer_id, "attn")}
                wm.ensure_headroom_mb(extra_headroom_mb, exclude=excl)

            # Attention计算 - 使用compute stream
            nvtx.range_push(f"layer_{self.layer_id}_attention_compute")
            do_profile_gpu = bool(self.enable_profiling and x.is_cuda)

            # 🔥 使用 Flash Attention - 统一使用旧 API（更稳定）
            # 注意：PyTorch 2.4+ 的新 API (torch.nn.attention.sdpa_kernel) 参数不同
            # 为了兼容性，统一使用 torch.backends.cuda.sdp_kernel
            is_causal = hasattr(self, 'apply_causal_mask') and self.apply_causal_mask
            from contextlib import nullcontext
            try:
                from torch.backends.cuda import sdp_kernel as sdpa_kernel
                # 允许 math 回退，避免“无可用内核”的硬错误
                sdpa_ctx = sdpa_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True)
            except Exception:
                sdpa_ctx = nullcontext()

            if compute_stream:
                with torch.cuda.stream(compute_stream):
                    with cuda_timer("attn_us", self.layer_id):
                        # mha
                        if do_profile_gpu:
                            attn_evt_start = torch.cuda.Event(enable_timing=True)
                            attn_evt_end = torch.cuda.Event(enable_timing=True)
                            attn_evt_start.record()

                        # 避免物化 [B,H,T,T] 的 scores/attn_weights
                        with sdpa_ctx:
                            out = torch.nn.functional.scaled_dot_product_attention(
                                q, k_full, v_full, attn_mask=None, dropout_p=0.0, is_causal=is_causal
                            )
                        # 🚀 立刻释放不再需要的中间激活，降低峰值内存
                        del q
                        del k
                        del v
                        del k_full
                        del v_full

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

                    # 避免物化 [B,H,T,T] 的 scores/attn_weights
                    with sdpa_ctx:
                        out = torch.nn.functional.scaled_dot_product_attention(
                            q, k_full, v_full, attn_mask=None, dropout_p=0.0, is_causal=is_causal
                        )
                    del q
                    del k
                    del v
                    del k_full
                    del v_full

                    if do_profile_gpu:
                        attn_evt_end.record()
                        if not attn_evt_end.query():
                            attn_evt_end.synchronize()
                        self.attn_time = attn_evt_start.elapsed_time(attn_evt_end) * 1000
                        PERF_TRACKER.add_layer_stat(self.layer_id, "attn_us", self.attn_time)
                    else:
                        self.attn_time = 0
            nvtx.range_pop()  # attention_compute

            # 转换回 [B, T, H, D] 格式
            out = out.transpose(1, 2).contiguous()

            # --- 统计信息（若有） ---
            stats = PERF_TRACKER.layer_stats.get(self.layer_id, {})
            self.kv_elapsed_time = stats.get("kv_fetch_us", 0)
            self.attn_time       = stats.get("attn_us",     0)


            feat = self.n_heads_q * self.head_dim  # = dim
            B, Tq = bsz, seqlen
            w = self.wo.weight
        
            # ⚠️ 注意：使用 Flash Attention 后，attn_weights 不再物化
            # 重要度统计功能已被禁用，因为 scaled_dot_product_attention 不返回权重矩阵
            # 这是内存优化的预期行为：避免物化 [B,H,T,T] 的巨大矩阵
            # 如果需要 token importance 统计，需要使用其他方法（例如梯度、探针等）
                    
                    
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
            del out
            w = self.wo.weight
            if getattr(out2d, "is_meta", False) or (hasattr(out2d, "device") and out2d.device.type == "meta"):
                out2d = torch.zeros((B*Tq, feat), dtype=w.dtype, device=w.device)
            else:
                if out2d.dtype != w.dtype:
                    out2d = out2d.to(w.dtype)
                if out2d.device != w.device:
                    out2d = out2d.to(w.device, non_blocking=True)

            # 输出投影（可用 compute_stream）
            if compute_stream:
                with torch.cuda.stream(compute_stream):
                    res2d = self.wo(out2d)
            else:
                res2d = self.wo(out2d)
            del out2d

            result = res2d.view(B, Tq, -1).contiguous()
            del res2d

            # --- 统计收尾 ---
            total_time = (time.time() - start_time) * 1e6  # μs
            PERF_TRACKER.add_layer_stat(self.layer_id, "total_forward_us", total_time)
            # print(f"[ATTN] Layer {self.layer_id} computation done")

            if wm and hasattr(wm, "notify_group_compute_done"):
                evt = torch.cuda.Event()
                evt.record(self.compute_stream if self.compute_stream is not None else torch.cuda.current_stream())
                wm.notify_group_compute_done(self.layer_id, "attn", evt)

            # ============================================================
            # 3.2) Eager KV Spill: 在 prefill 分支结束前，将本层生成的 KV 异步写入 SSD
            # 在 prefill 阶段（start_pos==0），本层的 KV 已用于注意力计算，可立即下放到 SSD
            # ============================================================
            if start_pos == 0 and getattr(self, "offloader", None) is not None:
                # 把本层刚生成的 KV 覆盖到的 token 全部甩到 SSD
                # upto_token = start_pos + seqlen 表示当前层已处理到的 token 位置
                self.offloader.eager_spill_layer(
                    self.layer_id,
                    upto_token=start_pos + seqlen,
                    async_write=True
                )

            return result
        finally:
            # 对称解除：ATTN 阶段 pin 的配对 FFN
            if wm is not None and hasattr(wm, "unpin_group"):
                wm.unpin_group(self.layer_id, "ffn")
            # 解除 IN_USE
            if in_use and hasattr(wm, "_unmark_group_in_use"):
                wm._unmark_group_in_use(self.layer_id, "attn")

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
        self.weight_manager = None  # Will be injected by _integrate_wsm_to_layers

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
        if wm is None:
            return
        compute_stream = getattr(self.streams, "compute_ffn", None)
        if hasattr(wm, "wait_group_ready"):
            wm.wait_group_ready(self.layer_id, "ffn", compute_stream=compute_stream)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # --- 设备健康检查（与 SelfAttention 一致） ---
    #     if x.is_cuda:
    #         try:
    #             torch.cuda.current_device()
    #         except RuntimeError as e:
    #             raise RuntimeError("CUDA context is corrupted") from e

    #     print(f"[FFN] Layer {self.layer_id} forward starting...")

    #     wm = getattr(self, "weight_manager", None)
    #     in_use = False
    #     try:
    #         # ============================================================
    #         # 只用事件、不阻塞：标记组使用 + 等待组 ready 事件
    #         # ============================================================
    #         if wm and hasattr(wm, "_mark_group_in_use"):
    #             wm._mark_group_in_use(self.layer_id, "ffn")
    #             in_use = True

    #         # ⭐ 只用事件等待，不做同步阻塞
    #         # 在 compute_ffn 流上等待 ffn 组的 ready 事件（非阻塞式，只让流依赖事件）
    #         compute_stream = getattr(self.streams, "compute_ffn", None)
    #         if wm is not None and hasattr(wm, "wait_group_ready"):
    #             wm.wait_group_ready(self.layer_id, "ffn", compute_stream=compute_stream)

    #         print(f"[FFN] Layer {self.layer_id} weights event wait done (non-blocking)")

    #         # ⭐⭐⭐ Background prefetch during FFN compute（后续 D 层 ATTn）
    #         try:
    #             if wm is not None and hasattr(wm, "prefetch_group_async"):
    #                 D = max(1, int(os.getenv("WSM_GROUP_PREFETCH_DEPTH", "2")))
    #                 nL = getattr(wm, "n_layers", 0)
    #                 used = getattr(wm, "_gpu_group_lru", [])  # 仅做轻量预算估计
    #                 budget = max(0, int(os.getenv("WSM_GPU_MAX_GROUPS", "10")) - len(used) - 1)
    #                 # 只在有预算时推进预取队列
    #                 depth = min(D, budget) if budget > 0 else 0
    #                 for off in range(1, depth + 1):
    #                     nxt = self.layer_id + off
    #                     if nxt < nL:
    #                         wm.prefetch_group_async(nxt, "attn")
    #         except Exception as e:
    #             if getattr(wm, "verbose", False):
    #                 print(f"[FFN][L{self.layer_id}] background prefetch skipped: {e}")

    #         # --- FFN 计算 ---
    #         compute_stream = getattr(self.streams, "compute_ffn", None) 
    #         if compute_stream:
    #             with torch.cuda.stream(compute_stream):
    #                 with cuda_timer("ffn_us", self.layer_id):
    #                     gate = self.w1(x)         # (B,T,28672)
    #                     up   = self.w3(x)         # (B,T,28672)
    #                     gate = F.silu(gate, inplace=True)       # in-place：覆盖 gate
    #                     up.mul_(gate)             # in-place：up 直接变成 hidden
    #                     result  = self.w2(up)        # 仅两块大张量存活
    #         else:
    #             with cuda_timer("ffn_us", self.layer_id):
    #                 gate = self.w1(x)         # (B,T,28672)
    #                 up   = self.w3(x)         # (B,T,28672)
    #                 gate = F.silu(gate, inplace=True)       # in-place：覆盖 gate
    #                 up.mul_(gate)             # in-place：up 直接变成 hidden
    #                 result  = self.w2(up)        # 仅两块大张量存活

    #         # 通知：FFN 组计算完成（便于组级 LRU 收缩/回收）
    #         if wm and hasattr(wm, "notify_group_compute_done"):
    #             evt = torch.cuda.Event()
    #             evt.record(compute_stream if compute_stream is not None else torch.cuda.current_stream())
    #             wm.notify_group_compute_done(self.layer_id, "ffn", evt)

    #         print(f"[FFN] Layer {self.layer_id} computation done")
    #         return result

    #     finally:
    #         # ⭐ 计算收尾：解除 FFN 组的 pin（对称于 ATTN 阶段的 pin）
    #         if wm is not None and hasattr(wm, "unpin_group"):
    #             wm.unpin_group(self.layer_id, "ffn")
    #         # 解除 in_use 标记
    #         if in_use and hasattr(wm, "_unmark_group_in_use"):
    #             wm._unmark_group_in_use(self.layer_id, "ffn")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        if x.device.type != "cuda":
            raise RuntimeError(
                f"[FeedForward L{self.layer_id}] Input tensor must be on CUDA, got {x.device}"
            )

        # ============================================================
        # ⭐ 回绕通知 + 事件等待（彻底不兜底）
        # ============================================================
        wm = getattr(self, "weight_manager", None)

        # 没有 WSM 时直接执行计算
        if wm is None:
            gate = self.w1(x)
            up   = self.w3(x)
            gate = F.silu(gate, inplace=True)
            up.mul_(gate)
            del gate
            out = self.w2(up)
            del up
            return out

        in_use = False
        if hasattr(wm, "_mark_group_in_use"):
            wm._mark_group_in_use(self.layer_id, "ffn")
            in_use = True

        try:
            compute_stream = getattr(self.streams, "compute_ffn", None)

            # ⭐ 只用事件等待，不做同步阻塞
            if wm is not None and hasattr(wm, "wait_group_ready"):
                wm.wait_group_ready(self.layer_id, "ffn", compute_stream=compute_stream)

            # ---- Device alignment (no synchronous fallback) ----
            dev = self.w1.weight.device
            if x.device != dev:
                x = x.to(dev, non_blocking=True)

            # 在 FFN 期间尝试预取下一层 ATTN
            try:
                if hasattr(wm, "prefetch_group_async"):
                    nxt = self.layer_id + 1
                    if nxt < getattr(self, "n_layer", 1 << 30):
                        gpu_count = wm.num_gpu_groups()
                        budget    = int(getattr(wm, "gpu_max_groups", 4))
                        if gpu_count < budget:
                            wm.prefetch_group_async(nxt, "attn")
            except Exception:
                pass

            if compute_stream is not None:
                with torch.cuda.stream(compute_stream):
                    gate = self.w1(x)
                    up   = self.w3(x)
                    gate = F.silu(gate, inplace=True)
                    up.mul_(gate)
                    del gate
                    result = self.w2(up)
                    del up
            else:
                gate = self.w1(x)
                up   = self.w3(x)
                gate = F.silu(gate, inplace=True)
                up.mul_(gate)
                del gate
                result = self.w2(up)
                del up

            if hasattr(wm, "notify_group_compute_done"):
                evt = torch.cuda.Event()
                evt.record(compute_stream if compute_stream is not None else torch.cuda.current_stream())
                wm.notify_group_compute_done(self.layer_id, "ffn", evt)

            return result

        finally:
            if in_use and hasattr(wm, "_unmark_group_in_use"):
                wm._unmark_group_in_use(self.layer_id, "ffn")


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
    
    def forward_async(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor,
                      wait_on: Optional[torch.cuda.Event] = None) -> tuple:
        """
        轻量异步 forward：只做事件排队，返回 (out, done_evt)。

        Args:
            x: 输入激活
            start_pos: 序列起始位置
            freqs_complex: RoPE 频率
            wait_on: 可选的前置事件（上一层的 done_evt）

        Returns:
            (out, done_evt): 输出张量和完成事件（在 FFN 流上记录）
        """
        import torch
        from llama3 import stream_mnt

        target_device = self.attention_norm.weight.device
        dtype = getattr(self, "param_dtype", torch.bfloat16)

        # 设备/dtype 对齐（尽量避免同步）
        copy_stream = None
        if x.device != target_device:
            if target_device.type == "cpu" and x.device.type == "cuda":
                logger.warning(f"Layer {self.layer_id}: Attempted to move CUDA activation to CPU. Keeping on CUDA.")
            else:
                copy_stream = torch.cuda.current_stream(device=target_device)
                x = x.to(device=target_device, dtype=dtype, non_blocking=True)
        elif x.dtype != dtype:
            x = x.to(dtype=dtype)

        freq_copy_stream = None
        if freqs_complex.device != target_device:
            if target_device.type == "cpu" and freqs_complex.device.type == "cuda":
                logger.warning(f"Layer {self.layer_id}: Attempted to move CUDA freqs to CPU. Keeping on CUDA.")
            else:
                freq_copy_stream = torch.cuda.current_stream(device=target_device)
                freqs_complex = freqs_complex.to(device=target_device, non_blocking=True)

        wm = getattr(self, "weight_manager", None)
        if wm is not None and hasattr(wm, "note_compute_advance"):
            wm.note_compute_advance(self.layer_id)

        if x.device.type != "cuda":
            raise RuntimeError(f"Layer {self.layer_id}: input activation must be on CUDA, got {x.device}")

        streams = self.streams
        device = x.device

        # -------- 1) MHA：只挂事件等待权重 + 可选的前置事件 --------
        if wm is not None and hasattr(wm, "wait_group_ready"):
            wm.wait_group_ready(self.layer_id, "attn",
                                compute_stream=getattr(streams, "compute_mha", None))

        if streams and streams.compute_mha:
            with torch.cuda.stream(streams.compute_mha):
                if copy_stream is not None:
                    streams.compute_mha.wait_stream(copy_stream)
                if freq_copy_stream is not None:
                    streams.compute_mha.wait_stream(freq_copy_stream)
                if wait_on is not None:
                    streams.compute_mha.wait_event(wait_on)
                attn_in = self.attention_norm(x)
                attn_out = self.attention(attn_in, start_pos, freqs_complex)
            mha_eid, mha_evt = stream_mnt.record_event_on(streams.compute_mha, device=device)
        else:
            if wait_on is not None:
                torch.cuda.current_stream().wait_event(wait_on)
            attn_out = self.attention(self.attention_norm(x), start_pos, freqs_complex)
            mha_eid, mha_evt = None, None

        # 残差（在 MHA 流完成后）
        if streams and streams.compute_mha and mha_evt is not None:
            with torch.cuda.device(device):
                torch.cuda.current_stream().wait_event(mha_evt)
        h = x
        h.add_(attn_out)
        del attn_out

        # -------- 2) FFN：只挂事件等待权重；FFN 流等待 MHA 事件 --------
        if wm is not None and hasattr(wm, "wait_group_ready"):
            wm.wait_group_ready(self.layer_id, "ffn",
                                compute_stream=getattr(streams, "compute_ffn", None))

        if streams and streams.compute_ffn and mha_evt is not None:
            streams.compute_ffn.wait_event(mha_evt)

        if streams and streams.compute_ffn:
            with torch.cuda.stream(streams.compute_ffn):
                ffn_in = self.ffn_norm(h)
                ffn_out = self.feed_forward(ffn_in)
            ffn_eid, ffn_evt = stream_mnt.record_event_on(streams.compute_ffn, device=device)
        else:
            ffn_out = self.feed_forward(self.ffn_norm(h))
            ffn_eid, ffn_evt = None, None

        h.add_(ffn_out)
        del ffn_out
        out = h

        # -------- 3) （可选）预取 L+1 的 KV 窗口 --------
        try:
            offloader = getattr(self.attention, "offloader", None)
            kv_stream = getattr(streams, "kv_h2d", None)
            nxt = self.layer_id + 1
            if (offloader is not None) and (nxt < self.n_layer) and (kv_stream is not None):
                window = int(getattr(offloader, "block_size", 256))
                seqlen = int(x.size(1))
                blocks = offloader.plan_tail_window_blocks(start_pos, seqlen, window_tokens=window)
                if hasattr(offloader, "prefetch_blocks_async"):
                    offloader.prefetch_blocks_async(nxt, blocks, stream=kv_stream)
        except Exception:
            pass

        # -------- 4) 清理 MHA 事件 --------
        if mha_eid is not None:
            stream_mnt.release_event(mha_eid, device=device)

        # 返回输出和 FFN 完成事件（不在这里等待）
        return out, ffn_evt

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        import torch
        from llama3 import stream_mnt

        # ⭐⭐⭐ 修复：激活应该跟随权重设备，而不是 self.device（后者可能在 OOM 时被改成 "cpu"）
        # 使用 attention_norm.weight 的设备作为目标设备（因为它是第一个会用到的权重）
        target_device = self.attention_norm.weight.device
        dtype = getattr(self, "param_dtype", torch.bfloat16)

        # 只在必要时迁移，且必须确保不会把 CUDA 激活迁移到 CPU
        if x.device != target_device:
            if target_device.type == "cpu" and x.device.type == "cuda":
                # 警告：不应该把 CUDA 激活迁移到 CPU
                logger.warning(f"Layer {self.layer_id}: Attempted to move CUDA activation to CPU. Keeping on CUDA.")
            else:
                x = x.to(device=target_device, dtype=dtype, non_blocking=True)
        elif x.dtype != dtype:
            x = x.to(dtype=dtype)

        if freqs_complex.device != target_device:
            if target_device.type == "cpu" and freqs_complex.device.type == "cuda":
                logger.warning(f"Layer {self.layer_id}: Attempted to move CUDA freqs to CPU. Keeping on CUDA.")
            else:
                freqs_complex = freqs_complex.to(device=target_device, non_blocking=True)

        # Norm 模块应该已经在正确的设备上（通过 weight streaming 管理）
        # 如果不在，这里也不需要强制迁移（因为 RMSNorm.forward 会自动处理）

        wm = getattr(self, "weight_manager", None)
        if wm is not None and hasattr(wm, "note_compute_advance"):
            wm.note_compute_advance(self.layer_id)

        # ⭐ 只检查激活是否在 CUDA 上（权重可能在 SSD streaming 模式下动态加载）
        if x.device.type != "cuda":
            raise RuntimeError(f"Layer {self.layer_id}: input activation must be on CUDA, got {x.device}")

        streams = self.streams
        device  = x.device

        # -------- 1) MHA：只挂事件等待权重 → compute_mha 流执行 → 记录事件 --------
        if wm is not None and hasattr(wm, "wait_group_ready"):
            wm.wait_group_ready(self.layer_id, "attn",
                                compute_stream=getattr(streams, "compute_mha", None))

        if streams and streams.compute_mha:
            with torch.cuda.stream(streams.compute_mha):
                attn_in  = self.attention_norm(x)
                attn_out = self.attention(attn_in, start_pos, freqs_complex)
            # 记录 MHA 完成事件
            mha_eid, mha_evt = stream_mnt.record_event_on(streams.compute_mha, device=device)
        else:
            attn_out = self.attention(self.attention_norm(x), start_pos, freqs_complex)
            mha_eid, mha_evt = None, None  # 无独立流则不产生命名事件

        # 残差最好也在 MHA 流完成后再落到默认流
        if streams and streams.compute_mha and mha_evt is not None:
            with torch.cuda.device(device):
                torch.cuda.current_stream().wait_event(mha_evt)
        h = x
        h.add_(attn_out)
        del attn_out

        # -------- 2) FFN：只挂事件等待权重；FFN 流等待 MHA 事件 → 计算 --------
        if wm is not None and hasattr(wm, "wait_group_ready"):
            wm.wait_group_ready(self.layer_id, "ffn",
                                compute_stream=getattr(streams, "compute_ffn", None))

        if streams and streams.compute_ffn and mha_evt is not None:
            streams.compute_ffn.wait_event(mha_evt)

        if streams and streams.compute_ffn:
            with torch.cuda.stream(streams.compute_ffn):
                ffn_in   = self.ffn_norm(h)
                ffn_out  = self.feed_forward(ffn_in)
            # FFN 完成事件
            ffn_eid, ffn_evt = stream_mnt.record_event_on(streams.compute_ffn, device=device)
        else:
            ffn_out = self.feed_forward(self.ffn_norm(h))
            ffn_eid, ffn_evt = None, None

        h.add_(ffn_out)
        del ffn_out
        out = h  # 最终残差复用了 x 的存储

        # -------- 3) （可选）在 FFN 期间预取 L+1 的 KV 窗口 --------
        try:
            offloader = getattr(self.attention, "offloader", None)
            kv_stream = getattr(streams, "kv_h2d", None)
            nxt = self.layer_id + 1
            if (offloader is not None) and (nxt < self.n_layer) and (kv_stream is not None):
                window = int(getattr(offloader, "block_size", 256))
                seqlen = int(x.size(1))
                blocks = offloader.plan_tail_window_blocks(start_pos, seqlen, window_tokens=window)
                if hasattr(offloader, "prefetch_blocks_async"):
                    offloader.prefetch_blocks_async(nxt, blocks, stream=kv_stream)
        except Exception:
            pass

        # -------- 4) 在默认流上等待 FFN 完成事件（只事件依赖），然后返回 --------
        if ffn_evt is not None:
            with torch.cuda.device(device):
                torch.cuda.current_stream().wait_event(ffn_evt)
            if ffn_eid is not None:
                stream_mnt.release_event(ffn_eid, device=device)
        if mha_eid is not None:
            stream_mnt.release_event(mha_eid, device=device)

        return out
    
    # def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
    #     forward_start = time.time()
        
    #     # dev   = self.device
    #     dev = str(self.device)
    #     if not dev.startswith("cuda"):
    #         try:
    #             dev = str(self.attention.wq.weight.device)
    #             self.device = dev  # 缓存，避免每次判断
    #         except Exception:
    #             pass
    #     dtype = getattr(self, "param_dtype", torch.bfloat16)

    #     if x.device != dev or x.dtype != dtype:
    #         x = x.to(device=dev, dtype=dtype, non_blocking=True)
    #     if freqs_complex.device != dev or freqs_complex.dtype != dtype:
    #         freqs_complex = freqs_complex.to(device=dev,  non_blocking=True)

    #     # 告知 WSM：计算前移（驱动 CPU/SSD 滑窗）
    #     wm = getattr(self, "weight_manager", None)
    #     if wm is not None and hasattr(wm, "note_compute_advance"):
    #         wm.note_compute_advance(self.layer_id)

    #     # 入口防呆：输入与归一化权重必须在 CUDA
    #     if x.device.type != "cuda":
    #         raise RuntimeError(f"Layer {self.layer_id} EncoderBlock: input x is on {x.device}, but only CUDA is supported")
    #     if self.attention_norm.weight.device.type != "cuda":
    #         raise RuntimeError(f"Layer {self.layer_id} EncoderBlock: attention_norm on {self.attention_norm.weight.device}, must be on CUDA")
    #     if self.ffn_norm.weight.device.type != "cuda":
    #         raise RuntimeError(f"Layer {self.layer_id} EncoderBlock: ffn_norm on {self.ffn_norm.weight.device}, must be on CUDA")

    #     dev = x.device
    #     streams = self.streams

    #     nvtx.range_push(f"layer_{self.layer_id}_forward")
    #     with cuda_timer("total_forward_us", self.layer_id):

    #         # -------- MHA 阶段：只做"事件依赖"，不再阻塞 ensure -----------
    #         if wm is not None:
    #             # REMOVED: wm.ensure_group_on_gpu(self.layer_id, "attn")
    #             if streams and streams.compute_mha and hasattr(wm, "wait_group_ready"):
    #                 wm.wait_group_ready(self.layer_id, "attn", compute_stream=streams.compute_mha)  # NEW: 纯事件挂载
    #             elif hasattr(wm, "wait_group_ready"):
    #                 wm.wait_group_ready(self.layer_id, "attn", compute_stream=None)                # NEW

    #         nvtx.range_push(f"layer_{self.layer_id}_attention")
    #         if streams and streams.compute_mha:
    #             torch.cuda.current_stream(dev).wait_stream(streams.compute_mha)  # default 等 MHA 流（安全）

    #             # ⭐⭐⭐ 在计算流中执行 MHA
    #             with torch.cuda.stream(streams.compute_mha):
    #                 # 注意：预取逻辑已移至 FFN 阶段（避免重复预取和 OOM）
    #                 attn_in  = self.attention_norm(x)
    #                 attn_out = self.attention(attn_in, start_pos, freqs_complex)  # 在 compute_mha 上排队
    #             # 在 MHA 流记录一个事件，供 FFN 流等待
    #             mha_eid, mha_evt = None, None
    #             try:
    #                 from llama3 import stream_mnt
    #                 mha_eid, mha_evt = stream_mnt.record_event_on(streams.compute_mha, device=dev)
    #             except Exception:
    #                 mha_evt = torch.cuda.Event()
    #                 mha_evt.record(streams.compute_mha)
    #         else:
    #             # 回退到默认流（不推荐，但保证可运行）
    #             # 注意：预取逻辑已移至 FFN 阶段（避免重复预取和 OOM）
    #             attn_out = self.attention(self.attention_norm(x), start_pos, freqs_complex)

    #         # 在 MHA 完成之前不要在默认流上消费 attn_out；先做残差也放到 MHA 流里
    #         if streams and streams.compute_mha:
    #             with torch.cuda.stream(streams.compute_mha):
    #                 h = x + attn_out
    #         else:
    #             h = x + attn_out
    #         nvtx.range_pop()  # attention

    #         # -------- FFN 阶段：只做"事件依赖"，同时前置预取 L+1 的 ATTN --------
    #         if wm is not None:
    #             # REMOVED: wm.ensure_group_on_gpu(self.layer_id, "ffn")
    #             if streams and streams.compute_ffn and hasattr(wm, "wait_group_ready"):
    #                 wm.wait_group_ready(self.layer_id, "ffn", compute_stream=streams.compute_ffn)  # NEW: 纯事件挂载
    #             elif hasattr(wm, "wait_group_ready"):
    #                 wm.wait_group_ready(self.layer_id, "ffn", compute_stream=None)                # NEW

    #         # 让 FFN 流等待 MHA 事件（只挂事件，不同步 CPU）
    #         if streams and streams.compute_ffn and 'mha_evt' in locals():
    #             streams.compute_ffn.wait_event(mha_evt)

    #         nvtx.range_push(f"layer_{self.layer_id}_ffn")
    #         if streams and streams.compute_ffn:
    #             with torch.cuda.stream(streams.compute_ffn):

    #                 # NEW ⭐ 在 L 的 FFN 计算期间，启动 L+1 的 ATTN 预取（高优先级/加 pin）
    #                 # 但先检查 GPU 剩余容量，避免过度预取导致 OOM
    #                 if wm is not None and hasattr(wm, "prefetch_group_async"):
    #                     nxt = self.layer_id + 1
    #                     if nxt < self.n_layer:
    #                         # 预算检查：只有在 GPU 未满时才预取
    #                         gpu_count = len(getattr(wm, "_gpu_group_lru", []))
    #                         gpu_limit = int(os.getenv("WSM_GPU_MAX_GROUPS", "10"))
    #                         # 留 2 个位置给当前层 FFN + 未来清理
    #                         if gpu_count + 2 < gpu_limit:
    #                             try:
    #                                 wm.prefetch_group_async(nxt, "attn", pin=True, priority="high")
    #                             except TypeError:
    #                                 # 兼容老签名
    #                                 wm.prefetch_group_async(nxt, "attn")

    #                 # 原 FFN 计算
    #                 ffn_in  = self.ffn_norm(h)
    #                 ffn_out = self.feed_forward(ffn_in)   # 在 compute_ffn 上排队
    #                 out     = h + ffn_out

    #             # 默认流等待 FFN 完成事件（仅事件）
    #             ffn_evt = torch.cuda.Event()
    #             ffn_evt.record(streams.compute_ffn)
    #             torch.cuda.current_stream(dev).wait_event(ffn_evt)
    #         else:
    #             # 回退到默认流（不推荐，但保证可运行）
    #             # 注意：预取逻辑已整合到上方 compute_ffn 分支（避免重复）
    #             out = h + self.feed_forward(self.ffn_norm(h))

    #         # NEW ⭐ 在 FFN 结束处：预测并预拉"下一层"需要的 KV blocks（异步 H2D）
    #         try:
    #             offloader = getattr(self.attention, "offloader", None)
    #             kv_stream = getattr(self.streams, "kv_h2d", None)
    #             nxt = self.layer_id + 1
    #             if (offloader is not None) and (nxt < self.n_layer) and (kv_stream is not None):
    #                 # window_tokens：优先取 offloader.block_size；否则使用一个安全默认值
    #                 window = int(getattr(offloader, "block_size", 256))
    #                 seqlen = int(x.size(1))
    #                 blocks = offloader.plan_tail_window_blocks(start_pos, seqlen, window_tokens=window)
    #                 if hasattr(offloader, "prefetch_blocks_async"):
    #                     offloader.prefetch_blocks_async(nxt, blocks, stream=kv_stream)   # 事件会在 KV H2D 上记录
    #                 else:
    #                     # 兼容：用已有的"下一层预取"API
    #                     offloader.prefetch_for_next_layer(nxt, start_pos, seqlen, D=1)
    #         except Exception:
    #             pass

    #         nvtx.range_pop()  # ffn

    #         # 清理 MHA 事件
    #         if streams and streams.compute_mha and 'mha_eid' in locals() and mha_eid is not None:
    #             try:
    #                 from llama3 import stream_mnt
    #                 stream_mnt.release_event(mha_eid, device=dev)
    #             except Exception:
    #                 pass

    #     nvtx.range_pop()  # layer_forward

    #     self.forward_count += 1
    #     self.total_forward_time += time.time() - forward_start

    #     # 周期性 GC 事件池
    #     self._gc_counter += 1
    #     if self._gc_counter % 10 == 0:
    #         try:
    #             from llama3 import stream_mnt
    #             stream_mnt.gc_event_pool(device=dev, force=False)
    #         except Exception:
    #             pass

    #     return out

    
    # def forward_async(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
    #     """
    #     异步前向：立即返回 Future，不阻塞调用线程。
    #     语义等价于 forward()，但内部把 MHA/FFN 放在各自 compute 流，并仅用"事件"建立依赖。

    #     Returns:
    #         Future[torch.Tensor]: 异步结果，调用 .result() 时会等待计算完成
    #     """
    #     from concurrent.futures import Future

    #     # ---- 设备与 dtype 协调（复用 forward() 里的逻辑）----
    #     dev = str(self.device)
    #     if not dev.startswith("cuda"):
    #         try:
    #             dev = str(self.attention.wq.weight.device)
    #             self.device = dev
    #         except Exception:
    #             pass
    #     dtype = getattr(self, "param_dtype", torch.bfloat16)

    #     if x.device != dev or x.dtype != dtype:
    #         x = x.to(device=dev, dtype=dtype, non_blocking=True)
    #     if freqs_complex.device != dev or freqs_complex.dtype != dtype:
    #         freqs_complex = freqs_complex.to(device=dev, non_blocking=True)

    #     # ---- 通知 WSM：推进 CPU 窗口（滚动模式/环形窗下立即入队缺失层）----
    #     wm = getattr(self, "weight_manager", None)
    #     if wm is not None:
    #         # 轻推进：刷新保留/PD 窗口
    #         if hasattr(wm, "note_compute_advance"):
    #             wm.note_compute_advance(self.layer_id)  # 轻量更新+窗口估计
    #         # 强推进：把缺失层入队到 CPU 预取线程（不在当前线程做 SSD 同步 IO）
    #         if hasattr(wm, "_advance_cpu_window_by_compute"):
    #             wm._advance_cpu_window_by_compute(self.layer_id)

    #     streams = getattr(self, "streams", None)
    #     compute_mha = getattr(streams, "compute_mha", None) if streams else None
    #     compute_ffn = getattr(streams, "compute_ffn", None) if streams else None

    #     # ---- 1) 在 compute_mha 上排 MHA，并记录事件 ----
    #     if wm is not None:
    #         # 保留 ensure_group_on_gpu 以确保权重已加载
    #         if hasattr(wm, "ensure_group_on_gpu"):
    #             wm.ensure_group_on_gpu(self.layer_id, "attn")
    #         if hasattr(wm, "wait_group_ready"):   # 纯事件依赖，绝不 CPU 同步
    #             wm.wait_group_ready(self.layer_id, "attn", compute_stream=compute_mha)

    #     if compute_mha:
    #         with torch.cuda.stream(compute_mha):
    #             attn_in  = self.attention_norm(x)
    #             attn_out = self.attention(attn_in, start_pos, freqs_complex)
    #             h = x + attn_out
    #             mha_done = torch.cuda.Event()
    #             mha_done.record(compute_mha)
    #     else:
    #         attn_in  = self.attention_norm(x)
    #         attn_out = self.attention(attn_in, start_pos, freqs_complex)
    #         h = x + attn_out
    #         mha_done = torch.cuda.Event()
    #         mha_done.record(torch.cuda.current_stream())

    #     # ---- 趁 MHA/FFN 进行时，异步预取后续组（L+1 的 attn 等）----
    #     try:
    #         if wm is not None and hasattr(wm, "prefetch_group_async"):
    #             nxt = self.layer_id + 1
    #             if nxt < self.n_layer:
    #                 # 预算检查：避免 OOM
    #                 gpu_count = len(getattr(wm, "_gpu_group_lru", []))
    #                 gpu_limit = int(os.getenv("WSM_GPU_MAX_GROUPS", "10"))
    #                 if gpu_count + 2 < gpu_limit:
    #                     wm.prefetch_group_async(nxt, "attn")   # 下一层ATTN
    #     except Exception:
    #         pass

    #     # ---- 2) 在 compute_ffn 上排 FFN，等待 MHA 事件 ----
    #     if wm is not None:
    #         if hasattr(wm, "ensure_group_on_gpu"):
    #             wm.ensure_group_on_gpu(self.layer_id, "ffn")
    #         if hasattr(wm, "wait_group_ready"):
    #             wm.wait_group_ready(self.layer_id, "ffn", compute_stream=compute_ffn)

    #     if compute_ffn:
    #         with torch.cuda.stream(compute_ffn):
    #             compute_ffn.wait_event(mha_done)
    #             ffn_in  = self.ffn_norm(h)
    #             ffn_out = self.feed_forward(ffn_in)
    #             out     = h + ffn_out
    #             ffn_done = torch.cuda.Event()
    #             ffn_done.record(compute_ffn)
    #     else:
    #         torch.cuda.current_stream().wait_event(mha_done)
    #         ffn_in  = self.ffn_norm(h)
    #         ffn_out = self.feed_forward(ffn_in)
    #         out     = h + ffn_out
    #         ffn_done = torch.cuda.Event()
    #         ffn_done.record(torch.cuda.current_stream())

    #     # ---- 3) 在 FFN 尾部预取下一层的 KV（可选：仅解码阶段有效）----
    #     try:
    #         offloader = getattr(self.attention, "offloader", None)
    #         kv_stream = getattr(self.streams, "kv_h2d", None)
    #         nxt = self.layer_id + 1
    #         if (offloader is not None) and (nxt < self.n_layer) and (kv_stream is not None):
    #             window = int(getattr(offloader, "block_size", 256))
    #             seqlen = int(x.size(1))
    #             blocks = offloader.plan_tail_window_blocks(start_pos, seqlen, window_tokens=window)
    #             if hasattr(offloader, "prefetch_blocks_async"):
    #                 offloader.prefetch_blocks_async(nxt, blocks, stream=kv_stream)
    #     except Exception:
    #         pass

    #     # ---- 4) 返回 Future：在一个轻线程里把"默认流等待FFN事件 + set_result"做完 ----
    #     fut: Future = _get_executor().submit(self._finalize_and_return,
    #                                          out, ffn_done, dev)
    #     return fut

    def _finalize_and_return(self, out_tensor: torch.Tensor, done_evt: torch.cuda.Event, device: str):
        """
        在一个线程里把默认流与 FFN 完成事件建立依赖，然后返回 out_tensor。
        这样调用者在拿到 Future 时仍是非阻塞的；真正的"结果就绪"由事件保证。
        """
        with torch.cuda.device(device):
            cur = torch.cuda.current_stream()
            cur.wait_event(done_evt)  # 事件依赖，非 CPU 同步
        return out_tensor

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
