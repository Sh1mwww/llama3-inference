import contextlib
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

ATTN_MICRO_B = int(os.getenv("ATTN_MICRO_B", "0") or 0)


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
        # 全局累计（保持原来的字段，避免 break 现有逻辑）
        self.stats = {
            "weights_hbm_us": 0,
            "kv_fetch_us": 0,
            "attn_us": 0,
            "ffn_us": 0,
            "total_forward_us": 0,
            "memory_alloc_us": 0,
        }
        # 每层累计（跨整个 run）
        # layer_stats[layer_id][stat_name] = us
        self.layer_stats: Dict[int, Dict[str, float]] = {}

        # 新增：按 phase + step + layer 的细粒度统计
        # step_stats[phase]["prefill"/"decode"][step_idx][layer_id][stat_name] = us
        # 目前我们只会在 decoder 里设置 step_idx
        self.step_stats: Dict[str, Dict[int, Dict[int, Dict[str, float]]]] = {}

        # 当前 forward 调用的上下文（由 InferenceProfiler 设置）
        self.current_phase: Optional[str] = None  # "prefill" / "decode" / None
        self.current_step: Optional[int] = None   # decode token idx，prefill 时为 None

        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            for key in self.stats:
                self.stats[key] = 0
            self.layer_stats.clear()
            self.step_stats.clear()
            self.current_phase = None
            self.current_step = None

    def get_stats(self) -> Dict:
        with self.lock:
            # 返回 shallow copy，避免外面修改内部 dict
            return {
                "global": self.stats.copy(),
                "per_layer": {lid: stats.copy() for lid, stats in self.layer_stats.items()},
                "per_step": {
                    phase: {
                        step: {lid: s.copy() for lid, s in per_layer.items()}
                        for step, per_layer in steps.items()
                    }
                    for phase, steps in self.step_stats.items()
                },
            }

    def add_layer_stat(self, layer_id: int, stat_name: str, value: float):
        # 这个接口保持不动，给别的地方补充统计用
        with self.lock:
            if layer_id not in self.layer_stats:
                self.layer_stats[layer_id] = {}
            if stat_name not in self.layer_stats[layer_id]:
                self.layer_stats[layer_id][stat_name] = 0.0
            self.layer_stats[layer_id][stat_name] += float(value)

    def set_step_context(self, phase: Optional[str], step_idx: Optional[int]):
        """
        由 InferenceProfiler.wrap_model_forward 在每次 model.forward 入口/出口调用。

        phase: "prefill" / "decode" / None
        step_idx: decode 的 token index（0-based），prefill/非 decode 时为 None
        """
        with self.lock:
            self.current_phase = phase
            self.current_step = step_idx


PERF_TRACKER = PerformanceTracker()

# Profiling control via environment variable
PROFILE = os.getenv("LLM_PROFILE", "0") == "1"

@contextmanager
def cuda_timer(key: str, layer_id: Optional[int] = None):
    # No-op when profiling is disabled（避免任何同步开销）
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
                # 只同步当前流的事件，避免全局阻塞其它流（尤其是 H2D）
                end_event.synchronize()

                elapsed_us = int(start_event.elapsed_time(end_event) * 1000)

                with PERF_TRACKER.lock:
                    # 1) 全局统计（保持原来的逻辑）
                    if key not in PERF_TRACKER.stats:
                        PERF_TRACKER.stats[key] = 0
                    PERF_TRACKER.stats[key] += elapsed_us

                    # 2) 每层累计（保持原来的逻辑）
                    if layer_id is not None:
                        if layer_id not in PERF_TRACKER.layer_stats:
                            PERF_TRACKER.layer_stats[layer_id] = {}
                        if key not in PERF_TRACKER.layer_stats[layer_id]:
                            PERF_TRACKER.layer_stats[layer_id][key] = 0
                        PERF_TRACKER.layer_stats[layer_id][key] += elapsed_us

                        # 3) 新增：按 phase + step 细分（只在 current_phase/current_step 有值时记录）
                        phase = PERF_TRACKER.current_phase
                        step_idx = PERF_TRACKER.current_step
                        if phase is not None and step_idx is not None:
                            phase_dict = PERF_TRACKER.step_stats.setdefault(phase, {})
                            step_dict = phase_dict.setdefault(int(step_idx), {})
                            layer_dict = step_dict.setdefault(int(layer_id), {})
                            layer_dict[key] = layer_dict.get(key, 0.0) + float(elapsed_us)

            except Exception as e:
                logger.warning(f"Error in cuda_timer cleanup for {key}: {e}")
                # 不要在清理阶段再抛异常


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
    fc = fc.to(dtype=torch.complex64, device=x.device)          # (Lx,D/2)
    fc = fc.unsqueeze(0).unsqueeze(2)                           # (1,Lx,1,D/2)

    # ---- 分块处理以节省内存 ----
    # 当 batch size 大时，分块处理避免峰值内存过高
    chunk_size = 16  # 每次处理16个batch
    if B <= chunk_size:
        # 小batch直接处理
        x_ = x.to(torch.float32).reshape(B, Lx, H, D // 2, 2)       # (B,L,H,D/2,2)
        x_complex = torch.view_as_complex(x_)                       # (B,L,H,D/2)
        out = torch.view_as_real(x_complex * fc)                    # (B,L,H,D/2,2)
        out = out.reshape(B, Lx, H, D).to(dtype=x.dtype)
    else:
        # 大batch分块处理
        out_chunks = []
        for i in range(0, B, chunk_size):
            end_i = min(i + chunk_size, B)
            x_chunk = x[i:end_i]                                    # (chunk,L,H,D)
            x_chunk_ = x_chunk.to(torch.float32).reshape(end_i - i, Lx, H, D // 2, 2)
            x_chunk_complex = torch.view_as_complex(x_chunk_)
            out_chunk = torch.view_as_real(x_chunk_complex * fc)
            out_chunk = out_chunk.reshape(end_i - i, Lx, H, D).to(dtype=x.dtype)
            out_chunks.append(out_chunk)
        out = torch.cat(out_chunks, dim=0)

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
        
        self.attn_micro_batch = ATTN_MICRO_B
        
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
        """
        只在需要时，把当前层的 attn 组权重准备到 GPU 上。
        - 优先通过 get_group_ready_event 拿到 H2D 完成事件，并在 compute stream 上等待该事件
          （非阻塞 CPU）。
        - 若事件还没建立，则触发一次 on-demand 预取。
        - 仅在完全没事件、没预取的极端情况下，才回退到 wait_group_ready 的“阻塞”版本。
        """
        wm = getattr(self, "weight_manager", None)
        if wm is None:
            return

        compute_stream = getattr(self.streams, "compute_mha", None)
        if not torch.cuda.is_available():
            # CPU 模式直接返回
            return

        # 优先走事件+流依赖路径（非阻塞 CPU）
        evt = None
        if hasattr(wm, "get_group_ready_event"):
            try:
                evt = wm.get_group_ready_event(self.layer_id, "attn")
            except Exception:
                evt = None

        # 若还没有事件，说明该组可能还没开始 H2D，触发一次 on-demand 预取
        if evt is None and hasattr(wm, "prefetch_group_async"):
            try:
                wm.prefetch_group_async(self.layer_id, "attn", reason="ensure_weights_cuda")
                evt = wm.get_group_ready_event(self.layer_id, "attn")
            except Exception:
                evt = None

        # 如果拿到了事件：只在 compute stream 上等待事件（非阻塞 CPU）
        if evt is not None:
            s = compute_stream or torch.cuda.current_stream()
            try:
                s.wait_event(evt)
            except Exception:
                # 极端情况下 fallback 到阻塞版本
                evt = None

        # 兜底：完全没有事件时，用老的 wait_group_ready（可能阻塞）
        if evt is None and hasattr(wm, "wait_group_ready"):
            wm.wait_group_ready(self.layer_id, "attn", compute_stream=compute_stream)

        if os.getenv("WSM_ASSERT_AFTER_WAIT", "0") == "1":
            assert self.wq.weight.is_cuda and self.wq.weight.numel() > 0, \
                f"L{self.layer_id}.wq still stub after _ensure_weights_cuda()"


    def _kv_gather_after_prefetch(
        self,
        blocks,
        bsz: int,
        stream: Optional[torch.cuda.Stream] = None,
        window_tokens: Optional[int] = None,
        batch_offset: int = 0,
    ):
        """
        在完成 prefetch 之后，从 KVOffloader 把指定 blocks 的 K/V 拉回到 GPU。

        Args:
            blocks: 需要拉取的 block indices（list[int]）
            bsz: 本次要用的 batch 大小（micro-batch 大小）
            stream: 用于 H2D 的 CUDA stream
            window_tokens: 如果不为 None，则只保留最后 window_tokens 个 token
            batch_offset: 在 KVOffloader 里写入/读取的 batch 偏移
                          也就是这次 micro-batch 对应全局 batch 里的起始行号
        """
        if getattr(self, "offloader", None) is None:
            raise RuntimeError("SelfAttention._kv_gather_after_prefetch called without offloader")

        if stream is None and torch.cuda.is_available():
            streams = getattr(self, "streams", None)
            stream = getattr(streams, "kv_h2d", None) or torch.cuda.current_stream()

        # 规范化 blocks，去重排序
        blocks_t = torch.as_tensor(blocks, dtype=torch.long, device=self.device)
        uniq_blocks = blocks_t.unique(sorted=True)

        # 等待这些 block 的 H2D 事件完成（如果 prefetch 时已经设置）
        self.offloader.wait_blocks_ready(self.layer_id, uniq_blocks.tolist(), stream=stream)

        # 从 offloader.fetch 拉到 GPU 上
        with torch.cuda.stream(stream):
            k_full, v_full = self.offloader.fetch(
                layer=self.layer_id,
                blocks=uniq_blocks,
                batch_idx=0,
                bsz=bsz,
                stream=stream,
                batch_offset=batch_offset,   # ⭐ 关键：带上 batch_offset
            )

        # 可选窗口裁剪：仅 decode 使用
        if window_tokens is not None:
            kv_len = k_full.size(2)
            if kv_len > window_tokens:
                k_full = k_full[..., -window_tokens:, :].contiguous()
                v_full = v_full[..., -window_tokens:, :].contiguous()

        return k_full, v_full


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

    def forward_with_microbatch(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_complex: torch.Tensor,
        kv_cache=None,
        mask=None,
        batch_idx: int = 0,
    ) -> torch.Tensor:
        bsz = x.size(0)
        micro_b = int(getattr(self, "attn_micro_batch", 0) or 0)

        if micro_b <= 0 or micro_b >= bsz:
            return self.forward(
                x,
                start_pos,
                freqs_complex,
                kv_cache=kv_cache,
                mask=mask,
                batch_idx=batch_idx,
                batch_offset=0,
            )

        out = torch.empty_like(x)
        offset = 0
        while offset < bsz:
            end = min(bsz, offset + micro_b)
            x_chunk = x[offset:end]

            out_chunk = self.forward(
                x_chunk,
                start_pos,
                freqs_complex,
                kv_cache=kv_cache,
                mask=mask,
                batch_idx=batch_idx,
                batch_offset=offset,  
            )
            out[offset:end] = out_chunk
            offset = end

        return out


    
    # def _forward_prefill_micro_batch(
    #     self,
    #     x: torch.Tensor,
    #     start_pos: int,
    #     freqs_complex: torch.Tensor,
    #     micro_b: int,
    # ) -> torch.Tensor:
    #     """
    #     只在 prefill 阶段(start_pos == 0)使用的按 batch 维 micro-batch 的 attention 实现。
    #     - 整个前向仍然算完 B 个样本的输出，
    #     - 但每次只在 GPU 上放 micro_b 个样本的 Q/K/V 和 attention，
    #       其余样本顺序串行，从而降低峰值显存。
    #     - KV 缓存在计算完每个 micro-batch 时写入 offloader（使用 batch_offset）。
    #     """
    #     assert start_pos == 0, "_forward_prefill_micro_batch 只用于 prefill"

    #     import torch.nn.functional as F
    #     from .global_state_tracker import get_global_tracker

    #     device = x.device
    #     dtype = x.dtype
    #     B, T, _ = x.shape

    #     if B <= micro_b:
    #         # B 本身就不大，没必要走这个分支
    #         raise RuntimeError("micro_b >= batch_size, 不应该进 _forward_prefill_micro_batch")

    #     offloader = getattr(self, "offloader", None)
    #     if offloader is None:
    #         raise RuntimeError("需要 KVOffloader 才有意义（要把 prefill 的 KV 存起来）")

    #     # 当前“批次 id”用于 global tracker 统计（不是 batch 维索引）
    #     tracker = get_global_tracker()
    #     if tracker:
    #         batch_idx = tracker.current_batch
    #     else:
    #         batch_idx = 0

    #     # 确保本层 attn 权重已经在 CUDA 上（走你原来封装好的逻辑）
    #     self._ensure_weights_cuda()

    #     # 结果缓冲区：最终返回 [B, T, dim]
    #     out = torch.empty(B, T, self.n_heads_q * self.head_dim, dtype=dtype, device=device)

    #     # 是否使用 causal mask
    #     is_causal = getattr(self, "apply_causal_mask", True)

    #     # 为了用 Flash Attention，统一转成 (B, H, T, D)
    #     def _repeat_kv_for_q_heads(k_or_v: torch.Tensor) -> torch.Tensor:
    #         # k_or_v: [mb, T, n_kv_heads, head_dim]
    #         k_or_v = k_or_v.transpose(1, 2).contiguous()  # [mb, n_kv_heads, T, D]
    #         if self.n_rep != 1:
    #             # 重复 KV head 到 Q head
    #             k_or_v = k_or_v.repeat_interleave(self.n_rep, dim=1)  # [mb, n_heads_q, T, D]
    #         return k_or_v

    #     # 主循环：沿 batch 维分块
    #     for b0 in range(0, B, micro_b):
    #         b1 = min(B, b0 + micro_b)
    #         mb = b1 - b0
    #         xb = x[b0:b1]                     # [mb, T, dim]

    #         # -------- Q/K/V 投影 + RoPE（只在这一小块上）--------
    #         # 注意：这里不再一次性对全 B 做 Q/K/V，从而避免大 tensor
    #         q = self.wq(xb).view(mb, T, self.n_heads_q, self.head_dim)
    #         k = self.wk(xb).view(mb, T, self.n_kv_heads, self.head_dim)
    #         v = self.wv(xb).view(mb, T, self.n_kv_heads, self.head_dim)

    #         # RoPE：仍然用你文件里已经优化过的 apply_rotary_embeddings（它内部有 B 维 chunk）
    #         q = apply_rotary_embeddings(q, freqs_complex, start_pos=start_pos)
    #         k = apply_rotary_embeddings(k, freqs_complex, start_pos=start_pos)

    #         # -------- 把当前 micro-batch 的 KV 写入 offloader（用 batch_offset=b0）--------
    #         # 这样每个样本的 KV 都被放在 [batch_offset : batch_offset+mb) 这一段，
    #         # 整体 pinned KV 的 batch 维大小仍然是 max_batch（比如 64/128）
    #         for t in range(T):
    #             token_idx = start_pos + t
    #             blk_idx   = token_idx // self.block_sz
    #             offloader.push(
    #                 layer=self.layer_id,
    #                 blk=blk_idx,
    #                 k=k[:, t, :, :],           # [mb, n_kv_heads, head_dim]
    #                 v=v[:, t, :, :],
    #                 token_idx=token_idx,
    #                 batch_idx=batch_idx,       # 统计用
    #                 batch_offset=b0,           # 关键：在 KV 中的 batch 偏移
    #             )

    #         # -------- 注意力计算（仅用当前 micro-batch 的 Q / K / V）--------
    #         # 这里不从 offloader 取 KV，而是直接用刚算出的 k/v，
    #         # 因为 prefill 阶段 K/V 就是当前层刚算出来的值
    #         q_t = q.transpose(1, 2)                      # [mb, n_heads_q, T, D]
    #         k_full = _repeat_kv_for_q_heads(k)           # [mb, n_heads_q, T, D]
    #         v_full = _repeat_kv_for_q_heads(v)           # [mb, n_heads_q, T, D]

    #         # Flash Attention（和你原来的实现保持一致）
    #         from contextlib import nullcontext
    #         try:
    #             from torch.backends.cuda import sdp_kernel as sdpa_kernel
    #             sdpa_ctx = sdpa_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True)
    #         except Exception:
    #             sdpa_ctx = nullcontext()

    #         with sdpa_ctx:
    #             attn_out = torch.nn.functional.scaled_dot_product_attention(
    #                 q_t, k_full, v_full,
    #                 attn_mask=None,
    #                 dropout_p=0.0,
    #                 is_causal=is_causal,
    #             )  # [mb, n_heads_q, T, D]

    #         # 回到 [mb, T, dim]
    #         attn_out = attn_out.transpose(1, 2).contiguous().view(mb, T, self.n_heads_q * self.head_dim)
    #         out[b0:b1] = self.wo(attn_out)

    #     return out

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_complex: torch.Tensor,
        kv_cache=None,
        mask=None,
        batch_idx: int = 0,
        batch_offset: int = 0,
    ) -> torch.Tensor:

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

        # 使用 CUDA timer 来准确测量计算时间（仅在 PROFILE 模式下）
        forward_start_event = None
        forward_end_event = None
        if PROFILE and torch.cuda.is_available():
            try:
                forward_start_event = torch.cuda.Event(enable_timing=True)
                forward_end_event = torch.cuda.Event(enable_timing=True)
                forward_start_event.record()
            except Exception:
                forward_start_event = None
                forward_end_event = None

        assert x.dim()==3, f"x dim={x.dim()}, shape={x.shape}"
        bsz, seqlen, _ = x.shape
        
        # micro_env = os.getenv("ATTN_MICRO_B", "0").strip()
        # micro_b = int(micro_env) if micro_env.isdigit() else 0

        # # 只在 prefill（start_pos==0）且 B > micro_b 时启用
        # if start_pos == 0 and micro_b > 0 and micro_b < bsz:
        #     return self._forward_prefill_micro_batch(
        #         x=x,
        #         start_pos=start_pos,
        #         freqs_complex=freqs_complex,
        #         micro_b=micro_b,
        #     )
        is_prefill = seqlen > 1
        is_decode  = (seqlen == 1 and start_pos > 0)


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
        # 1) ⭐⭐⭐ P0 FIX: 先预取(异步) → 再等待(事件依赖)，实现IO/计算overlap
        # ============================================================
        wm = getattr(self, "weight_manager", None)
        in_use = False
        try:
            if wm and hasattr(wm, "_mark_group_in_use"):
                wm._mark_group_in_use(self.layer_id, "attn")
                in_use = True

            # ⭐⭐⭐ STEP 1: 立即发起异步预取（不等待，让H2D与后续计算overlap）
            # 这是关键优化：预取必须在wait_event之前，否则零overlap
            try:
                if wm is not None and hasattr(wm, "prefetch_group_async"):
                    # (1) 配对预取：当前层的 FFN（将与当前attn计算overlap）
                    wm.prefetch_group_async(self.layer_id, "ffn", reason="pair")
                    # (2) 向前预取：L+1 .. L+D 的 ATTN（将与后续计算overlap）
                    D  = max(1, int(os.getenv("WSM_GROUP_PREFETCH_DEPTH", "3")))
                    nL = getattr(wm, "n_layers", 0)
                    for off in range(1, min(D+1, nL - self.layer_id)):
                        wm.prefetch_group_async(self.layer_id + off, "attn", reason="window_ahead")
            except Exception:
                pass

            # ⭐⭐⭐ STEP 2 & 3: 确保当前attn权重就绪（强制等待，保证权重在GPU）
            # 优化：先尝试直接取事件建立依赖，极端情况再兜底 wait_group_ready
            stream = self.compute_stream or torch.cuda.current_stream()
            evt = None
            try:
                if wm is not None and hasattr(wm, "get_group_ready_event"):
                    evt = wm.get_group_ready_event(self.layer_id, "attn")
            except Exception:
                evt = None

            # if evt is not None:
            #     stream.wait_event(evt)  # 非阻塞，仅建立依赖
            # else:
            #     # 极端兜底：沿用原 wait_group_ready（内部同样是 wait_event，但会触发补救逻辑）
            #     if wm is not None and hasattr(wm, "wait_group_ready"):
            #         wm.wait_group_ready(self.layer_id, "attn", compute_stream=stream)
            if evt is not None:
                stream.wait_event(evt)  # GPU 侧依赖
            else:
                # 没有事件 -> 触发一次主动预取，再取事件
                if wm is not None and hasattr(wm, "prefetch_group_async"):
                    try: wm.prefetch_group_async(self.layer_id, "attn", reason="late")
                    except Exception: pass
                if wm is not None and hasattr(wm, "get_group_ready_event"):
                    evt = wm.get_group_ready_event(self.layer_id, "attn")
                    if evt is not None:
                        stream.wait_event(evt)
                # 仍然取不到 -> 最终兜底的“强等待”
                if (evt is None) and wm is not None and hasattr(wm, "wait_group_ready"):
                    wm.wait_group_ready(self.layer_id, "attn", compute_stream=stream)

            # 绝不允许用到 stub/CPU 权重
            if (self.wq.weight.numel() == 0) or (self.wq.weight.device.type != "cuda"):
                if wm is not None and hasattr(wm, "wait_group_ready"):
                    wm.wait_group_ready(self.layer_id, "attn", compute_stream=stream)
                assert self.wq.weight.numel() > 0 and self.wq.weight.is_cuda, \
                    f"L{self.layer_id}.wq not ready after wait()"

            # 额外保护：在 ATTN 开始时预 pin 同层 FFN，避免缝隙被逐出
            if wm is not None and hasattr(wm, "pin_group"):
                try: wm.pin_group(self.layer_id, "ffn", reason="pair")
                except Exception: pass

            # ⭐ 可选：等待 KV 块 ready 事件（如果有预取）
            # 在 decode 阶段等待本层所需的 KV 块 H2D 完成
            if is_decode and self.offloader is not None and hasattr(self.offloader, "wait_blocks_ready"):
                # 计算本层需要的块：decode 时使用最近窗口 tokens
                decode_window_tokens = int(os.getenv("KV_DECODE_WINDOW_TOKENS", str(getattr(self, "_win_tokens", 128))))
                blocks = self.offloader.plan_tail_window_blocks(start_pos, seqlen, window_tokens=decode_window_tokens)
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

            # ⭐ KV prefetch (仅保留 KV offloading 相关的预取)
            try:
                # --- KV：为"下一层注意力"预拉 KV 到 HBM ---
                # decode: 预取窗口; prefill: 不预取（避免和 push/写盘打架）
                if (start_pos > 0) and getattr(self, "offloader", None) is not None:
                    decode_window_tokens = int(os.getenv("KV_DECODE_WINDOW_TOKENS", str(getattr(self, "_win_tokens", 128))))
                    # 该方法会：必要时先 SSD->DRAM，再在 kv_h2d 流发起 DRAM->HBM；并记录事件，fetch() 将命中
                    # prefetch_for_next_layer 内部已经做了 is_decode 判断，只在 decode 时才预取
                    self.offloader.prefetch_for_next_layer(
                        current_layer=self.layer_id,
                        start_pos=int(start_pos),
                        seqlen=int(seqlen),
                        bsz=int(bsz),
                        window_tokens=decode_window_tokens,
                        is_decode=is_decode,
                    )
            except Exception as e:
                # 非致命：任何预取异常都不影响主计算路径
                wm = getattr(self, "weight_manager", None)
                if wm and getattr(wm, "verbose", False):
                    print(f"[ATTN][L{self.layer_id}] KV prefetch skipped: {e}")

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
                    t_in_blk  = token_idx % self.block_sz

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
                        batch_offset=batch_offset, 
                    )

                    if is_decode:
                        self.offloader.append_token_to_gpu(
                            layer=self.layer_id,
                            blk=blk_idx,
                            t_in_blk=t_in_blk,
                            k=k_curr,
                            v=v_curr
                        )

            # ------------------------- (B) 选择并取回需要的 blocks -------------------------
            blk_idx = start_pos // self.block_sz

            # ⭐ Decode 取块策略：使用"尾窗"而非 Top-K
            # 原因：Flash Attention 不返回 attention weights，无法更新真实的 importance scores
            # 改用确定性的"最近 N tokens"窗口策略，更稳定且可预测
            nvtx.range_push(f"layer_{self.layer_id}_kv_fetch")
            with cuda_timer("kv_fetch_us", self.layer_id):
                do_profile_gpu = bool(self.enable_profiling and x.is_cuda)
                if do_profile_gpu:
                    fetch_evt_start = torch.cuda.Event(enable_timing=True)
                    fetch_evt_end = torch.cuda.Event(enable_timing=True)
                    fetch_evt_start.record()

                if getattr(self, "offloader", None) is not None and is_decode:
                    # ⭐ 只在真正 decode 时才根据显存估算 window size 并 fetch KV
                    if not hasattr(self, "_win_tokens"):
                        # 显存预算（默认 20% 可调）：KV 每 token 字节 = bsz * heads_kv * dim * 2(dtypebytes) * 2(K+V)
                        try:
                            free_bytes, _ = torch.cuda.mem_get_info(x.device)
                        except Exception:
                            free_bytes = 2 * (1024 ** 3)  # 回退到 2GB

                        frac = float(os.getenv("KV_DECODE_WINDOW_FRAC", "0.20"))
                        budget = int(free_bytes * max(0.1, min(frac, 0.9)))
                        dtype_bytes = getattr(self.offloader, "dtype_bytes", 2)
                        per_tok = int(bsz * self.n_kv_heads * self.head_dim * dtype_bytes * 2)
                        layers = int(getattr(self.offloader, "layers", 1))
                        tokens = max(128, min((budget // max(1, per_tok * layers)), 8192))
                        self._win_tokens = (tokens // BLOCK) * BLOCK

                        if os.getenv("DEBUG_KV_WINDOW", "0") == "1":
                            print(
                                f"[KV][L{self.layer_id}] Init decode window tokens = {self._win_tokens}, "
                                f"free_bytes={free_bytes/1e9:.2f}GB, per_token≈{per_tok/1024:.1f}KB"
                            )

                    decode_window_tokens = int(os.getenv("KV_DECODE_WINDOW_TOKENS", str(getattr(self, "_win_tokens", 128))))

                    # ✅ 真正 decode：只取尾部 window
                    blocks = self.offloader.plan_tail_window_blocks(
                            start_pos=start_pos,
                            seqlen=seqlen,
                            window_tokens=decode_window_tokens
                        )

                    # ⭐ 统一在 _kv_gather_after_prefetch 中处理窗口裁剪
                    # 传入 window_tokens 让 fetch 逻辑直接返回裁剪好的 KV
                    k_full, v_full = self._kv_gather_after_prefetch(
                        blocks, bsz,
                        window_tokens=decode_window_tokens,
                        batch_offset=batch_offset,
                    )
                else:
                    # prefill 或无 offloader：直接使用当前序列的 K/V（转为 (B,H,T,D)）
                    # prefill 时不 fetch，避免和 push/写盘的状态打架
                    k_full = k.transpose(1, 2).contiguous()
                    v_full = v.transpose(1, 2).contiguous()
                    blocks = [blk_idx]  # 为后续 update_importances 提供 blocks 列表

                if do_profile_gpu:
                    fetch_evt_end.record()
                    # if not fetch_evt_end.query():
                    #     fetch_evt_end.synchronize()
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

            # ⭐ 窗口裁剪已统一在 _kv_gather_after_prefetch 中处理，此处不再重复裁剪

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
                    # 仅保留事件计时，避免与 cuda_timer 重复；且不在热路径同步
                    if do_profile_gpu:
                        attn_evt_start = torch.cuda.Event(enable_timing=True)
                        attn_evt_end = torch.cuda.Event(enable_timing=True)
                        attn_evt_start.record()

                    with sdpa_ctx:
                        out = torch.nn.functional.scaled_dot_product_attention(
                            q, k_full, v_full, attn_mask=None, dropout_p=0.0, is_causal=is_causal
                        )

                    # 立即释放确定存在的中间激活
                    del q, k_full, v_full
                    try: del k
                    except NameError: pass
                    try: del v
                    except NameError: pass

                    if do_profile_gpu:
                        attn_evt_end.record()
                        PERF_TRACKER.add_event_pair(self.layer_id, "attn_us_evt", attn_evt_start, attn_evt_end)

                    # ✅ 后处理仍在同一流: [B,H,Tq,D] -> [B,Tq,H,D] -> [B,Tq,feat]
                    out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
                    feat = self.n_heads_q * self.head_dim
                    assert out.size(-1) == feat

                    # 线性层: 2D -> Linear -> 3D
                    w = self.wo.weight
                    out2d = out.view(-1, feat).contiguous()
                    if out2d.dtype != w.dtype:
                        out2d = out2d.to(w.dtype)
                    if out2d.device != w.device:
                        out2d = out2d.to(w.device, non_blocking=True)

                    res2d = self.wo(out2d)
                    result = res2d.view(bsz, seqlen, -1).contiguous()
                    del out, out2d, res2d

                nvtx.range_pop()  # attention_compute
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

            # --- 统计收尾：使用 CUDA timer 准确测量计算时间 ---
            if forward_end_event is not None:
                forward_end_event.record()
                try:
                    forward_end_event.synchronize()
                    total_time = forward_start_event.elapsed_time(forward_end_event) * 1000  # ms -> μs
                    PERF_TRACKER.add_layer_stat(self.layer_id, "total_forward_us", total_time)
                except Exception as e:
                    logger.warning(f"Failed to record total_forward_us for layer {self.layer_id}: {e}")
            # print(f"[ATTN] Layer {self.layer_id} computation done")

            if wm and hasattr(wm, "notify_group_compute_done"):
                evt = torch.cuda.Event()
                evt.record(self.compute_stream if self.compute_stream is not None else torch.cuda.current_stream())
                wm.notify_group_compute_done(self.layer_id, "attn", evt)

            # ============================================================
            # 3.2) Eager KV Spill: prefill 阶段（含 chunk prefill）将本层生成的 KV 异步写入 SSD
            # chunk prefill 也会逐步推进 eager spill，保证 KV 会被及时写出到 SSD，不挤爆 CPU/GPU
            # ============================================================
            if self.offloader is not None and os.getenv("KV_EAGER_SPILL", "1") != "0":
                if is_prefill:
                    try:
                        # 把本层刚生成的 KV 覆盖到的 token 全部甩到 SSD
                        # upto_token = start_pos + seqlen 表示当前层已处理到的 token 位置
                        self.offloader.eager_spill_layer(
                            self.layer_id,
                            upto_token=start_pos + seqlen,
                            async_write=True
                        )
                    except Exception as e:
                        if os.getenv("DEBUG_KV_WINDOW", "0") == "1":
                            print(f"[KV][WARN] eager_spill_layer failed at layer {self.layer_id}: {e}")

            # ============================================================
            # 3.3) Decode 阶段的快速 eviction：维持 decode window
            # 只在单 token decode 时做 window 内的 eviction，保持 tail window 大小稳定
            # ============================================================
            if getattr(self, "offloader", None) is not None and os.getenv("KV_EAGER_DECODE_EVICT", "1") != "0":
                if is_decode:
                    try:
                        # 每隔 N 个 token 调用一次（避免过于频繁；可配置）
                        spill_interval = int(os.getenv("KV_DECODE_SPILL_INTERVAL", "1024"))
                        current_token = start_pos + seqlen

                        # 简单策略：当 current_token 是 spill_interval 的倍数时触发下放
                        if current_token % spill_interval == 0:
                            # 保留最近 N 个 block 在 DRAM，其余下放到 SSD
                            keep_tail_blocks = int(os.getenv("KV_DECODE_KEEP_TAIL_BLOCKS", "2"))
                            self.offloader.eager_spill_decode_window(
                                upto_token=current_token,
                                keep_tail_blocks=keep_tail_blocks,
                                include_partial=False,  # 只下放完整块，避免反复写
                                layers=[self.layer_id]  # 只下放当前层
                            )
                    except Exception as e:
                        if os.getenv("DEBUG_KV_WINDOW", "0") == "1":
                            print(f"[KV][WARN] eager_spill_decode_window failed at layer {self.layer_id}: {e}")

            return result
        finally:
            # 对称解除：ATTN 阶段 pin 的配对 FFN
            # if wm is not None and hasattr(wm, "unpin_group"):
            #     wm.unpin_group(self.layer_id, "ffn")
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        if x.device.type != "cuda":
            raise RuntimeError(
                f"[FeedForward L{self.layer_id}] Input tensor must be on CUDA, got {x.device}"
            )

        # ============================================================
        # 使用 CUDA timer 来准确测量计算时间（仅在 PROFILE 模式下）
        # ============================================================
        forward_start_event = None
        forward_end_event = None
        if PROFILE and torch.cuda.is_available():
            try:
                forward_start_event = torch.cuda.Event(enable_timing=True)
                forward_end_event = torch.cuda.Event(enable_timing=True)
                forward_start_event.record()
            except Exception:
                forward_start_event = None
                forward_end_event = None

        # ============================================================
        # ⭐ 回绕通知 + 事件等待（彻底不兜底）
        # ============================================================
        wm = getattr(self, "weight_manager", None)

        # 没有 WSM 时直接执行计算
        if wm is None:
            micro = int(os.getenv("FFN_MICRO_B") )
            bsz = int(x.size(0))
            with cuda_timer("ffn_us", self.layer_id):
                if micro <= 0 or micro >= bsz:
                    gate = self.w1(x); up = self.w3(x)
                    gate = F.silu(gate, inplace=True); up.mul_(gate); del gate
                    out = self.w2(up); del up
                else:
                    out = torch.empty(bsz, x.size(1), self.w2.out_features,
                                      dtype=x.dtype, device=x.device)
                    for s in range(0, bsz, micro):
                        e = min(bsz, s + micro)
                        xi = x[s:e]
                        g  = self.w1(xi); u = self.w3(xi)
                        g  = F.silu(g, inplace=True); u.mul_(g); del g
                        out[s:e] = self.w2(u); del u

            # --- 统计收尾：使用 CUDA timer 准确测量计算时间 ---
            if forward_end_event is not None:
                forward_end_event.record()
                try:
                    forward_end_event.synchronize()
                    total_time = forward_start_event.elapsed_time(forward_end_event) * 1000  # ms -> μs
                    PERF_TRACKER.add_layer_stat(self.layer_id, "total_forward_us", total_time)
                except Exception as e:
                    logger.warning(f"Failed to record total_forward_us for FFN layer {self.layer_id}: {e}")

            return out

        in_use = False
        if hasattr(wm, "_mark_group_in_use"):
            wm._mark_group_in_use(self.layer_id, "ffn")
            in_use = True

        try:
            compute_stream = getattr(self.streams, "compute_ffn", None)

            # # ⭐⭐⭐ STEP 1: 先预取下一层（异步，与FFN计算overlap）
            # try:
            #     if wm is not None and hasattr(wm, "prefetch_group_async"):
            #         D  = max(1, int(os.getenv("WSM_GROUP_PREFETCH_DEPTH", "3")))
            #         nL = getattr(wm, "n_layers", 0)
            #         for off in range(1, min(D+1, nL - self.layer_id)):
            #             wm.prefetch_group_async(self.layer_id + off, "attn", reason="ffn_ahead")
            # except Exception:
            #     pass

            # ⭐⭐⭐ STEP 2 & 3: 确保FFN权重就绪（强制等待，保证权重在GPU）
            # 优化：先尝试直接取事件建立依赖，极端情况再兜底 wait_group_ready
            stream = compute_stream or torch.cuda.current_stream()
            evt = None
            try:
                if wm is not None and hasattr(wm, "get_group_ready_event"):
                    evt = wm.get_group_ready_event(self.layer_id, "ffn")
            except Exception:
                evt = None

            if evt is not None:
                stream.wait_event(evt)  # 非阻塞，仅建立依赖
            else:
                # 极端兜底：沿用原 wait_group_ready（内部同样是 wait_event，但会触发补救逻辑）
                if wm is not None and hasattr(wm, "wait_group_ready"):
                    wm.wait_group_ready(self.layer_id, "ffn", compute_stream=stream)

            # ---- Device alignment (no synchronous fallback) ----
            dev = self.w1.weight.device
            if x.device != dev:
                x = x.to(dev, non_blocking=True)

            # if compute_stream is not None:
            #     with torch.cuda.stream(compute_stream):
            #         gate = self.w1(x)
            #         up   = self.w3(x)
            #         gate = F.silu(gate, inplace=True)
            #         up.mul_(gate)
            #         del gate
            #         result = self.w2(up)
            #         del up
            # else:
            #     gate = self.w1(x)
            #     up   = self.w3(x)
            #     gate = F.silu(gate, inplace=True)
            #     up.mul_(gate)
            #     del gate
            #     result = self.w2(up)
            #     del up
            micro = int(os.getenv("FFN_MICRO_B") )
            bsz   = int(x.size(0))
            result = (torch.empty(bsz, x.size(1), self.w2.out_features,
                                  dtype=x.dtype, device=dev)
                      if (0 < micro < bsz) else None)
            stream_ctx = (torch.cuda.stream(compute_stream)
                          if compute_stream is not None else contextlib.nullcontext())
            with stream_ctx:
                with cuda_timer("ffn_us", self.layer_id):
                    if 0 < micro < bsz:
                        for s in range(0, bsz, micro):
                            e = min(bsz, s + micro)
                            xi = x[s:e]
                            g  = self.w1(xi); u = self.w3(xi)
                            g  = F.silu(g, inplace=True); u.mul_(g); del g
                            result[s:e] = self.w2(u); del u
                    else:
                        g = self.w1(x); u = self.w3(x)
                        g = F.silu(g, inplace=True); u.mul_(g); del g
                        result = self.w2(u); del u

            # --- 统计收尾：使用 CUDA timer 准确测量计算时间 ---
            if forward_end_event is not None:
                forward_end_event.record()
                try:
                    forward_end_event.synchronize()
                    total_time = forward_start_event.elapsed_time(forward_end_event) * 1000  # ms -> μs
                    PERF_TRACKER.add_layer_stat(self.layer_id, "total_forward_us", total_time)
                except Exception as e:
                    logger.warning(f"Failed to record total_forward_us for FFN layer {self.layer_id}: {e}")

            if hasattr(wm, "notify_group_compute_done"):
                evt = torch.cuda.Event()
                evt.record(compute_stream if compute_stream is not None else torch.cuda.current_stream())
                wm.notify_group_compute_done(self.layer_id, "ffn", evt)

            return result

        finally:
            if wm is not None and hasattr(wm, "unpin_group"):
                wm.unpin_group(self.layer_id, "ffn")  
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
        
        wm = getattr(self, "weight_manager", None)
        streams = getattr(self, "streams", None)

        if wm is not None and streams is not None:
            # 提前为本层 attn/ffn 获取“可能尚未完成”的 ready 事件
            try:
                evt_attn = wm.get_group_ready_event(self.layer_id, "attn")
                if evt_attn is not None and getattr(streams, "compute_mha", None) is not None:
                    streams.compute_mha.wait_event(evt_attn)   # 仅 GPU 侧依赖
            except Exception:
                pass
            try:
                evt_ffn = wm.get_group_ready_event(self.layer_id, "ffn")
                if evt_ffn is not None and getattr(streams, "compute_ffn", None) is not None:
                    streams.compute_ffn.wait_event(evt_ffn)   # 仅 GPU 侧依赖
            except Exception:
                pass

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

        # wm = getattr(self, "weight_manager", None)
        if wm is not None and hasattr(wm, "note_compute_advance"):
            wm.note_compute_advance(self.layer_id)

        if x.device.type != "cuda":
            raise RuntimeError(f"Layer {self.layer_id}: input activation must be on CUDA, got {x.device}")

        streams = self.streams
        device = x.device

        # -------- 1) MHA：只挂事件等待权重 + 可选的前置事件 --------
        # 优化：先尝试直接取事件建立依赖，极端情况再兜底 wait_group_ready
        mha_stream = getattr(streams, "compute_mha", None)
        evt_attn_fallback = None
        try:
            if wm is not None and hasattr(wm, "get_group_ready_event"):
                evt_attn_fallback = wm.get_group_ready_event(self.layer_id, "attn")
        except Exception:
            evt_attn_fallback = None

        if evt_attn_fallback is not None and mha_stream is not None:
            mha_stream.wait_event(evt_attn_fallback)  # 非阻塞，仅建立依赖
        else:
            # 极端兜底：沿用原 wait_group_ready（内部同样是 wait_event，但会触发补救逻辑）
            if wm is not None and hasattr(wm, "wait_group_ready"):
                wm.wait_group_ready(self.layer_id, "attn", compute_stream=mha_stream)

        if streams and streams.compute_mha:
            with torch.cuda.stream(streams.compute_mha):
                if copy_stream is not None:
                    streams.compute_mha.wait_stream(copy_stream)
                if freq_copy_stream is not None:
                    streams.compute_mha.wait_stream(freq_copy_stream)
                if wait_on is not None:
                    streams.compute_mha.wait_event(wait_on)

                attn_in = self.attention_norm(x)
                # attn_out = self.attention(attn_in, start_pos, freqs_complex)
                attn_out = self.attention.forward_with_microbatch(
                    attn_in, start_pos, freqs_complex
                )
            mha_eid, mha_evt = stream_mnt.record_event_on(streams.compute_mha, device=device)
        else:
            if wait_on is not None:
                torch.cuda.current_stream().wait_event(wait_on)
            # attn_out = self.attention(self.attention_norm(x), start_pos, freqs_complex)
            attn_out = self.attention.forward_with_microbatch(
                self.attention_norm(x), start_pos, freqs_complex
            )
            mha_eid, mha_evt = None, None

        # 残差（在 MHA 流完成后）
        if streams and streams.compute_mha and mha_evt is not None:
            with torch.cuda.device(device):
                torch.cuda.current_stream().wait_event(mha_evt)
        h = x
        h.add_(attn_out)
        del attn_out

        # -------- 2) FFN：只挂事件等待权重；FFN 流等待 MHA 事件 --------
        # 优化：先尝试直接取事件建立依赖，极端情况再兜底 wait_group_ready
        ffn_stream = getattr(streams, "compute_ffn", None)
        evt_ffn_fallback = None
        try:
            if wm is not None and hasattr(wm, "get_group_ready_event"):
                evt_ffn_fallback = wm.get_group_ready_event(self.layer_id, "ffn")
        except Exception:
            evt_ffn_fallback = None

        if evt_ffn_fallback is not None and ffn_stream is not None:
            ffn_stream.wait_event(evt_ffn_fallback)  # 非阻塞，仅建立依赖
        else:
            # 极端兜底：沿用原 wait_group_ready（内部同样是 wait_event，但会触发补救逻辑）
            if wm is not None and hasattr(wm, "wait_group_ready"):
                wm.wait_group_ready(self.layer_id, "ffn", compute_stream=ffn_stream)

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

        # -------- 3) （可选）预取 L+1 的 KV --------
        try:
            offloader = getattr(self.attention, "offloader", None)
            kv_stream = getattr(streams, "kv_h2d", None)
            nxt = self.layer_id + 1
            if (offloader is not None) and (nxt < self.n_layer) and (kv_stream is not None):
                seqlen = int(x.size(1))
                is_decode = (seqlen == 1 and start_pos > 0)

                if is_decode:
                    # decode: 预取窗口
                    window = int(getattr(offloader, "block_size", 256))
                    blocks = offloader.plan_tail_window_blocks(start_pos, seqlen, window_tokens=window)
                    if hasattr(offloader, "prefetch_blocks_async") and blocks:
                        bsz = int(x.size(0))
                        offloader.prefetch_blocks_async(nxt, blocks, bsz=bsz, stream=kv_stream)
                # else:  prefill/chunk-prefill 不做 KV 预取，避免和 push/写盘的状态打架
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
            # ⭐ 调试日志：验证 MHA 流被使用
            if os.getenv("DEBUG_STREAM_USAGE", "0") == "1":
                print(f"[Layer {self.layer_id}] MHA using stream: {streams.compute_mha}")
            with torch.cuda.stream(streams.compute_mha):
                attn_in  = self.attention_norm(x)
                # attn_out = self.attention(attn_in, start_pos, freqs_complex)
                attn_out = self.attention.forward_with_microbatch(
                    x=attn_in, start_pos=start_pos, freqs_complex=freqs_complex
                )
            # 记录 MHA 完成事件
            mha_eid, mha_evt = stream_mnt.record_event_on(streams.compute_mha, device=device)
        else:
            if os.getenv("DEBUG_STREAM_USAGE", "0") == "1":
                print(f"[Layer {self.layer_id}] MHA fallback to default stream (streams={streams})")
            # attn_out = self.attention(self.attention_norm(x), start_pos, freqs_complex)
            attn_out = self.attention.forward_with_microbatch(
                self.attention_norm(x), start_pos, freqs_complex
            )
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
            # ⭐ 调试日志：验证 FFN 流被使用
            if os.getenv("DEBUG_STREAM_USAGE", "0") == "1":
                print(f"[Layer {self.layer_id}] FFN using stream: {streams.compute_ffn}")
            with torch.cuda.stream(streams.compute_ffn):
                ffn_in   = self.ffn_norm(h)
                ffn_out  = self.feed_forward(ffn_in)
            # FFN 完成事件
            ffn_eid, ffn_evt = stream_mnt.record_event_on(streams.compute_ffn, device=device)
        else:
            if os.getenv("DEBUG_STREAM_USAGE", "0") == "1":
                print(f"[Layer {self.layer_id}] FFN fallback to default stream (streams={streams})")
            ffn_out = self.feed_forward(self.ffn_norm(h))
            ffn_eid, ffn_evt = None, None

        h.add_(ffn_out)
        del ffn_out
        out = h  # 最终残差复用了 x 的存储

        # -------- 3) （可选）在 FFN 期间预取 L+1、L+2 的 KV --------
        # 与 vLLM PagedAttention 思路一致：提前搬热点 KV 页到 HBM，减少解码步等待
        try:
            offloader = getattr(self.attention, "offloader", None)
            kv_stream = getattr(streams, "kv_h2d", None)
            if offloader is not None and kv_stream is not None:
                seqlen = int(x.size(1))
                is_decode = (seqlen == 1 and start_pos > 0)

                if is_decode:
                    for nxt in (self.layer_id + 1, self.layer_id + 2):
                        if nxt >= self.n_layer:
                            break
                        window = int(getattr(offloader, "block_size", 256))
                        blocks = offloader.plan_tail_window_blocks(
                            start_pos, seqlen, window_tokens=window
                        )
                        if hasattr(offloader, "prefetch_blocks_async") and blocks:
                            bsz = int(x.size(0))
                            offloader.prefetch_blocks_async(nxt, blocks, bsz=bsz, stream=kv_stream)
                # else:  prefill/chunk-prefill 不做 KV 预取，避免和 push/写盘的状态打架
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

        return out, ffn_evt
    


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
