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
                        PERF_TRACKER.add_layer_stat(layer_id, key, elapsed_us)
            except Exception as e:
                logger.warning(f"Error in cuda_timer cleanup for {key}: {e}")
                # Don't re-raise exceptions in cleanup

def set_weight_manager(manager):
    global WEIGHT_MANAGER
    WEIGHT_MANAGER = manager

def get_weight_manager(device: str):
    global WEIGHT_MANAGER
    return WEIGHT_MANAGER

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x: torch.Tensor):
        with cuda_timer("memory_alloc_us"):
            norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x * norm)

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
        
        # Linear权重初始化在CPU
        self.wq = nn.Linear(args.dim, self.n_heads_q * self.head_dim, bias=False, device="cpu")
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False, device="cpu")
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False, device="cpu")
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False, device="cpu")
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
        
        if not self.wq.weight.is_cuda:
            for mod in (self.wq, self.wk, self.wv, self.wo):
                mod.to(x.device, non_blocking=True)
        
        start_time = time.time()
        bsz, seqlen, _ = x.shape
        
        w_dtype = self.wq.weight.dtype
        if x.dtype != w_dtype:
            x = x.to(w_dtype)
        
        # 确保权重在GPU上
        self._ensure_weights_cuda()
        
        # 更新全局状态跟踪器
        tracker = get_global_tracker()
        # 獲取當前batch索引（從global tracker中讀取，而不是重新設置）
        if tracker:
            batch_idx = tracker.current_batch  # 使用tracker中已設置的值
        else:
            batch_idx = 0  # 回退值
        
        # 预取下一层权重
        if hasattr(self, '_next_layer_modules'):
            if self.weight_manager:
                self.weight_manager.prefetch_weights([self.layer_id + 1], {self.layer_id + 1: self._next_layer_modules})
        
        # 确保权重H2D传输完成后再开始计算（使用统一的事件化等待）
        if self.streams and x.is_cuda:
            try:
                self.streams.wait_weight_ready_on_current(self.device)
            except Exception as e:
                logger.warning(f"Layer {self.layer_id} SelfAttention: weight sync failed: {e}")

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
            
        #  batch
        if k_full.dim() == 3:
            # (seq_len, n_heads, head_dim) -> (1, n_heads, seq_len, head_dim)
            k_full = k_full.permute(1, 0, 2).unsqueeze(0)
            v_full = v_full.permute(1, 0, 2).unsqueeze(0)
        elif k_full.dim() == 4:
            # (bsz, seq_len, n_heads, head_dim) -> (bsz, n_heads, seq_len, head_dim)
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
        nvtx.range_pop()  # attention_compute
        out = out.transpose(1, 2).reshape(bsz, seqlen, -1)
        
        stats = PERF_TRACKER.layer_stats.get(self.layer_id, {})
        self.kv_elapsed_time = stats.get("kv_fetch_us", 0)
        self.attn_time       = stats.get("attn_us",     0)
        
        # Update block importances (only if offloader exists)
        if getattr(self, "offloader", None) is not None:
            with torch.no_grad():
                token_imp = attn_weights.mean(dim=(0, 1, 2))  # (Tkv,)
                block_scores = []
                for i, _ in enumerate(blocks):
                    s = i * self.block_sz
                    e = min(s + self.block_sz, token_imp.size(0))
                    score = float(token_imp[s:e].sum().item()) if s < token_imp.size(0) else 0.0
                    block_scores.append(score)

                self.offloader.update_importances(self.layer_id, blocks, block_scores, batch_idx=batch_idx)

        # 输出投影 - 使用compute stream
        if compute_stream:
            with torch.cuda.stream(compute_stream):
                result = self.wo(out)
        else:
            result = self.wo(out)

        total_time = (time.time() - start_time) * 1000000  # 转换为微秒
        PERF_TRACKER.add_layer_stat(self.layer_id, "total_forward_us", total_time)
        return result

# ---------- Optimized FeedForward ----------
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = int(4 * args.dim * 2 / 3)
        if args.ffn_dim_multiplier:
            hidden_dim = int(hidden_dim * args.ffn_dim_multiplier)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
    
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False, device="cpu")
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False, device="cpu")
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False, device="cpu")

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
        modules = self._get_modules_dict()
        if self.weight_manager:
            self.weight_manager.ensure_weights_cuda(self.layer_id, modules, priority=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check CUDA context health
        if x.is_cuda:
            try:
                torch.cuda.current_device()
            except RuntimeError as e:
                logger.error(f"CUDA context error in feedforward: {e}")
                raise RuntimeError("CUDA context is corrupted") from e

        self._ensure_weights_cuda()

        # 确保权重H2D传输完成后再开始计算（使用统一的事件化等待）
        if self.streams and x.is_cuda:
            try:
                self.streams.wait_weight_ready_on_current(self.device)
            except Exception as e:
                logger.warning(f"Layer {self.layer_id} FeedForward: weight sync failed: {e}")


        # FFN计算 - 使用compute stream
        # compute_stream = self.streams.weight_compute if self.streams else None
        compute_stream = getattr(self.streams, "compute_ffn", None) or getattr(self.streams, "weight_compute", None)
        if compute_stream:
            with torch.cuda.stream(compute_stream):
                with cuda_timer("ffn_us", self.layer_id):
                    # SwiGLU
                    gate = self.w1(x)
                    up = self.w3(x)
                    hidden = F.silu(gate) * up
                    result = self.w2(hidden)
            return result
        else:
            with cuda_timer("ffn_us", self.layer_id):
                # SwiGLU
                gate = self.w1(x)
                up = self.w3(x)
                hidden = F.silu(gate) * up
                return self.w2(hidden)

# ---------- Optimized EncoderBlock ----------
class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = args.n_layers

        self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
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

    def _ensure_weights_cuda(self):
        wm = getattr(self, "weight_manager", None)
        if wm is None:
            return
        modules = self._get_modules_dict()
        wm.ensure_weights_cuda(self.layer_id, modules, priority=True)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        forward_start = time.time()

        nvtx.range_push(f"layer_{self.layer_id}_forward")
        with cuda_timer("total_forward_us", self.layer_id):

            nvtx.range_push(f"layer_{self.layer_id}_attention")
            h = x + self.attention(self.attn_norm(x), start_pos, freqs_complex)
            nvtx.range_pop()  # attention

            # 确保attention计算完成后再开始FFN (使用事件化等待)
            if self.streams and self.streams.compute_mha:
                try:
                    from llama3 import stream_mnt
                    eid, evt = stream_mnt.record_event_on(self.streams.compute_mha, device=self.device)
                    stream_mnt.wait_event_on(torch.cuda.current_stream(self.device), evt)
                    stream_mnt.release_event(eid, device=self.device)
                except Exception as e:
                    logger.warning(f"Layer {self.layer_id}: MHA event sync failed ({e}), fallback to wait_stream")
                    torch.cuda.current_stream(self.device).wait_stream(self.streams.compute_mha)

            nvtx.range_push(f"layer_{self.layer_id}_ffn")
            out = h + self.feed_forward(self.ffn_norm(h))
            nvtx.range_pop()  # ffn

            # 确保FFN计算完成后再返回 (使用事件化等待)
            if self.streams and self.streams.compute_ffn:
                try:
                    from llama3 import stream_mnt
                    eid, evt = stream_mnt.record_event_on(self.streams.compute_ffn, device=self.device)
                    stream_mnt.wait_event_on(torch.cuda.current_stream(self.device), evt)
                    stream_mnt.release_event(eid, device=self.device)
                except Exception as e:
                    logger.warning(f"Layer {self.layer_id}: FFN event sync failed ({e}), fallback to wait_stream")
                    torch.cuda.current_stream(self.device).wait_stream(self.streams.compute_ffn)

        nvtx.range_pop()  # layer_forward

        self.forward_count += 1
        self.total_forward_time += time.time() - forward_start

        # 定期触发事件池 GC，避免事件积累
        self._gc_counter += 1
        if self._gc_counter % 10 == 0:
            try:
                from llama3 import stream_mnt
                stream_mnt.gc_event_pool(device=self.device, force=False)
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