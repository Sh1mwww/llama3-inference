# ============================================================
# 优化版本：
#   • 改进权重管理和异步加载
#   • 集成优化的KV offloader
#   • 增强的性能统计和监控
#   • 智能的内存管理
# ============================================================

import math
from typing import Optional, Tuple, List, Dict
import torch, torch.nn as nn, torch.nn.functional as F
import threading
from contextlib import contextmanager
import time

from .config import ModelArgs, LayerInfo, KVCacheArgs
from .kv_offload import OptimizedKVOffloader, BLOCK

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
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    try:
        yield
    finally:
        end_event.record()
        torch.cuda.synchronize()
        elapsed_us = int(start_event.elapsed_time(end_event) * 1000)
        
        with PERF_TRACKER.lock:
            PERF_TRACKER.stats[key] += elapsed_us
            if layer_id is not None:
                PERF_TRACKER.add_layer_stat(layer_id, key, elapsed_us)

# ---------- Optimized weight management ----------
class WeightManager:
    """管理权重的异步加载和缓存"""
    
    def __init__(self, device: str, max_cached_layers: int = 4):
        self.device = device
        self.is_cuda = device.startswith("cuda") and torch.cuda.is_available()
        self.weight_stream = torch.cuda.Stream(device=device) if self.is_cuda else None
        
        # 权重缓存管理
        self.cached_weights = {}  # layer_id -> {module_name: weight_dict}
        self.access_order = []    # LRU for weight cache
        self.max_cached = max_cached_layers
        self.loading_weights = set()  # track which layers are being loaded
        self.lock = threading.Lock()
    
    def ensure_weights_cuda(self, layer_id: int, modules: Dict[str, nn.Module], 
                           priority: bool = False):
        """确保指定layer的权重在GPU上"""
        with self.lock:
            if layer_id in self.cached_weights:
                # 更新访问顺序
                if layer_id in self.access_order:
                    self.access_order.remove(layer_id)
                self.access_order.append(layer_id)
                return
            
            if layer_id in self.loading_weights:
                return  # 已在加载中
            
            self.loading_weights.add(layer_id)
        
        try:
            self._load_weights_to_gpu(layer_id, modules, priority)
        finally:
            with self.lock:
                self.loading_weights.discard(layer_id)
    
    def _load_weights_to_gpu(self, layer_id: int, modules: Dict[str, nn.Module], 
                            priority: bool):
        """实际加载权重到GPU"""
        with cuda_timer("weights_hbm_us", layer_id):
            if self.weight_stream and not priority:
                with torch.cuda.stream(self.weight_stream):
                    for name, module in modules.items():
                        for param in module.parameters():
                            if not param.is_cuda:
                                param.data = param.data.to(self.device, non_blocking=True)
                torch.cuda.current_stream().wait_stream(self.weight_stream)
            else:
                # 同步加载，用于高优先级请求
                for name, module in modules.items():
                    for param in module.parameters():
                        if not param.is_cuda:
                            param.data = param.data.to(self.device)
        
        # 更新缓存
        with self.lock:
            self.cached_weights[layer_id] = {
                name: {p_name: param for p_name, param in module.named_parameters()}
                for name, module in modules.items()
            }
            self.access_order.append(layer_id)
            self._maybe_evict_weights()
    
    def _maybe_evict_weights(self):
        """LRU驱逐权重缓存"""
        while len(self.cached_weights) > self.max_cached:
            if not self.access_order:
                break
            
            evict_layer = self.access_order.pop(0)
            if evict_layer in self.cached_weights:
                # 将权重移回CPU
                for module_weights in self.cached_weights[evict_layer].values():
                    for param in module_weights.values():
                        if param.is_cuda:
                            param.data = param.data.cpu()
                
                del self.cached_weights[evict_layer]
    
    def prefetch_weights(self, layer_ids: List[int], modules_dict: Dict[int, Dict[str, nn.Module]]):
        """预取权重"""
        def _prefetch():
            for layer_id in layer_ids:
                if layer_id in modules_dict:
                    self.ensure_weights_cuda(layer_id, modules_dict[layer_id], priority=False)
        
        if self.is_cuda:
            threading.Thread(target=_prefetch, daemon=True).start()

# 全局权重管理器
WEIGHT_MANAGER = None

def get_weight_manager(device: str) -> WeightManager:
    global WEIGHT_MANAGER
    if WEIGHT_MANAGER is None:
        WEIGHT_MANAGER = WeightManager(device)
    return WEIGHT_MANAGER

# ---------- helpers (保持不变但优化) ----------
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

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor) -> torch.Tensor:
    b, l, h, d = x.shape
    x_ = x.float().reshape(b, l, h, d // 2, 2)
    x_complex = torch.view_as_complex(x_)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    out = torch.view_as_real(x_complex * freqs_complex)
    return out.reshape(b, l, h, d).type_as(x)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1: 
        return x
    b, t, h, d = x.shape
    return x[:, :, :, None, :].expand(b, t, h, n_rep, d).contiguous().view(b, t, h * n_rep, d)

# ---------- Optimized SelfAttention ----------
class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        self.topk_blk = args.topk_blk
        self.device = args.device
        self.is_cuda = str(self.device).startswith("cuda") and torch.cuda.is_available()
        
        # Linear权重初始化在CPU
        self.wq = nn.Linear(args.dim, self.n_heads_q * self.head_dim, bias=False, device="cpu")
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False, device="cpu")
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False, device="cpu")
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False, device="cpu")
        
        # 使用优化的KV offloader
        self.block_sz = BLOCK
        self.offloader = OptimizedKVOffloader(
            layers=args.n_layers,
            heads=self.n_kv_heads,
            dim=self.head_dim,
            max_seq=args.max_seq_len,
            max_batch=args.max_batch_size,
            device=args.device,
            dtype_bytes=2  # float16
        )
        
        self.layer_id = -1
        self.weight_manager = get_weight_manager(args.device)
        self.attention_history = []  # 用于分析注意力模式
        
        # 预分配缓冲区减少内存分配
        self.qkv_buffer = None
        self.scores_buffer = None
    
    def _get_modules_dict(self):
        """获取模块字典用于权重管理"""
        return {
            "wq": self.wq,
            "wk": self.wk, 
            "wv": self.wv,
            "wo": self.wo
        }
    
    def _ensure_weights_cuda(self):
        """确保权重在GPU上"""
        modules = self._get_modules_dict()
        self.weight_manager.ensure_weights_cuda(self.layer_id, modules, priority=True)
    
    def _allocate_buffers(self, batch_size: int, seq_len: int, max_kv_len: int):
        """预分配计算缓冲区"""
        if (self.qkv_buffer is None or 
            self.qkv_buffer[0].size(0) < batch_size or 
            self.qkv_buffer[0].size(1) < seq_len):
            
            with cuda_timer("memory_alloc_us", self.layer_id):
                # QKV缓冲区
                q_shape = (batch_size, seq_len, self.n_heads_q, self.head_dim)
                kv_shape = (batch_size, seq_len, self.n_kv_heads, self.head_dim)
                
                self.qkv_buffer = (
                    torch.empty(q_shape, dtype=torch.float16, device=self.device),
                    torch.empty(kv_shape, dtype=torch.float16, device=self.device),
                    torch.empty(kv_shape, dtype=torch.float16, device=self.device)
                )
                
                # 注意力分数缓冲区
                scores_shape = (batch_size, self.n_heads_q, seq_len, max_kv_len)
                self.scores_buffer = torch.empty(scores_shape, dtype=torch.float16, device=self.device)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        start_time = time.time()
        bsz, seqlen, _ = x.shape
        
        # 确保权重在GPU上
        self._ensure_weights_cuda()
        
        # 预取下一层权重
        if hasattr(self, '_next_layer_modules'):
            self.weight_manager.prefetch_weights([self.layer_id + 1], {self.layer_id + 1: self._next_layer_modules})
        
        # QKV投影
        with cuda_timer("attn_us", self.layer_id):
            q = self.wq(x).view(bsz, seqlen, self.n_heads_q, self.head_dim)
            k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
            v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
            
            # 应用旋转位置编码
            q = apply_rotary_embeddings(q, freqs_complex)
            k = apply_rotary_embeddings(k, freqs_complex)
        
        # Push当前block到offloader
        blk_idx = start_pos // self.block_sz
        self.offloader.push(self.layer_id, blk_idx, k.squeeze(0), v.squeeze(0))
        
        # 获取Top-K blocks
        blocks = self.offloader.topk_blocks(self.layer_id, self.topk_blk)
        if blk_idx not in blocks:
            blocks.append(blk_idx)
        blocks = sorted(blocks)
        needed = torch.tensor(blocks, device=x.device)
        
        # 异步获取KV
        with cuda_timer("kv_fetch_us", self.layer_id):
            k_full, v_full = self.offloader.fetch(self.layer_id, needed)
        
        # 形状调整
        if k_full.dim() == 3:
            k_full = k_full.permute(1, 0, 2).unsqueeze(0)
            v_full = v_full.permute(1, 0, 2).unsqueeze(0)
        
        k_full = k_full.to(q.dtype)
        v_full = v_full.to(q.dtype)
        
        # 重复KV heads
        if self.n_heads_q != self.n_kv_heads:
            k_full = k_full.repeat_interleave(self.n_rep, dim=1)
            v_full = v_full.repeat_interleave(self.n_rep, dim=1)
        
        # 注意力计算
        q = q.transpose(1, 2)  # (B, H, Tq, D)
        
        with cuda_timer("attn_us", self.layer_id):
            # 使用优化的注意力计算
            scores = torch.matmul(q, k_full.transpose(2, 3))
            scores = scores / math.sqrt(self.head_dim)
            
            # 应用causal mask（如果需要）
            if hasattr(self, 'apply_causal_mask') and self.apply_causal_mask:
                seq_len_q = q.size(2)
                seq_len_k = k_full.size(2)
                mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=q.device), diagonal=1)
                scores = scores.masked_fill(mask.bool(), float('-inf'))
            
            # Softmax和输出计算
            attn_weights = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn_weights, v_full)
        
        out = out.transpose(1, 2).reshape(bsz, seqlen, -1)
        
        # 更新block重要性
        with torch.no_grad():
            token_imp = attn_weights.mean(dim=(0, 1, 2))  # (Tkv,)
            block_scores = []
            for i, block_id in enumerate(blocks):
                s = i * self.block_sz
                e = min(s + self.block_sz, token_imp.size(0))
                score = float(token_imp[s:e].sum().item()) if s < token_imp.size(0) else 0.0
                block_scores.append(score)
            
            self.offloader.update_importances(self.layer_id, blocks, block_scores)
        
        # 输出投影
        result = self.wo(out)
        
        # 记录性能统计
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
        
        # Linear层CPU初始化
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False, device="cpu")
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False, device="cpu")
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False, device="cpu")
        
        self.device = args.device
        self.layer_id = -1
        self.weight_manager = get_weight_manager(args.device)
        
        # 激活函数缓冲区
        self.activation_buffer = None
    
    def _get_modules_dict(self):
        return {
            "w1": self.w1,
            "w2": self.w2,
            "w3": self.w3
        }
    
    def _ensure_weights_cuda(self):
        modules = self._get_modules_dict()
        self.weight_manager.ensure_weights_cuda(self.layer_id, modules, priority=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_weights_cuda()
        
        with cuda_timer("ffn_us", self.layer_id):
            # SwiGLU激活函数
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
        
        # 规范化层
        self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # 注意力和前馈网络
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        
        # 设置layer_id
        self.attention.layer_id = layer_id
        self.feed_forward.layer_id = layer_id
        
        # 为预取设置下一层模块引用
        if layer_id < args.n_layers - 1:
            self.attention._next_layer_modules = self.feed_forward._get_modules_dict()
        
        # 性能监控
        self.forward_count = 0
        self.total_forward_time = 0.0
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        forward_start = time.time()
        
        with cuda_timer("total_forward_us", self.layer_id):
            # 注意力块
            h = x + self.attention(self.attn_norm(x), start_pos, freqs_complex)
            
            # 前馈块
            out = h + self.feed_forward(self.ffn_norm(h))
        
        # 更新统计
        self.forward_count += 1
        self.total_forward_time += time.time() - forward_start
        
        return out
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
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
    """获取全局性能统计"""
    return PERF_TRACKER.get_stats()

def reset_performance_stats():
    """重置性能统计"""
    PERF_TRACKER.reset()

def optimize_layer_execution_order(layers: List[EncoderBlock]) -> List[int]:
    """基于性能统计优化层执行顺序"""
    layer_stats = [(i, layer.get_performance_stats()["avg_time_ms"]) 
                   for i, layer in enumerate(layers)]
    
    # 按平均执行时间排序，优化调度
    layer_stats.sort(key=lambda x: x[1])
    return [layer_id for layer_id, _ in layer_stats]