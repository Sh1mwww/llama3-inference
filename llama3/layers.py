# ============================================================
# 优化版本：
#   • 改进权重管理和异步加载
#   • 集成优化的KV offloader
#   • 增强的性能统计和监控
#   • 智能的内存管理
# ============================================================

import math
from typing import Optional, List, Dict
import torch, torch.nn as nn, torch.nn.functional as F
import threading
import logging
from contextlib import contextmanager
import time

# 配置日志
logger = logging.getLogger(__name__)

from .config import ModelArgs
from .kv_offload import KVOffloader, BLOCK

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
    
    try:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        yield
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA OOM in timer for {key}: {e}")
        torch.cuda.empty_cache()
        raise
    except RuntimeError as e:
        if "CUDA" in str(e):
            logger.error(f"CUDA error in timer for {key}: {e}")
            raise
        else:
            logger.error(f"Runtime error in timer for {key}: {e}")
            raise
    finally:
        try:
            if end_event is not None:
                end_event.record()
                torch.cuda.synchronize()
                
                if start_event is not None:
                    elapsed_us = int(start_event.elapsed_time(end_event) * 1000)
                    
                    with PERF_TRACKER.lock:
                        PERF_TRACKER.stats[key] += elapsed_us
                        if layer_id is not None:
                            PERF_TRACKER.add_layer_stat(layer_id, key, elapsed_us)
        except Exception as e:
            logger.warning(f"Error in cuda_timer cleanup for {key}: {e}")

# ---------- Optimized weight management ----------
class WeightManager:
    """管理权重的异步加载和缓存"""
    
    def __init__(self, device: str, max_cached_layers: int = 4):
        # 导入GPU工具
        from .gpu_utils import SafeGPUManager, gpu_memory_guard, gpu_safe_operation
        from .memory_manager import GlobalMemoryManager
        
        self.gpu_manager = SafeGPUManager(device, auto_fallback=True)
        self.device = self.gpu_manager.current_device
        self.is_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        
        # 获取内存管理器
        self.memory_manager = GlobalMemoryManager.get_instance()
        
        # 安全创建CUDA流
        self.weight_stream = None
        if self.is_cuda:
            try:
                device_id = int(self.device.split(":")[1]) if ":" in self.device else 0
                with self.gpu_manager.safe_cuda_context(device_id):
                    self.weight_stream = torch.cuda.Stream(device=self.device)
            except Exception as e:
                logger.warning(f"Failed to create CUDA stream: {e}")
                self.weight_stream = None
        
        # 权重缓存管理
        self.cached_weights = {}  # layer_id -> {module_name: weight_dict}
        self.access_order = []    # LRU for weight cache
        self.max_cached = max_cached_layers
        self.loading_weights = set()  # track which layers are being loaded
        self.lock = threading.Lock()
        self.memory_threshold_gb = 0.5  # GPU内存保护阈值
    
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
            
            # 更新缓存
            with self.lock:
                self.cached_weights[layer_id] = {
                    name: {p_name: param for p_name, param in module.named_parameters()}
                    for name, module in modules.items()
                }
                self.access_order.append(layer_id)
                self._maybe_evict_weights()
                
        finally:
            with self.lock:
                self.loading_weights.discard(layer_id)
    
    def _load_weights_to_gpu(self, layer_id: int, modules: Dict[str, nn.Module], 
                            priority: bool):
        """实际加载权重到GPU"""
        from .gpu_utils import gpu_memory_guard, gpu_safe_operation
        
        @gpu_safe_operation(retry_count=2, cleanup_on_error=True)
        def _safe_load():
            return self._do_load_weights(layer_id, modules, priority)
        
        return _safe_load()
    
    def _do_load_weights(self, layer_id: int, modules: Dict[str, nn.Module], 
                        priority: bool):
        """执行实际的权重加载"""
        from .gpu_utils import gpu_memory_guard
        
        device_id = int(self.device.split(":")[1]) if ":" in self.device else 0
        
        # 检查内存是否足够
        if self.memory_manager and self.is_cuda:
            total_weight_size = 0
            for module in modules.values():
                for param in module.parameters():
                    if not param.is_cuda:
                        total_weight_size += param.numel() * param.element_size()
            
            if not self.memory_manager.can_allocate(total_weight_size):
                logger.warning(f"Insufficient memory for layer {layer_id} weights ({total_weight_size/(1024**3):.2f}GB)")
                # 尝试清理缓存
                self._emergency_weight_cleanup()
                
                if not self.memory_manager.can_allocate(total_weight_size):
                    raise RuntimeError(f"Cannot load layer {layer_id}: insufficient GPU memory")
        
        with gpu_memory_guard(self.device, self.memory_threshold_gb):
            with cuda_timer("weights_hbm_us", layer_id):
                if self.weight_stream and not priority and self.is_cuda:
                    try:
                        with self.gpu_manager.safe_cuda_context(device_id):
                            with torch.cuda.stream(self.weight_stream):
                                for module_name, module in modules.items():
                                    for param in module.parameters():
                                        if not param.is_cuda:
                                            param.data = param.data.to(self.device, non_blocking=True)
                            
                            # 安全的流等待，避免死锁
                            try:
                                # 添加超时检查，避免无限等待
                                current_stream = torch.cuda.current_stream(self.device)
                                if current_stream != self.weight_stream:
                                    current_stream.wait_stream(self.weight_stream)
                                    # 验证操作完成
                                    if not self.weight_stream.query():
                                        logger.warning(f"Weight stream not completed for layer {layer_id}")
                            except RuntimeError as e:
                                logger.error(f"Stream wait failed for layer {layer_id}: {e}")
                                # 降级到同步操作
                                torch.cuda.synchronize(self.device)
                                raise
                    except Exception as e:
                        logger.warning(f"Async weight loading failed for layer {layer_id}: {e}")
                        # 回退到同步加载
                        self._sync_load_weights(modules)
                else:
                    # 同步加载，用于高优先级请求或CPU设备
                    self._sync_load_weights(modules)
    
    def _emergency_weight_cleanup(self):
        """紧急权重清理"""
        with self.lock:
            # 清理一半的缓存权重
            cleanup_count = len(self.cached_weights) // 2
            for _ in range(cleanup_count):
                if self.access_order:
                    evict_layer = self.access_order.pop(0)
                    if evict_layer in self.cached_weights:
                        del self.cached_weights[evict_layer]
                        logger.info(f"Emergency cleanup: evicted layer {evict_layer}")
        
        # 清理GPU缓存
        if self.is_cuda:
            torch.cuda.empty_cache()
    
    def _sync_load_weights(self, modules: Dict[str, nn.Module]):
        """同步加载权重"""
        try:
            for module_name, module in modules.items():
                for param in module.parameters():
                    if not param.is_cuda and self.is_cuda:
                        param.data = param.data.to(self.device)
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU OOM during weight loading: {e}")
            torch.cuda.empty_cache()
            raise
        except Exception as e:
            logger.error(f"Weight loading failed: {e}")
            raise
        
        # 更新缓存（移到主函数中处理）
    
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
        
        # 使用优化的KV offloader
        self.block_sz = BLOCK
        self.offloader = KVOffloader(
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
        """安全预分配计算缓冲区，防止OOM"""
        if (self.qkv_buffer is None or 
            self.qkv_buffer[0].size(0) < batch_size or 
            self.qkv_buffer[0].size(1) < seq_len):
            
            with cuda_timer("memory_alloc_us", self.layer_id):
                # 限制缓冲区大小防止OOM
                MAX_BUFFER_ELEMENTS = 50_000_000  # 约100MB for float16
                
                # 计算所需内存
                q_elements = batch_size * seq_len * self.n_heads_q * self.head_dim
                kv_elements = batch_size * seq_len * self.n_kv_heads * self.head_dim
                scores_elements = batch_size * self.n_heads_q * seq_len * max_kv_len
                
                # 如果scores缓冲区过大，限制max_kv_len
                if scores_elements > MAX_BUFFER_ELEMENTS:
                    logger.warning(f"Large attention buffer requested ({scores_elements} elements), limiting to prevent OOM")
                    # 计算安全的max_kv_len
                    safe_kv_len = MAX_BUFFER_ELEMENTS // (batch_size * self.n_heads_q * seq_len)
                    max_kv_len = min(max_kv_len, max(safe_kv_len, 1024))  # 至少保留1024
                    scores_elements = batch_size * self.n_heads_q * seq_len * max_kv_len
                    self.use_chunked_attention = True
                else:
                    self.use_chunked_attention = False
                
                # 检查内存管理器
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
                    # QKV缓冲区
                    q_shape = (batch_size, seq_len, self.n_heads_q, self.head_dim)
                    kv_shape = (batch_size, seq_len, self.n_kv_heads, self.head_dim)
                    
                    self.qkv_buffer = (
                        torch.empty(q_shape, dtype=torch.float16, device=self.device),
                        torch.empty(kv_shape, dtype=torch.float16, device=self.device),
                        torch.empty(kv_shape, dtype=torch.float16, device=self.device)
                    )
                    
                    # 注意力分数缓冲区 - 使用限制后的大小
                    scores_shape = (batch_size, self.n_heads_q, seq_len, max_kv_len)
                    self.scores_buffer = torch.empty(scores_shape, dtype=torch.float16, device=self.device)
                    
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"GPU OOM during buffer allocation: batch={batch_size}, seq={seq_len}, kv_len={max_kv_len}")
                    # 清理并抛出有用的错误
                    torch.cuda.empty_cache()
                    raise RuntimeError(f"GPU OOM: Cannot allocate attention buffers. Try reducing batch_size (current: {batch_size}) or max sequence length.") from e
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        if not self.wq.weight.is_cuda:
            for mod in (self.wq, self.wk, self.wv, self.wo):
                mod.to(x.device, non_blocking=True)
        
        start_time = time.time()
        bsz, seqlen, _ = x.shape
        
        # 确保权重在GPU上
        self._ensure_weights_cuda()
        
        # 预取下一层权重
        if hasattr(self, '_next_layer_modules'):
            self.weight_manager.prefetch_weights([self.layer_id + 1], {self.layer_id + 1: self._next_layer_modules})
        
        # QKV投影
        with cuda_timer("attn_us", self.layer_id):
            # assert x.is_cuda, "x 不在 GPU？"
            # print("wq.weight.device =", self.wq.weight.device)
            q = self.wq(x).view(bsz, seqlen, self.n_heads_q, self.head_dim)
            k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
            v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
            
            # 应用旋转位置编码
            q = apply_rotary_embeddings(q, freqs_complex)
            k = apply_rotary_embeddings(k, freqs_complex)
        
        # Push当前block到offloader  
        # 支持多token和多batch处理
        bsz, seqlen, n_heads, head_dim = k.shape
        
        # 对于每个token位置，计算对应的block并push
        for seq_idx in range(seqlen):
            blk_idx = (start_pos + seq_idx) // self.block_sz
            # 提取当前位置的k,v: (bsz, n_heads, head_dim)
            k_curr = k[:, seq_idx, :, :]  # (bsz, n_heads, head_dim)
            v_curr = v[:, seq_idx, :, :]  # (bsz, n_heads, head_dim)
            
            # 如果batch_size=1，保持原有行为移除batch维度
            if bsz == 1:
                k_curr = k_curr.squeeze(0)  # (n_heads, head_dim)
                v_curr = v_curr.squeeze(0)  # (n_heads, head_dim)
            
            self.offloader.push(self.layer_id, blk_idx, k_curr, v_curr)
        
        # 使用第一个token的位置来计算需要的blocks（简化处理）
        blk_idx = start_pos // self.block_sz 
        
        # 获取Top-K blocks
        blocks = self.offloader.topk_blocks(self.layer_id, self.topk_blk)
        if blk_idx not in blocks:
            blocks.append(blk_idx)
        blocks = sorted(blocks)
        needed = torch.tensor(blocks, device=x.device)
        
        # 异步获取KV
        with cuda_timer("kv_fetch_us", self.layer_id):
        #     k_full, v_full = self.offloader.fetch(self.layer_id, needed)
            fetch_evt_start = torch.cuda.Event(enable_timing=True)
            fetch_evt_end = torch.cuda.Event(enable_timing=True)
            fetch_evt_start.record()
            k_full, v_full = self.offloader.fetch(self.layer_id, needed)
            fetch_evt_end.record()
            
            # 避免不必要的同步，仅在profiling时同步
            if getattr(self, 'enable_profiling', False):
                torch.cuda.synchronize()  # 仅在profiling模式下同步
                self.kv_elapsed_time = fetch_evt_start.elapsed_time(fetch_evt_end) * 1000
                PERF_TRACKER.add_layer_stat(self.layer_id, "kv_fetch_us", self.kv_elapsed_time)
            else:
                # 非profiling模式：不阻塞GPU管道
                self.kv_elapsed_time = 0
            

        # 形状调整以支持多batch
        if k_full.dim() == 3:
            # 原始单batch情况: (seq_len, n_heads, head_dim) -> (1, n_heads, seq_len, head_dim)
            k_full = k_full.permute(1, 0, 2).unsqueeze(0)
            v_full = v_full.permute(1, 0, 2).unsqueeze(0)
        elif k_full.dim() == 4:
            # 多batch情况: (bsz, seq_len, n_heads, head_dim) -> (bsz, n_heads, seq_len, head_dim)
            k_full = k_full.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
            v_full = v_full.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        
        # 确保k_full和v_full与q的batch维度一致
        if k_full.size(0) == 1 and bsz > 1:
            # 单batch的KV需要扩展到多batch
            k_full = k_full.expand(bsz, -1, -1, -1)
            v_full = v_full.expand(bsz, -1, -1, -1)
        
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
            attn_evt_start = torch.cuda.Event(enable_timing=True)
            attn_evt_end = torch.cuda.Event(enable_timing=True)
            attn_evt_start.record()  
            scores = torch.matmul(q, k_full.transpose(2, 3))
            scores = scores / math.sqrt(self.head_dim)
            
            # 应用causal mask 
            if hasattr(self, 'apply_causal_mask') and self.apply_causal_mask:
                seq_len_q = q.size(2)
                seq_len_k = k_full.size(2)
                mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=q.device), diagonal=1)
                scores = scores.masked_fill(mask.bool(), float('-inf'))
            
            # Softmax和输出计算
            attn_weights = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn_weights, v_full)
            attn_evt_end.record()
            
            # 避免不必要的同步，仅在profiling时同步
            if getattr(self, 'enable_profiling', False):
                torch.cuda.synchronize()
                self.attn_time = attn_evt_start.elapsed_time(attn_evt_end) * 1000
                PERF_TRACKER.add_layer_stat(self.layer_id, "attn_us", self.attn_time)
            else:
                # 非profiling模式：不阻塞GPU管道
                self.attn_time = 0
        out = out.transpose(1, 2).reshape(bsz, seqlen, -1)
        
        stats = PERF_TRACKER.layer_stats.get(self.layer_id, {})
        self.kv_elapsed_time = stats.get("kv_fetch_us", 0)
        self.attn_time       = stats.get("attn_us",     0)
        
        # 更新block重要性
        with torch.no_grad():
            token_imp = attn_weights.mean(dim=(0, 1, 2))  # (Tkv,)
            block_scores = []
            for i, _ in enumerate(blocks):
                s = i * self.block_sz
                e = min(s + self.block_sz, token_imp.size(0))
                score = float(token_imp[s:e].sum().item()) if s < token_imp.size(0) else 0.0
                block_scores.append(score)
            
            self.offloader.update_importances(self.layer_id, blocks, block_scores, batch_idx= blk_idx)
        
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