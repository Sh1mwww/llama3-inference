import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List, Optional
import time
import json
from .stream_mnt import get_streams

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

def _pinned_clone_cpu(t: torch.Tensor) -> torch.Tensor:
    x = t.detach().cpu().contiguous()
    try:
        x = x.pin_memory()
    except Exception:
        pass
    return x

def _collect_block_modules(block: nn.Module) -> Dict[str, nn.Module]:
    """
    收集 block 中所有需要流式管理的模块：
    1. attention 和 feed_forward 的子模块（通过 _get_modules_dict）
    2. norm 层（attn_norm, ffn_norm）- 这些是小权重，适合常驻 GPU
    3. 若都不存在，则退化为把整个 block 视为一个模块整体搬运
    """
    mods: Dict[str, nn.Module] = {}
    
    # 收集 attention 子模块
    if hasattr(block, "attention") and hasattr(block.attention, "_get_modules_dict"):
        try:
            mods.update(block.attention._get_modules_dict())
        except Exception:
            pass
    
    # 收集 feed_forward 子模块        
    if hasattr(block, "feed_forward") and hasattr(block.feed_forward, "_get_modules_dict"):
        try:
            mods.update(block.feed_forward._get_modules_dict())
        except Exception:
            pass
    
    # 收集 norm 层（小权重，适合常驻 GPU）
    norm_modules = {}
    for norm_name in ["attn_norm", "ffn_norm", "attention_norm", "ffn_norm_pre", "ffn_norm_post"]:
        if hasattr(block, norm_name):
            norm_module = getattr(block, norm_name)
            if isinstance(norm_module, nn.Module):
                norm_modules[f"norm_{norm_name}"] = norm_module
    
    if norm_modules:
        mods.update(norm_modules)
    
    # 兜底：若没有收集到任何模块，把整个 block 搬运
    if not mods:
        mods["__block__"] = block
    
    return mods

class WeightStreamingManager:
    """
    权重流式管理器（最小侵入）：
      - CPU 常驻主副本pinned,GPU 仅按层临时拷贝
      - 进入层 i 前:ensure_on_gpu(i) + wait_stream
      - 层内：异步预取 i+1 .. i+prefetch_distance(不等待)
      - LRU: 最多保留 max_cached_layers 层在 GPU;逐出不D2H,只指针切回CPU主副本
    用法：
        wsm = WeightStreamingManager(model, device="cuda", prefetch_distance=1, max_cached_layers=4)
        # 创建后立刻生效(安装了每层 forward_pre_hook)
    """
    def __init__(self,
                 model: nn.Module,
                 device: str = "cuda",
                 prefetch_distance: int = 1,
                 max_cached_layers: int = 4,
                 layers_attr: str = "layers",
                 warmup_layers: int = 1,
                 verbose: bool = False,
                 monitor_fragmentation: bool = False):
        assert torch.cuda.is_available(), "CUDA not available"
        self.device = device
        self.verbose = verbose
        self.streams = get_streams(device)
        self.prefetch_distance = int(prefetch_distance)
        self.max_cached_layers = int(max_cached_layers)
        
        # 内存碎片化监控
        self.monitor_fragmentation = monitor_fragmentation
        self.memory_stats = []
        self.fragmentation_threshold = 0.3  # 30%碎片化阈值

        # 找到 block 列表
        blocks: Optional[List[nn.Module]] = None
        if hasattr(model, "layer_infos"):
            try:
                blocks = [info["block"] for info in getattr(model, "layer_infos")]
            except Exception:
                blocks = None
        if blocks is None and hasattr(model, layers_attr):
            blocks = list(getattr(model, layers_attr))
        if not blocks:
            # 最后兜底：从 children 里找一个最长的顺序容器
            cands = []
            for _name, child in model.named_children():
                if isinstance(child, (nn.Sequential, nn.ModuleList)):
                    cands.append(list(child))
            blocks = max(cands, key=len) if cands else []
        if not blocks:
            raise RuntimeError("Cannot locate transformer blocks (layers) on model.")

        self.blocks: List[nn.Module] = blocks
        # 每层需要搬运的模块集合
        self.block_mods: Dict[int, Dict[str, nn.Module]] = {
            i: _collect_block_modules(b) for i, b in enumerate(self.blocks)
        }

        # CPU 主副本：param_id -> pinned cpu tensor
        self.cpu_stash: Dict[int, torch.Tensor] = {}
        # 哪些层当前在 GPU（LRU，值无用）
        self.gpu_cache: "OrderedDict[int, None]" = OrderedDict()
        self.layer_events: Dict[int, torch.cuda.Event] = {}

        # 安装 forward_pre_hook：确保当前层 + 预取后续
        for i, blk in enumerate(self.blocks):
            blk.register_forward_pre_hook(self._pre_hook_factory(i))

        # 将 norm 层移到 GPU 并常驻（这些是小权重）
        self._setup_resident_norms()
        
        # （可选）预热前几层，降低第 0 层等待
        if warmup_layers > 0:
            warm = list(range(min(warmup_layers, len(self.blocks))))
            self.prefetch(warm)
            if self.verbose:
                print(f"[WSM] warmup prefetch: {warm}")

    def _setup_resident_norms(self):
        """将 norm 层移到 GPU 并常驻（这些是小权重，不需要流式传输）"""
        if self.verbose:
            print("[WSM] Setting up resident norm layers...")
            
        norm_count = 0
        for layer_id, modules_dict in self.block_mods.items():
            for module_name, module in modules_dict.items():
                # 如果是 norm 模块，移到 GPU 并从流式管理中排除
                if module_name.startswith("norm_"):
                    try:
                        module.to(self.device)
                        norm_count += 1
                        if self.verbose:
                            print(f"[WSM] Layer {layer_id}: {module_name} -> {self.device} (resident)")
                    except Exception as e:
                        if self.verbose:
                            print(f"[WSM] Warning: Failed to move {module_name} to {self.device}: {e}")
        
        if self.verbose:
            print(f"[WSM] {norm_count} norm layers set as resident on GPU")

    # ============ 基元：参数/模块的 CPU/GPU 切换 ============

    def _record_layer_ready_event(self, idx: int):
        """在权重H2D流上记录：该层权重拷贝已全部排队后的事件"""
        if not torch.cuda.is_available() or getattr(self.streams, "weight_h2d", None) is None:
            return
        evt = self.layer_events.get(idx)
        if evt is None:
            # 非阻塞事件即可；用于被 compute 流 wait_event
            evt = torch.cuda.Event(blocking=False)
            self.layer_events[idx] = evt
        # 在权重H2D流上记录事件，表示“该层所有已排队的H2D完成时触发”
        evt.record(self.streams.weight_h2d)

    def _wait_layer_ready(self, idx: int):
        """在当前流上等待该层已记录的事件；若无事件则退化为流间等待"""
        evt = self.layer_events.get(idx)
        if evt is not None:
            torch.cuda.current_stream(self.device).wait_event(evt)
        else:
            # 兜底：如果还没记录事件，就退化为等待权重流
            self.streams.wait_weight_ready_on_current(self.device)

    def _ensure_param_cpu_stash_inplace(self, p: torch.nn.Parameter):
        """把 p.data 原地置换为 pinned CPU 主副本，并记录到 cpu_stash。避免 DRAM 双持。"""
        pid = id(p)
        if pid in self.cpu_stash:
            return
        pinned = _pinned_clone_cpu(p.data)
        p.data = pinned               # 置换：原CPU张量释放
        self.cpu_stash[pid] = p.data  # 记录主副本（就是现在的 p.data）

    def _ensure_param_on_gpu(self, p: torch.nn.Parameter):
        """从主副本（pinned CPU）H2D 到 GPU，并把 p.data 切到 GPU 张量（在高优先级流上）。"""
        nvtx.range_push("param_h2d")
        if p.device.type != "cpu":
            nvtx.range_pop()
            return
        pid = id(p)
        
        nvtx.range_push("ensure_cpu_stash")
        self._ensure_param_cpu_stash_inplace(p)
        nvtx.range_pop()  # ensure_cpu_stash
        
        nvtx.range_push("weight_h2d_stream")
        with torch.cuda.stream(self.streams.weight_h2d):
            nvtx.range_push("cpu_to_gpu_transfer")
            p_gpu = self.cpu_stash[pid].to(self.device, non_blocking=True)
            nvtx.range_pop()  # cpu_to_gpu_transfer
        nvtx.range_pop()  # weight_h2d_stream
        
        p.data = p_gpu  # 切到 GPU
        nvtx.range_pop()  # param_h2d

    def _evict_param_to_cpu(self, p: torch.nn.Parameter):
        """逐出：不做 D2H，p.data 直接指回 cpu_stash（CPU 主副本）。"""
        pid = id(p)
        if pid in self.cpu_stash:
            p.data = self.cpu_stash[pid]

    def _ensure_module_on_gpu(self, m: nn.Module):
        for p in m.parameters(recurse=True):
            self._ensure_param_on_gpu(p)
        for b in m.buffers(recurse=True):
            if b.device.type == "cpu":
                with torch.cuda.stream(self.streams.weight_h2d):
                    b_gpu = b.detach().to(self.device, non_blocking=True)
                try:
                    b.data = b_gpu
                except Exception:
                    pass

    def _evict_module_to_cpu(self, m: nn.Module):
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

    # ============ 层级：ensure / prefetch / evict ============

    def _pre_hook_factory(self, idx: int):
        def _pre_hook(_module, _inputs):
            # 1) 确保本层在 GPU
            self.ensure_on_gpu(idx, wait=True)
            # 2) 预取后续若干层（不等待，重叠 H2D 与 compute）
            if self.prefetch_distance > 0:
                nxt = [j for j in range(idx + 1, min(idx + 1 + self.prefetch_distance, len(self.blocks)))]
                self.prefetch(nxt)
        return _pre_hook

    def ensure_on_gpu(self, idx: int, wait: bool):
        nvtx.range_push(f"ensure_layer_{idx}")
        self._record_memory_stats(f"ensure_start", idx)
        
        if idx in self.gpu_cache:
            nvtx.range_push(f"cache_hit_layer_{idx}")
            self.gpu_cache.move_to_end(idx)
            # 命中也可能还是预取中的层，需要按层事件来精确等待
            if wait:
                nvtx.range_push(f"wait_layer_{idx}")
                self._wait_layer_ready(idx)
                nvtx.range_pop()  # wait_layer
            nvtx.range_pop()  # cache_hit
        else:
            nvtx.range_push(f"cache_miss_layer_{idx}")
            
            while len(self.gpu_cache) >= self.max_cached_layers:
                old, _ = self.gpu_cache.popitem(last=False)
                nvtx.range_push(f"evict_layer_{old}")
                self._record_memory_stats(f"evict_start", old)
                self._evict_layer_to_cpu(old)
                self._record_memory_stats(f"evict_end", old)
                nvtx.range_pop()  # evict
                
            # H2D 当前层（逐模块发起H2D）
            nvtx.range_push(f"h2d_transfer_layer_{idx}")
            self._record_memory_stats(f"h2d_start", idx)
            for module_name, mod in self.block_mods[idx].items():
                if not module_name.startswith("norm_"):  # 跳过常驻的norm模块
                    nvtx.range_push(f"h2d_{module_name}")
                    self._ensure_module_on_gpu(mod)
                    nvtx.range_pop()  # h2d_module
            self._record_memory_stats(f"h2d_end", idx)
            nvtx.range_pop()  # h2d_transfer
            
            self.gpu_cache[idx] = None
            # 关键：所有该层的H2D已排队后，记录"该层就绪事件"
            nvtx.range_push(f"record_event_layer_{idx}")
            self._record_layer_ready_event(idx)
            nvtx.range_pop()  # record_event

            if self.verbose:
                print(f"[WSM] ->GPU layer={idx}")
            if wait:
                nvtx.range_push(f"wait_layer_{idx}")
                self._wait_layer_ready(idx)
                nvtx.range_pop()  # wait_layer
                
            nvtx.range_pop()  # cache_miss
        
        self._record_memory_stats(f"ensure_end", idx)
        nvtx.range_pop()  # ensure_layer


    def prefetch(self, ids: List[int]):
        if not ids:
            return
        nvtx.range_push(f"prefetch_layers_{ids}")
        
        for idx in ids:
            nvtx.range_push(f"prefetch_layer_{idx}")
            
            if idx in self.gpu_cache:
                nvtx.range_push(f"prefetch_cache_hit_{idx}")
                self.gpu_cache.move_to_end(idx)
                # 命中预取就不再重复H2D；可能已经有事件了，无需操作
                nvtx.range_pop()  # prefetch_cache_hit
                nvtx.range_pop()  # prefetch_layer
                continue
                
            nvtx.range_push(f"prefetch_cache_miss_{idx}")
            while len(self.gpu_cache) >= self.max_cached_layers:
                old, _ = self.gpu_cache.popitem(last=False)
                nvtx.range_push(f"prefetch_evict_{old}")
                self._evict_layer_to_cpu(old)
                nvtx.range_pop()  # prefetch_evict
                
            nvtx.range_push(f"prefetch_h2d_layer_{idx}")
            for module_name, mod in self.block_mods[idx].items():
                if not module_name.startswith("norm_"):  # 跳过常驻的norm模块
                    nvtx.range_push(f"prefetch_h2d_{module_name}")
                    self._ensure_module_on_gpu(mod)
                    nvtx.range_pop()  # prefetch_h2d_module
            nvtx.range_pop()  # prefetch_h2d_layer
            
            self.gpu_cache[idx] = None
            # 关键：预取层H2D排队后，记录"该层就绪事件"（不等待）
            nvtx.range_push(f"prefetch_record_event_{idx}")
            self._record_layer_ready_event(idx)
            nvtx.range_pop()  # prefetch_record_event

            if self.verbose:
                print(f"[WSM] prefetch layer={idx}")
                
            nvtx.range_pop()  # prefetch_cache_miss
            nvtx.range_pop()  # prefetch_layer
            
        nvtx.range_pop()  # prefetch_layers

    def _evict_layer_to_cpu(self, idx: int):
        for _name, mod in self.block_mods[idx].items():
            self._evict_module_to_cpu(mod)
        # 该层从GPU缓存移除后，对应的事件也一并清理（下次会重新创建/记录）
        self.layer_events.pop(idx, None)
        if self.verbose:
            print(f"[WSM]   evict layer={idx}")

    
    # ============ 兼容 layers.py 的接口 ============
    
    def ensure_weights_cuda(self, layer_id: int, modules: Dict[str, nn.Module], priority: bool = False):
        """
        兼容 layers.py 中 _ensure_weights_cuda 的调用
        将指定层的模块权重确保在 GPU 上
        """
        if layer_id >= len(self.blocks):
            return  # 超出范围，忽略
        
        # 使用现有的 ensure_on_gpu 逻辑
        self.ensure_on_gpu(layer_id, wait=priority)
        
        # 额外确保提供的模块在 GPU 上（如果与我们管理的不同）
        for name, module in modules.items():
            try:
                self._ensure_module_on_gpu(module)
            except Exception as e:
                if self.verbose:
                    print(f"[WSM] Warning: Failed to ensure {name} on GPU for layer {layer_id}: {e}")
    
    def prefetch_weights(self, layer_ids: List[int], modules_dict: Dict[int, Dict[str, nn.Module]] = None):
        """
        兼容 layers.py 中预取权重的调用
        """
        self.prefetch(layer_ids)
    
    def _get_memory_info(self):
        """获取当前GPU内存信息"""
        if not torch.cuda.is_available():
            return None
        
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        max_allocated = torch.cuda.max_memory_allocated(self.device)
        max_reserved = torch.cuda.max_memory_reserved(self.device)
        
        memory_stats = torch.cuda.memory_stats(self.device)
        
        return {
            'timestamp': time.time(),
            'allocated_mb': allocated / 1024**2,
            'reserved_mb': reserved / 1024**2,
            'max_allocated_mb': max_allocated / 1024**2,
            'max_reserved_mb': max_reserved / 1024**2,
            'fragmentation_ratio': (reserved - allocated) / reserved if reserved > 0 else 0,
            'allocation_count': memory_stats.get('allocation.all.current', 0),
            'segment_count': memory_stats.get('segment.all.current', 0),
            'large_pool_allocated': memory_stats.get('allocated_bytes.large_pool.current', 0) / 1024**2,
            'small_pool_allocated': memory_stats.get('allocated_bytes.small_pool.current', 0) / 1024**2,
        }
    
    def _record_memory_stats(self, operation: str, layer_id: int = -1):
        """记录内存统计信息"""
        if not self.monitor_fragmentation:
            return
        
        info = self._get_memory_info()
        if info:
            info['operation'] = operation
            info['layer_id'] = layer_id
            self.memory_stats.append(info)
            
            # 检查碎片化阈值
            if info['fragmentation_ratio'] > self.fragmentation_threshold:
                print(f"⚠️  High fragmentation detected: {info['fragmentation_ratio']:.3f} "
                      f"during {operation} (layer {layer_id})")
    
    def get_fragmentation_report(self):
        """生成碎片化分析报告"""
        if not self.memory_stats:
            return {"error": "No memory statistics collected"}
        
        fragmentation_ratios = [s['fragmentation_ratio'] for s in self.memory_stats]
        segment_counts = [s['segment_count'] for s in self.memory_stats]
        
        report = {
            'total_operations': len(self.memory_stats),
            'max_fragmentation': max(fragmentation_ratios),
            'avg_fragmentation': sum(fragmentation_ratios) / len(fragmentation_ratios),
            'min_fragmentation': min(fragmentation_ratios),
            'max_segments': max(segment_counts),
            'avg_segments': sum(segment_counts) / len(segment_counts),
            'high_fragmentation_count': sum(1 for r in fragmentation_ratios if r > self.fragmentation_threshold),
            'peak_allocated_mb': max(s['allocated_mb'] for s in self.memory_stats),
            'peak_reserved_mb': max(s['reserved_mb'] for s in self.memory_stats),
        }
        
        # 严重程度评估
        if report['max_fragmentation'] > 0.4:
            report['severity'] = 'CRITICAL'
        elif report['max_fragmentation'] > 0.3:
            report['severity'] = 'HIGH'
        elif report['max_fragmentation'] > 0.15:
            report['severity'] = 'MEDIUM'
        else:
            report['severity'] = 'LOW'
        
        return report
    
    def save_memory_stats(self, filename: str):
        """保存内存统计到文件"""
        if not self.memory_stats:
            print("⚠️  No memory statistics to save")
            return
        
        with open(filename, 'w') as f:
            json.dump({
                'memory_stats': self.memory_stats,
                'fragmentation_report': self.get_fragmentation_report(),
                'config': {
                    'max_cached_layers': self.max_cached_layers,
                    'prefetch_distance': self.prefetch_distance,
                    'device': self.device,
                    'fragmentation_threshold': self.fragmentation_threshold
                }
            }, f, indent=2)
        
        print(f"💾 Memory statistics saved to {filename}")
    
    def clear_memory_stats(self):
        """清空内存统计"""
        self.memory_stats.clear()
        torch.cuda.reset_peak_memory_stats(self.device)
