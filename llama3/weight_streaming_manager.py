# weight_streaming_manager.py
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List, Optional
from .stream_mnt import get_streams

def _pinned_clone_cpu(t: torch.Tensor) -> torch.Tensor:
    x = t.detach().cpu().contiguous()
    try:
        x = x.pin_memory()
    except Exception:
        pass
    return x

def _collect_block_modules(block: nn.Module) -> Dict[str, nn.Module]:
    """
    优先使用 block.attention/_get_modules_dict 和 block.feed_forward/_get_modules_dict。
    若不存在，则退化为把整个 block 视为一个模块整体搬运（也可用）。
    """
    mods: Dict[str, nn.Module] = {}
    if hasattr(block, "attention") and hasattr(block.attention, "_get_modules_dict"):
        try:
            mods.update(block.attention._get_modules_dict())
        except Exception:
            pass
    if hasattr(block, "feed_forward") and hasattr(block.feed_forward, "_get_modules_dict"):
        try:
            mods.update(block.feed_forward._get_modules_dict())
        except Exception:
            pass
    if not mods:
        # 兜底：把整个 block 搬运（参数会一次性切换）
        mods["__block__"] = block
    return mods

class WeightStreamingManager:
    """
    权重流式管理器（最小侵入）：
      - CPU 常驻主副本（pinned），GPU 仅按层临时拷贝
      - 进入层 i 前：ensure_on_gpu(i) + wait_stream
      - 层内：异步预取 i+1 .. i+prefetch_distance（不等待）
      - LRU：最多保留 max_cached_layers 层在 GPU；逐出不D2H，只指针切回CPU主副本
    用法：
        wsm = WeightStreamingManager(model, device="cuda", prefetch_distance=1, max_cached_layers=4)
        # 创建后立刻生效（安装了每层 forward_pre_hook）
    """
    def __init__(self,
                 model: nn.Module,
                 device: str = "cuda",
                 prefetch_distance: int = 1,
                 max_cached_layers: int = 4,
                 layers_attr: str = "layers",
                 warmup_layers: int = 1,
                 verbose: bool = False):
        assert torch.cuda.is_available(), "CUDA not available"
        self.device = device
        self.verbose = verbose
        self.streams = get_streams(device)
        self.prefetch_distance = int(prefetch_distance)
        self.max_cached_layers = int(max_cached_layers)

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

        # 安装 forward_pre_hook：确保当前层 + 预取后续
        for i, blk in enumerate(self.blocks):
            blk.register_forward_pre_hook(self._pre_hook_factory(i))

        # （可选）预热前几层，降低第 0 层等待
        if warmup_layers > 0:
            warm = list(range(min(warmup_layers, len(self.blocks))))
            self.prefetch(warm)
            if self.verbose:
                print(f"[WSM] warmup prefetch: {warm}")

    # ============ 基元：参数/模块的 CPU/GPU 切换 ============

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
        if p.device.type != "cpu":
            return
        pid = id(p)
        self._ensure_param_cpu_stash_inplace(p)
        with torch.cuda.stream(self.streams.weight_h2d):
            p_gpu = self.cpu_stash[pid].to(self.device, non_blocking=True)
        p.data = p_gpu  # 切到 GPU

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
        if idx in self.gpu_cache:
            self.gpu_cache.move_to_end(idx)
        else:
            while len(self.gpu_cache) >= self.max_cached_layers:
                old, _ = self.gpu_cache.popitem(last=False)
                self._evict_layer_to_cpu(old)
            # H2D 当前层
            for _name, mod in self.block_mods[idx].items():
                self._ensure_module_on_gpu(mod)
            self.gpu_cache[idx] = None
            if self.verbose:
                print(f"[WSM] ->GPU layer={idx}")
        if wait:
            self.streams.wait_weight_ready_on_current()

    def prefetch(self, ids: List[int]):
        for idx in ids:
            if idx in self.gpu_cache:
                self.gpu_cache.move_to_end(idx)
                continue
            while len(self.gpu_cache) >= self.max_cached_layers:
                old, _ = self.gpu_cache.popitem(last=False)
                self._evict_layer_to_cpu(old)
            for _name, mod in self.block_mods[idx].items():
                self._ensure_module_on_gpu(mod)
            self.gpu_cache[idx] = None
            if self.verbose:
                print(f"[WSM] prefetch layer={idx}")

    def _evict_layer_to_cpu(self, idx: int):
        for _name, mod in self.block_mods[idx].items():
            self._evict_module_to_cpu(mod)
        if self.verbose:
            print(f"[WSM]   evict layer={idx}")
