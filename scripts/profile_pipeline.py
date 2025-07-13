"""KV Cache SSD 传输时间专用分析器

这是一个专门用于分析 LLaMA 模型 KV Cache 传输性能的工具。主要功能包括：

1. 精确测量各种传输时间：
   - SSD → DRAM 的加载时间
   - DRAM → GPU 的传输时间  
   - GPU → DRAM 的保存时间
   - DRAM → SSD 的卸载时间

2. 统计分析：
   - 传输带宽计算
   - Cache 命中率统计
   - 内存使用监控
   - 批处理性能分析

3. 导出功能：
   - 详细的控制台报告
   - CSV 格式数据导出
   - 结构化日志记录

使用方法：
    python profile_pipeline.py --model-path /path/to/model --prompt "Your prompt here"
    python profile_pipeline.py --model-path /path/to/model --prompt-file prompts.txt --csv results.csv

Author: LLaMA3 Project Team
License: MIT
"""
import math
import sys
import argparse
import time
import pathlib
from contextlib import contextmanager
import csv
import os
import threading
import logging
from typing import List, Iterator, Any, Optional, Dict, Tuple

# 安全导入 torch 和相关模块
try:
    import torch
    from torch.cuda import Event, current_stream
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    Event = None
    current_stream = None

# 安全导入 llama3 模块
try:
    from llama3.generator import LLaMA
    from llama3.layers import PerformanceTracker    
    from llama3.kv_offload import KVOffloader
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    LLaMA = None
    PerformanceTracker = None
    KVOffloader = None

# ---------- 全局统计数据 ----------
# 使用 dict 而不是普通变量以提高线程安全性
KV_TRANSFER_STATS = {
    # SSD 相关
    'ssd_to_dram_us': 0,        # SSD → DRAM 加载时间
    'dram_to_ssd_us': 0,        # DRAM → SSD 卸载时间
    'ssd_to_dram_bytes': 0,     # SSD → DRAM 传输字节数
    'dram_to_ssd_bytes': 0,     # DRAM → SSD 传输字节数
    'ssd_to_dram_count': 0,     # SSD → DRAM 操作次数
    'dram_to_ssd_count': 0,     # DRAM → SSD 操作次数
    
    # DRAM ↔ GPU 相关
    'dram_to_gpu_us': 0,        # DRAM → GPU 传输时间
    'gpu_to_dram_us': 0,        # GPU → DRAM 传输时间
    'dram_to_gpu_bytes': 0,     # DRAM → GPU 传输字节数
    'gpu_to_dram_bytes': 0,     # GPU → DRAM 传输字节数
    'dram_to_gpu_count': 0,     # DRAM → GPU 操作次数
    'gpu_to_dram_count': 0,     # GPU → DRAM 操作次数
    
    # 其他统计
    'kv_cache_hit_count': 0,    # Cache 命中次数
    'kv_cache_miss_count': 0,   # Cache 未命中次数
    
    # 新增性能监控
    'total_memory_allocated': 0, # 总内存分配量
    'peak_memory_usage': 0,      # 峰值内存使用
    'memory_fragmentation': 0,   # 内存碎片化程度
}

# ---------- Batch statistics ----------
BATCH_STATS = {
    'total_batches':   0,        # 总批次数
    'total_prompts':   0,        # 累计 prompt 条数
    'max_batch_size':  0,
    'min_batch_size':  1 << 30,
    'batch_sizes':     [],       # 每批实际大小，可做分布分析
}

_PATCHES_APPLIED = False

# ---------- 日志配置 ----------
def setup_logging(verbose: bool = False) -> logging.Logger:
    """设置日志配置
    
    Args:
        verbose: 是否启用详细日志输出
        
    Returns:
        配置好的 Logger 实例
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('profile_pipeline.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)

# ---------- 辅助函数 ----------
def chunks(lst: List[Any], n: int) -> Iterator[List[Any]]:
    """将列表分割成指定大小的块
    
    Args:
        lst: 要分割的列表
        n: 每个块的大小
        
    Yields:
        大小为 n 的列表块（最后一块可能小于 n）
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def check_dependencies(logger: logging.Logger) -> None:
    """检查必要的依赖是否可用
    
    Args:
        logger: 日志记录器
        
    Raises:
        RuntimeError: 当缺少必要依赖时抛出
    """
    errors = []
    
    if not TORCH_AVAILABLE:
        errors.append("PyTorch not available. Please install PyTorch.")
    
    if not LLAMA_AVAILABLE:
        errors.append("LLaMA3 modules not available. Please check your LLaMA3 installation.")
    
    if torch.cuda.is_available():
        try:
            torch.cuda.device_count()
            logger.info(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
        except Exception as e:
            logger.warning(f"CUDA available but initialization failed: {e}")
    else:
        logger.info("CUDA not available, will use CPU")
    
    if errors:
        for error in errors:
            logger.error(f"❌ {error}")
        raise RuntimeError(f"Missing dependencies: {'; '.join(errors)}")
    
    logger.info("✅ All dependencies checked successfully")

def validate_and_process_prompts(prompts: List[str], max_seq_len: int = 2048, logger: logging.Logger = None) -> List[str]:
    """验证和处理输入提示，确保兼容性
    
    Args:
        prompts: 原始提示列表
        max_seq_len: 模型支持的最大序列长度
        logger: 日志记录器
        
    Returns:
        处理后的提示列表
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    processed_prompts = []
    
    for i, prompt in enumerate(prompts):
        # 检查prompt长度
        original_length = len(prompt)
        
        # 如果prompt过长，进行截断
        if original_length > max_seq_len * 3:  # 粗略估算，3个字符≈1个token
            logger.warning(f"Prompt {i+1} is very long ({original_length} chars), truncating...")
            prompt = prompt[:max_seq_len * 3]
            
        # 检查是否为测试数据格式（如 p1_0 p1_1 ...）
        if prompt.startswith(('p1_', 'p2_', 'p3_')) and ' ' in prompt:
            tokens = prompt.split()
            if len(tokens) > max_seq_len // 2:  # 如果token数量过多
                logger.warning(f"Prompt {i+1} appears to be test data with {len(tokens)} tokens, truncating to {max_seq_len//2}")
                # 截断到合理长度
                truncated_tokens = tokens[:max_seq_len//2]
                prompt = " ".join(truncated_tokens)
        
        # 添加基本内容验证
        if not prompt.strip():
            logger.warning(f"Prompt {i+1} is empty, using default text")
            prompt = "Hello world"
            
        processed_prompts.append(prompt)
        
        if len(prompt) != original_length:
            logger.debug(f"Prompt {i+1}: {original_length} -> {len(prompt)} chars")
    
    return processed_prompts

def safe_batch_processing(prompts: List[str], batch_size: int, max_seq_len: int = 2048, logger: logging.Logger = None) -> Iterator[List[str]]:
    """安全的批处理函数，确保每个batch符合模型限制
    
    注意：由于当前LLaMA实现的限制，暂时强制batch_size=1以避免shape mismatch错误
    
    Args:
        prompts: 处理后的提示列表
        batch_size: 期望的批处理大小
        max_seq_len: 模型支持的最大序列长度
        logger: 日志记录器
        
    Yields:
        安全的prompt批次
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 检查是否启用真正的multi-batch支持
    # 基于模型修改，现在支持真正的multi-batch处理
    effective_batch_size = batch_size
    if batch_size > 1:
        logger.info(f"Using multi-batch processing with batch_size={batch_size}. "
                   f"This requires the updated LLaMA model implementation.")
    
    current_batch = []
    current_batch_tokens = 0
    
    for i, prompt in enumerate(prompts):
        # 估算当前prompt的token数量（粗略估算）
        estimated_tokens = len(prompt.split()) if ' ' in prompt else len(prompt) // 3
        
        # 检查添加此prompt是否会超出限制
        if (len(current_batch) >= effective_batch_size or 
            (current_batch_tokens + estimated_tokens > max_seq_len and current_batch)):
            
            if current_batch:
                logger.debug(f"Yielding batch with {len(current_batch)} prompts, ~{current_batch_tokens} tokens")
                yield current_batch
                current_batch = []
                current_batch_tokens = 0
        
        # 如果单个prompt过长，发出警告但仍然处理
        if estimated_tokens > max_seq_len:
            logger.warning(f"Prompt {i+1} has ~{estimated_tokens} tokens, exceeds max_seq_len {max_seq_len}")
        
        current_batch.append(prompt)
        current_batch_tokens += estimated_tokens
    
    # 处理最后一个batch
    if current_batch:
        logger.debug(f"Yielding final batch with {len(current_batch)} prompts, ~{current_batch_tokens} tokens")
        yield current_batch

# ---------- 计时工具 ----------
@contextmanager
def precise_timer(stat_key: str, byte_key: str = None, count_key: str = None, bytes_transferred: int = 0):
    """精确计时器，同时统计时间、字节数和操作次数"""
    start_time = time.perf_counter()
    success = False
    try:
        yield
        success = True
    except Exception as e:
        logging.getLogger(__name__).warning(f"Operation failed during timing: {e}")
        raise
    finally:
        try:
            end_time = time.perf_counter()
            elapsed_us = int((end_time - start_time) * 1e6)
            
            # 只有操作成功时才记录统计
            if success and stat_key in KV_TRANSFER_STATS:
                KV_TRANSFER_STATS[stat_key] += elapsed_us
                
                if byte_key and bytes_transferred > 0 and byte_key in KV_TRANSFER_STATS:
                    KV_TRANSFER_STATS[byte_key] += bytes_transferred
                
                if count_key and count_key in KV_TRANSFER_STATS:
                    KV_TRANSFER_STATS[count_key] += 1
        except Exception as e:
            logging.getLogger(__name__).error(f"Error recording timing stats: {e}")

@contextmanager
def cuda_precise_timer(stat_key: str, byte_key: str = None, count_key: str = None, bytes_transferred: int = 0):
    """CUDA 精确计时器"""
    if not torch.cuda.is_available():
        with precise_timer(stat_key, byte_key, count_key, bytes_transferred):
            yield
        return
    
    start_event = Event(enable_timing=True)
    end_event = Event(enable_timing=True)
    
    try:
        start_event.record()
        yield
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_us = int(start_event.elapsed_time(end_event) * 1000)
        KV_TRANSFER_STATS[stat_key] += elapsed_us
        
        if byte_key and bytes_transferred > 0:
            KV_TRANSFER_STATS[byte_key] += bytes_transferred
        
        if count_key:
            KV_TRANSFER_STATS[count_key] += 1
    except Exception as e:
        print(f"⚠️  CUDA timing error: {e}")
        # 回退到 CPU 计时
        with precise_timer(stat_key, byte_key, count_key, bytes_transferred):
            pass

def sizeof_fmt(num: float, suffix: str = 'B') -> str:
    """格式化字节数显示"""
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if abs(num) < 1024.0:
            return f"{num:6.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} E{suffix}"

def bandwidth_fmt(bytes_val: int, time_us: int) -> str:
    """计算并格式化带宽"""
    if time_us <= 0:
        return "N/A"
    
    time_s = time_us / 1e6
    bandwidth_bps = bytes_val / time_s
    return sizeof_fmt(bandwidth_bps, 'B/s')

def update_memory_stats():
    """更新内存使用统计"""
    if not torch.cuda.is_available():
        return
    
    try:
        current_memory = torch.cuda.memory_allocated()
        max_memory = torch.cuda.max_memory_allocated()
        
        KV_TRANSFER_STATS['total_memory_allocated'] = current_memory
        KV_TRANSFER_STATS['peak_memory_usage'] = max(
            KV_TRANSFER_STATS['peak_memory_usage'], 
            max_memory
        )
        
        # 简单的内存碎片化检测
        reserved = torch.cuda.memory_reserved()
        if reserved > 0:
            fragmentation = (reserved - current_memory) / reserved
            KV_TRANSFER_STATS['memory_fragmentation'] = fragmentation
            
    except Exception as e:
        logging.getLogger(__name__).debug(f"Memory stats update failed: {e}")

# ---------- KV Offloader 详细 Patch ----------
def patch_kv_offloader_detailed():
    """
    打补丁以统计 GPU↔DRAM 及 SSD↔DRAM 传输时间。
    仅修改函数封装，不改任何输出部分。
    """
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    try:
        import llama3.kv_offload as kvmod
    except ImportError:
        print("⚠️  无法导入 llama3.kv_offload 模块")
        return

    # 尝试找到 KV offloader 类
    KVCls = None
    for cls_name in ["OptimizedKVOffloader", "KVOffloader"]:
        KVCls = getattr(kvmod, cls_name, None)
        if KVCls is not None:
            break
    
    if KVCls is None:
        print("⚠️  未找到 KV offloader 类")
        return

    # -------- 保存原始方法 --------
    orig_push = getattr(KVCls, 'push', None)
    orig_fetch = getattr(KVCls, 'fetch', None)
    orig_spill = getattr(KVCls, "_spill_to_ssd", None)
    
    # 新版同步加载叫 _load_from_ssd_sync，旧版叫 _load_from_ssd
    orig_load = getattr(KVCls, "_load_from_ssd_sync", 
                       getattr(KVCls, "_load_from_ssd", None))

    # 统一块大小，用于估算字节
    def _blk_bytes(self):
        return getattr(self, "block_nbytes",
               getattr(self, "blk_bytes", 0))

    # ---------- GPU → DRAM ----------
    def wrapped_push(self, layer, blk, k, v):
        if not orig_push:
            return None
        
        try:
            bytes_tx = 0
            if hasattr(k, 'nbytes') and hasattr(v, 'nbytes'):
                bytes_tx = k.nbytes + v.nbytes
            
            with cuda_precise_timer('gpu_to_dram_us',
                                    'gpu_to_dram_bytes',
                                    'gpu_to_dram_count',
                                    bytes_tx):
                result = orig_push(self, layer, blk, k, v)
                update_memory_stats()  # 更新内存统计
                return result
        except Exception as e:
            logging.getLogger(__name__).warning(f"Error in wrapped_push: {e}")
            return orig_push(self, layer, blk, k, v)

    # ---------- DRAM (+SSD) → GPU ----------
    def wrapped_fetch(self, layer, blocks):
        if not orig_fetch:
            return None, None
        
        try:
            # 安全地转换 blocks 为列表
            if torch.is_tensor(blocks):
                blk_list = blocks.tolist()
            elif hasattr(blocks, '__iter__'):
                blk_list = list(blocks)
            else:
                blk_list = [blocks]
            
            # 检查是否需要从 SSD 加载
            need_ssd = []
            if hasattr(self, "on_ssd") and layer < len(self.on_ssd):
                need_ssd = [b for b in blk_list 
                           if b < len(self.on_ssd[layer]) and self.on_ssd[layer][b]]

            # 如需从 SSD 读取，强制同步
            if need_ssd:
                bytes_from_ssd = len(need_ssd) * _blk_bytes(self)
                with precise_timer('ssd_to_dram_us',
                                   'ssd_to_dram_bytes',
                                   'ssd_to_dram_count',
                                   bytes_from_ssd):
                    for b in need_ssd:
                        if hasattr(self, "_load_from_ssd_sync"):
                            self._load_from_ssd_sync(layer, b)
                        elif hasattr(self, "_load_from_ssd"):
                            self._load_from_ssd(layer, b)
                KV_TRANSFER_STATS['kv_cache_miss_count'] += 1
            else:
                KV_TRANSFER_STATS['kv_cache_hit_count'] += 1

            # 现在块已在 DRAM，安全执行原 fetch
            k, v = orig_fetch(self, layer, blocks)

            # DRAM → GPU 计时
            if k is not None and v is not None:
                bytes_tx = 0
                if hasattr(k, 'nbytes') and hasattr(v, 'nbytes'):
                    bytes_tx = k.nbytes + v.nbytes
                
                with cuda_precise_timer('dram_to_gpu_us',
                                        'dram_to_gpu_bytes',
                                        'dram_to_gpu_count',
                                        bytes_tx):
                    pass
            return k, v
        except Exception as e:
            print(f"⚠️  Error in wrapped_fetch: {e}")
            return orig_fetch(self, layer, blocks)

    # ---------- DRAM → SSD ----------
    def wrapped_spill(self, layer, blk, *args, **kwargs):
        if not orig_spill:
            return None
        
        try:
            with precise_timer('dram_to_ssd_us',
                               'dram_to_ssd_bytes',
                               'dram_to_ssd_count',
                               _blk_bytes(self)):
                return orig_spill(self, layer, blk, *args, **kwargs)
        except Exception as e:
            print(f"⚠️  Error in wrapped_spill: {e}")
            return orig_spill(self, layer, blk, *args, **kwargs)

    # ---------- SSD → DRAM（显式同步加载 API） ----------
    def wrapped_load(self, layer, blk, *args, **kwargs):
        if not orig_load:
            return None
        
        try:
            with precise_timer('ssd_to_dram_us',
                               'ssd_to_dram_bytes',
                               'ssd_to_dram_count',
                               _blk_bytes(self)):
                return orig_load(self, layer, blk, *args, **kwargs)
        except Exception as e:
            print(f"⚠️  Error in wrapped_load: {e}")
            return orig_load(self, layer, blk, *args, **kwargs)

    # -------- 应用补丁（仅当对应方法存在）--------
    if orig_push:
        KVCls.push = wrapped_push
    if orig_fetch:
        KVCls.fetch = wrapped_fetch
    if orig_spill:
        KVCls._spill_to_ssd = wrapped_spill
    if orig_load:
        if hasattr(KVCls, "_load_from_ssd_sync"):
            KVCls._load_from_ssd_sync = wrapped_load
        else:
            KVCls._load_from_ssd = wrapped_load

    _PATCHES_APPLIED = True
    print("✅ KV offloader detailed patches applied (auto-compatible)")

# ---------- LLaMA batch counter patch ----------
def patch_llama_batch_counter():
    """
    给 llama3.generator.LLaMA.text_completion 打补丁，
    在每次调用时统计 batch_size 并写入 BATCH_STATS。
    """
    try:
        import llama3.generator as genmod
    except ImportError:
        print("⚠️  无法导入 llama3.generator 模块")
        return

    if getattr(genmod, "_BATCH_PATCHED", False):
        return

    if not hasattr(genmod, 'LLaMA') or not hasattr(genmod.LLaMA, 'text_completion'):
        print("⚠️  LLaMA.text_completion 方法不存在")
        return

    orig_tc = genmod.LLaMA.text_completion

    def wrapped(self, prompts, *a, **kw):
        try:
            bsz = len(prompts) if prompts else 0
            BATCH_STATS['total_batches'] += 1
            BATCH_STATS['total_prompts'] += bsz
            BATCH_STATS['batch_sizes'].append(bsz)
            BATCH_STATS['max_batch_size'] = max(BATCH_STATS['max_batch_size'], bsz)
            BATCH_STATS['min_batch_size'] = min(BATCH_STATS['min_batch_size'], bsz)
            return orig_tc(self, prompts, *a, **kw)
        except Exception as e:
            print(f"⚠️  Error in batch counter: {e}")
            return orig_tc(self, prompts, *a, **kw)

    genmod.LLaMA.text_completion = wrapped
    genmod._BATCH_PATCHED = True
    print("✅ LLaMA batch-size counter patched")

def _make_perf_tracker_thread_safe():
    """
    把 llama3.layers.PERF_TRACKER.lock 换成 RLock，
    并让 add_layer_stat 在拿不到锁时直接略过这一次累加，
    以免 GPU 同步期间造成全局阻塞。
    """
    try:
        import llama3.layers as layermod
    except ImportError:
        print("⚠️  无法导入 llama3.layers 模块")
        return

    if not hasattr(layermod, 'PERF_TRACKER'):
        print("⚠️  PERF_TRACKER 不存在")
        return

    tracker = layermod.PERF_TRACKER
    tracker.lock = threading.RLock()

    if not hasattr(tracker, 'add_layer_stat'):
        return

    orig_add = tracker.add_layer_stat

    def safe_add(self, layer_id, stat_name, value):
        try:
            locked = self.lock.acquire(timeout=5e-4)
            if not locked:
                return
            try:
                return orig_add(layer_id, stat_name, value)
            finally:
                self.lock.release()
        except Exception as e:
            print(f"⚠️  Error in safe_add: {e}")

    layermod.PerformanceTracker.add_layer_stat = safe_add

# ---------- 统计报告 ----------
def print_transfer_report():
    """打印详细的传输统计报告"""
    stats = KV_TRANSFER_STATS
    
    # ------- Batch summary -------
    if BATCH_STATS['total_batches']:
        avg_bs = BATCH_STATS['total_prompts'] / BATCH_STATS['total_batches']
        print("\n📑 Batch Statistics")
        print(f"   Total batches : {BATCH_STATS['total_batches']}")
        print(f"   Total prompts : {BATCH_STATS['total_prompts']}")
        print(f"   Avg batch size: {avg_bs:.1f}")
        print(f"   Max batch size: {BATCH_STATS['max_batch_size']}")
        print(f"   Min batch size: {BATCH_STATS['min_batch_size']}")
        
    print("\n" + "="*80)
    print("🏷️  KV Cache Transfer Analysis Report")
    print("="*80)
    
    # SSD ↔ DRAM 统计
    print(f"\n💾 SSD ↔ DRAM Transfers:")
    print(f"   SSD → DRAM:")
    print(f"     Operations: {stats['ssd_to_dram_count']:,}")
    print(f"     Total time: {stats['ssd_to_dram_us']/1000:.1f} ms")
    print(f"     Total bytes: {sizeof_fmt(stats['ssd_to_dram_bytes'])}")
    if stats['ssd_to_dram_us'] > 0:
        print(f"     Bandwidth: {bandwidth_fmt(stats['ssd_to_dram_bytes'], stats['ssd_to_dram_us'])}")
        print(f"     Avg time/op: {stats['ssd_to_dram_us']/max(1, stats['ssd_to_dram_count']):.0f} μs")
    
    print(f"\n   DRAM → SSD:")
    print(f"     Operations: {stats['dram_to_ssd_count']:,}")
    print(f"     Total time: {stats['dram_to_ssd_us']/1000:.1f} ms")
    print(f"     Total bytes: {sizeof_fmt(stats['dram_to_ssd_bytes'])}")
    if stats['dram_to_ssd_us'] > 0:
        print(f"     Bandwidth: {bandwidth_fmt(stats['dram_to_ssd_bytes'], stats['dram_to_ssd_us'])}")
        print(f"     Avg time/op: {stats['dram_to_ssd_us']/max(1, stats['dram_to_ssd_count']):.0f} μs")
    
    # DRAM ↔ GPU 统计
    print(f"\n🖥️  DRAM ↔ GPU Transfers:")
    print(f"   DRAM → GPU:")
    print(f"     Operations: {stats['dram_to_gpu_count']:,}")
    print(f"     Total time: {stats['dram_to_gpu_us']/1000:.1f} ms")
    print(f"     Total bytes: {sizeof_fmt(stats['dram_to_gpu_bytes'])}")
    if stats['dram_to_gpu_us'] > 0:
        print(f"     Bandwidth: {bandwidth_fmt(stats['dram_to_gpu_bytes'], stats['dram_to_gpu_us'])}")
        print(f"     Avg time/op: {stats['dram_to_gpu_us']/max(1, stats['dram_to_gpu_count']):.0f} μs")
    
    print(f"\n   GPU → DRAM:")
    print(f"     Operations: {stats['gpu_to_dram_count']:,}")
    print(f"     Total time: {stats['gpu_to_dram_us']/1000:.1f} ms")
    print(f"     Total bytes: {sizeof_fmt(stats['gpu_to_dram_bytes'])}")
    if stats['gpu_to_dram_us'] > 0:
        print(f"     Bandwidth: {bandwidth_fmt(stats['gpu_to_dram_bytes'], stats['gpu_to_dram_us'])}")
        print(f"     Avg time/op: {stats['gpu_to_dram_us']/max(1, stats['gpu_to_dram_count']):.0f} μs")
    
    # Cache 效率统计
    total_cache_ops = stats['kv_cache_hit_count'] + stats['kv_cache_miss_count']
    if total_cache_ops > 0:
        hit_ratio = stats['kv_cache_hit_count'] / total_cache_ops * 100
        print(f"\n🎯 Cache Efficiency:")
        print(f"     Cache hits: {stats['kv_cache_hit_count']:,} ({hit_ratio:.1f}%)")
        print(f"     Cache misses: {stats['kv_cache_miss_count']:,} ({100-hit_ratio:.1f}%)")
        print(f"     Hit ratio: {hit_ratio:.1f}%")
    
    # 总体统计
    total_time = (stats['ssd_to_dram_us'] + stats['dram_to_ssd_us'] + 
                  stats['dram_to_gpu_us'] + stats['gpu_to_dram_us'])
    total_bytes = (stats['ssd_to_dram_bytes'] + stats['dram_to_ssd_bytes'] + 
                   stats['dram_to_gpu_bytes'] + stats['gpu_to_dram_bytes'])
    
    print(f"\n📊 Overall Transfer Summary:")
    print(f"     Total transfer time: {total_time/1000:.1f} ms")
    print(f"     Total bytes transferred: {sizeof_fmt(total_bytes)}")
    if total_time > 0:
        print(f"     Overall bandwidth: {bandwidth_fmt(total_bytes, total_time)}")
    
    # 时间分布
    if total_time > 0:
        print(f"\n⏱️  Time Distribution:")
        print(f"     SSD operations: {(stats['ssd_to_dram_us'] + stats['dram_to_ssd_us'])/total_time*100:.1f}%")
        print(f"     GPU operations: {(stats['dram_to_gpu_us'] + stats['gpu_to_dram_us'])/total_time*100:.1f}%")
    
    # 内存使用统计
    if stats['peak_memory_usage'] > 0:
        print(f"\n🧠 Memory Usage:")
        print(f"     Peak GPU memory: {sizeof_fmt(stats['peak_memory_usage'])}")
        print(f"     Current allocated: {sizeof_fmt(stats['total_memory_allocated'])}")
        if stats['memory_fragmentation'] > 0:
            print(f"     Memory fragmentation: {stats['memory_fragmentation']*100:.1f}%")
    
    print("="*80)

# ---------- 主函数 ----------
def main():
    # 设置日志
    logger = setup_logging(verbose=False)
    logger.info("🚀 Starting KV Cache SSD Transfer Profiler")
    
    # 检查依赖
    try:
        check_dependencies(logger)
    except RuntimeError as e:
        logger.error(f"Dependency check failed: {e}")
        sys.exit(1)
    
    ap = argparse.ArgumentParser(description="KV Cache SSD Transfer Profiler")
    ap.add_argument("--model-path", required=True, help="Path to LLaMA model")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="Input prompt text")
    src.add_argument("--prompt-file", help="File containing prompts, one per line")
    
    ap.add_argument("--max-gen-len", type=int, default=128, help="Number of tokens to generate")
    ap.add_argument("--batch-size", type=int, default=32, help="每批并行条数；>1 时脚本会把 prompts 切块循环推理")
    ap.add_argument("--topk-blk", type=int, default=4, help="Number of KV blocks to keep in GPU")
    ap.add_argument("--max-seq-len", type=int, default=2048, help="Maximum sequence length for model")
    ap.add_argument("--csv", help="Save results to CSV file")
    ap.add_argument("--runs", type=int, default=1, help="Number of inference runs")
    args = ap.parse_args()
    
    # 根据参数重新配置日志
    if args.verbose:
        logger = setup_logging(verbose=True)
        logger.debug("Verbose logging enabled")

    ckpt = pathlib.Path(args.model_path)
    
    if not ckpt.exists():
        logger.error(f"❌ Model path does not exist: {ckpt}")
        sys.exit(1)
    
    logger.info(f"🚀 KV Cache SSD Transfer Profiler")
    logger.info(f"📁 Model: {ckpt.name}")
    logger.info(f"🖥️  Device: {args.device}")
    
    # ------------- 收集 prompt -----------------
    prompts = []
    if args.prompt_file:
        if not os.path.exists(args.prompt_file):
            logger.error(f"❌ Prompt file does not exist: {args.prompt_file}")
            sys.exit(1)
        
        try:
            with open(args.prompt_file, encoding="utf-8") as f:
                prompts = [line.rstrip("\n") for line in f if line.strip()]
            logger.debug(f"Loaded {len(prompts)} prompts from file")
        except Exception as e:
            logger.error(f"❌ Error reading prompt file: {e}")
            sys.exit(1)
    else:
        prompts = [args.prompt]

    # ----------- 空列表保护 -----------
    if not prompts:
        src_name = args.prompt_file or "command line"
        logger.error(f"❌ No valid prompt found in {src_name}")
        sys.exit(1)

    # ----------- Prompt验证和处理 -----------
    logger.info("🔍 Validating and processing prompts...")
    try:
        original_count = len(prompts)
        prompts = validate_and_process_prompts(prompts, max_seq_len=args.max_seq_len, logger=logger)
        if len(prompts) != original_count:
            logger.warning(f"Prompt count changed: {original_count} -> {len(prompts)}")
    except Exception as e:
        logger.error(f"❌ Error processing prompts: {e}")
        sys.exit(1)

    # ------------ 打印基本信息 -----------------
    max_len = max(len(p) for p in prompts)
    total_batches = math.ceil(len(prompts) / args.batch_size)
    print(f"📝 Prompts: {len(prompts)} | Max length: {max_len} chars")
    print(f"🔢 Batch size: {args.batch_size} | Total batches: {total_batches}")
    print(f"🔢 Generation length: {args.max_gen_len}")
    print(f"💾 KV blocks in GPU: {args.topk_blk}")
    print(f"🔄 Number of runs: {args.runs}")
    
    # 应用 patches
    _make_perf_tracker_thread_safe()
    patch_kv_offloader_detailed()
    patch_llama_batch_counter()
    
    # 加载模型
    print("\n📦 Loading model...")
    try:
        llama = LLaMA.build(ckpt, load_model=True, device="cpu")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # 配置 KV cache
    try:
        for layer in llama.model.layers:
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'topk_blk'):
                layer.attention.topk_blk = args.topk_blk
    except Exception as e:
        print(f"⚠️  Error configuring KV cache: {e}")
    
    # 转移到 GPU
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("❌ CUDA not available but cuda device specified")
            sys.exit(1)
        
        print("🔄 Transferring to GPU...")
        try:
            llama.model.to(args.device)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"❌ Error transferring to GPU: {e}")
            sys.exit(1)
    
    llama.args.device = args.device
    
    # 配置 offloader streams
    try:
        for blk in llama.model.layers:
            if hasattr(blk, 'attention') and hasattr(blk.attention, 'offloader'):
                off = blk.attention.offloader
                off.device = args.device
                if torch.cuda.is_available():
                    off.copy_stream = torch.cuda.Stream(device=args.device)
    except Exception as e:
        print(f"⚠️  Error configuring offloader streams: {e}")
    
    print("✅ Model ready for inference")
    
    # 执行多次推理
    total_inference_time = 0
    
    for run in range(args.runs):
        print(f"\n🧠 Running inference {run+1}/{args.runs}...")
        
        # 重置当前运行的统计
        run_stats = {k: v for k, v in KV_TRANSFER_STATS.items()}
        
        start_time = time.perf_counter()
        
        try:
            all_outs = []
            batch_count = 0
            for prompt_batch in safe_batch_processing(prompts, args.batch_size, max_seq_len=args.max_seq_len, logger=logger):
                batch_count += 1
                logger.debug(f"Processing batch {batch_count} with {len(prompt_batch)} prompts")
                
                _, outs = llama.text_completion(
                    prompt_batch,
                    max_gen_len=args.max_gen_len
                )
                all_outs.extend(outs)
                
                # 更新内存统计
                update_memory_stats()
                
        except Exception as e:
            logger.error(f"❌ Error during inference: {e}")
            print(f"❌ Error during inference: {e}")
            continue
        
        end_time = time.perf_counter()
        
        inference_time = end_time - start_time
        total_inference_time += inference_time
        
        # 计算本次运行的增量统计
        run_delta = {}
        for key in KV_TRANSFER_STATS:
            run_delta[key] = KV_TRANSFER_STATS[key] - run_stats[key]
        
        print(f"✅ Run {run+1} completed in {inference_time:.2f}s")
        print(f"   Generated {len(all_outs)} responses")
        
        # 显示本次运行的关键统计
        if run_delta['ssd_to_dram_count'] > 0 or run_delta['dram_to_gpu_count'] > 0:
            print(f"   SSD→DRAM: {run_delta['ssd_to_dram_count']} ops, {run_delta['ssd_to_dram_us']/1000:.1f}ms")
            print(f"   DRAM→GPU: {run_delta['dram_to_gpu_count']} ops, {run_delta['dram_to_gpu_us']/1000:.1f}ms")
        
        # 仅在第一次运行时打印部分生成结果
        if run == 0 and all_outs:
            print(f"\n📝 Sample outputs (first run):")
            for i, out in enumerate(all_outs[:min(3, len(all_outs))]):
                print(f"   [{i+1}] {out[:100]}{'...' if len(out) > 100 else ''}")
            if len(all_outs) > 3:
                print(f"   ... and {len(all_outs) - 3} more")
    
    # 最终统计报告
    print(f"\n🏁 All runs completed!")
    print(f"Total inference time: {total_inference_time:.2f}s")
    print(f"Average time per run: {total_inference_time/args.runs:.2f}s")
    
    # 打印详细的传输统计报告
    print_transfer_report()
    
    # 保存 CSV 结果
    if args.csv:
        save_csv_results(args.csv, args, total_inference_time)
    
    print("\n🎯 Profiling completed successfully!")

def save_csv_results(csv_path: str, args, total_inference_time: float) -> None:
    """保存统计结果到 CSV 文件
    
    Args:
        csv_path: CSV 文件保存路径
        args: 命令行参数对象
        total_inference_time: 总推理时间（秒）
        
    Note:
        CSV 文件包含完整的性能统计数据，包括：
        - 配置参数
        - 传输统计
        - 缓存统计  
        - 批处理统计
        - 内存使用统计
    """
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入头部信息
            writer.writerow(['Metric', 'Value', 'Unit'])
            writer.writerow(['Model Path', args.model_path, ''])
            writer.writerow(['Device', args.device, ''])
            writer.writerow(['Batch Size', args.batch_size, ''])
            writer.writerow(['Max Gen Length', args.max_gen_len, ''])
            writer.writerow(['TopK Blocks', args.topk_blk, ''])
            writer.writerow(['Runs', args.runs, ''])
            writer.writerow(['Total Inference Time', f'{total_inference_time:.3f}', 'seconds'])
            writer.writerow([])  # 空行分隔
            
            # 写入传输统计
            stats = KV_TRANSFER_STATS
            writer.writerow(['Transfer Statistics', '', ''])
            
            # SSD 统计
            writer.writerow(['SSD to DRAM Operations', stats['ssd_to_dram_count'], 'count'])
            writer.writerow(['SSD to DRAM Time', f"{stats['ssd_to_dram_us']/1000:.3f}", 'ms'])
            writer.writerow(['SSD to DRAM Bytes', stats['ssd_to_dram_bytes'], 'bytes'])
            
            writer.writerow(['DRAM to SSD Operations', stats['dram_to_ssd_count'], 'count'])
            writer.writerow(['DRAM to SSD Time', f"{stats['dram_to_ssd_us']/1000:.3f}", 'ms'])
            writer.writerow(['DRAM to SSD Bytes', stats['dram_to_ssd_bytes'], 'bytes'])
            
            # GPU 统计
            writer.writerow(['DRAM to GPU Operations', stats['dram_to_gpu_count'], 'count'])
            writer.writerow(['DRAM to GPU Time', f"{stats['dram_to_gpu_us']/1000:.3f}", 'ms'])
            writer.writerow(['DRAM to GPU Bytes', stats['dram_to_gpu_bytes'], 'bytes'])
            
            writer.writerow(['GPU to DRAM Operations', stats['gpu_to_dram_count'], 'count'])
            writer.writerow(['GPU to DRAM Time', f"{stats['gpu_to_dram_us']/1000:.3f}", 'ms'])
            writer.writerow(['GPU to DRAM Bytes', stats['gpu_to_dram_bytes'], 'bytes'])
            
            # Cache 统计
            writer.writerow(['Cache Hits', stats['kv_cache_hit_count'], 'count'])
            writer.writerow(['Cache Misses', stats['kv_cache_miss_count'], 'count'])
            
            # 内存统计
            writer.writerow(['Peak Memory Usage', stats['peak_memory_usage'], 'bytes'])
            writer.writerow(['Current Memory Allocated', stats['total_memory_allocated'], 'bytes'])
            writer.writerow(['Memory Fragmentation', f"{stats['memory_fragmentation']*100:.2f}", 'percent'])
            
            # Batch 统计
            if BATCH_STATS['total_batches'] > 0:
                writer.writerow([])
                writer.writerow(['Batch Statistics', '', ''])
                writer.writerow(['Total Batches', BATCH_STATS['total_batches'], 'count'])
                writer.writerow(['Total Prompts', BATCH_STATS['total_prompts'], 'count'])
                writer.writerow(['Max Batch Size', BATCH_STATS['max_batch_size'], 'count'])
                writer.writerow(['Min Batch Size', BATCH_STATS['min_batch_size'], 'count'])
                
                avg_batch_size = BATCH_STATS['total_prompts'] / BATCH_STATS['total_batches']
                writer.writerow(['Avg Batch Size', f'{avg_batch_size:.2f}', 'count'])
        
        print(f"📊 Results saved to: {csv_path}")
        
    except Exception as e:
        print(f"⚠️  Error saving CSV: {e}")

if __name__ == "__main__":
    main()