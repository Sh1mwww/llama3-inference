# (完整文件) profile_pipeline.py
"""KV Cache SSD 传输时间专用分析器
...（顶部注释保持不变）...
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
from typing import List, Iterator, Any

# 按层权重流式 & 流管理
from llama3.weight_streaming_manager import WeightStreamingManager
import llama3.stream_mnt as _stream_mnt

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
KV_TRANSFER_STATS = {
    'ssd_to_dram_us': 0, 'dram_to_ssd_us': 0,
    'ssd_to_dram_bytes': 0, 'dram_to_ssd_bytes': 0,
    'ssd_to_dram_count': 0, 'dram_to_ssd_count': 0,
    'dram_to_gpu_us': 0, 'gpu_to_dram_us': 0,
    'dram_to_gpu_bytes': 0, 'gpu_to_dram_bytes': 0,
    'dram_to_gpu_count': 0, 'gpu_to_dram_count': 0,
    'kv_cache_hit_count': 0, 'kv_cache_miss_count': 0,
    'total_memory_allocated': 0, 'peak_memory_usage': 0, 'memory_fragmentation': 0,
}

# ---------- Batch statistics ----------
BATCH_STATS = {
    'total_batches':   0,
    'total_prompts':   0,
    'max_batch_size':  0,
    'min_batch_size':  1 << 30,
    'batch_sizes':     [],
}

_PATCHES_APPLIED = False

# ---------- 日志配置 ----------
def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler('profile_pipeline.log', mode='a')]
    )
    return logging.getLogger(__name__)

# ---------- 辅助函数 ----------
def chunks(lst: List[Any], n: int) -> Iterator[List[Any]]:
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def check_dependencies(logger: logging.Logger) -> None:
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
        for error in errors: logger.error(f"❌ {error}")
        raise RuntimeError(f"Missing dependencies: {'; '.join(errors)}")
    logger.info("✅ All dependencies checked successfully")

def validate_and_process_prompts(prompts: List[str], max_seq_len: int = 2048, logger: logging.Logger = None) -> List[str]:
    logger = logger or logging.getLogger(__name__)
    processed = []
    for i, prompt in enumerate(prompts):
        original_length = len(prompt)
        if original_length > max_seq_len * 3:
            logger.warning(f"Prompt {i+1} is very long ({original_length} chars), truncating...")
            prompt = prompt[:max_seq_len * 3]
        if prompt.startswith(('p1_', 'p2_', 'p3_')) and ' ' in prompt:
            toks = prompt.split()
            if len(toks) > max_seq_len // 2:
                logger.warning(f"Prompt {i+1} appears to be test data with {len(toks)} tokens, truncating to {max_seq_len//2}")
                prompt = " ".join(toks[:max_seq_len//2])
        if not prompt.strip():
            logger.warning(f"Prompt {i+1} is empty, using default text")
            prompt = "Hello world"
        processed.append(prompt)
        if len(prompt) != original_length:
            logger.debug(f"Prompt {i+1}: {original_length} -> {len(prompt)} chars")
    return processed

def safe_batch_processing(prompts: List[str], batch_size: int, max_seq_len: int = 2048, logger: logging.Logger = None) -> Iterator[List[str]]:
    logger = logger or logging.getLogger(__name__)
    effective_batch_size = batch_size
    if batch_size > 1:
        logger.info(f"Using multi-batch processing with batch_size={batch_size}. This requires the updated LLaMA model implementation.")
    current_batch, current_tokens = [], 0
    for i, prompt in enumerate(prompts):
        est_tokens = len(prompt.split()) if ' ' in prompt else len(prompt) // 3
        if (len(current_batch) >= effective_batch_size or (current_tokens + est_tokens > max_seq_len and current_batch)):
            if current_batch:
                logger.debug(f"Yielding batch with {len(current_batch)} prompts, ~{current_tokens} tokens")
                yield current_batch
                current_batch, current_tokens = [], 0
        if est_tokens > max_seq_len:
            logger.warning(f"Prompt {i+1} has ~{est_tokens} tokens, exceeds max_seq_len {max_seq_len}")
        current_batch.append(prompt)
        current_tokens += est_tokens
    if current_batch:
        logger.debug(f"Yielding final batch with {len(current_batch)} prompts, ~{current_tokens} tokens")
        yield current_batch

# ---------- 计时工具 ----------
@contextmanager
def precise_timer(stat_key: str, byte_key: str = None, count_key: str = None, bytes_transferred: int = 0):
    t0 = time.perf_counter(); ok = False
    try:
        yield; ok = True
    finally:
        try:
            dt_us = int((time.perf_counter() - t0) * 1e6)
            if ok and stat_key in KV_TRANSFER_STATS:
                KV_TRANSFER_STATS[stat_key] += dt_us
                if byte_key and bytes_transferred > 0 and byte_key in KV_TRANSFER_STATS:
                    KV_TRANSFER_STATS[byte_key] += bytes_transferred
                if count_key and count_key in KV_TRANSFER_STATS:
                    KV_TRANSFER_STATS[count_key] += 1
        except Exception as e:
            logging.getLogger(__name__).error(f"Error recording timing stats: {e}")

@contextmanager
def cuda_precise_timer(stat_key: str, byte_key: str = None, count_key: str = None, bytes_transferred: int = 0):
    if not torch.cuda.is_available():
        with precise_timer(stat_key, byte_key, count_key, bytes_transferred): yield
        return
    start_event = Event(enable_timing=True); end_event = Event(enable_timing=True)
    try:
        start_event.record(); yield; end_event.record(); torch.cuda.synchronize()
        elapsed_us = int(start_event.elapsed_time(end_event) * 1000)
        KV_TRANSFER_STATS[stat_key] += elapsed_us
        if byte_key and bytes_transferred > 0: KV_TRANSFER_STATS[byte_key] += bytes_transferred
        if count_key: KV_TRANSFER_STATS[count_key] += 1
    except Exception:
        with precise_timer(stat_key, byte_key, count_key, bytes_transferred): pass

def sizeof_fmt(num: float, suffix: str = 'B') -> str:
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if abs(num) < 1024.0: return f"{num:6.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} E{suffix}"

def bandwidth_fmt(bytes_val: int, time_us: int) -> str:
    if time_us <= 0: return "N/A"
    return sizeof_fmt(bytes_val / (time_us / 1e6), 'B/s')

def update_memory_stats():
    if not torch.cuda.is_available(): return
    try:
        cur = torch.cuda.memory_allocated(); peak = torch.cuda.max_memory_allocated()
        KV_TRANSFER_STATS['total_memory_allocated'] = cur
        KV_TRANSFER_STATS['peak_memory_usage'] = max(KV_TRANSFER_STATS['peak_memory_usage'], peak)
        reserved = torch.cuda.memory_reserved()
        if reserved > 0:
            KV_TRANSFER_STATS['memory_fragmentation'] = (reserved - cur) / reserved
    except Exception as e:
        logging.getLogger(__name__).debug(f"Memory stats update failed: {e}")

# ---------- KV Offloader 详细 Patch ----------
def patch_kv_offloader_detailed():
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED: return
    try:
        import llama3.kv_offload as kvmod
    except ImportError:
        print("⚠️  无法导入 llama3.kv_offload 模块"); return
    KVCls = None
    for cls_name in ["OptimizedKVOffloader", "KVOffloader"]:
        KVCls = getattr(kvmod, cls_name, None)
        if KVCls is not None: break
    if KVCls is None:
        print("⚠️  未找到 KV offloader 类"); return

    orig_push = getattr(KVCls, 'push', None)
    orig_fetch = getattr(KVCls, 'fetch', None)
    orig_spill = getattr(KVCls, "_spill_to_ssd", None)
    orig_load  = getattr(KVCls, "_load_from_ssd_sync", getattr(KVCls, "_load_from_ssd", None))

    def _blk_bytes(self):
        return getattr(self, "block_nbytes", getattr(self, "blk_bytes", 0))

    def wrapped_push(self, layer, blk, k, v, *args, **kwargs):
        if not orig_push: return None
        try:
            bytes_tx = (getattr(k, 'nbytes', 0) + getattr(v, 'nbytes', 0))
            with cuda_precise_timer('gpu_to_dram_us', 'gpu_to_dram_bytes', 'gpu_to_dram_count', bytes_tx):
                out = orig_push(self, layer, blk, k, v, *args, **kwargs)
                update_memory_stats(); return out
        except Exception:
            return orig_push(self, layer, blk, k, v, *args, **kwargs)

    def wrapped_fetch(self, layer, blocks):
        if not orig_fetch: return (None, None)
        try:
            if torch.is_tensor(blocks): blk_list = blocks.tolist()
            elif hasattr(blocks, '__iter__'): blk_list = list(blocks)
            else: blk_list = [blocks]
            need_ssd = []
            if hasattr(self, "on_ssd") and layer < len(self.on_ssd):
                need_ssd = [b for b in blk_list if b < len(self.on_ssd[layer]) and self.on_ssd[layer][b]]
            if need_ssd:
                with precise_timer('ssd_to_dram_us', 'ssd_to_dram_bytes', 'ssd_to_dram_count', len(need_ssd) * _blk_bytes(self)):
                    for b in need_ssd:
                        if hasattr(self, "_load_from_ssd_sync"): self._load_from_ssd_sync(layer, b)
                        elif hasattr(self, "_load_from_ssd"):   self._load_from_ssd(layer, b)
                KV_TRANSFER_STATS['kv_cache_miss_count'] += 1
            else:
                KV_TRANSFER_STATS['kv_cache_hit_count'] += 1
            k, v = orig_fetch(self, layer, blocks)
            if k is not None and v is not None:
                bytes_tx = getattr(k, 'nbytes', 0) + getattr(v, 'nbytes', 0)
                with cuda_precise_timer('dram_to_gpu_us', 'dram_to_gpu_bytes', 'dram_to_gpu_count', bytes_tx):
                    pass
            return k, v
        except Exception:
            return orig_fetch(self, layer, blocks)

    def wrapped_spill(self, layer, blk, *args, **kwargs):
        if not orig_spill: return None
        try:
            with precise_timer('dram_to_ssd_us', 'dram_to_ssd_bytes', 'dram_to_ssd_count', _blk_bytes(self)):
                return orig_spill(self, layer, blk, *args, **kwargs)
        except Exception:
            return orig_spill(self, layer, blk, *args, **kwargs)

    def wrapped_load(self, layer, blk, *args, **kwargs):
        if not orig_load: return None
        try:
            with precise_timer('ssd_to_dram_us', 'ssd_to_dram_bytes', 'ssd_to_dram_count', _blk_bytes(self)):
                return orig_load(self, layer, blk, *args, **kwargs)
        except Exception:
            return orig_load(self, layer, blk, *args, **kwargs)

    if orig_push:  KVCls.push = wrapped_push
    if orig_fetch: KVCls.fetch = wrapped_fetch
    if orig_spill: KVCls._spill_to_ssd = wrapped_spill
    if orig_load:
        if hasattr(KVCls, "_load_from_ssd_sync"): KVCls._load_from_ssd_sync = wrapped_load
        else:                                      KVCls._load_from_ssd      = wrapped_load

    _PATCHES_APPLIED = True
    print("✅ KV offloader detailed patches applied (auto-compatible)")

# ---------- LLaMA batch counter patch（纯 Python，无 torch 张量） ----------
def patch_llama_batch_counter():
    try:
        import llama3.generator as genmod
    except ImportError:
        print("⚠️  无法导入 llama3.generator 模块"); return
    if getattr(genmod, "_BATCH_PATCHED", False): return
    if not hasattr(genmod, 'LLaMA') or not hasattr(genmod.LLaMA, 'text_completion'):
        print("⚠️  LLaMA.text_completion 方法不存在"); return

    orig_tc = genmod.LLaMA.text_completion

    def wrapped(self, prompts, *a, **kw):
        bsz = len(prompts) if prompts else 0
        BATCH_STATS['total_batches'] += 1
        BATCH_STATS['total_prompts'] += bsz
        BATCH_STATS['batch_sizes'].append(bsz)
        BATCH_STATS['max_batch_size'] = max(BATCH_STATS['max_batch_size'], bsz)
        BATCH_STATS['min_batch_size'] = min(BATCH_STATS['min_batch_size'], bsz)
        return orig_tc(self, prompts, *a, **kw)

    genmod.LLaMA.text_completion = wrapped
    genmod._BATCH_PATCHED = True
    print("✅ LLaMA batch-size counter patched")

def _make_perf_tracker_thread_safe():
    try:
        import llama3.layers as layermod
    except ImportError:
        print("⚠️  无法导入 llama3.layers 模块"); return
    if not hasattr(layermod, 'PERF_TRACKER'):
        print("⚠️  PERF_TRACKER 不存在"); return
    tracker = layermod.PERF_TRACKER
    tracker.lock = threading.RLock()
    if not hasattr(tracker, 'add_layer_stat'): return
    orig_add = tracker.add_layer_stat
    def safe_add(self, layer_id, stat_name, value):
        try:
            if not self.lock.acquire(timeout=5e-4): return
            try: return orig_add(layer_id, stat_name, value)
            finally: self.lock.release()
        except Exception: pass
    layermod.PerformanceTracker.add_layer_stat = safe_add

# ---------- 统计报告 ----------
def print_transfer_report():
    s = KV_TRANSFER_STATS
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
    print(f"\n💾 SSD ↔ DRAM Transfers:")
    print(f"   SSD → DRAM:")
    print(f"     Operations: {s['ssd_to_dram_count']:,}")
    print(f"     Total time: {s['ssd_to_dram_us']/1000:.1f} ms")
    print(f"     Total bytes: {sizeof_fmt(s['ssd_to_dram_bytes'])}")
    if s['ssd_to_dram_us'] > 0:
        print(f"     Bandwidth: {bandwidth_fmt(s['ssd_to_dram_bytes'], s['ssd_to_dram_us'])}")
        print(f"     Avg time/op: {s['ssd_to_dram_us']/max(1, s['ssd_to_dram_count']):.0f} μs")
    print(f"\n   DRAM → SSD:")
    print(f"     Operations: {s['dram_to_ssd_count']:,}")
    print(f"     Total time: {s['dram_to_ssd_us']/1000:.1f} ms")
    print(f"     Total bytes: {sizeof_fmt(s['dram_to_ssd_bytes'])}")
    if s['dram_to_ssd_us'] > 0:
        print(f"     Bandwidth: {bandwidth_fmt(s['dram_to_ssd_bytes'], s['dram_to_ssd_us'])}")
        print(f"     Avg time/op: {s['dram_to_ssd_us']/max(1, s['dram_to_ssd_count']):.0f} μs")
    print(f"\n🖥️  DRAM ↔ GPU Transfers:")
    print(f"   DRAM → GPU:")
    print(f"     Operations: {s['dram_to_gpu_count']:,}")
    print(f"     Total time: {s['dram_to_gpu_us']/1000:.1f} ms")
    print(f"     Total bytes: {sizeof_fmt(s['dram_to_gpu_bytes'])}")
    if s['dram_to_gpu_us'] > 0:
        print(f"     Bandwidth: {bandwidth_fmt(s['dram_to_gpu_bytes'], s['dram_to_gpu_us'])}")
        print(f"     Avg time/op: {s['dram_to_gpu_us']/max(1, s['dram_to_gpu_count']):.0f} μs")
    print(f"\n   GPU → DRAM:")
    print(f"     Operations: {s['gpu_to_dram_count']:,}")
    print(f"     Total time: {s['gpu_to_dram_us']/1000:.1f} ms")
    print(f"     Total bytes: {sizeof_fmt(s['gpu_to_dram_bytes'])}")
    if s['gpu_to_dram_us'] > 0:
        print(f"     Bandwidth: {bandwidth_fmt(s['gpu_to_dram_bytes'], s['gpu_to_dram_us'])}")
        print(f"     Avg time/op: {s['gpu_to_dram_us']/max(1, s['gpu_to_dram_count']):.0f} μs")
    total_time = (s['ssd_to_dram_us'] + s['dram_to_ssd_us'] + s['dram_to_gpu_us'] + s['gpu_to_dram_us'])
    total_bytes = (s['ssd_to_dram_bytes'] + s['dram_to_ssd_bytes'] + s['dram_to_gpu_bytes'] + s['gpu_to_dram_bytes'])
    print(f"\n📊 Overall Transfer Summary:")
    print(f"     Total transfer time: {total_time/1000:.1f} ms")
    print(f"     Total bytes transferred: {sizeof_fmt(total_bytes)}")
    if total_time > 0:
        print(f"     Overall bandwidth: {bandwidth_fmt(total_bytes, total_time)}")
        print(f"\n⏱️  Time Distribution:")
        print(f"     SSD operations: {(s['ssd_to_dram_us'] + s['dram_to_ssd_us'])/total_time*100:.1f}%")
        print(f"     GPU operations: {(s['dram_to_gpu_us'] + s['gpu_to_dram_us'])/total_time*100:.1f}%")
    if s['peak_memory_usage'] > 0:
        print(f"\n🧠 Memory Usage:")
        print(f"     Peak GPU memory: {sizeof_fmt(s['peak_memory_usage'])}")
        print(f"     Current allocated: {sizeof_fmt(s['total_memory_allocated'])}")
        if s['memory_fragmentation'] > 0:
            print(f"     Memory fragmentation: {s['memory_fragmentation']*100:.1f}%")
    print("="*80)

# ---------- 主函数 ----------
def main():
    logger = setup_logging(verbose=False)
    logger.info("🚀 Starting KV Cache SSD Transfer Profiler")
    try:
        check_dependencies(logger)
    except RuntimeError as e:
        logger.error(f"Dependency check failed: {e}"); sys.exit(1)

    ap = argparse.ArgumentParser(description="KV Cache SSD Transfer Profiler")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--verbose", "-v", action="store_true")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt")
    src.add_argument("--prompt-file")
    ap.add_argument("--max-gen-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--topk-blk", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--csv")
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    if args.verbose:
        logger = setup_logging(verbose=True)
        logger.debug("Verbose logging enabled")

    ckpt = pathlib.Path(args.model_path)
    if not ckpt.exists():
        logger.error(f"❌ Model path does not exist: {ckpt}"); sys.exit(1)

    logger.info(f"🚀 KV Cache SSD Transfer Profiler")
    logger.info(f"📁 Model: {ckpt.name}")
    logger.info(f"🖥️  Device: {args.device}")

    prompts = []
    if args.prompt_file:
        if not os.path.exists(args.prompt_file):
            logger.error(f"❌ Prompt file does not exist: {args.prompt_file}"); sys.exit(1)
        try:
            with open(args.prompt_file, encoding="utf-8") as f:
                prompts = [line.rstrip("\n") for line in f if line.strip()]
        except Exception as e:
            logger.error(f"❌ Error reading prompt file: {e}"); sys.exit(1)
    else:
        prompts = [args.prompt]

    if not prompts:
        logger.error("❌ No valid prompt found"); sys.exit(1)

    logger.info("🔍 Validating and processing prompts...")
    try:
        original_count = len(prompts)
        prompts = validate_and_process_prompts(prompts, max_seq_len=args.max_seq_len, logger=logger)
        if len(prompts) != original_count:
            logger.warning(f"Prompt count changed: {original_count} -> {len(prompts)}")
    except Exception as e:
        logger.error(f"❌ Error processing prompts: {e}"); sys.exit(1)

    max_len = max(len(p) for p in prompts)
    total_batches = math.ceil(len(prompts) / args.batch_size)
    print(f"📝 Prompts: {len(prompts)} | Max length: {max_len} chars")
    print(f"🔢 Batch size: {args.batch_size} | Total batches: {total_batches}")
    print(f"🔢 Generation length: {args.max_gen_len}")
    print(f"💾 KV blocks in GPU: {args.topk_blk}")
    print(f"🔄 Number of runs: {args.runs}")

    _make_perf_tracker_thread_safe()
    patch_kv_offloader_detailed()
    patch_llama_batch_counter()

    print("\n📦 Loading model...")
    try:
        # 配置权重流式传输参数
        enable_streaming = args.device.startswith("cuda")
        streaming_config = {
            'prefetch_distance': 1,
            'max_cached_layers': 4,
            'warmup_layers': 1,
            'verbose': args.verbose
        } if enable_streaming else None
        
        # 使用新的 LLaMA.build() API 进行统一构建
        llama = LLaMA.build(
            ckpt, 
            load_model=True, 
            device=args.device,
            enable_weight_streaming=enable_streaming,
            streaming_config=streaming_config,
            topk_blk=args.topk_blk,
            max_seq_len=args.max_seq_len,
            max_batch_size=args.batch_size
        )
        
        if not hasattr(llama, 'model') or llama.model is None:
            raise RuntimeError("Model failed to load properly - llama.model is None")
        print(f"✅ Model loaded successfully: {type(llama.model)}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}"); sys.exit(1)
    
    print("✅ Model ready for inference")

    # 执行多次推理
    total_inference_time = 0
    for run in range(args.runs):
        print(f"\n🧠 Running inference {run+1}/{args.runs}...")
        run_stats = {k: v for k, v in KV_TRANSFER_STATS.items()}
        t0 = time.perf_counter()
        try:
            all_outs = []
            batch_count = 0
            total_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
            try:
                from llama3.global_state_tracker import get_global_tracker
                tracker = get_global_tracker()
                if tracker:
                    tracker.register_future_batch(list(range(total_batches)))
                    print(f"[INFO] Pre-registered {total_batches} batches: {list(range(total_batches))}")
            except Exception as e:
                print(f"[WARNING] Failed to pre-register batches: {e}")

            for prompt_batch in safe_batch_processing(prompts, args.batch_size, max_seq_len=args.max_seq_len, logger=logger):
                batch_count += 1
                print(f"[INFO] Processing batch {batch_count}/{total_batches} with {len(prompt_batch)} prompts")
                logger.debug(f"Processing batch {batch_count} with {len(prompt_batch)} prompts")
                try:
                    tracker = get_global_tracker()
                    if tracker: tracker.set_current_execution(batch_count - 1, 0)
                except Exception as e:
                    print(f"[WARNING] Failed to set current batch: {e}")
                _, outs = llama.text_completion(prompt_batch, max_gen_len=args.max_gen_len)
                all_outs.extend(outs)
                update_memory_stats()
        except Exception as e:
            logger.error(f"❌ Error during inference: {e}")
            print(f"❌ Error during inference: {e}")
            continue
        dt = time.perf_counter() - t0
        total_inference_time += dt
        delta = {k: KV_TRANSFER_STATS[k] - run_stats[k] for k in KV_TRANSFER_STATS}
        print(f"✅ Run {run+1} completed in {dt:.2f}s")
        print(f"   Generated {len(all_outs)} responses")
        if delta['ssd_to_dram_count'] > 0 or delta['dram_to_gpu_count'] > 0:
            print(f"   SSD→DRAM: {delta['ssd_to_dram_count']} ops, {delta['ssd_to_dram_us']/1000:.1f}ms")
            print(f"   DRAM→GPU: {delta['dram_to_gpu_count']} ops, {delta['dram_to_gpu_us']/1000:.1f}ms")
        if run == 0 and all_outs:
            print(f"\n📝 Sample outputs (first run):")
            for i, out in enumerate(all_outs[:min(3, len(all_outs))]):
                print(f"   [{i+1}] {out[:100]}{'...' if len(out) > 100 else ''}")
            if len(all_outs) > 3:
                print(f"   ... and {len(all_outs) - 3} more")

    print(f"\n🏁 All runs completed!")
    print(f"Total inference time: {total_inference_time:.2f}s")
    print(f"Average time per run: {total_inference_time/args.runs:.2f}s")
    print_transfer_report()
    if args.csv:
        save_csv_results(args.csv, args, total_inference_time)
    print("\n🎯 Profiling completed successfully!")

def save_csv_results(csv_path: str, args, total_inference_time: float) -> None:
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            import csv
            w = csv.writer(f)
            w.writerow(['Metric', 'Value', 'Unit'])
            w.writerow(['Model Path', args.model_path, ''])
            w.writerow(['Device', args.device, ''])
            w.writerow(['Batch Size', args.batch_size, ''])
            w.writerow(['Max Gen Length', args.max_gen_len, ''])
            w.writerow(['TopK Blocks', args.topk_blk, ''])
            w.writerow(['Runs', args.runs, ''])
            w.writerow(['Total Inference Time', f'{total_inference_time:.3f}', 'seconds'])
            w.writerow([])
            s = KV_TRANSFER_STATS
            w.writerow(['Transfer Statistics', '', ''])
            w.writerow(['SSD to DRAM Operations', s['ssd_to_dram_count'], 'count'])
            w.writerow(['SSD to DRAM Time', f"{s['ssd_to_dram_us']/1000:.3f}", 'ms'])
            w.writerow(['SSD to DRAM Bytes', s['ssd_to_dram_bytes'], 'bytes'])
            w.writerow(['DRAM to SSD Operations', s['dram_to_ssd_count'], 'count'])
            w.writerow(['DRAM to SSD Time', f"{s['dram_to_ssd_us']/1000:.3f}", 'ms'])
            w.writerow(['DRAM to SSD Bytes', s['dram_to_ssd_bytes'], 'bytes'])
            w.writerow(['DRAM to GPU Operations', s['dram_to_gpu_count'], 'count'])
            w.writerow(['DRAM to GPU Time', f"{s['dram_to_gpu_us']/1000:.3f}", 'ms'])
            w.writerow(['DRAM to GPU Bytes', s['dram_to_gpu_bytes'], 'bytes'])
            w.writerow(['GPU to DRAM Operations', s['gpu_to_dram_count'], 'count'])
            w.writerow(['GPU to DRAM Time', f"{s['gpu_to_dram_us']/1000:.3f}', 'ms"])
            w.writerow(['GPU to DRAM Bytes', s['gpu_to_dram_bytes'], 'bytes'])
            w.writerow(['Cache Hits', s['kv_cache_hit_count'], 'count'])
            w.writerow(['Cache Misses', s['kv_cache_miss_count'], 'count'])
            w.writerow(['Peak Memory Usage', s['peak_memory_usage'], 'bytes'])
            w.writerow(['Current Memory Allocated', s['total_memory_allocated'], 'bytes'])
            w.writerow(['Memory Fragmentation', f"{s['memory_fragmentation']*100:.2f}", 'percent'])
            if BATCH_STATS['total_batches'] > 0:
                w.writerow([]); w.writerow(['Batch Statistics', '', ''])
                w.writerow(['Total Batches', BATCH_STATS['total_batches'], 'count'])
                w.writerow(['Total Prompts', BATCH_STATS['total_prompts'], 'count'])
                w.writerow(['Max Batch Size', BATCH_STATS['max_batch_size'], 'count'])
                w.writerow(['Min Batch Size', BATCH_STATS['min_batch_size'], 'count'])
                avg_bs = BATCH_STATS['total_prompts'] / BATCH_STATS['total_batches']
                w.writerow(['Avg Batch Size', f'{avg_bs:.2f}', 'count'])
        print(f"📊 Results saved to: {csv_path}")
    except Exception as e:
        print(f"⚠️  Error saving CSV: {e}")

if __name__ == "__main__":
    main()
