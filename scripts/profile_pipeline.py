"""KV Cache SSD 传输时间专用分析器
专门计算和统计：
1. SSD → DRAM 的加载时间
2. DRAM → GPU 的传输时间  
3. GPU → DRAM 的保存时间
4. DRAM → SSD 的卸载时间
"""
import math
import argparse, time, pathlib
from contextlib import contextmanager
import torch, csv, os
from llama3.generator import LLaMA
from llama3.layers import PerformanceTracker    
from llama3.kv_offload import KVOffloader, BLOCK 
from torch.cuda import Event, current_stream
import threading

# ---------- 全局统计数据 ----------
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
}

_PATCHES_APPLIED = False

# ---------- 计时工具 ----------
@contextmanager
def precise_timer(stat_key: str, byte_key: str = None, count_key: str = None, bytes_transferred: int = 0):
    """精确计时器，同时统计时间、字节数和操作次数"""
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    
    elapsed_us = int((end_time - start_time) * 1e6)
    KV_TRANSFER_STATS[stat_key] += elapsed_us
    
    if byte_key and bytes_transferred > 0:
        KV_TRANSFER_STATS[byte_key] += bytes_transferred
    
    if count_key:
        KV_TRANSFER_STATS[count_key] += 1

@contextmanager
def cuda_precise_timer(stat_key: str, byte_key: str = None, count_key: str = None, bytes_transferred: int = 0):
    """CUDA 精确计时器"""
    if not torch.cuda.is_available():
        with precise_timer(stat_key, byte_key, count_key, bytes_transferred):
            yield
        return
    
    start_event = Event(enable_timing=True)
    end_event = Event(enable_timing=True)
    
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

def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if abs(num) < 1024.0:
            return f"{num:6.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} E{suffix}"

def bandwidth_fmt(bytes_val, time_us):
    """计算并格式化带宽"""
    if time_us <= 0:
        return "N/A"
    
    time_s = time_us / 1e6
    bandwidth_bps = bytes_val / time_s
    return sizeof_fmt(bandwidth_bps, 'B/s')

# ---------- KV Offloader 详细 Patch ----------
# ---------- KV Offloader 详细 Patch（兼容 OptimizedKVOffloader & 旧版） ----------
def patch_kv_offloader_detailed():
    """
    打补丁以统计 GPU↔DRAM 及 SSD↔DRAM 传输时间。
    仅修改函数封装，不改任何输出部分。
    """
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    import llama3.kv_offload as kvmod          # 统一入口
    KVCls = getattr(kvmod, "OptimizedKVOffloader",
                    getattr(kvmod, "KVOffloader", None))
    if KVCls is None:
        raise RuntimeError("⚠️  未找到 KV offloader 类")

    # -------- 保存原始方法 --------
    orig_push  = KVCls.push
    orig_fetch = KVCls.fetch
    orig_spill = getattr(KVCls, "_spill_to_ssd",  None)
    # 新版同步加载叫 _load_from_ssd_sync，旧版叫 _load_from_ssd
    orig_load  = getattr(KVCls, "_load_from_ssd_sync",
                         getattr(KVCls, "_load_from_ssd", None))

    # 统一块大小，用于估算字节
    def _blk_bytes(self):
        return getattr(self, "block_nbytes",
               getattr(self, "blk_bytes", 0))

    # ---------- GPU → DRAM ----------
    def wrapped_push(self, layer, blk, k, v):
        bytes_tx = k.nbytes + v.nbytes
        with cuda_precise_timer('gpu_to_dram_us',
                                'gpu_to_dram_bytes',
                                'gpu_to_dram_count',
                                bytes_tx):
            return orig_push(self, layer, blk, k, v)

    # ---------- DRAM (+SSD) → GPU ----------
    # ---------- DRAM (+SSD) → GPU ----------
    def wrapped_fetch(self, layer, blocks):
        blk_list = (blocks.tolist() if torch.is_tensor(blocks) else list(blocks))
        need_ssd = [b for b in blk_list if getattr(self, "on_ssd")[layer][b]]

        # ---------- 如需从 SSD 读取，强制同步 ----------
        if need_ssd:
            bytes_from_ssd = len(need_ssd) * _blk_bytes(self)
            with precise_timer('ssd_to_dram_us',
                               'ssd_to_dram_bytes',
                               'ssd_to_dram_count',
                               bytes_from_ssd):
                for b in need_ssd:           # 逐块同步读取，保证可用
                    if hasattr(self, "_load_from_ssd_sync"):
                        self._load_from_ssd_sync(layer, b)
                    else:                    # 兼容旧名
                        self._load_from_ssd(layer, b)
            KV_TRANSFER_STATS['kv_cache_miss_count'] += 1
        else:
            KV_TRANSFER_STATS['kv_cache_hit_count']  += 1

        # ---------- 现在块已在 DRAM，安全执行原 fetch ----------
        k, v = orig_fetch(self, layer, blocks)

        # ---------- DRAM → GPU 计时 ----------
        if k is not None and v is not None:
            with cuda_precise_timer('dram_to_gpu_us',
                                    'dram_to_gpu_bytes',
                                    'dram_to_gpu_count',
                                    k.nbytes + v.nbytes):
                pass
        return k, v


    # ---------- DRAM → SSD ----------
    def wrapped_spill(self, layer, blk, *args, **kwargs):
        with precise_timer('dram_to_ssd_us',
                           'dram_to_ssd_bytes',
                           'dram_to_ssd_count',
                           _blk_bytes(self)):
            return orig_spill(self, layer, blk, *args, **kwargs)

    # ---------- SSD → DRAM（显式同步加载 API） ----------
    def wrapped_load(self, layer, blk, *args, **kwargs):
        with precise_timer('ssd_to_dram_us',
                           'ssd_to_dram_bytes',
                           'ssd_to_dram_count',
                           _blk_bytes(self)):
            return orig_load(self, layer, blk, *args, **kwargs)

    # -------- 应用补丁（仅当对应方法存在）--------
    KVCls.push   = wrapped_push
    KVCls.fetch  = wrapped_fetch
    if orig_spill:
        KVCls._spill_to_ssd      = wrapped_spill
    if orig_load:
        if hasattr(KVCls, "_load_from_ssd_sync"):
            KVCls._load_from_ssd_sync = wrapped_load
        else:
            KVCls._load_from_ssd      = wrapped_load

    _PATCHES_APPLIED = True
    print("✅ KV offloader detailed patches applied (auto-compatible)")

def _make_perf_tracker_thread_safe():
    """
    把 llama3.layers.PERF_TRACKER.lock 换成 RLock，
    并让 add_layer_stat 在拿不到锁时直接略过这一次累加，
    以免 GPU 同步期间造成全局阻塞。
    """
    import llama3.layers as layermod
    import threading

    tracker = layermod.PERF_TRACKER
    tracker.lock = threading.RLock()          # 1. 换成可重入锁

    orig_add = tracker.add_layer_stat

    def safe_add(self, layer_id, stat_name, value):
        # 2. 尝试 0.5ms 拿锁；拿不到就放弃本次统计
        locked = self.lock.acquire(timeout=5e-4)
        if not locked:
            return
        try:
            return orig_add(layer_id, stat_name, value)
        finally:
            self.lock.release()

    # 3. 替换方法
    layermod.PerformanceTracker.add_layer_stat = safe_add

_make_perf_tracker_thread_safe()

# ---------- 统计报告 ----------
def print_transfer_report():
    """打印详细的传输统计报告"""
    stats = KV_TRANSFER_STATS
    
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
    
    print("="*80)

# ---------- 主函数 ----------
def main():
    ap = argparse.ArgumentParser(description="KV Cache SSD Transfer Profiler")
    ap.add_argument("--model-path", required=True, help="Path to LLaMA model")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. This is a longer prompt to generate more KV cache activity.")
    ap.add_argument("--max-gen-len", type=int, default=128, help="Number of tokens to generate")
    ap.add_argument("--topk-blk", type=int, default=4, help="Number of KV blocks to keep in GPU")
    ap.add_argument("--csv", help="Save results to CSV file")
    ap.add_argument("--runs", type=int, default=1, help="Number of inference runs")
    args = ap.parse_args()

    ckpt = pathlib.Path(args.model_path)
    
    print(f"🚀 KV Cache SSD Transfer Profiler")
    print(f"📁 Model: {ckpt.name}")
    print(f"🖥️  Device: {args.device}")
    print(f"📝 Prompt length: {len(args.prompt)} chars")
    print(f"🔢 Generation length: {args.max_gen_len}")
    print(f"💾 KV blocks in GPU: {args.topk_blk}")
    print(f"🔄 Number of runs: {args.runs}")
    
    # 应用 patches
    patch_kv_offloader_detailed()
    
    # 加载模型
    print("\n📦 Loading model...")
    llama = LLaMA.build(ckpt, load_model=True, device="cpu")
    
    # 配置 KV cache
    for layer in llama.model.layers:
        if hasattr(layer, 'attention') and hasattr(layer.attention, 'topk_blk'):
            layer.attention.topk_blk = args.topk_blk
    
    # 转移到 GPU
    if args.device.startswith("cuda"):
        print("🔄 Transferring to GPU...")
        llama.model.to(args.device)
        torch.cuda.synchronize()
    
    llama.args.device = args.device
    
    # 配置 offloader streams
    for blk in llama.model.layers:
        if hasattr(blk, 'attention') and hasattr(blk.attention, 'offloader'):
            off = blk.attention.offloader
            off.device = args.device
            if torch.cuda.is_available():
                off.copy_stream = torch.cuda.Stream(device=args.device)
    
    print("✅ Model ready for inference")
    
    # 执行多次推理
    total_inference_time = 0
    
    for run in range(args.runs):
        print(f"\n🧠 Running inference {run+1}/{args.runs}...")
        
        # 重置当前运行的统计
        run_stats = {k: v for k, v in KV_TRANSFER_STATS.items()}
        
        start_time = time.perf_counter()
        result = llama.text_completion([args.prompt], max_gen_len=args.max_gen_len)
        end_time = time.perf_counter()
        
        run_time = (end_time - start_time) * 1000
        total_inference_time += run_time
        
        # 计算这次运行的传输统计
        run_transfer_time = 0
        for key in ['ssd_to_dram_us', 'dram_to_ssd_us', 'dram_to_gpu_us', 'gpu_to_dram_us']:
            run_transfer_time += KV_TRANSFER_STATS[key] - run_stats[key]
        
        print(f"   Inference time: {run_time:.1f} ms")
        print(f"   Transfer time: {run_transfer_time/1000:.1f} ms ({run_transfer_time/run_time/10:.1f}%)")
        
        if args.runs == 1:  # 只有一次运行时显示生成的文本
            print(f"   Generated: {result[0][len(args.prompt):len(args.prompt)+50]}...")
    
    # 打印详细报告
    print_transfer_report()
    
    # 最终统计
    avg_inference_time = total_inference_time / args.runs
    tokens_per_sec = args.max_gen_len / (avg_inference_time / 1000)
    
    print(f"\n📈 Performance Summary:")
    print(f"   Average inference time: {avg_inference_time:.1f} ms")
    print(f"   Tokens per second: {tokens_per_sec:.1f}")
    
    # 保存到 CSV
    if args.csv:
        headers = ['model', 'prompt_len', 'gen_len', 'topk_blk', 'runs', 'avg_inference_ms'] + list(KV_TRANSFER_STATS.keys())
        row = [ckpt.name, len(args.prompt), args.max_gen_len, args.topk_blk, args.runs, avg_inference_time] + list(KV_TRANSFER_STATS.values())
        
        file_exists = os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)
        print(f"\n📊 Results saved to {args.csv}")

if __name__ == "__main__":
    main()