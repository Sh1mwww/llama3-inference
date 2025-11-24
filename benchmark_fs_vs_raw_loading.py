#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比普通文件系统(fs)加载和 raw 方式加载 MHA/FFN 权重的时间

测试场景:
1. 普通文件系统加载 (Filesystem + buffered IO, 单文件顺序读)
2. Raw 方式加载 (RawBlockKVBackend + DirectIO/O_DIRECT)

对比指标:
- MHA 权重加载时间
- FFN 权重加载时间
- 加载带宽 (GB/s)

说明:
- 为了专注于 I/O 与用户态拷贝，本脚本使用随机数据，只匹配参数规模，不依赖真实模型权重。
- Filesystem 场景用顺序 read + frombuffer + clone 近似 torch.load/safetensors 的 I/O 部分。
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Any

import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from llama3.config import ModelArgs, KVCacheArgs
from llama3.SSDBacked import RawBlockKVBackend


# ======================== 全局配置 ========================

# 用于 FS benchmark 的目录：请挂在和 raw device 同一块 NVMe 上
FS_BENCH_DIR = "/data1/fs_bench"

# 模型配置与 checkpoint 路径
PARAMS_JSON = "/data1/.llama/checkpoints/Llama3.1-70B/params.json"
CHECKPOINT_DIR = "/data1/.llama/checkpoints/Llama3.1-70B"

# 是否在每个大场景（FS/Raw）开始前尝试 drop_caches
DROP_CACHES_BEFORE_EACH_TEST = True

# 每个场景的 warmup / 正式迭代次数
WARMUP_ITERS = 2
BENCH_ITERS = 10


# ======================== 工具函数 ========================

def drop_caches() -> None:
    """尝试清理 Linux page cache，用于模拟冷启动。需要免密 sudo。"""
    print("\n[INFO] 尝试 drop_caches (sync + echo 3 > /proc/sys/vm/drop_caches)...")
    ret1 = os.system("sync")
    ret2 = os.system("echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1")
    if ret1 != 0 or ret2 != 0:
        print("[WARN] drop_caches 可能失败（需要无密码 sudo），结果可能偏向 warm 场景。")
    else:
        print("[INFO] 已请求内核 drop_caches。")
    time.sleep(0.5)


def ensure_fs_bench_dir() -> str:
    """确保 FS_BENCH_DIR 存在，失败则回退到系统 tmp 目录。"""
    d = FS_BENCH_DIR
    try:
        os.makedirs(d, exist_ok=True)
        print(f"[INFO] FS benchmark 目录: {d}")
        return d
    except Exception as e:
        tmp = tempfile.gettempdir()
        print(f"[WARN] 无法创建 FS_BENCH_DIR={d}: {e}")
        print(f"       回退到系统临时目录: {tmp}")
        print("       注意：若 /tmp 是 tmpfs，则不会打到真实 SSD，上报论文数据时请改成挂在 NVMe 上的目录。")
        return tmp


# ======================== 权重大小计算 ========================

def get_mha_weight_size(args: ModelArgs) -> Tuple[int, int]:
    """计算 MHA 部分的权重大小 (bfloat16)"""
    dim = args.dim
    n_heads = args.n_heads
    n_kv_heads = args.n_kv_heads
    head_dim = dim // n_heads
    kv_dim = n_kv_heads * head_dim

    # wq, wk, wv, wo
    mha_params = (
        dim * dim +      # wq
        kv_dim * dim +   # wk
        kv_dim * dim +   # wv
        dim * dim        # wo
    )

    # bfloat16 = 2 bytes per parameter
    mha_bytes = mha_params * 2
    return mha_bytes, mha_params


def _calculate_hidden_dim(args: ModelArgs) -> int:
    """标准的 hidden_dim 计算"""
    dim = args.dim
    if args.ffn_dim_multiplier is not None:
        hidden_dim = int(args.ffn_dim_multiplier * dim)
    else:
        hidden_dim = 4 * dim
    hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
    return hidden_dim


def get_ffn_weight_size(args: ModelArgs) -> Tuple[int, int, int]:
    """
    计算 FFN 部分的权重大小，优先考虑 Tensor Parallel 分片。

    注意：对于 70B 模型：
    - 模型使用 8 路 Tensor Parallel
    - 每 shard 的 hidden_dim_per_shard ~3584，完整 hidden_dim ~28672
    """
    dim = args.dim
    hidden_dim: int

    checkpoint_path = CHECKPOINT_DIR
    if os.path.exists(checkpoint_path):
        import glob
        ckpt_files = sorted(glob.glob(f"{checkpoint_path}/consolidated.*.pth"))
        n_shards = len(ckpt_files)

        if n_shards > 1:
            # 有多个分片，说明使用了 Tensor Parallel
            try:
                ckpt = torch.load(ckpt_files[0], map_location='cpu')
                # 查找 w1 的形状
                hidden_dim_per_shard = None
                for key in ckpt.keys():
                    if 'layers.0.feed_forward.w1' in key:
                        w1_shape = ckpt[key].shape
                        hidden_dim_per_shard = w1_shape[0]
                        hidden_dim = hidden_dim_per_shard * n_shards
                        print(f"[INFO] 从 checkpoint 推断 hidden_dim: {hidden_dim} "
                              f"(分片数: {n_shards}, 每分片: {hidden_dim_per_shard})")
                        break
                else:
                    # 如果找不到，使用标准计算
                    print("[WARN] 未在 checkpoint 中找到 w1，使用标准公式计算 hidden_dim。")
                    hidden_dim = _calculate_hidden_dim(args)
                del ckpt
            except Exception as e:
                print(f"[WARN] 无法从 checkpoint 推断 hidden_dim: {e}")
                hidden_dim = _calculate_hidden_dim(args)
        else:
            hidden_dim = _calculate_hidden_dim(args)
    else:
        hidden_dim = _calculate_hidden_dim(args)

    # w1, w2, w3
    ffn_params = (
        hidden_dim * dim +  # w1
        dim * hidden_dim +  # w2
        hidden_dim * dim    # w3
    )

    ffn_bytes = ffn_params * 2  # bfloat16 = 2 bytes
    return ffn_bytes, ffn_params, hidden_dim


# ======================== Filesystem benchmark ========================

def benchmark_fs_load(weight_size_bytes: int,
                      name: str,
                      fs_dir: str,
                      warmup: int = WARMUP_ITERS,
                      iterations: int = BENCH_ITERS) -> Tuple[float, float, float, float]:
    """
    测量普通文件系统加载时间 (buffered IO)。

    Args:
        weight_size_bytes: 权重大小（字节）
        name: 测试名称（MHA或FFN）
        fs_dir: 测试文件所在目录 (挂在 NVMe 上)
        warmup: 预热次数
        iterations: 测试迭代次数

    返回:
        (avg_ms, min_ms, max_ms, avg_bandwidth_GBps)
    """
    print(f"\n{'='*80}")
    print(f"测试: {name} 普通文件系统加载 (buffered)")
    print(f"{'='*80}")

    num_elements = weight_size_bytes // 2  # bfloat16 = 2 bytes

    print(f"[FS] 生成 {weight_size_bytes / (1024**3):.3f} GB 测试数据...")
    test_data = torch.randn(num_elements, dtype=torch.float16)

    os.makedirs(fs_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin', dir=fs_dir) as tmp_file:
        tmp_path = tmp_file.name

    print(f"[FS] 保存测试数据到临时文件: {tmp_path}")
    with open(tmp_path, 'wb') as f:
        f.write(test_data.numpy().tobytes())

    # 整个 FS 场景开始前做一次 drop_caches（冷启动）
    if DROP_CACHES_BEFORE_EACH_TEST:
        drop_caches()

    time.sleep(0.2)

    # Warmup
    print(f"[FS] Warmup ({warmup} iterations)...")
    for i in range(warmup):
        with open(tmp_path, 'rb') as f:
            buf = f.read()
            _ = torch.frombuffer(memoryview(buf), dtype=torch.float16).clone()
        print(f"  Warmup iter {i+1} done.")

    # Benchmark
    print(f"[FS] Benchmarking ({iterations} iterations)...")
    times = []

    for i in range(iterations):
        start_time = time.perf_counter()
        with open(tmp_path, 'rb') as f:
            buf = f.read()
            _ = torch.frombuffer(memoryview(buf), dtype=torch.float16).clone()
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000
        times.append(elapsed_ms)
        print(f"  Iteration {i+1}: {elapsed_ms:.2f} ms")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_ms = (sum((t - avg_time)**2 for t in times) / len(times))**0.5
    bandwidth_gbps = weight_size_bytes / (avg_time / 1000) / (1024**3)

    print(f"\n[FS] 结果:")
    print(f"  平均时间: {avg_time:.2f} ms")
    print(f"  最小时间: {min_time:.2f} ms")
    print(f"  最大时间: {max_time:.2f} ms")
    print(f"  标准差: {std_ms:.2f} ms")
    print(f"  平均带宽: {bandwidth_gbps:.2f} GB/s")

    os.unlink(tmp_path)
    del test_data, buf

    return avg_time, min_time, max_time, bandwidth_gbps


# ======================== Raw benchmark ========================

def benchmark_raw_load(weight_size_bytes: int,
                       name: str,
                       device_path: str,
                       warmup: int = WARMUP_ITERS,
                       iterations: int = BENCH_ITERS):
    """
    测量 Raw 方式加载时间 (使用 RawBlockKVBackend)

    Args:
        weight_size_bytes: 权重大小（字节）
        name: 测试名称（MHA或FFN）
        device_path: SSD 原始设备或分区路径（务必是专门预留的 raw 区域）
        warmup: 预热次数
        iterations: 测试迭代次数
    """
    print(f"\n{'='*80}")
    print(f"测试: {name} Raw方式加载 (DirectIO + RawBlockKVBackend)")
    print(f"{'='*80}")

    print(f"[RAW] 使用设备: {device_path}")
    if not os.path.exists(device_path):
        raise RuntimeError(f"指定的 raw 设备不存在: {device_path}")

    # 初始化 RawBlockKVBackend
    n_layers = 1
    blk_per_layer = 1

    # 对齐到4KB（DirectIO要求）
    aligned_size = ((weight_size_bytes + 4095) // 4096) * 4096

    print(f"[RAW] 初始化 RawBlockKVBackend...")
    print(f"  块大小: {aligned_size / (1024**2):.2f} MB (4KB 对齐)")

    backend = RawBlockKVBackend(
        dev_path=device_path,
        n_layers=n_layers,
        blk_bytes=aligned_size,
        blk_per_layer=blk_per_layer,
        max_concurrent_io=getattr(KVCacheArgs, "max_concurrent_io", 4),
    )

    # 创建 pinned memory buffer 用于读取
    print(f"[RAW] 创建 {aligned_size / (1024**3):.3f} GB 的 pinned memory buffer...")
    buffer = torch.empty(aligned_size, dtype=torch.uint8, pin_memory=True)

    # 预写入测试数据
    print(f"[RAW] 预写入测试数据...")
    test_data = torch.randn(aligned_size // 2, dtype=torch.bfloat16, pin_memory=True)
    backend.write(0, 0, test_data)

    time.sleep(0.5)

    # 确定使用哪种读取方法
    use_zero_copy = hasattr(backend, 'read_into_pinned_aligned')
    read_method = "read_into_pinned_aligned" if use_zero_copy else "read"
    print(f"[RAW] 使用读取方法: {read_method}")

    if DROP_CACHES_BEFORE_EACH_TEST:
        # 严格来说 O_DIRECT 不走 page cache，这里只是为了流程对齐
        drop_caches()

    # Warmup
    print(f"[RAW] Warmup ({warmup} iterations)...")
    for i in range(warmup):
        if use_zero_copy:
            backend.read_into_pinned_aligned(0, 0, buffer)
        else:
            backend.read(0, 0, buffer)
        print(f"  Warmup iter {i+1} done.")

    # Benchmark
    print(f"[RAW] Benchmarking ({iterations} iterations)...")
    times = []
    for i in range(iterations):
        start_time = time.perf_counter()

        if use_zero_copy:
            backend.read_into_pinned_aligned(0, 0, buffer)
        else:
            backend.read(0, 0, buffer)

        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000
        times.append(elapsed_ms)
        print(f"  Iteration {i+1}: {elapsed_ms:.2f} ms")

    # 统计
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_ms = (sum((t - avg_time)**2 for t in times) / len(times))**0.5
    bandwidth_gbps = weight_size_bytes / (avg_time / 1000) / (1024**3)

    print(f"\n[RAW] 结果:")
    print(f"  平均时间: {avg_time:.2f} ms")
    print(f"  最小时间: {min_time:.2f} ms")
    print(f"  最大时间: {max_time:.2f} ms")
    print(f"  标准差: {std_ms:.2f} ms")
    print(f"  平均带宽: {bandwidth_gbps:.2f} GB/s")

    # 清理
    del buffer, test_data, backend
    torch.cuda.empty_cache()

    return avg_time, min_time, max_time, bandwidth_gbps


# ======================== main ========================

def main():
    print("="*80)
    print("文件系统 vs Raw 方式加载对比测试 (Llama3.1-70B 单层 MHA/FFN)")
    print("="*80)

    # 检查CUDA
    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        return 1

    print(f"\nGPU信息:")
    print(f"  设备: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  显存: {total_mem:.2f} GB")

    # 加载70B模型配置
    params_path = PARAMS_JSON
    if not os.path.exists(params_path):
        print(f"\n错误: 找不到模型配置文件 {params_path}")
        return 1

    print(f"\n加载模型配置: {params_path}")
    args = ModelArgs.from_json(
        params_path,
        max_seq_len=2048,
        max_batch_size=1,
        device="cuda:0"
    )

    # 计算权重大小
    mha_bytes, mha_params = get_mha_weight_size(args)
    ffn_bytes, ffn_params, ffn_hidden_dim = get_ffn_weight_size(args)

    print(f"\n模型配置:")
    print(f"  dim: {args.dim}")
    print(f"  n_layers: {args.n_layers}")
    print(f"  n_heads: {args.n_heads}")
    print(f"  n_kv_heads: {args.n_kv_heads}")
    print(f"\nMHA权重:")
    print(f"  参数数量: {mha_params:,}")
    print(f"  权重大小: {mha_bytes / (1024**3):.3f} GB")
    print(f"\nFFN权重:")
    print(f"  参数数量: {ffn_params:,}")
    print(f"  权重大小: {ffn_bytes / (1024**3):.3f} GB")
    print(f"  hidden_dim: {ffn_hidden_dim}")

    fs_dir = ensure_fs_bench_dir()

    # Raw 设备从 KVCacheArgs 读取，不再默认 /dev/nvme0n1，以免误伤系统盘
    raw_dev = getattr(KVCacheArgs, "ssd_device_path", None)
    if raw_dev is None:
        print("\n错误: KVCacheArgs.ssd_device_path 未配置，请指定专门用于 RawBlockKVBackend 的 raw 设备或分区。")
        return 1

    # 存储结果
    results: Dict[str, Dict[str, Any]] = {}

    # ========== MHA测试 ==========

    print(f"\n{'='*80}")
    print("第一部分: MHA 权重加载测试")
    print(f"{'='*80}")

    # MHA - Filesystem（buffered IO）
    mha_fs_avg, mha_fs_min, mha_fs_max, mha_fs_bw = benchmark_fs_load(
        mha_bytes, "MHA", fs_dir
    )
    results['mha_fs_buffered'] = {
        'avg': mha_fs_avg,
        'min': mha_fs_min,
        'max': mha_fs_max,
        'bandwidth': mha_fs_bw
    }

    # MHA - Raw 方式
    mha_raw_avg, mha_raw_min, mha_raw_max, mha_raw_bw = benchmark_raw_load(
        mha_bytes, "MHA", device_path=raw_dev
    )
    results['mha_raw'] = {
        'avg': mha_raw_avg,
        'min': mha_raw_min,
        'max': mha_raw_max,
        'bandwidth': mha_raw_bw
    }

    # ========== FFN测试 ==========

    print(f"\n{'='*80}")
    print("第二部分: FFN 权重加载测试")
    print(f"{'='*80}")

    # FFN - Filesystem（buffered IO）
    ffn_fs_avg, ffn_fs_min, ffn_fs_max, ffn_fs_bw = benchmark_fs_load(
        ffn_bytes, "FFN", fs_dir
    )
    results['ffn_fs_buffered'] = {
        'avg': ffn_fs_avg,
        'min': ffn_fs_min,
        'max': ffn_fs_max,
        'bandwidth': ffn_fs_bw
    }

    # FFN - Raw 方式
    ffn_raw_avg, ffn_raw_min, ffn_raw_max, ffn_raw_bw = benchmark_raw_load(
        ffn_bytes, "FFN", device_path=raw_dev
    )
    results['ffn_raw'] = {
        'avg': ffn_raw_avg,
        'min': ffn_raw_min,
        'max': ffn_raw_max,
        'bandwidth': ffn_raw_bw
    }

    # ========== 结果总结 ==========

    print(f"\n{'='*80}")
    print("测试结果总结")
    print(f"{'='*80}")

    print(f"\n权重大小:")
    print(f"  MHA: {mha_bytes / (1024**3):.3f} GB")
    print(f"  FFN: {ffn_bytes / (1024**3):.3f} GB")

    # MHA对比
    print(f"\n{'='*80}")
    print("MHA 权重加载对比")
    print(f"{'='*80}")

    m_fs = results['mha_fs_buffered']
    m_raw = results['mha_raw']

    print(f"\n文件系统 (Buffered IO):")
    print(f"  平均时间: {m_fs['avg']:.2f} ms")
    print(f"  最小时间: {m_fs['min']:.2f} ms")
    print(f"  平均带宽: {m_fs['bandwidth']:.2f} GB/s")

    print(f"\nRaw方式 (DirectIO + RawBlockKVBackend):")
    print(f"  平均时间: {m_raw['avg']:.2f} ms")
    print(f"  最小时间: {m_raw['min']:.2f} ms")
    print(f"  平均带宽: {m_raw['bandwidth']:.2f} GB/s")

    speedup_mha = m_fs['avg'] / m_raw['avg']
    print(f"\n加速比 (MHA, 平均时间): Raw vs FS = {speedup_mha:.2f}x")

    # FFN对比
    print(f"\n{'='*80}")
    print("FFN 权重加载对比")
    print(f"{'='*80}")

    f_fs = results['ffn_fs_buffered']
    f_raw = results['ffn_raw']

    print(f"\n文件系统 (Buffered IO):")
    print(f"  平均时间: {f_fs['avg']:.2f} ms")
    print(f"  最小时间: {f_fs['min']:.2f} ms")
    print(f"  平均带宽: {f_fs['bandwidth']:.2f} GB/s")

    print(f"\nRaw方式 (DirectIO + RawBlockKVBackend):")
    print(f"  平均时间: {f_raw['avg']:.2f} ms")
    print(f"  最小时间: {f_raw['min']:.2f} ms")
    print(f"  平均带宽: {f_raw['bandwidth']:.2f} GB/s")

    speedup_ffn = f_fs['avg'] / f_raw['avg']
    print(f"\n加速比 (FFN, 平均时间): Raw vs FS = {speedup_ffn:.2f}x")

    # 总体对比
    print(f"\n{'='*80}")
    print("总体对比")
    print(f"{'='*80}")

    layer_fs_buffered = m_fs['avg'] + f_fs['avg']
    layer_raw = m_raw['avg'] + f_raw['avg']

    print(f"\n文件系统 (Buffered) - 单层加载时间:")
    print(f"  MHA + FFN = {layer_fs_buffered:.2f} ms")

    print(f"\nRaw方式 - 单层加载时间:")
    print(f"  MHA + FFN = {layer_raw:.2f} ms")

    total_speedup_buffered = layer_fs_buffered / layer_raw
    print(f"\n单层总加速比:")
    print(f"  Raw vs Buffered: {total_speedup_buffered:.2f}x")

    # 推算全模型（n_layers 层）的加载时间
    full_model_raw = layer_raw * args.n_layers
    full_model_fs_buffered = layer_fs_buffered * args.n_layers

    print(f"\n全模型 ({args.n_layers}层) 预估加载时间:")
    print(f"  Raw方式: {full_model_raw:.2f} ms = {full_model_raw/1000:.2f} s")
    print(f"  文件系统(Buffered): {full_model_fs_buffered:.2f} ms = {full_model_fs_buffered/1000:.2f} s")
    print(f"  节省时间: {(full_model_fs_buffered - full_model_raw)/1000:.2f} s")

    # 方便直接丢论文表格的 summary
    print("\n--- Summary (for paper table, avg over runs) ---")
    print("case, part, avg_ms, min_ms, GBps")
    print(f"FS,  MHA, {m_fs['avg']:.2f}, {m_fs['min']:.2f}, {m_fs['bandwidth']:.2f}")
    print(f"Raw, MHA, {m_raw['avg']:.2f}, {m_raw['min']:.2f}, {m_raw['bandwidth']:.2f}")
    print(f"FS,  FFN, {f_fs['avg']:.2f}, {f_fs['min']:.2f}, {f_fs['bandwidth']:.2f}")
    print(f"Raw, FFN, {f_raw['avg']:.2f}, {f_raw['min']:.2f}, {f_raw['bandwidth']:.2f}")

    print(f"\n{'='*80}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
