#!/usr/bin/env python3
"""
测量70B模型单层操作的详细时间 - 分离MHA和FFN:
1. MHA计算时间 (Prefill & Decode)
2. FFN计算时间 (Prefill & Decode)
3. MHA权重传输时间 (Pinned->GPU, SSD->Pinned)
4. FFN权重传输时间 (Pinned->GPU, SSD->Pinned)

注意:
- Decode测试是在~256 tokens context下进行的单token推理
- Pinned->GPU测试给出乐观的带宽上界参考
- SSD->Pinned测试使用真实的DirectIO/WSM接口
"""
import torch
import torch.nn as nn
import time
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from llama3.config import ModelArgs, KVCacheArgs
from llama3.layers import SelfAttention, FeedForward, RMSNorm, precompute_theta_pos_frequencies
from llama3.SSDBacked import RawBlockKVBackend


def get_mha_weight_size(args: ModelArgs) -> int:
    """计算MHA部分的权重大小"""
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


def get_ffn_weight_size(args: ModelArgs) -> int:
    """计算FFN部分的权重大小"""
    dim = args.dim

    # 计算FFN hidden dim
    if args.ffn_dim_multiplier is not None:
        hidden_dim = int(args.ffn_dim_multiplier * dim)
    else:
        hidden_dim = 4 * dim

    hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

    # w1, w2, w3
    ffn_params = (
        hidden_dim * dim +  # w1
        dim * hidden_dim +  # w2
        hidden_dim * dim    # w3
    )

    # bfloat16 = 2 bytes per parameter
    ffn_bytes = ffn_params * 2

    return ffn_bytes, ffn_params, hidden_dim


def benchmark_mha_computation(args: ModelArgs, seq_len: int = 2048, is_prefill: bool = True,
                               warmup: int = 3, iterations: int = 10):
    """测量MHA计算时间"""
    phase_name = "Prefill" if is_prefill else "Decode"
    print(f"\n{'='*80}")
    print(f"测试: MHA {phase_name}计算时间 (seq_len={seq_len if is_prefill else 1})")
    print(f"{'='*80}")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # 创建MHA模块
    mha = SelfAttention(args).to(device=device, dtype=dtype)

    # 预计算位置编码
    freqs = precompute_theta_pos_frequencies(
        args.dim // args.n_heads,
        args.max_seq_len * 2,
        device=device,
        theta=args.rope_theta,
    )

    if is_prefill:
        # Prefill: 处理整个序列
        batch_size = 1
        x = torch.randn(batch_size, seq_len, args.dim, device=device, dtype=dtype)
        start_pos = 0
        test_iterations = iterations
    else:
        # Decode: 先做prefill初始化KV cache，然后测试单token
        batch_size = 1
        prefill_len = 256
        print(f"初始化KV cache (prefill {prefill_len} tokens)...")
        prefill_x = torch.randn(batch_size, prefill_len, args.dim, device=device, dtype=dtype)
        with torch.no_grad():
            _ = mha(prefill_x, 0, freqs)
        del prefill_x
        torch.cuda.synchronize()

        # 准备单token输入
        x = torch.randn(batch_size, 1, args.dim, device=device, dtype=dtype)
        start_pos = prefill_len
        test_iterations = min(iterations * 10, 100)  # Decode测试更多次

    # Warmup
    print(f"Warmup ({warmup} iterations)...")
    for i in range(warmup):
        with torch.no_grad():
            _ = mha(x, start_pos + (i if not is_prefill else 0), freqs)
    torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({test_iterations} iterations)...")
    times = []
    for i in range(test_iterations):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            output = mha(x, start_pos + (warmup + i if not is_prefill else 0), freqs)

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000
        times.append(elapsed_ms)
        if i < 10:  # 只打印前10次
            print(f"  Iteration {i+1}: {elapsed_ms:.2f} ms")

    if test_iterations > 10:
        print(f"  ... ({test_iterations - 10} more iterations)")

    # 统计
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\n结果:")
    print(f"  平均时间: {avg_time:.2f} ms")
    print(f"  最小时间: {min_time:.2f} ms")
    print(f"  最大时间: {max_time:.2f} ms")
    print(f"  标准差: {(sum((t - avg_time)**2 for t in times) / len(times))**0.5:.2f} ms")

    # 清理
    del mha, x, output, freqs
    torch.cuda.empty_cache()

    return avg_time, min_time, max_time


def benchmark_ffn_computation(args: ModelArgs, seq_len: int = 2048, is_prefill: bool = True,
                               warmup: int = 3, iterations: int = 10):
    """测量FFN计算时间"""
    phase_name = "Prefill" if is_prefill else "Decode"
    print(f"\n{'='*80}")
    print(f"测试: FFN {phase_name}计算时间 (seq_len={seq_len if is_prefill else 1})")
    print(f"{'='*80}")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # 创建FFN模块
    ffn = FeedForward(args).to(device=device, dtype=dtype)

    if is_prefill:
        # Prefill: 处理整个序列
        batch_size = 1
        x = torch.randn(batch_size, seq_len, args.dim, device=device, dtype=dtype)
        test_iterations = iterations
    else:
        # Decode: 单token
        batch_size = 1
        x = torch.randn(batch_size, 1, args.dim, device=device, dtype=dtype)
        test_iterations = min(iterations * 10, 100)

    # Warmup
    print(f"Warmup ({warmup} iterations)...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = ffn(x)
    torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({test_iterations} iterations)...")
    times = []
    for i in range(test_iterations):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            output = ffn(x)

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000
        times.append(elapsed_ms)
        if i < 10:
            print(f"  Iteration {i+1}: {elapsed_ms:.2f} ms")

    if test_iterations > 10:
        print(f"  ... ({test_iterations - 10} more iterations)")

    # 统计
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\n结果:")
    print(f"  平均时间: {avg_time:.2f} ms")
    print(f"  最小时间: {min_time:.2f} ms")
    print(f"  最大时间: {max_time:.2f} ms")
    print(f"  标准差: {(sum((t - avg_time)**2 for t in times) / len(times))**0.5:.2f} ms")

    # 清理
    del ffn, x, output
    torch.cuda.empty_cache()

    return avg_time, min_time, max_time


def benchmark_weight_transfer(weight_size_bytes: int, name: str, warmup: int = 3, iterations: int = 20):
    """测量权重传输时间 (Pinned Memory -> GPU)"""
    print(f"\n{'='*80}")
    print(f"测试: {name} 权重传输时间 (Pinned -> GPU)")
    print(f"{'='*80}")

    device = torch.device("cuda:0")

    # 计算需要的元素数量（bfloat16 = 2 bytes）
    num_elements = weight_size_bytes // 2

    # 创建pinned memory tensor
    print(f"创建 {weight_size_bytes / (1024**3):.3f} GB 的 pinned memory tensor...")
    pinned_tensor = torch.randn(num_elements, dtype=torch.bfloat16, pin_memory=True)
    gpu_tensor = torch.empty(num_elements, dtype=torch.bfloat16, device=device)

    # Warmup
    print(f"Warmup ({warmup} iterations)...")
    for _ in range(warmup):
        gpu_tensor.copy_(pinned_tensor, non_blocking=True)
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({iterations} iterations)...")
    times = []
    for i in range(iterations):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        gpu_tensor.copy_(pinned_tensor, non_blocking=True)

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000
        times.append(elapsed_ms)
        print(f"  Iteration {i+1}: {elapsed_ms:.2f} ms")

    # 统计
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    bandwidth_gbps = weight_size_bytes / (avg_time / 1000) / (1024**3)

    print(f"\n结果:")
    print(f"  平均时间: {avg_time:.2f} ms")
    print(f"  最小时间: {min_time:.2f} ms")
    print(f"  最大时间: {max_time:.2f} ms")
    print(f"  标准差: {(sum((t - avg_time)**2 for t in times) / len(times))**0.5:.2f} ms")
    print(f"  平均带宽: {bandwidth_gbps:.2f} GB/s")

    # 清理
    del pinned_tensor, gpu_tensor
    torch.cuda.empty_cache()

    return avg_time, min_time, max_time, bandwidth_gbps


def benchmark_ssd_read(weight_size_bytes: int, name: str,
                       device_path: str = None,
                       warmup: int = 2, iterations: int = 10):
    """
    测量SSD读取时间 (SSD -> Pinned Memory)
    使用真实的RawBlockKVBackend接口进行测试
    """
    print(f"\n{'='*80}")
    print(f"测试: {name} SSD读取时间 (SSD -> Pinned)")
    print(f"{'='*80}")

    # 使用KVCacheArgs中配置的设备路径
    if device_path is None:
        device_path = getattr(KVCacheArgs, "ssd_device_path", "/dev/nvme0n1")

    print(f"使用设备: {device_path}")

    try:
        # 初始化RawBlockKVBackend
        # 我们只需要测试一个layer，一个block
        n_layers = 1
        blk_per_layer = 1

        # 对齐到4KB（DirectIO要求）
        aligned_size = ((weight_size_bytes + 4095) // 4096) * 4096

        print(f"初始化 RawBlockKVBackend...")
        print(f"  块大小: {aligned_size / (1024**2):.2f} MB (4KB对齐)")

        backend = RawBlockKVBackend(
            dev_path=device_path,
            n_layers=n_layers,
            blk_bytes=aligned_size,
            blk_per_layer=blk_per_layer,
            max_concurrent_io=getattr(KVCacheArgs, "max_concurrent_io", 4),
        )

        # 创建pinned memory buffer用于读取
        print(f"创建 {aligned_size / (1024**3):.3f} GB 的 pinned memory buffer...")
        buffer = torch.empty(aligned_size, dtype=torch.uint8, pin_memory=True)

        # 先写入一些数据（否则读取可能是空的）
        print(f"预写入测试数据...")
        test_data = torch.randn(aligned_size // 2, dtype=torch.bfloat16, pin_memory=True)
        backend.write(0, 0, test_data)

        # Warmup
        print(f"Warmup ({warmup} iterations)...")
        for _ in range(warmup):
            if hasattr(backend, 'read_into_pinned_aligned'):
                backend.read_into_pinned_aligned(0, 0, buffer)
            else:
                backend.read(0, 0, buffer)

        # Benchmark
        print(f"Benchmarking ({iterations} iterations)...")
        times = []
        for i in range(iterations):
            start_time = time.perf_counter()

            if hasattr(backend, 'read_into_pinned_aligned'):
                # 使用零拷贝读取（如果支持）
                backend.read_into_pinned_aligned(0, 0, buffer)
            else:
                # 回退到标准读取
                backend.read(0, 0, buffer)

            end_time = time.perf_counter()

            elapsed_ms = (end_time - start_time) * 1000
            times.append(elapsed_ms)
            print(f"  Iteration {i+1}: {elapsed_ms:.2f} ms")

        # 统计
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        bandwidth_gbps = weight_size_bytes / (avg_time / 1000) / (1024**3)

        print(f"\n结果:")
        print(f"  平均时间: {avg_time:.2f} ms")
        print(f"  最小时间: {min_time:.2f} ms")
        print(f"  最大时间: {max_time:.2f} ms")
        print(f"  标准差: {(sum((t - avg_time)**2 for t in times) / len(times))**0.5:.2f} ms")
        print(f"  平均带宽: {bandwidth_gbps:.2f} GB/s")

        # 清理
        del buffer, test_data, backend
        torch.cuda.empty_cache()

        return avg_time, min_time, max_time, bandwidth_gbps

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def main():
    print("="*80)
    print("70B 模型 MHA/FFN 分离性能测试")
    print("="*80)

    # 禁用KV cache mirror以避免干扰纯计算时间测量
    KVCacheArgs.mirror_on_push = False
    KVCacheArgs.verbose_pool = False
    print("已禁用 KV cache mirroring 和 verbose logging\n")

    # 检查CUDA
    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        return 1

    print(f"\nGPU信息:")
    print(f"  设备: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  显存: {total_mem:.2f} GB")

    # 加载70B模型配置
    params_path = "/data1/.llama/checkpoints/Llama3.1-70B/params.json"
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
    print(f"\n总权重大小: {(mha_bytes + ffn_bytes) / (1024**3):.3f} GB")

    # 存储结果
    results = {}

    # ========== 计算时间测试 ==========

    try:
        print(f"\n{'='*80}")
        print("第一部分: 计算时间测试")
        print(f"{'='*80}")

        # MHA Prefill
        mha_prefill_avg, mha_prefill_min, mha_prefill_max = benchmark_mha_computation(
            args, seq_len=2048, is_prefill=True
        )
        results['mha_prefill'] = {
            'avg': mha_prefill_avg,
            'min': mha_prefill_min,
            'max': mha_prefill_max
        }
    except Exception as e:
        print(f"\nMHA Prefill测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['mha_prefill'] = None

    try:
        # MHA Decode
        mha_decode_avg, mha_decode_min, mha_decode_max = benchmark_mha_computation(
            args, seq_len=1, is_prefill=False
        )
        results['mha_decode'] = {
            'avg': mha_decode_avg,
            'min': mha_decode_min,
            'max': mha_decode_max
        }
    except Exception as e:
        print(f"\nMHA Decode测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['mha_decode'] = None

    try:
        # FFN Prefill
        ffn_prefill_avg, ffn_prefill_min, ffn_prefill_max = benchmark_ffn_computation(
            args, seq_len=2048, is_prefill=True
        )
        results['ffn_prefill'] = {
            'avg': ffn_prefill_avg,
            'min': ffn_prefill_min,
            'max': ffn_prefill_max
        }
    except Exception as e:
        print(f"\nFFN Prefill测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['ffn_prefill'] = None

    try:
        # FFN Decode
        ffn_decode_avg, ffn_decode_min, ffn_decode_max = benchmark_ffn_computation(
            args, seq_len=1, is_prefill=False
        )
        results['ffn_decode'] = {
            'avg': ffn_decode_avg,
            'min': ffn_decode_min,
            'max': ffn_decode_max
        }
    except Exception as e:
        print(f"\nFFN Decode测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['ffn_decode'] = None

    # ========== IO时间测试 ==========

    try:
        print(f"\n{'='*80}")
        print("第二部分: IO时间测试")
        print(f"{'='*80}")

        # MHA权重传输 (Pinned -> GPU)
        mha_pin2gpu_avg, mha_pin2gpu_min, mha_pin2gpu_max, mha_pin2gpu_bw = benchmark_weight_transfer(
            mha_bytes, "MHA"
        )
        results['mha_pin2gpu'] = {
            'avg': mha_pin2gpu_avg,
            'min': mha_pin2gpu_min,
            'max': mha_pin2gpu_max,
            'bandwidth': mha_pin2gpu_bw
        }
    except Exception as e:
        print(f"\nMHA Pin->GPU测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['mha_pin2gpu'] = None

    try:
        # FFN权重传输 (Pinned -> GPU)
        ffn_pin2gpu_avg, ffn_pin2gpu_min, ffn_pin2gpu_max, ffn_pin2gpu_bw = benchmark_weight_transfer(
            ffn_bytes, "FFN"
        )
        results['ffn_pin2gpu'] = {
            'avg': ffn_pin2gpu_avg,
            'min': ffn_pin2gpu_min,
            'max': ffn_pin2gpu_max,
            'bandwidth': ffn_pin2gpu_bw
        }
    except Exception as e:
        print(f"\nFFN Pin->GPU测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['ffn_pin2gpu'] = None

    try:
        # MHA SSD读取
        mha_ssd2pin_avg, mha_ssd2pin_min, mha_ssd2pin_max, mha_ssd2pin_bw = benchmark_ssd_read(
            mha_bytes, "MHA"
        )
        results['mha_ssd2pin'] = {
            'avg': mha_ssd2pin_avg,
            'min': mha_ssd2pin_min,
            'max': mha_ssd2pin_max,
            'bandwidth': mha_ssd2pin_bw
        }
    except Exception as e:
        print(f"\nMHA SSD读取测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['mha_ssd2pin'] = None

    try:
        # FFN SSD读取
        ffn_ssd2pin_avg, ffn_ssd2pin_min, ffn_ssd2pin_max, ffn_ssd2pin_bw = benchmark_ssd_read(
            ffn_bytes, "FFN"
        )
        results['ffn_ssd2pin'] = {
            'avg': ffn_ssd2pin_avg,
            'min': ffn_ssd2pin_min,
            'max': ffn_ssd2pin_max,
            'bandwidth': ffn_ssd2pin_bw
        }
    except Exception as e:
        print(f"\nFFN SSD读取测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['ffn_ssd2pin'] = None

    # ========== 结果总结 ==========

    print(f"\n{'='*80}")
    print("测试结果总结")
    print(f"{'='*80}")

    print(f"\n权重大小:")
    print(f"  MHA: {mha_bytes / (1024**3):.3f} GB")
    print(f"  FFN: {ffn_bytes / (1024**3):.3f} GB")
    print(f"  总计: {(mha_bytes + ffn_bytes) / (1024**3):.3f} GB")

    # 计算时间
    print(f"\n{'='*80}")
    print("计算时间 (纯计算，无IO)")
    print(f"{'='*80}")

    if results.get('mha_prefill'):
        print(f"\nMHA Prefill (2048 tokens):")
        print(f"  平均: {results['mha_prefill']['avg']:.2f} ms")
        print(f"  最小: {results['mha_prefill']['min']:.2f} ms")
        print(f"  最大: {results['mha_prefill']['max']:.2f} ms")

    if results.get('ffn_prefill'):
        print(f"\nFFN Prefill (2048 tokens):")
        print(f"  平均: {results['ffn_prefill']['avg']:.2f} ms")
        print(f"  最小: {results['ffn_prefill']['min']:.2f} ms")
        print(f"  最大: {results['ffn_prefill']['max']:.2f} ms")

    if results.get('mha_prefill') and results.get('ffn_prefill'):
        total_prefill = results['mha_prefill']['avg'] + results['ffn_prefill']['avg']
        print(f"\nPrefill总计:")
        print(f"  MHA + FFN = {total_prefill:.2f} ms")
        print(f"  MHA占比: {results['mha_prefill']['avg']/total_prefill*100:.1f}%")
        print(f"  FFN占比: {results['ffn_prefill']['avg']/total_prefill*100:.1f}%")

    if results.get('mha_decode'):
        print(f"\nMHA Decode (1 token, ~256 context):")
        print(f"  平均: {results['mha_decode']['avg']:.2f} ms")
        print(f"  最小: {results['mha_decode']['min']:.2f} ms")
        print(f"  最大: {results['mha_decode']['max']:.2f} ms")

    if results.get('ffn_decode'):
        print(f"\nFFN Decode (1 token, ~256 context):")
        print(f"  平均: {results['ffn_decode']['avg']:.2f} ms")
        print(f"  最小: {results['ffn_decode']['min']:.2f} ms")
        print(f"  最大: {results['ffn_decode']['max']:.2f} ms")

    if results.get('mha_decode') and results.get('ffn_decode'):
        total_decode = results['mha_decode']['avg'] + results['ffn_decode']['avg']
        print(f"\nDecode总计:")
        print(f"  MHA + FFN = {total_decode:.2f} ms")
        print(f"  MHA占比: {results['mha_decode']['avg']/total_decode*100:.1f}%")
        print(f"  FFN占比: {results['ffn_decode']['avg']/total_decode*100:.1f}%")

    # IO时间
    print(f"\n{'='*80}")
    print("IO时间")
    print(f"{'='*80}")

    if results.get('mha_pin2gpu'):
        print(f"\nMHA Pinned->GPU (乐观上界):")
        print(f"  平均: {results['mha_pin2gpu']['avg']:.2f} ms")
        print(f"  带宽: {results['mha_pin2gpu']['bandwidth']:.2f} GB/s")

    if results.get('ffn_pin2gpu'):
        print(f"\nFFN Pinned->GPU (乐观上界):")
        print(f"  平均: {results['ffn_pin2gpu']['avg']:.2f} ms")
        print(f"  带宽: {results['ffn_pin2gpu']['bandwidth']:.2f} GB/s")

    if results.get('mha_ssd2pin') and results['mha_ssd2pin']['avg'] is not None:
        print(f"\nMHA SSD->Pinned (使用真实DirectIO/WSM):")
        print(f"  平均: {results['mha_ssd2pin']['avg']:.2f} ms")
        print(f"  带宽: {results['mha_ssd2pin']['bandwidth']:.2f} GB/s")

    if results.get('ffn_ssd2pin') and results['ffn_ssd2pin']['avg'] is not None:
        print(f"\nFFN SSD->Pinned (使用真实DirectIO/WSM):")
        print(f"  平均: {results['ffn_ssd2pin']['avg']:.2f} ms")
        print(f"  带宽: {results['ffn_ssd2pin']['bandwidth']:.2f} GB/s")

    # 重叠分析
    print(f"\n{'='*80}")
    print("重叠可行性分析")
    print(f"{'='*80}")

    # Prefill阶段
    if (results.get('mha_prefill') and results.get('ffn_prefill') and
        results.get('mha_pin2gpu') and results.get('ffn_pin2gpu') and
        results.get('mha_ssd2pin') and results.get('ffn_ssd2pin') and
        results['mha_ssd2pin']['avg'] is not None and results['ffn_ssd2pin']['avg'] is not None):

        print(f"\nPrefill阶段:")

        mha_comp = results['mha_prefill']['avg']
        ffn_comp = results['ffn_prefill']['avg']
        mha_h2d = results['mha_pin2gpu']['avg']
        ffn_h2d = results['ffn_pin2gpu']['avg']
        mha_ssd = results['mha_ssd2pin']['avg']
        ffn_ssd = results['ffn_ssd2pin']['avg']

        # 串行时间
        serial_time = mha_comp + ffn_comp + mha_h2d + ffn_h2d + mha_ssd + ffn_ssd

        print(f"  串行执行总时间: {serial_time:.2f} ms")
        print(f"    MHA计算: {mha_comp:.2f} ms ({mha_comp/serial_time*100:.1f}%)")
        print(f"    FFN计算: {ffn_comp:.2f} ms ({ffn_comp/serial_time*100:.1f}%)")
        print(f"    MHA Pin->GPU: {mha_h2d:.2f} ms ({mha_h2d/serial_time*100:.1f}%)")
        print(f"    FFN Pin->GPU: {ffn_h2d:.2f} ms ({ffn_h2d/serial_time*100:.1f}%)")
        print(f"    MHA SSD->Pin: {mha_ssd:.2f} ms ({mha_ssd/serial_time*100:.1f}%)")
        print(f"    FFN SSD->Pin: {ffn_ssd:.2f} ms ({ffn_ssd/serial_time*100:.1f}%)")

        # 理想流水线: MHA计算时预取FFN，FFN计算时预取下一层MHA
        # 假设层内流水线: MHA_compute || (FFN_h2d + FFN_ssd) 和 FFN_compute || (MHA_next_h2d + MHA_next_ssd)
        layer_pipeline_time = max(
            mha_comp + mha_h2d,  # MHA计算+加载
            ffn_comp + ffn_h2d   # FFN计算+加载
        ) + max(mha_ssd, ffn_ssd)  # SSD读取瓶颈

        print(f"\n  层内流水线分析:")
        print(f"    MHA阶段: max(计算={mha_comp:.2f}ms, IO={mha_h2d + mha_ssd:.2f}ms) = {max(mha_comp, mha_h2d + mha_ssd):.2f} ms")
        print(f"    FFN阶段: max(计算={ffn_comp:.2f}ms, IO={ffn_h2d + ffn_ssd:.2f}ms) = {max(ffn_comp, ffn_h2d + ffn_ssd):.2f} ms")

        if mha_comp > mha_h2d + mha_ssd:
            print(f"    ✓ MHA计算可以隐藏MHA的IO")
        else:
            print(f"    ✗ MHA计算无法隐藏MHA的IO (等待 {mha_h2d + mha_ssd - mha_comp:.2f}ms)")

        if ffn_comp > ffn_h2d + ffn_ssd:
            print(f"    ✓ FFN计算可以隐藏FFN的IO")
        else:
            print(f"    ✗ FFN计算无法隐藏FFN的IO (等待 {ffn_h2d + ffn_ssd - ffn_comp:.2f}ms)")

        ideal_time = max(mha_comp, mha_h2d + mha_ssd) + max(ffn_comp, ffn_h2d + ffn_ssd)
        print(f"\n  理想流水线时间: {ideal_time:.2f} ms")
        print(f"  理论加速比: {serial_time / ideal_time:.2f}x")

    # Decode阶段
    if (results.get('mha_decode') and results.get('ffn_decode') and
        results.get('mha_pin2gpu') and results.get('ffn_pin2gpu') and
        results.get('mha_ssd2pin') and results.get('ffn_ssd2pin') and
        results['mha_ssd2pin']['avg'] is not None and results['ffn_ssd2pin']['avg'] is not None):

        print(f"\nDecode阶段:")

        mha_comp = results['mha_decode']['avg']
        ffn_comp = results['ffn_decode']['avg']
        mha_h2d = results['mha_pin2gpu']['avg']
        ffn_h2d = results['ffn_pin2gpu']['avg']
        mha_ssd = results['mha_ssd2pin']['avg']
        ffn_ssd = results['ffn_ssd2pin']['avg']

        # 串行时间
        serial_time = mha_comp + ffn_comp + mha_h2d + ffn_h2d + mha_ssd + ffn_ssd

        print(f"  串行执行总时间: {serial_time:.2f} ms")
        print(f"    MHA计算: {mha_comp:.2f} ms ({mha_comp/serial_time*100:.1f}%)")
        print(f"    FFN计算: {ffn_comp:.2f} ms ({ffn_comp/serial_time*100:.1f}%)")
        print(f"    MHA Pin->GPU: {mha_h2d:.2f} ms ({mha_h2d/serial_time*100:.1f}%)")
        print(f"    FFN Pin->GPU: {ffn_h2d:.2f} ms ({ffn_h2d/serial_time*100:.1f}%)")
        print(f"    MHA SSD->Pin: {mha_ssd:.2f} ms ({mha_ssd/serial_time*100:.1f}%)")
        print(f"    FFN SSD->Pin: {ffn_ssd:.2f} ms ({ffn_ssd/serial_time*100:.1f}%)")

        print(f"\n  层内流水线分析:")
        print(f"    MHA阶段: max(计算={mha_comp:.2f}ms, IO={mha_h2d + mha_ssd:.2f}ms) = {max(mha_comp, mha_h2d + mha_ssd):.2f} ms")
        print(f"    FFN阶段: max(计算={ffn_comp:.2f}ms, IO={ffn_h2d + ffn_ssd:.2f}ms) = {max(ffn_comp, ffn_h2d + ffn_ssd):.2f} ms")

        if mha_comp > mha_h2d + mha_ssd:
            print(f"    ✓ MHA计算可以隐藏MHA的IO")
        else:
            print(f"    ✗ MHA计算无法隐藏MHA的IO (等待 {mha_h2d + mha_ssd - mha_comp:.2f}ms)")

        if ffn_comp > ffn_h2d + ffn_ssd:
            print(f"    ✓ FFN计算可以隐藏FFN的IO")
        else:
            print(f"    ✗ FFN计算无法隐藏FFN的IO (等待 {ffn_h2d + ffn_ssd - ffn_comp:.2f}ms)")

        ideal_time = max(mha_comp, mha_h2d + mha_ssd) + max(ffn_comp, ffn_h2d + ffn_ssd)
        print(f"\n  理想流水线时间: {ideal_time:.2f} ms")
        print(f"  理论加速比: {serial_time / ideal_time:.2f}x")

        # 预取距离计算
        print(f"\n  预取距离建议:")
        mha_prefetch_dist = max(1, int((mha_h2d + mha_ssd) / mha_comp) + 1)
        ffn_prefetch_dist = max(1, int((ffn_h2d + ffn_ssd) / ffn_comp) + 1)
        print(f"    MHA: {mha_prefetch_dist} 层")
        print(f"    FFN: {ffn_prefetch_dist} 层")

    print(f"\n{'='*80}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
