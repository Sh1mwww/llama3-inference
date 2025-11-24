#!/usr/bin/env python3
"""
测量70B模型单层操作的时间:
1. Prefill计算一个layer (2048 tokens)
2. 将一个layer的weight从pinned memory到GPU
3. 将一个layer的weight从raw block device到pinned memory
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
from llama3.layers import EncoderBlock, precompute_theta_pos_frequencies


def get_layer_weight_size(args: ModelArgs) -> int:
    """
    计算一个EncoderBlock的权重总大小（字节）

    对于70B模型:
    - dim = 8192
    - n_heads = 64
    - n_kv_heads = 8
    - ffn_hidden_dim ≈ dim * 4 * 1.3 ≈ 43000

    Attention权重:
    - wq: (dim, dim) = (8192, 8192)
    - wk: (kv_dim, dim) = (1024, 8192)
    - wv: (kv_dim, dim) = (1024, 8192)
    - wo: (dim, dim) = (8192, 8192)

    FFN权重:
    - w1: (hidden, dim) ≈ (43008, 8192)
    - w2: (dim, hidden) ≈ (8192, 43008)
    - w3: (hidden, dim) ≈ (43008, 8192)
    """
    dim = args.dim
    n_heads = args.n_heads
    n_kv_heads = args.n_kv_heads
    head_dim = dim // n_heads
    kv_dim = n_kv_heads * head_dim

    # 计算FFN hidden dim
    if args.ffn_dim_multiplier is not None:
        hidden_dim = int(args.ffn_dim_multiplier * dim)
    else:
        hidden_dim = 4 * dim

    hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

    # Attention权重参数量
    attn_params = (
        dim * dim +      # wq
        kv_dim * dim +   # wk
        kv_dim * dim +   # wv
        dim * dim        # wo
    )

    # FFN权重参数量
    ffn_params = (
        hidden_dim * dim +  # w1
        dim * hidden_dim +  # w2
        hidden_dim * dim    # w3
    )

    # RMSNorm参数量（2个norm层）
    norm_params = dim * 2

    total_params = attn_params + ffn_params + norm_params

    # bfloat16 = 2 bytes per parameter
    total_bytes = total_params * 2

    return total_bytes, {
        'attn_params': attn_params,
        'ffn_params': ffn_params,
        'norm_params': norm_params,
        'total_params': total_params,
        'hidden_dim': hidden_dim,
        'kv_dim': kv_dim
    }


def benchmark_prefill_computation(args: ModelArgs, seq_len: int = 2048, warmup: int = 3, iterations: int = 10):
    """
    测量prefill阶段单个layer的计算时间
    """
    print(f"\n{'='*80}")
    print(f"测试1a: Prefill计算时间 (seq_len={seq_len})")
    print(f"{'='*80}")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # 创建一个EncoderBlock
    layer = EncoderBlock(args, layer_id=0).to(device=device, dtype=dtype)

    # 禁用KV offloading以测量纯计算时间
    if hasattr(layer.attention, 'kv_offloader'):
        layer.attention.kv_offloader = None
        print("已禁用 KV offloading (测量纯计算时间)")

    # 准备输入
    batch_size = 1
    x = torch.randn(batch_size, seq_len, args.dim, device=device, dtype=dtype)
    start_pos = 0

    # 预计算位置编码
    freqs = precompute_theta_pos_frequencies(
        args.dim // args.n_heads,
        args.max_seq_len * 2,
        device=device,
        theta=args.rope_theta,
    )

    # Warmup
    print(f"Warmup ({warmup} iterations)...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = layer(x, start_pos, freqs)
    torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({iterations} iterations)...")
    times = []
    for i in range(iterations):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            output = layer(x, start_pos, freqs)

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000
        times.append(elapsed_ms)
        print(f"  Iteration {i+1}: {elapsed_ms:.2f} ms")

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
    del layer, x, output, freqs
    torch.cuda.empty_cache()

    return avg_time, min_time, max_time


def benchmark_decode_computation(args: ModelArgs, warmup: int = 3, iterations: int = 100):
    """
    测量decode阶段单个layer的计算时间（每次1个token）
    """
    print(f"\n{'='*80}")
    print(f"测试1b: Decode计算时间 (1 token per step, KV cache = 256 tokens)")
    print(f"{'='*80}")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # 创建一个EncoderBlock
    layer = EncoderBlock(args, layer_id=0).to(device=device, dtype=dtype)

    # 禁用KV offloading以测量纯计算时间
    if hasattr(layer.attention, 'kv_offloader'):
        layer.attention.kv_offloader = None
        print("已禁用 KV offloading (测量纯计算时间)")

    # 预计算位置编码
    freqs = precompute_theta_pos_frequencies(
        args.dim // args.n_heads,
        args.max_seq_len * 2,
        device=device,
        theta=args.rope_theta,
    )

    # 先做一次prefill来初始化KV cache (使用256 tokens)
    print(f"初始化KV cache (prefill 256 tokens)...")
    batch_size = 1
    prefill_len = 256
    prefill_x = torch.randn(batch_size, prefill_len, args.dim, device=device, dtype=dtype)
    with torch.no_grad():
        _ = layer(prefill_x, 0, freqs)
    del prefill_x
    torch.cuda.synchronize()

    # 现在测试decode - 每次只有1个token
    seq_len = 1
    x = torch.randn(batch_size, seq_len, args.dim, device=device, dtype=dtype)
    start_pos = prefill_len  # 从prefill位置之后开始

    # Warmup
    print(f"Warmup ({warmup} iterations)...")
    for i in range(warmup):
        with torch.no_grad():
            _ = layer(x, start_pos + i, freqs)
    torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({iterations} iterations)...")
    times = []
    for i in range(iterations):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            output = layer(x, start_pos + warmup + i, freqs)

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000
        times.append(elapsed_ms)
        if i < 10:  # 只打印前10次
            print(f"  Iteration {i+1}: {elapsed_ms:.2f} ms")

    # 统计
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"  ... ({iterations - 10} more iterations)")
    print(f"\n结果:")
    print(f"  平均时间: {avg_time:.2f} ms")
    print(f"  最小时间: {min_time:.2f} ms")
    print(f"  最大时间: {max_time:.2f} ms")
    print(f"  标准差: {(sum((t - avg_time)**2 for t in times) / len(times))**0.5:.2f} ms")

    # 清理
    del layer, x, output, freqs
    torch.cuda.empty_cache()

    return avg_time, min_time, max_time


def benchmark_pinned_to_gpu(layer_size_bytes: int, warmup: int = 3, iterations: int = 20):
    """
    测量从pinned memory到GPU的传输时间
    """
    print(f"\n{'='*80}")
    print(f"测试2: Pinned Memory -> GPU 传输时间")
    print(f"{'='*80}")

    device = torch.device("cuda:0")

    # 计算需要的元素数量（bfloat16 = 2 bytes）
    num_elements = layer_size_bytes // 2

    # 创建pinned memory tensor
    print(f"创建 {layer_size_bytes / (1024**3):.2f} GB 的 pinned memory tensor...")
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
    bandwidth_gbps = layer_size_bytes / (avg_time / 1000) / (1024**3)

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


def benchmark_ssd_to_pinned(layer_size_bytes: int, device_path: str = "/dev/nvme0n1p4",
                            warmup: int = 2, iterations: int = 10):
    """
    测量从raw block device到pinned memory的读取时间
    """
    print(f"\n{'='*80}")
    print(f"测试3: Raw Block Device -> Pinned Memory 读取时间")
    print(f"{'='*80}")

    # 检查设备是否存在
    if not os.path.exists(device_path):
        print(f"错误: 块设备 {device_path} 不存在")
        print("可用的块设备:")
        os.system("lsblk -o NAME,SIZE,TYPE,MOUNTPOINT | grep -E 'disk|part'")
        return None, None, None, None

    # 检查权限
    if not os.access(device_path, os.R_OK):
        print(f"错误: 没有读取 {device_path} 的权限")
        print(f"请运行: sudo chmod 666 {device_path}")
        return None, None, None, None

    try:
        # 对齐到4KB（O_DIRECT要求）
        aligned_size = ((layer_size_bytes + 4095) // 4096) * 4096

        # 创建pinned memory buffer
        print(f"创建 {aligned_size / (1024**3):.2f} GB 的 pinned memory buffer...")
        buffer = torch.empty(aligned_size // 2, dtype=torch.bfloat16, pin_memory=True)
        buffer_ptr = buffer.data_ptr()

        # 使用O_DIRECT打开设备
        print(f"打开块设备: {device_path} (O_DIRECT mode)")
        import fcntl
        fd = os.open(device_path, os.O_RDONLY | os.O_DIRECT)

        # Warmup
        print(f"Warmup ({warmup} iterations)...")
        for _ in range(warmup):
            os.lseek(fd, 0, os.SEEK_SET)
            os.read(fd, aligned_size)

        # Benchmark
        print(f"Benchmarking ({iterations} iterations)...")
        times = []
        for i in range(iterations):
            # 随机选择一个offset（对齐到4KB）
            # 假设设备至少有100GB
            max_offset_blocks = min(100 * 1024**3 // aligned_size, 1000)
            offset_blocks = i % max_offset_blocks
            offset = offset_blocks * aligned_size

            start_time = time.perf_counter()

            os.lseek(fd, offset, os.SEEK_SET)
            data = os.read(fd, aligned_size)

            end_time = time.perf_counter()

            elapsed_ms = (end_time - start_time) * 1000
            times.append(elapsed_ms)
            print(f"  Iteration {i+1} (offset={offset/(1024**3):.2f}GB): {elapsed_ms:.2f} ms")

        os.close(fd)

        # 统计
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        bandwidth_gbps = layer_size_bytes / (avg_time / 1000) / (1024**3)

        print(f"\n结果:")
        print(f"  平均时间: {avg_time:.2f} ms")
        print(f"  最小时间: {min_time:.2f} ms")
        print(f"  最大时间: {max_time:.2f} ms")
        print(f"  标准差: {(sum((t - avg_time)**2 for t in times) / len(times))**0.5:.2f} ms")
        print(f"  平均带宽: {bandwidth_gbps:.2f} GB/s")

        # 清理
        del buffer

        return avg_time, min_time, max_time, bandwidth_gbps

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def main():
    print("="*80)
    print("70B 模型单层操作性能测试")
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

    # 计算单层权重大小
    layer_size_bytes, details = get_layer_weight_size(args)

    print(f"\n模型配置:")
    print(f"  dim: {args.dim}")
    print(f"  n_layers: {args.n_layers}")
    print(f"  n_heads: {args.n_heads}")
    print(f"  n_kv_heads: {args.n_kv_heads}")
    print(f"  vocab_size: {args.vocab_size}")
    print(f"\n单层权重详情:")
    print(f"  Attention 参数: {details['attn_params']:,} ({details['attn_params']*2/(1024**3):.3f} GB)")
    print(f"  FFN 参数: {details['ffn_params']:,} ({details['ffn_params']*2/(1024**3):.3f} GB)")
    print(f"  Norm 参数: {details['norm_params']:,} ({details['norm_params']*2/(1024**3):.3f} GB)")
    print(f"  总参数: {details['total_params']:,} ({details['total_params']*2/(1024**3):.3f} GB)")
    print(f"  单层大小: {layer_size_bytes / (1024**3):.3f} GB")
    print(f"  FFN hidden_dim: {details['hidden_dim']}")
    print(f"  KV dim: {details['kv_dim']}")

    # 运行测试
    results = {}

    try:
        # 测试1a: Prefill计算
        prefill_avg, prefill_min, prefill_max = benchmark_prefill_computation(args, seq_len=2048)
        results['prefill_computation'] = {
            'avg': prefill_avg,
            'min': prefill_min,
            'max': prefill_max
        }
    except Exception as e:
        print(f"\n测试1a失败: {e}")
        import traceback
        traceback.print_exc()
        results['prefill_computation'] = None

    try:
        # 测试1b: Decode计算
        decode_avg, decode_min, decode_max = benchmark_decode_computation(args)
        results['decode_computation'] = {
            'avg': decode_avg,
            'min': decode_min,
            'max': decode_max
        }
    except Exception as e:
        print(f"\n测试1b失败: {e}")
        import traceback
        traceback.print_exc()
        results['decode_computation'] = None

    try:
        # 测试2: Pinned -> GPU
        pin2gpu_avg, pin2gpu_min, pin2gpu_max, pin2gpu_bw = benchmark_pinned_to_gpu(layer_size_bytes)
        results['pinned_to_gpu'] = {
            'avg': pin2gpu_avg,
            'min': pin2gpu_min,
            'max': pin2gpu_max,
            'bandwidth': pin2gpu_bw
        }
    except Exception as e:
        print(f"\n测试2失败: {e}")
        import traceback
        traceback.print_exc()
        results['pinned_to_gpu'] = None

    try:
        # 测试3: SSD -> Pinned
        ssd2pin_avg, ssd2pin_min, ssd2pin_max, ssd2pin_bw = benchmark_ssd_to_pinned(
            layer_size_bytes,
            device_path="/dev/nvme0n1p4"
        )
        results['ssd_to_pinned'] = {
            'avg': ssd2pin_avg,
            'min': ssd2pin_min,
            'max': ssd2pin_max,
            'bandwidth': ssd2pin_bw
        }
    except Exception as e:
        print(f"\n测试3失败: {e}")
        import traceback
        traceback.print_exc()
        results['ssd_to_pinned'] = None

    # 打印总结
    print(f"\n{'='*80}")
    print("测试结果总结")
    print(f"{'='*80}")

    print(f"\n单层权重大小: {layer_size_bytes / (1024**3):.3f} GB")

    if results['prefill_computation']:
        print(f"\n1a. Prefill计算时间 (1 layer, 2048 tokens):")
        print(f"   平均: {results['prefill_computation']['avg']:.2f} ms")
        print(f"   最小: {results['prefill_computation']['min']:.2f} ms")
        print(f"   最大: {results['prefill_computation']['max']:.2f} ms")

    if results['decode_computation']:
        print(f"\n1b. Decode计算时间 (1 layer, 1 token):")
        print(f"   平均: {results['decode_computation']['avg']:.2f} ms")
        print(f"   最小: {results['decode_computation']['min']:.2f} ms")
        print(f"   最大: {results['decode_computation']['max']:.2f} ms")
        if results['prefill_computation']:
            speedup = results['prefill_computation']['avg'] / results['decode_computation']['avg']
            print(f"   Decode比Prefill快: {speedup:.1f}x")

    if results['pinned_to_gpu']:
        print(f"\n2. Pinned Memory -> GPU 传输时间:")
        print(f"   平均: {results['pinned_to_gpu']['avg']:.2f} ms")
        print(f"   最小: {results['pinned_to_gpu']['min']:.2f} ms")
        print(f"   最大: {results['pinned_to_gpu']['max']:.2f} ms")
        print(f"   带宽: {results['pinned_to_gpu']['bandwidth']:.2f} GB/s")

    if results['ssd_to_pinned'] and results['ssd_to_pinned']['avg'] is not None:
        print(f"\n3. Raw Block Device -> Pinned Memory 读取时间:")
        print(f"   平均: {results['ssd_to_pinned']['avg']:.2f} ms")
        print(f"   最小: {results['ssd_to_pinned']['min']:.2f} ms")
        print(f"   最大: {results['ssd_to_pinned']['max']:.2f} ms")
        print(f"   带宽: {results['ssd_to_pinned']['bandwidth']:.2f} GB/s")

    # 重叠分析 - Prefill阶段
    if results.get('prefill_computation') and results.get('pinned_to_gpu') and results.get('ssd_to_pinned') and results['ssd_to_pinned']['avg'] is not None:
        print(f"\n{'='*80}")
        print("重叠可行性分析 - Prefill阶段")
        print(f"{'='*80}")

        comp_time = results['prefill_computation']['avg']
        pin2gpu_time = results['pinned_to_gpu']['avg']
        ssd2pin_time = results['ssd_to_pinned']['avg']

        total_serial = comp_time + pin2gpu_time + ssd2pin_time

        print(f"\n串行执行总时间: {total_serial:.2f} ms")
        print(f"  计算: {comp_time:.2f} ms ({comp_time/total_serial*100:.1f}%)")
        print(f"  Pin->GPU: {pin2gpu_time:.2f} ms ({pin2gpu_time/total_serial*100:.1f}%)")
        print(f"  SSD->Pin: {ssd2pin_time:.2f} ms ({ssd2pin_time/total_serial*100:.1f}%)")

        # 分析重叠可能性
        if comp_time > pin2gpu_time:
            print(f"\n✓ 计算可以隐藏 Pin->GPU 传输 (计算时间 > 传输时间)")
            print(f"  节省时间: {pin2gpu_time:.2f} ms")
        else:
            print(f"\n✗ 计算无法完全隐藏 Pin->GPU 传输")
            print(f"  额外等待: {pin2gpu_time - comp_time:.2f} ms")

        if comp_time > ssd2pin_time:
            print(f"\n✓ 计算可以隐藏 SSD->Pin 读取 (计算时间 > 读取时间)")
            print(f"  节省时间: {ssd2pin_time:.2f} ms")
        else:
            print(f"\n✗ 计算无法完全隐藏 SSD->Pin 读取")
            print(f"  额外等待: {ssd2pin_time - comp_time:.2f} ms")

        # 理想流水线分析
        pipeline_time = max(comp_time, pin2gpu_time, ssd2pin_time)
        print(f"\n理想流水线时间 (完美重叠): {pipeline_time:.2f} ms")
        print(f"理论加速比: {total_serial / pipeline_time:.2f}x")

        # 预取距离建议
        print(f"\n预取距离建议:")
        if ssd2pin_time < comp_time:
            prefetch_dist = 1
            print(f"  SSD->Pin: 预取距离 = {prefetch_dist} (读取可被计算隐藏)")
        else:
            prefetch_dist = max(1, int(ssd2pin_time / comp_time) + 1)
            print(f"  SSD->Pin: 预取距离 = {prefetch_dist} (需要提前 {prefetch_dist} 层开始读取)")

        if pin2gpu_time < comp_time:
            gpu_prefetch_dist = 1
            print(f"  Pin->GPU: 预取距离 = {gpu_prefetch_dist} (传输可被计算隐藏)")
        else:
            gpu_prefetch_dist = max(1, int(pin2gpu_time / comp_time) + 1)
            print(f"  Pin->GPU: 预取距离 = {gpu_prefetch_dist} (需要提前 {gpu_prefetch_dist} 层开始传输)")

    # 重叠分析 - Decode阶段
    if results.get('decode_computation') and results.get('pinned_to_gpu') and results.get('ssd_to_pinned') and results['ssd_to_pinned']['avg'] is not None:
        print(f"\n{'='*80}")
        print("重叠可行性分析 - Decode阶段")
        print(f"{'='*80}")

        comp_time = results['decode_computation']['avg']
        pin2gpu_time = results['pinned_to_gpu']['avg']
        ssd2pin_time = results['ssd_to_pinned']['avg']

        total_serial = comp_time + pin2gpu_time + ssd2pin_time

        print(f"\n串行执行总时间: {total_serial:.2f} ms")
        print(f"  计算: {comp_time:.2f} ms ({comp_time/total_serial*100:.1f}%)")
        print(f"  Pin->GPU: {pin2gpu_time:.2f} ms ({pin2gpu_time/total_serial*100:.1f}%)")
        print(f"  SSD->Pin: {ssd2pin_time:.2f} ms ({ssd2pin_time/total_serial*100:.1f}%)")

        # 分析重叠可能性
        if comp_time > pin2gpu_time:
            print(f"\n✓ 计算可以隐藏 Pin->GPU 传输")
        else:
            print(f"\n✗ 计算无法隐藏 Pin->GPU 传输 (需要等待 {pin2gpu_time - comp_time:.2f} ms)")

        if comp_time > ssd2pin_time:
            print(f"\n✓ 计算可以隐藏 SSD->Pin 读取")
        else:
            print(f"\n✗ 计算无法隐藏 SSD->Pin 读取 (需要等待 {ssd2pin_time - comp_time:.2f} ms)")

        # 理想流水线分析
        pipeline_time = max(comp_time, pin2gpu_time, ssd2pin_time)
        print(f"\n理想流水线时间 (完美重叠): {pipeline_time:.2f} ms")
        print(f"理论加速比: {total_serial / pipeline_time:.2f}x")

        # 预取距离建议
        print(f"\n预取距离建议 (Decode):")
        if ssd2pin_time < comp_time:
            prefetch_dist = 1
            print(f"  SSD->Pin: 预取距离 = {prefetch_dist}")
        else:
            prefetch_dist = max(1, int(ssd2pin_time / comp_time) + 1)
            print(f"  SSD->Pin: 预取距离 = {prefetch_dist} (需要提前 {prefetch_dist} 层)")

        if pin2gpu_time < comp_time:
            gpu_prefetch_dist = 1
            print(f"  Pin->GPU: 预取距离 = {gpu_prefetch_dist}")
        else:
            gpu_prefetch_dist = max(1, int(pin2gpu_time / comp_time) + 1)
            print(f"  Pin->GPU: 预取距离 = {gpu_prefetch_dist} (需要提前 {gpu_prefetch_dist} 层)")

    print(f"\n{'='*80}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
