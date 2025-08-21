#!/usr/bin/env python3
"""
GPU内存碎片化检测工具
监控CUDA内存分配和碎片化情况
"""

import torch
import gc
import time
import json
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

def get_gpu_memory_info():
    """获取详细的GPU内存信息"""
    if not torch.cuda.is_available():
        return None
    
    # 基础内存信息
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_allocated = torch.cuda.max_memory_allocated()
    max_reserved = torch.cuda.max_memory_reserved()
    
    # 内存统计信息
    memory_stats = torch.cuda.memory_stats()
    
    return {
        'allocated_mb': allocated / 1024**2,
        'reserved_mb': reserved / 1024**2,
        'max_allocated_mb': max_allocated / 1024**2,
        'max_reserved_mb': max_reserved / 1024**2,
        'fragmentation_ratio': (reserved - allocated) / reserved if reserved > 0 else 0,
        'allocation_count': memory_stats.get('allocation.all.current', 0),
        'free_count': memory_stats.get('free_retries.all.current', 0),
        'segment_count': memory_stats.get('segment.all.current', 0),
        'large_pool_allocated': memory_stats.get('allocated_bytes.large_pool.current', 0) / 1024**2,
        'small_pool_allocated': memory_stats.get('allocated_bytes.small_pool.current', 0) / 1024**2,
    }

def test_allocation_pattern(sizes: List[int], pattern: str = "sequential") -> List[Dict]:
    """测试不同分配模式的碎片化情况"""
    print(f"🧪 测试分配模式: {pattern}")
    
    results = []
    tensors = []
    
    # 清理初始状态
    torch.cuda.empty_cache()
    initial_info = get_gpu_memory_info()
    results.append({
        'step': 'initial',
        'pattern': pattern,
        **initial_info
    })
    
    if pattern == "sequential":
        # 顺序分配
        for i, size in enumerate(sizes):
            try:
                tensor = torch.randn(size, device='cuda')
                tensors.append(tensor)
                
                info = get_gpu_memory_info()
                results.append({
                    'step': f'alloc_{i}',
                    'size_mb': size * 4 / 1024**2,  # float32 = 4 bytes
                    'pattern': pattern,
                    **info
                })
                
            except torch.cuda.OutOfMemoryError:
                print(f"❌ OOM at allocation {i}, size: {size}")
                break
    
    elif pattern == "random_free":
        # 随机分配和释放
        for i, size in enumerate(sizes):
            try:
                tensor = torch.randn(size, device='cuda')
                tensors.append(tensor)
                
                # 随机释放一些tensor
                if len(tensors) > 3 and i % 3 == 0:
                    idx_to_free = np.random.randint(0, len(tensors))
                    del tensors[idx_to_free]
                    tensors = [t for j, t in enumerate(tensors) if j != idx_to_free]
                    gc.collect()
                    torch.cuda.empty_cache()
                
                info = get_gpu_memory_info()
                results.append({
                    'step': f'alloc_free_{i}',
                    'size_mb': size * 4 / 1024**2,
                    'pattern': pattern,
                    **info
                })
                
            except torch.cuda.OutOfMemoryError:
                print(f"❌ OOM at allocation {i}, size: {size}")
                break
    
    elif pattern == "alternating_sizes":
        # 交替大小分配
        large_size = max(sizes)
        small_size = min(sizes)
        
        for i in range(len(sizes)):
            try:
                size = large_size if i % 2 == 0 else small_size
                tensor = torch.randn(size, device='cuda')
                tensors.append(tensor)
                
                info = get_gpu_memory_info()
                results.append({
                    'step': f'alternating_{i}',
                    'size_mb': size * 4 / 1024**2,
                    'pattern': pattern,
                    **info
                })
                
            except torch.cuda.OutOfMemoryError:
                print(f"❌ OOM at allocation {i}")
                break
    
    # 清理
    del tensors
    gc.collect()
    torch.cuda.empty_cache()
    
    final_info = get_gpu_memory_info()
    results.append({
        'step': 'final_cleanup',
        'pattern': pattern,
        **final_info
    })
    
    return results

def analyze_fragmentation(results: List[Dict]) -> Dict:
    """分析碎片化程度"""
    
    max_fragmentation = max(r['fragmentation_ratio'] for r in results if 'fragmentation_ratio' in r)
    avg_fragmentation = np.mean([r['fragmentation_ratio'] for r in results if 'fragmentation_ratio' in r])
    
    max_segments = max(r['segment_count'] for r in results if 'segment_count' in r)
    
    # 内存效率：分配的内存 / 保留的内存
    efficiency_ratios = [r['allocated_mb'] / r['reserved_mb'] if r['reserved_mb'] > 0 else 1.0 
                        for r in results if 'allocated_mb' in r and 'reserved_mb' in r]
    min_efficiency = min(efficiency_ratios) if efficiency_ratios else 1.0
    
    return {
        'max_fragmentation_ratio': max_fragmentation,
        'avg_fragmentation_ratio': avg_fragmentation,
        'max_segments': max_segments,
        'min_efficiency': min_efficiency,
        'severity': 'HIGH' if max_fragmentation > 0.3 else 'MEDIUM' if max_fragmentation > 0.15 else 'LOW'
    }

def plot_fragmentation_timeline(results_by_pattern: Dict[str, List[Dict]]):
    """绘制碎片化时间线图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('GPU Memory Fragmentation Analysis', fontsize=16)
    
    for pattern, results in results_by_pattern.items():
        steps = list(range(len(results)))
        fragmentation_ratios = [r.get('fragmentation_ratio', 0) for r in results]
        allocated_mb = [r.get('allocated_mb', 0) for r in results]
        reserved_mb = [r.get('reserved_mb', 0) for r in results]
        segment_counts = [r.get('segment_count', 0) for r in results]
        
        # 碎片化比例
        axes[0, 0].plot(steps, fragmentation_ratios, label=pattern, marker='o')
        axes[0, 0].set_title('Fragmentation Ratio Over Time')
        axes[0, 0].set_ylabel('Fragmentation Ratio')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 内存使用
        axes[0, 1].plot(steps, allocated_mb, label=f'{pattern} (allocated)', linestyle='-')
        axes[0, 1].plot(steps, reserved_mb, label=f'{pattern} (reserved)', linestyle='--')
        axes[0, 1].set_title('Memory Usage Over Time')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 段数量
        axes[1, 0].plot(steps, segment_counts, label=pattern, marker='s')
        axes[1, 0].set_title('Memory Segments Over Time')
        axes[1, 0].set_ylabel('Segment Count')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 效率比
        efficiency = [a/r if r > 0 else 1.0 for a, r in zip(allocated_mb, reserved_mb)]
        axes[1, 1].plot(steps, efficiency, label=pattern, marker='^')
        axes[1, 1].set_title('Memory Efficiency Over Time')
        axes[1, 1].set_ylabel('Efficiency (allocated/reserved)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('memory_fragmentation_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 图表已保存为 memory_fragmentation_analysis.png")

def simulate_weight_streaming_memory_pattern():
    """模拟权重流式传输的内存分配模式"""
    print("🔄 模拟权重流式传输内存模式")
    
    # 模拟不同大小的权重矩阵
    weight_sizes = [
        1024 * 1024,      # 1M elements ~4MB
        2048 * 2048,      # 4M elements ~16MB  
        1024 * 4096,      # 4M elements ~16MB
        4096 * 1024,      # 4M elements ~16MB
        512 * 512,        # 256K elements ~1MB
        8192 * 512,       # 4M elements ~16MB
    ]
    
    results = []
    weight_cache = {}
    max_cached_layers = 4
    
    torch.cuda.empty_cache()
    initial_info = get_gpu_memory_info()
    results.append({
        'step': 'initial',
        'operation': 'start',
        **initial_info
    })
    
    # 模拟多层推理过程
    for layer_id in range(8):
        print(f"  处理Layer {layer_id}")
        
        # 1. 检查缓存，如果超出限制则逐出
        if len(weight_cache) >= max_cached_layers:
            # LRU逐出
            oldest_layer = min(weight_cache.keys())
            del weight_cache[oldest_layer]
            gc.collect()
            
            info = get_gpu_memory_info()
            results.append({
                'step': f'layer_{layer_id}_evict',
                'operation': f'evict_layer_{oldest_layer}',
                **info
            })
        
        # 2. 加载当前层权重
        try:
            layer_weights = []
            for i, size in enumerate(weight_sizes):
                weight = torch.randn(size, device='cuda')
                layer_weights.append(weight)
            
            weight_cache[layer_id] = layer_weights
            
            info = get_gpu_memory_info()
            results.append({
                'step': f'layer_{layer_id}_load',
                'operation': f'load_layer_{layer_id}',
                **info
            })
            
        except torch.cuda.OutOfMemoryError:
            print(f"❌ OOM loading layer {layer_id}")
            break
        
        # 3. 模拟计算（小量激活内存）
        try:
            activations = torch.randn(2048 * 1024, device='cuda')  # ~8MB
            
            info = get_gpu_memory_info()
            results.append({
                'step': f'layer_{layer_id}_compute',
                'operation': f'compute_layer_{layer_id}',
                **info
            })
            
            del activations
            
        except torch.cuda.OutOfMemoryError:
            print(f"❌ OOM during computation layer {layer_id}")
    
    # 清理
    weight_cache.clear()
    gc.collect()
    torch.cuda.empty_cache()
    
    final_info = get_gpu_memory_info()
    results.append({
        'step': 'final_cleanup',
        'operation': 'cleanup',
        **final_info
    })
    
    return results

def main():
    print("🔍 GPU内存碎片化检测开始")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return
    
    # 获取GPU信息
    gpu_name = torch.cuda.get_device_name()
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🔧 GPU: {gpu_name}")
    print(f"💾 总内存: {total_memory:.1f} GB")
    
    # 测试大小（元素数量）
    test_sizes = [
        512 * 512,        # ~1MB
        1024 * 1024,      # ~4MB
        2048 * 1024,      # ~8MB
        2048 * 2048,      # ~16MB
        1024 * 4096,      # ~16MB
        4096 * 1024,      # ~16MB
    ]
    
    results_by_pattern = {}
    
    # 测试不同分配模式
    patterns = ["sequential", "random_free", "alternating_sizes"]
    
    for pattern in patterns:
        print(f"\n🧪 测试模式: {pattern}")
        results = test_allocation_pattern(test_sizes, pattern)
        results_by_pattern[pattern] = results
        
        # 分析这个模式的碎片化
        analysis = analyze_fragmentation(results)
        print(f"  最大碎片化比例: {analysis['max_fragmentation_ratio']:.3f}")
        print(f"  平均碎片化比例: {analysis['avg_fragmentation_ratio']:.3f}")
        print(f"  最大段数: {analysis['max_segments']}")
        print(f"  最低效率: {analysis['min_efficiency']:.3f}")
        print(f"  严重程度: {analysis['severity']}")
    
    # 测试权重流式传输模式
    print(f"\n🔄 测试权重流式传输模式")
    streaming_results = simulate_weight_streaming_memory_pattern()
    results_by_pattern["weight_streaming"] = streaming_results
    
    streaming_analysis = analyze_fragmentation(streaming_results)
    print(f"  权重流式传输碎片化: {streaming_analysis['max_fragmentation_ratio']:.3f}")
    print(f"  权重流式传输严重程度: {streaming_analysis['severity']}")
    
    # 保存结果
    with open('memory_fragmentation_results.json', 'w') as f:
        json.dump(results_by_pattern, f, indent=2)
    print(f"\n💾 结果已保存到 memory_fragmentation_results.json")
    
    # 生成分析报告
    print(f"\n📊 生成分析报告...")
    try:
        plot_fragmentation_timeline(results_by_pattern)
    except Exception as e:
        print(f"❌ 生成图表失败: {e}")
    
    # 总结建议
    print(f"\n📋 碎片化检测总结")
    print("=" * 30)
    
    all_analyses = {}
    for pattern, results in results_by_pattern.items():
        all_analyses[pattern] = analyze_fragmentation(results)
    
    worst_pattern = max(all_analyses.keys(), 
                       key=lambda k: all_analyses[k]['max_fragmentation_ratio'])
    best_pattern = min(all_analyses.keys(), 
                      key=lambda k: all_analyses[k]['max_fragmentation_ratio'])
    
    print(f"🔴 最严重碎片化模式: {worst_pattern} ({all_analyses[worst_pattern]['max_fragmentation_ratio']:.3f})")
    print(f"🟢 最轻微碎片化模式: {best_pattern} ({all_analyses[best_pattern]['max_fragmentation_ratio']:.3f})")
    
    # 建议
    if all_analyses[worst_pattern]['max_fragmentation_ratio'] > 0.3:
        print("\n⚠️  高碎片化风险检测到!")
        print("建议:")
        print("1. 增加权重缓存大小 (max_cached_layers)")
        print("2. 使用内存池预分配")
        print("3. 定期调用 torch.cuda.empty_cache()")
        print("4. 优化权重加载顺序")

if __name__ == "__main__":
    main()