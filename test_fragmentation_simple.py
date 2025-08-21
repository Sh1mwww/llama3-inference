#!/usr/bin/env python3
"""
简单的GPU内存碎片化检测脚本
快速检测权重流式传输过程中的内存碎片化情况
"""

import torch
import gc
import time
import json

def quick_fragmentation_test():
    """快速碎片化测试"""
    print("🔍 GPU内存碎片化快速检测")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return
    
    # 清理初始状态
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    def get_fragmentation_info():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        stats = torch.cuda.memory_stats()
        
        return {
            'allocated_mb': allocated / 1024**2,
            'reserved_mb': reserved / 1024**2,
            'fragmentation_ratio': (reserved - allocated) / reserved if reserved > 0 else 0,
            'segments': stats.get('segment.all.current', 0),
        }
    
    print("📊 基础内存信息:")
    initial = get_fragmentation_info()
    print(f"   初始分配: {initial['allocated_mb']:.1f} MB")
    print(f"   初始保留: {initial['reserved_mb']:.1f} MB")
    print(f"   初始碎片化: {initial['fragmentation_ratio']:.3f}")
    
    # 模拟权重分配和释放模式
    tensors = []
    fragmentation_history = []
    
    # 模拟不同大小的权重矩阵
    weight_sizes = [
        (2048, 2048),    # ~16MB
        (4096, 1024),    # ~16MB  
        (1024, 4096),    # ~16MB
        (8192, 512),     # ~16MB
        (512, 8192),     # ~16MB
        (1024, 1024),    # ~4MB
    ]
    
    print("\n🔄 模拟权重加载模式:")
    max_cached = 4  # 模拟max_cached_layers=4
    
    for step in range(10):
        print(f"   Step {step+1}: ", end="")
        
        try:
            # 模拟权重加载
            if len(tensors) >= max_cached:
                # LRU逐出
                del tensors[0]
                tensors = tensors[1:]
                gc.collect()
                print("逐出+", end="")
            
            # 加载新权重（选择不同大小）
            size = weight_sizes[step % len(weight_sizes)]
            weight = torch.randn(size, device='cuda')
            tensors.append(weight)
            
            info = get_fragmentation_info()
            fragmentation_history.append(info)
            
            print(f"加载{size} -> 碎片化: {info['fragmentation_ratio']:.3f} "
                  f"(段数: {info['segments']})")
            
            # 检查高碎片化
            if info['fragmentation_ratio'] > 0.3:
                print(f"      ⚠️  高碎片化检测到!")
                
        except torch.cuda.OutOfMemoryError:
            print("❌ OOM")
            break
    
    # 分析结果
    if fragmentation_history:
        max_frag = max(h['fragmentation_ratio'] for h in fragmentation_history)
        avg_frag = sum(h['fragmentation_ratio'] for h in fragmentation_history) / len(fragmentation_history)
        max_segments = max(h['segments'] for h in fragmentation_history)
        
        print(f"\n📈 碎片化分析结果:")
        print(f"   最大碎片化比例: {max_frag:.3f}")
        print(f"   平均碎片化比例: {avg_frag:.3f}")
        print(f"   最大内存段数: {max_segments}")
        
        # 严重程度评估
        if max_frag > 0.4:
            severity = "🔴 严重"
            recommendations = [
                "立即增加max_cached_layers到6-8",
                "使用内存池预分配",
                "考虑权重量化减少内存使用"
            ]
        elif max_frag > 0.3:
            severity = "🟡 中等"
            recommendations = [
                "增加max_cached_layers到5-6",
                "定期调用torch.cuda.empty_cache()"
            ]
        elif max_frag > 0.15:
            severity = "🟢 轻微"
            recommendations = [
                "当前配置可接受",
                "可以考虑小幅优化"
            ]
        else:
            severity = "✅ 良好"
            recommendations = [
                "内存使用效率很高",
                "无需调整"
            ]
        
        print(f"   严重程度: {severity}")
        print(f"   建议:")
        for rec in recommendations:
            print(f"     - {rec}")
    
    # 清理
    del tensors
    gc.collect()
    torch.cuda.empty_cache()

def monitor_model_fragmentation():
    """监控模型推理过程中的碎片化（如果有模型的话）"""
    print(f"\n🤖 监控模型碎片化（如果启用权重流式传输）")
    
    try:
        # 检查是否可以导入
        from llama3.generator import LLaMA
        from llama3.weight_streaming_manager import WeightStreamingManager
        
        print("✅ 可以创建带碎片化监控的WeightStreamingManager:")
        print("   在generator中添加monitor_fragmentation=True参数")
        print("   例如:")
        print("   streaming_config = {")
        print("       'monitor_fragmentation': True,")
        print("       'max_cached_layers': 4,")
        print("       'prefetch_distance': 2")
        print("   }")
        
    except ImportError as e:
        print(f"❌ 无法导入模型组件: {e}")
        print("   可以使用test_memory_fragmentation.py进行详细测试")

def main():
    """主函数"""
    quick_fragmentation_test()
    monitor_model_fragmentation()
    
    print(f"\n💡 其他检测方法:")
    print(f"1. 使用nvidia-smi监控GPU内存使用")
    print(f"2. 运行test_memory_fragmentation.py进行详细分析")
    print(f"3. 在权重流式传输中启用monitor_fragmentation=True")
    print(f"4. 使用torch.profiler.profile()进行内存分析")

if __name__ == "__main__":
    main()