#!/usr/bin/env python3
"""
简化版NVTX测试脚本
用于快速验证NVTX标记是否正确工作

使用方法:
nsys profile --trace=nvtx --output=nvtx_test python test_nvtx_simple.py
"""

import torch
import torch.cuda.nvtx as nvtx
import time
import sys
import os

def test_nvtx_markers():
    """测试NVTX标记功能"""
    
    print("🧪 开始NVTX标记测试...")
    
    nvtx.range_push("test_suite")
    
    # 测试1: 基础标记
    nvtx.range_push("basic_operations")
    print("✅ 基础操作测试")
    
    for i in range(3):
        nvtx.range_push(f"operation_{i}")
        time.sleep(0.1)  # 模拟操作
        nvtx.range_pop()
    
    nvtx.range_pop()  # basic_operations
    
    # 测试2: CUDA操作标记
    if torch.cuda.is_available():
        nvtx.range_push("cuda_operations")
        print("✅ CUDA操作测试")
        
        # 创建张量
        nvtx.range_push("tensor_creation")
        a = torch.randn(1000, 1000, device="cuda")
        b = torch.randn(1000, 1000, device="cuda")
        nvtx.range_pop()  # tensor_creation
        
        # 矩阵乘法
        nvtx.range_push("matrix_multiplication")
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        nvtx.range_pop()  # matrix_multiplication
        
        # 内存传输
        nvtx.range_push("memory_transfer")
        c_cpu = c.cpu()
        nvtx.range_pop()  # memory_transfer
        
        nvtx.range_pop()  # cuda_operations
    else:
        print("⚠️  CUDA不可用，跳过CUDA测试")
    
    # 测试3: 嵌套标记
    nvtx.range_push("nested_operations")
    print("✅ 嵌套操作测试")
    
    for layer in range(2):
        nvtx.range_push(f"layer_{layer}")
        
        nvtx.range_push(f"layer_{layer}_attention")
        time.sleep(0.05)
        nvtx.range_pop()  # attention
        
        nvtx.range_push(f"layer_{layer}_ffn")
        time.sleep(0.05)
        nvtx.range_pop()  # ffn
        
        nvtx.range_pop()  # layer
    
    nvtx.range_pop()  # nested_operations
    
    nvtx.range_pop()  # test_suite
    
    print("✅ NVTX标记测试完成")

def test_weight_streaming_markers():
    """测试权重流式传输相关的标记"""
    
    print("🔄 开始权重流式传输标记测试...")
    
    nvtx.range_push("weight_streaming_simulation")
    
    # 模拟权重管理操作
    for layer_id in range(3):
        nvtx.range_push(f"ensure_layer_{layer_id}")
        
        # 模拟缓存检查
        nvtx.range_push(f"cache_miss_layer_{layer_id}")
        
        # 模拟H2D传输
        nvtx.range_push(f"h2d_transfer_layer_{layer_id}")
        
        for module in ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]:
            nvtx.range_push(f"h2d_{module}")
            time.sleep(0.01)  # 模拟传输时间
            nvtx.range_pop()  # h2d_module
        
        nvtx.range_pop()  # h2d_transfer
        
        # 模拟事件记录
        nvtx.range_push(f"record_event_layer_{layer_id}")
        time.sleep(0.001)
        nvtx.range_pop()  # record_event
        
        nvtx.range_pop()  # cache_miss
        nvtx.range_pop()  # ensure_layer
        
        # 模拟预取操作
        if layer_id < 2:
            next_layers = [layer_id + 1]
            nvtx.range_push(f"prefetch_layers_{next_layers}")
            
            nvtx.range_push(f"prefetch_layer_{layer_id + 1}")
            time.sleep(0.02)  # 模拟预取时间
            nvtx.range_pop()  # prefetch_layer
            
            nvtx.range_pop()  # prefetch_layers
    
    nvtx.range_pop()  # weight_streaming_simulation
    
    print("✅ 权重流式传输标记测试完成")

def main():
    """主测试函数"""
    
    print("🎯 NVTX功能验证测试")
    print("=" * 40)
    
    # 检查NVTX是否可用
    try:
        nvtx.range_push("test_nvtx_availability")
        nvtx.range_pop()
        print("✅ NVTX可用")
    except Exception as e:
        print(f"❌ NVTX不可用: {e}")
        return
    
    # 运行测试
    test_nvtx_markers()
    test_weight_streaming_markers()
    
    print("\n🎉 所有测试完成！")
    print("\n📊 使用以下命令查看结果:")
    print("nsys profile --trace=nvtx --output=nvtx_test python test_nvtx_simple.py")
    print("nsys-ui nvtx_test.nsys-rep")

if __name__ == "__main__":
    main()