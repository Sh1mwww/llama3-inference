#!/usr/bin/env python3
"""
Nsight Systems 性能分析测试脚本
用于测试IO和compute重叠效果

使用方法:
nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
    --output=weight_streaming_analysis python test_nsight_profiling.py
"""

import torch
import torch.cuda.nvtx as nvtx
import time
import sys
import os

# 添加项目路径
sys.path.append('/home/roger/llama3_project')

from llama3.generator import LlamaGenerator

def test_io_compute_overlap():
    """测试权重流式传输的IO和计算重叠效果"""
    
    nvtx.range_push("model_initialization")
    print("🚀 开始初始化模型...")
    
    # 检查checkpoint路径
    ckpt_dir = "/home/roger/llama3_project/checkpoints"  # 请根据实际路径修改
    if not os.path.exists(ckpt_dir):
        print(f"❌ Checkpoint目录不存在: {ckpt_dir}")
        print("请修改ckpt_dir为实际的checkpoint路径")
        return
    
    try:
        generator = LlamaGenerator.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=f"{ckpt_dir}/tokenizer.model",
            max_seq_len=1024,
            max_batch_size=1,
            device="cuda:0",
            enable_weight_streaming=True,
            streaming_config={
                "prefetch_distance": 2,
                "max_cached_layers": 4,
                "verbose": True
            }
        )
        print("✅ 模型初始化完成")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        print("请检查checkpoint路径和文件")
        nvtx.range_pop()  # model_initialization
        return
    
    nvtx.range_pop()  # model_initialization
    
    # 测试用例
    test_cases = [
        {
            "name": "短文本生成",
            "prompt": "The future of artificial intelligence is",
            "max_gen_len": 50
        },
        {
            "name": "技术解释",
            "prompt": "Machine learning is a subset of artificial intelligence that",
            "max_gen_len": 80
        },
        {
            "name": "创意写作",
            "prompt": "In a world where robots and humans coexist,",
            "max_gen_len": 100
        }
    ]
    
    print(f"\n🧪 开始执行 {len(test_cases)} 个测试用例...")
    
    for i, test_case in enumerate(test_cases):
        nvtx.range_push(f"test_case_{i}_{test_case['name']}")
        print(f"\n📝 测试 {i+1}: {test_case['name']}")
        print(f"   提示词: {test_case['prompt']}")
        
        try:
            nvtx.range_push(f"generation_{i}")
            start_time = time.time()
            
            results = generator.text_completion(
                prompts=[test_case['prompt']],
                max_gen_len=test_case['max_gen_len'],
                temperature=0.7,
                top_p=0.9
            )
            
            end_time = time.time()
            nvtx.range_pop()  # generation
            
            # 输出结果
            generation_time = end_time - start_time
            result_text = results[0] if results else "生成失败"
            
            print(f"   ⏱️  生成时间: {generation_time:.2f}秒")
            print(f"   📄 生成结果: {result_text[:100]}...")
            
            # 添加延迟确保流程清晰可见
            nvtx.range_push(f"cooldown_{i}")
            time.sleep(0.5)
            nvtx.range_pop()  # cooldown
            
        except Exception as e:
            print(f"   ❌ 测试失败: {e}")
        
        nvtx.range_pop()  # test_case
    
    print("\n✅ 所有测试完成!")

def test_baseline_comparison():
    """对比测试：带权重流式传输 vs 不带权重流式传输"""
    
    print("\n🔬 开始基准对比测试...")
    test_prompt = "Artificial intelligence will transform"
    
    # 测试1: 不启用权重流式传输
    nvtx.range_push("baseline_test")
    print("📊 基准测试 (无权重流式传输)")
    
    try:
        generator_baseline = LlamaGenerator.build(
            ckpt_dir="/home/roger/llama3_project/checkpoints",
            tokenizer_path="/home/roger/llama3_project/checkpoints/tokenizer.model",
            max_seq_len=512,
            max_batch_size=1,
            device="cuda:0",
            enable_weight_streaming=False  # 关闭权重流式传输
        )
        
        start_time = time.time()
        results_baseline = generator_baseline.text_completion(
            prompts=[test_prompt],
            max_gen_len=50,
            temperature=0.7
        )
        baseline_time = time.time() - start_time
        
        print(f"   ⏱️  基准测试时间: {baseline_time:.2f}秒")
        
    except Exception as e:
        print(f"   ❌ 基准测试失败: {e}")
        baseline_time = 0
    
    nvtx.range_pop()  # baseline_test
    
    # 测试2: 启用权重流式传输
    nvtx.range_push("streaming_test")
    print("🚀 流式传输测试 (启用权重流式传输)")
    
    try:
        generator_streaming = LlamaGenerator.build(
            ckpt_dir="/home/roger/llama3_project/checkpoints",
            tokenizer_path="/home/roger/llama3_project/checkpoints/tokenizer.model",
            max_seq_len=512,
            max_batch_size=1,
            device="cuda:0",
            enable_weight_streaming=True,
            streaming_config={
                "prefetch_distance": 2,
                "max_cached_layers": 4,
                "verbose": True
            }
        )
        
        start_time = time.time()
        results_streaming = generator_streaming.text_completion(
            prompts=[test_prompt],
            max_gen_len=50,
            temperature=0.7
        )
        streaming_time = time.time() - start_time
        
        print(f"   ⏱️  流式传输时间: {streaming_time:.2f}秒")
        
        # 计算性能改进
        if baseline_time > 0:
            improvement = ((baseline_time - streaming_time) / baseline_time) * 100
            print(f"   📈 性能提升: {improvement:.1f}%")
        
    except Exception as e:
        print(f"   ❌ 流式传输测试失败: {e}")
    
    nvtx.range_pop()  # streaming_test

def main():
    """主测试函数"""
    nvtx.range_push("main_test_suite")
    
    print("🎯 Nsight Systems 性能分析测试")
    print("=" * 50)
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法执行测试")
        return
    
    print(f"🔧 CUDA设备: {torch.cuda.get_device_name()}")
    print(f"🔧 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 执行主要测试
    test_io_compute_overlap()
    
    # 执行对比测试（可选）
    # test_baseline_comparison()
    
    nvtx.range_pop()  # main_test_suite
    print("\n🎉 测试套件完成！")
    print("\n📊 Nsight分析建议:")
    print("1. 查看Timeline中的NVTX ranges")
    print("2. 检查weight_h2d stream与default stream的重叠")
    print("3. 关注prefetch操作的时机")
    print("4. 分析GPU利用率和内存带宽")

if __name__ == "__main__":
    main()