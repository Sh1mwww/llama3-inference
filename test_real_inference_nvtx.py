#!/usr/bin/env python3
"""
真实inference的NVTX性能分析脚本
使用实际的LLaMA模型进行推理，并分析权重流式传输的性能

使用方法:
nsys profile --trace=cuda,nvtx --output=real_inference python test_real_inference_nvtx.py
"""

import torch
import time
import os
import sys
from pathlib import Path
import argparse

# NVTX profiling support
try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
    print("✅ NVTX可用")
except ImportError:
    print("❌ NVTX不可用，使用fallback")
    # Fallback no-op functions
    class nvtx:
        @staticmethod
        def range_push(name): pass
        @staticmethod
        def range_pop(): pass
    NVTX_AVAILABLE = False

def get_model_path():
    """获取模型路径"""
    # 可能的模型路径
    possible_paths = [
        "/home/roger/llama3_project/checkpoints",
        "./checkpoints",
        "../checkpoints",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # 检查是否包含必要的文件
            path_obj = Path(path)
            if (path_obj / "params.json").exists():
                return str(path_obj)
    
    return None

def run_real_inference_with_nvtx(model_path, prompts, enable_streaming=True):
    """运行带NVTX标记的真实inference"""
    
    nvtx.range_push("real_inference_session")
    
    print(f"🎯 开始真实inference分析")
    print(f"📁 模型路径: {model_path}")
    print(f"🔧 权重流式传输: {'启用' if enable_streaming else '禁用'}")
    print(f"📝 Prompt数量: {len(prompts)}")
    
    # 1. 模型加载阶段
    nvtx.range_push("model_loading")
    print("🔄 正在加载模型...")
    
    try:
        from llama3.generator import LLaMA
        
        nvtx.range_push("llama_build")
        llama = LLaMA.build(
            model_path,
            load_model=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            enable_weight_streaming=enable_streaming,
            streaming_config={
                'prefetch_distance': 2,
                'max_cached_layers': 4,
                'warmup_layers': 1,
                'verbose': True
            } if enable_streaming else None,
            max_seq_len=1024,  # 较小的序列长度用于测试
            max_batch_size=8
        )
        nvtx.range_pop()  # llama_build
        
        print("✅ 模型加载完成")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        nvtx.range_pop()  # model_loading
        nvtx.range_pop()  # real_inference_session
        return None
    
    nvtx.range_pop()  # model_loading
    
    # 2. 推理阶段
    nvtx.range_push("inference_execution")
    
    results = []
    
    for i, prompt in enumerate(prompts):
        nvtx.range_push(f"prompt_{i}_inference")
        
        print(f"🔄 处理Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        try:
            nvtx.range_push(f"prompt_{i}_text_completion")
            
            start_time = time.time()
            _, completions = llama.text_completion(
                [prompt],
                max_gen_len=32,  # 较短的生成长度用于快速测试
                temperature=0.6,
                top_p=0.9
            )
            inference_time = time.time() - start_time
            
            nvtx.range_pop()  # text_completion
            
            completion = completions[0] if completions else ""
            results.append({
                'prompt': prompt,
                'completion': completion,
                'time': inference_time
            })
            
            print(f"✅ Prompt {i+1} 完成 ({inference_time:.2f}s)")
            print(f"   结果: {completion[:100]}...")
            
        except Exception as e:
            print(f"❌ Prompt {i+1} 推理失败: {e}")
            results.append({
                'prompt': prompt,
                'completion': f"ERROR: {e}",
                'time': 0.0
            })
        
        nvtx.range_pop()  # prompt_i_inference
    
    nvtx.range_pop()  # inference_execution
    nvtx.range_pop()  # real_inference_session
    
    return results

def analyze_streaming_performance():
    """分析权重流式传输性能"""
    
    nvtx.range_push("streaming_performance_analysis")
    
    print("\n🔍 权重流式传输性能分析")
    print("=" * 50)
    
    # 测试Prompt列表
    test_prompts = [
        "你好，请介绍一下自己。",
        "什么是人工智能？",
        "请解释一下深度学习的基本原理。",
        "如何优化神经网络的性能？"
    ]
    
    model_path = get_model_path()
    if not model_path:
        print("❌ 找不到模型文件，请确保模型路径正确")
        nvtx.range_pop()  # streaming_performance_analysis
        return
    
    # 运行带权重流式传输的inference
    nvtx.range_push("streaming_enabled_test")
    print("\n🔧 测试1: 启用权重流式传输")
    streaming_results = run_real_inference_with_nvtx(
        model_path, test_prompts, enable_streaming=True
    )
    nvtx.range_pop()  # streaming_enabled_test
    
    if streaming_results:
        total_time = sum(r['time'] for r in streaming_results)
        print(f"✅ 权重流式传输测试完成，总时间: {total_time:.2f}s")
        
        # 显示结果
        for i, result in enumerate(streaming_results):
            print(f"\n📝 Prompt {i+1}: {result['prompt'][:50]}...")
            print(f"🤖 回答: {result['completion'][:100]}...")
            print(f"⏱️  时间: {result['time']:.2f}s")
    
    nvtx.range_pop()  # streaming_performance_analysis

def simulate_layer_processing():
    """模拟实际的层处理过程（用于无模型的测试）"""
    
    nvtx.range_push("layer_processing_simulation")
    
    print("\n🎭 模拟层处理过程")
    print("=" * 30)
    
    num_layers = 8
    seq_len = 64
    
    for token_pos in range(seq_len):
        nvtx.range_push(f"token_{token_pos}_processing")
        
        for layer_id in range(num_layers):
            nvtx.range_push(f"layer_{layer_id}_token_{token_pos}")
            
            # 模拟权重确保在GPU
            nvtx.range_push(f"ensure_layer_{layer_id}_weights")
            time.sleep(0.001)  # 模拟权重加载时间
            nvtx.range_pop()  # ensure_weights
            
            # 模拟实际计算
            nvtx.range_push(f"layer_{layer_id}_computation")
            if torch.cuda.is_available():
                # 实际的CUDA操作
                a = torch.randn(512, 512, device="cuda")
                b = torch.randn(512, 512, device="cuda") 
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
            else:
                time.sleep(0.002)
            nvtx.range_pop()  # computation
            
            nvtx.range_pop()  # layer_processing
        
        nvtx.range_pop()  # token_processing
        
        # 每隔几个token打印进度
        if token_pos % 10 == 0:
            print(f"🔄 处理token {token_pos}/{seq_len}")
    
    nvtx.range_pop()  # layer_processing_simulation

def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description="真实inference的NVTX性能分析")
    parser.add_argument("--mode", choices=["full", "simulation"], default="simulation",
                       help="运行模式: full(完整模型) 或 simulation(模拟)")
    parser.add_argument("--model-path", help="模型路径（覆盖自动检测）")
    
    args = parser.parse_args()
    
    print("🎯 真实inference的NVTX性能分析")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print(f"🔧 使用GPU: {torch.cuda.get_device_name()}")
    else:
        print("🔧 使用CPU模拟")
    
    if args.mode == "full":
        # 完整模型测试
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = get_model_path()
        
        if model_path:
            analyze_streaming_performance()
        else:
            print("❌ 找不到模型，切换到模拟模式")
            simulate_layer_processing()
    else:
        # 模拟模式
        simulate_layer_processing()
    
    print("\n✅ 分析完成!")
    print("\n📊 使用以下命令查看结果:")
    print("nsys profile --trace=cuda,nvtx --output=real_inference python test_real_inference_nvtx.py")
    print("nsys stats --report nvtxsum real_inference.nsys-rep")
    print("python analyze_nsys_report.py real_inference.nsys-rep")

if __name__ == "__main__":
    main()