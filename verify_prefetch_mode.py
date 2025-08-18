#!/usr/bin/env python3
"""
验证 Weight Streaming 和 Prefetch 模式的诊断工具
"""

import subprocess
import sys
import re
from typing import Dict, List, Tuple

def analyze_wsm_logs(stdout: str) -> Dict:
    """分析 WSM 日志输出"""
    
    # 统计不同类型的操作
    prefetch_count = len(re.findall(r'\[WSM\] prefetch layer=\d+', stdout))
    evict_count = len(re.findall(r'\[WSM\]\s+evict layer=\d+', stdout))
    gpu_load_count = len(re.findall(r'\[WSM\] ->GPU layer=\d+', stdout))
    warmup_count = len(re.findall(r'\[WSM\] warmup prefetch:', stdout))
    
    # 提取层操作序列
    layer_ops = []
    for match in re.finditer(r'\[WSM\] (prefetch|evict|->GPU) layer=(\d+)', stdout):
        op_type = match.group(1)
        layer_id = int(match.group(2))
        layer_ops.append((op_type, layer_id))
    
    # 分析内存使用
    memory_match = re.search(r'Peak GPU memory:\s+([0-9.]+)\s+([A-Z]+)', stdout)
    peak_memory = None
    if memory_match:
        value = float(memory_match.group(1))
        unit = memory_match.group(2)
        peak_memory = f"{value} {unit}"
    
    # 分析传输统计
    dram_gpu_ops = 0
    gpu_dram_ops = 0
    
    dram_gpu_match = re.search(r'DRAM → GPU:\s+Operations: (\d+)', stdout)
    if dram_gpu_match:
        dram_gpu_ops = int(dram_gpu_match.group(1))
        
    gpu_dram_match = re.search(r'GPU → DRAM:\s+Operations: (\d+)', stdout)
    if gpu_dram_match:
        gpu_dram_ops = int(gpu_dram_match.group(1))
    
    return {
        'prefetch_count': prefetch_count,
        'evict_count': evict_count,
        'gpu_load_count': gpu_load_count,
        'warmup_count': warmup_count,
        'layer_ops': layer_ops,
        'peak_memory': peak_memory,
        'dram_gpu_ops': dram_gpu_ops,
        'gpu_dram_ops': gpu_dram_ops,
        'total_transfer_ops': dram_gpu_ops + gpu_dram_ops
    }

def check_streaming_indicators(stdout: str) -> Dict[str, bool]:
    """检查权重流式传输的各种指标"""
    
    indicators = {}
    
    # 1. WSM 启用检查
    indicators['wsm_enabled'] = 'Weight streaming enabled' in stdout
    
    # 2. 检查是否有层权重在 CPU（应该有）
    indicators['layers_on_cpu'] = 'first block param device: cpu' in stdout
    
    # 3. 检查是否有 WSM 操作日志
    indicators['wsm_operations'] = '[WSM]' in stdout
    
    # 4. 检查是否有设备同步
    indicators['device_sync'] = 'Synchronizing layer norms to GPU' in stdout
    
    # 5. 检查是否有预热
    indicators['warmup_prefetch'] = 'warmup prefetch:' in stdout
    
    # 6. 检查错误
    indicators['no_device_errors'] = 'Expected all tensors to be on the same device' not in stdout
    indicators['no_method_errors'] = 'ensure_weights_cuda' not in stdout
    
    return indicators

def print_analysis_report(analysis: Dict, indicators: Dict[str, bool]):
    """打印分析报告"""
    
    print("🔍 Weight Streaming & Prefetch 模式验证报告")
    print("=" * 60)
    
    # 1. 基本指标
    print("\n📊 WSM 操作统计:")
    print(f"   Prefetch 操作: {analysis['prefetch_count']}")
    print(f"   Evict 操作: {analysis['evict_count']}")  
    print(f"   GPU 加载操作: {analysis['gpu_load_count']}")
    print(f"   预热操作: {analysis['warmup_count']}")
    
    # 2. 传输统计
    print(f"\n💾 内存传输统计:")
    print(f"   DRAM→GPU 操作: {analysis['dram_gpu_ops']}")
    print(f"   GPU→DRAM 操作: {analysis['gpu_dram_ops']}")
    print(f"   总传输操作: {analysis['total_transfer_ops']}")
    
    # 3. 内存使用
    if analysis['peak_memory']:
        print(f"\n🧠 GPU 内存使用:")
        print(f"   峰值内存: {analysis['peak_memory']}")
    
    # 4. 流式传输指标
    print(f"\n✅ 流式传输指标检查:")
    for key, value in indicators.items():
        status = "✅" if value else "❌"
        key_desc = {
            'wsm_enabled': 'WSM 已启用',
            'layers_on_cpu': '层权重保持在 CPU',
            'wsm_operations': 'WSM 操作日志存在',
            'device_sync': '设备同步执行',
            'warmup_prefetch': '预热预取执行',
            'no_device_errors': '无设备不匹配错误',
            'no_method_errors': '无方法缺失错误'
        }
        print(f"   {status} {key_desc.get(key, key)}: {value}")
    
    # 5. 总结判断
    print(f"\n🎯 **最终判断**:")
    
    is_prefetch_mode = (
        analysis['prefetch_count'] > 0 and 
        analysis['evict_count'] > 0 and
        analysis['total_transfer_ops'] > 50 and  # 足够多的传输操作
        indicators['wsm_enabled'] and
        indicators['layers_on_cpu']
    )
    
    if is_prefetch_mode:
        print("   ✅ **确认：正在使用 Prefetch 逐层加载模式**")
        print("   📋 证据:")
        print(f"      - WSM prefetch 操作: {analysis['prefetch_count']} 次")
        print(f"      - WSM evict 操作: {analysis['evict_count']} 次")
        print(f"      - 内存传输操作: {analysis['total_transfer_ops']} 次")
        print("      - 层权重保持在 CPU，按需加载到 GPU")
    else:
        print("   ❌ **未使用 Prefetch 逐层加载模式**")
        print("   可能原因:")
        if analysis['prefetch_count'] == 0:
            print("      - 没有检测到 prefetch 操作")
        if analysis['total_transfer_ops'] < 50:
            print("      - 内存传输操作太少，可能是全量加载")
        if not indicators['wsm_enabled']:
            print("      - WSM 未启用")

def main():
    """主函数"""
    
    print("🧪 运行 Weight Streaming 验证测试...")
    
    # 运行测试
    cmd = [
        'python3', '/home/roger/llama3_project/scripts/profile_pipeline.py',
        '--model-path', '/mnt/model/llama/checkpoints/Llama3.2-3B',
        '--prompt', 'Hello world test for WSM prefetch mode verification',
        '--max-gen-len', '5',  # 短一点，快速测试
        '--batch-size', '1',
        '--device', 'cuda',
        '--verbose'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        stdout = result.stdout
        stderr = result.stderr
        
        if result.returncode != 0:
            print(f"❌ 测试失败，返回码: {result.returncode}")
            print(f"错误输出: {stderr}")
            return False
        
        # 分析结果
        analysis = analyze_wsm_logs(stdout)
        indicators = check_streaming_indicators(stdout)
        
        # 打印报告
        print_analysis_report(analysis, indicators)
        
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ 测试超时")
        return False
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)