#!/usr/bin/env python3
"""
命令行分析Nsight Systems报告
当GUI不可用时使用此脚本分析.nsys-rep文件
"""

import subprocess
import sys
import os
import json
import re

def run_nsys_stats(rep_file, report_type):
    """运行nsys stats命令并返回结果"""
    try:
        cmd = ["nsys", "stats", "--report", report_type, rep_file]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ 执行nsys stats失败: {e}")
        return None
    except FileNotFoundError:
        print("❌ nsys命令未找到，请确保Nsight Systems已安装")
        return None

def analyze_gpu_utilization(rep_file):
    """分析GPU利用率"""
    print("📊 GPU利用率分析")
    print("=" * 50)
    
    output = run_nsys_stats(rep_file, "gpustats")
    if output:
        print(output)
    else:
        print("无法获取GPU统计信息")

def analyze_cuda_api(rep_file):
    """分析CUDA API调用"""
    print("\n🔧 CUDA API调用分析")
    print("=" * 50)
    
    output = run_nsys_stats(rep_file, "cudaapisum")
    if output:
        # 提取关键信息
        lines = output.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['stream', 'event', 'memory', 'kernel']):
                print(line)

def analyze_nvtx_ranges(rep_file):
    """分析NVTX范围"""
    print("\n🏷️  NVTX标记分析")
    print("=" * 50)
    
    output = run_nsys_stats(rep_file, "nvtxsum")
    if output:
        lines = output.split('\n')
        
        # 查找权重相关操作
        weight_ops = []
        layer_ops = []
        prefetch_ops = []
        
        for line in lines:
            if 'ensure_layer' in line or 'h2d_transfer' in line:
                weight_ops.append(line.strip())
            elif 'layer_' in line and ('attention' in line or 'ffn' in line):
                layer_ops.append(line.strip())
            elif 'prefetch' in line:
                prefetch_ops.append(line.strip())
        
        if weight_ops:
            print("\n🔄 权重加载操作:")
            for op in weight_ops[:10]:  # 显示前10个
                print(f"  {op}")
        
        if layer_ops:
            print("\n🧠 层计算操作:")
            for op in layer_ops[:10]:
                print(f"  {op}")
        
        if prefetch_ops:
            print("\n⚡ 预取操作:")
            for op in prefetch_ops[:10]:
                print(f"  {op}")
        
        print(f"\n📈 NVTX完整输出:")
        print(output)

def analyze_memory_operations(rep_file):
    """分析内存操作"""
    print("\n💾 内存操作分析")
    print("=" * 50)
    
    output = run_nsys_stats(rep_file, "memop")
    if output:
        lines = output.split('\n')
        h2d_ops = []
        d2h_ops = []
        
        for line in lines:
            if 'HtoD' in line or 'Host to Device' in line:
                h2d_ops.append(line.strip())
            elif 'DtoH' in line or 'Device to Host' in line:
                d2h_ops.append(line.strip())
        
        if h2d_ops:
            print("📤 Host to Device 传输:")
            for op in h2d_ops[:5]:
                print(f"  {op}")
        
        if d2h_ops:
            print("📥 Device to Host 传输:")
            for op in d2h_ops[:5]:
                print(f"  {op}")

def analyze_kernels(rep_file):
    """分析GPU kernel"""
    print("\n⚙️  GPU Kernel分析")
    print("=" * 50)
    
    output = run_nsys_stats(rep_file, "gpukernsum")
    if output:
        lines = output.split('\n')
        
        # 查找矩阵乘法相关kernel
        matmul_kernels = []
        attention_kernels = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['gemm', 'matmul', 'sgemm']):
                matmul_kernels.append(line.strip())
            elif any(keyword in line.lower() for keyword in ['attention', 'softmax']):
                attention_kernels.append(line.strip())
        
        if matmul_kernels:
            print("🔢 矩阵乘法Kernel:")
            for kernel in matmul_kernels[:5]:
                print(f"  {kernel}")
        
        if attention_kernels:
            print("🎯 注意力相关Kernel:")
            for kernel in attention_kernels[:5]:
                print(f"  {kernel}")

def generate_summary_report(rep_file):
    """生成总结报告"""
    print("\n📋 总结报告")
    print("=" * 50)
    
    summary = run_nsys_stats(rep_file, "summary")
    if summary:
        print(summary)

def calculate_overlap_efficiency(rep_file):
    """计算重叠效率（简化版）"""
    print("\n⚡ IO/计算重叠分析")
    print("=" * 50)
    
    # 这是一个简化的分析，实际的重叠计算需要更复杂的逻辑
    nvtx_output = run_nsys_stats(rep_file, "nvtxsum")
    if nvtx_output:
        lines = nvtx_output.split('\n')
        
        ensure_times = []
        prefetch_times = []
        compute_times = []
        
        for line in lines:
            # 简单的时间提取（这里需要根据实际输出格式调整）
            if 'ensure_layer' in line and 'Total Time' in line:
                try:
                    time_match = re.search(r'(\d+\.?\d*)\s*(us|ms|s)', line)
                    if time_match:
                        time_val = float(time_match.group(1))
                        unit = time_match.group(2)
                        if unit == 'ms':
                            time_val *= 1000
                        elif unit == 's':
                            time_val *= 1000000
                        ensure_times.append(time_val)
                except:
                    pass
        
        if ensure_times:
            total_ensure_time = sum(ensure_times)
            avg_ensure_time = total_ensure_time / len(ensure_times)
            print(f"🔄 平均权重加载时间: {avg_ensure_time:.2f} us")
            print(f"📊 权重加载操作数: {len(ensure_times)}")
            print(f"⏱️  总权重加载时间: {total_ensure_time:.2f} us")

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("使用方法: python analyze_nsys_report.py <file.nsys-rep>")
        sys.exit(1)
    
    rep_file = sys.argv[1]
    
    if not os.path.exists(rep_file):
        print(f"❌ 文件不存在: {rep_file}")
        sys.exit(1)
    
    print("🎯 Nsight Systems 报告分析")
    print(f"📁 分析文件: {rep_file}")
    print("=" * 70)
    
    # 执行各种分析
    analyze_gpu_utilization(rep_file)
    analyze_cuda_api(rep_file)
    analyze_nvtx_ranges(rep_file)
    analyze_memory_operations(rep_file)
    analyze_kernels(rep_file)
    calculate_overlap_efficiency(rep_file)
    generate_summary_report(rep_file)
    
    print("\n✅ 分析完成!")
    print("\n💡 优化建议:")
    print("1. 查看权重加载和层计算的时间分布")
    print("2. 检查prefetch操作是否与计算重叠")
    print("3. 关注GPU kernel的执行效率")
    print("4. 监控内存传输的带宽利用率")

if __name__ == "__main__":
    main()