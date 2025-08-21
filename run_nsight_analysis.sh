#!/bin/bash

# Nsight Systems 权重流式传输分析脚本
# 作者: Claude Code Assistant
# 功能: 分析IO和compute重叠效果

set -e

echo "🚀 启动 Nsight Systems 性能分析..."
echo "=========================================="

# 检查nsys是否安装
if ! command -v nsys &> /dev/null; then
    echo "❌ nsys (Nsight Systems) 未安装"
    echo "请安装: sudo apt install nsight-systems-cli"
    echo "或从 https://developer.nvidia.com/nsight-systems 下载"
    exit 1
fi

# 检查CUDA设备
if ! nvidia-smi &> /dev/null; then
    echo "❌ 未检测到NVIDIA GPU或驱动"
    exit 1
fi

echo "✅ 环境检查通过"

# 设置输出目录
OUTPUT_DIR="./nsight_analysis"
mkdir -p $OUTPUT_DIR

# 基础分析配置
BASIC_TRACE="cuda,nvtx,osrt"
ADVANCED_TRACE="cuda,nvtx,osrt,cublas,cudnn"

echo "📊 开始基础性能分析..."

# 1. 基础分析 - 快速概览
nsys profile \
    --trace=$BASIC_TRACE \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --output=$OUTPUT_DIR/basic_analysis \
    python test_nsight_profiling.py

echo "✅ 基础分析完成"

echo "🔬 开始详细性能分析..."

# 2. 详细分析 - 完整追踪
nsys profile \
    --trace=$ADVANCED_TRACE \
    --cuda-memory-usage=true \
    --cuda-um-cpu-page-faults=true \
    --cuda-um-gpu-page-faults=true \
    --sample=cpu \
    --cpuctxsw=true \
    --force-overwrite=true \
    --output=$OUTPUT_DIR/detailed_analysis \
    python test_nsight_profiling.py

echo "✅ 详细分析完成"

# 3. 生成报告
echo "📈 生成分析报告..."

echo "基础统计信息:" > $OUTPUT_DIR/analysis_report.txt
echo "===================" >> $OUTPUT_DIR/analysis_report.txt

nsys stats --report gputrace,cudaapisum $OUTPUT_DIR/basic_analysis.nsys-rep >> $OUTPUT_DIR/analysis_report.txt

echo "" >> $OUTPUT_DIR/analysis_report.txt
echo "内存传输统计:" >> $OUTPUT_DIR/analysis_report.txt
echo "===================" >> $OUTPUT_DIR/analysis_report.txt

nsys stats --report memop $OUTPUT_DIR/basic_analysis.nsys-rep >> $OUTPUT_DIR/analysis_report.txt

echo "" >> $OUTPUT_DIR/analysis_report.txt
echo "NVTX Range统计:" >> $OUTPUT_DIR/analysis_report.txt
echo "===================" >> $OUTPUT_DIR/analysis_report.txt

nsys stats --report nvtxsum $OUTPUT_DIR/basic_analysis.nsys-rep >> $OUTPUT_DIR/analysis_report.txt

echo "✅ 分析报告生成完成"

# 4. 输出结果信息
echo ""
echo "🎉 Nsight Systems 分析完成！"
echo "=========================================="
echo "📁 分析文件位置: $OUTPUT_DIR/"
echo ""
echo "📊 主要输出文件:"
echo "   • basic_analysis.nsys-rep    - 基础分析结果"
echo "   • detailed_analysis.nsys-rep - 详细分析结果"  
echo "   • analysis_report.txt        - 文本报告"
echo ""
echo "🔍 查看结果的方法:"
echo "   1. GUI界面:   nsys-ui $OUTPUT_DIR/basic_analysis.nsys-rep"
echo "   2. 命令行:    cat $OUTPUT_DIR/analysis_report.txt"
echo "   3. 网页版:    nsys-ui --port 8080 (然后上传.nsys-rep文件)"
echo ""
echo "📋 关键分析要点:"
echo "   • 查看Timeline中的NVTX ranges"
echo "   • 检查不同CUDA streams的重叠情况"
echo "   • 关注weight_h2d stream的活动"
echo "   • 分析GPU利用率和内存带宽"
echo "   • 查看prefetch和ensure操作的时间分布"
echo ""
echo "🎯 重叠效率评估:"
echo "   • 理想重叠率 > 60%"
echo "   • GPU空闲时间 < 15%"
echo "   • 传输等待时间 < 10%"

# 如果有X11显示，自动打开GUI
if [ -n "$DISPLAY" ]; then
    echo ""
    echo "🖥️  正在启动Nsight Systems GUI..."
    nsys-ui $OUTPUT_DIR/basic_analysis.nsys-rep &
fi