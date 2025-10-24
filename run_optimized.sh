#!/bin/bash
# GPU 优化配置 - 快速启动脚本
# 用法: ./run_optimized.sh [profile]
# profile 可选: balanced (默认), aggressive, conservative

PROFILE="${1:-balanced}"

echo "========================================="
echo "LLaMA3 推理 - GPU 优化配置"
echo "========================================="
echo "配置档案: $PROFILE"
echo ""

case "$PROFILE" in
  "aggressive")
    echo "🚀 激进模式 - 最大性能（需要更多内存）"
    echo ""
    # GPU 端
    export WSM_AGGRESSIVE_GPU_PREFETCH=5
    export WSM_MAX_CACHED_LAYERS=8
    export WSM_GPU_FREE_GUARD_MB=1024

    # CPU 端
    export WSM_CPU_CACHE_CAP_LAYERS=60
    export WSM_CPU_CACHE_HWM_LAYERS=65
    export WSM_CPU_PREFETCH_DISTANCE=60

    # SSD 端
    export WSM_STAGING_MB=256
    ;;

  "conservative")
    echo "🛡️  保守模式 - 节省内存"
    echo ""
    # GPU 端
    export WSM_AGGRESSIVE_GPU_PREFETCH=2
    export WSM_MAX_CACHED_LAYERS=4
    export WSM_GPU_FREE_GUARD_MB=2048

    # CPU 端
    export WSM_CPU_CACHE_CAP_LAYERS=40
    export WSM_CPU_CACHE_HWM_LAYERS=45
    export WSM_CPU_PREFETCH_DISTANCE=40

    # SSD 端
    export WSM_STAGING_MB=64
    ;;

  "balanced"|*)
    echo "⚖️  平衡模式 - 推荐配置（RTX 5080 16GB）"
    echo ""
    # GPU 端
    export WSM_AGGRESSIVE_GPU_PREFETCH=3
    export WSM_MAX_CACHED_LAYERS=6
    export WSM_GPU_FREE_GUARD_MB=1024

    # CPU 端
    export WSM_CPU_CACHE_CAP_LAYERS=50
    export WSM_CPU_CACHE_HWM_LAYERS=55
    export WSM_CPU_PREFETCH_DISTANCE=50

    # SSD 端
    export WSM_STAGING_MB=128
    ;;
esac

echo "配置详情:"
echo "  GPU 激进预取: $WSM_AGGRESSIVE_GPU_PREFETCH 层"
echo "  GPU Cache 容量: $WSM_MAX_CACHED_LAYERS 层"
echo "  CPU Cache 容量: $WSM_CPU_CACHE_CAP_LAYERS-$WSM_CPU_CACHE_HWM_LAYERS 层"
echo "  CPU 预取窗口: $WSM_CPU_PREFETCH_DISTANCE 层"
echo "  SSD Staging: $WSM_STAGING_MB MB"
echo ""

# 可选：启用详细日志
read -p "启用详细日志？(y/n, 默认 n): " ENABLE_VERBOSE
if [[ "$ENABLE_VERBOSE" == "y" || "$ENABLE_VERBOSE" == "Y" ]]; then
  export WSM_VERBOSE=1
  echo "✅ 已启用详细日志"
else
  echo "ℹ️  详细日志已禁用"
fi

echo ""
echo "========================================="
echo "开始推理..."
echo "========================================="
echo ""
echo "💡 提示："
echo "  - 打开另一个终端运行 'nvitop -m full' 监控 GPU"
echo "  - 预期 GPU 利用率: 60-80%"
echo "  - 预期 GPU 频率: 1500MHz+"
echo "  - 预期功耗: 200W+"
echo ""
echo "按 Ctrl+C 可以随时终止"
echo ""

# 运行推理
python test_70b_prefill_ssd.py

echo ""
echo "========================================="
echo "推理完成"
echo "========================================="
