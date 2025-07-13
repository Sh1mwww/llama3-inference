#!/bin/bash
# GSP监控器启动脚本

cd "$(dirname "$0")"

echo "🛡️ Starting GSP Monitor..."

# 检查是否已经在运行
if pgrep -f "gsp_monitor_optimized.py" > /dev/null; then
    echo "⚠️ GSP Monitor is already running!"
    echo "PID: $(pgrep -f gsp_monitor_optimized.py)"
    exit 1
fi

# 检查Python和依赖
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found!"
    exit 1
fi

if ! python3 -c "import torch" 2>/dev/null; then
    echo "❌ PyTorch not found!"
    exit 1
fi

# 启动选项
case "${1:-foreground}" in
    "background"|"bg")
        echo "🚀 Starting GSP Monitor in background..."
        nohup python3 gsp_monitor_optimized.py > /dev/null 2>&1 &
        PID=$!
        echo "✅ GSP Monitor started with PID: $PID"
        echo "📝 Check logs: tail -f gsp_monitor.log"
        echo "🛑 To stop: kill $PID"
        ;;
    "foreground"|"fg"|*)
        echo "🚀 Starting GSP Monitor in foreground..."
        echo "📝 Logs will be saved to gsp_monitor.log"
        echo "🛑 Press Ctrl+C to stop"
        python3 gsp_monitor_optimized.py
        ;;
esac