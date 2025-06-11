# ================================
# profile_llama3.sh
# ================================
# 交互式外壳脚本：一键运行 profiler 并打印结果
# 用法:
#   ./profile_llama3.sh "prompt text or /path/to/prompt.txt" 128
#   参数2 可选：生成长度，默认 64

#!/bin/bash

MODEL="/mnt/model/llama/checkpoints/Llama3.2-3B"
DEVICE="cuda"
LEN=${2:-64}

# 读取 prompt 内容：支持路径或直接文本
if [ -f "$1" ]; then
  PROMPT=$(cat "$1" | tr -d '\n' | sed 's/"/\\"/g')
else
  PROMPT=$(echo "$1" | tr -d '\n' | sed 's/"/\\"/g')
fi

echo "▶ MODEL = $MODEL"
echo "▶ DEVICE = $DEVICE"
echo "▶ PROMPT = \"$PROMPT\""
echo "▶ MAX_LEN = $LEN"
echo "-------------------------------"

python3 scripts/profile_pipeline.py \
  --model-path "$MODEL" \
  --device "$DEVICE" \
  --prompt "$PROMPT" \
  --max-gen-len "$LEN"
