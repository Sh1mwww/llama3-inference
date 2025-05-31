# ================================
# profile_llama3.sh
# ================================
# 交互式外壳脚本：一键运行 profiler 并打印结果
# 用法:
#   ./profile_llama3.sh "prompt text or /path/to/prompt.txt" 128
#   参数2 可选：生成长度，默认 64

#!/bin/bash

MODEL="/home/roger/.llama/checkpoints/Llama3.2-3B"
DEVICE="cuda"
LEN=${2:-64}

if [ -f "$1" ]; then
  PROMPT="$(cat "$1")"
else
  PROMPT="${1:-'Hello'}"
fi

python scripts/profile_pipeline.py \
  --model-path "$MODEL" \
  --device "$DEVICE" \
  --prompt "$PROMPT" \
  --max-gen-len $LEN