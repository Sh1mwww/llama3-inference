#!/bin/bash

# 参数设定
MODEL_PATH=${1:-"/home/roger/.llama/checkpoints/Llama3.2-1B"}
DEVICE=${2:-"cuda"}
PROMPT_FILE=${3:-"prompts/alpaca.txt"}
MAX_GEN_LEN=${4:-64}

# 读取 prompt 文件内容（确保支持多行）
if [ ! -f "$PROMPT_FILE" ]; then
  echo "[ERROR] Prompt file not found: $PROMPT_FILE"
  exit 1
fi

# 将 prompt 文件的每一行转成参数（支持多条 prompt）
PROMPTS=()
while IFS= read -r line || [ -n "$line" ]; do
  PROMPTS+=("$line")
done < "$PROMPT_FILE"

# 构造 prompt 参数列表
PROMPT_ARGS=""
for p in "${PROMPTS[@]}"; do
  PROMPT_ARGS+="\"$p\" "
done

# 执行推理
eval python scripts/run_inference.py \
  --model-path "$MODEL_PATH" \
  --device "$DEVICE" \
  --prompt $PROMPT_ARGS \
  --max-gen-len "$MAX_GEN_LEN"
