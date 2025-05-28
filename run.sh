#!/bin/bash

# ---------- 参数设定 ----------
MODEL_PATH=${1:-"/home/roger/.llama/checkpoints/Llama3.2-1B"}
DEVICE=${2:-"cuda"}
PROMPT_DIR="./prompts"
MAX_GEN_LEN=${3:-64}

# ---------- 检查 prompts 目录 ----------
if [ ! -d "$PROMPT_DIR" ]; then
  echo "[ERROR] Prompt directory not found: $PROMPT_DIR"
  exit 1
fi

# ---------- 显示选单 ----------
echo "请选择要使用的 prompt 文件："
PS3="请输入对应数字编号："

# 建立文件列表数组
prompt_files=($(ls "$PROMPT_DIR"/*.txt))
select file in "${prompt_files[@]}"; do
  if [ -n "$file" ]; then
    PROMPT_FILE="$file"
    echo "select: $PROMPT_FILE"
    break
  else
    echo "无效的选择，请重试。"
  fi
done

# ---------- 读取 prompt 文件内容 ----------
PROMPTS=()
while IFS= read -r line || [ -n "$line" ]; do
  PROMPTS+=("$line")
done < "$PROMPT_FILE"

PROMPT_ARGS=""
for p in "${PROMPTS[@]}"; do
  PROMPT_ARGS+="\"$p\" "
done

# ---------- 执行推理 ----------
eval python scripts/run_inference.py \
  --model-path "$MODEL_PATH" \
  --device "$DEVICE" \
  --prompt $PROMPT_ARGS \
  --max-gen-len "$MAX_GEN_LEN"
