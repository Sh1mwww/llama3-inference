#!/usr/bin/env bash
# ------------------------------------------------------------
# run.sh  --  multi-batch wrapper for scripts/run_inference.py
#
# 用法：
#   ./run.sh [PROMPT_FILE] [MODEL_PATH] [BATCH_SIZE] [MAX_LEN] [DEVICE]
#
#   PROMPT_FILE : (必填) 纯文本，每行一条 prompt。
#   MODEL_PATH  : (可选) 模型目录，默认为 $HOME/.llama/checkpoints/Llama3.2-3B
#   BATCH_SIZE  : (可选) 每批条数，默认 32
#   MAX_LEN     : (可选) 生成长度，默认 64
#   DEVICE      : (可选) cuda / cpu，默认 cuda（若可用）
# ------------------------------------------------------------

set -e

PROMPT_FILE=${1:? "❌ 必须提供 prompt 文件路径"}
MODEL_PATH=${2:-"$HOME/.llama/checkpoints/Llama3.2-3B"}
BATCH_SIZE=${3:-4}  # 降低預設批次大小
MAX_LEN=${4:-64}
DEVICE=${5:-"cuda"}

if [ ! -f "$PROMPT_FILE" ]; then
  echo "[ERROR] prompt file not found: $PROMPT_FILE"
  exit 1
fi

echo "▶ MODEL       = $MODEL_PATH"
echo "▶ DEVICE      = $DEVICE"
echo "▶ PROMPT_FILE = $PROMPT_FILE"
echo "▶ BATCH_SIZE  = $BATCH_SIZE"
echo "▶ MAX_LEN     = $MAX_LEN"
echo "------------------------------------------------------------"

python scripts/run_inference.py \
  --model-path "$MODEL_PATH" \
  --device "$DEVICE" \
  --prompt-file "$PROMPT_FILE" \
  --batch-size "$BATCH_SIZE" \
  --max-gen-len "$MAX_LEN"
