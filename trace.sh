#!/usr/bin/env bash
# ---------------------------------------------
# run_trace.sh  —  trace Llama-3 推理并产出 trace.json / trace_summary.csv
#
# 用法:
#   ./run_trace.sh [prompt_or_file] [max_len] [model_dir] [cuda_id]
#
# 例:
#   ./run_trace.sh "Why is the sky blue?" 128
#   ./run_trace.sh prompts/chat.txt 64 /data/llama3/3B 0
# ---------------------------------------------

# ---------- 参数 ----------
PROMPT_OR_FILE=${1:-"Hello, world!"}
MAX_LEN=${2:-64}
MODEL_DIR=${3:-"/home/roger/.llama/checkpoints/Llama3.2-3B"}
CUDA_ID=${4:-0}

# ---------- 处理 prompt ----------
if [[ -f "$PROMPT_OR_FILE" ]]; then
  PROMPT=$(cat "$PROMPT_OR_FILE")
else
  PROMPT="$PROMPT_OR_FILE"
fi

# ---------- 环境 ----------
export CUDA_VISIBLE_DEVICES=$CUDA_ID

# ---------- 运行 ----------
python scripts/trace_kv_weight.py \
  --model-path "$MODEL_DIR" \
  --device cuda \
  --prompt "$PROMPT" \
  --max-gen-len "$MAX_LEN"

echo -e "\n✅ trace 完成：trace.json · trace_summary.csv 已在当前目录"
