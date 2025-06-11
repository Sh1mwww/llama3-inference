#!/usr/bin/env bash
# ------------------------------------------------
# run_trace.sh v2
# ⌁ 把 trace.json & trace_summary.csv 统一搬到 /home/roger/jsonfile
#
# 用法:
#   ./run_trace.sh [prompt_or_file] [max_len] [model_dir] [cuda_id]
# ------------------------------------------------

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

# ---------- 跑 trace ----------
python scripts/trace_kv_weight.py \
  --model-path "$MODEL_DIR" \
  --device cuda \
  --prompt "$PROMPT" \
  --max-gen-len "$MAX_LEN"

# # ---------- 归档到目标文件夹 ----------
# OUTDIR="/home/roger/jsonfile"
# mkdir -p "$OUTDIR"

# mv trace.json "$OUTDIR"/trace.json
# mv trace_summary.csv "$OUTDIR"/trace_summary.csv

# echo -e "✅ trace 完成，文件已移动到 $OUTDIR"
