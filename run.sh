#!/bin/bash

# 默认参数，可通过命令行覆盖
MODEL_PATH=${1:-"/home/roger/.llama/checkpoints/Llama3.2-1B"}
DEVICE=${2:-"cuda"}
PROMPT=${3:-"The theory of relativity states that"}
MAX_GEN_LEN=${4:-64}

# 激活 Python 环境（如有需要）
# source ~/.pyenv/versions/3.13.2/bin/activate

# 执行推理
python scripts/run_inference.py \
  --model-path "$MODEL_PATH" \
  --device "$DEVICE" \
  --prompt "$PROMPT" \
  --max-gen-len "$MAX_GEN_LEN"
