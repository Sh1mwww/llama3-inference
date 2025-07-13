#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-batch inference launcher for llama3.generator.LLaMA

用法示例
--------
# 单文件、多行 prompt，按 batch_size 送入 GPU
python scripts/run_inference.py \
    --model-path /path/to/model \
    --prompt-file ./prompts/web_questions.txt \
    --batch-size 32 \
    --max-gen-len 64

# 命令行直接给多条 prompt（与旧版兼容）
python scripts/run_inference.py \
    --model-path /path/to/model \
    --prompt "你好" "请用中文解释牛顿第二定律" \
    --batch-size 8
"""
import argparse
import itertools
import math
import sys
from pathlib import Path
from typing import List

import torch

from llama3.generator import LLaMA


# ------------------------- Helpers ------------------------- #
def chunks(lst: List[str], n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def read_prompts_from_file(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    # 过滤空行
    return [ln for ln in lines if ln.strip()]


# ---------------------- Main routine ---------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Run LLaMA-3 inference with optional multi-batch splitting"
    )
    p.add_argument(
        "--model-path",
        required=True,
        help="目录下需包含 *.pth / tokenizer.model / params.json",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    prompt_group = p.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt",
        nargs="+",
        help="一次输入多条 prompt, 每条用引号包住（旧接口，兼容单/多 Batch）",
    )
    prompt_group.add_argument(
        "--prompt-file",
        help="文本文件路径；每行视作一条 prompt（推荐大规模推理时使用）",
    )
    p.add_argument("--batch-size", type=int, default=32,help="每个 Batch 实际并行条数")
    p.add_argument("--max-seq-len", type=int, default=2048,help="模型上下文窗口大小（预留显存用）")
    p.add_argument("--max-batch-size", type=int, default=512,help="缓存预分配的最大批容量，要 ≥ --batch-size")
    args = p.parse_args()

    # -------- Collect prompts -------- #
    if args.prompt_file:
        prompt_path = Path(args.prompt_file).expanduser().resolve()
        if not prompt_path.is_file():
            sys.exit(f"[ERROR] prompt-file not found: {prompt_path}")
        prompts = read_prompts_from_file(prompt_path)
    else:
        prompts = args.prompt or []

    if not prompts:
        sys.exit("[ERROR] No prompt specified.")

    print(
        f"[INFO] Prompts: {len(prompts)}  |  Batch size: {args.batch_size}  |  "
        f"Total batches: {math.ceil(len(prompts)/args.batch_size)}"
    )

    # -------- Build model (一次加载即可) -------- #
    llama = LLaMA.build(args.model_path, load_model=True, device=args.device)

    # -------- Run batched inference -------- #
    all_outputs: List[str] = []
    idx_offset = 0
    for batch_idx, prompt_batch in enumerate(chunks(prompts, args.batch_size), 1):
        print(f"\n🟢 Running batch {batch_idx}  (size={len(prompt_batch)}) …")
        _, outs = llama.text_completion(
            prompt_batch, max_gen_len=args.max_gen_len
        )
        all_outputs.extend(outs)

        # 实时打印本批次结果
        for i, (inp, out) in enumerate(zip(prompt_batch, outs), start=idx_offset):
            print(f"\n▼ [{i}] Prompt: {inp}")
            print(f"▲ [{i}] Completion: {out}")
            print("-" * 60)
        idx_offset += len(prompt_batch)

    # -------- 可选：将全部结果写文件（示例） -------- #
    # with open("inference_outputs.txt", "w", encoding="utf-8") as f:
    #     for i, (inp, out) in enumerate(zip(prompts, all_outputs)):
    #         f.write(f"[{i}] PROMPT:\n{inp}\n\n[{i}] COMPLETION:\n{out}\n\n{'='*80}\n")

    print(f"\n✅ Finished. Generated {len(all_outputs)} completions.")


if __name__ == "__main__":
    main()
