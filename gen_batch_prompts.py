#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic prompts for throughput experiments.
- Exactly N tokens per prompt (as measured by the chosen tokenizer)
- B prompts per batch
- Output as a single .txt file (default), JSONL, or one-file-per-prompt

Example:
  python gen_batch_prompts.py \
      --tokenizer /path/to/your/model_or_tokenizer_dir \
      --batch 512 --tokens 2048 \
      --out prompts_batch512_len2048.txt --format txt

Requirements:
  pip install transformers sentencepiece
"""
import argparse
import math
import random
from pathlib import Path
from typing import List

from transformers import AutoTokenizer


LOREM = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis "
    "nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat "
    "duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore "
    "eu fugiat nulla pariatur excepteur sint occaecat cupidatat non proident sunt "
    "in culpa qui officia deserunt mollit anim id est laborum"
).split()


def build_block_text(i: int, min_words: int = 80) -> str:
    """Make a deterministic, readable base block of text for prompt i."""
    # deterministic "random" but reproducible per i
    rng = random.Random(i * 9973 + 42)
    words = []
    while len(words) < min_words:
        # mix a short header with shuffled lorem words
        words.extend(rng.sample(LOREM, k=min(len(LOREM), min_words - len(words))))
    header = f"[Synthetic throughput prompt #{i}] task=benchmark domain=generic note=do_not_answer.\n"
    return header + " ".join(words)


def make_prompt_with_exact_tokens(tokenizer, n_tokens: int, i: int) -> str:
    """
    Create a text prompt that tokenizes to exactly `n_tokens` tokens under `tokenizer`.
    Strategy: encode a readable base block once, tile its token ids to length, slice to N, decode.
    """
    base_text = build_block_text(i=i, min_words=96)
    block_ids: List[int] = tokenizer.encode(base_text, add_special_tokens=False)
    if len(block_ids) == 0:
        # extremely unlikely, but keep a safe fallback
        block_ids = tokenizer.encode("benchmark", add_special_tokens=False)
        if len(block_ids) == 0:
            raise RuntimeError("Tokenizer produced empty ids for fallback text. Please specify a different tokenizer.")
    reps = math.ceil(n_tokens / len(block_ids))
    ids = (block_ids * reps)[:n_tokens]
    prompt_text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # double-check exact length; if mismatch, repair by re-encoding → slice → decode
    check_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if len(check_ids) != n_tokens:
        ids = check_ids[:n_tokens]
        prompt_text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # one more assert (should match now)
        assert len(tokenizer.encode(prompt_text, add_special_tokens=False)) == n_tokens, \
            "Unable to create prompt with exact token length; try a different tokenizer or reduce n_tokens."
    return prompt_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=str, required=True,
                    help="HF model or local dir containing tokenizer files (use the SAME tokenizer as your model).")
    ap.add_argument("--batch", type=int, default=512, help="Number of prompts to generate.")
    ap.add_argument("--tokens", type=int, default=2048, help="Exact token length per prompt.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output path. Defaults to ./prompts_batch{batch}_len{tokens}.txt for --format txt.")
    ap.add_argument("--format", choices=["txt", "jsonl", "lines", "split"], default="txt",
                    help="txt: single file with separators; jsonl: one JSON per line; "
                         "lines: one prompt per line; split: one .txt per prompt in a new folder.")
    ap.add_argument("--seed", type=int, default=123, help="Seed for reproducibility.")
    args = ap.parse_args()

    random.seed(args.seed)

    # Load tokenizer (fast if available). Use trust_remote_code=False for safety.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, trust_remote_code=False)

    # Resolve output path
    if args.out is None:
        if args.format == "split":
            out_path = Path(f"./prompts_batch{args.batch}_len{args.tokens}")
        else:
            out_path = Path(f"./prompts_batch{args.batch}_len{args.tokens}.txt")
    else:
        out_path = Path(args.out)

    prompts = []
    for i in range(args.batch):
        p = make_prompt_with_exact_tokens(tokenizer, args.tokens, i)
        prompts.append(p)

    if args.format == "split":
        out_path.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(prompts):
            (out_path / f"prompt_{i:04d}.txt").write_text(p, encoding="utf-8")
        print(f"Wrote {args.batch} files to {out_path.resolve()}")
        return

    if args.format == "jsonl":
        with open(out_path, "w", encoding="utf-8") as f:
            for i, p in enumerate(prompts):
                # minimal JSON to keep things simple
                obj = {"id": i, "prompt": p}
                f.write(__import__("json").dumps(obj, ensure_ascii=False) + "\n")
        print(f"Wrote JSONL to {out_path.resolve()}")
        return

    if args.format == "lines":
        with open(out_path, "w", encoding="utf-8") as f:
            for p in prompts:
                f.write(p.replace("\n", " ") + "\n")
        print(f"Wrote one-prompt-per-line text to {out_path.resolve()}")
        return

    # default: human-friendly txt with section headers
    with open(out_path, "w", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            f.write(f"===== PROMPT {i:04d} (len={args.tokens} tokens) =====\n")
            f.write(p)
            f.write("\n\n")
    print(f"Wrote combined TXT to {out_path.resolve()}")


if __name__ == "__main__":
    main()
