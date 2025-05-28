#!/usr/bin/env python
import argparse
import torch

from llama3.generator import LLaMA


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="目录下需包含 *.pth / tokenizer.model / params.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--prompt",
        nargs="+",
        required=True,
        help="支持一次输入多条 prompt,每条用引号包住",
    )
    p.add_argument("--max-gen-len", type=int, default=64)
    args = p.parse_args()

    llama = LLaMA.build(args.model_path, load_model=True, device=args.device)
    _, outs = llama.text_completion(args.prompt, max_gen_len=args.max_gen_len)
    for prompt, out in zip(args.prompt, outs):
        print("▼ Prompt:", prompt)
        print("▲ Completion:", out)
        print("-" * 60)


if __name__ == "__main__":
    main()
