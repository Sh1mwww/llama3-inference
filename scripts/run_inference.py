"""
# 单文件、多行 prompt，按 batch_size 送入 GPU
python scripts/run_inference.py \
    --model-path /path/to/model \
    --prompt-file ./prompts/web_questions.txt \
    --batch-size 32 \
    --max-gen-len 64

# 命令行直接给多条 prompt
python scripts/run_inference.py \
    --model-path /path/to/model \
    --prompt "你好" "请用中文解释牛顿第二定律" \
    --batch-size 8
"""
import argparse
import math
import sys
import time
from pathlib import Path
from typing import List, Iterator

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


def safe_batch_processing(prompts: List[str], batch_size: int, max_seq_len: int = 2048) -> Iterator[List[str]]:
    """安全的批处理函数，确保每个batch符合模型限制"""
    current_batch = []
    current_batch_tokens = 0
    
    for i, prompt in enumerate(prompts):
        # 估算当前prompt的token数量（粗略估算）
        estimated_tokens = len(prompt.split()) if ' ' in prompt else len(prompt) // 3
        
        # 检查添加此prompt是否会超出限制
        if (len(current_batch) >= batch_size or 
            (current_batch_tokens + estimated_tokens > max_seq_len and current_batch)):
            
            if current_batch:
                yield current_batch
                current_batch = []
                current_batch_tokens = 0
        
        # 如果单个prompt过长，发出警告但仍然处理
        if estimated_tokens > max_seq_len:
            print(f"[WARNING] Prompt {i+1} has ~{estimated_tokens} tokens, exceeds max_seq_len {max_seq_len}")
        
        current_batch.append(prompt)
        current_batch_tokens += estimated_tokens
    
    # 处理最后一个batch
    if current_batch:
        yield current_batch


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
    p.add_argument("--batch-size", type=int, default=1, help="每个 Batch 实际并行条数")
    p.add_argument("--max-seq-len", type=int, default=2048, help="模型上下文窗口大小（预留显存用）")
    p.add_argument("--max-batch-size", type=int, default=512, help="缓存预分配的最大批容量，要 ≥ --batch-size")
    p.add_argument("--max-gen-len", type=int, default=64, help="生成文本的最大长度")
    args = p.parse_args()

    # Validate arguments
    if args.max_batch_size < args.batch_size:
        sys.exit(f"[ERROR] max-batch-size ({args.max_batch_size}) must be >= batch-size ({args.batch_size})")
    
    # 检查模型路径
    model_path = Path(args.model_path)
    if not model_path.exists():
        sys.exit(f"[ERROR] Model path does not exist: {model_path}")
    
    # 检查CUDA可用性
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARNING] CUDA not available, falling back to CPU")
        args.device = "cpu"

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

    # -------- Build model (先CPU后GPU，避免卡住) -------- #
    print("[INFO] Loading model on CPU first...")
    llama = LLaMA.build(args.model_path, load_model=True, device="cpu")
    
    # 如果需要GPU，单独转移
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[ERROR] CUDA not available but cuda device specified")
            sys.exit(1)
        
        print(f"[INFO] Transferring model to {args.device}...")
        try:
            llama.model.to(args.device)
            llama.args.device = args.device
            torch.cuda.synchronize()
            print(f"[INFO] Model successfully moved to {args.device}")
        except Exception as e:
            print(f"[ERROR] Failed to move model to GPU: {e}")
            sys.exit(1)
    else:
        llama.args.device = args.device

    # -------- Run batched inference -------- #
    all_outputs: List[str] = []
    idx_offset = 0
    total_batches = math.ceil(len(prompts) / args.batch_size)
    
    print(f"[INFO] Starting inference with {total_batches} batches...")
    
    try:
        for batch_idx, prompt_batch in enumerate(safe_batch_processing(prompts, args.batch_size, args.max_seq_len), 1):
            print(f"\n🟢 Running batch {batch_idx}/{total_batches} (size={len(prompt_batch)}) …")
            
            start_time = time.time()
            _, outs = llama.text_completion(
                prompt_batch, max_gen_len=args.max_gen_len
            )
            batch_time = time.time() - start_time
            
            all_outputs.extend(outs)
            print(f"✅ Batch {batch_idx} completed in {batch_time:.2f}s")

            # 实时打印本批次结果
            for i, (inp, out) in enumerate(zip(prompt_batch, outs), start=idx_offset):
                print(f"\n▼ [{i}] Prompt: {inp}")
                print(f"▲ [{i}] Completion: {out}")
                print("-" * 60)
            idx_offset += len(prompt_batch)
            
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        sys.exit(1)

    # -------- 可选：将全部结果写文件（示例） -------- #
    # with open("inference_outputs.txt", "w", encoding="utf-8") as f:
    #     for i, (inp, out) in enumerate(zip(prompts, all_outputs)):
    #         f.write(f"[{i}] PROMPT:\n{inp}\n\n[{i}] COMPLETION:\n{out}\n\n{'='*80}\n")

    print(f"\n✅ Finished. Generated {len(all_outputs)} completions.")


if __name__ == "__main__":
    main()
