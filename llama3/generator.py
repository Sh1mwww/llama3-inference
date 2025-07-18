import json
import os
import time
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import LlamaTokenizerFast

from .config import ModelArgs
from .model import Transformer


class LLaMA:
    """
    推理包装：负责
      1. 加载 tokenizer / checkpoint
      2. 采样 & top-p
      3. KV profiling 保存
    """

    def __init__(self, tokenizer, checkpoint, args: ModelArgs):
        self.tokenizer = tokenizer
        self.args = args
        
        # 初始化全局状态跟踪器
        from .global_state_tracker import init_global_tracker, get_global_tracker
        from .kv_offload import BLOCK
        if get_global_tracker() is None:
            print(f"[INFO] Initializing global state tracker...")
            n_blocks = (args.max_seq_len + BLOCK - 1) // BLOCK  # 计算需要的block数量
            tracker = init_global_tracker(
                max_batch=args.max_batch_size,
                layers=args.n_layers,
                n_blocks=n_blocks
            )
            # 不设置默认的future batches，等待实际使用时再设置
            print(f"[INFO] Global state tracker initialized, waiting for actual batch registration")
        
        print(f"[INFO] Initializing model on device: {args.device}")
        self.model = Transformer(args)
        
        print(f"[INFO] Moving model to {args.device}...")
        self.model = self.model.to(args.device)
        
        print(f"[INFO] Converting to half precision...")
        self.model = self.model.half()
        
        if checkpoint is not None:
            print(f"[INFO] Loading state dict...")
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint, strict=False)
            if missing_keys:
                print(f"[WARNING] Missing keys: {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"[WARNING] Unexpected keys: {len(unexpected_keys)} keys")
            print(f"[INFO] Model weights loaded successfully")
        

    # ---------- 构建 ----------
    @staticmethod
    def build(
        checkpoints_dir: str,
        load_model: bool = True,
        device: str = "cuda",
    ) -> "LLaMA":
        ckpt_dir = Path(checkpoints_dir)
        tokenizer = LlamaTokenizerFast.from_pretrained(ckpt_dir, legacy=True)
        params_path = ckpt_dir / "params.json"
        args = ModelArgs.from_json(
            str(params_path), max_seq_len=2048, max_batch_size=512, device=device
        )
        
        args.checkpoints_dir = str(ckpt_dir)

        checkpoint = None
        if load_model:
            ckpt_file = sorted(ckpt_dir.glob("*.pth"))[0]
            print(f"[INFO] Loading checkpoint: {ckpt_file}")
            t0 = time.time()
            checkpoint = torch.load(ckpt_file, map_location="cpu")
            print(f"[INFO] Done ({time.time() - t0:.1f}s)")

        return LLaMA(tokenizer, checkpoint, args)

    # ---------- 推理 ----------
    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        profile_output_dir: Optional[str] = None,
        batch_size: int = 4,  # 每个batch的prompt数量
    ):
        # 根据prompts数量和batch_size计算需要的batch数量
        num_batches = (len(prompts) + batch_size - 1) // batch_size
        
        # 注册实际需要的batch indices
        from .global_state_tracker import get_global_tracker
        tracker = get_global_tracker()
        if tracker:
            actual_batches = list(range(num_batches))
            tracker.register_future_batch(actual_batches)
            print(f"[INFO] Registered {num_batches} batches for {len(prompts)} prompts (batch_size={batch_size}): {actual_batches}")
        
        self.args.max_batch_size = max(self.args.max_batch_size, len(prompts))

        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1

        prompts_tok = [
            self.tokenizer.encode(p, add_special_tokens=False) for p in prompts
        ]
        
        # 按batch_size分组处理prompts
        all_out_tokens, all_out_text = [], []
        kv_profile = []
        
        for batch_idx in range(num_batches):
            try:
                # 确定当前批次的prompts范围
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(prompts_tok))
                batch_prompts = prompts_tok[start_idx:end_idx]
                
                # 更新tracker的当前批次
                if tracker:
                    tracker.set_current_execution(batch_idx, 0)
                
                print(f"[INFO] Processing batch {batch_idx + 1}/{num_batches} with {len(batch_prompts)} prompts")
                
                # 处理当前批次
                bsz = len(batch_prompts)
                max_prompt = max(len(x) for x in batch_prompts)
                total_len = min(self.args.max_seq_len, max_gen_len + max_prompt)
                
            except Exception as e:
                print(f"❌ Error during batch {batch_idx + 1} initialization: {e}")
                continue

            try:
                pad_id = (
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                )
                tokens = torch.full(
                    (bsz, total_len),
                    pad_id,
                    dtype=torch.long,
                    device=self.args.device,
                )
                for i, tok in enumerate(batch_prompts):
                    tokens[i, : len(tok)] = torch.tensor(tok, device=self.args.device)

                eos_mask = torch.zeros(bsz, dtype=torch.bool, device=self.args.device)
                prompt_mask = tokens != pad_id
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"❌ CUDA OOM during batch {batch_idx + 1} tensor allocation: {e}")
                torch.cuda.empty_cache()
                continue
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"❌ CUDA error during batch {batch_idx + 1} tensor allocation: {e}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

            for cur_pos in tqdm(range(1, total_len), desc=f"Generating tokens for batch {batch_idx + 1}"):
                try:
                    # ---- forward ----
                    with torch.no_grad():
                        logits = self.model(tokens[:, cur_pos - 1 : cur_pos], cur_pos)

                    if temperature > 0:
                        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                        next_tok = self._sample_top_p(probs, top_p)
                    else:
                        next_tok = torch.argmax(logits[:, -1], dim=-1)

                    next_tok = next_tok.reshape(-1)
                    next_tok = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_tok)
                    tokens[:, cur_pos] = next_tok
                    eos_mask |= (~prompt_mask[:, cur_pos]) & (next_tok == self.tokenizer.eos_token_id)
                    if eos_mask.all():
                        break
                        
                except torch.cuda.OutOfMemoryError as e:
                    print(f"❌ CUDA OOM during inference at position {cur_pos}: {e}")
                    torch.cuda.empty_cache()
                    raise RuntimeError(f"GPU out of memory during inference") from e
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        print(f"❌ CUDA error during inference at position {cur_pos}: {e}")
                        torch.cuda.empty_cache()
                        raise RuntimeError(f"CUDA error during inference") from e
                    else:
                        raise

                kv_re_time = sum(self.model.kv_times)
                bytes_per_token = (                
                    2 * self.model.args.n_kv_heads
                    * self.model.args.dim // self.model.args.n_heads
                    * self.model.embed_tokens.weight.element_size()
                )
                kv_bytes = bytes_per_token * cur_pos * self.model.args.n_layers
                kv_profile.append(
                    {
                        "batch_idx": batch_idx,
                        "token_idx": int(cur_pos),
                        "phase": "prefill" if cur_pos < max_prompt else "decode",
                        "kv_re_ms": float(kv_re_time),
                        "kv_kb": float(kv_bytes / 1024),
                    }
                )
                
            # ---- 处理当前批次输出 ----
            for row in tokens.tolist():
                if self.tokenizer.eos_token_id in row:
                    row = row[: row.index(self.tokenizer.eos_token_id)]
                all_out_tokens.append(row)
                all_out_text.append(self.tokenizer.decode(row))
        
        # 使用处理后的结果
        out_tokens, out_text = all_out_tokens, all_out_text

        # 保存 profiling
        if profile_output_dir:
            os.makedirs(profile_output_dir, exist_ok=True)
            save_name = os.path.join(profile_output_dir, f"{Path(self.args.checkpoints_dir).name}_kv_profile.json")
            with open(save_name, "w", encoding="utf-8") as f:
                json.dump(kv_profile, f, indent=2)
            print(f"[INFO] KV profile saved → {save_name}")

        return out_tokens, out_text

    # ---------- utils ----------
    @staticmethod
    def _sample_top_p(probs, p):
        sort_probs, sort_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sort_probs, dim=-1)
        sort_probs[cumsum - sort_probs > p] = 0.0
        sort_probs.div_(sort_probs.sum(dim=-1, keepdim=True))
        next_tok = torch.multinomial(sort_probs, 1)
        return torch.gather(sort_idx, -1, next_tok)

    def encode(self, text: str):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)
