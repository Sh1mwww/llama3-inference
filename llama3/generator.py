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
        self.model = Transformer(args).to(args.device)
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint, strict=False)

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
            str(params_path), max_seq_len=2048, max_batch_size=32, device=device
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
    ):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1

        prompts_tok = [
            self.tokenizer.encode(p, add_special_tokens=False) for p in prompts
        ]
        bsz = len(prompts_tok)
        max_prompt = max(len(x) for x in prompts_tok)
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt)

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
        for i, tok in enumerate(prompts_tok):
            tokens[i, : len(tok)] = torch.tensor(tok, device=self.args.device)

        eos_mask = torch.zeros(bsz, dtype=torch.bool, device=self.args.device)
        prompt_mask = tokens != pad_id
        kv_profile = []

        for cur_pos in tqdm(range(1, total_len), desc="Generating tokens"):
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

            # profiling
            kv_re_time = sum(self.model.kv_times)
            kv_bytes = sum(
                (layer.attention.cache_k[:, :cur_pos].numel()
                 + layer.attention.cache_v[:, :cur_pos].numel())
                * layer.attention.cache_k.element_size()
                for layer in self.model.layers
            )
            kv_profile.append(
                {
                    "token_idx": int(cur_pos),
                    "phase": "prefill" if cur_pos < max_prompt else "decode",
                    "kv_re_ms": float(kv_re_time),
                    "kv_kb": float(kv_bytes / 1024),
                    "per_layer_kv_ms": self.model.kv_times.copy(),
                }
            )

        # ---- 输出整理 ----
        out_tokens, out_text = [], []
        for row in tokens.tolist():
            if self.tokenizer.eos_token_id in row:
                row = row[: row.index(self.tokenizer.eos_token_id)]
            out_tokens.append(row)
            out_text.append(self.tokenizer.decode(row))

        # 保存 profiling
        save_name = (
            f"{Path(self.args.checkpoints_dir).name}_kv_profile.json"
        )
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
