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
        batch_size: int = 4,  
    ):
        
        num_batches = (len(prompts) + batch_size - 1) // batch_size
        
        # register future batches in the global state tracker
        from .global_state_tracker import get_global_tracker
        tracker = get_global_tracker()
        if tracker:
            actual_batches = list(range(num_batches))
            # 只在future_batches為空時註冊，避免覆蓋已存在的batch序列
            if not tracker.future_batches:
                tracker.register_future_batch(actual_batches)
                print(f"[INFO] Registered {num_batches} batches for {len(prompts)} prompts (batch_size={batch_size}): {actual_batches}")
        else:
            print("[WARNING] Global tracker not found during batch registration")
        
        self.args.max_batch_size = max(self.args.max_batch_size, len(prompts))

        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1

        prompts_tok = [
            self.tokenizer.encode(p, add_special_tokens=False) for p in prompts
        ]
        
        '''
        all_out_tokens: 最终每个 prompt 生成的 token ID 序列
        all_out_text: 对上面 token 的 decode 结果
        kv_profile: 每个 token 的 KV 访问 profile 记录（带时间和内存）
        '''
        all_out_tokens, all_out_text = [], []
        kv_profile = []
        
        for batch_idx in range(num_batches):
            try:
                # 确定当前批次的prompts范围
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(prompts_tok))
                batch_prompts = prompts_tok[start_idx:end_idx]
                
                # 顯示全局batch進度（如果可用）
                try:
                    if tracker and tracker.future_batches:
                        global_batch_idx = tracker.current_batch
                        total_global_batches = len(tracker.future_batches)
                        print(f"[INFO] Processing batch {global_batch_idx + 1}/{total_global_batches} with {len(batch_prompts)} prompts")
                    else:
                        print(f"[INFO] Processing batch {batch_idx + 1}/{num_batches} with {len(batch_prompts)} prompts")
                except:
                    print(f"[INFO] Processing batch {batch_idx + 1}/{num_batches} with {len(batch_prompts)} prompts")
                
                '''
                bsz: 当前 batch 的样本数
                max_prompt: 当前 batch 中最长的 prompt token 数
                total_len: 当前 batch 需要分配的最大序列长度 (最长 prompt + 可生成的 token)
                '''
                bsz = len(batch_prompts)
                max_prompt = max(len(x) for x in batch_prompts)
                total_len = min(self.args.max_seq_len, max_gen_len + max_prompt)
                
            except Exception as e:
                print(f"❌ Error during batch {batch_idx + 1} initialization: {e}")
                continue

            try:
                '''
                获取 tokenizer 里用于 padding 的 token ID;
                如果 tokenizer 没定义 pad_token_id(例如原生 LLaMA 就没有），则 fallback 使用 eos_token_id 来填充；
                这个 pad_id 将用于填满每条 prompt 后面的空白位置。
                '''
                pad_id = (
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                )
                '''
                tokens 是输入模型的 token ID 二维张量,shape 为 (bsz, total_len);
                初始化时全部填充为 pad_id,即“空”的标记;
                '''
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
            '''
            每个 step(cur_pos)做：
            1.调用模型进行一次 forward输入当前序列的最后一个 token;
            2.得到 logits → 根据温度采样或 argmax,得到下一个 token;
            3.写入 tokens;
            4.如果所有样本都生成了 <eos>，提前退出；
            5.同时收集 KV cache profiling 信息（时间、空间）；
            '''
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
