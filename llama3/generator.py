import json
import os
import time
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import LlamaTokenizerFast,AutoTokenizer

from .config import ModelArgs
from .model import Transformer

# NVTX profiling support
try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False
    # Fallback no-op functions
    class nvtx:
        @staticmethod
        def range_push(name): pass
        @staticmethod
        def range_pop(): pass

# tokenizer, checkpoint, args
class LLaMA:
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
    
    def _configure_weight_streaming(self, streaming_config: dict):
        """配置权重流式传输"""
        print("🔧 Configuring Weight Streaming...")
        
        # 导入必要的模块
        from .weight_streaming_manager import WeightStreamingManager
        from . import layers
        from . import stream_mnt
        
        # 设置默认配置
        config = {
            'prefetch_distance': 1,
            'max_cached_layers': 4,
            'warmup_layers': 1,
            'verbose': False
        }
        config.update(streaming_config)
        
        # 确保模型的 layers 属性可访问（供 WSM 使用）
        if hasattr(self.model, "layer_infos"):
            try:
                blocks = [info.block for info in self.model.layer_infos if info.block is not None]
                if blocks and not hasattr(self.model, "layers"):
                    self.model.layers = blocks
            except Exception:
                pass
        
        # 配置核心组件到目标设备（小模块常驻 HBM）
        self._configure_core_components()
        
        # 创建和配置 WSM
        wsm = WeightStreamingManager(
            self.model, 
            device=self.args.device,
            prefetch_distance=config['prefetch_distance'],
            max_cached_layers=config['max_cached_layers'],
            warmup_layers=config['warmup_layers'],
            verbose=True  # 强制启用详细日志以便验证
        )
        
        # 集成 WSM 到模型层
        self._integrate_wsm_to_layers(wsm)
        
        # 配置 KV streams
        self._configure_kv_streams()
        
        # 验证并修复设备放置
        self._verify_and_fix_device_placement()
        
        # 输出关键的诊断信息
        try:
            first_blk = getattr(self.model, "layers", [None])[0]
            if first_blk is not None:
                print("[CHECK] first block param device:", next(first_blk.parameters()).device)
        except Exception:
            pass
        
        print("✅ Weight streaming enabled (activations on GPU, weights streamed per-layer).")
        print("⚙️  Running on GPU")
        
        return wsm
    
    def _configure_core_components(self):
        """配置核心组件到目标设备"""
        device = self.args.device
        model = self.model
        
        # 小模块常驻 HBM
        model.embed_tokens = model.embed_tokens.to(device)
        model.norm = model.norm.to(device)
        model.output = model.output.to(device)
        
        # 处理 freqs_complex
        self._handle_freqs_complex(device)
    
    def _handle_freqs_complex(self, device: str):
        """处理 freqs_complex 的设备放置与重建"""
        model = self.model
        
        if hasattr(model, "freqs_complex"):
            try:
                model.freqs_complex = model.freqs_complex.to(device)
            except Exception as e:
                print(f"⚠️ Warning: Failed to move freqs_complex to {device}: {e}")
                # 重新创建 freqs_complex 在目标设备上
                self._recreate_freqs_complex(device)
    
    def _recreate_freqs_complex(self, device: str):
        """重新创建 freqs_complex 在目标设备上"""
        try:
            from .layers import precompute_theta_pos_frequencies
            print(f"   Attempting to recreate freqs_complex on {device}...")
            
            # 使用 ModelArgs 中的配置
            dim = self.args.dim
            n_heads = self.args.n_heads
            max_seq_len = self.args.max_seq_len
            rope_theta = self.args.rope_theta
            
            self.model.freqs_complex = precompute_theta_pos_frequencies(
                dim // n_heads,
                max_seq_len * 2,
                device=device,
                theta=rope_theta,
            )
            print(f"   Successfully recreated freqs_complex on {device}")
        except Exception as e:
            print(f"   Failed to recreate freqs_complex: {e}")
            raise RuntimeError(f"Cannot ensure freqs_complex is on {device}") from e
    
    def _integrate_wsm_to_layers(self, wsm):
        """将 WSM 集成到模型层"""
        try:
            from . import layers
            layers.set_weight_manager(wsm)  # 设置全局引用
            
            # 为现有层手动注入
            if hasattr(self.model, "layers"):
                for i, layer in enumerate(self.model.layers):
                    if hasattr(layer, "attention"):
                        layer.attention.weight_manager = wsm
                        layer.attention.layer_id = getattr(layer, "layer_id", i)
                    if hasattr(layer, "feed_forward"):
                        layer.feed_forward.weight_manager = wsm
                        layer.feed_forward.layer_id = getattr(layer, "layer_id", i)
        except Exception as e:
            print(f"[WARN] failed to set_weight_manager on blocks: {e}")
    
    def _configure_kv_streams(self):
        """配置 KV streams"""
        try:
            from . import stream_mnt
            streams = stream_mnt.get_streams(self.args.device)
            
            if hasattr(self.model, "layers"):
                for layer in self.model.layers:
                    if hasattr(layer, "attention"):
                        off = getattr(layer.attention, "offloader", None)
                        if off is not None:
                            off.h2d_stream = streams.kv_h2d
                            off.d2h_stream = streams.kv_d2h
        except Exception as e:
            print(f"[WARN] failed to configure KV streams: {e}")
    
    def _verify_and_fix_device_placement(self):
        """验证并修复设备放置"""
        device = self.args.device
        model = self.model
        
        if not device.startswith("cuda"):
            return
            
        print("🔍 Verifying device placement before inference...")
        
        try:
            # 强制同步所有核心组件到目标设备
            print("🔧 Synchronizing all components to target device...")
            model.embed_tokens = model.embed_tokens.to(device)
            model.norm = model.norm.to(device)
            model.output = model.output.to(device)
            if hasattr(model, 'freqs_complex'):
                model.freqs_complex = model.freqs_complex.to(device)
            
            # 同步所有层的 norm 权重到 GPU
            print("🔧 Synchronizing layer norms to GPU...")
            if hasattr(model, "layers"):
                for layer in model.layers:
                    if hasattr(layer, 'attn_norm'):
                        layer.attn_norm = layer.attn_norm.to(device)
                    if hasattr(layer, 'ffn_norm'):
                        layer.ffn_norm = layer.ffn_norm.to(device)
            
            # GPU 同步确保所有操作完成
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            print("✅ All layer components synchronized to target device")
                
        except Exception as e:
            print(f"⚠️ Error during device synchronization: {e}")

    # ---------- 构建 ----------
    @staticmethod
    def build(
        checkpoints_dir: str,
        load_model: bool = True,
        device: str = "cuda",
        enable_weight_streaming: bool = False,
        streaming_config: Optional[dict] = None,
        topk_blk: Optional[int] = None,
        max_seq_len: int = 2048,
        max_batch_size: int = 512,
    ) -> "LLaMA":
        ckpt_dir = Path(checkpoints_dir)
        tokenizer = LlamaTokenizerFast.from_pretrained(pretrained_model_name_or_path=ckpt_dir/tokenizer.model, legacy=True)
        params_path = ckpt_dir / "params.json"
        args = ModelArgs.from_json(
            str(params_path), max_seq_len=max_seq_len, max_batch_size=max_batch_size, device=device
        )
        
        # 如果指定了 topk_blk，更新到 ModelArgs 中
        if topk_blk is not None:
            args.topk_blk = topk_blk
        
        args.checkpoints_dir = str(ckpt_dir)

        checkpoint = None
        if load_model:
            ckpt_file = sorted(ckpt_dir.glob("*.pth"))[0]
            print(f"[INFO] Loading checkpoint: {ckpt_file}")
            t0 = time.time()
            checkpoint = torch.load(ckpt_file, map_location="cpu")
            print(f"[INFO] Done ({time.time() - t0:.1f}s)")

        # 先在 CPU 上创建 LLaMA 实例（避免 OOM）
        cpu_args = ModelArgs.from_json(
            str(params_path), max_seq_len=max_seq_len, max_batch_size=max_batch_size, device="cpu"
        )
        if topk_blk is not None:
            cpu_args.topk_blk = topk_blk
        cpu_args.checkpoints_dir = str(ckpt_dir)
        
        llama = LLaMA(tokenizer, checkpoint, cpu_args)
        
        # 如果启用权重流式传输且设备是 CUDA
        if enable_weight_streaming and device.startswith("cuda"):
            llama._configure_weight_streaming(streaming_config or {})
        elif device.startswith("cuda"):
            # 非流式传输模式：直接移动到 GPU
            try:
                llama.model = llama.model.to(device)
                llama.args.device = device
            except torch.cuda.OutOfMemoryError:
                print("❌ CUDA OOM when moving model. Keeping on CPU...")
                device = "cpu"
                llama.args.device = "cpu"
        
        return llama

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
        nvtx.range_push("text_completion")
        
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

        nvtx.range_push("tokenization")
        prompts_tok = [
            self.tokenizer.encode(p, add_special_tokens=False) for p in prompts
        ]
        nvtx.range_pop()  # tokenization
        
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
            # 获取全局批次信息以显示更准确的进度
            try:
                tracker = get_global_tracker()
                if tracker and hasattr(tracker, 'current_batch') and tracker.current_batch is not None:
                    global_batch_num = tracker.current_batch + 1
                    total_global_batches = len(tracker.future_batches) if hasattr(tracker, 'future_batches') else 'Unknown'
                    desc = f"Generating tokens for batch {global_batch_num}/{total_global_batches} (local {batch_idx + 1}/{num_batches})"
                else:
                    desc = f"Generating tokens for batch {batch_idx + 1}/{num_batches}"
            except:
                desc = f"Generating tokens for batch {batch_idx + 1}/{num_batches}"
            
            for cur_pos in tqdm(range(1, total_len), desc=desc):
                nvtx.range_push(f"token_{cur_pos}_generation")
                try:
                    # ---- forward ----
                    nvtx.range_push(f"token_{cur_pos}_forward")
                    with torch.no_grad():
                        logits = self.model(tokens[:, cur_pos - 1 : cur_pos], cur_pos)
                    nvtx.range_pop()  # forward

                    nvtx.range_push(f"token_{cur_pos}_sampling")
                    if temperature > 0:
                        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                        next_tok = self._sample_top_p(probs, top_p)
                    else:
                        next_tok = torch.argmax(logits[:, -1], dim=-1)

                    next_tok = next_tok.reshape(-1)
                    next_tok = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_tok)
                    tokens[:, cur_pos] = next_tok
                    eos_mask |= (~prompt_mask[:, cur_pos]) & (next_tok == self.tokenizer.eos_token_id)
                    nvtx.range_pop()  # sampling
                    
                    if eos_mask.all():
                        nvtx.range_pop()  # token_generation
                        break
                        
                except torch.cuda.OutOfMemoryError as e:
                    print(f"❌ CUDA OOM during inference at position {cur_pos}: {e}")
                    torch.cuda.empty_cache()
                    nvtx.range_pop()  # token_generation (error case)
                    raise RuntimeError(f"GPU out of memory during inference") from e
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        print(f"❌ CUDA error during inference at position {cur_pos}: {e}")
                        torch.cuda.empty_cache()
                        nvtx.range_pop()  # token_generation (error case)
                        raise RuntimeError(f"CUDA error during inference") from e
                    else:
                        nvtx.range_pop()  # token_generation (error case)
                        raise
                        
                nvtx.range_pop()  # token_generation

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
