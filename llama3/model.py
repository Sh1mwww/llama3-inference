from typing import List, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import ModelArgs,  LayerInfo
from .layers import (
    RMSNorm,
    EncoderBlock,
    precompute_theta_pos_frequencies,
)

@dataclass
class _RuntimeSwitch:
    """运行时开关：控制是否使用管线化 forward"""
    PIPELINED: bool = True   # 可通过外部配置/环境变量控制


class Transformer(nn.Module):
    """
    纯粹的 Forward 计算与 KV profiling，**不**负责权重载入 / 采样
    """

    def print_device_info(self, tokenizer=None):
        """
        打印设备和形状信息，用于调试设备一致性问题

        Args:
            tokenizer: 可选的 tokenizer，用于验证 vocab_size 一致性
        """
        print("=" * 80)
        print("Transformer 设备一致性检查")
        print("=" * 80)

        # 1. Tokenizer vocab size
        if tokenizer is not None:
            tokenizer_vocab_size = len(tokenizer)
            print(f"Tokenizer vocab_size:           {tokenizer_vocab_size}")

        # 2. Embedding 层信息
        print(f"embed_tokens.num_embeddings:    {self.embed_tokens.num_embeddings}")
        print(f"embed_tokens.embedding_dim:     {self.embed_tokens.embedding_dim}")
        print(f"embed_tokens.weight.shape:      {self.embed_tokens.weight.shape}")
        print(f"embed_tokens.weight.device:     {self.embed_tokens.weight.device}")
        print(f"embed_tokens.weight.dtype:      {self.embed_tokens.weight.dtype}")

        # 3. Output 层信息
        print(f"output.in_features:             {self.output.in_features}")
        print(f"output.out_features:            {self.output.out_features}")
        print(f"output.weight.shape:            {self.output.weight.shape}")
        print(f"output.weight.device:           {self.output.weight.device}")
        print(f"output.weight.dtype:            {self.output.weight.dtype}")

        # 4. 验证一致性
        print("\n" + "-" * 80)
        print("一致性验证:")
        print("-" * 80)

        # vocab_size 一致性
        if tokenizer is not None:
            if tokenizer_vocab_size == self.embed_tokens.num_embeddings:
                print(f"✅ Tokenizer vocab_size ({tokenizer_vocab_size}) == embed_tokens.num_embeddings ({self.embed_tokens.num_embeddings})")
            else:
                print(f"❌ Tokenizer vocab_size ({tokenizer_vocab_size}) != embed_tokens.num_embeddings ({self.embed_tokens.num_embeddings})")

        # embed vs output
        if self.embed_tokens.num_embeddings == self.output.out_features:
            print(f"✅ embed_tokens.num_embeddings ({self.embed_tokens.num_embeddings}) == output.out_features ({self.output.out_features})")
        else:
            print(f"❌ embed_tokens.num_embeddings ({self.embed_tokens.num_embeddings}) != output.out_features ({self.output.out_features})")

        # 设备一致性
        embed_dev = self.embed_tokens.weight.device
        output_dev = self.output.weight.device
        if embed_dev == output_dev:
            print(f"✅ 设备一致: embed ({embed_dev}) == output ({output_dev})")
        else:
            print(f"❌ 设备不一致: embed ({embed_dev}) != output ({output_dev})")

        print("=" * 80)

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size

        # ★ 确保 vocab_size 合法（必须 > 0）
        assert args.vocab_size > 0, f"vocab_size 必须 > 0，当前值: {args.vocab_size}"

        # 创建 embedding 层：num_embeddings = vocab_size，embedding_dim = dim
        # 注意：num_embeddings 必须等于后续加载权重的 shape[0]，否则前向查表会出错
        self.embed_tokens = nn.Embedding(args.vocab_size, args.dim)
        # self.layers = nn.ModuleList(
        #     [EncoderBlock(args, i) for i in range(args.n_layers)]
        # )
        
        self.layers = nn.ModuleList()
        self.layer_infos: List[Dict[str, Any]] = []
        for i in range(args.n_layers):
            blk = EncoderBlock(args, i)
            self.layers.append(blk)
            self.layer_infos.append({
                "layer_id": i,
                "block": blk,          # 方便从 info 直接拿模块
                "extra": {}            # 留给后续任意字段
            })     
               
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        # ★ 输出投影层：必须与 embed_tokens 在 vocab 维度对齐
        # nn.Linear(in_features=dim, out_features=vocab_size)
        # output.weight.shape = [vocab_size, dim]，与 embed_tokens.weight 同形状
        # 许多模型会 tie weights（共享权重），但即使不共享，vocab_size 也必须一致
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(
            args.dim // args.n_heads,
            args.max_seq_len * 2,
            device=args.device,
            theta=args.rope_theta,
        )

        self.kv_times: List[float] = [0.0] * args.n_layers
        self.attn_times: List[float] = [0.0] * args.n_layers

    def _forward_pipelined(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        """
        跨层事件串接：全程异步提交 + 仅在最后等待。

        改进点：
        1) 使用 forward_async() 返回 (out, done_evt)
        2) L+1 的 forward_async 在提交时对 prev_done_evt 做 wait_event
        3) CPU 不在每层同步，而是在所有层排完后再等待最后一个事件
        4) 这样即使下一层不能提前算（数据依赖），CPU 也能提前把权重 H2D 挂到计算流
        """
        # ⭐ 关键修复：使用 embed_tokens 的设备（已在 _verify_and_fix_device_placement 中确保在 CUDA）
        # 而不是 self.args.device（可能还是字符串 "cuda" 或被错误设置为 "cpu"）
        embed_dev = self.embed_tokens.weight.device
        if not embed_dev.type.startswith("cuda"):
            raise RuntimeError(
                f"embed_tokens.weight must be on CUDA, got {embed_dev}. "
                f"This should have been caught by _verify_and_fix_device_placement."
            )

        dev = embed_dev  # 使用实际的 CUDA device 对象，而不是字符串
        dtype = getattr(self, "param_dtype", torch.bfloat16)

        # 1) embed：确保 tokens 与 embedding 在同一设备（与原实现一致）
        if tokens.device != embed_dev:
            if tokens.device.type == "cpu" and not tokens.is_pinned():
                tokens = tokens.pin_memory()
            tokens = tokens.to(embed_dev, non_blocking=True)
        if tokens.dtype != torch.long:
            tokens = tokens.long()

        # 2) 执行 embedding
        h = self.embed_tokens(tokens)  # 与权重同设备（必然是 CUDA）

        # 3) 如有不一致，迁移到目标计算设备 + 统一 dtype
        # ⭐ 安全网：即便 embedding 权重在 CPU，激活也会被立刻转回 CUDA，不会让后面崩掉
        # （但由于上面已经检查了 embed_dev 必须是 CUDA，这里理论上不会触发设备转换）
        if h.device != dev or h.dtype != dtype:
            h = h.to(device=dev, dtype=dtype, non_blocking=True)

        # 2) freqs 只在设备不一致时搬一次（沿用缓存逻辑）
        freqs_dev_key = str(dev)  # 用字符串作为缓存键
        if getattr(self, "_freqs_cached_dev", None) != freqs_dev_key:
            self._freqs_cached = self.freqs_complex.to(dev, non_blocking=True)
            self._freqs_cached_dev = freqs_dev_key
        freqs = self._freqs_cached

        # 3) 跨层事件串接：逐层调用 forward_async，传递前一层的 done_evt
        prev_done = None
        for idx, info in enumerate(self.layer_infos):
            blk = info["block"]
            # forward_async 返回 (out, done_evt)
            # wait_on=prev_done 让 GPU 自动建立依赖，CPU 不阻塞
            h, prev_done = blk.forward_async(h, start_pos, freqs, wait_on=prev_done)

            # 更新性能统计（与原 forward 一致）
            self.kv_times[idx] = blk.attention.kv_elapsed_time
            self.attn_times[idx] = blk.attention.attn_time

        # 4) 到真正要用输出时再等待最后一层事件（同步点）
        # ⭐ 使用 with torch.cuda.device(dev) 避免隐式同步（PyTorch 解析字符串设备时可能触发同步）
        if prev_done is not None:
            with torch.cuda.device(dev):
                torch.cuda.current_stream(dev).wait_event(prev_done)
            # 释放最后一层的事件（可选：如果 stream_mnt 需要）
            try:
                from llama3 import stream_mnt
                # 注意：forward_async 中只释放了 mha_evt，ffn_evt 由这里释放
                # 但我们无法拿到 ffn_eid，所以这里只做 wait_event
                # 真正的 release 可以在 stream_mnt 中实现自动清理
            except Exception:
                pass

        h = self.norm(h)
        out = self.output(h).float()
        return out

    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        """
        保留原 forward 语义；当开关打开时，走 pipelined 包装（内部仍保证正确性）。
        """
        # 检查是否使用管线化模式
        use_pipe = getattr(self, "_runtime_switch", None)
        if use_pipe is None:
            # 首次调用：初始化运行时开关（默认开启管线化）
            self._runtime_switch = _RuntimeSwitch(PIPELINED=True)
            use_pipe = self._runtime_switch

        if use_pipe.PIPELINED:
            return self._forward_pipelined(tokens, start_pos)

        # === 原 forward 的安全版（保持现有逻辑；如需保留，可继续使用）===
        bsz, seqlen = tokens.shape

        # ⭐ 关键修复：使用 embed_tokens 的设备（已在 _verify_and_fix_device_placement 中确保在 CUDA）
        # 而不是 self.args.device（可能还是字符串 "cuda" 或被错误设置为 "cpu"）
        embed_dev = self.embed_tokens.weight.device
        if not embed_dev.type.startswith("cuda"):
            raise RuntimeError(
                f"embed_tokens.weight must be on CUDA, got {embed_dev}. "
                f"This should have been caught by _verify_and_fix_device_placement."
            )

        dev = embed_dev  # 使用实际的 CUDA device 对象，而不是字符串
        dtype = getattr(self, "param_dtype", torch.bfloat16)

        # 首次调用时打印设备信息（调试用）
        if not hasattr(self, '_device_info_printed'):
            print(f"[DEVICE] 计算设备: {dev}")
            print(f"[DEVICE] embed_tokens.weight.device: {self.embed_tokens.weight.device}")
            if len(self.layer_infos) > 0 and hasattr(self.layer_infos[0]['block'], 'attention'):
                first_attn = self.layer_infos[0]['block'].attention
                if hasattr(first_attn, 'wq') and hasattr(first_attn.wq, 'weight'):
                    print(f"[DEVICE] layer[0].attention.wq.device: {first_attn.wq.weight.device}")
            self._device_info_printed = True

        # ★ 设备一致性检查：input tokens 必须与 embed.weight 在同一设备
        # nn.Embedding/F.embedding 要求：input 和 weight 必须在同一设备（CPU 或同一块 GPU）
        if tokens.device != embed_dev:
            if tokens.device.type == "cpu" and not tokens.is_pinned():
                tokens = tokens.pin_memory()
            tokens = tokens.to(embed_dev, non_blocking=True)
        if tokens.dtype != torch.long:
            tokens = tokens.long()

        # 2) 执行 embedding 查表：此时 tokens 和 weight 已在同一设备
        h = self.embed_tokens(tokens)  # -> 与 embedding 同设备（必然是 CUDA）

        # 3) 如有不一致，迁移到目标计算设备 + 统一 dtype
        # ⭐ 安全网：即便 embedding 权重在 CPU，激活也会被立刻转回 CUDA，不会让后面崩掉
        # （但由于上面已经检查了 embed_dev 必须是 CUDA，这里理论上不会触发设备转换）
        if h.device != dev or h.dtype != dtype:
            # 打印调试信息（首次调用时）
            if not hasattr(self, '_device_warning_printed'):
                print(f"[DEBUG] Embedding 输出设备: {h.device}, 目标计算设备: {dev}")
                print(f"[DEBUG] embed_tokens.weight.device: {self.embed_tokens.weight.device}")
                print(f"[DEBUG] 正在将激活转换到目标设备...")
                self._device_warning_printed = True
            h = h.to(device=dev, dtype=dtype, non_blocking=True)

        # 2) freqs 只在设备不一致时搬一次，避免每步重复 .to()
        freqs_dev_key = str(dev)  # 用字符串作为缓存键
        if getattr(self, "_freqs_cached_dev", None) != freqs_dev_key:
            self._freqs_cached = self.freqs_complex.to(dev, non_blocking=True)
            self._freqs_cached_dev = freqs_dev_key
        freqs = self._freqs_cached

        # 3) 运行时防呆：任何层若把激活搬回 CPU 立即报错（早失败）
        for idx, info in enumerate(self.layer_infos):
            blk: EncoderBlock = info["block"]
            h = blk(h, start_pos, freqs)
            if h.device != dev:
                raise RuntimeError(
                    f"EncoderBlock[{idx}] returned activation on {h.device}, "
                    f"expected {dev}. Check for any '.to(\"cpu\")' fallback."
                )
            self.kv_times[idx]  = blk.attention.kv_elapsed_time
            self.attn_times[idx] = blk.attention.attn_time

        h = self.norm(h)                  # 确认 norm 模块常驻 GPU
        out = self.output(h).float()      # 输出用 float 以便后续 logits 运算
        return out
