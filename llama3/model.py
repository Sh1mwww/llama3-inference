from typing import List, Dict, Any   

import torch
import torch.nn as nn

from .config import ModelArgs,  LayerInfo
from .layers import (
    RMSNorm,
    EncoderBlock,
    precompute_theta_pos_frequencies,
)


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

    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        bsz, seqlen = tokens.shape

        if hasattr(self.args, 'device') and self.args.device and not self.args.device.startswith('meta'):
            # 优先使用配置的设备
            dev = torch.device(self.args.device)
        elif len(self.layer_infos) > 0:
            # 从第一个 layer 的任意参数推断计算设备（适用于 streaming）
            first_layer = self.layer_infos[0]["block"]
            # 尝试从 attention.wq 获取设备
            if hasattr(first_layer, 'attention') and hasattr(first_layer.attention, 'wq'):
                if hasattr(first_layer.attention.wq, 'weight'):
                    dev = first_layer.attention.wq.weight.device
                else:
                    # meta 或未初始化，回退到 embed_tokens
                    dev = self.embed_tokens.weight.device
            else:
                dev = self.embed_tokens.weight.device
        else:
            # 回退到 embed_tokens
            dev = self.embed_tokens.weight.device

        dtype = getattr(self, "param_dtype", torch.bfloat16)   # ★ 统一的参数/激活 dtype

        # 首次调用时打印设备信息（调试用）
        if not hasattr(self, '_device_info_printed'):
            print(f"[DEVICE] 计算设备: {dev}")
            print(f"[DEVICE] embed_tokens.weight.device: {self.embed_tokens.weight.device}")
            if len(self.layer_infos) > 0 and hasattr(self.layer_infos[0]['block'], 'attention'):
                first_attn = self.layer_infos[0]['block'].attention
                if hasattr(first_attn, 'wq') and hasattr(first_attn.wq, 'weight'):
                    print(f"[DEVICE] layer[0].attention.wq.device: {first_attn.wq.weight.device}")
            self._device_info_printed = True

        # ★ 处理 embed_tokens：它可能在 CPU（resident）或 GPU
        # tokens 必须先转到 embed_tokens.weight 所在设备（满足 nn.Embedding 要求）
        embed_dev = self.embed_tokens.weight.device
        if embed_dev.type == "cuda":
            dev = embed_dev
        else:
            # 退化到 args.device 或第一层可用权重的设备
            dev = torch.device(str(getattr(self.args, "device", embed_dev)))
            
        dtype = getattr(self, "param_dtype", torch.bfloat16)
        print(f"[DEVICE] 计算设备: {dev}")
        print(f"[DEVICE] embed_tokens.weight.device: {self.embed_tokens.weight.device}")
        try:
            print(f"[DEVICE] layer[0].attention.wq.device: {self.layers[0].attention.wq.weight.device}")
        except Exception:
            pass

        # ★ 设备一致性检查：input tokens 必须与 embed.weight 在同一设备
        # nn.Embedding/F.embedding 要求：input 和 weight 必须在同一设备（CPU 或同一块 GPU）
        if tokens.device != embed_dev:
            if tokens.device.type == "cpu" and not tokens.is_pinned():
                tokens = tokens.pin_memory()
            tokens = tokens.to(embed_dev, non_blocking=True)
        if tokens.dtype != torch.long:
            tokens = tokens.long()

        # 2) 执行 embedding 查表：此时 tokens 和 weight 已在同一设备
        h = self.embed_tokens(tokens)  # -> 与 embedding 同设备

        # 3) 验证 embedding 输出设备并确保在正确设备上
        # 注意：h 的设备取决于 embed_tokens.weight 的设备，而 dev 是我们期望的计算设备
        # 如果 embed_tokens 在 CPU，但我们需要在 GPU 上计算，必须转换
        if h.device != dev or h.dtype != dtype:
            # 打印调试信息（首次调用时）
            if not hasattr(self, '_device_warning_printed'):
                print(f"[DEBUG] Embedding 输出设备: {h.device}, 目标计算设备: {dev}")
                print(f"[DEBUG] embed_tokens.weight.device: {self.embed_tokens.weight.device}")
                print(f"[DEBUG] 正在将激活转换到目标设备...")
                self._device_warning_printed = True
            h = h.to(device=dev, dtype=dtype, non_blocking=True)

        # 2) freqs 只在设备不一致时搬一次，避免每步重复 .to()
        if getattr(self, "_freqs_cached_dev", None) != dev:
            self._freqs_cached = self.freqs_complex.to(dev, non_blocking=True)
            self._freqs_cached_dev = dev
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
