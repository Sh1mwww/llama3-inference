#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流式合并 Meta 分片权重（低内存版本）
一次只加载和处理一层的权重，避免 OOM

用法:
  python merge_consolidated_streaming.py /home/roger/.llama/checkpoints/Llama3.1-70B /data1/70b.consolidated.pth
"""

import re
import os
import sys
import glob
import json
import torch
from pathlib import Path
from typing import Dict, Tuple, List
from collections import defaultdict

ATTN_TAG_RE = re.compile(r"\.attention\.(wq|wk|wv|wo)\.weight$")
FFN_TAG_RE = re.compile(r"\.feed_forward\.(w1|w2|w3)\.weight$")

def _hf_to_internal_name(name: str) -> str:
    n = name
    if n.startswith("model."):
        n = n[len("model."):]
    n = n.replace(".self_attn.q_proj.", ".attention.wq.")
    n = n.replace(".self_attn.k_proj.", ".attention.wk.")
    n = n.replace(".self_attn.v_proj.", ".attention.wv.")
    n = n.replace(".self_attn.o_proj.", ".attention.wo.")
    n = n.replace(".input_layernorm.", ".attention_norm.")
    n = n.replace(".post_attention_layernorm.", ".ffn_norm.")
    n = n.replace(".mlp.gate_proj.", ".feed_forward.w1.")
    n = n.replace(".mlp.up_proj.",   ".feed_forward.w3.")
    n = n.replace(".mlp.down_proj.", ".feed_forward.w2.")
    if n == "model.embed_tokens.weight":
        n = "embed_tokens.weight"
    if n == "lm_head.weight":
        n = "output.weight"
    return n

def _load_model_args_from_dir(root: Path) -> Dict[str, any]:
    for fname in ("params.json", "config.json"):
        p = root / fname
        if p.exists():
            js = json.loads(p.read_text(encoding="utf-8"))
            dim        = int(js.get("dim") or js.get("hidden_size"))
            n_heads    = int(js.get("n_heads") or js.get("num_attention_heads"))
            n_kv_heads = int(js.get("n_kv_heads") or js.get("num_key_value_heads") or n_heads)
            n_layers   = int(js.get("n_layers") or js.get("num_hidden_layers"))
            head_dim   = dim // n_heads
            ffn_mult   = js.get("ffn_dim_multiplier", 1.0)
            multiple_of = js.get("multiple_of", 256)
            return {
                "dim": dim,
                "n_heads": n_heads,
                "n_kv_heads": n_kv_heads,
                "n_layers": n_layers,
                "head_dim": head_dim,
                "ffn_dim_multiplier": ffn_mult,
                "multiple_of": multiple_of,
            }
    raise FileNotFoundError(f"params.json / config.json not found in {root}")

def _calc_ffn_intermediate_size(args: Dict) -> int:
    """计算 FFN intermediate_size"""
    dim = args["dim"]
    ffn_mult = args["ffn_dim_multiplier"]
    multiple_of = args["multiple_of"]

    hidden_dim = int(2 * (4 * dim) / 3)
    hidden_dim = int(ffn_mult * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim

def _index_all_params(shards: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    """
    扫描所有分片，建立参数索引：param_name -> [(shard_path, original_key), ...]
    不实际加载权重，只记录位置
    """
    print("📋 建立参数索引...")
    param_index = defaultdict(list)

    for shard_path in shards:
        print(f"  索引 {Path(shard_path).name}")
        sd = torch.load(shard_path, map_location="cpu", weights_only=False)
        for orig_key in sd.keys():
            if not isinstance(sd[orig_key], torch.Tensor):
                continue
            internal_name = _hf_to_internal_name(orig_key)
            param_index[internal_name].append((shard_path, orig_key))
        del sd  # 释放内存

    return dict(param_index)

def _concat_param_from_shards(param_name: str, locations: List[Tuple[str, str]],
                               args: Dict, is_attn: bool, is_ffn: bool) -> torch.Tensor:
    """
    从多个分片中加载并拼接参数
    """
    if len(locations) == 1:
        # 只有一个分片，直接加载
        shard_path, orig_key = locations[0]
        sd = torch.load(shard_path, map_location="cpu", weights_only=False)
        tensor = sd[orig_key].detach().cpu()
        del sd
        return tensor

    # 多个分片，需要拼接
    parts = []
    for shard_path, orig_key in locations:
        sd = torch.load(shard_path, map_location="cpu", weights_only=False)
        parts.append(sd[orig_key].detach().cpu())
        del sd

    # 确定拼接维度
    if is_attn:
        # Attention 权重拼接逻辑
        if ".wq.weight" in param_name or ".wo.weight" in param_name:
            # wq, wo 通常沿 dim=0 或 dim=1 拼接，取决于分片形状
            sample_shape = parts[0].shape
            if sample_shape[1] == args["dim"]:
                axis = 0  # (n_kv_heads*head_dim, dim) -> (dim, dim)
            else:
                axis = 1
        elif ".wk.weight" in param_name or ".wv.weight" in param_name:
            axis = 0  # (head_dim, dim) -> (n_kv_heads*head_dim, dim)
        else:
            axis = 0  # 默认

    elif is_ffn:
        # FFN 权重拼接逻辑
        if ".w1.weight" in param_name or ".w3.weight" in param_name:
            axis = 0  # (intermediate_size/n_shards, dim) -> (intermediate_size, dim)
        elif ".w2.weight" in param_name:
            axis = 1  # (dim, intermediate_size/n_shards) -> (dim, intermediate_size)
        else:
            axis = 0
    else:
        axis = 0  # 默认

    result = torch.cat(parts, dim=axis).contiguous()
    return result

def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()
    out_pth = Path(sys.argv[2]).resolve()

    if not root.is_dir():
        print(f"❌ Not a directory: {root}")
        sys.exit(1)

    # 加载模型配置
    args = _load_model_args_from_dir(root)
    intermediate_size = _calc_ffn_intermediate_size(args)

    print(f"📋 模型配置:")
    print(f"   dim={args['dim']} n_layers={args['n_layers']}")
    print(f"   n_heads={args['n_heads']} n_kv_heads={args['n_kv_heads']}")
    print(f"   ffn_intermediate_size={intermediate_size}")

    # 查找分片文件
    shards = sorted(glob.glob(str(root / "consolidated.*.pth")))
    if not shards:
        shards = [p for p in sorted(glob.glob(str(root / "*.pth")))
                  if Path(p).name != "consolidated.pth"]
    if not shards:
        print(f"❌ No consolidated.*.pth found in {root}")
        sys.exit(1)

    print(f"📦 找到 {len(shards)} 个分片")

    # 建立参数索引（不加载权重）
    param_index = _index_all_params(shards)

    print(f"\n🔄 开始流式合并（逐参数处理）...")
    merged = {}
    total_params = len(param_index)

    for i, (param_name, locations) in enumerate(param_index.items(), 1):
        if i % 50 == 0:
            print(f"  进度: {i}/{total_params} ({100*i//total_params}%)")

        # 判断参数类型
        is_attn = ATTN_TAG_RE.search(param_name) is not None
        is_ffn = FFN_TAG_RE.search(param_name) is not None

        # 加载并拼接（如果需要）
        tensor = _concat_param_from_shards(param_name, locations, args, is_attn, is_ffn)
        merged[param_name] = tensor

        # 验证关键权重的形状
        if ".w1.weight" in param_name and "layers.0" in param_name:
            expected = (intermediate_size, args["dim"])
            if tuple(tensor.shape) != expected:
                print(f"⚠️  警告: {param_name} 形状 {tuple(tensor.shape)} != 期望 {expected}")
            else:
                print(f"✅ {param_name}: {tuple(tensor.shape)} (正确)")

    # 保存
    print(f"\n💾 保存到 {out_pth}")
    out_pth.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, str(out_pth))

    # 验证
    print(f"\n📊 验证权重形状:")
    for L in [0, 1]:
        w1_key = f"layers.{L}.feed_forward.w1.weight"
        wq_key = f"layers.{L}.attention.wq.weight"
        if w1_key in merged:
            print(f"  Layer {L} w1: {tuple(merged[w1_key].shape)}")
        if wq_key in merged:
            print(f"  Layer {L} wq: {tuple(merged[wq_key].shape)}")

    print(f"\n✅ 合并完成！")

if __name__ == "__main__":
    main()
