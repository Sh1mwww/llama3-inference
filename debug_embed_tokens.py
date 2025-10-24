#!/usr/bin/env python3
"""
诊断脚本：检查 embed_tokens 在各个阶段的设备位置
"""
import json
import torch
from pathlib import Path
from llama3.model import Transformer
from llama3.config import ModelArgs

# 配置
MANIFEST = "/data1/70b-fixed.runtime_manifest.json"
PARAMS_JSON = "/home/roger/.llama/checkpoints/Llama3.1-70B/params.json"
DEVICE = "cuda:0"

print("=" * 80)
print("Stage 1: 创建 CPU 模型骨架")
print("=" * 80)

cpu_args = ModelArgs.from_json(PARAMS_JSON, max_seq_len=8192, max_batch_size=1, device="cpu")
setattr(cpu_args, "use_stub_params", True)
setattr(cpu_args, "weight_source", "raw-ssd")

model = Transformer(cpu_args)
print(f"embed_tokens.weight device: {model.embed_tokens.weight.device}")
print(f"embed_tokens.weight shape: {model.embed_tokens.weight.shape}")
print(f"embed_tokens.weight dtype: {model.embed_tokens.weight.dtype}")
print(f"embed_tokens.weight numel: {model.embed_tokens.weight.numel()}")
print(f"embed_tokens.weight mean: {model.embed_tokens.weight.mean().item():.6f}")
print(f"embed_tokens.weight std: {model.embed_tokens.weight.std().item():.6f}")

embed_before_load = model.embed_tokens.weight.data.clone()

print("\n" + "=" * 80)
print("Stage 2: 手动调用 load_resident_to_gpu")
print("=" * 80)

from llama3.weights_io_ssd_dram import load_resident_to_gpu

manifest = json.loads(Path(MANIFEST).read_text())
load_resident_to_gpu(model, manifest, device=DEVICE)

print(f"embed_tokens.weight device: {model.embed_tokens.weight.device}")
print(f"embed_tokens.weight shape: {model.embed_tokens.weight.shape}")
print(f"embed_tokens.weight dtype: {model.embed_tokens.weight.dtype}")
print(f"embed_tokens.weight mean: {model.embed_tokens.weight.mean().item():.6f}")
print(f"embed_tokens.weight std: {model.embed_tokens.weight.std().item():.6f}")

# 检查权重是否真的改变了（shape/dtype 都变了，肯定变了）
print(f"权重形状是否改变: {embed_before_load.shape} -> {model.embed_tokens.weight.shape}")
print(f"权重dtype是否改变: {embed_before_load.dtype} -> {model.embed_tokens.weight.dtype}")
weights_changed = (embed_before_load.shape != model.embed_tokens.weight.shape or
                  embed_before_load.dtype != model.embed_tokens.weight.dtype)
print(f"权重是否改变: {weights_changed}")

# 保存 hash 值和统计信息，而不是整个张量（避免 OOM）
embed_after_load_mean = model.embed_tokens.weight.mean().item()
embed_after_load_std = model.embed_tokens.weight.std().item()
embed_after_load_ptr = model.embed_tokens.weight.data_ptr()

print(f"embed_tokens.num_embeddings: {model.embed_tokens.num_embeddings}")
print(f"embed_tokens.embedding_dim: {model.embed_tokens.embedding_dim}")

print("\n" + "=" * 80)
print("Stage 3: 调用 embed_tokens.to(device)")
print("=" * 80)

model.embed_tokens = model.embed_tokens.to(DEVICE)

print(f"embed_tokens.weight device: {model.embed_tokens.weight.device}")
print(f"embed_tokens.weight shape: {model.embed_tokens.weight.shape}")
print(f"embed_tokens.weight dtype: {model.embed_tokens.weight.dtype}")
print(f"embed_tokens.weight mean: {model.embed_tokens.weight.mean().item():.6f}")
print(f"embed_tokens.weight std: {model.embed_tokens.weight.std().item():.6f}")

# 检查权重是否又改变了（通过统计值和指针）
new_mean = model.embed_tokens.weight.mean().item()
new_std = model.embed_tokens.weight.std().item()
new_ptr = model.embed_tokens.weight.data_ptr()

print(f"mean 变化: {embed_after_load_mean:.6f} -> {new_mean:.6f}")
print(f"std 变化: {embed_after_load_std:.6f} -> {new_std:.6f}")
print(f"data_ptr 变化: {embed_after_load_ptr} -> {new_ptr}")

weights_changed_again = (abs(embed_after_load_mean - new_mean) > 1e-5 or
                        abs(embed_after_load_std - new_std) > 1e-5 or
                        embed_after_load_ptr != new_ptr)
print(f"权重是否再次改变: {weights_changed_again}")

if weights_changed_again:
    print("⚠️  警告：embed_tokens.to(device) 改变了权重！")
    print("   这就是 bug 的根源！")
else:
    print("✅ embed_tokens.to(device) 没有改变权重")
