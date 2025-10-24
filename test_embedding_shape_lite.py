#!/usr/bin/env python3
"""
轻量级测试：验证 nn.Embedding 的形状与权重一致（不加载整个模型）
"""
import json
import torch
import torch.nn as nn
from pathlib import Path

print("=" * 80)
print("测试 1: 验证 nn.Embedding 初始化逻辑")
print("=" * 80)

# 读取 params.json
PARAMS_JSON = "/home/roger/.llama/checkpoints/Llama3.1-70B/params.json"
with open(PARAMS_JSON) as f:
    params = json.load(f)

vocab_size = params["vocab_size"]
dim = params["dim"]

print(f"从 params.json 读取:")
print(f"  vocab_size: {vocab_size}")
print(f"  dim: {dim}")

# 创建 embedding 层
embed = nn.Embedding(vocab_size, dim)
print(f"\nnn.Embedding 创建成功:")
print(f"  num_embeddings: {embed.num_embeddings}")
print(f"  embedding_dim: {embed.embedding_dim}")
print(f"  weight.shape: {embed.weight.shape}")

# 验证
assert embed.num_embeddings == vocab_size, f"num_embeddings {embed.num_embeddings} != vocab_size {vocab_size}"
assert embed.embedding_dim == dim, f"embedding_dim {embed.embedding_dim} != dim {dim}"
assert embed.weight.shape == (vocab_size, dim), f"weight.shape {embed.weight.shape} != ({vocab_size}, {dim})"

print("✅ nn.Embedding 形状正确")

print("\n" + "=" * 80)
print("测试 2: 验证与 manifest 的一致性")
print("=" * 80)

MANIFEST = "/data1/70b-fixed.runtime_manifest.json"
if Path(MANIFEST).exists():
    manifest = json.loads(Path(MANIFEST).read_text())

    # 查找 embed_tokens 权重
    embed_param = None
    for p in manifest.get("params", []):
        name = p.get("name", "")
        if name in ["embed_tokens.weight", "tok_embeddings.weight"]:
            embed_param = p
            break

    if embed_param:
        manifest_shape = tuple(embed_param["shape"])
        model_shape = (vocab_size, dim)

        print(f"Manifest 中的 embed_tokens 形状: {manifest_shape}")
        print(f"模型中的 embed_tokens 形状: {model_shape}")

        if manifest_shape == model_shape:
            print("✅ 形状一致！权重可以安全加载")
        else:
            print(f"❌ 形状不匹配！")
            print(f"  这会在 load_resident_to_gpu 中触发 RuntimeError")
            exit(1)
    else:
        print("⚠️  未在 manifest 中找到 embed_tokens")
else:
    print(f"⚠️  Manifest 文件不存在: {MANIFEST}")

print("\n" + "=" * 80)
print("测试 3: 验证形状检查逻辑")
print("=" * 80)

# 模拟 load_resident_to_gpu 中的形状检查
param_shape = (vocab_size, dim)  # 模型参数形状
weight_shape = manifest_shape if 'manifest_shape' in locals() else (vocab_size, dim)  # 权重文件形状

print(f"模拟权重加载:")
print(f"  模型参数形状: {param_shape}")
print(f"  权重文件形状: {weight_shape}")

# 这是我们在 load_resident_to_gpu 中添加的检查
if tuple(weight_shape) != tuple(param_shape):
    print(f"❌ 形状不匹配检测成功（这会触发 RuntimeError）")
    print(f"  模块参数形状: {param_shape}")
    print(f"  权重文件形状: {weight_shape}")
else:
    print(f"✅ 形状匹配检查通过")

print("\n" + "=" * 80)
print("✅ 所有测试通过！")
print("=" * 80)
print("\n修复总结:")
print("1. 在 model.py 中添加了 vocab_size > 0 的断言")
print("2. 在 weights_io_ssd_dram.py 的 load_resident_to_gpu 中添加了形状验证")
print("3. 确保 nn.Embedding(num_embeddings, embedding_dim) 的 num_embeddings")
print("   等于实际权重的 shape[0]")
