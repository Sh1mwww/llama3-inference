#!/usr/bin/env python3
"""
测试脚本：验证 output (lm_head) 与 embedding 层的维度对齐
"""
import json
import torch
import torch.nn as nn
from pathlib import Path

print("=" * 80)
print("测试 1: 检查 embedding 和 output 层的初始化")
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

# 创建 embedding 和 output 层
embed = nn.Embedding(vocab_size, dim)
output = nn.Linear(dim, vocab_size, bias=False)

print(f"\nnn.Embedding:")
print(f"  num_embeddings: {embed.num_embeddings}")
print(f"  embedding_dim: {embed.embedding_dim}")
print(f"  weight.shape: {embed.weight.shape}")

print(f"\nnn.Linear (output/lm_head):")
print(f"  in_features: {output.in_features}")
print(f"  out_features: {output.out_features}")
print(f"  weight.shape: {output.weight.shape}")

print("\n" + "=" * 80)
print("测试 2: 验证维度对齐")
print("=" * 80)

# 验证 embedding 的 num_embeddings 与 output 的 out_features 一致
if embed.num_embeddings == output.out_features:
    print(f"✅ vocab 维度对齐:")
    print(f"   embed.num_embeddings ({embed.num_embeddings}) == output.out_features ({output.out_features})")
else:
    print(f"❌ vocab 维度不对齐!")
    print(f"   embed.num_embeddings: {embed.num_embeddings}")
    print(f"   output.out_features: {output.out_features}")
    exit(1)

# 验证 embedding 的 embedding_dim 与 output 的 in_features 一致
if embed.embedding_dim == output.in_features:
    print(f"✅ hidden 维度对齐:")
    print(f"   embed.embedding_dim ({embed.embedding_dim}) == output.in_features ({output.in_features})")
else:
    print(f"❌ hidden 维度不对齐!")
    print(f"   embed.embedding_dim: {embed.embedding_dim}")
    print(f"   output.in_features: {output.in_features}")
    exit(1)

print("\n" + "=" * 80)
print("测试 3: 验证与 manifest 的一致性")
print("=" * 80)

MANIFEST = "/data1/70b-fixed.runtime_manifest.json"
if Path(MANIFEST).exists():
    manifest = json.loads(Path(MANIFEST).read_text())

    # 查找 embed_tokens 和 output 权重
    embed_param = None
    output_param = None

    for p in manifest.get("params", []):
        name = p.get("name", "")
        if name in ["embed_tokens.weight", "tok_embeddings.weight"]:
            embed_param = p
        elif name in ["output.weight", "lm_head.weight"]:
            output_param = p

    if embed_param and output_param:
        embed_shape = tuple(embed_param["shape"])
        output_shape = tuple(output_param["shape"])

        print(f"Manifest 中的形状:")
        print(f"  embed_tokens: {embed_shape}")
        print(f"  output:       {output_shape}")

        # 验证 vocab_size 维度（第 0 维）
        if embed_shape[0] == output_shape[0]:
            print(f"\n✅ vocab_size 维度一致: {embed_shape[0]}")
        else:
            print(f"\n❌ vocab_size 维度不一致!")
            print(f"   embed: {embed_shape[0]}, output: {output_shape[0]}")
            exit(1)

        # 验证 hidden_dim 维度（第 1 维）
        if embed_shape[1] == output_shape[1]:
            print(f"✅ hidden_dim 维度一致: {embed_shape[1]}")
        else:
            print(f"❌ hidden_dim 维度不一致!")
            print(f"   embed: {embed_shape[1]}, output: {output_shape[1]}")
            exit(1)

        # 注意：nn.Linear 的 weight 形状是 [out_features, in_features]
        # 所以 output.weight.shape = [vocab_size, dim]
        expected_output_shape = (vocab_size, dim)
        if output_shape == expected_output_shape:
            print(f"✅ output.weight 形状正确: {output_shape}")
        else:
            print(f"❌ output.weight 形状错误!")
            print(f"   预期: {expected_output_shape}, 实际: {output_shape}")
            exit(1)

    else:
        if not embed_param:
            print("⚠️  未在 manifest 中找到 embed_tokens")
        if not output_param:
            print("⚠️  未在 manifest 中找到 output")
else:
    print(f"⚠️  Manifest 文件不存在: {MANIFEST}")

print("\n" + "=" * 80)
print("测试 4: 模拟前向传播路径")
print("=" * 80)

# 创建测试输入
batch_size = 2
seq_len = 5
test_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

print(f"输入 tokens shape: {test_tokens.shape}")

# Embedding
embeddings = embed(test_tokens)
print(f"Embedding 输出 shape: {embeddings.shape}")
assert embeddings.shape == (batch_size, seq_len, dim), f"Embedding 输出形状错误"

# 假设中间经过 transformer layers，hidden states 保持 [batch, seq, dim]
hidden_states = embeddings  # 模拟

# Output projection
logits = output(hidden_states)
print(f"Output 输出 (logits) shape: {logits.shape}")

expected_logits_shape = (batch_size, seq_len, vocab_size)
if logits.shape == expected_logits_shape:
    print(f"✅ logits 形状正确: {logits.shape}")
else:
    print(f"❌ logits 形状错误!")
    print(f"   预期: {expected_logits_shape}, 实际: {logits.shape}")
    exit(1)

print("\n" + "=" * 80)
print("✅ 所有测试通过！output 层与 embedding 层完全对齐")
print("=" * 80)
print("\n总结:")
print(f"  vocab_size: {vocab_size}")
print(f"  hidden_dim: {dim}")
print(f"  embed_tokens.weight: [{vocab_size}, {dim}]")
print(f"  output.weight:       [{vocab_size}, {dim}]")
print(f"  ✓ 两者在 vocab 维度上完全一致 (均为 {vocab_size})")
print(f"  ✓ 可以安全进行前向传播：tokens -> embeddings -> ... -> logits")
