#!/usr/bin/env python3
"""
测试脚本：验证 nn.Embedding 的形状与权重一致
"""
import json
import torch
from pathlib import Path
from llama3.model import Transformer
from llama3.config import ModelArgs

print("=" * 80)
print("测试 1: 检查模型初始化时 embed_tokens 的形状")
print("=" * 80)

# 测试 70B 模型
PARAMS_JSON = "/home/roger/.llama/checkpoints/Llama3.1-70B/params.json"

args = ModelArgs.from_json(PARAMS_JSON, max_seq_len=8192, max_batch_size=1, device="cpu")
print(f"ModelArgs.vocab_size: {args.vocab_size}")
print(f"ModelArgs.dim: {args.dim}")

model = Transformer(args)
print(f"\nembed_tokens.num_embeddings: {model.embed_tokens.num_embeddings}")
print(f"embed_tokens.embedding_dim: {model.embed_tokens.embedding_dim}")
print(f"embed_tokens.weight.shape: {model.embed_tokens.weight.shape}")

# 验证形状一致性
expected_shape = (args.vocab_size, args.dim)
actual_shape = tuple(model.embed_tokens.weight.shape)

print(f"\n预期形状: {expected_shape}")
print(f"实际形状: {actual_shape}")

if expected_shape == actual_shape:
    print("✅ 形状一致！")
else:
    print(f"❌ 形状不一致！预期 {expected_shape}，实际 {actual_shape}")
    exit(1)

# 验证 num_embeddings 与 weight.shape[0] 一致
if model.embed_tokens.num_embeddings == model.embed_tokens.weight.shape[0]:
    print(f"✅ num_embeddings ({model.embed_tokens.num_embeddings}) == weight.shape[0] ({model.embed_tokens.weight.shape[0]})")
else:
    print(f"❌ num_embeddings ({model.embed_tokens.num_embeddings}) != weight.shape[0] ({model.embed_tokens.weight.shape[0]})")
    exit(1)

print("\n" + "=" * 80)
print("测试 2: 模拟权重加载时的形状验证")
print("=" * 80)

# 读取 manifest 中的 embed_tokens 形状
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
        print(f"Manifest 中的形状: {manifest_shape}")
        print(f"模型参数的形状: {actual_shape}")

        if manifest_shape == actual_shape:
            print("✅ Manifest 形状与模型参数一致！")
        else:
            print(f"❌ 形状不匹配！Manifest: {manifest_shape}, 模型: {actual_shape}")
            print("\n这会在 load_resident_to_gpu 中触发 RuntimeError")
            exit(1)
    else:
        print("⚠️  未在 manifest 中找到 embed_tokens")
else:
    print(f"⚠️  Manifest 文件不存在: {MANIFEST}")

print("\n" + "=" * 80)
print("测试 3: 测试前向传播（确保查表不会越界）")
print("=" * 80)

# 创建一些测试 token IDs
test_tokens = torch.tensor([[1, 2, 3, 100, 1000]], dtype=torch.long)
print(f"测试 tokens: {test_tokens}")
print(f"最大 token ID: {test_tokens.max().item()}")
print(f"vocab_size: {model.embed_tokens.num_embeddings}")

if test_tokens.max().item() < model.embed_tokens.num_embeddings:
    print("✅ Token IDs 在有效范围内")

    # 执行 embedding 查表
    try:
        embeddings = model.embed_tokens(test_tokens)
        print(f"Embedding 输出形状: {embeddings.shape}")
        expected_output_shape = (test_tokens.shape[0], test_tokens.shape[1], model.embed_tokens.embedding_dim)
        if tuple(embeddings.shape) == expected_output_shape:
            print(f"✅ 输出形状正确: {embeddings.shape}")
        else:
            print(f"❌ 输出形状错误！预期 {expected_output_shape}，实际 {embeddings.shape}")
            exit(1)
    except Exception as e:
        print(f"❌ Embedding 查表失败: {e}")
        exit(1)
else:
    print(f"❌ Token IDs 超出范围！最大 ID {test_tokens.max().item()} >= vocab_size {model.embed_tokens.num_embeddings}")
    exit(1)

print("\n" + "=" * 80)
print("✅ 所有测试通过！nn.Embedding 形状与权重一致")
print("=" * 80)
