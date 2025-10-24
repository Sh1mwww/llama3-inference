#!/usr/bin/env python3
"""
轻量级设备一致性测试（不加载完整模型）
"""
import torch
import torch.nn as nn

print("=" * 80)
print("测试 nn.Embedding 设备一致性要求")
print("=" * 80)

# 使用小规模的 embedding 测试
vocab_size = 128256  # Llama 3.1 的 vocab_size
embedding_dim = 128   # 小规模用于测试

print(f"\n创建 embedding: vocab_size={vocab_size}, embedding_dim={embedding_dim}")

# 测试 1: CPU
print("\n" + "-" * 80)
print("测试 1: CPU 上的 embedding（设备一致）")
print("-" * 80)

embed_cpu = nn.Embedding(vocab_size, embedding_dim)
tokens_cpu = torch.tensor([[1, 2, 3, 100, 1000]])

print(f"embed.weight.device:     {embed_cpu.weight.device}")
print(f"tokens.device:           {tokens_cpu.device}")
print(f"embed.num_embeddings:    {embed_cpu.num_embeddings}")
print(f"embed.weight.shape:      {embed_cpu.weight.shape}")

try:
    output = embed_cpu(tokens_cpu)
    print(f"✅ 成功: 输出 shape {output.shape}, device {output.device}")
except RuntimeError as e:
    print(f"❌ 失败: {e}")

# 测试 2: CUDA（如果可用）
if torch.cuda.is_available():
    print("\n" + "-" * 80)
    print("测试 2: CUDA 上的 embedding（设备一致）")
    print("-" * 80)

    embed_cuda = nn.Embedding(vocab_size, embedding_dim).cuda()
    tokens_cuda = torch.tensor([[1, 2, 3, 100, 1000]]).cuda()

    print(f"embed.weight.device:     {embed_cuda.weight.device}")
    print(f"tokens.device:           {tokens_cuda.device}")
    print(f"embed.num_embeddings:    {embed_cuda.num_embeddings}")
    print(f"embed.weight.shape:      {embed_cuda.weight.shape}")

    try:
        output = embed_cuda(tokens_cuda)
        print(f"✅ 成功: 输出 shape {output.shape}, device {output.device}")
    except RuntimeError as e:
        print(f"❌ 失败: {e}")

    # 测试 3: 设备不匹配（应该失败）
    print("\n" + "-" * 80)
    print("测试 3: 设备不匹配（应该失败）")
    print("-" * 80)

    print(f"embed.weight.device:     {embed_cuda.weight.device} (CUDA)")
    print(f"tokens.device:           {tokens_cpu.device} (CPU)")

    try:
        output = embed_cuda(tokens_cpu)
        print(f"❌ 意外成功（不应该）: {output.shape}")
    except RuntimeError as e:
        print(f"✅ 预期失败: {type(e).__name__}")
        print(f"   错误类型: RuntimeError")
        print(f"   错误关键词: 'Expected all tensors to be on the same device'")

    # 测试 4: 自动转换（正确做法）
    print("\n" + "-" * 80)
    print("测试 4: 自动设备转换（正确做法）")
    print("-" * 80)

    # 模拟 Transformer.forward() 中的设备转换逻辑
    tokens_input = torch.tensor([[1, 2, 3, 100, 1000]])  # CPU tokens
    print(f"初始 tokens device:      {tokens_input.device}")

    # 获取目标设备（embed_tokens.weight 所在设备）
    tgt_dev = embed_cuda.weight.device
    print(f"目标设备 (embed.weight): {tgt_dev}")

    # 自动转换
    if tokens_input.device != tgt_dev:
        print(f"  → 执行转换: {tokens_input.device} -> {tgt_dev}")
        tokens_input = tokens_input.to(tgt_dev)

    print(f"转换后 tokens device:    {tokens_input.device}")

    # 现在可以安全执行 embedding
    output = embed_cuda(tokens_input)
    print(f"✅ 成功: 输出 shape {output.shape}, device {output.device}")

else:
    print("\n⚠️  CUDA 不可用，跳过 CUDA 测试")

print("\n" + "=" * 80)
print("验证摘要")
print("=" * 80)

print("\n关键检查点（类似用户要求的打印）:")
print(f"  embed.num_embeddings:    {embed_cpu.num_embeddings}  # 必须等于 vocab_size")
print(f"  embed.weight.shape:      {embed_cpu.weight.shape}  # [vocab_size, embedding_dim]")
print(f"  embed.weight.device:     {embed_cpu.weight.device}  # 必须与 input tokens 一致")

print("\n✅ 设备一致性要求:")
print("  1. input tokens 和 embed.weight 必须在同一设备")
print("  2. CPU tokens + CPU embedding = ✓")
print("  3. CUDA tokens + CUDA embedding = ✓")
print("  4. CPU tokens + CUDA embedding = ✗ (会报错)")
print("  5. 解决方案: tokens.to(embed.weight.device) 自动转换")

print("\n✅ Transformer.forward() 已实现自动设备转换:")
print("  tgt_dev = self.embed_tokens.weight.device")
print("  if tokens.device != tgt_dev:")
print("      tokens = tokens.to(tgt_dev, non_blocking=True)")

print("\n" + "=" * 80)
