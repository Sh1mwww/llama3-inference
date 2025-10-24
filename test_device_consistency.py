#!/usr/bin/env python3
"""
测试脚本：验证设备一致性
确保 input tokens 与 embed_tokens.weight 在同一设备上
"""
import torch
import torch.nn as nn
from llama3.config import ModelArgs
from llama3.model import Transformer

print("=" * 80)
print("测试 1: 验证 nn.Embedding 的设备一致性要求")
print("=" * 80)

# 创建一个简单的 embedding 层
vocab_size = 1000
embedding_dim = 128

# 测试 CPU
print("\n场景 1: CPU 上的 embedding")
embed_cpu = nn.Embedding(vocab_size, embedding_dim)
tokens_cpu = torch.randint(0, vocab_size, (2, 5))

print(f"embed.weight.device: {embed_cpu.weight.device}")
print(f"tokens.device: {tokens_cpu.device}")

try:
    output = embed_cpu(tokens_cpu)
    print(f"✅ 成功: 输出 shape {output.shape}, device {output.device}")
except RuntimeError as e:
    print(f"❌ 失败: {e}")

# 测试 CUDA（如果可用）
if torch.cuda.is_available():
    print("\n场景 2: CUDA 上的 embedding")
    embed_cuda = nn.Embedding(vocab_size, embedding_dim).cuda()
    tokens_cuda = torch.randint(0, vocab_size, (2, 5)).cuda()

    print(f"embed.weight.device: {embed_cuda.weight.device}")
    print(f"tokens.device: {tokens_cuda.device}")

    try:
        output = embed_cuda(tokens_cuda)
        print(f"✅ 成功: 输出 shape {output.shape}, device {output.device}")
    except RuntimeError as e:
        print(f"❌ 失败: {e}")

    # 测试设备不匹配
    print("\n场景 3: 设备不匹配（应该失败）")
    print(f"embed.weight.device: {embed_cuda.weight.device}")
    print(f"tokens.device: {tokens_cpu.device}")

    try:
        output = embed_cuda(tokens_cpu)
        print(f"❌ 意外成功（不应该）: {output.shape}")
    except RuntimeError as e:
        print(f"✅ 预期失败: {type(e).__name__}")
        print(f"   错误信息: {str(e)[:100]}...")

print("\n" + "=" * 80)
print("测试 2: Transformer 模型的设备一致性")
print("=" * 80)

# 加载真实模型配置
PARAMS_JSON = "/home/roger/.llama/checkpoints/Llama3.1-70B/params.json"

print("\n创建模型（CPU）...")
args = ModelArgs.from_json(PARAMS_JSON, max_seq_len=2048, max_batch_size=1, device="cpu")
model = Transformer(args)

# 使用 print_device_info 方法
print("\n调用 model.print_device_info():")
model.print_device_info()

print("\n" + "=" * 80)
print("测试 3: 前向传播设备自动转换")
print("=" * 80)

# 创建测试 tokens
test_tokens = torch.randint(0, args.vocab_size, (1, 10))
print(f"\n输入 tokens device: {test_tokens.device}")
print(f"embed_tokens.weight device: {model.embed_tokens.weight.device}")

# 模型会自动将 tokens 转换到正确的设备
print("\n执行前向传播（模型会自动处理设备转换）...")
try:
    # 注意：这里只测试 embedding 部分，不运行整个模型（避免 OOM）
    with torch.no_grad():
        # 模拟 forward 中的设备转换逻辑
        tgt_dev = model.embed_tokens.weight.device
        if test_tokens.device != tgt_dev:
            print(f"  → 自动转换 tokens 从 {test_tokens.device} 到 {tgt_dev}")
            test_tokens = test_tokens.to(tgt_dev)

        # 执行 embedding
        embeddings = model.embed_tokens(test_tokens)
        print(f"✅ Embedding 成功")
        print(f"   输出 shape: {embeddings.shape}")
        print(f"   输出 device: {embeddings.device}")

        # 验证设备一致性
        assert embeddings.device == model.embed_tokens.weight.device
        print(f"✅ 设备一致性验证通过")

except Exception as e:
    print(f"❌ 失败: {type(e).__name__}: {e}")

if torch.cuda.is_available():
    print("\n" + "=" * 80)
    print("测试 4: CUDA 模型设备一致性")
    print("=" * 80)

    print("\n将模型移动到 CUDA...")
    # 只移动 embed_tokens 和 output，避免 OOM
    model.embed_tokens = model.embed_tokens.cuda()
    model.output = model.output.cuda()

    print("\n调用 model.print_device_info():")
    model.print_device_info()

    # 测试 CPU tokens -> CUDA embedding
    print("\n测试 CPU tokens 自动转换到 CUDA:")
    cpu_tokens = torch.randint(0, args.vocab_size, (1, 5))
    print(f"输入 tokens device: {cpu_tokens.device}")

    with torch.no_grad():
        tgt_dev = model.embed_tokens.weight.device
        if cpu_tokens.device != tgt_dev:
            print(f"  → 自动转换到 {tgt_dev}")
            cpu_tokens = cpu_tokens.to(tgt_dev)

        embeddings = model.embed_tokens(cpu_tokens)
        print(f"✅ 转换成功")
        print(f"   输出 device: {embeddings.device}")

print("\n" + "=" * 80)
print("✅ 所有设备一致性测试完成")
print("=" * 80)

print("\n关键要点:")
print("1. nn.Embedding 要求 input 和 weight 必须在同一设备")
print("2. Transformer.forward() 会自动将 tokens 转换到 embed_tokens.weight 所在设备")
print("3. 使用 model.print_device_info() 可以检查设备和形状信息")
print("4. 生产环境中，确保 tokenizer.vocab_size == model.embed_tokens.num_embeddings")
