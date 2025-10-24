#!/usr/bin/env python3
"""
验证脚本：打印模型的设备和形状信息（按用户要求的格式）

使用方法:
    # 在你的代码中添加：
    from verify_model_device import verify_model_device
    verify_model_device(model, tokenizer)
"""
import json
from pathlib import Path

def verify_model_device(model, tokenizer=None):
    """
    验证并打印模型设备和形状信息

    Args:
        model: Transformer 模型实例
        tokenizer: 可选的 tokenizer 实例
    """
    print("\n" + "=" * 80)
    print("模型设备和形状验证")
    print("=" * 80)

    # 1. Tokenizer
    if tokenizer is not None:
        tokenizer_vocab = len(tokenizer)
        print(f"len(tokenizer):                     {tokenizer_vocab}")
    else:
        print("tokenizer:                          None (未提供)")

    # 2. Embedding 层
    print(f"model.embed_tokens.num_embeddings:  {model.embed_tokens.num_embeddings}")
    print(f"model.embed_tokens.weight.shape:    {model.embed_tokens.weight.shape}")
    print(f"model.embed_tokens.weight.device:   {model.embed_tokens.weight.device}")

    # 3. Output 层
    print(f"model.output.out_features:          {model.output.out_features}")
    print(f"model.output.weight.shape:          {model.output.weight.shape}")
    print(f"model.output.weight.device:         {model.output.weight.device}")

    # 4. 一致性检查
    print("\n" + "-" * 80)
    print("一致性检查:")
    print("-" * 80)

    checks_passed = True

    # vocab_size 一致性
    if tokenizer is not None:
        if tokenizer_vocab == model.embed_tokens.num_embeddings:
            print(f"✅ len(tokenizer) == embed_tokens.num_embeddings: {tokenizer_vocab}")
        else:
            print(f"❌ len(tokenizer) [{tokenizer_vocab}] != embed_tokens.num_embeddings [{model.embed_tokens.num_embeddings}]")
            checks_passed = False

    # embed vs output
    if model.embed_tokens.num_embeddings == model.output.out_features:
        print(f"✅ embed_tokens.num_embeddings == output.out_features: {model.embed_tokens.num_embeddings}")
    else:
        print(f"❌ embed_tokens.num_embeddings [{model.embed_tokens.num_embeddings}] != output.out_features [{model.output.out_features}]")
        checks_passed = False

    # 设备一致性
    embed_dev = model.embed_tokens.weight.device
    output_dev = model.output.weight.device
    if embed_dev == output_dev:
        print(f"✅ embed_tokens.device == output.device: {embed_dev}")
    else:
        print(f"❌ embed_tokens.device [{embed_dev}] != output.device [{output_dev}]")
        checks_passed = False

    # 形状检查
    expected_embed_shape = (model.embed_tokens.num_embeddings, model.embed_tokens.embedding_dim)
    expected_output_shape = (model.output.out_features, model.output.in_features)

    if tuple(model.embed_tokens.weight.shape) == expected_embed_shape:
        print(f"✅ embed_tokens.weight 形状正确: {model.embed_tokens.weight.shape}")
    else:
        print(f"❌ embed_tokens.weight 形状错误: 期望 {expected_embed_shape}, 实际 {model.embed_tokens.weight.shape}")
        checks_passed = False

    if tuple(model.output.weight.shape) == expected_output_shape:
        print(f"✅ output.weight 形状正确: {model.output.weight.shape}")
    else:
        print(f"❌ output.weight 形状错误: 期望 {expected_output_shape}, 实际 {model.output.weight.shape}")
        checks_passed = False

    print("=" * 80)

    if checks_passed:
        print("✅ 所有检查通过！模型可以安全使用")
    else:
        print("❌ 存在问题，请检查上述错误")

    print("=" * 80 + "\n")

    return checks_passed


# 独立运行时的测试
if __name__ == "__main__":
    print("独立测试模式：创建小规模模型验证")

    import torch
    import torch.nn as nn
    from llama3.config import ModelArgs
    from llama3.model import Transformer

    # 读取配置（但创建小规模模型）
    PARAMS_JSON = "/home/roger/.llama/checkpoints/Llama3.1-70B/params.json"

    with open(PARAMS_JSON) as f:
        params = json.load(f)

    print(f"\n从 params.json 读取配置:")
    print(f"  vocab_size: {params['vocab_size']}")
    print(f"  dim: {params['dim']}")

    # 创建轻量级 embedding 测试
    print("\n创建测试 embedding 层...")
    vocab_size = params['vocab_size']
    dim = 128  # 小规模用于测试

    class SimpleModel:
        def __init__(self):
            self.embed_tokens = nn.Embedding(vocab_size, dim)
            self.output = nn.Linear(dim, vocab_size, bias=False)

    model = SimpleModel()

    # 如果 CUDA 可用，移动到 GPU
    if torch.cuda.is_available():
        print("检测到 CUDA，移动模型到 GPU...")
        model.embed_tokens = model.embed_tokens.cuda()
        model.output = model.output.cuda()

    # 模拟 tokenizer
    class DummyTokenizer:
        def __len__(self):
            return vocab_size

    tokenizer = DummyTokenizer()

    # 调用验证函数
    verify_model_device(model, tokenizer)

    # 测试打印格式（与用户要求的一致）
    print("\n按照用户要求的打印格式:")
    print("-" * 80)
    print(f"len(tokenizer)                       # {len(tokenizer)}")
    print(f"model.embed_tokens.num_embeddings    # {model.embed_tokens.num_embeddings}")
    print(f"model.embed_tokens.weight.shape      # {model.embed_tokens.weight.shape}")
    print(f"model.embed_tokens.weight.device     # {model.embed_tokens.weight.device}")
    print("-" * 80)
