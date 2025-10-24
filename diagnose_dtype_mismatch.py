#!/usr/bin/env python3
"""
诊断 dtype 不匹配问题
检查模型参数的初始 dtype 和从 SSD 加载的权重 dtype
"""
import json
import torch
import torch.nn as nn
from pathlib import Path

def check_manifest_dtypes(manifest_path: str):
    """检查 manifest 中的权重 dtype"""
    print("=" * 60)
    print("检查 Manifest 中的权重 dtype")
    print("=" * 60)

    manifest = json.loads(Path(manifest_path).read_text())

    # 收集所有 dtype
    dtype_counts = {}
    sample_params = {}

    for layer_idx, params in manifest["layers_params"].items():
        for param_info in params:
            dtype = param_info["dtype"]
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

            # 保存第一个样本
            if dtype not in sample_params:
                sample_params[dtype] = param_info["name"]

    print(f"\nManifest 文件: {manifest_path}")
    print(f"总层数: {len(manifest['layers_params'])}")
    print(f"\nDtype 统计:")
    for dtype, count in sorted(dtype_counts.items()):
        print(f"  {dtype}: {count} 个参数")
        print(f"    示例: {sample_params[dtype]}")

    return dtype_counts, sample_params

def check_model_param_dtypes():
    """检查模型参数的默认 dtype"""
    print("\n" + "=" * 60)
    print("检查模型参数的默认 dtype")
    print("=" * 60)

    # 测试 nn.Linear 的默认 dtype
    print("\n测试 nn.Linear 默认 dtype:")
    test_linear = nn.Linear(10, 10, bias=False)
    print(f"  未指定 dtype: {test_linear.weight.dtype}")

    # 测试指定 device 但不指定 dtype
    test_linear_cpu = nn.Linear(10, 10, bias=False, device="cpu")
    print(f"  device='cpu': {test_linear_cpu.weight.dtype}")

    # 测试 meta device
    test_linear_meta = nn.Linear(10, 10, bias=False, device="meta")
    print(f"  device='meta': {test_linear_meta.weight.dtype}")

    # 测试 nn.Embedding
    print("\n测试 nn.Embedding 默认 dtype:")
    test_emb = nn.Embedding(100, 10)
    print(f"  未指定 dtype: {test_emb.weight.dtype}")

    # 测试 nn.Parameter
    print("\n测试 nn.Parameter 默认 dtype:")
    test_param = nn.Parameter(torch.ones(10))
    print(f"  torch.ones: {test_param.dtype}")

    test_param_empty = nn.Parameter(torch.empty(10))
    print(f"  torch.empty: {test_param_empty.dtype}")

def check_dtype_compatibility():
    """检查 dtype 之间的兼容性"""
    print("\n" + "=" * 60)
    print("检查 dtype 赋值兼容性")
    print("=" * 60)

    # 测试 float32 参数是否可以赋值 bfloat16 tensor
    print("\n测试 1: float32 param.data = bfloat16 tensor")
    try:
        param_f32 = nn.Parameter(torch.randn(3, 3, dtype=torch.float32))
        tensor_bf16 = torch.randn(3, 3, dtype=torch.bfloat16)
        param_f32.data = tensor_bf16
        print(f"  ✓ 成功! param dtype 变为: {param_f32.dtype}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")

    # 测试 meta device 参数
    print("\n测试 2: meta param.data = bfloat16 tensor")
    try:
        param_meta = nn.Parameter(torch.empty(3, 3, dtype=torch.float32, device="meta"))
        tensor_bf16 = torch.randn(3, 3, dtype=torch.bfloat16)
        param_meta.data = tensor_bf16
        print(f"  ✓ 成功! param dtype 变为: {param_meta.dtype}, device: {param_meta.device}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")

    # 测试 CUDA 设备上的类型转换
    if torch.cuda.is_available():
        print("\n测试 3: cuda float32 param.data = cuda bfloat16 tensor")
        try:
            param_cuda_f32 = nn.Parameter(torch.randn(3, 3, dtype=torch.float32, device="cuda"))
            tensor_cuda_bf16 = torch.randn(3, 3, dtype=torch.bfloat16, device="cuda")
            param_cuda_f32.data = tensor_cuda_bf16
            print(f"  ✓ 成功! param dtype 变为: {param_cuda_f32.dtype}")
        except Exception as e:
            print(f"  ✗ 失败: {e}")

        # 测试跨设备赋值
        print("\n测试 4: cpu float32 param.data = cuda bfloat16 tensor")
        try:
            param_cpu_f32 = nn.Parameter(torch.randn(3, 3, dtype=torch.float32))
            tensor_cuda_bf16 = torch.randn(3, 3, dtype=torch.bfloat16, device="cuda")
            param_cpu_f32.data = tensor_cuda_bf16
            print(f"  ✓ 成功! param dtype: {param_cpu_f32.dtype}, device: {param_cpu_f32.device}")
        except Exception as e:
            print(f"  ✗ 失败: {e}")

def check_actual_model():
    """检查实际模型的参数 dtype"""
    print("\n" + "=" * 60)
    print("检查实际模型的参数 dtype")
    print("=" * 60)

    try:
        from llama3.config import ModelArgs
        from llama3.layers import SelfAttention

        # 创建一个简单的 Attention 层
        args = ModelArgs(
            dim=8192,
            n_layers=80,
            n_heads=64,
            n_kv_heads=8,
            vocab_size=128256,
            multiple_of=256,
            ffn_dim_multiplier=1.3,
            norm_eps=1e-5,
            rope_theta=500000.0,
            use_scaled_rope=True,
            max_batch_size=1,
            max_seq_len=8192,
            device="cpu",
            param_init_device="meta"  # 使用 meta device 初始化
        )

        attn = SelfAttention(args)

        print(f"\nSelfAttention 参数 dtype:")
        print(f"  wq.weight: {attn.wq.weight.dtype}, device: {attn.wq.weight.device}")
        print(f"  wk.weight: {attn.wk.weight.dtype}, device: {attn.wk.weight.device}")
        print(f"  wv.weight: {attn.wv.weight.dtype}, device: {attn.wv.weight.device}")
        print(f"  wo.weight: {attn.wo.weight.dtype}, device: {attn.wo.weight.device}")

    except Exception as e:
        print(f"  ✗ 创建模型失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("dtype 不匹配诊断工具")
    print()

    # 1. 检查 manifest
    manifest_path = "/mnt/ssd/llama3_70b_ssd/ssd_weights_manifest.json"
    if Path(manifest_path).exists():
        check_manifest_dtypes(manifest_path)
    else:
        print(f"⚠ Manifest 文件不存在: {manifest_path}")

    # 2. 检查模型默认 dtype
    check_model_param_dtypes()

    # 3. 检查 dtype 兼容性
    check_dtype_compatibility()

    # 4. 检查实际模型
    check_actual_model()

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
