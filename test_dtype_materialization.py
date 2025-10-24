#!/usr/bin/env python3
"""
测试使用正确 dtype 物化模块
"""
import torch
import torch.nn as nn

def materialize_module_from_meta(module, example_tensor, device):
    """
    将 module 从 meta 物化到 GPU，dtype 与 example_tensor 保持一致。
    """
    target_device = device
    target_dtype = example_tensor.dtype

    print(f"  Materializing to device={target_device}, dtype={target_dtype}")

    # 优先走 to_empty
    try:
        module.to_empty(device=target_device, dtype=target_dtype)
        return
    except Exception as e:
        print(f"  to_empty failed ({e}), using fallback")

    # 兜底：逐参数替换
    for name, p in list(module.named_parameters(recurse=False)):
        if getattr(p, "is_meta", False) or str(p.device).startswith("meta"):
            new_p = nn.Parameter(
                torch.empty_like(p, device=target_device, dtype=target_dtype),
                requires_grad=False
            )
            setattr(module, name, new_p)

if torch.cuda.is_available():
    print("测试正确 dtype 的模块物化")
    print("=" * 60)

    # 创建 meta device 上的模块
    linear = nn.Linear(10, 10, bias=False, device="meta")
    print(f"\n初始状态:")
    print(f"  linear.weight.device: {linear.weight.device}")
    print(f"  linear.weight.dtype: {linear.weight.dtype}")

    # 模拟从 SSD 加载的 bfloat16 权重
    src_tensor = torch.randn(10, 10, dtype=torch.bfloat16)
    print(f"\n源权重:")
    print(f"  src_tensor.dtype: {src_tensor.dtype}")

    # 物化模块
    print(f"\n物化模块...")
    materialize_module_from_meta(linear, src_tensor, "cuda")

    print(f"\n物化后:")
    print(f"  linear.weight.device: {linear.weight.device}")
    print(f"  linear.weight.dtype: {linear.weight.dtype}")

    # 测试赋值
    print(f"\n测试赋值 bfloat16 tensor...")
    dst_tensor = src_tensor.to("cuda")
    print(f"  dst_tensor.dtype: {dst_tensor.dtype}, device: {dst_tensor.device}")

    try:
        linear.weight.data = dst_tensor
        print(f"  ✓ 赋值成功!")
        print(f"  最终 dtype: {linear.weight.dtype}")
        print(f"  最终 device: {linear.weight.device}")
    except Exception as e:
        print(f"  ✗ 赋值失败: {e}")

    print("\n" + "=" * 60)
    print("✓ 测试完成")

else:
    print("CUDA not available")
