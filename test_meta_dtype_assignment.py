#!/usr/bin/env python3
"""
精确测试 meta device 上的 dtype 赋值行为
"""
import torch
import torch.nn as nn

print("=" * 60)
print("测试 meta device 参数的 dtype 赋值")
print("=" * 60)

# 测试 1: meta float32 param.data = cpu bfloat16
print("\n测试 1: meta float32 param.data = cpu bfloat16")
try:
    param = nn.Parameter(torch.empty(3, 3, dtype=torch.float32, device="meta"))
    tensor = torch.randn(3, 3, dtype=torch.bfloat16, device="cpu")
    print(f"  Before: param.dtype={param.dtype}, param.device={param.device}")
    print(f"  Assigning: tensor.dtype={tensor.dtype}, tensor.device={tensor.device}")
    param.data = tensor
    print(f"  ✓ 成功!")
    print(f"  After: param.dtype={param.dtype}, param.device={param.device}")
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 测试 2: meta float32 param.data = cuda bfloat16
if torch.cuda.is_available():
    print("\n测试 2: meta float32 param.data = cuda bfloat16")
    try:
        param = nn.Parameter(torch.empty(3, 3, dtype=torch.float32, device="meta"))
        tensor = torch.randn(3, 3, dtype=torch.bfloat16, device="cuda")
        print(f"  Before: param.dtype={param.dtype}, param.device={param.device}")
        print(f"  Assigning: tensor.dtype={tensor.dtype}, tensor.device={tensor.device}")
        param.data = tensor
        print(f"  ✓ 成功!")
        print(f"  After: param.dtype={param.dtype}, param.device={param.device}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")

# 测试 3: meta bfloat16 param.data = cpu bfloat16
print("\n测试 3: meta bfloat16 param.data = cpu bfloat16")
try:
    param = nn.Parameter(torch.empty(3, 3, dtype=torch.bfloat16, device="meta"))
    tensor = torch.randn(3, 3, dtype=torch.bfloat16, device="cpu")
    print(f"  Before: param.dtype={param.dtype}, param.device={param.device}")
    print(f"  Assigning: tensor.dtype={tensor.dtype}, tensor.device={tensor.device}")
    param.data = tensor
    print(f"  ✓ 成功!")
    print(f"  After: param.dtype={param.dtype}, param.device={param.device}")
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 测试 4: meta bfloat16 param.data = cuda bfloat16
if torch.cuda.is_available():
    print("\n测试 4: meta bfloat16 param.data = cuda bfloat16")
    try:
        param = nn.Parameter(torch.empty(3, 3, dtype=torch.bfloat16, device="meta"))
        tensor = torch.randn(3, 3, dtype=torch.bfloat16, device="cuda")
        print(f"  Before: param.dtype={param.dtype}, param.device={param.device}")
        print(f"  Assigning: tensor.dtype={tensor.dtype}, tensor.device={tensor.device}")
        param.data = tensor
        print(f"  ✓ 成功!")
        print(f"  After: param.dtype={param.dtype}, param.device={param.device}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")

# 测试 5: 模拟 nn.Linear 的情况
print("\n测试 5: nn.Linear meta float32 param.data = cuda bfloat16")
if torch.cuda.is_available():
    try:
        linear = nn.Linear(10, 10, bias=False, device="meta")
        tensor = torch.randn(10, 10, dtype=torch.bfloat16, device="cuda")
        print(f"  Before: linear.weight.dtype={linear.weight.dtype}, linear.weight.device={linear.weight.device}")
        print(f"  Assigning: tensor.dtype={tensor.dtype}, tensor.device={tensor.device}")
        linear.weight.data = tensor
        print(f"  ✓ 成功!")
        print(f"  After: linear.weight.dtype={linear.weight.dtype}, linear.weight.device={linear.weight.device}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")

print("\n" + "=" * 60)
print("结论:")
print("=" * 60)
print("meta device 上的参数能否被不同 dtype 的 tensor 替换？")
print("需要根据上面的测试结果来判断")
