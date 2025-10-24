#!/usr/bin/env python3
"""
测试从 meta device 物化参数的正确方法
"""
import torch
import torch.nn as nn

print("=" * 60)
print("测试 meta device 参数物化的正确方法")
print("=" * 60)

# 方法 1: 使用 to_empty() + copy_()
print("\n方法 1: to_empty() + copy_()")
if torch.cuda.is_available():
    try:
        param = nn.Parameter(torch.empty(3, 3, dtype=torch.float32, device="meta"))
        tensor = torch.randn(3, 3, dtype=torch.bfloat16, device="cuda")
        print(f"  Before: param.dtype={param.dtype}, param.device={param.device}")
        print(f"  Target: tensor.dtype={tensor.dtype}, tensor.device={tensor.device}")

        # 先物化到目标设备和 dtype
        param.data = torch.empty_like(tensor, device=tensor.device, dtype=tensor.dtype)
        param.data.copy_(tensor)

        print(f"  ✓ 成功!")
        print(f"  After: param.dtype={param.dtype}, param.device={param.device}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()

# 方法 2: 直接使用 torch.empty + copy_
print("\n方法 2: torch.empty + copy_")
if torch.cuda.is_available():
    try:
        param = nn.Parameter(torch.empty(3, 3, dtype=torch.float32, device="meta"))
        tensor = torch.randn(3, 3, dtype=torch.bfloat16, device="cuda")
        print(f"  Before: param.dtype={param.dtype}, param.device={param.device}")
        print(f"  Target: tensor.dtype={tensor.dtype}, tensor.device={tensor.device}")

        # 创建一个新的空 tensor，然后copy
        new_data = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        new_data.copy_(tensor)
        param.data = new_data

        print(f"  ✓ 成功!")
        print(f"  After: param.dtype={param.dtype}, param.device={param.device}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()

# 方法 3: 先物化为空tensor再赋值
print("\n方法 3: 先创建空tensor再直接赋值")
if torch.cuda.is_available():
    try:
        param = nn.Parameter(torch.empty(3, 3, dtype=torch.float32, device="meta"))
        tensor = torch.randn(3, 3, dtype=torch.bfloat16, device="cuda")
        print(f"  Before: param.dtype={param.dtype}, param.device={param.device}")
        print(f"  Target: tensor.dtype={tensor.dtype}, tensor.device={tensor.device}")

        # 先让参数不再是 meta
        param.data = torch.empty(param.shape, dtype=torch.float32, device="cuda")
        # 然后再赋值（此时可以改变dtype）
        param.data = tensor

        print(f"  ✓ 成功!")
        print(f"  After: param.dtype={param.dtype}, param.device={param.device}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()

# 方法 4: 使用 module.to_empty()
print("\n方法 4: 使用 module.to_empty()")
if torch.cuda.is_available():
    try:
        linear = nn.Linear(10, 10, bias=False, device="meta")
        tensor = torch.randn(10, 10, dtype=torch.bfloat16, device="cuda")
        print(f"  Before: linear.weight.dtype={linear.weight.dtype}, linear.weight.device={linear.weight.device}")
        print(f"  Target: tensor.dtype={tensor.dtype}, tensor.device={tensor.device}")

        # 先物化模块
        linear = linear.to_empty(device="cuda")
        # 然后赋值（可能还需要调整 dtype）
        linear.weight.data = tensor

        print(f"  ✓ 成功!")
        print(f"  After: linear.weight.dtype={linear.weight.dtype}, linear.weight.device={linear.weight.device}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()

# 方法 5: 不改变参数对象，只替换 .data 为同 shape/dtype 的空 tensor 然后 copy
print("\n方法 5: 分两步 - 先物化为同dtype的空tensor，再copy")
if torch.cuda.is_available():
    try:
        param = nn.Parameter(torch.empty(3, 3, dtype=torch.float32, device="meta"))
        tensor = torch.randn(3, 3, dtype=torch.bfloat16, device="cuda")
        print(f"  Before: param.dtype={param.dtype}, param.device={param.device}")
        print(f"  Target: tensor.dtype={tensor.dtype}, tensor.device={tensor.device}")

        # 先物化为目标dtype和device的空tensor
        materialized = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        # copy数据
        materialized.copy_(tensor)
        # 赋值
        param.data = materialized

        print(f"  ✓ 成功!")
        print(f"  After: param.dtype={param.dtype}, param.device={param.device}")
        print(f"  Data matches: {torch.allclose(param.data, tensor, rtol=1e-3)}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("结论")
print("=" * 60)
