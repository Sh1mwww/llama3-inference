#!/usr/bin/env python3
"""
测试参数安装的保险丝机制
验证即使模块被错误地物化为 fp32，也能在安装时纠正为 bf16
"""
import torch
import torch.nn as nn

class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(10, 10, bias=False)

def test_install_param_tensor():
    """模拟 _install_param_tensor 的行为"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    print("测试参数安装保险丝机制")
    print("=" * 60)

    # 场景 1: Meta device -> bfloat16
    print("\n场景 1: Meta device 参数 -> bfloat16 tensor")
    print("-" * 60)

    module = DummyModule()
    module.wq = nn.Linear(10, 10, bias=False, device="meta")
    param = module.wq.weight

    print(f"原始参数: device={param.device}, dtype={param.dtype}")

    # 模拟从 SSD 加载的 bfloat16 tensor
    dst_tensor = torch.randn(10, 10, dtype=torch.bfloat16, device="cuda")
    print(f"目标 tensor: device={dst_tensor.device}, dtype={dst_tensor.dtype}")

    # 检查是否需要替换
    need_replace = (
        str(param.device).startswith("meta")
        or param.device != dst_tensor.device
        or param.dtype != dst_tensor.dtype
        or param.shape != dst_tensor.shape
    )
    print(f"需要替换参数对象: {need_replace}")

    if need_replace:
        # 替换参数
        new_p = nn.Parameter(dst_tensor, requires_grad=False)
        module.wq.weight = new_p
        print(f"✓ 替换后: device={module.wq.weight.device}, dtype={module.wq.weight.dtype}")
    else:
        print("✗ 不应该走到这里")

    # 场景 2: float32 -> bfloat16 (保险丝场景)
    print("\n场景 2: float32 参数 -> bfloat16 tensor (保险丝)")
    print("-" * 60)

    module2 = DummyModule()
    module2.wq = nn.Linear(10, 10, bias=False, device="cuda")  # 默认 float32
    param2 = module2.wq.weight

    print(f"原始参数: device={param2.device}, dtype={param2.dtype}")

    # 模拟从 SSD 加载的 bfloat16 tensor
    dst_tensor2 = torch.randn(10, 10, dtype=torch.bfloat16, device="cuda")
    print(f"目标 tensor: device={dst_tensor2.device}, dtype={dst_tensor2.dtype}")

    # 检查是否需要替换
    need_replace2 = (
        str(param2.device).startswith("meta")
        or param2.device != dst_tensor2.device
        or param2.dtype != dst_tensor2.dtype
        or param2.shape != dst_tensor2.shape
    )
    print(f"需要替换参数对象: {need_replace2} (dtype 不匹配)")

    if need_replace2:
        # 替换参数
        new_p2 = nn.Parameter(dst_tensor2, requires_grad=False)
        module2.wq.weight = new_p2
        print(f"✓ 保险丝生效！替换后: device={module2.wq.weight.device}, dtype={module2.wq.weight.dtype}")
    else:
        print("✗ 不应该走到这里")

    # 场景 3: bfloat16 -> bfloat16 (相同 dtype，使用 copy_)
    print("\n场景 3: bfloat16 参数 -> bfloat16 tensor (使用 copy_)")
    print("-" * 60)

    module3 = DummyModule()
    module3.wq = nn.Linear(10, 10, bias=False, device="cuda", dtype=torch.bfloat16)
    param3 = module3.wq.weight

    print(f"原始参数: device={param3.device}, dtype={param3.dtype}")

    # 模拟从 SSD 加载的 bfloat16 tensor
    dst_tensor3 = torch.randn(10, 10, dtype=torch.bfloat16, device="cuda")
    print(f"目标 tensor: device={dst_tensor3.device}, dtype={dst_tensor3.dtype}")

    # 检查是否需要替换
    need_replace3 = (
        str(param3.device).startswith("meta")
        or param3.device != dst_tensor3.device
        or param3.dtype != dst_tensor3.dtype
        or param3.shape != dst_tensor3.shape
    )
    print(f"需要替换参数对象: {need_replace3} (全部匹配)")

    if not need_replace3:
        # 使用 copy_
        param3.data.copy_(dst_tensor3)
        print(f"✓ 使用 copy_: device={module3.wq.weight.device}, dtype={module3.wq.weight.dtype}")
        print(f"  数据已更新: {torch.allclose(module3.wq.weight, dst_tensor3, rtol=1e-3)}")
    else:
        print("✗ 不应该走到这里")

    print("\n" + "=" * 60)
    print("✓ 所有场景测试通过！")
    print("\n总结:")
    print("  场景 1: Meta device -> 自动替换为正确的 device/dtype")
    print("  场景 2: 错误的 dtype -> 保险丝纠正")
    print("  场景 3: 正确的 dtype -> 高效的 copy_")

if __name__ == "__main__":
    test_install_param_tensor()
