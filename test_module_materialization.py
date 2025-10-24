#!/usr/bin/env python3
"""
测试模块物化逻辑
"""
import torch
import torch.nn as nn

# 创建一个在 meta device 上的模块
class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(10, 10, bias=False, device="meta")
        self.wk = nn.Linear(10, 10, bias=False, device="meta")

class TestBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = TestModule()

if torch.cuda.is_available():
    print("测试模块物化")
    print("=" * 60)

    # 创建block
    block = TestBlock()

    # 检查初始状态
    print(f"\n初始状态:")
    print(f"  wq.weight device: {block.attention.wq.weight.device}")
    print(f"  wq.weight dtype: {block.attention.wq.weight.dtype}")

    # 模拟参数名
    pname = "layers.0.attention.wq.weight"
    parts = pname.split('.')
    layer_idx = 0
    module_path = '.'.join(parts[2:-1])  # "attention.wq"

    print(f"\n解析参数名:")
    print(f"  pname: {pname}")
    print(f"  module_path: {module_path}")

    # 获取目标模块
    target_module = block.attention
    for attr in module_path.split('.')[1:]:  # Skip 'attention', just get 'wq'
        target_module = getattr(target_module, attr, None)
        print(f"  Getting attr '{attr}': {target_module}")

    print(f"\n目标模块: {target_module}")
    print(f"  模块类型: {type(target_module)}")
    print(f"  参数device: {next(target_module.parameters()).device}")

    # 物化模块
    print(f"\n物化到 cuda...")
    materialized = target_module.to_empty(device="cuda")
    print(f"  物化后device: {materialized.weight.device}")
    print(f"  物化后dtype: {materialized.weight.dtype}")

    # 更新模块引用
    parent = block  # 从 block 开始，不是 block.attention
    attrs = module_path.split('.')
    print(f"\n更新模块引用:")
    print(f"  attrs: {attrs}")
    print(f"  Starting from: {parent}")
    for attr in attrs[:-1]:
        print(f"    parent = parent.{attr}")
        parent = getattr(parent, attr)
    print(f"  Final parent: {parent}")
    print(f"  Setting {attrs[-1]} on parent")
    setattr(parent, attrs[-1], materialized)

    # 验证
    print(f"\n验证:")
    print(f"  block.attention.wq.weight.device: {block.attention.wq.weight.device}")
    print(f"  block.attention.wq.weight.dtype: {block.attention.wq.weight.dtype}")

    # 测试赋值
    print(f"\n测试赋值bfloat16 tensor:")
    tensor = torch.randn(10, 10, dtype=torch.bfloat16, device="cuda")
    print(f"  tensor.dtype: {tensor.dtype}, tensor.device: {tensor.device}")

    try:
        block.attention.wq.weight.data = tensor
        print(f"  ✓ 赋值成功!")
        print(f"  新的 dtype: {block.attention.wq.weight.dtype}")
        print(f"  新的 device: {block.attention.wq.weight.device}")
    except Exception as e:
        print(f"  ✗ 赋值失败: {e}")

else:
    print("CUDA not available, skipping test")
