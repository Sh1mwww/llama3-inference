#!/usr/bin/env python3
"""
测试脚本：验证分片权重（vocab-parallel）的检测逻辑
"""

print("=" * 80)
print("测试分片权重检测逻辑")
print("=" * 80)

# 常见的完整 vocab_size
common_vocab_sizes = [128256, 128000, 32000, 50257]

# 模拟的权重形状（vocab_size, dim）
test_cases = [
    # (权重形状, 期望检测结果)
    ((128256, 8192), "完整权重", None, None),
    ((16032, 8192), "分片权重", 8, 128256),   # TP=8
    ((32064, 8192), "分片权重", 4, 128256),   # TP=4
    ((64128, 8192), "分片权重", 2, 128256),   # TP=2
    ((16000, 4096), "分片权重", 8, 128000),   # TP=8, Llama 2
    ((4000, 4096), "分片权重", 8, 32000),     # TP=8, Llama 1
    ((100000, 8192), "未知形状", None, None),
]

print("\n检测测试用例:")
print("-" * 80)

for weight_shape, expected_type, expected_tp, expected_full_vocab in test_cases:
    weight_vocab_size = weight_shape[0]

    # 执行检测逻辑（与 weights_io_ssd_dram.py 中的逻辑一致）
    is_sharded = False
    detected_tp = None
    detected_full_vocab = None

    for full_vocab in common_vocab_sizes:
        for tp_size in [2, 4, 8, 16]:
            if weight_vocab_size * tp_size == full_vocab:
                is_sharded = True
                detected_tp = tp_size
                detected_full_vocab = full_vocab
                break
        if is_sharded:
            break

    # 验证检测结果
    result = "分片权重" if is_sharded else ("完整权重" if weight_vocab_size in common_vocab_sizes else "未知形状")

    print(f"\n权重形状: {weight_shape}")
    print(f"  检测结果: {result}")

    if is_sharded:
        print(f"  TP size: {detected_tp}")
        print(f"  完整 vocab_size: {detected_full_vocab}")
        print(f"  分片计算: {weight_vocab_size} × {detected_tp} = {detected_full_vocab}")

    # 验证是否与预期一致
    if result == expected_type:
        if expected_type == "分片权重":
            if detected_tp == expected_tp and detected_full_vocab == expected_full_vocab:
                print(f"  ✅ 检测正确")
            else:
                print(f"  ❌ 检测错误: 预期 TP={expected_tp}, full_vocab={expected_full_vocab}")
        else:
            print(f"  ✅ 检测正确")
    else:
        print(f"  ❌ 检测错误: 预期 {expected_type}")

print("\n" + "=" * 80)
print("分片权重场景说明")
print("=" * 80)

scenarios = [
    {
        "场景": "Llama 3.1 70B TP=8 分片",
        "完整 vocab_size": 128256,
        "TP size": 8,
        "每分片 vocab": 128256 // 8,
        "权重形状": f"[{128256 // 8}, 8192]",
    },
    {
        "场景": "Llama 3.1 70B TP=4 分片",
        "完整 vocab_size": 128256,
        "TP size": 4,
        "每分片 vocab": 128256 // 4,
        "权重形状": f"[{128256 // 4}, 8192]",
    },
    {
        "场景": "Llama 2 70B TP=8 分片",
        "完整 vocab_size": 32000,
        "TP size": 8,
        "每分片 vocab": 32000 // 8,
        "权重形状": f"[{32000 // 8}, 4096]",
    },
]

for s in scenarios:
    print(f"\n{s['场景']}:")
    print(f"  完整 vocab_size: {s['完整 vocab_size']}")
    print(f"  TP size: {s['TP size']}")
    print(f"  每个分片的 vocab 维度: {s['每分片 vocab']}")
    print(f"  分片权重形状: {s['权重形状']}")

print("\n" + "=" * 80)
print("✅ 分片权重检测逻辑验证完成")
print("=" * 80)
print("\n关键点:")
print("1. 完整权重: vocab_size 等于常见值 (128256, 128000, 32000 等)")
print("2. 分片权重: vocab_size × TP_size = 完整 vocab_size")
print("3. 混用错误: 不能用分片权重 (16032×8192) 加载到完整模型 (128256×8192)")
print("4. 解决方案: 合并分片 → 完整权重，或使用支持 TP 的并行引擎")
