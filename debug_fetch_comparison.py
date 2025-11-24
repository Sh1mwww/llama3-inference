#!/usr/bin/env python3
"""
对比两种方法读取的数据，找出差异
"""

import torch
from llama3.raw_param_store import ParamStore

manifest_path = "/data1/70b-fixed.runtime_manifest.json"

with ParamStore(manifest_path, staging_mb=64) as store:
    layer_id = 0

    print("=" * 80)
    print("对比 fetch_layer 和 fetch_layer_mha_ffn_2io")
    print("=" * 80)

    # 方法1: 原方法
    tensors_old = store.fetch_layer(layer_id, only_stream=True)

    # 方法2: 新方法
    tensors_new = store.fetch_layer_mha_ffn_2io(layer_id, only_stream=True)

    # 对比每个参数
    for name in sorted(tensors_old.keys()):
        old_t = tensors_old[name]
        new_t = tensors_new[name]

        print(f"\n{name}:")
        print(f"  Shape: {old_t.shape}")
        print(f"  Dtype: {old_t.dtype}")
        print(f"  Size: {old_t.numel() * old_t.element_size() / (1024**2):.2f} MB")

        # 检查前几个值
        old_flat = old_t.view(-1)
        new_flat = new_t.view(-1)

        print(f"  Old前10个值: {old_flat[:10].tolist()}")
        print(f"  New前10个值: {new_flat[:10].tolist()}")

        if torch.allclose(old_t, new_t, rtol=1e-5, atol=1e-7):
            print(f"  ✅ 数值一致")
        else:
            # 计算差异
            diff = (old_t != new_t).sum().item()
            total = old_t.numel()
            print(f"  ❌ 数值不一致: {diff}/{total} 元素不同 ({diff/total*100:.2f}%)")

            # 显示第一个不同的位置
            diff_mask = (old_t != new_t)
            if diff_mask.any():
                first_diff_idx = diff_mask.view(-1).nonzero()[0].item()
                print(f"     第一个差异位置: index={first_diff_idx}")
                print(f"     Old值: {old_flat[first_diff_idx]}")
                print(f"     New值: {new_flat[first_diff_idx]}")

                # 检查差异区域是否集中
                diff_indices = diff_mask.view(-1).nonzero().squeeze()
                if len(diff_indices) > 1:
                    print(f"     差异区域: 从 {diff_indices[0].item()} 到 {diff_indices[-1].item()}")
