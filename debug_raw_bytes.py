#!/usr/bin/env python3
"""
对比原始字节数据
"""

import torch
from llama3.raw_param_store import ParamStore

manifest_path = "/data1/70b-fixed.runtime_manifest.json"

with ParamStore(manifest_path, staging_mb=64) as store:
    layer_id = 0

    # 只看 wq（第一个参数）
    print("=" * 80)
    print("对比 layers.0.attention.wq.weight 的原始字节")
    print("=" * 80)

    # 方法1: 原方法
    tensors_old = store.fetch_layer(layer_id, only_stream=True, names=["layers.0.attention.wq.weight"])
    old_wq = tensors_old["layers.0.attention.wq.weight"]

    # 方法2: 新方法
    tensors_new = store.fetch_layer_mha_ffn_2io(layer_id, only_stream=True)
    new_wq = tensors_new["layers.0.attention.wq.weight"]

    print(f"Old shape: {old_wq.shape}, dtype: {old_wq.dtype}")
    print(f"New shape: {new_wq.shape}, dtype: {new_wq.dtype}")

    # 转成字节 (flat 1D 视图)
    old_bytes = old_wq.contiguous().view(-1).view(torch.uint8).cpu()
    new_bytes = new_wq.contiguous().view(-1).view(torch.uint8).cpu()

    print(f"\nOld bytes shape: {old_bytes.shape}")
    print(f"New bytes shape: {new_bytes.shape}")

    # 对比前50个字节
    print(f"\n前50个字节:")
    print(f"Old: {list(old_bytes[:50].numpy())}")
    print(f"New: {list(new_bytes[:50].numpy())}")

    # 检查是否一致
    if torch.equal(old_bytes, new_bytes):
        print("\n✅ 字节完全一致")
    else:
        # 找到第一个不同的字节
        diff_mask = (old_bytes != new_bytes)
        first_diff_idx = diff_mask.nonzero()[0].item()
        print(f"\n❌ 第一个不同的字节位置: {first_diff_idx} ({first_diff_idx/(1024**2):.2f} MB)")
        print(f"   Old byte: {old_bytes[first_diff_idx].item()}")
        print(f"   New byte: {new_bytes[first_diff_idx].item()}")

        # 统计不同的字节数
        diff_count = diff_mask.sum().item()
        print(f"   不同的字节数: {diff_count}/{len(old_bytes)} ({diff_count/len(old_bytes)*100:.2f}%)")

        # 显示前几个不同的字节
        diff_indices = diff_mask.nonzero().squeeze()[:10]
        print(f"\n   前10个不同的字节位置:")
        for idx in diff_indices:
            idx = idx.item()
            print(f"      [{idx:>10}]: old={old_bytes[idx].item():>3}, new={new_bytes[idx].item():>3}")
