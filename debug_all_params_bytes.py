#!/usr/bin/env python3
"""
对比所有参数的字节数据
"""

import torch
from llama3.raw_param_store import ParamStore

manifest_path = "/data1/70b-fixed.runtime_manifest.json"

with ParamStore(manifest_path, staging_mb=128) as store:
    layer_id = 0

    print("=" * 80)
    print("对比所有参数的字节")
    print("=" * 80)

    # 方法1: 原方法
    tensors_old = store.fetch_layer(layer_id, only_stream=True)

    # 方法2: 新方法
    tensors_new = store.fetch_layer_mha_ffn_2io(layer_id, only_stream=True)

    for name in sorted(tensors_old.keys()):
        old_t = tensors_old[name]
        new_t = tensors_new[name]

        # 转成字节
        old_bytes = old_t.contiguous().view(-1).view(torch.uint8).cpu()
        new_bytes = new_t.contiguous().view(-1).view(torch.uint8).cpu()

        if torch.equal(old_bytes, new_bytes):
            print(f"✅ {name}: 字节完全一致")
        else:
            diff_mask = (old_bytes != new_bytes)
            diff_count = diff_mask.sum().item()
            first_diff = diff_mask.nonzero()[0].item()
            print(f"❌ {name}: {diff_count}/{len(old_bytes)} 字节不同 ({diff_count/len(old_bytes)*100:.2f}%)")
            print(f"      第一个差异: 字节位置 {first_diff}")
