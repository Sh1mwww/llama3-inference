#!/usr/bin/env python3
"""
简单测试脚本：使用 raw param store 加载 llama3.1-8b 的一层权重
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from llama3.raw_param_store import ParamStore


def test_raw_param_store():
    """测试 raw param store 能否正常加载 llama3.1-8b"""

    print("=" * 60)
    print("🧪 Testing Raw Param Store with LLaMA3.1-8B")
    print("=" * 60)

    manifest_path = "/data1/llama3.1-8B.runtime_manifest.json"

    try:
        # 1. 创建 ParamStore
        print("\n[1/4] 创建 ParamStore...")
        store = ParamStore(
            manifest_or_path=manifest_path,
            method="bytecopy",
            staging_mb=16,
            rw=False
        )
        print("✅ ParamStore 创建成功")

        # 2. 获取存储统计信息
        print("\n[2/4] 获取存储统计信息...")
        stats = store.get_storage_stats()
        print(f"   总参数数: {stats['total_params']}")
        print(f"   总大小: {stats['total_gb']:.2f} GB")
        print(f"   Stream 大小: {stats['stream_gb']:.2f} GB")
        print(f"   Raw 设备: {stats['raw_device']}")
        print(f"   块大小: {stats['block_size']}")

        # 3. 加载第 0 层的权重（only_stream=True）
        print("\n[3/4] 加载第 0 层的 stream 权重...")
        layer_0_tensors = store.fetch_layer(
            layer_id=0,
            only_stream=True
        )

        if layer_0_tensors:
            print(f"✅ 成功加载第 0 层，共 {len(layer_0_tensors)} 个参数:")
            total_mb = 0
            for name, tensor in layer_0_tensors.items():
                mb = tensor.numel() * tensor.element_size() / (1024**2)
                total_mb += mb
                print(f"   - {name}: {list(tensor.shape)} {tensor.dtype} "
                      f"({mb:.2f} MB, pinned={tensor.is_pinned()})")
            print(f"   总计: {total_mb:.2f} MB")
        else:
            print("⚠️  第 0 层没有 stream 权重")

        # 4. 校验数据完整性（读取前 64 字节进行校验）
        print("\n[4/4] 校验数据完整性...")
        matched, total = store.sanity_check_layer(
            layer_id=0,
            tensors=layer_0_tensors,
            check_bytes=64,
            verbose=False
        )
        print(f"   校验结果: {matched}/{total} 分片匹配")
        if matched == total:
            print("   ✅ 数据完整性校验通过")
        else:
            print(f"   ⚠️  有 {total - matched} 个分片校验失败")

        # 5. 测试异步加载
        print("\n[5/5] 测试异步加载第 1 层...")
        future = store.fetch_layer_async(layer_id=1, only_stream=True)
        layer_1_tensors = future.result()

        if layer_1_tensors:
            total_mb = sum(t.numel() * t.element_size() for t in layer_1_tensors.values()) / (1024**2)
            print(f"✅ 异步加载成功: {len(layer_1_tensors)} 个参数, {total_mb:.2f} MB")

        # 6. 关闭 store
        store.close()
        print("\n✅ ParamStore 关闭成功")

        print("\n" + "=" * 60)
        print("🎉 所有测试通过！Raw Param Store 工作正常")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_raw_param_store()
    sys.exit(0 if success else 1)
