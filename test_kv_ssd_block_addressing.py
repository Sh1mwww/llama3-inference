#!/usr/bin/env python3
"""
测试 KV cache SSD 读写功能 - 使用 block 索引
验证 push() 写入和 _load_from_ssd() 读取的一致性
"""

import torch
import numpy as np
import sys
import os

# 添加 llama3 模块路径
sys.path.insert(0, os.path.dirname(__file__))

from llama3.SSDBacked import RawBlockKVBackend


def test_single_block_rw():
    """测试单个 block 的读写"""
    print("=" * 60)
    print("Test 1: Single Block Read/Write")
    print("=" * 60)

    # 参数设置
    ssd_device = "/dev/nvme0n1p4"
    n_layers = 2
    max_batch = 1
    heads = 8
    dim = 128
    n_blocks = 10

    # 计算 block 大小 (K + V)
    block_nbytes = (max_batch * heads * dim * 2) * 2  # fp16 = 2 bytes

    try:
        # 初始化 SSD backend
        ssd = RawBlockKVBackend(
            dev_path=ssd_device,
            n_layers=n_layers,
            blk_bytes=block_nbytes,
            blk_per_layer=n_blocks,
            max_concurrent_io=4
        )
        print(f"✓ SSD backend initialized: {ssd_device}")
        print(f"  - Layers: {n_layers}")
        print(f"  - Blocks per layer: {n_blocks}")
        print(f"  - Block size: {block_nbytes} bytes ({block_nbytes / 1024:.2f} KB)")
        print(f"  - Aligned stride: {ssd.stride} bytes")

    except Exception as e:
        print(f"✗ Failed to initialize SSD backend: {e}")
        return False

    # 测试数据生成
    layer = 0
    blk = 3

    # 创建测试数据：K 和 V
    k_test = torch.randn(max_batch, heads, dim, dtype=torch.float16)
    v_test = torch.randn(max_batch, heads, dim, dtype=torch.float16)

    # 打包为 KV (max_batch, heads, 2*dim)
    kv_pack = torch.cat([k_test, v_test], dim=-1)

    print(f"\n📝 Writing test data to Layer {layer}, Block {blk}")
    print(f"  - K shape: {k_test.shape}")
    print(f"  - V shape: {v_test.shape}")
    print(f"  - KV pack shape: {kv_pack.shape}")
    print(f"  - Data size: {kv_pack.numel() * 2} bytes")

    # 写入 SSD
    try:
        ssd.write(layer, blk, kv_pack, sync=True)
        print("✓ Write completed (sync)")
    except Exception as e:
        print(f"✗ Write failed: {e}")
        return False

    # 从 SSD 读取
    print(f"\n📖 Reading data from Layer {layer}, Block {blk}")
    kv_read_gpu = torch.empty_like(kv_pack, device='cuda:0')

    try:
        ssd.read(layer, blk, kv_read_gpu)
        torch.cuda.synchronize()
        print("✓ Read completed")
    except Exception as e:
        print(f"✗ Read failed: {e}")
        return False

    # 验证数据
    kv_read = kv_read_gpu.cpu()
    k_read, v_read = kv_read.split(dim, dim=-1)

    k_diff = (k_test - k_read).abs().max().item()
    v_diff = (v_test - v_read).abs().max().item()

    print(f"\n🔍 Data verification:")
    print(f"  - K max diff: {k_diff}")
    print(f"  - V max diff: {v_diff}")

    if k_diff < 1e-5 and v_diff < 1e-5:
        print("✓ Data integrity verified!")
        return True
    else:
        print("✗ Data mismatch detected!")
        return False


def test_batch_write_contiguous():
    """测试连续 blocks 的批量写入"""
    print("\n" + "=" * 60)
    print("Test 2: Batch Write (Contiguous Blocks)")
    print("=" * 60)

    ssd_device = "/dev/nvme0n1p4"
    n_layers = 2
    max_batch = 1
    heads = 8
    dim = 128
    n_blocks = 20

    block_nbytes = (max_batch * heads * dim * 2) * 2

    try:
        ssd = RawBlockKVBackend(
            dev_path=ssd_device,
            n_layers=n_layers,
            blk_bytes=block_nbytes,
            blk_per_layer=n_blocks,
            max_concurrent_io=4
        )
        print(f"✓ SSD backend initialized")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return False

    # 准备连续的 blocks: 5, 6, 7, 8
    layer = 1
    slots = [5, 6, 7, 8]
    tensors = []

    print(f"\n📝 Preparing {len(slots)} contiguous blocks for Layer {layer}")
    print(f"  - Block indices: {slots}")

    for i, slot in enumerate(slots):
        k = torch.full((max_batch, heads, dim), fill_value=float(slot * 10 + i), dtype=torch.float16)
        v = torch.full((max_batch, heads, dim), fill_value=float(slot * 10 + i + 0.5), dtype=torch.float16)
        kv_pack = torch.cat([k, v], dim=-1)
        tensors.append(kv_pack)

    # 批量写入
    try:
        ssd.write_batch(layer, slots, tensors, sync=True)
        print("✓ Batch write completed (contiguous optimization)")
    except Exception as e:
        print(f"✗ Batch write failed: {e}")
        return False

    # 逐个读取验证
    print(f"\n📖 Reading and verifying each block...")
    all_ok = True

    for i, slot in enumerate(slots):
        kv_read_gpu = torch.empty(max_batch, heads, dim * 2, dtype=torch.float16, device='cuda:0')
        ssd.read(layer, slot, kv_read_gpu)
        torch.cuda.synchronize()

        kv_read = kv_read_gpu.cpu()
        k_read, v_read = kv_read.split(dim, dim=-1)

        expected_k = float(slot * 10 + i)
        expected_v = float(slot * 10 + i + 0.5)

        k_val = k_read[0, 0, 0].item()
        v_val = v_read[0, 0, 0].item()

        # Use relative tolerance for large values due to fp16 precision limits
        # fp16 has ~3 decimal digits of precision, so use 0.1% relative tolerance
        k_tol = max(0.01, abs(expected_k) * 0.001)
        v_tol = max(0.01, abs(expected_v) * 0.001)
        k_ok = abs(k_val - expected_k) < k_tol
        v_ok = abs(v_val - expected_v) < v_tol

        status = "✓" if (k_ok and v_ok) else "✗"
        print(f"  {status} Block {slot}: K={k_val:.1f} (expect {expected_k}), V={v_val:.1f} (expect {expected_v})")

        if not (k_ok and v_ok):
            all_ok = False

    if all_ok:
        print("✓ All blocks verified successfully!")
        return True
    else:
        print("✗ Some blocks failed verification")
        return False


def test_batch_write_non_contiguous():
    """测试非连续 blocks 的批量写入"""
    print("\n" + "=" * 60)
    print("Test 3: Batch Write (Non-Contiguous Blocks)")
    print("=" * 60)

    ssd_device = "/dev/nvme0n1p4"
    n_layers = 2
    max_batch = 1
    heads = 8
    dim = 128
    n_blocks = 20

    block_nbytes = (max_batch * heads * dim * 2) * 2

    try:
        ssd = RawBlockKVBackend(
            dev_path=ssd_device,
            n_layers=n_layers,
            blk_bytes=block_nbytes,
            blk_per_layer=n_blocks,
            max_concurrent_io=4
        )
        print(f"✓ SSD backend initialized")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return False

    # 准备非连续的 blocks: 2, 5, 9, 15
    layer = 0
    slots = [2, 5, 9, 15]
    tensors = []

    print(f"\n📝 Preparing {len(slots)} non-contiguous blocks for Layer {layer}")
    print(f"  - Block indices: {slots}")

    for i, slot in enumerate(slots):
        k = torch.full((max_batch, heads, dim), fill_value=float(slot * 100 + i), dtype=torch.float16)
        v = torch.full((max_batch, heads, dim), fill_value=float(slot * 100 + i + 0.5), dtype=torch.float16)
        kv_pack = torch.cat([k, v], dim=-1)
        tensors.append(kv_pack)

    # 批量写入
    try:
        ssd.write_batch(layer, slots, tensors, sync=True)
        print("✓ Batch write completed (individual writes)")
    except Exception as e:
        print(f"✗ Batch write failed: {e}")
        return False

    # 逐个读取验证
    print(f"\n📖 Reading and verifying each block...")
    all_ok = True

    for i, slot in enumerate(slots):
        kv_read_gpu = torch.empty(max_batch, heads, dim * 2, dtype=torch.float16, device='cuda:0')
        ssd.read(layer, slot, kv_read_gpu)
        torch.cuda.synchronize()

        kv_read = kv_read_gpu.cpu()
        k_read, v_read = kv_read.split(dim, dim=-1)

        expected_k = float(slot * 100 + i)
        expected_v = float(slot * 100 + i + 0.5)

        k_val = k_read[0, 0, 0].item()
        v_val = v_read[0, 0, 0].item()

        # Use relative tolerance for large values due to fp16 precision limits
        # fp16 has ~3 decimal digits of precision, so use 0.1% relative tolerance
        k_tol = max(0.01, abs(expected_k) * 0.001)
        v_tol = max(0.01, abs(expected_v) * 0.001)
        k_ok = abs(k_val - expected_k) < k_tol
        v_ok = abs(v_val - expected_v) < v_tol

        status = "✓" if (k_ok and v_ok) else "✗"
        print(f"  {status} Block {slot}: K={k_val:.1f} (expect {expected_k}), V={v_val:.1f} (expect {expected_v})")

        if not (k_ok and v_ok):
            all_ok = False

    if all_ok:
        print("✓ All blocks verified successfully!")
        return True
    else:
        print("✗ Some blocks failed verification")
        return False


def main():
    print("KV Cache SSD Block Addressing Test Suite")
    print("=" * 60)
    print()

    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        print("✗ CUDA is not available. Cannot run tests.")
        return 1

    print(f"✓ CUDA is available: {torch.cuda.get_device_name(0)}")
    print()

    # 运行测试
    results = []

    # Test 1: 单个 block 读写
    try:
        results.append(("Single Block R/W", test_single_block_rw()))
    except Exception as e:
        print(f"\n✗ Test 1 crashed: {e}")
        results.append(("Single Block R/W", False))

    # Test 2: 连续 blocks 批量写入
    try:
        results.append(("Batch Contiguous", test_batch_write_contiguous()))
    except Exception as e:
        print(f"\n✗ Test 2 crashed: {e}")
        results.append(("Batch Contiguous", False))

    # Test 3: 非连续 blocks 批量写入
    try:
        results.append(("Batch Non-Contiguous", test_batch_write_non_contiguous()))
    except Exception as e:
        print(f"\n✗ Test 3 crashed: {e}")
        results.append(("Batch Non-Contiguous", False))

    # 总结
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = 0
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
