#!/usr/bin/env python3
"""
测试 Layer Block Table (LBT) 功能
"""

import json
import sys
import time
from pathlib import Path

import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from llama3.weight_lbt import (
    build_layer_block_table,
    build_all_layer_block_tables,
    load_layer_from_raw_block,
    GroupBlockDescriptor,
    LayerBlockTable,
)
from llama3.weights_io_ssd_dram import DirectIOFile, alloc_pinned_aligned


def test_build_block_table(manifest_path: str):
    """测试构建块表"""
    print("=" * 80)
    print("TEST 1: Build Layer Block Table")
    print("=" * 80)

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # 构建所有层的块表
    print("\n📊 Building block tables for all layers...")
    tables = build_all_layer_block_tables(manifest)

    print(f"✅ Built {len(tables)} layer block tables")
    print()

    # 统计
    total_current_ios = sum(t.total_ios_current() for t in tables.values())
    total_optimized_ios = sum(t.total_ios_optimized() for t in tables.values())
    reduction = (total_current_ios - total_optimized_ios) / total_current_ios * 100

    print(f"📈 Overall Statistics:")
    print(f"   Current IOs:    {total_current_ios}")
    print(f"   Optimized IOs:  {total_optimized_ios}")
    print(f"   Reduction:      {reduction:.1f}%")
    print()

    # 显示前 5 层的详细信息
    print("📋 Sample Layers (0-4):")
    print(f"{'Layer':<8} {'Mode':<10} {'ATTN IOs':<12} {'FFN IOs':<12} {'Total Reduction':<15}")
    print("-" * 70)

    for layer_id in range(min(5, len(tables))):
        table = tables[layer_id]

        attn_mode = "✅ Block" if table.attn_block.can_merge() else "❌ Scatter"
        ffn_mode = "✅ Block" if table.ffn_block.can_merge() else "❌ Scatter"

        attn_ios = f"{len(table.attn_block.params)} → 1" if table.attn_block.can_merge() else f"{len(table.attn_block.params)}"
        ffn_ios = f"{len(table.ffn_block.params)} → 1" if table.ffn_block.can_merge() else f"{len(table.ffn_block.params)}"

        mode_str = f"{attn_mode}/{ffn_mode}"
        reduction_str = f"{table.io_reduction() * 100:.1f}%"

        print(f"{layer_id:<8} {mode_str:<10} {attn_ios:<12} {ffn_ios:<12} {reduction_str:<15}")

    print()
    return tables


def test_load_layer_optimized(manifest_path: str, layer_id: int = 5):
    """测试使用 LBT 加载层"""
    print("=" * 80)
    print(f"TEST 2: Load Layer {layer_id} Using LBT (Optimized)")
    print("=" * 80)

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    raw_device = manifest["raw_device"]
    block_size = manifest["block_size"]

    # 打开 raw device
    print(f"\n📂 Opening raw device: {raw_device}")
    dio = DirectIOFile(raw_device, mode="r", block_size=block_size)

    # 构建块表
    print(f"🔧 Building block table for layer {layer_id}...")
    layer_table = build_layer_block_table(manifest, layer_id)

    # 打印块表信息
    print(f"\n📊 Layer {layer_id} Block Table:")
    print(f"   ATTN Group:")
    print(f"      Mode:         {layer_table.attn_block.mode}")
    print(f"      Params:       {len(layer_table.attn_block.params)}")
    print(f"      Total span:   {layer_table.attn_block.total_span / (1024**2):.2f} MB")
    print(f"      Useful bytes: {layer_table.attn_block.useful_bytes / (1024**2):.2f} MB")
    print(f"      Fragmentation: {layer_table.attn_block.fragmentation * 100:.2f}%")
    print()
    print(f"   FFN Group:")
    print(f"      Mode:         {layer_table.ffn_block.mode}")
    print(f"      Params:       {len(layer_table.ffn_block.params)}")
    print(f"      Total span:   {layer_table.ffn_block.total_span / (1024**2):.2f} MB")
    print(f"      Useful bytes: {layer_table.ffn_block.useful_bytes / (1024**2):.2f} MB")
    print(f"      Fragmentation: {layer_table.ffn_block.fragmentation * 100:.2f}%")
    print()

    # 分配 staging buffer（需要能容纳 FFN 组，它更大）
    max_span = max(layer_table.attn_block.total_span, layer_table.ffn_block.total_span)
    staging_size = ((max_span + block_size - 1) // block_size) * block_size

    print(f"💾 Allocating staging buffer: {staging_size / (1024**2):.2f} MB")
    staging_buffer = alloc_pinned_aligned(staging_size, block_size)

    # 加载层（使用 LBT 优化）
    print(f"\n⚡ Loading layer {layer_id} with LBT optimization...")
    start = time.time()

    layer_weights = load_layer_from_raw_block(
        layer_table, dio, staging_buffer, block_size
    )

    elapsed = time.time() - start

    print(f"✅ Loaded {len(layer_weights)} parameters in {elapsed*1000:.2f} ms")
    print()

    # 显示加载的参数
    print("📦 Loaded Parameters:")
    total_bytes = 0
    for name, tensor in sorted(layer_weights.items()):
        param_bytes = tensor.numel() * tensor.element_size()
        total_bytes += param_bytes
        print(f"   {name:<50} {str(tensor.shape):<25} {param_bytes/(1024**2):>8.2f} MB")

    print()
    print(f"   Total: {total_bytes / (1024**2):.2f} MB")
    print()

    # 关闭文件
    dio.close()

    return layer_weights


def test_load_layer_baseline(manifest_path: str, layer_id: int = 5):
    """测试传统逐参数加载（作为对比基准）"""
    print("=" * 80)
    print(f"TEST 3: Load Layer {layer_id} Using Baseline (Per-Param)")
    print("=" * 80)

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    from llama3.weights_io_ssd_dram import DTYPE_MAP

    raw_device = manifest["raw_device"]
    block_size = manifest["block_size"]

    # 打开 raw device
    print(f"\n📂 Opening raw device: {raw_device}")
    dio = DirectIOFile(raw_device, mode="r", block_size=block_size)

    # 提取该层的参数
    layer_params = [p for p in manifest["params"]
                   if p.get("layer") == layer_id and p.get("policy") == "stream"]

    print(f"📋 Found {len(layer_params)} stream parameters for layer {layer_id}")
    print()

    # 分配 staging buffer
    max_stride = max(p["stride"] for p in layer_params)
    staging_size = ((max_stride + block_size - 1) // block_size) * block_size

    print(f"💾 Allocating staging buffer: {staging_size / (1024**2):.2f} MB")
    staging_buffer = alloc_pinned_aligned(staging_size, block_size)

    # 逐参数加载
    print(f"\n🐌 Loading layer {layer_id} with baseline (per-param)...")
    start = time.time()

    layer_weights = {}
    for param_info in layer_params:
        offset = param_info["offset"]
        stride = param_info["stride"]
        nbytes = param_info["nbytes"]
        param_name = param_info["name"]

        # 单个参数读取
        dio.pread_into_tensor(staging_buffer, stride, offset)

        # 创建目标 tensor
        param_tensor = torch.empty(
            param_info["shape"],
            dtype=DTYPE_MAP[param_info["dtype"]],
            pin_memory=True
        )

        # 拷贝有效字节
        param_tensor.view(-1).view(torch.uint8)[:nbytes].copy_(
            staging_buffer[:nbytes]
        )

        layer_weights[param_name] = param_tensor

    elapsed = time.time() - start

    print(f"✅ Loaded {len(layer_weights)} parameters in {elapsed*1000:.2f} ms")
    print()

    # 关闭文件
    dio.close()

    return layer_weights


def compare_performance(manifest_path: str, layer_id: int = 5, iterations: int = 3):
    """性能对比测试"""
    print("=" * 80)
    print(f"TEST 4: Performance Comparison (Layer {layer_id}, {iterations} iterations)")
    print("=" * 80)

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    from llama3.weights_io_ssd_dram import DTYPE_MAP

    raw_device = manifest["raw_device"]
    block_size = manifest["block_size"]

    # 构建块表
    layer_table = build_layer_block_table(manifest, layer_id)

    # 提取参数（用于 baseline）
    layer_params = [p for p in manifest["params"]
                   if p.get("layer") == layer_id and p.get("policy") == "stream"]

    # 分配 staging buffers
    max_span = max(layer_table.attn_block.total_span, layer_table.ffn_block.total_span)
    staging_size = ((max_span + block_size - 1) // block_size) * block_size
    staging_buffer = alloc_pinned_aligned(staging_size, block_size)

    # Baseline 测试
    print(f"\n🐌 Baseline (per-param) - {iterations} iterations:")
    baseline_times = []

    for i in range(iterations):
        dio = DirectIOFile(raw_device, mode="r", block_size=block_size)

        start = time.time()
        layer_weights = {}
        for param_info in layer_params:
            dio.pread_into_tensor(staging_buffer, param_info["stride"], param_info["offset"])
            param_tensor = torch.empty(param_info["shape"], dtype=DTYPE_MAP[param_info["dtype"]], pin_memory=True)
            param_tensor.view(-1).view(torch.uint8)[:param_info["nbytes"]].copy_(staging_buffer[:param_info["nbytes"]])
            layer_weights[param_info["name"]] = param_tensor

        elapsed = time.time() - start
        baseline_times.append(elapsed * 1000)
        dio.close()

        print(f"   Iteration {i+1}: {elapsed*1000:.2f} ms")

    baseline_avg = sum(baseline_times) / len(baseline_times)
    print(f"   Average: {baseline_avg:.2f} ms")

    # LBT 优化测试
    print(f"\n⚡ LBT Optimized (block mode) - {iterations} iterations:")
    lbt_times = []

    for i in range(iterations):
        dio = DirectIOFile(raw_device, mode="r", block_size=block_size)

        start = time.time()
        layer_weights = load_layer_from_raw_block(layer_table, dio, staging_buffer, block_size)
        elapsed = time.time() - start
        lbt_times.append(elapsed * 1000)
        dio.close()

        print(f"   Iteration {i+1}: {elapsed*1000:.2f} ms")

    lbt_avg = sum(lbt_times) / len(lbt_times)
    print(f"   Average: {lbt_avg:.2f} ms")

    # 性能提升
    speedup = (baseline_avg - lbt_avg) / baseline_avg * 100
    print()
    print(f"📊 Performance Improvement:")
    print(f"   Baseline:    {baseline_avg:.2f} ms")
    print(f"   LBT:         {lbt_avg:.2f} ms")
    print(f"   Improvement: {speedup:.1f}% faster")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test Layer Block Table functionality")
    parser.add_argument("--manifest", default="/data1/70b-fixed.runtime_manifest.json",
                       help="Path to runtime manifest")
    parser.add_argument("--layer", type=int, default=5,
                       help="Layer to test (default: 5)")
    parser.add_argument("--iterations", type=int, default=3,
                       help="Number of iterations for performance test")

    args = parser.parse_args()

    try:
        # Test 1: Build block tables
        tables = test_build_block_table(args.manifest)

        # Test 2: Load layer with LBT
        lbt_weights = test_load_layer_optimized(args.manifest, args.layer)

        # Test 3: Load layer with baseline
        baseline_weights = test_load_layer_baseline(args.manifest, args.layer)

        # Verify they loaded the same parameters
        print("=" * 80)
        print("VERIFICATION")
        print("=" * 80)
        print()

        if set(lbt_weights.keys()) == set(baseline_weights.keys()):
            print("✅ Both methods loaded the same parameters")
        else:
            print("❌ Parameter mismatch!")
            print(f"   LBT:      {set(lbt_weights.keys())}")
            print(f"   Baseline: {set(baseline_weights.keys())}")

        print()

        # Test 4: Performance comparison
        compare_performance(args.manifest, args.layer, args.iterations)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
