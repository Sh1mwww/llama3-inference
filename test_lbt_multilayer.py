#!/usr/bin/env python3
"""
测试 LBT 在多层连续加载场景下的性能
这更接近实际推理时的使用模式
"""

import json
import time
import torch
from pathlib import Path

from llama3.weight_lbt import build_all_layer_block_tables, load_layer_from_raw_block
from llama3.weights_io_ssd_dram import DirectIOFile, alloc_pinned_aligned, DTYPE_MAP


def baseline_load_layers(manifest, dio, staging_buffer, layer_ids):
    """Baseline: 逐参数加载多层"""
    all_weights = {}

    for layer_id in layer_ids:
        layer_params = [p for p in manifest["params"]
                       if p.get("layer") == layer_id and p.get("policy") == "stream"]

        layer_weights = {}
        for param_info in layer_params:
            dio.pread_into_tensor(staging_buffer, param_info["stride"], param_info["offset"])

            param_tensor = torch.empty(
                param_info["shape"],
                dtype=DTYPE_MAP[param_info["dtype"]],
                pin_memory=True
            )
            param_tensor.view(-1).view(torch.uint8)[:param_info["nbytes"]].copy_(
                staging_buffer[:param_info["nbytes"]]
            )

            layer_weights[param_info["name"]] = param_tensor

        all_weights[layer_id] = layer_weights

    return all_weights


def lbt_load_layers(tables, dio, staging_buffer, block_size, layer_ids):
    """LBT: 使用块表加载多层"""
    all_weights = {}

    for layer_id in layer_ids:
        layer_weights = load_layer_from_raw_block(
            tables[layer_id], dio, staging_buffer, block_size
        )
        all_weights[layer_id] = layer_weights

    return all_weights


def test_multilayer_performance(manifest_path, num_layers=10, iterations=3):
    """测试多层连续加载性能"""

    print("=" * 80)
    print(f"Multi-Layer Loading Performance Test")
    print(f"  Layers: 0-{num_layers-1}")
    print(f"  Iterations: {iterations}")
    print("=" * 80)
    print()

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    raw_device = manifest["raw_device"]
    block_size = manifest["block_size"]

    # 构建所有层的块表
    print("📊 Building block tables...")
    tables = build_all_layer_block_tables(manifest)
    print(f"✅ Built {len(tables)} tables")
    print()

    # 计算需要的 staging buffer 大小（FFN 组最大）
    max_span = max(
        max(tables[lid].attn_block.total_span, tables[lid].ffn_block.total_span)
        for lid in range(num_layers)
    )
    staging_size = ((max_span + block_size - 1) // block_size) * block_size

    print(f"💾 Staging buffer: {staging_size / (1024**2):.2f} MB")
    staging_buffer = alloc_pinned_aligned(staging_size, block_size)
    print()

    layer_ids = list(range(num_layers))

    # Baseline 测试
    print(f"🐌 Baseline (per-param) - {iterations} iterations:")
    baseline_times = []

    for i in range(iterations):
        dio = DirectIOFile(raw_device, mode="r", block_size=block_size)

        start = time.time()
        weights = baseline_load_layers(manifest, dio, staging_buffer, layer_ids)
        elapsed = time.time() - start

        baseline_times.append(elapsed)
        dio.close()

        total_params = sum(len(w) for w in weights.values())
        print(f"   Iteration {i+1}: {elapsed:.3f}s ({total_params} params)")

    baseline_avg = sum(baseline_times) / len(baseline_times)
    print(f"   Average: {baseline_avg:.3f}s")
    print()

    # LBT 测试
    print(f"⚡ LBT Optimized - {iterations} iterations:")
    lbt_times = []

    for i in range(iterations):
        dio = DirectIOFile(raw_device, mode="r", block_size=block_size)

        start = time.time()
        weights = lbt_load_layers(tables, dio, staging_buffer, block_size, layer_ids)
        elapsed = time.time() - start

        lbt_times.append(elapsed)
        dio.close()

        total_params = sum(len(w) for w in weights.values())
        print(f"   Iteration {i+1}: {elapsed:.3f}s ({total_params} params)")

    lbt_avg = sum(lbt_times) / len(lbt_times)
    print(f"   Average: {lbt_avg:.3f}s")
    print()

    # 统计
    speedup = (baseline_avg - lbt_avg) / baseline_avg * 100
    io_reduction = (num_layers * 7 - num_layers * 2) / (num_layers * 7) * 100

    print("=" * 80)
    print("📊 Results Summary")
    print("=" * 80)
    print(f"  Baseline avg:     {baseline_avg:.3f}s")
    print(f"  LBT avg:          {lbt_avg:.3f}s")
    print(f"  Speedup:          {speedup:+.1f}%")
    print(f"  IO reduction:     {io_reduction:.1f}%")
    print(f"  Baseline IOs:     {num_layers * 7}")
    print(f"  LBT IOs:          {num_layers * 2}")
    print()

    # 吞吐量
    total_mb = num_layers * 1632  # 每层 1632 MB
    baseline_throughput = total_mb / baseline_avg
    lbt_throughput = total_mb / lbt_avg

    print(f"  Data loaded:      {total_mb:.0f} MB")
    print(f"  Baseline throughput: {baseline_throughput:.1f} MB/s")
    print(f"  LBT throughput:      {lbt_throughput:.1f} MB/s")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/data1/70b-fixed.runtime_manifest.json")
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=3)

    args = parser.parse_args()

    test_multilayer_performance(args.manifest, args.layers, args.iterations)
