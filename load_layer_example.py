#!/usr/bin/env python3
"""
从raw block device加载单层权重到pinned memory的示例
"""

import sys
import json
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from llama3.weights_io_ssd_dram import DirectIOFile, alloc_pinned_aligned, DTYPE_MAP


def load_single_layer_from_ssd(manifest_path: str, layer_id: int, verbose: bool = True):
    """
    从SSD加载指定层的所有权重到pinned memory

    Args:
        manifest_path: runtime manifest文件路径
        layer_id: 要加载的层ID (0-79 for LLaMA3-70B)
        verbose: 是否打印详细信息

    Returns:
        dict: {param_name: torch.Tensor} 包含该层所有权重的字典
    """

    if verbose:
        print(f"🔄 Loading layer {layer_id} from SSD...")

    # 1. 加载manifest
    if not Path(manifest_path).exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    if verbose:
        print(f"✅ Manifest loaded: {manifest['raw_device']}")

    # 2. 打开SSD设备
    dio = DirectIOFile(
        manifest["raw_device"],
        mode="r",
        block_size=manifest["block_size"]
    )

    # 3. 找到指定层的所有参数
    layer_params = []
    for param in manifest["params"]:
        if param["layer"] == layer_id and param["policy"] == "stream":
            layer_params.append(param)

    if not layer_params:
        dio.close()
        raise ValueError(f"No stream parameters found for layer {layer_id}")

    if verbose:
        print(f"📋 Found {len(layer_params)} parameters for layer {layer_id}")

    # 4. 计算最大staging buffer需求
    max_stride = max(p["stride"] for p in layer_params)
    staging_buffer = alloc_pinned_aligned(max_stride, manifest["block_size"])

    if verbose:
        print(f"💾 Allocated {max_stride} byte staging buffer")

    # 5. 逐个加载参数
    layer_weights = {}

    for param_info in layer_params:
        param_name = param_info["name"]
        stride = param_info["stride"]
        offset = param_info["offset"]
        nbytes = param_info["nbytes"]
        shape = param_info["shape"]
        dtype_str = param_info["dtype"]

        if verbose:
            print(f"   Loading {param_name}: {shape} ({nbytes} bytes)")

        # 扩展staging buffer如果需要
        if stride > len(staging_buffer):
            staging_buffer = alloc_pinned_aligned(stride, manifest["block_size"])
            if verbose:
                print(f"   Expanded staging buffer to {stride} bytes")

        # 从SSD读取到staging buffer
        dio.pread_into_tensor(staging_buffer, stride, offset)

        # 创建目标tensor (pinned)
        param_tensor = torch.empty(
            shape,
            dtype=DTYPE_MAP[dtype_str],
            pin_memory=True
        )

        # 复制数据
        param_tensor.view(-1).view(torch.uint8)[:nbytes].copy_(
            staging_buffer[:nbytes]
        )

        layer_weights[param_name] = param_tensor

        if verbose:
            print(f"   ✅ {param_name} loaded to pinned memory")

    # 6. 清理
    dio.close()

    if verbose:
        total_mb = sum(t.numel() * t.element_size() for t in layer_weights.values()) / (1024**2)
        print(f"✅ Layer {layer_id} loaded: {len(layer_weights)} params, {total_mb:.1f} MB")

    return layer_weights


def load_specific_parameter(manifest_path: str, param_name: str, verbose: bool = True):
    """
    从SSD加载指定的单个参数

    Args:
        manifest_path: runtime manifest文件路径
        param_name: 参数名称 (如 "layers.0.attention.wq.weight")
        verbose: 是否打印详细信息

    Returns:
        torch.Tensor: 加载的参数tensor (pinned memory)
    """

    if verbose:
        print(f"🔄 Loading parameter: {param_name}")

    # 1. 加载manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # 2. 找到指定参数
    param_info = None
    for param in manifest["params"]:
        if param["name"] == param_name:
            param_info = param
            break

    if param_info is None:
        raise ValueError(f"Parameter {param_name} not found in manifest")

    if verbose:
        print(f"📋 Found parameter: {param_info['shape']} ({param_info['nbytes']} bytes)")

    # 3. 打开SSD设备
    dio = DirectIOFile(
        manifest["raw_device"],
        mode="r",
        block_size=manifest["block_size"]
    )

    # 4. 分配staging buffer
    stride = param_info["stride"]
    staging_buffer = alloc_pinned_aligned(stride, manifest["block_size"])

    # 5. 读取参数
    dio.pread_into_tensor(staging_buffer, stride, param_info["offset"])

    # 6. 创建目标tensor
    param_tensor = torch.empty(
        param_info["shape"],
        dtype=DTYPE_MAP[param_info["dtype"]],
        pin_memory=True
    )

    # 7. 复制数据
    param_tensor.view(-1).view(torch.uint8)[:param_info["nbytes"]].copy_(
        staging_buffer[:param_info["nbytes"]]
    )

    # 8. 清理
    dio.close()

    if verbose:
        print(f"✅ Parameter loaded to pinned memory: {param_tensor.shape}")

    return param_tensor


def batch_load_layers(manifest_path: str, layer_ids: list, verbose: bool = True):
    """
    批量加载多个层的权重

    Args:
        manifest_path: runtime manifest文件路径
        layer_ids: 层ID列表
        verbose: 是否打印详细信息

    Returns:
        dict: {layer_id: {param_name: torch.Tensor}}
    """

    if verbose:
        print(f"🔄 Batch loading layers: {layer_ids}")

    all_layers = {}

    for layer_id in layer_ids:
        try:
            layer_weights = load_single_layer_from_ssd(manifest_path, layer_id, verbose=False)
            all_layers[layer_id] = layer_weights

            if verbose:
                total_mb = sum(t.numel() * t.element_size() for t in layer_weights.values()) / (1024**2)
                print(f"✅ Layer {layer_id}: {len(layer_weights)} params, {total_mb:.1f} MB")

        except Exception as e:
            if verbose:
                print(f"❌ Failed to load layer {layer_id}: {e}")
            continue

    if verbose:
        total_layers = len(all_layers)
        total_mb = sum(
            sum(t.numel() * t.element_size() for t in layer_weights.values())
            for layer_weights in all_layers.values()
        ) / (1024**2)
        print(f"🎉 Batch load complete: {total_layers} layers, {total_mb:.1f} MB total")

    return all_layers


def demonstrate_usage():
    """演示如何使用这些函数"""

    manifest_path = "/data1/llama-70b.runtime_manifest.json"

    print("🚀 SSD Layer Loading Demonstration")
    print("=" * 50)

    try:
        # 示例1: 加载单个层
        print("\n1. 加载单个层 (Layer 0)")
        print("-" * 30)
        layer_0_weights = load_single_layer_from_ssd(manifest_path, 0)

        print(f"\nLayer 0 参数列表:")
        for name, tensor in layer_0_weights.items():
            print(f"  {name}: {tensor.shape} {tensor.dtype} (pinned: {tensor.is_pinned()})")

        # 示例2: 加载特定参数
        print("\n\n2. 加载特定参数")
        print("-" * 30)
        if layer_0_weights:
            first_param_name = list(layer_0_weights.keys())[0]
            param_tensor = load_specific_parameter(manifest_path, first_param_name)
            print(f"参数详情: {param_tensor.shape}, pinned: {param_tensor.is_pinned()}")

        # 示例3: 批量加载多层
        print("\n\n3. 批量加载多层 (Layer 0-2)")
        print("-" * 30)
        multi_layers = batch_load_layers(manifest_path, [0, 1, 2])

        # 示例4: 转移到GPU
        if torch.cuda.is_available():
            print("\n\n4. 转移到GPU")
            print("-" * 30)
            first_layer = list(multi_layers.values())[0]
            first_param = list(first_layer.values())[0]

            print(f"原始: {first_param.device}")
            gpu_param = first_param.cuda(non_blocking=True)
            print(f"GPU: {gpu_param.device}")
            print("✅ 非阻塞传输成功")

        print(f"\n🎉 演示完成！所有权重都在pinned memory中，可以高效传输到GPU")

    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🔧 Raw Block Device Layer Loading")
    print("=" * 40)

    # 运行演示
    demonstrate_usage()

    print(f"\n💡 使用方法总结:")
    print(f"1. load_single_layer_from_ssd(manifest_path, layer_id)")
    print(f"2. load_specific_parameter(manifest_path, param_name)")
    print(f"3. batch_load_layers(manifest_path, layer_ids)")
    print(f"4. 所有返回的tensor都是pinned memory，可以non_blocking传输到GPU")