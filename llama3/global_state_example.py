#!/usr/bin/env python3
"""
全局状态跟踪器使用示例

这个示例展示了如何使用全局状态跟踪器来监控LLaMA3项目中的
batch和layer在HBM、DRAM、SSD之间的分布情况。
"""

import torch
import numpy as np
from llama3.kv_offload import KVOffloader
from llama3.global_state_tracker import GlobalStateTracker, StorageType, init_global_tracker, get_global_tracker

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 初始化KVOffloader（会自动创建全局跟踪器）
    offloader = KVOffloader(
        layers=32,
        heads=32,
        dim=128,
        max_seq=2048,
        max_batch=4,
        device="cuda:0",
        dtype_bytes=2
    )
    
    # 获取全局跟踪器
    tracker = get_global_tracker()
    
    # 模拟一些操作
    print("\n1. 设置当前执行状态")
    offloader.set_current_execution(0, layer_idx=5)
    
    print("\n2. 模拟push操作（HBM → DRAM）")
    # 创建一些dummy数据
    k_tensor = torch.randn(4, 32, 128, dtype=torch.float16, device="cuda:0")
    v_tensor = torch.randn(4, 32, 128, dtype=torch.float16, device="cuda:0")
    
    # Push几个blocks到DRAM
    offloader.push(layer=5, blk=0, k=k_tensor, v=v_tensor, batch_idx=0)
    offloader.push(layer=5, blk=1, k=k_tensor, v=v_tensor, batch_idx=0)
    offloader.push(layer=5, blk=2, k=k_tensor, v=v_tensor, batch_idx=1)
    
    print("\n3. 查看当前状态")
    offloader.print_global_state()
    
    print("\n4. 模拟fetch操作（DRAM → HBM）")
    # 模拟fetch一些blocks
    blocks_to_fetch = torch.tensor([0, 1], dtype=torch.long, device="cuda:0")
    try:
        k_fetched, v_fetched = offloader.fetch(layer=5, blocks=blocks_to_fetch, batch_idx=0)
        print(f"Fetched K shape: {k_fetched.shape}, V shape: {v_fetched.shape}")
    except Exception as e:
        print(f"Fetch error (expected in demo): {e}")
    
    print("\n5. 查看更新后的状态")
    offloader.print_global_state()

def example_batch_analysis():
    """批次分析示例"""
    print("\n\n=== 批次分析示例 ===")
    
    # 获取全局跟踪器
    tracker = get_global_tracker()
    if tracker is None:
        print("请先运行基本使用示例来初始化跟踪器")
        return
    
    # 分析特定batch的分布
    print("\n1. Batch 0 的存储分布:")
    batch0_dist = tracker.get_batch_distribution(0)
    for storage_type, layers in batch0_dist.items():
        print(f"  {storage_type.upper()}: {layers}")
    
    print("\n2. Batch 1 的存储分布:")
    batch1_dist = tracker.get_batch_distribution(1)
    for storage_type, layers in batch1_dist.items():
        print(f"  {storage_type.upper()}: {layers}")
    
    # 分析特定layer的分布
    print("\n3. Layer 5 的存储分布:")
    layer5_dist = tracker.get_layer_distribution(5)
    for storage_type, batches in layer5_dist.items():
        print(f"  {storage_type.upper()}: {batches}")

def example_detailed_tracking():
    """详细跟踪示例"""
    print("\n\n=== 详细跟踪示例 ===")
    
    tracker = get_global_tracker()
    if tracker is None:
        print("请先运行基本使用示例来初始化跟踪器")
        return
    
    # 查看block详细信息
    print("\n1. 所有block的详细信息:")
    all_blocks = tracker.get_block_details()
    for i, block in enumerate(all_blocks[:5]):  # 只显示前5个
        print(f"  Block {i+1}: Batch={block.batch_idx}, Layer={block.layer_idx}, "
              f"Block={block.block_idx}, Storage={block.storage_type.value}, "
              f"Importance={block.importance_score:.6f}, Access={block.access_count}")
    
    # 查看存储利用率
    print("\n2. 存储利用率:")
    utilization = tracker.get_storage_utilization()
    for storage_type, info in utilization.items():
        print(f"  {storage_type.upper()}: {info['used']}/{info['capacity']} "
              f"({info['utilization_rate']:.1%}) - 剩余: {info['free']}")
    
    # 查看操作历史
    print("\n3. 最近的操作历史:")
    history = tracker.get_operation_history(5)
    for op in history:
        print(f"  {op['operation']}: {op['details']} (batch={op['current_batch']}, layer={op['current_layer']})")

def example_advanced_operations():
    """高级操作示例"""
    print("\n\n=== 高级操作示例 ===")
    
    tracker = get_global_tracker()
    if tracker is None:
        print("请先运行基本使用示例来初始化跟踪器")
        return
    
    # 根据重要性查找blocks
    print("\n1. 查找重要性高于0.0的blocks:")
    important_blocks = tracker.find_blocks_by_importance(min_importance=0.0)
    for block in important_blocks[:3]:  # 只显示前3个
        print(f"  Batch={block.batch_idx}, Layer={block.layer_idx}, "
              f"Block={block.block_idx}, Importance={block.importance_score:.6f}")
    
    # 模拟清除某个batch的数据
    print("\n2. 清除Batch 1的数据前:")
    tracker.print_current_state()
    
    tracker.clear_batch_data(1)
    print("\n3. 清除Batch 1的数据后:")
    tracker.print_current_state()

def example_integration_with_model():
    """与模型集成示例"""
    print("\n\n=== 与模型集成示例 ===")
    
    # 模拟模型推理过程中的状态跟踪
    offloader = KVOffloader(
        layers=4,  # 简化为4层
        heads=8,
        dim=64,
        max_seq=512,
        max_batch=2,
        device="cuda:0",
        dtype_bytes=2
    )
    
    tracker = get_global_tracker()
    
    # 模拟推理过程
    print("\n模拟推理过程中的状态变化:")
    
    for batch_idx in range(2):
        for layer_idx in range(4):
            # 设置当前执行状态
            offloader.set_current_execution(batch_idx, layer_idx)
            
            # 模拟处理一些blocks
            k_tensor = torch.randn(2, 8, 64, dtype=torch.float16, device="cuda:0")
            v_tensor = torch.randn(2, 8, 64, dtype=torch.float16, device="cuda:0")
            
            # Push当前layer的数据
            offloader.push(layer=layer_idx, blk=0, k=k_tensor, v=v_tensor, batch_idx=batch_idx)
            
            print(f"  处理 Batch {batch_idx}, Layer {layer_idx}")
    
    print("\n最终状态:")
    offloader.print_global_state()
    
    # 显示每个batch的分布情况
    print("\n各batch的存储分布:")
    for batch_idx in range(2):
        print(f"\nBatch {batch_idx}:")
        dist = tracker.get_batch_distribution(batch_idx)
        for storage_type, layers in dist.items():
            if layers:
                print(f"  {storage_type.upper()}: {layers}")

def main():
    """主函数"""
    print("全局状态跟踪器使用示例")
    print("=" * 50)
    
    try:
        # 运行各种示例
        example_basic_usage()
        example_batch_analysis()
        example_detailed_tracking()
        example_advanced_operations()
        example_integration_with_model()
        
        print("\n" + "=" * 50)
        print("所有示例执行完成!")
        
    except Exception as e:
        print(f"\n执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()