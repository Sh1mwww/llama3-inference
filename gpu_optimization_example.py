#!/usr/bin/env python3
"""
GPU优化示例代码
展示如何使用新的GPU错误处理和HBM内存限制功能
"""

import torch
from llama3.gpu_utils import (
    GPUHealthMonitor, 
    SafeGPUManager, 
    gpu_safe_operation,
    gpu_memory_guard,
    get_optimal_device
)
from llama3.memory_manager import (
    hbm_memory_limit,
    set_global_memory_limit,
    get_memory_info,
    MemoryConfig,
    HBMMemoryManager
)
from llama3.config import ModelArgs, MemoryLimitArgs

def check_gpu_status():
    """检查GPU状态示例"""
    print("=== GPU状态检查 ===")
    
    monitor = GPUHealthMonitor()
    
    if monitor.gpu_available:
        for i in range(monitor.device_count):
            health = monitor.check_gpu_health(i)
            print(f"GPU {i}: {health['status']} - {health['message']}")
            
            if 'memory_info' in health:
                mem = health['memory_info']
                print(f"  内存: {mem['free']/(1024**3):.2f}GB 可用 / {mem['total']/(1024**3):.2f}GB 总计")
    else:
        print("CUDA不可用")

@gpu_safe_operation(retry_count=2)
def safe_tensor_operation():
    """安全的张量操作示例"""
    print("\n=== 安全张量操作 ===")
    
    device = get_optimal_device()
    print(f"使用设备: {device}")
    
    # 创建大张量进行测试
    try:
        with gpu_memory_guard(device, threshold_gb=0.5):
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            
            result = torch.mm(x, y)
            print(f"矩阵乘法结果形状: {result.shape}")
            
            # 清理
            del x, y, result
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
                
    except Exception as e:
        print(f"操作失败: {e}")

def test_config_with_gpu_optimization():
    """测试配置文件的GPU优化"""
    print("\n=== 配置GPU优化测试 ===")
    
    try:
        # 测试自动设备选择
        args = ModelArgs.from_json(
            params_path="params.json",  # 假设存在
            max_seq_len=2048,
            max_batch_size=16,
            device=None  # 自动选择
        )
        print(f"自动选择的设备: {args.device}")
        
    except FileNotFoundError:
        print("params.json文件不存在，跳过配置测试")
    except Exception as e:
        print(f"配置测试失败: {e}")

def test_safe_gpu_manager():
    """测试安全GPU管理器"""
    print("\n=== 安全GPU管理器测试 ===")
    
    manager = SafeGPUManager("cuda", auto_fallback=True)
    print(f"当前设备: {manager.current_device}")
    
    device_id = 0
    try:
        with manager.safe_cuda_context(device_id):
            print("安全CUDA上下文创建成功")
            
            # 模拟一些GPU操作
            if manager.current_device.startswith("cuda"):
                test_tensor = torch.randn(100, 100, device=manager.current_device)
                print(f"测试张量创建成功: {test_tensor.shape}")
                del test_tensor
                
    except Exception as e:
        print(f"安全上下文测试失败: {e}")

def test_hbm_memory_limit():
    """测试HBM内存限制"""
    print("\n=== HBM内存限制测试 ===")
    
    device = get_optimal_device()
    if not device.startswith("cuda"):
        print("非CUDA设备，跳过HBM测试")
        return
    
    # 获取GPU总内存
    mem_info = get_memory_info(device)
    print(f"GPU总内存: {mem_info.get('total_gb', 0):.2f}GB")
    
    # 设置较小的内存限制进行测试
    limit_gb = 2.0  # 限制为2GB
    print(f"设置内存限制: {limit_gb}GB")
    
    try:
        with hbm_memory_limit(limit_gb, device, reserved_gb=0.5) as manager:
            print("HBM内存管理器创建成功")
            
            # 显示内存信息
            info = manager.get_memory_info()
            print(f"可用内存: {info['available_gb']:.2f}GB")
            print(f"当前使用: {info['current_allocated_gb']:.2f}GB")
            
            # 测试安全张量分配
            print("\n测试安全张量分配...")
            try:
                with manager.allocate_tensor((1000, 1000), torch.float32) as tensor:
                    print(f"成功分配张量: {tensor.shape}")
                    
                    # 显示分配后的内存状态
                    info = manager.get_memory_info()
                    print(f"分配后使用: {info['current_allocated_gb']:.2f}GB")
                    
            except Exception as e:
                print(f"张量分配失败: {e}")
            
            # 测试内存限制
            print("\n测试内存限制...")
            large_size = (5000, 5000)  # 很大的张量
            can_alloc = manager.can_allocate(manager.estimate_tensor_size(large_size))
            print(f"是否可以分配{large_size}张量: {can_alloc}")
            
    except Exception as e:
        print(f"HBM测试失败: {e}")

def test_global_memory_limit():
    """测试全局内存限制"""
    print("\n=== 全局内存限制测试 ===")
    
    device = get_optimal_device()
    if not device.startswith("cuda"):
        print("非CUDA设备，跳过全局内存限制测试")
        return
    
    try:
        # 设置全局内存限制
        set_global_memory_limit(3.0, device, reserved_gb=1.0)
        print("全局内存限制设置成功: 3.0GB")
        
        # 显示内存信息
        info = get_memory_info(device)
        print(f"最大允许: {info.get('max_allowed_gb', 0):.2f}GB")
        print(f"当前可用: {info.get('can_allocate_gb', 0):.2f}GB")
        
    except Exception as e:
        print(f"全局内存限制测试失败: {e}")

def test_memory_with_model_config():
    """测试模型配置的内存管理"""
    print("\n=== 模型配置内存管理测试 ===")
    
    try:
        # 创建内存限制配置
        memory_config = MemoryLimitArgs(
            max_hbm_gb=4.0,
            reserved_hbm_gb=1.0,
            enable_monitoring=True,
            auto_limit=False
        )
        
        print(f"内存配置: 最大{memory_config.max_hbm_gb}GB, 预留{memory_config.reserved_hbm_gb}GB")
        
        # 这里可以测试模型配置，但需要实际的params.json文件
        print("内存配置创建成功")
        
    except Exception as e:
        print(f"模型配置测试失败: {e}")

def run_optimization_tests():
    """运行所有优化测试"""
    print("开始GPU优化测试...\n")
    
    # 1. 检查GPU状态
    check_gpu_status()
    
    # 2. 测试安全张量操作
    safe_tensor_operation()
    
    # 3. 测试HBM内存限制
    test_hbm_memory_limit()
    
    # 4. 测试全局内存限制
    test_global_memory_limit()
    
    # 5. 测试模型配置内存管理
    test_memory_with_model_config()
    
    # 6. 测试配置优化
    test_config_with_gpu_optimization()
    
    # 7. 测试安全GPU管理器
    test_safe_gpu_manager()
    
    print("\n=== 测试完成 ===")
    print("HBM内存限制优化建议:")
    print("1. 设置合理的HBM内存限制(如总内存的80%)")
    print("2. 预留足够的内存给系统和其他进程(如20%)")
    print("3. 启用内存监控以实时跟踪使用情况")
    print("4. 在模型初始化时自动设置内存限制")
    print("5. 使用内存保护上下文防止OOM错误")
    print("6. 定期清理GPU缓存释放内存碎片")

def main():
    """主函数 - 展示如何在实际项目中使用HBM限制"""
    print("=== HBM内存限制实际使用示例 ===")
    
    # 示例1: 为整个项目设置全局内存限制
    device = get_optimal_device()
    if device.startswith("cuda"):
        try:
            # 自动检测并设置为80%的显存
            mem_info = get_memory_info(device)
            total_gb = mem_info.get('total_gb', 8.0)
            limit_gb = total_gb * 0.8
            
            set_global_memory_limit(limit_gb, device, reserved_gb=total_gb * 0.2)
            print(f"✓ 全局内存限制已设置: {limit_gb:.1f}GB (预留: {total_gb * 0.2:.1f}GB)")
            
        except Exception as e:
            print(f"✗ 全局内存限制设置失败: {e}")
    
    # 示例2: 在特定任务中使用临时内存限制
    print("\n--- 临时内存限制示例 ---")
    try:
        with hbm_memory_limit(2.0, device, reserved_gb=0.5) as manager:
            print("✓ 临时内存限制生效: 2.0GB")
            
            # 在这个上下文中，所有GPU操作都受到2GB限制
            info = manager.get_memory_info()
            print(f"  可用: {info['available_gb']:.2f}GB")
            
    except Exception as e:
        print(f"✗ 临时内存限制失败: {e}")
    
    print("\n--- 运行完整测试 ---")
    run_optimization_tests()

if __name__ == "__main__":
    main()