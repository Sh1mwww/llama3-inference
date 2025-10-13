#!/usr/bin/env python3
"""
完整压力测试脚本：使用 raw param store 对 llama3.1-8b 进行全层多次验证
- 遍历所有层（32 层）
- 多次加载验证
- 带 evict 机制防止 OOM
- 测量加载速度和吞吐量
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import time
import gc
import psutil
import torch
from llama3.raw_param_store import ParamStore


def get_memory_info():
    """获取当前内存使用情况"""
    process = psutil.Process()
    rss_mb = process.memory_info().rss / (1024**2)

    # 系统内存
    vm = psutil.virtual_memory()
    total_mb = vm.total / (1024**2)
    available_mb = vm.available / (1024**2)
    used_percent = vm.percent

    return {
        "process_rss_mb": rss_mb,
        "system_total_mb": total_mb,
        "system_available_mb": available_mb,
        "system_used_percent": used_percent
    }


def evict_layer(tensors_dict):
    """
    释放层权重，防止 OOM
    """
    if tensors_dict:
        for tensor in tensors_dict.values():
            del tensor
        tensors_dict.clear()

    # 强制垃圾回收
    gc.collect()

    # 如果有 CUDA，也清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_single_layer(store, layer_id, check_integrity=True, verbose=False):
    """
    测试单个层的加载

    Returns:
        dict: 包含 success, load_time_ms, size_mb, num_params, tensors
    """
    mem_before = get_memory_info()

    start_time = time.perf_counter()

    try:
        # 加载层
        tensors = store.fetch_layer(layer_id=layer_id, only_stream=True)

        load_time = (time.perf_counter() - start_time) * 1000  # ms

        if not tensors:
            return {
                "success": False,
                "error": "No stream tensors found",
                "load_time_ms": load_time,
            }

        # 计算大小
        total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        size_mb = total_bytes / (1024**2)
        num_params = len(tensors)

        # 可选：完整性校验
        integrity_ok = True
        if check_integrity:
            matched, total = store.sanity_check_layer(
                layer_id=layer_id,
                tensors=tensors,
                check_bytes=64,
                verbose=False
            )
            integrity_ok = (matched == total)

        mem_after = get_memory_info()
        mem_delta_mb = mem_after["process_rss_mb"] - mem_before["process_rss_mb"]

        result = {
            "success": True,
            "load_time_ms": load_time,
            "size_mb": size_mb,
            "num_params": num_params,
            "throughput_mbps": size_mb / (load_time / 1000) if load_time > 0 else 0,
            "integrity_ok": integrity_ok,
            "mem_delta_mb": mem_delta_mb,
            "mem_before_mb": mem_before["process_rss_mb"],
            "mem_after_mb": mem_after["process_rss_mb"],
            "tensors": tensors,
        }

        if verbose:
            print(f"   Layer {layer_id}: {size_mb:.2f} MB, "
                  f"{load_time:.2f} ms, {result['throughput_mbps']:.2f} MB/s, "
                  f"integrity={'✅' if integrity_ok else '❌'}")

        return result

    except Exception as e:
        load_time = (time.perf_counter() - start_time) * 1000
        return {
            "success": False,
            "error": str(e),
            "load_time_ms": load_time,
        }


def test_all_layers_sequential(store, num_layers, check_integrity=True, evict_after_load=True):
    """
    顺序测试所有层（带 evict 防止 OOM）

    Args:
        store: ParamStore 实例
        num_layers: 总层数
        check_integrity: 是否进行完整性校验
        evict_after_load: 是否在加载后立即释放（防止 OOM）

    Returns:
        dict: 测试结果统计
    """
    print(f"\n{'='*70}")
    print(f"📊 顺序加载所有 {num_layers} 层（evict={evict_after_load}）")
    print(f"{'='*70}")

    results = []
    total_size_mb = 0
    total_time_ms = 0
    failed_layers = []

    mem_start = get_memory_info()
    print(f"初始内存: 进程 {mem_start['process_rss_mb']:.2f} MB, "
          f"系统可用 {mem_start['system_available_mb']:.2f} MB "
          f"({mem_start['system_used_percent']:.1f}% 已使用)")

    for layer_id in range(num_layers):
        result = test_single_layer(store, layer_id, check_integrity=check_integrity, verbose=True)
        results.append(result)

        if result["success"]:
            total_size_mb += result["size_mb"]
            total_time_ms += result["load_time_ms"]

            # 释放内存防止 OOM
            if evict_after_load:
                evict_layer(result.get("tensors", {}))
                result["tensors"] = None  # 清除引用
        else:
            failed_layers.append(layer_id)
            print(f"   ❌ Layer {layer_id} 失败: {result.get('error', 'Unknown')}")

    mem_end = get_memory_info()
    mem_growth_mb = mem_end["process_rss_mb"] - mem_start["process_rss_mb"]

    # 统计
    success_count = sum(1 for r in results if r["success"])
    avg_load_time_ms = total_time_ms / success_count if success_count > 0 else 0
    avg_throughput_mbps = sum(r.get("throughput_mbps", 0) for r in results if r["success"]) / success_count if success_count > 0 else 0

    integrity_failures = sum(1 for r in results if r["success"] and not r.get("integrity_ok", True))

    print(f"\n{'='*70}")
    print(f"📈 顺序加载统计:")
    print(f"   成功加载: {success_count}/{num_layers} 层")
    print(f"   总数据量: {total_size_mb:.2f} MB")
    print(f"   总耗时: {total_time_ms:.2f} ms ({total_time_ms/1000:.2f} s)")
    print(f"   平均每层: {avg_load_time_ms:.2f} ms")
    print(f"   平均吞吐: {avg_throughput_mbps:.2f} MB/s")
    print(f"   完整性校验: {success_count - integrity_failures}/{success_count} 通过")
    print(f"   内存增长: {mem_growth_mb:+.2f} MB")
    print(f"   最终内存: 进程 {mem_end['process_rss_mb']:.2f} MB, "
          f"系统可用 {mem_end['system_available_mb']:.2f} MB")

    if failed_layers:
        print(f"   ❌ 失败的层: {failed_layers}")

    print(f"{'='*70}")

    return {
        "success_count": success_count,
        "total_layers": num_layers,
        "failed_layers": failed_layers,
        "total_size_mb": total_size_mb,
        "total_time_ms": total_time_ms,
        "avg_load_time_ms": avg_load_time_ms,
        "avg_throughput_mbps": avg_throughput_mbps,
        "integrity_failures": integrity_failures,
        "mem_growth_mb": mem_growth_mb,
        "results": results,
    }


def test_random_access_pattern(store, num_layers, num_iterations=3):
    """
    测试随机访问模式（模拟实际推理场景）
    每次只保留当前层，立即释放

    Args:
        store: ParamStore 实例
        num_layers: 总层数
        num_iterations: 完整遍历次数
    """
    print(f"\n{'='*70}")
    print(f"🔀 随机访问测试（{num_iterations} 次完整遍历）")
    print(f"{'='*70}")

    import random

    all_load_times = []
    all_throughputs = []
    iteration_stats = []

    for iteration in range(num_iterations):
        print(f"\n--- 第 {iteration+1}/{num_iterations} 次遍历 ---")

        # 生成随机访问顺序
        layer_order = list(range(num_layers))
        random.shuffle(layer_order)

        iteration_start = time.perf_counter()
        iteration_size = 0
        success_in_iteration = 0

        for layer_id in layer_order:
            result = test_single_layer(store, layer_id, check_integrity=False, verbose=False)

            if result["success"]:
                all_load_times.append(result["load_time_ms"])
                all_throughputs.append(result["throughput_mbps"])
                iteration_size += result["size_mb"]
                success_in_iteration += 1

                # 立即释放
                evict_layer(result.get("tensors", {}))

            # 每 8 层打印一次进度
            if (layer_order.index(layer_id) + 1) % 8 == 0:
                progress = (layer_order.index(layer_id) + 1) / num_layers * 100
                print(f"   进度: {progress:.1f}% ({layer_order.index(layer_id)+1}/{num_layers} 层)")

        iteration_time = (time.perf_counter() - iteration_start) * 1000
        iteration_throughput = iteration_size / (iteration_time / 1000) if iteration_time > 0 else 0

        iteration_stats.append({
            "iteration": iteration + 1,
            "success_count": success_in_iteration,
            "total_size_mb": iteration_size,
            "total_time_ms": iteration_time,
            "throughput_mbps": iteration_throughput,
        })

        print(f"   ✅ 完成: {success_in_iteration}/{num_layers} 层, "
              f"{iteration_size:.2f} MB, {iteration_time:.2f} ms, "
              f"{iteration_throughput:.2f} MB/s")

    # 总结统计
    print(f"\n{'='*70}")
    print(f"📊 随机访问统计:")
    print(f"   总加载次数: {len(all_load_times)}")
    print(f"   平均加载时间: {sum(all_load_times)/len(all_load_times):.2f} ms")
    print(f"   最小/最大加载时间: {min(all_load_times):.2f} / {max(all_load_times):.2f} ms")
    print(f"   平均吞吐量: {sum(all_throughputs)/len(all_throughputs):.2f} MB/s")
    print(f"   最小/最大吞吐量: {min(all_throughputs):.2f} / {max(all_throughputs):.2f} MB/s")

    print(f"\n   各次遍历对比:")
    for stat in iteration_stats:
        print(f"      第 {stat['iteration']} 次: {stat['total_time_ms']:.2f} ms, "
              f"{stat['throughput_mbps']:.2f} MB/s")

    print(f"{'='*70}")

    return {
        "num_iterations": num_iterations,
        "total_loads": len(all_load_times),
        "avg_load_time_ms": sum(all_load_times) / len(all_load_times) if all_load_times else 0,
        "avg_throughput_mbps": sum(all_throughputs) / len(all_throughputs) if all_throughputs else 0,
        "iteration_stats": iteration_stats,
    }


def test_concurrent_loading(store, num_layers, batch_size=4):
    """
    测试并发批量加载（使用 fetch_layer_batch）
    """
    print(f"\n{'='*70}")
    print(f"⚡ 并发批量加载测试（batch_size={batch_size}）")
    print(f"{'='*70}")

    batches = [list(range(i, min(i + batch_size, num_layers)))
               for i in range(0, num_layers, batch_size)]

    total_size_mb = 0
    total_time_ms = 0
    successful_batches = 0

    mem_start = get_memory_info()

    for batch_idx, batch_layers in enumerate(batches):
        print(f"\n批次 {batch_idx+1}/{len(batches)}: 层 {batch_layers}")

        start_time = time.perf_counter()

        try:
            # 并发加载
            batch_tensors = store.fetch_layer_batch(batch_layers, only_stream=True)

            load_time = (time.perf_counter() - start_time) * 1000

            # 计算大小
            batch_size_mb = 0
            for layer_id, tensors in batch_tensors.items():
                layer_size = sum(t.numel() * t.element_size() for t in tensors.values())
                batch_size_mb += layer_size / (1024**2)

            throughput = batch_size_mb / (load_time / 1000) if load_time > 0 else 0

            print(f"   ✅ 加载成功: {len(batch_tensors)} 层, "
                  f"{batch_size_mb:.2f} MB, {load_time:.2f} ms, "
                  f"{throughput:.2f} MB/s")

            total_size_mb += batch_size_mb
            total_time_ms += load_time
            successful_batches += 1

            # 释放
            for tensors in batch_tensors.values():
                evict_layer(tensors)

        except Exception as e:
            print(f"   ❌ 批次加载失败: {e}")

    mem_end = get_memory_info()

    print(f"\n{'='*70}")
    print(f"📊 并发加载统计:")
    print(f"   成功批次: {successful_batches}/{len(batches)}")
    print(f"   总数据量: {total_size_mb:.2f} MB")
    print(f"   总耗时: {total_time_ms:.2f} ms")
    print(f"   平均吞吐: {total_size_mb / (total_time_ms/1000) if total_time_ms > 0 else 0:.2f} MB/s")
    print(f"   内存变化: {mem_end['process_rss_mb'] - mem_start['process_rss_mb']:+.2f} MB")
    print(f"{'='*70}")


def main():
    """主测试流程"""

    print("=" * 70)
    print("🚀 LLaMA3.1-8B Raw Param Store 完整压力测试")
    print("=" * 70)

    manifest_path = "/data1/llama3.1-8B.runtime_manifest.json"

    try:
        # 创建 ParamStore
        print("\n[1/6] 初始化 ParamStore...")
        store = ParamStore(
            manifest_or_path=manifest_path,
            method="bytecopy",
            staging_mb=32,
            rw=False,
            max_concurrent_io=4
        )
        print("✅ ParamStore 创建成功")

        # 获取模型信息
        stats = store.get_storage_stats()
        print(f"\n📋 模型信息:")
        print(f"   总参数: {stats['total_params']}")
        print(f"   总大小: {stats['total_gb']:.2f} GB")
        print(f"   Stream: {stats['stream_gb']:.2f} GB ({stats['stream_bytes']/stats['total_bytes']*100:.1f}%)")
        print(f"   Raw 设备: {stats['raw_device']}")

        # 确定层数
        num_layers = 32  # llama3.1-8b
        print(f"   模型层数: {num_layers}")

        # 测试 1: 顺序加载所有层（带 evict）
        print("\n[2/6] 测试 1: 顺序加载所有层（带 evict）")
        seq_result = test_all_layers_sequential(
            store, num_layers,
            check_integrity=True,
            evict_after_load=True
        )

        # 强制清理
        gc.collect()
        time.sleep(1)

        # 测试 2: 随机访问模式（多次遍历）
        print("\n[3/6] 测试 2: 随机访问模式")
        random_result = test_random_access_pattern(store, num_layers, num_iterations=3)

        gc.collect()
        time.sleep(1)

        # 测试 3: 并发批量加载
        print("\n[4/6] 测试 3: 并发批量加载")
        test_concurrent_loading(store, num_layers, batch_size=4)

        gc.collect()
        time.sleep(1)

        # 测试 4: 不带 evict 的顺序加载（测试内存压力）
        print("\n[5/6] 测试 4: 不带 evict 的顺序加载（测试内存压力）")
        print("⚠️  警告: 此测试可能消耗大量内存")
        mem_before = get_memory_info()
        if mem_before["system_available_mb"] < 15000:  # 少于 15GB 可用内存
            print(f"❌ 跳过：系统可用内存不足 ({mem_before['system_available_mb']:.2f} MB)")
        else:
            no_evict_result = test_all_layers_sequential(
                store, num_layers,
                check_integrity=False,
                evict_after_load=False
            )
            print(f"   保留所有层后的内存: {get_memory_info()['process_rss_mb']:.2f} MB")

            # 清理
            gc.collect()

        # 测试 5: 异步加载压力测试
        print("\n[6/6] 测试 5: 异步加载压力测试")
        print("提交多个异步加载任务...")

        futures = []
        for layer_id in range(min(8, num_layers)):
            future = store.fetch_layer_async(layer_id, only_stream=True)
            futures.append((layer_id, future))

        print(f"   提交了 {len(futures)} 个异步任务")

        for layer_id, future in futures:
            result = future.result()
            size_mb = sum(t.numel() * t.element_size() for t in result.values()) / (1024**2)
            print(f"   ✅ Layer {layer_id}: {size_mb:.2f} MB")
            evict_layer(result)

        # 最终总结
        print(f"\n{'='*70}")
        print(f"🎉 所有压力测试完成！")
        print(f"{'='*70}")
        print(f"顺序加载: {seq_result['success_count']}/{seq_result['total_layers']} 层成功")
        print(f"   平均耗时: {seq_result['avg_load_time_ms']:.2f} ms/层")
        print(f"   平均吞吐: {seq_result['avg_throughput_mbps']:.2f} MB/s")
        print(f"   完整性: {seq_result['success_count'] - seq_result['integrity_failures']}/{seq_result['success_count']} 通过")
        print(f"\n随机访问: {random_result['num_iterations']} 次遍历")
        print(f"   平均耗时: {random_result['avg_load_time_ms']:.2f} ms/层")
        print(f"   平均吞吐: {random_result['avg_throughput_mbps']:.2f} MB/s")
        print(f"{'='*70}")

        # 关闭
        store.close()
        print("\n✅ ParamStore 已关闭")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
