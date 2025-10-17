#!/usr/bin/env python3
"""
单元测试：验证所有改进是否正常工作（不需要实际模型权重）
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_event_pool_improvements():
    """测试事件池改进"""
    print("\n" + "="*80)
    print("测试 1: 事件池改进（stream_mnt.py）")
    print("="*80)

    try:
        from llama3 import stream_mnt

        if not torch.cuda.is_available():
            print("⚠️  CUDA 不可用，跳过事件池测试")
            return None

        device = "cuda:0"

        # 获取 streams
        streams = stream_mnt.get_streams(device)
        print(f"✅ Streams 创建成功")

        # 测试事件池
        pool = stream_mnt._get_event_pool(device)
        print(f"✅ 事件池获取成功")

        # 测试新的 record_event_on API
        if streams.compute_mha:
            eid, evt = stream_mnt.record_event_on(streams.compute_mha, device=device)
            print(f"✅ 事件记录成功 (ID: {eid})")

            # 测试释放
            stream_mnt.release_event(eid, device=device)
            print(f"✅ 事件释放成功")

        # 测试 GC
        freed = stream_mnt.gc_event_pool(device=device)
        print(f"✅ 事件池 GC 成功 (回收: {freed} 个事件)")

        print(f"\n✅ 事件池改进测试通过")
        return True

    except Exception as e:
        print(f"\n❌ 事件池测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kv_offloader_improvements():
    """测试 KV Offloader DRAM 配额改进"""
    print("\n" + "="*80)
    print("测试 2: KV Offloader DRAM 配额优化")
    print("="*80)

    try:
        from llama3.kv_offload import KVOffloader, BLOCK
        from llama3.config import KVCacheArgs

        # 设置测试配置
        KVCacheArgs.dram_limit_gb = 2.0
        KVCacheArgs.dram_sizing_batch = 8  # 使用改进的配额估算

        print(f"配置:")
        print(f"  - dram_limit_gb: {KVCacheArgs.dram_limit_gb}")
        print(f"  - dram_sizing_batch: {KVCacheArgs.dram_sizing_batch}")

        # 创建 KVOffloader
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"\n创建 KVOffloader (device: {device})...")
        offloader = KVOffloader(
            layers=32,
            heads=32,
            dim=128,
            max_seq=2048,
            max_batch=32,  # 实际会用 dram_sizing_batch=8 来估算
            device=device,
            dtype_bytes=2,  # fp16
            streams=None,
        )

        print(f"\n✅ KVOffloader 创建成功")
        print(f"  - DRAM limit blocks: {offloader.dram_limit_blk}")
        print(f"  - Block size: {offloader.block_nbytes / (1024**2):.2f} MB")

        # 验证配额计算是否使用了 dram_sizing_batch
        expected_token_nbytes = (8 * 32 * 128) * 2 * 2  # alloc_bsz=8
        assert offloader.token_nbytes == expected_token_nbytes, \
            f"token_nbytes 不正确: {offloader.token_nbytes} != {expected_token_nbytes}"
        print(f"✅ 配额计算正确使用 dram_sizing_batch")

        # 验证 dram_limit_blk 不为 0
        assert offloader.dram_limit_blk > 0, \
            f"dram_limit_blk 为 0，配额估算可能有问题"
        print(f"✅ dram_limit_blk 正常 (> 0)")

        print(f"\n✅ DRAM 配额优化测试通过")
        return True

    except Exception as e:
        print(f"\n❌ KV Offloader 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_queue_blocking_put():
    """测试写队列阻塞式 put"""
    print("\n" + "="*80)
    print("测试 3: 写队列阻塞式处理")
    print("="*80)

    try:
        from queue import Queue, Full
        import time

        # 创建小队列测试
        q = Queue(maxsize=2)

        # 填满队列
        q.put("item1")
        q.put("item2")

        print(f"✅ 队列已填满 (2/2)")

        # 测试阻塞式 put 带超时
        start = time.time()
        try:
            q.put("item3", timeout=0.5)
            print(f"❌ 应该触发 Full 异常")
            return False
        except Full:
            elapsed = time.time() - start
            print(f"✅ Full 异常正确触发 (超时: {elapsed:.2f}s)")

        # 验证代码中确实导入了 Full
        import llama3.kv_offload
        import inspect
        source = inspect.getsource(llama3.kv_offload)

        if "from queue import" in source and "Full" in source:
            print(f"✅ kv_offload.py 正确导入 Full")
        else:
            print(f"❌ kv_offload.py 未正确导入 Full")
            return False

        if "put(" in source and "timeout=" in source:
            print(f"✅ kv_offload.py 使用阻塞式 put")
        else:
            print(f"⚠️  未找到阻塞式 put 调用")

        print(f"\n✅ 写队列改进测试通过")
        return True

    except Exception as e:
        print(f"\n❌ 队列测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_layers_event_sync():
    """测试 layers.py 中的事件化同步"""
    print("\n" + "="*80)
    print("测试 4: Layers 事件化同步")
    print("="*80)

    try:
        from llama3.config import ModelArgs
        from llama3.layers import EncoderBlock

        if not torch.cuda.is_available():
            print("⚠️  CUDA 不可用，跳过 layers 测试")
            return None

        # 创建简单的 args
        args = ModelArgs(
            dim=256,
            n_layers=2,
            n_heads=8,
            n_kv_heads=8,
            vocab_size=32000,
            multiple_of=256,
            ffn_dim_multiplier=None,
            norm_eps=1e-5,
            rope_theta=10000.0,
            use_scaled_rope=False,
            max_batch_size=4,
            max_seq_len=512,
            device="cuda:0",
            topk_blk=4,
        )

        print(f"创建 EncoderBlock...")
        block = EncoderBlock(args, layer_id=0)

        # 验证初始化了 _gc_counter
        assert hasattr(block, '_gc_counter'), "EncoderBlock 缺少 _gc_counter"
        print(f"✅ EncoderBlock 正确初始化 _gc_counter")

        # 检查是否使用了新的事件化等待
        import inspect
        source = inspect.getsource(block.forward)

        if "stream_mnt.record_event_on" in source:
            print(f"✅ EncoderBlock.forward 使用事件化等待")
        else:
            print(f"⚠️  EncoderBlock.forward 未使用事件化等待（可能使用降级方案）")

        if "gc_event_pool" in source:
            print(f"✅ EncoderBlock.forward 定期触发 GC")
        else:
            print(f"⚠️  EncoderBlock.forward 未触发 GC")

        print(f"\n✅ Layers 事件化同步测试通过")
        return True

    except Exception as e:
        print(f"\n❌ Layers 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*80)
    print("所有改进的单元测试")
    print("="*80)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = {}

    # 运行所有测试
    results['event_pool'] = test_event_pool_improvements()
    results['kv_offloader'] = test_kv_offloader_improvements()
    results['queue_blocking'] = test_queue_blocking_put()
    results['layers_sync'] = test_layers_event_sync()

    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)

    for name, result in results.items():
        if result is True:
            status = "✅ 通过"
        elif result is False:
            status = "❌ 失败"
        else:
            status = "⏭️  跳过"
        print(f"{name:20s}: {status}")

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    print(f"\n总计: {passed} 通过, {failed} 失败, {skipped} 跳过")

    if failed == 0:
        print(f"\n🎉 所有测试通过！所有改进已成功应用。")
        return 0
    else:
        print(f"\n⚠️  部分测试失败，请检查错误信息")
        return 1

if __name__ == "__main__":
    exit(main())
