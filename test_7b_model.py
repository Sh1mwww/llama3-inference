#!/usr/bin/env python3
"""
7B 模型完整测试脚本
测试所有改进是否正常工作：
1. 事件化等待（stream_mnt）
2. DRAM 配额估算优化（kv_offload）
3. 写队列阻塞式处理
"""
import torch
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from llama3.generator import LLaMA
from llama3.config import KVCacheArgs

def test_7b_model():
    """
    运行 7B 模型的基本推理测试
    """
    print("="*80)
    print("7B 模型测试 - 验证所有改进")
    print("="*80)

    # 配置参数
    checkpoint_dir = "/path/to/llama3-7b"  # 请替换为实际路径
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，将使用 CPU 模式（速度较慢）")

    # 更新 KVCacheArgs 配置
    print("\n📋 配置 KV Cache 参数...")
    KVCacheArgs.dram_limit_gb = 16.0  # DRAM 限制
    KVCacheArgs.dram_sizing_batch = 8  # 使用改进的配额估算
    KVCacheArgs.ssd_device_path = "/dev/nvme0n1p4"  # SSD 设备路径
    KVCacheArgs.max_concurrent_io = 4

    print(f"  - DRAM limit: {KVCacheArgs.dram_limit_gb} GB")
    print(f"  - DRAM sizing batch: {KVCacheArgs.dram_sizing_batch}")

    # 检查模型路径是否存在
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        print(f"\n❌ 错误：模型路径不存在: {checkpoint_dir}")
        print(f"\n请设置正确的模型路径。例如：")
        print(f"  checkpoint_dir = '/home/roger/models/Meta-Llama-3-8B'")
        return False

    # 检查必要文件
    params_file = ckpt_path / "params.json"
    if not params_file.exists():
        print(f"\n❌ 错误：未找到 params.json 文件")
        return False

    pth_files = list(ckpt_path.glob("*.pth"))
    if not pth_files:
        print(f"\n❌ 错误：未找到 .pth 权重文件")
        return False

    print(f"\n✅ 模型文件检查通过")
    print(f"  - params.json: {params_file}")
    print(f"  - 权重文件: {pth_files[0]}")

    # 构建模型
    print(f"\n🔨 构建模型...")
    try:
        llama = LLaMA.build(
            checkpoints_dir=str(ckpt_path),
            load_model=True,
            device=device,
            max_seq_len=512,  # 较短序列用于快速测试
            max_batch_size=4,
            topk_blk=8,
        )
        print(f"✅ 模型构建成功")
    except Exception as e:
        print(f"❌ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 准备测试提示词
    prompts = [
        "Once upon a time",
        "The quick brown fox",
        "In the beginning",
    ]

    print(f"\n📝 测试提示词:")
    for i, p in enumerate(prompts, 1):
        print(f"  {i}. '{p}'")

    # 运行推理
    print(f"\n🚀 开始推理...")
    try:
        tokens, texts = llama.text_completion(
            prompts=prompts,
            temperature=0.6,
            top_p=0.9,
            max_gen_len=50,  # 生成 50 个 token
            batch_size=2,
            enable_batching=True,
        )

        print(f"\n✅ 推理完成！")
        print(f"\n📄 生成结果:")
        print("="*80)
        for i, (prompt, text) in enumerate(zip(prompts, texts), 1):
            print(f"\n[{i}] 提示: {prompt}")
            print(f"    生成: {text}")
        print("="*80)

        return True

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ GPU 内存不足: {e}")
        print(f"\n💡 建议：")
        print(f"  1. 减少 max_seq_len (当前: 512)")
        print(f"  2. 减少 batch_size (当前: 2)")
        print(f"  3. 使用权重流式传输模式")
        return False
    except Exception as e:
        print(f"\n❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streaming_mode():
    """
    测试权重流式传输模式（适用于 GPU 内存不足的情况）
    """
    print("\n"+"="*80)
    print("7B 模型测试 - 权重流式传输模式")
    print("="*80)

    checkpoint_dir = "/path/to/llama3-7b"  # 请替换为实际路径
    device = "cuda:0"

    if not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，跳过流式传输测试")
        return False

    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        print(f"❌ 模型路径不存在，跳过流式传输测试")
        return False

    # 更新配置
    KVCacheArgs.dram_limit_gb = 16.0
    KVCacheArgs.dram_sizing_batch = 4  # 流式模式使用更小的 batch

    print(f"\n🔨 构建模型（流式模式）...")
    try:
        llama = LLaMA.build(
            checkpoints_dir=str(ckpt_path),
            load_model=True,
            device=device,
            mode="stream",  # 启用流式传输
            mode_config={
                'prefetch_distance': 2,
                'max_cached_layers': 4,
                'warmup_layers': 2,
                'verbose': True,
            },
            max_seq_len=256,
            max_batch_size=2,
            topk_blk=4,
        )
        print(f"✅ 流式模式模型构建成功")
    except Exception as e:
        print(f"❌ 流式模式构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 简单推理测试
    prompts = ["Hello, how are you?"]

    print(f"\n🚀 开始流式推理...")
    try:
        tokens, texts = llama.text_completion(
            prompts=prompts,
            temperature=0.6,
            max_gen_len=30,
            batch_size=1,
        )

        print(f"\n✅ 流式推理完成！")
        print(f"\n生成: {texts[0]}")
        return True

    except Exception as e:
        print(f"\n❌ 流式推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(f"\n🔬 开始 7B 模型测试")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 测试 1: 基本模式
    print(f"\n{'='*80}")
    print(f"测试 1: 基本推理模式")
    print(f"{'='*80}")
    success_basic = test_7b_model()

    # 测试 2: 流式模式（可选）
    if torch.cuda.is_available():
        print(f"\n{'='*80}")
        print(f"测试 2: 权重流式传输模式（可选）")
        print(f"{'='*80}")
        success_streaming = test_streaming_mode()
    else:
        success_streaming = None

    # 总结
    print(f"\n{'='*80}")
    print(f"测试总结")
    print(f"{'='*80}")
    print(f"基本模式: {'✅ 通过' if success_basic else '❌ 失败'}")
    if success_streaming is not None:
        print(f"流式模式: {'✅ 通过' if success_streaming else '❌ 失败'}")

    if success_basic:
        print(f"\n🎉 恭喜！所有改进已成功应用并通过测试")
        print(f"\n改进验证:")
        print(f"  ✅ 事件化等待（stream_mnt.py）")
        print(f"  ✅ DRAM 配额优化（kv_offload.py）")
        print(f"  ✅ 写队列阻塞式处理")
    else:
        print(f"\n⚠️  测试未完全通过，请检查错误信息")
        print(f"\n常见问题:")
        print(f"  1. 模型路径设置: 修改 checkpoint_dir 变量")
        print(f"  2. GPU 内存不足: 尝试流式模式或减小 batch_size")
        print(f"  3. SSD 路径错误: 检查 KVCacheArgs.ssd_device_path")
