#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
70B 模型 Prefill 阶段测试脚本 (优化for 16GB 显存)

专门测试 70B 模型的 prefill 阶段性能和权重流式传输效果。
不进行 decode 阶段，只测试一次性处理长序列的能力。

针对 16GB 显存的配置优化：
- 极小的 GPU 缓存层数 (1层)
- 较小的 batch size (1)
- 中等的序列长度 (256 tokens)

使用方式：
    python test_70b_prefill.py
"""
import sys
from pathlib import Path
import torch
import time

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent))

from llama3.generator import LLaMA
from llama3.config import KVCacheArgs


def test_70b_prefill():
    """
    测试 70B 模型的 prefill 阶段 (16GB 显存优化配置)
    """
    print("=" * 80)
    print("70B 模型 Prefill 测试 - 权重流式传输模式 (16GB 显存)")
    print("=" * 80)

    # ---- 模型路径配置 ----
    checkpoint_dir = "/home/roger/.llama/checkpoints/Llama3.1-70B/"
    ckpt_path = Path(checkpoint_dir)

    # 检查模型路径
    if not ckpt_path.exists():
        print(f"\n❌ 错误：模型路径不存在: {checkpoint_dir}")
        print("请设置正确的 70B 模型路径")
        return False

    # 检查 params.json
    params_file = ckpt_path / "params.json"
    if not params_file.exists():
        print(f"\n❌ 错误：未找到 params.json 文件: {params_file}")
        return False

    # 检查权重文件
    pth_files = list(ckpt_path.glob("*.pth"))
    if not pth_files:
        print(f"\n❌ 错误：未找到 .pth 权重文件: {ckpt_path}")
        return False

    print("\n✅ 模型文件检查通过")
    print(f"  - params.json: {params_file}")
    print(f"  - 权重文件: {len(pth_files)} 个")

    # ---- 设备配置 ----
    if not torch.cuda.is_available():
        print("\n❌ 错误：CUDA 不可用，无法测试 70B 模型")
        return False

    device = "cuda:0"

    # 打印 GPU 信息
    print(f"\n📊 GPU 信息:")
    try:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  - 设备: {gpu_name}")
        print(f"  - 显存: {gpu_mem_gb:.2f} GB")

        if gpu_mem_gb < 16:
            print(f"\n⚠️  警告：GPU 显存小于 16GB，测试可能失败")
            return False
    except Exception as e:
        print(f"\n❌ 错误：无法获取 GPU 信息: {e}")
        return False

    # ---- KV Cache 参数配置 (16GB 优化) ----
    print("\n⚙️  配置 KV Cache 参数 (16GB 显存优化)...")
    try:
        KVCacheArgs.dram_limit_gb = 16.0              # DRAM 限制
        KVCacheArgs.dram_sizing_batch = 2             # 小批量估算
        KVCacheArgs.ssd_device_path = "/dev/nvme0n1p4"  # SSD 路径（可选）
        KVCacheArgs.max_concurrent_io = 4
        print(f"  - DRAM limit: {KVCacheArgs.dram_limit_gb} GB")
        print(f"  - DRAM sizing batch: {KVCacheArgs.dram_sizing_batch}")
    except Exception as e:
        print(f"\n❌ 错误：KV Cache 配置失败: {e}")
        return False

    # ---- Prefill 测试参数 (16GB 优化) ----
    prefill_seq_len = 256  # 较小的序列长度
    batch_size = 1         # 单batch

    print(f"\n📝 Prefill 测试参数 (16GB 优化):")
    print(f"  - Sequence length: {prefill_seq_len}")
    print(f"  - Batch size: {batch_size}")

    # ---- 构建模型（权重流式传输模式）----
    print("\n🔨 构建 70B 模型（权重流式传输模式）...")

    # 70B 模型流式配置 (16GB 显存优化)
    # 根据你的 GPU 内存情况调整:
    # - 如果有 20GB+: max_cached_layers=3-4, prefetch_distance=2-3
    # - 如果是 16GB: max_cached_layers=1-2, prefetch_distance=2
    # - 如果经常 OOM: max_cached_layers=1, batch_size=1, seq_len=128

    streaming_config = {
        "prefetch_distance": 2,      # 预取距离 (可以稍大，不占GPU内存)
        "max_cached_layers": 1,      # GPU缓存层数 (70B每层约1.8GB)
        "warmup_layers": 0,          # 不预热（节省初始化时间）
        "verbose": True,             # 启用详细日志
    }

    print(f"\n📦 流式配置 (16GB 显存):")
    print(f"  - Prefetch distance: {streaming_config['prefetch_distance']} (异步预取)")
    print(f"  - Max cached layers: {streaming_config['max_cached_layers']} (每层约1.8GB FP16)")
    print(f"  - Warmup layers: {streaming_config['warmup_layers']}")
    print(f"  - 预计峰值显存: ~5GB (核心) + {streaming_config['max_cached_layers']*1.8:.1f}GB (层缓存) + 2-3GB (KV+激活)")
    print(f"  - 总计预计: ~{5 + streaming_config['max_cached_layers']*1.8 + 2.5:.1f}GB")

    llama = None
    try:
        start_time = time.time()
        llama = LLaMA.build(
            checkpoints_dir=str(ckpt_path),
            load_model=True,
            device=device,
            mode="stream",
            mode_config=streaming_config,
            max_seq_len=prefill_seq_len,
            max_batch_size=batch_size,
            topk_blk=8,
        )
        build_time = time.time() - start_time
        print(f"\n✅ 模型构建成功 (耗时: {build_time:.2f}s)")
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ 错误：构建模型时 GPU 内存不足")
        print(f"详细信息: {e}")
        print("\n💡 建议:")
        print("  1) 关闭所有其他占用 GPU 的程序")
        print("  2) 减少 prefill_seq_len 到 128")
        print("  3) 确保系统有足够的 DRAM (建议 32GB+)")
        torch.cuda.empty_cache()
        return False
    except FileNotFoundError as e:
        print(f"\n❌ 错误：权重文件未找到: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 错误：模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 验证模型
    if llama is None or llama.model is None:
        print(f"\n❌ 错误：模型对象未正确初始化")
        return False

    # ---- 准备测试提示词 ----
    prompts = [
        "Once upon a time, in a land far far away, there lived a wise old wizard who possessed great knowledge.",
    ][:batch_size]

    print(f"\n📝 测试提示词 ({len(prompts)} 条):")
    for i, p in enumerate(prompts, 1):
        preview = p[:80] + "..." if len(p) > 80 else p
        print(f"  {i}. '{preview}'")

    # ---- 运行 Prefill 测试 ----
    print("\n" + "=" * 80)
    print("🚀 开始 Prefill 测试")
    print("=" * 80)

    try:
        # Tokenize
        print("\n[Prefill] Tokenizing prompts...")
        try:
            prompts_tok = [llama.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        except Exception as e:
            print(f"\n❌ 错误：Tokenization 失败: {e}")
            return False

        prompt_lens = [len(tok) for tok in prompts_tok]
        print(f"[Prefill] Token lengths: {prompt_lens}")
        max_prompt_len = max(prompt_lens)
        print(f"[Prefill] Max prompt length: {max_prompt_len} tokens")

        # 检查并截断
        if max_prompt_len > prefill_seq_len:
            print(f"\n⚠️  警告：提示词长度 ({max_prompt_len}) 超过配置 ({prefill_seq_len})")
            print("将截断到配置长度")
            max_prompt_len = prefill_seq_len

        # Prepare input tensors
        print(f"\n[Prefill] Preparing input tensors...")
        try:
            bsz = len(prompts_tok)
            pad_id = llama.tokenizer.pad_token_id if llama.tokenizer.pad_token_id else llama.tokenizer.eos_token_id

            tokens = torch.full(
                (bsz, max_prompt_len),
                pad_id,
                dtype=torch.long,
                device=device,
            )

            for i, tok in enumerate(prompts_tok):
                tok_len = min(len(tok), max_prompt_len)
                tokens[i, :tok_len] = torch.tensor(tok[:tok_len], device=device)

            print(f"[Prefill] Input tensor shape: {tokens.shape}")
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n❌ 错误：准备输入张量时 GPU 内存不足: {e}")
            torch.cuda.empty_cache()
            return False
        except Exception as e:
            print(f"\n❌ 错误：准备输入张量失败: {e}")
            return False

        # Run prefill forward pass
        print(f"\n[Prefill] Running forward pass...")
        print("=" * 80)
        print("⚠️  注意：16GB 显存下，70B 模型每层传输需要时间，请耐心等待...")
        print("=" * 80)

        start_time = time.time()
        logits = None

        try:
            with torch.no_grad():
                # 只做一次前向传播，start_pos=0 表示这是 prefill
                logits = llama.model(tokens, start_pos=0)

            prefill_time = time.time() - start_time

            # 验证输出
            if logits is None:
                print(f"\n❌ 错误：模型输出为 None")
                return False

            print("=" * 80)
            print(f"\n✅ Prefill 完成！")
            print(f"\n📊 性能统计:")
            print(f"  - Prefill time: {prefill_time:.3f}s")
            print(f"  - Tokens processed: {max_prompt_len * bsz}")
            print(f"  - Throughput: {(max_prompt_len * bsz) / prefill_time:.2f} tokens/s")
            print(f"  - Avg time per token: {(prefill_time / (max_prompt_len * bsz)) * 1000:.2f} ms")
            print(f"  - Logits shape: {logits.shape}")

        except torch.cuda.OutOfMemoryError as e:
            print(f"\n❌ 错误：Forward pass 时 GPU 内存不足: {e}")
            torch.cuda.empty_cache()
            return False
        except RuntimeError as e:
            print(f"\n❌ 错误：Forward pass 运行时错误: {e}")
            import traceback
            traceback.print_exc()
            return False
        except Exception as e:
            print(f"\n❌ 错误：Forward pass 失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        # GPU 内存统计
        if torch.cuda.is_available():
            try:
                mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
                peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
                print(f"\n💾 GPU 内存使用:")
                print(f"  - Current allocated: {mem_allocated:.2f} GB")
                print(f"  - Current reserved: {mem_reserved:.2f} GB")
                print(f"  - Peak allocated: {peak_mem:.2f} GB")
                print(f"  - Utilization: {(mem_allocated / gpu_mem_gb) * 100:.1f}%")
            except Exception as e:
                print(f"\n⚠️  警告：无法获取 GPU 内存统计: {e}")

        # Weight Streaming Manager 统计
        if hasattr(llama, 'weight_streaming_manager'):
            try:
                wsm = llama.weight_streaming_manager
                print(f"\n📦 Weight Streaming 统计:")
                print(f"  - GPU cache size: {len(wsm.gpu_cache)}/{wsm.max_cached_layers} layers")
                print(f"  - Total layers: {len(wsm.blocks)}")
                if wsm.ssd_enabled and hasattr(wsm, 'cpu_cache'):
                    print(f"  - CPU cache enabled: {len(wsm.cpu_cache)} layers cached")
            except Exception as e:
                print(f"\n⚠️  警告：无法获取 WSM 统计: {e}")

        return True

    except KeyboardInterrupt:
        print(f"\n\n⚠️  测试被用户中断")
        return False
    except Exception as e:
        print(f"\n❌ 错误：Prefill 测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    print("\n🔬 开始 70B 模型 Prefill 测试 (16GB 显存)")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        try:
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU 内存: {gpu_mem:.2f} GB")

            if gpu_mem > 16.5:
                print("✅ 显存充足 (>16GB)")
            elif gpu_mem >= 15.5:
                print("⚠️  显存刚好达到要求 (~16GB)")
            else:
                print(f"❌ 显存不足 ({gpu_mem:.1f}GB < 16GB)")
                print("测试可能会失败")
        except Exception as e:
            print(f"⚠️  警告：无法获取完整 GPU 信息: {e}")

    # 运行基础 prefill 测试
    success = False
    try:
        success = test_70b_prefill()
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生未捕获的异常: {e}")
        import traceback
        traceback.print_exc()

    # 总结
    print(f"\n{'=' * 80}")
    print("测试总结")
    print(f"{'=' * 80}")
    print(f"Prefill 测试: {'✅ 通过' if success else '❌ 失败'}")

    if success:
        print("\n🎉 恭喜！70B 模型 Prefill 测试通过 (16GB 显存)")
        print("\n关键验证点:")
        print("  ✅ 权重流式传输工作正常")
        print("  ✅ 16GB 显存成功运行 70B 模型")
        print("  ✅ GPU 内存管理有效")
        print("\n性能提示:")
        print("  - 16GB 显存下，每层需要从 CPU/SSD 传输")
        print("  - 吞吐量会低于大显存 GPU")
        print("  - 适合验证功能，不适合生产部署")
    else:
        print("\n⚠️  测试未通过，请检查上述错误信息")
        print("\n常见问题:")
        print("  1) 检查 70B 模型路径是否正确")
        print("  2) 确保没有其他程序占用 GPU")
        print("  3) 尝试减小 prefill_seq_len (当前 256)")
        print("  4) 确保系统有足够 DRAM (建议 32GB+)")
        print("  5) 检查 CUDA 和 PyTorch 安装")
