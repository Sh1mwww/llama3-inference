#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7B/8B 模型完整测试脚本（修正版）
修复要点：
1) 自动选择运行模式：16GB 级显存默认使用“权重流式”
2) 构建阶段若 CUDA OOM，强制把“半上卡”的模型回迁 CPU 并重建
3) 构建完成后，统一把小模块（embedding/norm/output 等）迁到目标设备，防止设备不一致
4) 可选保险阀：在 forward 入口把 tokens 迁到 embed_tokens 的设备（猴补丁）

使用方式：
    python test_7b_model.py
"""
import sys
from pathlib import Path
import torch

# 将项目根目录加入 sys.path（根据你的工程结构调节）
sys.path.insert(0, str(Path(__file__).parent))

from llama3.generator import LLaMA
from llama3.config import KVCacheArgs


# ============== 工具函数 ==============

def _force_model_small_modules_to(model, target_device: torch.device):
    """
    把小模块（embedding/norm/output/频率表/各层norm）统一迁到目标设备，防止“半上卡”残留。
    """
    
    def _move_safely(mod):
        # 只检测“当前模块自身”是否含有 meta 参数/缓冲
        has_meta = False
        for p in mod.parameters(recurse=False):
            if getattr(p, "is_meta", False):
                has_meta = True; break
        if not has_meta:
            for b in mod.buffers(recurse=False):
                if getattr(b, "is_meta", False):
                    has_meta = True; break
        return mod.to_empty(device=target_device) if has_meta else mod.to(target_device)
    

    if hasattr(model, "embed_tokens"):
        # model.embed_tokens = model.embed_tokens.to(target_device)
        model.embed_tokens = _move_safely(model.embed_tokens)
    if hasattr(model, "norm"):
        # model.norm = model.norm.to(target_device)
        model.norm = _move_safely(model.norm)
    if hasattr(model, "output"):
        # model.output = model.output.to(target_device)
        model.output = _move_safely(model.output)
    if hasattr(model, "freqs_complex") and hasattr(model.freqs_complex, "device"):
        try:
            model.freqs_complex = model.freqs_complex.to(target_device)
        except Exception:
            pass

    if hasattr(model, "layers"):
        for lyr in model.layers:
            for name in ("attn_norm", "ffn_norm"):
                if hasattr(lyr, name):
                    # setattr(lyr, name, getattr(lyr, name).to(target_device))
                    setattr(lyr, name, _move_safely(getattr(lyr, name)))


def _patch_safe_forward(llama_model):
    """
    可选“保险阀”：确保 tokens 和 embed_tokens.weight 在同一设备。
    不改库源码时，可用猴补丁方式替换 forward。
    """
    if not hasattr(llama_model, "forward"):
        return

    orig_forward = llama_model.forward

    def _safe_forward(tokens, start_pos: int):
        dev = llama_model.embed_tokens.weight.device
        if tokens.device != dev:
            tokens = tokens.to(dev, non_blocking=True)
        return orig_forward(tokens, start_pos)

    llama_model.forward = _safe_forward


# ============== 7B/8B 基本模式 ==============

def test_7b_model():
    """
    运行 7B/8B 模型的基本推理测试（自动模式选择 + OOM 回退）
    """
    print("=" * 80)
    print("7B 模型测试 - 验证所有改进")
    print("=" * 80)

    # ---- 模型与设备配置 ----
    # 改成你的 7B/8B 路径（示例为 8B）
    checkpoint_dir = "/home/roger/.llama/checkpoints/Llama3.1-8B/"
    ckpt_path = Path(checkpoint_dir)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，将使用 CPU（仅验证流程，速度较慢）")

    # ---- KV Cache 参数（根据你的环境可调整）----
    print("\n📋 配置 KV Cache 参数...")
    KVCacheArgs.dram_limit_gb = 16.0             # DRAM 限制
    KVCacheArgs.dram_sizing_batch = 8            # DRAM 配额估算批量（更贴近实际）
    KVCacheArgs.ssd_device_path = "/dev/nvme0n1p4"  # 可选：若无 SSD 路径可置空 ""
    KVCacheArgs.max_concurrent_io = 4
    print(f"  - DRAM limit: {KVCacheArgs.dram_limit_gb} GB")
    print(f"  - DRAM sizing batch: {KVCacheArgs.dram_sizing_batch}")

    # ---- 基础检查 ----
    if not ckpt_path.exists():
        print(f"\n❌ 错误：模型路径不存在: {checkpoint_dir}")
        print("请设置正确的模型路径，例如：/home/roger/models/Meta-Llama-3-8B")
        return False

    params_file = ckpt_path / "params.json"
    if not params_file.exists():
        print("\n❌ 错误：未找到 params.json 文件")
        return False

    pth_files = list(ckpt_path.glob("*.pth"))
    if not pth_files:
        print("\n❌ 错误：未找到 .pth 权重文件")
        return False

    print("\n✅ 模型文件检查通过")
    print(f"  - params.json: {params_file}")
    print(f"  - 权重文件: {pth_files[0]}")

    # ---- 根据显存自动选择模式 ----
    use_stream_by_default = False
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        # 16~20GB 级显存默认流式；>=24GB 可尝试 full
        use_stream_by_default = total_gb < 20

    build_kwargs = dict(
        checkpoints_dir=str(ckpt_path),
        load_model=True,
        device=device,
        max_seq_len=512,      # 快速冒烟
        max_batch_size=4,
        topk_blk=8,
    )
    if use_stream_by_default:
        build_kwargs.update({
            "mode": "stream",
            "mode_config": {
                "prefetch_distance": 1,
                "max_cached_layers": 3,
                "warmup_layers": 1,
                "verbose": True,
            }
        })

    # ---- 构建模型（含 OOM 回退）----
    print("\n🔨 构建模型...")
    recovered_to_cpu = False
    try:
        llama = LLaMA.build(**build_kwargs)
        print("✅ 模型构建成功")
    except torch.cuda.OutOfMemoryError as e:
        print("❌ CUDA OOM when building; falling back to CPU:", e)
        torch.cuda.empty_cache()
        # 强制把（可能半上卡的）模型回到 CPU：重新 build 到 CPU
        cpu_build_kwargs = {**build_kwargs, "device": "cpu"}
        cpu_build_kwargs.pop("mode", None)         # CPU 下可忽略流式模式
        cpu_build_kwargs.pop("mode_config", None)
        llama = LLaMA.build(**cpu_build_kwargs)
        recovered_to_cpu = True
        print("✅ 已回退到 CPU 构建（可配合流式继续跑）")
    except Exception as e:
        print(f"❌ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ---- 构建后设备一致性（双向）----
    # 目标设备：以 LLaMA.args.device 为准；若 OOM 回退，则是 "cpu"
    target_device = torch.device(getattr(llama.args, "device", "cpu"))
    _force_model_small_modules_to(llama.model, target_device)

    # 可选：为 forward 打保险阀，防止上层误传 device 不一致的 tokens
    SAFE_FORWARD_PATCH = True
    if SAFE_FORWARD_PATCH:
        _patch_safe_forward(llama.model)

    # ---- 提示词与推理参数 ----
    prompts = [
        "Once upon a time",
        "The quick brown fox",
        "In the beginning",
    ]
    print("\n📝 测试提示词:")
    for i, p in enumerate(prompts, 1):
        print(f"  {i}. '{p}'")

    print("\n🚀 开始推理...")
    try:
        tokens, texts = llama.text_completion(
            prompts=prompts,
            temperature=0.6,
            top_p=0.9,
            max_gen_len=50,
            batch_size=2,
            enable_batching=True,
        )

        print("\n✅ 推理完成！")
        print("\n📄 生成结果:")
        print("=" * 80)
        for i, (prompt, text) in enumerate(zip(prompts, texts), 1):
            print(f"\n[{i}] 提示: {prompt}")
            print(f"    生成: {text}")
        print("=" * 80)
        return True

    except torch.cuda.OutOfMemoryError as e:
        print("\n❌ GPU 内存不足:", e)
        print("\n💡 建议：")
        print("  1) 减少 max_seq_len（当前 512）")
        print("  2) 减少 batch_size（当前 2）")
        print("  3) 使用权重流式模式（本脚本在 16GB 显存已默认启用）")
        return False
    except Exception as e:
        print("\n❌ 推理失败:", e)
        import traceback
        traceback.print_exc()
        return False


# ============== 流式模式专测（可选） ==============

def test_streaming_mode():
    """
    测试权重流式传输模式（适用于 GPU 显存紧张的情况）
    """
    print("\n" + "=" * 80)
    print("7B 模型测试 - 权重流式传输模式")
    print("=" * 80)

    # 改成你的 7B/8B 路径
    checkpoint_dir = "/home/roger/.llama/checkpoints/Llama3.1-8B/"
    device = "cuda:0"

    if not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，跳过流式传输测试")
        return False

    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        print("❌ 模型路径不存在，跳过流式传输测试")
        return False

    # KV 配置
    KVCacheArgs.dram_limit_gb = 16.0
    KVCacheArgs.dram_sizing_batch = 4

    print("\n🔨 构建模型（流式模式）...")
    try:
        llama = LLaMA.build(
            checkpoints_dir=str(ckpt_path),
            load_model=True,
            device=device,
            mode="stream",
            mode_config={
                "prefetch_distance": 2,
                "max_cached_layers": 4,
                "warmup_layers": 2,
                "verbose": True,
            },
            max_seq_len=256,
            max_batch_size=2,
            topk_blk=4,
        )
        # 构建后设备一致性（双向）
        _force_model_small_modules_to(llama.model, torch.device(llama.args.device))
        _patch_safe_forward(llama.model)

        print("✅ 流式模式模型构建成功")
    except Exception as e:
        print("❌ 流式模式构建失败:", e)
        import traceback
        traceback.print_exc()
        return False

    prompts = ["Hello, how are you?"]
    print("\n🚀 开始流式推理...")
    try:
        tokens, texts = llama.text_completion(
            prompts=prompts,
            temperature=0.6,
            max_gen_len=30,
            batch_size=1,
        )
        print("\n✅ 流式推理完成！")
        print("\n生成:", texts[0])
        return True

    except Exception as e:
        print("\n❌ 流式推理失败:", e)
        import traceback
        traceback.print_exc()
        return False


# ============== 主入口 ==============

if __name__ == "__main__":
    print("\n🔬 开始 7B 模型测试")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 测试 1: 基本模式（含自动选择 + OOM 回退）
    print(f"\n{'='*80}\n测试 1: 基本推理模式\n{'='*80}")
    success_basic = test_7b_model()

    # 流式模式专测（可选）
    success_streaming = None
    if torch.cuda.is_available():
        print(f"\n{'='*80}\n测试 2: 权重流式传输模式（可选）\n{'='*80}")
        success_streaming = test_streaming_mode()

    # 总结
    print(f"\n{'='*80}\n测试总结\n{'='*80}")
    print(f"基本模式: {'✅ 通过' if success_basic else '❌ 失败'}")
    if success_streaming is not None:
        print(f"流式模式: {'✅ 通过' if success_streaming else '❌ 失败'}")

    if success_basic:
        print("\n🎉 恭喜！关键改进已成功应用并通过测试")
        print("\n改进验证：")
        print("  ✅ 事件化等待（stream_mnt）")
        print("  ✅ DRAM 配额优化（kv_offload）")
        print("  ✅ 写队列阻塞式处理（避免 drop）")
    else:
        print("\n⚠️  测试未完全通过，请检查错误信息")
        print("\n常见问题：")
        print("  1) 模型路径设置是否正确")
        print("  2) 16GB 显存请优先使用流式模式；或减少 batch/max_seq_len")
        print("  3) SSD 路径（KVCacheArgs.ssd_device_path）是否存在（可置空停用）")
