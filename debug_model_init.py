#!/usr/bin/env python3
"""
调试模型初始化问题
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import psutil
import gc
import torch

def monitor_memory():
    """监控内存使用"""
    process = psutil.Process()
    memory_info = process.memory_info()
    virtual_memory = psutil.virtual_memory()

    print(f"进程内存: {memory_info.rss / (1024**3):.2f} GB")
    print(f"系统可用内存: {virtual_memory.available / (1024**3):.2f} GB")

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        print(f"GPU内存: {gpu_memory:.2f} GB")

def test_model_creation_steps():
    """逐步测试模型创建过程"""

    checkpoints_dir = "/data1/.llama/checkpoints/Llama3.1-70B"

    print("🔍 Debug Model Initialization Steps")
    print("=" * 50)

    print("1. Initial memory state:")
    monitor_memory()

    try:
        print("\n2. Loading tokenizer...")
        from transformers import LlamaTokenizerFast
        tokenizer = LlamaTokenizerFast.from_pretrained(checkpoints_dir, legacy=True)
        print("✅ Tokenizer loaded")
        monitor_memory()

        print("\n3. Loading model config...")
        from llama3.config import ModelArgs
        params_path = Path(checkpoints_dir) / "params.json"

        # 创建一个小的测试配置来验证流程
        args = ModelArgs.from_json(
            str(params_path),
            max_seq_len=512,  # 减小序列长度
            max_batch_size=1, # 减小batch size
            device="cpu"      # 先在CPU上创建
        )
        print(f"✅ Config loaded: {args.n_layers} layers, {args.dim} dim")
        monitor_memory()

        print("\n4. Creating model on CPU...")
        from llama3.model import Transformer

        # 强制垃圾回收
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print("Memory before model creation:")
        monitor_memory()

        # 尝试创建模型
        model = Transformer(args)

        print("✅ Model created on CPU")
        monitor_memory()

        print("\n5. Model structure:")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,} ({total_params * 4 / (1024**3):.2f} GB in fp32)")

        # 检查是否可以移动到GPU（小批量测试）
        if torch.cuda.is_available():
            print("\n6. Testing GPU memory...")
            try:
                # 只移动一小部分来测试
                model.embed_tokens = model.embed_tokens.cuda()
                print("✅ Successfully moved embed_tokens to GPU")
                monitor_memory()
            except Exception as e:
                print(f"❌ GPU test failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Error at step: {e}")
        import traceback
        traceback.print_exc()
        monitor_memory()
        return False

def test_memory_limits():
    """测试不同的内存限制"""

    print(f"\n🧪 Testing Memory Limits")
    print("=" * 30)

    # 获取系统信息
    virtual_memory = psutil.virtual_memory()
    print(f"System memory: {virtual_memory.total / (1024**3):.1f} GB total")
    print(f"Available memory: {virtual_memory.available / (1024**3):.1f} GB")
    print(f"Memory usage: {virtual_memory.percent}%")

    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    # 计算LLaMA3-70B的理论内存需求
    # 70B参数 * 2字节(fp16) = 140GB
    print(f"\nLLaMA3-70B theoretical memory:")
    print(f"70B params × 2 bytes (fp16) = 140 GB")
    print(f"70B params × 4 bytes (fp32) = 280 GB")

    available_gb = virtual_memory.available / (1024**3)
    if available_gb < 150:
        print(f"❌ Insufficient memory for full model loading")
        print(f"💡 SSD streaming is essential for this model size")
        return False
    else:
        print(f"✅ Sufficient memory available")
        return True

if __name__ == "__main__":
    print("🔧 Model Initialization Debug")
    print("=" * 40)

    # 测试内存限制
    memory_ok = test_memory_limits()

    if not memory_ok:
        print(f"\n💡 Recommendation:")
        print(f"   - Use SSD streaming with smaller cpu_cache_layers")
        print(f"   - Consider using a smaller model variant")
        print(f"   - Add more system RAM")
    else:
        # 测试模型创建步骤
        success = test_model_creation_steps()

        if success:
            print(f"\n✅ Model initialization debug complete")
        else:
            print(f"\n❌ Model initialization failed")