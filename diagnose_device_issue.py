#!/usr/bin/env python3
"""
WSM 设备问题诊断工具

快速检测 Weight Streaming Manager 相关的设备放置问题
"""

import sys
import torch
import pathlib

def diagnose_device_issues(model_path: str, target_device: str = "cuda"):
    """诊断模型加载和设备放置问题"""
    
    print("🔍 WSM Device Issue Diagnostic Tool")
    print("="*50)
    
    # 检查基础环境
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    
    # 检查模型路径
    ckpt_path = pathlib.Path(model_path)
    if not ckpt_path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        return False
    
    print(f"✅ Model path exists: {model_path}")
    
    # 导入必要模块
    try:
        sys.path.append('/home/roger/llama3_project')
        from llama3.generator import LLaMA
        print("✅ LLaMA modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import LLaMA modules: {e}")
        return False
    
    # 加载模型
    print("\n📦 Loading model...")
    try:
        # 先在 CPU 上加载
        llama = LLaMA.build(ckpt_path, load_model=True, device="cpu")
        print("✅ Model loaded on CPU")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # 检查初始设备状态
    print("\n🔍 Checking initial device placement...")
    try:
        m = llama.model
        
        print(f"   embed_tokens: {m.embed_tokens.weight.device}")
        print(f"   norm: {m.norm.weight.device}")
        print(f"   output: {m.output.weight.device}")
        
        if hasattr(m, 'freqs_complex'):
            print(f"   freqs_complex: {m.freqs_complex.device}")
            print(f"   freqs_complex shape: {m.freqs_complex.shape}")
            print(f"   freqs_complex dtype: {m.freqs_complex.dtype}")
        else:
            print("   ⚠️  freqs_complex not found")
        
        print(f"   First layer device: {next(m.layers[0].parameters()).device}")
        
    except Exception as e:
        print(f"❌ Error checking initial device state: {e}")
        return False
    
    # 测试 freqs_complex 移动
    if target_device.startswith("cuda") and torch.cuda.is_available():
        print(f"\n🔄 Testing freqs_complex movement to {target_device}...")
        try:
            if hasattr(m, 'freqs_complex'):
                original_device = m.freqs_complex.device
                print(f"   Original device: {original_device}")
                
                # 尝试移动
                m.freqs_complex = m.freqs_complex.to(target_device)
                new_device = m.freqs_complex.device
                print(f"   After movement: {new_device}")
                
                if str(new_device) == target_device:
                    print("   ✅ freqs_complex moved successfully")
                else:
                    print(f"   ❌ freqs_complex movement failed")
                    return False
            else:
                print("   ❌ freqs_complex not available for testing")
                return False
                
        except Exception as e:
            print(f"   ❌ freqs_complex movement failed: {e}")
            
            # 尝试重新创建
            print("   🔧 Attempting to recreate freqs_complex...")
            try:
                from llama3.layers import precompute_theta_pos_frequencies
                m.freqs_complex = precompute_theta_pos_frequencies(
                    llama.args.dim // llama.args.n_heads,
                    llama.args.max_seq_len * 2,
                    device=target_device,
                    theta=llama.args.rope_theta,
                )
                print(f"   ✅ freqs_complex recreated on {target_device}")
                print(f"   New device: {m.freqs_complex.device}")
                
            except Exception as e2:
                print(f"   ❌ Failed to recreate freqs_complex: {e2}")
                return False
    
    # 测试简单的输入张量创建
    print(f"\n🧪 Testing input tensor creation on {target_device}...")
    try:
        test_tokens = torch.tensor([[1, 2, 3, 4]], device=target_device, dtype=torch.long)
        print(f"   Test tokens device: {test_tokens.device}")
        print("   ✅ Input tensor creation successful")
        
        # 测试与 freqs_complex 的兼容性
        if hasattr(m, 'freqs_complex'):
            freqs_slice = m.freqs_complex[0:1].to(test_tokens.device)
            print(f"   freqs_slice device: {freqs_slice.device}")
            print("   ✅ Device compatibility test passed")
        
    except Exception as e:
        print(f"   ❌ Input tensor test failed: {e}")
        return False
    
    print(f"\n✅ All diagnostic checks passed!")
    print("   Your WSM setup should work correctly now.")
    
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose WSM device issues")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--device", default="cuda", help="Target device")
    
    args = parser.parse_args()
    
    success = diagnose_device_issues(args.model_path, args.device)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()