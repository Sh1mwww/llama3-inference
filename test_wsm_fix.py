#!/usr/bin/env python3
"""
测试 Weight Streaming Manager 设备问题修复
"""

import sys
import os
sys.path.append('/home/roger/llama3_project')

def test_basic_inference():
    """测试基本推理功能"""
    print("🧪 Testing WSM device fix with basic inference...")
    
    # 使用修复后的 profile_pipeline 进行测试
    import subprocess
    
    # 简单的测试命令
    cmd = [
        'python3', '/home/roger/llama3_project/scripts/profile_pipeline.py',
        '--model-path', '/mnt/model/llama/checkpoints/Llama3.2-3B',
        '--prompt', 'Hello world',
        '--max-gen-len', '10',
        '--batch-size', '1',
        '--device', 'cuda',
        '--verbose'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        print("\n📤 STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\n📥 STDERR:")  
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ Test completed successfully!")
            return True
        else:
            print(f"\n❌ Test failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n⏰ Test timed out after 120 seconds")
        return False
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_inference()
    sys.exit(0 if success else 1)