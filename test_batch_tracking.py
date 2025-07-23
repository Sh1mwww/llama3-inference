#!/usr/bin/env python3
"""
測試batch tracking功能的簡單腳本
"""

from llama3.global_state_tracker import init_global_tracker, get_global_tracker

def test_batch_tracking():
    """測試batch tracking功能"""
    print("🧪 測試GlobalStateTracker的get_next_batch功能")
    
    # 初始化tracker
    tracker = init_global_tracker(max_batch=8, layers=32, n_blocks=100)
    
    # 設置future batches
    tracker.register_future_batch([0, 1, 2, 3])
    
    print(f"初始狀態: future_batches={tracker.future_batches}")
    
    # 測試不同current_batch的情況
    test_cases = [0, 1, 2, 3]
    
    for current in test_cases:
        tracker.set_current_execution(current, 0)
        next_batch = tracker.get_next_batch(1)
        third_batch = tracker.get_next_batch(3)
        
        print(f"當前batch={current}, 下一個batch={next_batch}, 第三個批次={third_batch}")
    
    print("✅ 測試完成")

if __name__ == "__main__":
    test_batch_tracking()