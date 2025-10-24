#!/usr/bin/env python3
"""
诊断滑动窗口问题的脚本
模拟窗口推进逻辑，找出问题所在
"""

# 模拟参数
n_layers = 80
cpu_cache_cap = 50
cpu_back_margin = 4

# 状态
cpu_win_base = 0
cache = set()  # 缓存的层

def _target_cpu_window():
    L0 = cpu_win_base
    L1 = min(n_layers - 1, cpu_win_base + cpu_cache_cap - 1)
    return L0, L1

def _advance_cpu_window_NEW(cur_layer):
    """新逻辑：平滑滑动"""
    global cpu_win_base

    L0, L1 = _target_cpu_window()

    # 只在当前层超出窗口或接近末尾时推进
    if cur_layer > L1 or (cur_layer >= L1 - 5):
        if cur_layer > L1:
            # 已超出，推进到刚好包含当前层
            new_base = cur_layer - cpu_cache_cap + 1
        elif cur_layer >= L1 - 5:
            # 接近末尾，推进 1 层
            new_base = cpu_win_base + 1
        else:
            new_base = cpu_win_base

        new_base = max(0, new_base)

        old_base = cpu_win_base
        if new_base > cpu_win_base:
            cpu_win_base = new_base
            print(f"  [ADVANCE] layer {cur_layer}: win_base {old_base} -> {cpu_win_base}")
        else:
            print(f"  [KEEP] layer {cur_layer}: win_base={cpu_win_base}")
    else:
        print(f"  [KEEP] layer {cur_layer}: win_base={cpu_win_base} (in window)")

def simulate_forward():
    global cache

    print("=== 初始预热：加载 0-49 ===")
    cache = set(range(50))
    print(f"Cache: {len(cache)} layers = {sorted(cache)[:5]}...{sorted(cache)[-5:]}")
    print(f"Window: {_target_cpu_window()}")
    print()

    print("=== 模拟 forward 过程 ===")
    for layer in range(50):  # 模拟前 50 层的 forward
        if layer % 10 == 0 or layer >= 44:  # 只打印关键层
            print(f"\n--- Layer {layer} forward ---")
            _advance_cpu_window_NEW(layer)

            L0, L1 = _target_cpu_window()
            print(f"  Window: [{L0}, {L1}]")

            # 模拟清理窗口外的层
            layers_to_evict = [l for l in cache if l < L0 or l > L1]
            if layers_to_evict:
                print(f"  [EVICT] {len(layers_to_evict)} layers: {layers_to_evict}")
                cache -= set(layers_to_evict)

            # 模拟加载缺失层
            missing = [l for l in range(L0, L1+1) if l not in cache]
            if missing:
                print(f"  [LOAD] {len(missing)} layers: {missing}")
                cache |= set(missing)

            print(f"  Cache size: {len(cache)}")

if __name__ == "__main__":
    simulate_forward()
