#!/usr/bin/env python3
"""
测试 CPU Cache 滚动窗口式预取机制

这个脚本模拟了滚动窗口的工作流程，展示：
1. 窗口如何随当前层推进而滚动
2. CPU cache 如何保持在指定容量范围内
3. LRU 淘汰策略如何工作
"""

class MockWSM:
    """简化的 WSM 模拟器，用于演示滚动窗口逻辑"""

    def __init__(self, n_layers=80, cpu_prefetch_distance=50,
                 cpu_cache_cap=50, cpu_cache_hwm=55, verbose=True):
        self.n_layers = n_layers
        self.cpu_prefetch_distance = cpu_prefetch_distance
        self.cpu_cache_cap_layers = cpu_cache_cap
        self.cpu_cache_hwm_layers = cpu_cache_hwm
        self.verbose = verbose

        self._cpu_cached_layers = set()
        self._cpu_lru = []  # 最近使用在末尾

    def _touch_cpu_layer(self, layer_idx):
        """标记层最近被使用"""
        if layer_idx in self._cpu_lru:
            self._cpu_lru.remove(layer_idx)
            self._cpu_lru.append(layer_idx)
        if self.verbose:
            print(f"  👆 Touch layer {layer_idx}")

    def _load_layer_to_cpu(self, layer_idx):
        """模拟加载层到 CPU cache"""
        if layer_idx in self._cpu_cached_layers:
            return

        self._cpu_cached_layers.add(layer_idx)
        if layer_idx in self._cpu_lru:
            self._cpu_lru.remove(layer_idx)
        self._cpu_lru.append(layer_idx)

        if self.verbose:
            print(f"  📥 Loaded layer {layer_idx} to CPU cache")

    def _shrink_cpu_cache_if_needed(self, current_layer_hint):
        """收缩 CPU cache"""
        count = len(self._cpu_cached_layers)
        if count <= self.cpu_cache_hwm_layers:
            return

        if current_layer_hint is None:
            cur = max(self._cpu_lru[-1], 0) if self._cpu_lru else 0
        else:
            cur = current_layer_hint

        lo = cur + 1
        hi = min(self.n_layers - 1, cur + self.cpu_prefetch_distance)

        if self.verbose:
            print(f"  ⚠️  CPU cache shrink: current={cur}, count={count}, window=[{lo}, {hi}]")

        # Phase 1: 淘汰窗口外的层
        i = 0
        while count > self.cpu_cache_cap_layers and i < len(self._cpu_lru):
            L = self._cpu_lru[i]
            if not (lo <= L <= hi):
                self._cpu_lru.pop(i)
                self._cpu_cached_layers.discard(L)
                count = len(self._cpu_cached_layers)
                if self.verbose:
                    print(f"  🗑️  Evict (out of window): layer {L}")
            else:
                i += 1

        # Phase 2: 淘汰窗口内的最老层
        i = 0
        while count > self.cpu_cache_cap_layers and i < len(self._cpu_lru):
            L = self._cpu_lru[i]
            if lo <= L <= hi:
                self._cpu_lru.pop(i)
                self._cpu_cached_layers.discard(L)
                count = len(self._cpu_cached_layers)
                if self.verbose:
                    print(f"  🗑️  Evict (in window, LRU): layer {L}")
            else:
                i += 1

    def _schedule_cpu_prefetch(self, current_layer):
        """滚动窗口式预取"""
        lo = current_layer + 1
        hi = min(self.n_layers - 1, current_layer + self.cpu_prefetch_distance)

        if self.verbose:
            print(f"  🔍 Prefetch window: [{lo}, {hi}]")

        for L in range(lo, hi + 1):
            if L not in self._cpu_cached_layers:
                self._load_layer_to_cpu(L)

        self._shrink_cpu_cache_if_needed(current_layer_hint=current_layer)

    def simulate_layer_forward(self, layer_idx):
        """模拟层 forward"""
        if self.verbose:
            print(f"\n🚀 Layer {layer_idx} forward:")

        self._touch_cpu_layer(layer_idx)
        self._schedule_cpu_prefetch(layer_idx)

        if self.verbose:
            print(f"  ✅ CPU cache: {len(self._cpu_cached_layers)} layers")
            print(f"  📊 Cached layers: {sorted(self._cpu_cached_layers)[:10]}{'...' if len(self._cpu_cached_layers) > 10 else ''}")
            print(f"  🔄 LRU queue (last 10): {self._cpu_lru[-10:]}")


def main():
    print("=" * 80)
    print("CPU Cache 滚动窗口式预取 - 演示")
    print("=" * 80)

    # 创建模拟器
    wsm = MockWSM(
        n_layers=80,
        cpu_prefetch_distance=50,
        cpu_cache_cap=50,
        cpu_cache_hwm=55,
        verbose=True
    )

    print("\n【场景 1】初始预取（层 0）")
    print("-" * 80)
    wsm.simulate_layer_forward(0)

    print("\n【场景 2】推进到层 5（窗口开始滚动）")
    print("-" * 80)
    wsm.simulate_layer_forward(5)

    print("\n【场景 3】推进到层 10（窗口继续滚动，开始淘汰旧层）")
    print("-" * 80)
    wsm.simulate_layer_forward(10)

    print("\n【场景 4】推进到层 30（窗口已完全滚动，大量淘汰）")
    print("-" * 80)
    wsm.simulate_layer_forward(30)

    print("\n【场景 5】推进到层 60（接近末尾）")
    print("-" * 80)
    wsm.simulate_layer_forward(60)

    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)
    print(f"\n关键观察点：")
    print(f"1. CPU cache 始终保持在 50~55 层范围内")
    print(f"2. 预取窗口随当前层滚动：[cur+1, cur+50]")
    print(f"3. 优先淘汰窗口外的旧层（如层 30 时，层 0~10 被淘汰）")
    print(f"4. LRU 队列保持最近使用的层在末尾")


if __name__ == "__main__":
    main()
