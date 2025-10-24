# CPU Cache 滚动窗口式预取 - 使用指南

## 快速开始

### 1. 环境变量配置

在运行推理之前，设置以下环境变量：

```bash
# CPU cache 硬上限（推荐值：50）
export WSM_CPU_CACHE_CAP_LAYERS=50

# CPU cache 高水位（推荐值：55，略大于硬上限）
export WSM_CPU_CACHE_HWM_LAYERS=55

# CPU 预取窗口距离（推荐值：50，与硬上限相同）
export WSM_CPU_PREFETCH_DISTANCE=50
```

### 2. 运行推理

```bash
# 启用 verbose 模式查看详细日志
python test_70b_prefill_ssd.py --verbose
```

### 3. 预期日志输出

启用 verbose 后，你会看到类似的日志：

```
[WSM] CPU cache shrink: current=5, count=56, window=[6, 56]
[WSM] CPU cache evict (out of window): layer 0
[WSM] CPU cache evict (out of window): layer 1
[WSM] ✅ Loaded layer 56 to CPU cache (7 params)
```

## 参数调优指南

### 基本原则

1. **`cpu_cache_cap_layers` ≤ `cpu_prefetch_distance`**
   - 硬上限不应超过预取窗口大小
   - 否则窗口内的层可能被提前淘汰

2. **`cpu_cache_hwm_layers` = `cpu_cache_cap_layers` + 5**
   - 高水位应略大于硬上限，留出缓冲空间
   - 差值建议在 3~10 之间

3. **根据 DRAM 容量调整**
   - 70B 模型单层约 3~4 GB（BF16）
   - 50 层需要约 150~200 GB DRAM

### 不同场景的推荐配置

#### 场景 1：大内存服务器（256GB+ DRAM）

```bash
# 保守配置，充分利用 DRAM
export WSM_CPU_CACHE_CAP_LAYERS=60
export WSM_CPU_CACHE_HWM_LAYERS=65
export WSM_CPU_PREFETCH_DISTANCE=60
```

**优点：**
- 更大的预取窗口，减少 SSD 读取频率
- 更高的吞吐量

**缺点：**
- 占用更多 DRAM

#### 场景 2：中等内存服务器（128GB DRAM）

```bash
# 平衡配置
export WSM_CPU_CACHE_CAP_LAYERS=40
export WSM_CPU_CACHE_HWM_LAYERS=45
export WSM_CPU_PREFETCH_DISTANCE=40
```

**优点：**
- 平衡 DRAM 使用和预取效率
- 适合大多数场景

#### 场景 3：小内存服务器（64GB DRAM）

```bash
# 激进配置，最小化 DRAM 使用
export WSM_CPU_CACHE_CAP_LAYERS=20
export WSM_CPU_CACHE_HWM_LAYERS=25
export WSM_CPU_PREFETCH_DISTANCE=20
```

**优点：**
- 最小化 DRAM 占用
- 避免 OOM

**缺点：**
- 更频繁的 SSD 读取
- 可能影响吞吐量

## 监控和调试

### 查看 CPU cache 状态

在推理过程中，启用 verbose 模式可以看到：

```python
wsm = WeightStreamingManager(
    model,
    verbose=True,  # 启用详细日志
    ssd_manifest_path="manifest.json",
    cpu_cache_layers=50
)
```

### 关键日志解读

1. **`📥 Loaded layer X to CPU cache`**
   - 表示从 SSD 加载层到 DRAM

2. **`🗑️ Evict (out of window): layer X`**
   - 表示淘汰窗口外的层（正常行为）

3. **`🗑️ Evict (in window, LRU): layer X`**
   - 表示淘汰窗口内的层（可能需要调大容量）

4. **`⚠️ CPU cache shrink: current=X, count=Y, window=[A, B]`**
   - 当前层：X
   - CPU cache 层数：Y
   - 预取窗口：[A, B]

### 常见问题诊断

#### 问题 1：频繁出现 "Evict (in window, LRU)"

**原因：** CPU cache 容量太小，窗口内的层无法全部缓存

**解决方案：**
```bash
# 增大 CPU cache 容量
export WSM_CPU_CACHE_CAP_LAYERS=60  # 从 50 增加到 60
export WSM_CPU_CACHE_HWM_LAYERS=65
export WSM_CPU_PREFETCH_DISTANCE=50  # 保持不变或适当减小
```

#### 问题 2：CPU cache 层数始终接近上限

**原因：** 预取窗口过大，持续触发高水位

**解决方案：**
```bash
# 减小预取窗口
export WSM_CPU_PREFETCH_DISTANCE=40  # 从 50 减小到 40
export WSM_CPU_CACHE_CAP_LAYERS=40
export WSM_CPU_CACHE_HWM_LAYERS=45
```

#### 问题 3：出现 "timeout, loading immediately"

**原因：** CPU prefetch 未及时完成，层尚未加载到 DRAM

**解决方案：**
```bash
# 增大预取窗口，提前加载
export WSM_CPU_PREFETCH_DISTANCE=60  # 从 50 增加到 60
export WSM_CPU_CACHE_CAP_LAYERS=60
```

## 性能优化技巧

### 1. 预热（Warmup）

在推理前预热 CPU cache：

```python
# 预热前 50 层
wsm.wait_for_preload_ready(timeout=300.0)
```

### 2. 调整预取距离

根据推理速度动态调整：

```bash
# 快速推理（低延迟优先）：小窗口
export WSM_CPU_PREFETCH_DISTANCE=30

# 慢速推理（大批量）：大窗口
export WSM_CPU_PREFETCH_DISTANCE=70
```

### 3. 与 KV offload 配合

确保 CPU cache 和 KV offload 不争抢 DRAM：

```bash
# 假设 256GB DRAM
# KV cache: 100GB
# 权重 cache: 150GB (50层 × 3GB)
# 系统预留: 6GB

export WSM_CPU_CACHE_CAP_LAYERS=50
```

## 测试和验证

### 运行模拟测试

```bash
# 运行滚动窗口演示脚本
python test_rolling_window.py
```

**预期输出：**
- CPU cache 保持在 50~55 层
- 窗口随层推进而滚动
- 优先淘汰窗口外的层

### 端到端测试

```bash
# 运行实际推理测试
export WSM_CPU_CACHE_CAP_LAYERS=50
export WSM_CPU_CACHE_HWM_LAYERS=55
export WSM_CPU_PREFETCH_DISTANCE=50

python test_70b_prefill_ssd.py
```

**关注指标：**
- CPU cache 层数是否稳定在 50~55
- 是否出现频繁的立即加载（timeout）
- 推理吞吐量是否满足预期

## 与其他机制的配合

### 组级预取（Group Prefetch）

滚动窗口与组级预取完全兼容：

```python
# 组级模式下，CPU cache 仍然按层管理
self.grouped_mode = True
self.gpu_max_groups = 4  # GPU 最多缓存 4 个组
self.cpu_cache_cap_layers = 50  # CPU 最多缓存 50 层
```

**工作流程：**
1. CPU cache 按层滚动预取（SSD → DRAM）
2. GPU cache 按组调度（DRAM → GPU）
3. 两级 cache 独立管理，互不干扰

### GPU LRU

CPU 和 GPU 各自维护独立的 LRU：

```python
# CPU LRU（按层）
self._cpu_lru = [0, 1, 2, ..., 50]

# GPU LRU（按组）
self._gpu_group_lru = [(5, 'attn'), (5, 'ffn'), (6, 'attn'), ...]
```

## 常见陷阱

### ❌ 错误配置 1：窗口大于容量

```bash
# 错误：窗口 60 > 容量 40
export WSM_CPU_CACHE_CAP_LAYERS=40
export WSM_CPU_PREFETCH_DISTANCE=60  # 太大！
```

**后果：** 窗口内的层会被提前淘汰，导致频繁重新加载

### ❌ 错误配置 2：高水位等于硬上限

```bash
# 错误：高水位 = 硬上限，没有缓冲空间
export WSM_CPU_CACHE_CAP_LAYERS=50
export WSM_CPU_CACHE_HWM_LAYERS=50  # 应该是 55！
```

**后果：** 每次加载新层都会立即触发淘汰，开销大

### ❌ 错误配置 3：容量远小于层数

```bash
# 错误：80 层模型，只缓存 10 层
export WSM_CPU_CACHE_CAP_LAYERS=10
export WSM_CPU_PREFETCH_DISTANCE=10
```

**后果：** SSD 读取频繁，吞吐量低

## 总结

### ✅ 推荐默认配置

```bash
# 70B 模型，256GB DRAM
export WSM_CPU_CACHE_CAP_LAYERS=50
export WSM_CPU_CACHE_HWM_LAYERS=55
export WSM_CPU_PREFETCH_DISTANCE=50
```

### ✅ 核心优势

1. **严格容量控制**：CPU cache 不会无限增长
2. **滚动窗口预取**：始终预取即将使用的层
3. **窗口感知淘汰**：优先淘汰窗口外的旧层
4. **LRU 优化**：保留最近使用的层

### ✅ 适用场景

- ✅ SSD 后端权重流式
- ✅ 大模型推理（70B+）
- ✅ 有限 DRAM 场景
- ✅ 长序列生成

### ❌ 不适用场景

- ❌ 所有权重都能放入 DRAM（无需流式）
- ❌ 随机访问层（非顺序推理）
- ❌ 极短序列（预取开销大于收益）
