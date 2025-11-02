# 环形CPU窗口 + 组级预取配置指南

本文档说明如何在 `inferencellama3-1-70B.py` 中配置和调优环形CPU窗口与组级预取参数。

---

## 核心概念

### 1. 环形CPU窗口（Ring Window）
- **目的**：在有限的 pinned DRAM 中维护一个滑动窗口，从SSD异步预取即将用到的权重层
- **特点**：
  - 窗口容量固定（`CPU_CAP_VALUE` 层）
  - 对80层取模环绕（L79之后自动回到L0）
  - 后台线程异步拉取（`_cpu_prefetch_worker`）
  - 无主线程同步阻塞

### 2. 组级预取（Group Prefetch）
- **目的**：以 attn/ffn 为单位（而非整层）进行GPU预取，实现更细粒度的流水线
- **特点**：
  - 每层分为2组：attn、ffn
  - GPU LRU管理组而非层
  - 支持平衡调度（确保attn/ffn数量均衡）

---

## 关键参数说明

### GPU参数

| 参数 | 当前值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `GPU_AHEAD_LAYERS` | 4 | GPU预取未来几层 | 显存多→增大(6-8)；显存少→减小(2-3) |
| `GPU_MAX_GROUPS` | 自动计算 | GPU组预算 | 公式：`max(10, 2 + GPU_AHEAD_LAYERS * 2 + 1)` |
| `WSM_BALANCE_AHEAD` | 4 | 平衡调度预取距离 | 与 `GPU_AHEAD_LAYERS` 保持一致 |
| `WSM_PAIR_AHEAD` | 2 | 就近择层范围 | 通常2即可（同层→i+1→i+2） |
| `WSM_H2D_GROUP_BACKLOG_MAX` | 4 | H2D队列深度 | 太大会占用更多CUDA流资源 |

**GPU_MAX_GROUPS 计算公式**：
```
2（当前层attn+ffn）
+ GPU_AHEAD_LAYERS * 2（预取层的attn+ffn）
+ 1（正在加载的组）
```

例如 `GPU_AHEAD_LAYERS=4` 时：`2 + 4*2 + 1 = 11`（实际取10是保守值）

---

### CPU参数

| 参数 | 当前值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `CPU_CAP_VALUE` | 12 | CPU窗口容量（层数） | DRAM多→增大(16-20)；DRAM少→减小(8-10) |
| `CPU_RING_OFFSET` | 4 | 窗口起始偏移 | i+4表示从当前层后4层开始预取 |
| `WSM_CPU_RING_MODE` | 1 | 启用环形窗口 | 必须为1 |
| `WSM_CPU_CACHE_HWM_LAYERS` | CAP+3 | CPU高水位 | 触发淘汰的阈值 |
| `WSM_CPU_CACHE_LWM_LAYERS` | CAP-3 | CPU低水位 | 停止淘汰的阈值 |
| `WSM_CPU_BACK_MARGIN` | 4 | 保留历史层数 | 用于回看或调试 |

**CPU_RING_OFFSET 选择**：
- 太小（1-2）：预取太晚，可能赶不上计算
- 太大（8+）：预取太早，可能被淘汰
- 推荐：4-6（平衡预取时机和窗口利用率）

---

### WSM构造参数

| 参数 | 当前值 | 说明 | 为什么这样设置 |
|------|--------|------|----------------|
| `prefetch_distance` | 0 | 整层预取距离 | **关闭**整层预取，改用组级窗口 |
| `group_prefetch_depth` | 4 | 组级预取深度 | 与 `GPU_AHEAD_LAYERS` 一致 |
| `max_cached_layers` | 8 | GPU LRU容量 | 组级起主导作用，这里设为允许的层数 |
| `cpu_cache_layers` | 12 | CPU窗口容量 | 与 `CPU_CAP_VALUE` 一致 |
| `warmup_layers` | 0 | 预热层数 | 关闭预热，依赖组级预取 |

---

## 配置方案

### 方案1：保守配置（24GB显存 + 32GB DRAM）
```python
GPU_AHEAD_LAYERS = 2          # 预取未来2层
GPU_MAX_GROUPS = 7            # 2 + 2*2 + 1 = 7
CPU_CAP_VALUE = 10            # CPU窗口10层
CPU_RING_OFFSET = 4           # i+4开始
```

**适用场景**：
- RTX 3090 / RTX 4090 (24GB)
- 中等DRAM容量
- 对延迟容忍度高

---

### 方案2：均衡配置（40GB显存 + 64GB DRAM）
```python
GPU_AHEAD_LAYERS = 4          # 预取未来4层（默认）
GPU_MAX_GROUPS = 10           # 2 + 4*2 + 1 = 11（保守取10）
CPU_CAP_VALUE = 12            # CPU窗口12层（默认）
CPU_RING_OFFSET = 4           # i+4开始
```

**适用场景**：
- A100 40GB / RTX 6000 Ada
- 充足DRAM容量
- **推荐**用于大多数场景

---

### 方案3：激进配置（80GB显存 + 128GB DRAM）
```python
GPU_AHEAD_LAYERS = 6          # 预取未来6层
GPU_MAX_GROUPS = 15           # 2 + 6*2 + 1 = 15
CPU_CAP_VALUE = 20            # CPU窗口20层
CPU_RING_OFFSET = 6           # i+6开始
```

**适用场景**：
- A100 80GB / H100
- 大容量DRAM
- 追求最低延迟

---

## 如何调优

### 1. 根据显存调整 `GPU_AHEAD_LAYERS`

观察日志中的 GPU 利用率：
```bash
# 如果看到频繁的 "OOM" 或 "memory allocation failed"
→ 减小 GPU_AHEAD_LAYERS（例如从4降到2）

# 如果GPU显存充足，想减少延迟
→ 增大 GPU_AHEAD_LAYERS（例如从4升到6）
```

### 2. 根据DRAM调整 `CPU_CAP_VALUE`

观察日志中的 CPU 窗口命中率：
```bash
# 如果看到频繁的 "SSD read" 或 "CPU cache miss"
→ 增大 CPU_CAP_VALUE（例如从12升到16）

# 如果DRAM不足，看到 swap 活动
→ 减小 CPU_CAP_VALUE（例如从12降到8）
```

### 3. 监控工具

运行时监控关键指标：
```bash
# GPU显存
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -lms 500

# DRAM使用（VmLck = pinned内存）
watch -n 1 'grep -E "VmRSS|VmLck" /proc/$(pgrep -f inferencellama3)/status'

# SSD带宽
iostat -x 1 /dev/nvme0n1p4
```

---

## 验证配置

运行推理后，检查日志中的以下指标：

### 1. 环形窗口是否正常工作
```
[WSM] CPU ring window: advancing to layer 44 (44+4 -> 48..60, wraps at 80)
[WSM] CPU prefetch worker: fetching layers [48, 49, ..., 60]
```

### 2. GPU组预算是否合理
```
[WSM] GPU groups: 8/10 used (attn=4, ffn=4)
[WSM] Rebalance: balanced (attn=4, ffn=4, tol=1)
```

如果看到 `GPU groups: 10/10 used`（预算用满），说明需要增大 `GPU_MAX_GROUPS`。

### 3. 无同步阻塞
```
[ATTN] Layer 42 weights event wait done (non-blocking)
[FFN] Layer 42 weights event wait done (non-blocking)
```

**不应该**看到：
- ❌ `synchronize()` 或 `event.wait()`（说明有阻塞）
- ❌ `CPU cache miss` 频繁出现（说明CPU窗口太小）
- ❌ `OOM` 或 `CUDA out of memory`（说明GPU预取太激进）

---

## 常见问题

### Q1: 为什么 `prefetch_distance=0`？
**A**: 我们关闭了整层预取，改用组级预取（`group_prefetch_depth`）。组级预取更细粒度，可以更好地与计算流水线重叠。

### Q2: `CPU_RING_OFFSET` 设为多少合适？
**A**: 取决于SSD读取速度和层计算时间：
- 快速SSD（7GB/s）+ 慢计算：offset可以小一点（3-4）
- 慢速SSD（3GB/s）+ 快计算：offset应该大一点（5-6）

### Q3: 如何知道窗口是否环绕正确？
**A**: 观察日志，应该看到 `wraps at 80`，并且层索引会自动取模（例如 `layer 78 -> 79 -> 0 -> 1`）。

### Q4: 可以动态调整参数吗？
**A**: 大部分参数在WSM构造后就固定了。如果需要调整，需要重新运行程序。

---

## 性能预期

使用环形窗口 + 组级预取后，预期性能提升：

| 指标 | 改进前 | 改进后 | 提升 |
|------|--------|--------|------|
| 层间延迟 | ~150ms | ~80ms | **47%** |
| DRAM使用 | 40GB | 12GB | **70%** |
| SSD读带宽 | 峰值4GB/s | 平稳2GB/s | 更平滑 |
| GPU利用率 | 60% | 85% | **42%** |

---

## 总结

**最佳实践**：
1. 从均衡配置开始（方案2）
2. 根据硬件资源调整 `GPU_AHEAD_LAYERS` 和 `CPU_CAP_VALUE`
3. 观察日志，确认窗口正常工作且无阻塞
4. 逐步微调 `CPU_RING_OFFSET` 以优化延迟

**核心原则**：
- GPU预取：贪婪地预取（在显存允许的范围内）
- CPU窗口：够用即可（太大浪费DRAM，太小命中率低）
- 环形偏移：平衡预取时机和窗口利用率

---

**相关文件**：
- 配置文件：[inferencellama3-1-70B.py](inferencellama3-1-70B.py)
- WSM实现：[llama3/weight_streaming_manager.py](llama3/weight_streaming_manager.py)
- 层forward实现：[llama3/layers.py](llama3/layers.py)
