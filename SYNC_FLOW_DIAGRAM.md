# 权重同步流程图

## 总体流程

```
Layer Forward 调用
    ↓
[1] _mark_group_in_use(layer_id, group)
    │   - 增加引用计数 _gpu_group_in_use[key]++
    │   - 调用 _touch_group() 更新 LRU 时间戳
    │   - 防止该组被驱逐
    ↓
[2] ensure_group_on_gpu(layer_id, group)
    │   - 检查组是否已在 GPU: _group_is_resident()
    │   - 若不在，发起 H2D 传输（在 weight_h2d 流上）
    │   - 记录 CUDA Event 到 _group_ready_events[key]
    ↓
[3] wait_group_ready(layer_id, group, compute_stream)
    │   - 快路径：若已在 GPU，直接返回
    │   - 检查 _gpu_group_inflight：若在加载中，等待完成
    │   - 将 compute_stream 挂到 _group_ready_events[key] 上
    │   - compute_stream.wait_event(cuda_event)
    │   ↓
    │   这是关键同步点！确保：
    │   • 计算流在继续前必须等待 H2D 完成
    │   • 不会读取未完成的权重
    ↓
[4] 执行计算（权重已就绪且可访问）
    │   - Q = wq(x)
    │   - K = wk(x)
    │   - V = wv(x)
    │   - attn = softmax(Q @ K.T)
    │   - out = attn @ V
    │   - result = wo(out)
    ↓
[5] _unmark_group_in_use(layer_id, group) [in finally block]
    │   - 减少引用计数 _gpu_group_in_use[key]--
    │   - 若计数降为 0，从 in_use 集合移除
    │   - 调用 _touch_group() 再次更新时间戳（避免立即被踢）
    │   - 可选：立即驱逐（若 evict_finished_group=True）
    │   - 或者收缩其它组到预算上限
    ↓
返回结果
```

## 时间线视图（多流并发）

```
时间轴 →

weight_h2d 流:
    ├─ [Layer N-1 FFN H2D]─┐
    │                       ├─ Event_N-1_ffn
    │                       │
    ├─────────────────── [Layer N ATTN H2D]─┐
                                             ├─ Event_N_attn

compute_mha 流:
    ├─ [Layer N-1 ATTN compute]─────────────┐
    │                                         │
    │   _mark_group_in_use(N, "attn")       │
    │   ensure_group_on_gpu(N, "attn")       │
    │   wait_group_ready(N, "attn")          │ ← 等待 Event_N_attn
    │          ↓ (wait_event 阻塞点)          │
    │          ├─ [等待 H2D 完成]              │
    │          ↓                               │
    ├──────── [Layer N ATTN compute] ─────────┤
    │          Q/K/V/attn/wo                   │
    │   _unmark_group_in_use(N, "attn")       │
    │                                         ↓

compute_ffn 流:
    ├─ [Layer N-1 FFN compute]───────────────┐
    │                                         │
    │   _mark_group_in_use(N, "ffn")         │
    │   ensure_group_on_gpu(N, "ffn")         │
    │   wait_group_ready(N, "ffn")            │ ← 等待 Event_N_ffn
    │          ↓ (wait_event 阻塞点)          │
    │          ├─ [等待 H2D 完成]              │
    │          ↓                               │
    ├──────── [Layer N FFN compute] ──────────┤
    │          w1/w3/silu/w2                   │
    │   _unmark_group_in_use(N, "ffn")        │
    │                                         ↓
```

## 关键同步点详解

### wait_group_ready() 内部逻辑

```python
def wait_group_ready(layer_idx, group, compute_stream):
    key = (layer_idx, group)

    # === 路径 1: 快路径（组已在 GPU）===
    if _group_is_resident(layer_idx, group):
        return  # ✓ 立即返回，无需等待

    # === 路径 2: Inflight 检测（组正在加载中）===
    evt = _gpu_group_inflight.get(key)
    if evt is not None:
        # threading.Event 或 torch.cuda.Event
        if isinstance(evt, threading.Event):
            evt.wait()  # 同步等待 H2D 线程完成
        else:
            compute_stream.wait_event(evt)  # 异步等待 CUDA Event

        # H2D 完成：从 inflight 转为 resident
        _gpu_group_inflight.pop(key)
        _gpu_group_lru.append(key)
        return  # ✓ 完成

    # === 路径 3: 事件同步（组已发起 H2D，等待事件）===
    cuda_evt = _group_ready_events.get(key)
    if cuda_evt is not None:
        s = compute_stream or torch.cuda.current_stream()
        s.wait_event(cuda_evt)  # ⭐ 关键同步点！
        return  # ✓ 计算流已挂到 H2D 事件上

    # === 路径 4: 兜底（组未加载，触发同步预取）===
    if not _group_is_resident(layer_idx, group):
        prefetch_group_async(layer_idx, group, pin=False)
        wait_group_ready(layer_idx, group, compute_stream)  # 递归
```

### 为什么需要 wait_event() 而不是 synchronize()?

| 方法 | 行为 | 影响 |
|------|------|------|
| `stream.synchronize()` | 阻塞 CPU，等待流上所有操作完成 | ❌ 阻塞整个流，影响其它组 |
| `stream.wait_event(evt)` | 在流上插入依赖，仅等待特定事件 | ✓ 仅阻塞依赖该组的操作 |

**示例**：
```python
# ❌ 错误方式（阻塞整个 weight_h2d 流）
weight_h2d_stream.synchronize()  # 等待所有正在传输的组！

# ✓ 正确方式（仅等待需要的组）
compute_stream.wait_event(group_ready_event)  # 仅等待该组
```

## 引用计数与驱逐策略

### in_use 引用计数

```python
_gpu_group_in_use = {
    (0, "attn"): 1,  # Layer 0 ATTN 正在使用
    (1, "ffn"):  2,  # Layer 1 FFN 被两个流引用（嵌套）
    # ...
}
```

### 驱逐时机

```python
def _unmark_group_in_use(layer_idx, group):
    key = (layer_idx, group)
    refcount = _gpu_group_in_use.get(key, 0)

    if refcount <= 1:
        _gpu_group_in_use.pop(key)  # 引用计数降为 0

        if evict_finished_group:
            # 策略 A: 立即驱逐刚完成的组
            _evict_group_immediately(layer_idx, group)
        else:
            # 策略 B: 保留刚完成的组，驱逐其它 LRU 组
            _shrink_gpu_groups_now(exclude={key})
    else:
        _gpu_group_in_use[key] = refcount - 1  # 减少计数
```

## 常见问题排查

### Q1: 为什么还是出现 "illegal memory access"？

**可能原因**：
1. `wait_group_ready()` 未被调用
2. `compute_stream` 传递错误（传了 None）
3. 事件记录位置错误（在 H2D 入队前记录）

**检查清单**：
```python
# ✓ 确保调用顺序正确
wm._mark_group_in_use(layer_id, group)
wm.ensure_group_on_gpu(layer_id, group)  # 记录事件
wm.wait_group_ready(layer_id, group, compute_stream=stream)  # 等待事件

# ✓ 确保 compute_stream 非 None
assert compute_stream is not None, "compute_stream must be set!"
```

### Q2: 为什么性能下降？

**可能原因**：
1. 同步点过多（每次都 synchronize）
2. 未启用后台预取
3. 预取深度不足

**优化建议**：
```python
# 在 ATTN 计算期间预取 FFN
wm.prefetch_group_async(layer_id, "ffn", pin=True)

# 在 FFN 计算期间预取后续层 ATTN
for off in range(1, D+1):
    wm.prefetch_group_async(layer_id + off, "attn")
```

### Q3: 如何调试同步问题？

```bash
# 启用详细日志
export WSM_VERBOSE=1
export WSM_PRINT_GROUPS=1

# 运行推理
python bench_infer.py

# 观察输出：
# [WSM] Marked group (0, attn) as IN_USE (refcount=1)
# [WSM] Group (0, attn) already resident, skip H2D
# [WSM] Unmarked group (0, attn) from IN_USE (refcount=0)
```

## 性能指标

正确实现后应观察到：

| 指标 | 预期值 |
|------|--------|
| 首次权重访问延迟 | < 1ms（已预取） |
| 计算流利用率 | > 90%（流水线重叠） |
| 内存访问错误 | 0 |
| 驱逐误踢率 | < 5%（in_use 保护） |

## 下一步

- [ ] 验证所有层都正确实现同步模式
- [ ] 运行压力测试（bench_infer.py）
- [ ] 检查事件池是否正确回收（防止内存泄漏）
- [ ] 调优预取深度和窗口大小
