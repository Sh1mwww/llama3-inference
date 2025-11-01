# 层内同步模式：必经等待点与 in_use 标记

## 概述

本文档说明了在 `SelfAttention` 和 `FeedForward` 层中实现的精确同步模式，确保权重在使用前已完成 H2D 传输。

## 核心模式

每个层的 `forward()` 方法遵循以下模式：

```python
wm = getattr(self, "weight_manager", None)
in_use = False
try:
    # 1. 标记组为使用中（影响驱逐优先级）
    if wm and hasattr(wm, "_mark_group_in_use"):
        wm._mark_group_in_use(layer_id, group_name)
        in_use = True

    # 2. 确保权重在 GPU 上
    if wm and hasattr(wm, "ensure_group_on_gpu"):
        wm.ensure_group_on_gpu(layer_id, group_name)

    # 3. 等待 H2D 完成（关键同步点）
    if wm and hasattr(wm, "wait_group_ready"):
        wm.wait_group_ready(layer_id, group_name, compute_stream=compute_stream)

    # 4. 执行计算...

    return result
finally:
    # 5. 解除使用标记（在 finally 中确保执行）
    if in_use and hasattr(wm, "_unmark_group_in_use"):
        wm._unmark_group_in_use(layer_id, group_name)
```

## 实现位置

### SelfAttention.forward()
- **文件**: [llama3/layers.py:515-535](llama3/layers.py#L515-L535)
- **组名**: `"attn"`
- **计算流**: `self.compute_stream` (compute_mha)
- **解除标记**: [llama3/layers.py:1027-1028](llama3/layers.py#L1027-L1028)

```python
# 在 forward() 开始
wm = getattr(self, "weight_manager", None)
in_use = False
try:
    # 标记组为使用中（影响驱逐优先级）
    if wm and hasattr(wm, "_mark_group_in_use"):
        wm._mark_group_in_use(self.layer_id, "attn")
        in_use = True

    if wm is not None:
        # 确保 attn 组在 GPU（阻塞式，直到权重加载完成）
        if hasattr(wm, "ensure_group_on_gpu"):
            wm.ensure_group_on_gpu(self.layer_id, "attn")
        else:
            # 回退到层级加载
            modules = self._get_modules_dict()
            wm.ensure_weights_cuda(self.layer_id, modules, priority=True)

        # 在 compute stream 上等待 attn 组 H2D 传输完成（关键同步点）
        # wait_group_ready 会把计算流与该组的就绪事件连起来，确保不读未完成的 H2D
        if hasattr(wm, "wait_group_ready"):
            wm.wait_group_ready(self.layer_id, "attn", compute_stream=self.compute_stream)

    # ... 注意力计算 ...

finally:
    if in_use and hasattr(wm, "_unmark_group_in_use"):
        wm._unmark_group_in_use(self.layer_id, "attn")
```

### FeedForward.forward()
- **文件**: [llama3/layers.py:1255-1269](llama3/layers.py#L1255-L1269)
- **组名**: `"ffn"`
- **计算流**: `compute_ffn` 或 `weight_compute`
- **解除标记**: [llama3/layers.py:1328-1330](llama3/layers.py#L1328-L1330)

```python
# 在 forward() 开始
wm = getattr(self, "weight_manager", None)
in_use = False
try:
    # 标记并确保 FFN 组权重在 GPU
    if wm and hasattr(wm, "_mark_group_in_use"):
        wm._mark_group_in_use(self.layer_id, "ffn")
        in_use = True

    if wm is not None:
        if hasattr(wm, "ensure_group_on_gpu"):
            wm.ensure_group_on_gpu(self.layer_id, "ffn")
            # 在 FFN 计算流上等待"组 ready"事件（避免误等别组）
            compute_stream = getattr(self.streams, "compute_ffn", None) or getattr(self.streams, "weight_compute", None)
            if hasattr(wm, "wait_group_ready"):
                wm.wait_group_ready(self.layer_id, "ffn", compute_stream=compute_stream)
        else:
            # 回退：层级 API（不建议，但保底）
            mods = self._get_modules_dict()
            wm.ensure_weights_cuda(self.layer_id, mods, priority=True)

    # ... FFN 计算 ...

finally:
    # 解除 in_use 标记
    if in_use and hasattr(wm, "_unmark_group_in_use"):
        wm._unmark_group_in_use(self.layer_id, "ffn")
```

## WSM API 说明

### `_mark_group_in_use(layer_idx, group)`
- **位置**: [weight_streaming_manager.py:2097-2110](llama3/weight_streaming_manager.py#L2097-L2110)
- **作用**: 标记组为使用中（引用计数），防止被驱逐
- **实现**: 增加引用计数 `_gpu_group_in_use[key]`，并调用 `_touch_group()` 更新 LRU

### `_unmark_group_in_use(layer_idx, group)`
- **位置**: [weight_streaming_manager.py:2112-2135](llama3/weight_streaming_manager.py#L2112-L2135)
- **作用**: 解除组的使用标记（减少引用计数）
- **实现**: 减少引用计数，若降为 0 则从 `_gpu_group_in_use` 中移除
- **驱逐策略**:
  - 若 `evict_finished_group=True`: 立即驱逐刚完成的组
  - 否则: 排除刚完成的组，收缩其它组到上限

### `wait_group_ready(layer_idx, group, compute_stream)`
- **位置**: [weight_streaming_manager.py:2218-2286](llama3/weight_streaming_manager.py#L2218-L2286)
- **作用**: 等待组权重在 H2D 流上的完成事件
- **快路径**: 若组已驻留在 GPU，直接返回
- **Inflight 检测**: 若组在加载中，等待 threading.Event 或 CUDA Event
- **事件同步**: 将 compute_stream 挂到组的 ready 事件上
- **兜底**: 若既不在 inflight 也没有事件，触发同步预取

## 关键优势

1. **精确同步**: `wait_group_ready()` 将计算流与组级 ready 事件连接，确保权重可用
2. **防止驱逐**: `_mark_group_in_use()` 提高使用中组的优先级，防止被意外驱逐
3. **异常安全**: 使用 `try-finally` 确保即使计算失败也会解除标记
4. **流水线友好**: 支持后台预取（在 MHA/FFN 计算期间预取后续层）

## 与预取的配合

在等待本组权重就绪后，层会触发后续层的预取：

### SelfAttention
- 预取本层 FFN 组（带 pin）
- 调用 `rebalance_and_topoff()` 填满窗口

### FeedForward
- 预取后续 D 层的 ATTN 组
- 根据预算限制预取深度

## 注意事项

1. **避免重复调用**: `ensure_group_on_gpu()` 后直接调用 `wait_group_ready()`，无需再次 ensure
2. **流选择**: 确保传递正确的 compute_stream（MHA 用 `compute_mha`，FFN 用 `compute_ffn`）
3. **引用计数**: `_mark_group_in_use()` 使用引用计数，支持嵌套使用（虽然当前不需要）
4. **性能**: `wait_group_ready()` 使用事件同步而非流同步，避免阻塞其它组的传输

## 测试验证

运行以下测试确保同步正确：

```bash
# 运行推理测试
python bench_infer.py

# 检查是否有权重访问错误
# 若同步正确，不应出现 "invalid pointer" 或 "illegal memory access" 错误
```

## 相关文件

- [llama3/layers.py](llama3/layers.py) - 层实现
- [llama3/weight_streaming_manager.py](llama3/weight_streaming_manager.py) - 权重流管理器
- [llama3/generator.py](llama3/generator.py) - 生成器（调用层）
- [llama3/model.py](llama3/model.py) - 模型定义

## 参考

此模式参考了您提供的伪代码，并已在实际代码中完整实现。
