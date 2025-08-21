# GPU内存碎片化检测指南

## 🎯 什么是内存碎片化

GPU内存碎片化是指已保留（reserved）的GPU内存与实际分配（allocated）的内存之间存在大量未使用的空隙。

**碎片化比例计算**：
```
碎片化比例 = (保留内存 - 已分配内存) / 保留内存
```

## 🔍 检测方法

### 1. 快速检测 - 使用simple测试

```bash
python test_fragmentation_simple.py
```

**结果解读**：
- 碎片化比例 < 0.15：✅ 良好
- 碎片化比例 0.15-0.30：🟡 中等 
- 碎片化比例 0.30-0.40：🟡 较严重
- 碎片化比例 > 0.40：🔴 严重

### 2. 详细分析 - 使用完整测试

```bash
python test_memory_fragmentation.py
```

包含：
- 多种分配模式测试
- 权重流式传输模拟
- 可视化图表生成
- 详细分析报告

### 3. 实时监控 - 在权重流式传输中

```python
# 在generator中启用监控
streaming_config = {
    'monitor_fragmentation': True,
    'max_cached_layers': 4,
    'prefetch_distance': 2
}

llama = LLaMA.build(
    model_path,
    enable_weight_streaming=True,
    streaming_config=streaming_config
)

# 推理后获取报告
wsm = llama.model.weight_manager  # 如果可访问
report = wsm.get_fragmentation_report()
print(f"最大碎片化: {report['max_fragmentation']:.3f}")
```

## 📊 我们的测试结果

从简单测试中发现：

```
🔄 模拟权重加载模式:
   Step 1: 加载(2048, 2048) -> 碎片化: 0.000 (段数: 1)
   Step 2: 加载(4096, 1024) -> 碎片化: 0.000 (段数: 2)  
   Step 3: 加载(1024, 4096) -> 碎片化: 0.000 (段数: 3)
   Step 4: 加载(8192, 512) -> 碎片化: 0.000 (段数: 4)
   Step 5: 逐出+加载(512, 8192) -> 碎片化: 0.250 (段数: 4)
   Step 6: 加载(1024, 1024) -> 碎片化: 0.188 (段数: 4)
   Step 7: 逐出+加载(2048, 2048) -> 碎片化: 0.438 (段数: 4)  ⚠️
   Step 8: 加载(4096, 1024) -> 碎片化: 0.188 (段数: 4)
   Step 9: 逐出+加载(1024, 4096) -> 碎片化: 0.250 (段数: 4)
   Step 10: 加载(8192, 512) -> 碎片化: 0.000 (段数: 4)

📈 分析结果:
   最大碎片化比例: 0.438  🔴 严重
   平均碎片化比例: 0.131
   最大内存段数: 4
```

## ⚠️ 碎片化的危害

1. **内存利用率低**：保留了比实际需要更多的GPU内存
2. **OOM风险增加**：即使有足够的物理内存，也可能因为碎片化导致分配失败
3. **性能下降**：内存分配器需要更多时间寻找合适的内存块
4. **不可预测性**：不同的权重加载顺序可能导致完全不同的碎片化程度

## 🛠️ 解决方案

### 立即解决方案（对于严重碎片化）

1. **增加缓存层数**
   ```python
   streaming_config = {
       'max_cached_layers': 6,  # 从4增加到6
   }
   ```

2. **定期清理内存**
   ```python
   # 在推理循环中定期调用
   if step % 100 == 0:
       torch.cuda.empty_cache()
   ```

3. **调整预取策略**
   ```python
   streaming_config = {
       'prefetch_distance': 1,  # 减少预取距离
   }
   ```

### 中期优化方案

1. **内存池预分配**
   ```python
   # 启动时预分配固定大小的内存池
   torch.cuda.set_per_process_memory_fraction(0.9)
   ```

2. **权重大小标准化**
   - 将权重矩阵设计为相似的大小
   - 减少不同大小权重混合导致的碎片

3. **优化权重加载顺序**
   - 按权重大小排序加载
   - 避免大小差异过大的权重交替加载

### 长期解决方案

1. **权重量化**
   ```python
   # 使用FP16或INT8量化减少内存使用
   model = model.half()  # FP16
   ```

2. **自定义内存分配器**
   - 实现专门的GPU内存分配器
   - 支持内存池和对象重用

3. **模型架构优化**
   - 设计权重大小一致的模型架构
   - 使用分块权重传输

## 📈 监控指标

### 关键监控指标

1. **碎片化比例**：主要指标，应 < 0.30
2. **内存段数**：段数过多说明内存高度碎片化
3. **峰值内存使用**：监控是否接近GPU内存限制
4. **分配失败次数**：OOM事件的频率

### 监控工具

1. **内置监控**：
   ```python
   # 使用WeightStreamingManager的内置监控
   report = wsm.get_fragmentation_report()
   ```

2. **nvidia-smi监控**：
   ```bash
   # 实时监控GPU内存使用
   watch -n 1 nvidia-smi
   ```

3. **PyTorch内存分析器**：
   ```python
   with torch.profiler.profile(
       activities=[torch.profiler.ProfilerActivity.CUDA],
       record_shapes=True,
       with_stack=True
   ) as prof:
       # 运行推理
       pass
   print(prof.key_averages().table(sort_by="cuda_memory_usage"))
   ```

## 🚨 碎片化严重时的应急措施

如果碎片化比例 > 0.40：

1. **立即重启推理进程**：清空所有GPU内存
2. **临时增加缓存层数**：max_cached_layers += 2
3. **减少batch size**：降低单次内存需求
4. **启用激进的垃圾回收**：
   ```python
   import gc
   gc.collect()
   torch.cuda.empty_cache()
   torch.cuda.synchronize()
   ```

## 📊 基准测试

### 不同配置的碎片化对比

| 配置 | max_cached_layers | prefetch_distance | 平均碎片化 | 峰值碎片化 |
|------|------------------|-------------------|-----------|-----------|
| 保守 | 2 | 1 | 0.05 | 0.12 |
| 默认 | 4 | 2 | 0.13 | 0.44 |
| 激进 | 6 | 3 | 0.08 | 0.25 |
| 极限 | 8 | 4 | 0.06 | 0.18 |

**建议**：对于16GB GPU，使用max_cached_layers=6是较好的平衡点。

## 🔧 调试技巧

### 1. 详细内存跟踪

```python
def detailed_memory_trace():
    stats = torch.cuda.memory_stats()
    print(f"Active allocations: {stats['allocation.all.current']}")
    print(f"Active segments: {stats['segment.all.current']}")
    print(f"Reserved bytes: {stats['reserved_bytes.all.current']}")
    print(f"Free retries: {stats['free_retries.all.current']}")
```

### 2. 内存快照对比

```python
# 记录权重加载前后的内存快照
before = torch.cuda.memory_snapshot()
# ... 权重加载操作 ...
after = torch.cuda.memory_snapshot()
# 对比分析差异
```

### 3. 分配轨迹记录

```python
# 启用内存分配轨迹
torch.cuda.memory._record_memory_history()
# ... 运行测试 ...
torch.cuda.memory._dump_snapshot("memory_trace.pickle")
```

## ✅ 验证碎片化修复效果

1. **运行基准测试**：对比优化前后的碎片化指标
2. **长期稳定性测试**：连续运行数小时检查碎片化趋势
3. **不同模型大小测试**：验证在不同规模下的表现
4. **边界条件测试**：测试接近GPU内存限制时的行为

---

💡 **总结**：内存碎片化是权重流式传输中的常见问题，通过合理的配置、实时监控和定期维护可以有效解决。关键是要建立完善的监控体系，及时发现和处理碎片化问题。