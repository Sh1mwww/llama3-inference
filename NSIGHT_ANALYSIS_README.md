# Nsight Systems 性能分析指南

本文档介绍如何使用Nsight Systems分析权重流式传输系统的IO/计算重叠效果。

## 🛠️ 环境准备

### 安装Nsight Systems

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nsight-systems-cli

# 或从官网下载
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/...
```

### 验证安装

```bash
nsys --version
nvidia-smi
```

## 🚀 快速开始

### 1. 验证NVTX标记

首先验证NVTX标记是否正确工作：

```bash
# 运行简单测试
nsys profile --trace=nvtx --output=nvtx_test python test_nvtx_simple.py

# 查看结果
nsys-ui nvtx_test.nsys-rep
```

### 2. 完整性能分析

运行完整的权重流式传输分析：

```bash
# 使用自动化脚本
./run_nsight_analysis.sh

# 或手动运行
nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
    --output=weight_analysis python test_nsight_profiling.py
```

## 📊 分析结果解读

### Timeline视图关键指标

#### 1. Stream重叠模式

**理想模式**：
```
CUDA Stream weight_h2d    ████████░░░░████████░░░░████████
CUDA Stream default       ░░░░████████░░░░████████░░░░████
NVTX Ranges              ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```

- ✅ H2D传输(█)与计算(█)时间错开
- ✅ 预取在前一层计算期间进行
- ✅ 最小等待间隙(░)

**问题模式**：
```
CUDA Stream weight_h2d    ████████████████████████████████
CUDA Stream default       ░░░░░░░░░░░░░░░░████████████████
```
- ❌ 序列化执行，无重叠
- ❌ 计算等待传输完成

#### 2. 关键NVTX Ranges

查找以下标记：
- `ensure_layer_X` - 当前层权重加载
- `prefetch_layers_[X,Y]` - 预取操作
- `layer_X_attention` - 注意力计算
- `layer_X_ffn` - 前馈网络计算
- `h2d_transfer_layer_X` - 权重传输
- `layer_X_kv_fetch` - KV缓存获取

### 性能指标目标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| GPU利用率 | > 85% | 计算核心使用率 |
| 内存带宽利用率 | > 80% | GPU内存带宽 |
| PCIe利用率 | 50-70% | CPU-GPU传输 |
| 重叠效率 | > 60% | IO与计算重叠比例 |
| 同步等待时间 | < 10% | 流同步开销 |

## 🔍 常见问题诊断

### 问题1：重叠效率低

**症状**：Timeline显示传输和计算串行执行
**可能原因**：
- prefetch_distance设置过小
- max_cached_layers不足
- 内存带宽饱和

**解决方案**：
```python
streaming_config = {
    "prefetch_distance": 3,      # 增加预取距离
    "max_cached_layers": 6,      # 增加缓存层数
}
```

### 问题2：GPU利用率低

**症状**：GPU Utilization < 70%
**可能原因**：
- 权重传输成为瓶颈
- 同步操作过多
- 缓存命中率低

**解决方案**：
- 检查同步点的频率
- 优化权重传输优先级
- 调整预取策略

### 问题3：内存带宽饱和

**症状**：Memory Bandwidth持续100%
**可能原因**：
- 过度预取
- 传输块大小不当
- 多个流竞争带宽

**解决方案**：
```python
# 减少并发传输
streaming_config = {
    "prefetch_distance": 1,      # 减少预取距离
}
```

## 📈 命令行分析

### 生成统计报告

```bash
# GPU操作统计
nsys stats --report gputrace,cudaapisum analysis.nsys-rep

# 内存传输统计
nsys stats --report memop analysis.nsys-rep

# NVTX范围统计
nsys stats --report nvtxsum analysis.nsys-rep

# 完整报告
nsys stats --report summary analysis.nsys-rep
```

### 筛选特定操作

```bash
# 只看权重相关操作
nsys stats --report nvtxsum analysis.nsys-rep | grep -E "layer_|prefetch_|h2d_"

# 查看Stream活动
nsys stats --report cudaapisum analysis.nsys-rep | grep -E "Stream|Event"
```

## 🔧 高级分析技巧

### 对比分析

```bash
# 基准测试（无流式传输）
nsys profile --output=baseline python test_baseline.py

# 流式传输测试
nsys profile --output=streaming python test_streaming.py

# 对比结果
nsys-ui baseline.nsys-rep streaming.nsys-rep
```

### 热点分析

```bash
# 采样CPU性能
nsys profile --sample=cpu --output=cpu_analysis python test_nsight_profiling.py

# 内存页面错误
nsys profile --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true \
    --output=memory_analysis python test_nsight_profiling.py
```

### 自定义筛选

在Timeline视图中：
1. 使用时间范围选择器聚焦特定时间段
2. 使用过滤器只显示特定类型的事件
3. 使用标尺测量精确的时间间隔

## 📝 报告模板

### 性能分析报告

```
权重流式传输性能分析报告
==============================

测试配置：
- 模型: LLaMA-XX
- 设备: [GPU型号]
- 配置: prefetch_distance=X, max_cached_layers=Y

关键指标：
- GPU利用率: XX%
- 内存带宽利用率: XX%
- 重叠效率: XX%
- 平均延迟: XX ms

优化建议：
1. [具体建议1]
2. [具体建议2]
3. [具体建议3]
```

## 🆘 故障排除

### 常见错误

1. **nsys: command not found**
   ```bash
   sudo apt install nsight-systems-cli
   ```

2. **CUDA context error**
   ```bash
   # 检查CUDA驱动
   nvidia-smi
   # 重新安装CUDA
   ```

3. **Permission denied**
   ```bash
   chmod +x run_nsight_analysis.sh
   ```

4. **No NVTX ranges visible**
   ```bash
   # 确保使用--trace=nvtx
   nsys profile --trace=nvtx,cuda ...
   ```

### 获取帮助

```bash
# 查看nsys帮助
nsys profile --help

# 查看可用追踪选项
nsys profile --trace=help

# 查看报告类型
nsys stats --help
```

## 📚 参考资源

- [Nsight Systems官方文档](https://docs.nvidia.com/nsight-systems/)
- [NVTX用户指南](https://docs.nvidia.com/cuda/nvtx/)
- [CUDA最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

---

📧 如有问题，请检查日志输出或参考官方文档。