# bench_infer.py 运行时配置同步说明

## 概述

本次更新将 `bench_infer.py` 的运行时配置与 `inferencellama3-1-70B.py` 完全对齐，确保 WSM/SSD/KV 行为一致。

## 主要改动

### 1. 环境变量同步（lines 46-75）

添加了与参考脚本完全一致的环境变量配置：

#### 基础环境
- `OMP_NUM_THREADS=8`
- `MALLOC_ARENA_MAX=2`

#### WSM 行为开关
- `WSM_CPU_ROLLING_MODE=1` - 启用滚动滑窗
- `WSM_CPU_RING_OFFSET=0`
- `WSM_CPU_WRAP_AROUND=1` - 窗口末尾回环到 L0
- `WSM_CPU_ROLL_STRIDE=1`
- `WSM_CPU_ROLL_SYNC=1` - 计算线程同步推进
- `WSM_AGGRESSIVE_GPU_PREFETCH=2` - 预取当前层 ffn + 下一层 attn
- `WSM_H2D_GROUP_BACKLOG_MAX=1`
- `WSM_GPU_MAX_GROUPS=10` - GPU 最大组数
- `WSM_SKIP_PRELOAD_WAIT=1` - 不等待预热，边跑边预取
- `WSM_EVICT_FINISHED=1` - 组算完立即驱逐，释放预算

#### KV 写入节流
- `WSM_KV_THROTTLE_THRESHOLD=8`
- `WSM_KV_THROTTLE_MS=16`

#### 预取平衡策略
- `WSM_BALANCE_PREFETCH=1`
- `WSM_BALANCE_TOL=1` - attn/ffn 允许相差 ≤1
- `WSM_PAIR_AHEAD=2` - 就近择层范围
- `WSM_KIND_AHEAD_CAP=2` - 单一类型最大前瞻距离

### 2. 设备一致性与安全裁剪（lines 149-181）

#### 关键改进
```python
# 1. 计算安全的最大 prompt token 数
max_prompt_tokens = llama.args.max_seq_len - max_new_tokens

# 2. 获取与模型一致的计算设备
dev = getattr(llama.model, "device", None)
if dev is None:
    try:
        dev = str(llama.model.embed_tokens.weight.device)
    except Exception:
        dev = llama.args.device
dev = str(dev)

# 3. 先逐条裁剪，再批量对齐分配
batch_tok = []
for p in prompts:
    t = tok.encode(p, add_special_tokens=False)
    if len(t) > max_prompt_tokens:
        # 保留结尾的 max_prompt_tokens 个 token
        t = t[-max_prompt_tokens:]
    batch_tok.append(t)

# 4. 分配 tokens 并回填（使用 torch.as_tensor 提高效率）
tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=dev)
for i, t in enumerate(batch_tok):
    tokens[i, :len(t)] = torch.as_tensor(t, device=dev)
```

### 3. 默认参数调整（lines 370-389）

与参考脚本对齐的默认值：
- `--batch-size`: 4 → **1**
- `--temperature`: 0.0 → **0.6**
- `--warmup-layers`: 2 → **4**
- 新增 `--verbose` 开关

### 4. 构建配置增强（lines 88-99）

添加了 `prefetch_distance_layers` 参数，用于 WSM 内部配置：
```python
mode_cfg.update({
    "prefetch_distance": args.prefetch_distance,
    "group_prefetch_depth": args.group_prefetch_depth,
    "prefetch_distance_layers": args.prefetch_distance,  # 新增
    "verbose": args.verbose,  # 使用命令行参数
    ...
})
```

### 5. Global Tracker 初始化顺序（lines 398-415）

现在在 `LLaMA.build` **之前**初始化 global tracker，与参考脚本一致：
```python
# 1) 先初始化 tracker
tracker = init_global_tracker(max_batch=args.batch_size, layers=80, n_blocks=128)

# 2) 再构建模型
llama = _build_llama(args)
```

## 运行建议

### 基本用法（与 inferencellama3-1-70B.py 一致）

```bash
python bench_infer.py \
  --checkpoints /path/to/Llama3.1-70B \
  --mode mixed \
  --raw-device /dev/nvme0n1p4 \
  --manifest /path/to/runtime_manifest.json \
  --prompts prompts_batch512_len2048.txt \
  --batch-size 1 \
  --max-new-tokens 32 \
  --max-seq-len 2048 \
  --prefetch-distance 6 \
  --group-prefetch-depth 4 \
  --gpu-cached-layers 4 \
  --cpu-cache-layers 40 \
  --warmup-layers 4 \
  --staging-mb 64 \
  --topk-blk 8 \
  --verbose
```

### 输出说明

脚本会输出：
1. **TTFT (Time To First Token)**: 首 token 延迟（ms）
2. **Prefill 时间**: prefill 阶段耗时（ms）
3. **Decode 时间**: decode 阶段耗时（ms）
4. **Throughput**: decode 阶段吞吐量（tokens/s）
5. **E2E 时间**: 端到端总时长（ms）
6. **I/O 带宽**: 权重 H2D、KV H2D/D2H 带宽（MB/s）
7. **GPU 内存**: 分配/保留内存（GB）

详细结果保存在 `bench_results.json` 中。

## 与参考脚本的一致性

| 配置项 | bench_infer.py | inferencellama3-1-70B.py | 状态 |
|--------|----------------|--------------------------|------|
| WSM 环境变量 | ✅ 15 个开关 | ✅ 15 个开关 | 一致 |
| 设备推断逻辑 | ✅ model.device → embed_tokens | ✅ 同样逻辑 | 一致 |
| Token 裁剪 | ✅ 保留尾部 max_prompt_tokens | ✅ 同样逻辑 | 一致 |
| 默认 batch_size | ✅ 1 | ✅ 1 | 一致 |
| 默认 temperature | ✅ 0.6 | ✅ 0.6 | 一致 |
| 默认 warmup_layers | ✅ 4 | ✅ 4 | 一致 |
| Global tracker 初始化 | ✅ build 之前 | ✅ build 之前 | 一致 |
| mode_config 参数 | ✅ 含 prefetch_distance_layers | ✅ 同样参数 | 一致 |

## 关键优势

1. **无越界风险**: 裁剪逻辑确保 `prompt_tokens + max_new_tokens ≤ max_seq_len`
2. **设备一致**: 所有张量（tokens、mask）都在同一设备上
3. **运行时一致**: WSM/SSD/KV 行为完全对齐
4. **低干扰计时**: 最小化 CUDA 同步，仅在关键点记录事件
5. **内存高效**: 使用 `torch.as_tensor` 避免不必要的拷贝

## 注意事项

1. **层数调整**: 如果模型不是 70B（80 层），需要调整 `init_global_tracker(layers=...)`
2. **manifest 路径**: 确保 `--manifest` 指向正确的 runtime_manifest.json
3. **原始设备**: `--raw-device` 需要指向实际的 NVMe 块设备
4. **内存限制**: batch_size=1 是 70B 模型在 24GB GPU 上的安全值

## 测试验证

建议运行以下命令验证配置：
```bash
# 简单测试（单个短 prompt）
python bench_infer.py \
  --checkpoints /path/to/checkpoints \
  --mode mixed \
  --raw-device /dev/nvmeXnY \
  --manifest /path/to/manifest.json \
  --prompt "Hello, world!" \
  --max-new-tokens 16 \
  --verbose

# 完整基准测试（与参考脚本一致）
python bench_infer.py \
  --checkpoints /path/to/Llama3.1-70B \
  --mode mixed \
  --raw-device /dev/nvme0n1p4 \
  --manifest /data1/70b-fixed.runtime_manifest.json \
  --prompts prompts/prompts_batch512_len2048.txt \
  --batch-size 1 \
  --max-new-tokens 32 \
  --warmup-rounds 1
```
