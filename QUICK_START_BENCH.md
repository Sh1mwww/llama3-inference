# bench_infer.py 快速启动指南

## 一键运行（与 inferencellama3-1-70B.py 完全一致的配置）

```bash
python bench_infer.py \
  --checkpoints /home/roger/.llama/checkpoints/Llama3.1-70B \
  --mode mixed \
  --raw-device /dev/nvme0n1p4 \
  --manifest /data1/70b-fixed.runtime_manifest.json \
  --prompts /home/roger/llama3-inference/prompts/prompts_batch512_len2048.txt \
  --batch-size 1 \
  --max-new-tokens 32 \
  --max-seq-len 2048 \
  --temperature 0.6 \
  --top-p 0.9 \
  --warmup-rounds 1 \
  --prefetch-distance 6 \
  --group-prefetch-depth 4 \
  --gpu-cached-layers 4 \
  --cpu-cache-layers 40 \
  --warmup-layers 4 \
  --staging-mb 64 \
  --topk-blk 8 \
  --out-json bench_70B_results.json \
  --verbose
```

## 环境变量（已自动设置，无需手动导出）

脚本内部已自动设置以下环境变量：

```bash
# 基础环境
OMP_NUM_THREADS=8
MALLOC_ARENA_MAX=2

# WSM 核心配置
WSM_CPU_ROLLING_MODE=1           # CPU 滑窗模式
WSM_CPU_WRAP_AROUND=1            # 回环到 L0
WSM_CPU_ROLL_SYNC=1              # 同步推进
WSM_AGGRESSIVE_GPU_PREFETCH=2    # 激进预取
WSM_GPU_MAX_GROUPS=10            # GPU 最大组数
WSM_SKIP_PRELOAD_WAIT=1          # 跳过预热等待
WSM_EVICT_FINISHED=1             # 组算完即驱逐

# KV 节流
WSM_KV_THROTTLE_THRESHOLD=8
WSM_KV_THROTTLE_MS=16

# 预取平衡
WSM_BALANCE_PREFETCH=1
WSM_BALANCE_TOL=1
WSM_PAIR_AHEAD=2
WSM_KIND_AHEAD_CAP=2
```

## 核心特性

### 1. 安全裁剪机制
- 自动限制 `prompt_tokens ≤ max_seq_len - max_new_tokens`
- 超长 prompt 保留**尾部** tokens（与参考脚本一致）

### 2. 设备一致性
- 自动检测模型计算设备：`model.device` → `embed_tokens.weight.device` → `args.device`
- 所有张量（tokens, masks）在同一设备上分配

### 3. 精准计时
- **TTFT**: Time to First Token
- **Prefill**: 预填充阶段耗时
- **Decode**: 解码阶段耗时
- **Throughput**: tokens/s（decode 阶段）
- **E2E**: 端到端总时长

### 4. I/O 带宽监控
- 权重 H2D 带宽（MB/s）
- KV H2D/D2H 带宽（MB/s）

## 输出示例

```
[INFO] Global state tracker initialized
[MODE-DECISION] called LLaMA.build(mode=mixed, load_model=False)
[MODE-DECISION] use_raw_ssd=True raw_device=/dev/nvme0n1p4
[INFO] raw-ssd mode: large weights replaced with CPU stubs
[WSM] Initializing SSD backend...
✅ Weight streaming enabled (SSD -> CPU(pinned) -> GPU by layer)

=== Benchmark Summary ===
                 ttft_ms: 1234.567
             prefill_ms: 1200.123
  first_token_decode_ms: 34.444
              decode_ms: 890.234
                 e2e_ms: 2090.357
        throughput_tok_s: 35.87
     end_to_end_tok_s: 28.34

详细 JSON 已保存到: bench_70B_results.json
```

## JSON 输出格式

```json
{
  "batch_size": 1,
  "max_new_tokens": 32,
  "prompt_lens": [2016],
  "prefill_tokens_total": 2016,
  "decode_tokens_total": 32,
  "prefill_ms": 1200.123,
  "first_token_decode_ms": 34.444,
  "ttft_ms": 1234.567,
  "decode_ms": 890.234,
  "e2e_ms": 2090.357,
  "throughput_tok_s": 35.87,
  "end_to_end_tok_s": 28.34,
  "io_bw_MBps": {
    "weight_h2d_MBps": 3456.78,
    "kv_h2d_MBps": 234.56,
    "kv_d2h_MBps": 189.12
  },
  "gpu_mem_GB": {
    "allocated_GB": 22.45,
    "reserved_GB": 23.12
  },
  "mode": "mixed",
  "device": "cuda:0",
  "wsm": {
    "enabled": true,
    "ssd_enabled": true,
    "gpu_max_groups": 10
  }
}
```

## 常见参数调整

### 调整吞吐量优先级
```bash
# 更激进的预取（提高吞吐，增加显存占用）
--prefetch-distance 8 \
--gpu-cached-layers 6 \
--cpu-cache-layers 50
```

### 调整延迟优先级
```bash
# 减少预取（降低 TTFT，降低吞吐）
--prefetch-distance 4 \
--gpu-cached-layers 2 \
--warmup-layers 6
```

### 调整内存占用
```bash
# 降低显存占用
--gpu-cached-layers 2 \
--staging-mb 32

# 降低 CPU pinned 内存占用
--cpu-cache-layers 20
```

## 故障排查

### 1. CUDA OOM
```bash
# 减少 GPU 缓存层数
--gpu-cached-layers 2

# 减小 staging buffer
--staging-mb 32

# 减小 batch size
--batch-size 1
```

### 2. CPU 内存不足
```bash
# 减少 CPU 缓存层数
--cpu-cache-layers 20

# 确保使用 mixed 模式（不加载完整 checkpoint）
--mode mixed --load_model=False
```

### 3. I/O 瓶颈（低带宽）
```bash
# 检查 NVMe 设备
sudo nvme smart-log /dev/nvme0n1

# 检查是否有其他进程竞争 I/O
iotop -o

# 调整 staging buffer 大小
--staging-mb 128
```

### 4. 预取不平衡
查看日志中的 `[WSM][groups]` 输出，如果 attn/ffn 不平衡：
```bash
# 调整平衡参数（已通过环境变量设置）
# WSM_BALANCE_TOL=1
# WSM_PAIR_AHEAD=2
```

## 性能基线（参考）

### Llama-3.1-70B @ 4090 24GB

| 配置 | TTFT (ms) | Decode (tok/s) | GPU Mem (GB) |
|------|-----------|----------------|--------------|
| mixed (SSD) | ~1200 | ~35 | ~22 |
| stream (CPU) | ~800 | ~30 | ~23 |
| preload | ~600 | ~40 | OOM |
| full | N/A | N/A | OOM |

*注：实际性能取决于硬件配置（NVMe 速度、PCIe 带宽、CPU 等）*

## 下一步

1. 运行基准测试并保存结果
2. 比对 `bench_70B_results.json` 与 `inferencellama3-1-70B.py` 的输出
3. 调整参数以优化 TTFT/Throughput 平衡
4. 使用 `--verbose` 查看详细的 WSM 日志
