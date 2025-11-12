#!/usr/bin/env python3
"""
完整计算分析:
1. 单个decoder token计算时间
2. Weight streaming时间 (SSD -> DRAM -> GPU -> DRAM -> SSD)
3. Overlap可行性分析
"""

import json
from pathlib import Path

print("=" * 80)
print("70B模型 Overlap可行性分析")
print("=" * 80)

# ============================================================================
# 1. 硬件配置
# ============================================================================
print("\n【1. 硬件配置】")
GPU_NAME = "RTX 5080"
GPU_VRAM_GB = 16
CPU_RAM_GB = 125
CPU_CORES = 20
CPU_MAX_MHZ = 5600

# PCIe带宽 (RTX 5080是PCIe 5.0 x16)
PCIE_GEN = 5
PCIE_LANES = 16
PCIE_BW_GBPS = PCIE_LANES * {3: 1.0, 4: 2.0, 5: 4.0}[PCIE_GEN]  # GB/s 单向
print(f"GPU: {GPU_NAME} ({GPU_VRAM_GB} GB VRAM)")
print(f"PCIe: Gen{PCIE_GEN} x{PCIE_LANES} → 理论带宽 {PCIE_BW_GBPS:.1f} GB/s 单向")

# NVMe SSD带宽 (从你的系统推测，可能是PCIe 4.0 SSD)
# 典型值: PCIe 4.0 x4 SSD → 读取 ~7 GB/s, 写入 ~5 GB/s
NVME_READ_GBPS = 7.0   # GB/s (保守估计)
NVME_WRITE_GBPS = 5.0  # GB/s
print(f"NVMe SSD: 读 {NVME_READ_GBPS} GB/s, 写 {NVME_WRITE_GBPS} GB/s (典型PCIe 4.0 x4)")

# RAM带宽 (DDR5, Intel Core Ultra 7 265K)
# DDR5-5600 dual channel → ~89 GB/s
RAM_BW_GBPS = 89.0  # GB/s
print(f"系统RAM: {CPU_RAM_GB} GB, 带宽 ~{RAM_BW_GBPS} GB/s (DDR5 dual-channel)")

# ============================================================================
# 2. 模型参数量
# ============================================================================
print("\n【2. 模型参数分析】")

# 读取manifest计算实际大小
manifest_path = Path("/data1/70b-fixed.runtime_manifest.json")
try:
    with open(manifest_path) as f:
        manifest = json.load(f)

    # 统计每层的权重大小
    layer_sizes = {}  # layer_idx -> total_bytes
    resident_bytes = 0

    for param in manifest["params"]:
        layer = param["layer"]
        nbytes = param["nbytes"]
        policy = param.get("policy", "stream")

        if policy == "resident":
            resident_bytes += nbytes
        else:
            if layer not in layer_sizes:
                layer_sizes[layer] = 0
            layer_sizes[layer] += nbytes

    # 计算统计
    n_layers = len([k for k in layer_sizes.keys() if k >= 0])
    per_layer_bytes = sum(layer_sizes.values()) / max(n_layers, 1)
    total_streaming_bytes = sum(layer_sizes.values())

    print(f"总层数: {n_layers}")
    print(f"Resident权重 (embedding+norm+output): {resident_bytes / (1<<30):.2f} GB")
    print(f"流式权重总量: {total_streaming_bytes / (1<<30):.2f} GB")
    print(f"平均每层权重: {per_layer_bytes / (1<<30):.3f} GB ({per_layer_bytes / (1<<20):.1f} MB)")

    # 估算单层各组大小 (基于Llama3架构)
    # 每层包含: attention (wq, wk, wv, wo) + feed_forward (w1, w2, w3) + 2x RMSNorm
    # 70B: dim=8192, n_heads=64, ffn_dim=28672
    # wq,wk,wv,wo: 4 * (8192 * 8192) * 2 bytes = 512 MB
    # w1,w2,w3: (8192*28672 + 28672*8192 + 8192*28672) * 2 = ~1.4 GB
    # 总计约 ~1.9 GB/layer

    ATTN_GROUP_MB = 512   # wq+wk+wv+wo
    FFN_GROUP_MB = 1400   # w1+w2+w3

    print(f"\n估算单层组大小:")
    print(f"  Attention组 (wq+wk+wv+wo): ~{ATTN_GROUP_MB} MB")
    print(f"  FFN组 (w1+w2+w3): ~{FFN_GROUP_MB} MB")

except Exception as e:
    print(f"无法读取manifest: {e}")
    # 使用默认值
    n_layers = 80
    per_layer_bytes = 1.9 * (1 << 30)  # 1.9 GB
    ATTN_GROUP_MB = 512
    FFN_GROUP_MB = 1400
    print(f"使用默认值: {n_layers}层, 每层 ~1.9 GB")

# ============================================================================
# 3. 计算decoder单token时间
# ============================================================================
print("\n【3. Decoder单token计算时间】")

# 从你的代码配置和经验数据估算
# 70B模型在RTX 5080 (16GB VRAM) 上做decode
# - Batch size = 1
# - Sequence length = 1 (decode阶段)
# - BF16精度

# 估算方法1: 基于FLOPs
# Decoder单token FLOPs ≈ 2 * n_params (每个参数一次乘法+一次加法)
MODEL_PARAMS = 70e9  # 70B
FLOPS_PER_TOKEN = 2 * MODEL_PARAMS  # 140 GFLOPs

# RTX 5080性能 (BF16/FP16)
# 官方未公布,但根据架构推测 (Blackwell架构, 基于4090的~83 TFLOPs FP16)
# RTX 5080可能在 ~60-80 TFLOPs FP16
GPU_TFLOPS_BF16 = 70.0  # 保守估计

theoretical_time_ms = (FLOPS_PER_TOKEN / 1e9) / GPU_TFLOPS_BF16 * 1000
print(f"理论计算时间 (基于FLOPs):")
print(f"  单token FLOPs: {FLOPS_PER_TOKEN/1e9:.1f} GFLOPs")
print(f"  GPU算力 (BF16): ~{GPU_TFLOPS_BF16} TFLOPs")
print(f"  理论时间: {theoretical_time_ms:.1f} ms/token")

# 估算方法2: 基于实际经验
# 70B模型在高端GPU上decode通常在 50-150 ms/token
# 考虑weight streaming的overhead, 保守估计
EMPIRICAL_TIME_MS = 100.0  # ms/token (经验值,需要实测修正)
print(f"\n实际预估 (考虑memory带宽瓶颈): {EMPIRICAL_TIME_MS:.1f} ms/token")

COMPUTE_TIME_PER_TOKEN = EMPIRICAL_TIME_MS  # 使用保守估计

# ============================================================================
# 4. Weight streaming时间计算
# ============================================================================
print("\n【4. Weight Streaming时间分析】")

print("\n4.1 单层完整流程:")
print("  SSD → DRAM → GPU (forward) → DRAM → SSD")

# 每层需要传输的数据量
LAYER_WEIGHT_GB = per_layer_bytes / (1 << 30)

# 时间拆解:
# (1) SSD → DRAM: per_layer_bytes / NVME_READ_GBPS
time_ssd_to_dram = (LAYER_WEIGHT_GB / NVME_READ_GBPS) * 1000  # ms

# (2) DRAM → GPU: per_layer_bytes / PCIE_BW_GBPS
time_dram_to_gpu = (LAYER_WEIGHT_GB / PCIE_BW_GBPS) * 1000  # ms

# (3) GPU → DRAM (KV cache + 中间结果, 假设和权重同量级或更小)
# 实际上KV cache很小 (~几MB), 但为了保守起见按权重量级估算
time_gpu_to_dram = time_dram_to_gpu  # 双向对称

# (4) DRAM → SSD (写回evicted weights)
time_dram_to_ssd = (LAYER_WEIGHT_GB / NVME_WRITE_GBPS) * 1000  # ms

print(f"\n单层 ({LAYER_WEIGHT_GB:.3f} GB) 各阶段时间:")
print(f"  (1) SSD → DRAM:   {time_ssd_to_dram:.2f} ms  ({NVME_READ_GBPS} GB/s)")
print(f"  (2) DRAM → GPU:   {time_dram_to_gpu:.2f} ms  ({PCIE_BW_GBPS} GB/s)")
print(f"  (3) GPU → DRAM:   {time_gpu_to_dram:.2f} ms  ({PCIE_BW_GBPS} GB/s)")
print(f"  (4) DRAM → SSD:   {time_dram_to_ssd:.2f} ms  ({NVME_WRITE_GBPS} GB/s)")

total_streaming_time = time_ssd_to_dram + time_dram_to_gpu + time_gpu_to_dram + time_dram_to_ssd
print(f"\n  单层总streaming时间: {total_streaming_time:.2f} ms")

# 4.2 组级分析 (Attention vs FFN)
print("\n4.2 组级streaming时间:")
attn_gb = ATTN_GROUP_MB / 1024
ffn_gb = FFN_GROUP_MB / 1024

attn_stream_time = (attn_gb / NVME_READ_GBPS + attn_gb / PCIE_BW_GBPS) * 1000
ffn_stream_time = (ffn_gb / NVME_READ_GBPS + ffn_gb / PCIE_BW_GBPS) * 1000

print(f"  Attention组 ({attn_gb:.2f} GB): SSD→GPU ~{attn_stream_time:.2f} ms")
print(f"  FFN组 ({ffn_gb:.2f} GB): SSD→GPU ~{ffn_stream_time:.2f} ms")

# ============================================================================
# 5. Overlap可行性分析
# ============================================================================
print("\n" + "=" * 80)
print("【5. Overlap可行性分析】")
print("=" * 80)

print(f"\n关键对比:")
print(f"  单token计算时间:        {COMPUTE_TIME_PER_TOKEN:.2f} ms")
print(f"  单层streaming时间:      {total_streaming_time:.2f} ms")
print(f"  比值 (streaming/compute): {total_streaming_time / COMPUTE_TIME_PER_TOKEN:.2f}x")

if total_streaming_time <= COMPUTE_TIME_PER_TOKEN:
    print(f"\n✅ 理论上可以完美overlap!")
    print(f"   Streaming比计算快 {COMPUTE_TIME_PER_TOKEN - total_streaming_time:.2f} ms")
    print(f"   只要提前预取, 计算时权重已就绪")
else:
    print(f"\n❌ 无法完美overlap!")
    print(f"   Streaming比计算慢 {total_streaming_time - COMPUTE_TIME_PER_TOKEN:.2f} ms")
    print(f"   每层会有 {total_streaming_time - COMPUTE_TIME_PER_TOKEN:.2f} ms 的等待")

# 5.1 流水线分析
print(f"\n5.1 流水线overlap策略:")
print(f"  如果提前 N 层预取, 可以让streaming和compute并行")

# 需要多少层的预取深度才能完美overlap?
required_prefetch_depth = total_streaming_time / COMPUTE_TIME_PER_TOKEN
print(f"  所需预取深度: {required_prefetch_depth:.1f} 层")
print(f"  当前配置 GPU_AHEAD_LAYERS=4 → ", end="")
if required_prefetch_depth <= 4:
    print("✅ 足够!")
else:
    print(f"⚠️  不足, 建议增加到 {int(required_prefetch_depth) + 1} 层")

# 5.2 内存需求分析
print(f"\n5.2 内存需求 (按当前配置):")
gpu_ahead = 4
cpu_cap = 40

gpu_memory_needed = gpu_ahead * LAYER_WEIGHT_GB * 2  # 2x因为有attn+ffn组
cpu_memory_needed = cpu_cap * LAYER_WEIGHT_GB

print(f"  GPU显存需求 (预取{gpu_ahead}层): {gpu_memory_needed:.2f} GB")
print(f"  可用GPU显存: {GPU_VRAM_GB} GB → ", end="")
if gpu_memory_needed < GPU_VRAM_GB * 0.7:  # 留30%给激活和KV
    print("✅ 充足")
else:
    print(f"⚠️  紧张 (实际可用需减去激活+KV ~{GPU_VRAM_GB * 0.3:.1f} GB)")

print(f"  DRAM需求 (窗口{cpu_cap}层): {cpu_memory_needed:.2f} GB")
print(f"  可用DRAM: {CPU_RAM_GB} GB → ", end="")
if cpu_memory_needed < CPU_RAM_GB * 0.8:
    print("✅ 充足")
else:
    print("⚠️  紧张")

# 5.3 瓶颈分析
print(f"\n5.3 瓶颈识别:")
bottlenecks = [
    ("SSD读取", time_ssd_to_dram),
    ("DRAM→GPU", time_dram_to_gpu),
    ("GPU→DRAM", time_gpu_to_dram),
    ("DRAM→SSD写回", time_dram_to_ssd),
    ("GPU计算", COMPUTE_TIME_PER_TOKEN / n_layers)  # 单层计算时间
]
bottlenecks.sort(key=lambda x: x[1], reverse=True)
print("  耗时排序:")
for i, (name, time_ms) in enumerate(bottlenecks, 1):
    print(f"    {i}. {name:20s}: {time_ms:.2f} ms")

# ============================================================================
# 6. 优化建议
# ============================================================================
print("\n" + "=" * 80)
print("【6. 优化建议】")
print("=" * 80)

# 6.1 基于瓶颈的建议
main_bottleneck = bottlenecks[0][0]
print(f"\n主要瓶颈: {main_bottleneck}")

if "SSD" in main_bottleneck:
    print("  建议:")
    print("  - 增加SSD读取并发 (多线程I/O)")
    print("  - 使用更快的PCIe 5.0 SSD")
    print("  - 压缩权重 (bf16 → int8/int4)")
elif "DRAM→GPU" in main_bottleneck or "GPU→DRAM" in main_bottleneck:
    print("  建议:")
    print("  - 优化PCIe传输 (pinned memory, 非阻塞传输)")
    print("  - 减少GPU↔DRAM往返 (延迟eviction)")
    print("  - 增大GPU显存占用 (减少换入换出)")
elif "GPU计算" in main_bottleneck:
    print("  建议:")
    print("  - 计算已经是瓶颈, streaming不是问题!")
    print("  - 可以适当增加预取深度以消除等待")

# 6.2 实测建议
print(f"\n实测验证:")
print(f"  运行 inferencellama3-1-70B_timed.py 并记录:")
print(f"  - decode阶段平均 ms/token (从GPU compute events)")
print(f"  - H2D传输时间分布 (从H2D events)")
print(f"  - 是否有GPU idle等待 (compute gaps)")
print(f"  - CPU I/O wait时间 (host_io_wait_ms)")

print("\n" + "=" * 80)
print("分析完成! 建议先运行实测,然后对比理论值")
print("=" * 80)
