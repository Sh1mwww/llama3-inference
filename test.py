import os, torch
from llama3.generator import LLaMA

# —— 环境开关（优化后的配置）——
os.environ.setdefault("WSM_SKIP_PRELOAD_WAIT", "1")
os.environ.setdefault("WSM_EVICT_FINISHED",   "1")
os.environ.setdefault("WSM_GPU_MAX_GROUPS",   "10")
os.environ.setdefault("WSM_GROUP_PREFETCH_DEPTH", "4")
# ✅ 关键修复：允许 3-4 个组同时排队 H2D（实现流水线）
# 之前=1 导致预取全部被放弃，完全串行！
os.environ.setdefault("WSM_H2D_GROUP_BACKLOG_MAX", "4")  # 从 1 改为 4
os.environ.setdefault("WSM_CPU_RING_MODE", "1")
os.environ.setdefault("WSM_CPU_RING_OFFSET", "1")
os.environ.setdefault("WSM_CPU_CACHE_CAP_LAYERS", "50")
os.environ.setdefault("WSM_CPU_CACHE_HWM_LAYERS", "55")
os.environ.setdefault("WSM_CPU_CACHE_LWM_LAYERS", "45")
os.environ.setdefault("WSM_KV_THROTTLE_THRESHOLD", "8")
os.environ.setdefault("WSM_KV_THROTTLE_MS", "16")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
RAW_DEV  = "/dev/nvme0n1p4"
MANIFEST = "/data1/70b-fixed.runtime_manifest.json"
CKPT_DIR = "/home/roger/.llama/checkpoints/Llama3.1-70B"

llama = LLaMA.build(
    checkpoints_dir=CKPT_DIR,
    load_model=False,   # 关键：raw-ssd 模式不把权重装入 CPU
    device=device,
    max_seq_len=2048,
    max_batch_size=32,
    topk_blk=8,
    mode="mixed",
    mode_config=dict(
        raw_device=RAW_DEV,
        ssd_manifest_path=MANIFEST,
        prefetch_distance=6,
        group_prefetch_depth=4,
        max_cached_layers=4,
        cpu_cache_layers=50,
        warmup_layers=4,
        staging_mb=64,
        verbose=True,
    )
)
text = "Explain how IO–compute overlap is achieved in weight streaming."
_, out = llama.text_completion([text], temperature=0.6, max_gen_len=32, batch_size=4)
print(out[0])