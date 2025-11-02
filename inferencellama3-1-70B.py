#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于 test_70b_prefill_ssd.py 的完整推理脚本：
- 维持相同的运行时配置（WSM/SSD 流式权重、KV 池、env 等）
- 仅将生成长度改为 max_gen_len=32，并真正 decode 输出文本
"""

import os
from pathlib import Path
import torch

# ===== 你项目内的模块 =====
from llama3.generator import LLaMA
from llama3.config import KVCacheArgs, load_runtime_config, runtime_config_to_dict
# === 在 import LLaMA 后立刻加（runtime 注入日志，不改库） ===
from llama3 import generator as _gen

# 取到原始 build 函数（类方法/静态方法都能用这个包装方式）
_orig_build = _gen.LLaMA.build

# 与原文件保持一致的 prompt 路径（从文件读入）
PROMPT_TXT = Path("/home/roger/llama3-inference/prompts/prompts_batch512_len2048.txt")

def _debug_build(*args, **kw):
    """
    只包一层日志，不改底层库逻辑。
    """
    mode       = kw.get("mode", None)
    load_model = kw.get("load_model", None)
    mode_cfg   = kw.get("mode_config", {})
    raw_dev    = (mode_cfg or {}).get("raw_device")
    manifest   = (mode_cfg or {}).get("manifest_path") or (mode_cfg or {}).get("ssd_manifest_path")

    # 决策前日志：mixed + raw-ssd 路径
    print(f"[MODE-DECISION] called LLaMA.build(mode={mode}, load_model={load_model})")
    use_raw_ssd = (mode in {"ssd", "mixed"}) or (mode_cfg and mode_cfg.get("weight_source") == "raw-ssd")
    print(f"[MODE-DECISION] use_raw_ssd={use_raw_ssd} raw_device={raw_dev} manifest={manifest}")

    # 调用原始 build
    llama = _orig_build(*args, **kw)

    # 返回前根据对象状态判别一次
    has_wsm = hasattr(llama, "weight_streaming_manager")
    if has_wsm:
        wsm = llama.weight_streaming_manager
        ssd = bool(getattr(wsm, "ssd_enabled", False) or getattr(wsm, "ssd", None))
        print(f"[MODE-DECISION] built: WSM present, ssd_enabled={ssd}")
    else:
        print("[MODE-DECISION] built: NO WSM (可能是 full-cpu/full-gpu/旧 streaming)")
    return llama

# 替换成带日志的 build（保持静态/类方法调用方式）
_gen.LLaMA.build = staticmethod(_debug_build)

# ===== 与原文件一致的路径与常量 =====
RAW_DEV  = "/dev/nvme0n1p4"
MANIFEST = "/data1/70b-fixed.runtime_manifest.json"
CKPT_DIR = "/home/roger/.llama/checkpoints/Llama3.1-70B"

# ---------- 系统/GPU 内存快照 ----------
def _read_status():
    def _grep(path, keys):
        out = {}
        try:
            with open(path, "r") as f:
                for line in f:
                    for k in keys:
                        if line.startswith(k + ":"):
                            out[k] = line.split(":")[1].strip()
        except:
            pass
        return out
    s = _grep("/proc/self/status", ["VmRSS","VmHWM","VmLck"])
    m = _grep("/proc/meminfo", ["MemAvailable","CommitLimit","Committed_AS","Cached","Buffers"])
    return s, m

def _gpu_mem():
    if not torch.cuda.is_available():
        return {}
    dev = torch.cuda.current_device()
    st  = torch.cuda.memory_stats(dev)
    return {
        "alloc_GB": st.get("allocated_bytes.all.current", 0)/(1<<30),
        "rsrv_GB":  st.get("reserved_bytes.all.current", 0)/(1<<30),
    }

def probe(stage: str):
    s, m = _read_status()
    g    = _gpu_mem()
    print(f"\n[MEM] {stage}")
    print(f"  VmRSS={s.get('VmRSS','?')}  VmLck(pinned)={s.get('VmLck','?')}  "
          f"CommitLimit={m.get('CommitLimit','?')}  Committed_AS={m.get('Committed_AS','?')}  "
          f"MemAvailable={m.get('MemAvailable','?')}")
    if g:
        print(f"  GPU: allocated={g['alloc_GB']:.2f} GiB  reserved={g['rsrv_GB']:.2f} GiB")
    print()

# ---------- 扫描参数在 meta/cpu/cuda 上的占用 ----------
def dump_param_inventory(model, tag):
    buckets = {"cpu":0, "cuda":0, "meta":0, "other":0}
    big_cpu = []
    for n,p in model.named_parameters(recurse=True):
        b = p.numel() * p.element_size()
        if getattr(p, "is_meta", False):
            buckets["meta"] += b
        elif hasattr(p, "device"):
            t = p.device.type
            if t == "cpu":
                buckets["cpu"] += b
                if b >= (64<<20):
                    big_cpu.append((n,b))
            elif t == "cuda":
                buckets["cuda"] += b
            else:
                buckets["other"] += b
        else:
            buckets["other"] += b
    f = lambda x: f"{x/(1<<30):.2f} GiB"
    print(f"[PARAMS] {tag}: cpu={f(buckets['cpu'])}, cuda={f(buckets['cuda'])}, meta={f(buckets['meta'])}, other={f(buckets['other'])}")
    if big_cpu:
        big_cpu.sort(key=lambda x:-x[1])
        print("  [big-cpu] top:")
        for n,b in big_cpu[:10]:
            print(f"   - {n}  {b/(1<<20):.1f} MiB")

# ---------- 运行时覆盖：收敛 pinned/注册池 ----------
def apply_runtime_overrides():
    """
    把注册总量钳在 ≤256MiB，并把 EXTENT_BYTES 降到 1MiB，降低高阶页 order 压力。
    """
    cfg = load_runtime_config({
        "pinned": {
            "WEIGHT_PINNED_BYTES":      8  << 30,
            "KV_PINNED_BYTES":          6  << 30,
            "EXTENT_BYTES":             1  << 20,   # 1MiB
            "PINNED_REGISTER_CHUNK":   16  << 20,   # 16MiB
            "PINNED_REGISTER_N":            8,      # 128MiB
        },
        "regpool": {
            "REG_POOL_N_BUFFERS":           8,
            "REG_POOL_BUF_BYTES":     16 << 20,     # ~128MiB 传送带
        },
        "io": {
            "RAW_IO_QD_WRITE":             24,      # 写队列深度
            "IO_RAW_THROTTLE_MS":          30,      # 写带宽窗口
        }
    })
    D = runtime_config_to_dict(cfg)
    p = D["pinned"]
    need  = int(p["WEIGHT_PINNED_BYTES"])
    chunk = int(p["PINNED_REGISTER_CHUNK"])
    target_total = min(need // 2, 256 << 20)  # 目标 ≤ 256MiB
    newN = max(1, target_total // chunk)
    p["PINNED_REGISTER_N"] = newN
    cfg = load_runtime_config({"pinned": p, "io": D["io"]})
    print("[RuntimeConfig] pinned =", runtime_config_to_dict(cfg)["pinned"])
    print("[RuntimeConfig] io =", runtime_config_to_dict(cfg)["io"])
    return cfg

# ---------- KV 池：懒分配 + 单块 ≥ 单个 KV 块 ----------
def configure_kv_pool():
    # DRAM 配置
    KVCacheArgs.dram_limit_gb     = 24.0     # 层内短驻，层尾甩掉
    KVCacheArgs.dram_sizing_batch = 32      # 必须 >= max_batch_size
    KVCacheArgs.block_bytes       = 4 * 1024 * 1024
    KVCacheArgs.preallocate       = False   # 懒分配
    KVCacheArgs.lazy_init         = True

    # 关闭 push 镜像，改为层尾批写
    KVCacheArgs.mirror_on_push = False

    # I/O 节流与写速率配置
    KVCacheArgs.IO_RAW_THROTTLE_MS     = 30
    KVCacheArgs.NVME_WRITE_TARGET_MBPS = 1200

    if hasattr(KVCacheArgs, "prefer_bf16"):
        KVCacheArgs.prefer_bf16 = True

    print(f"[KVArgs] dram_limit={KVCacheArgs.dram_limit_gb} GiB, "
          f"block_bytes={KVCacheArgs.block_bytes//(1<<20)} MiB, prealloc={KVCacheArgs.preallocate}")
    print(f"[KVArgs] mirror_on_push={KVCacheArgs.mirror_on_push}, "
          f"IO_RAW_THROTTLE_MS={KVCacheArgs.IO_RAW_THROTTLE_MS}, "
          f"NVME_WRITE_TARGET_MBPS={KVCacheArgs.NVME_WRITE_TARGET_MBPS}")

# ---------- 识别“实际运行的模式” ----------
def classify_mode(llama) -> str:
    """
    返回：'ssd-streaming' / 'cpu-gpu-streaming' / 'full-gpu' / 'full-cpu' / 'meta-only'
    并打印判据，方便肉眼确认现在到底跑的是什么。
    """
    m = llama.model
    # 1) 是否装了 WSM（并且带 SSD）
    if hasattr(llama, "weight_streaming_manager"):
        wsm = llama.weight_streaming_manager
        ssd = bool(getattr(wsm, "ssd_enabled", False) or getattr(wsm, "ssd", None))
        cpu_warm = getattr(wsm, "disable_cpu_warm", None)
        mode = "ssd-streaming" if ssd else "cpu-gpu-streaming"
        print(f"[MODE] detected={mode}  (has WSM, ssd={ssd}, disable_cpu_warm={cpu_warm})")
        return mode
    # 2) 无 WSM：看参数分布
    cpu, cuda, meta = 0,0,0
    for _,p in m.named_parameters():
        b = p.numel()*p.element_size()
        if getattr(p, "is_meta", False): meta += b
        elif p.device.type == "cpu":     cpu  += b
        elif p.device.type == "cuda":    cuda += b
    if cuda > 0 and cpu == 0 and meta == 0:
        print("[MODE] detected=full-gpu (all params on CUDA)"); return "full-gpu"
    if cpu  > 0 and cuda == 0 and meta == 0:
        print("[MODE] detected=full-cpu (all params on CPU)");  return "full-cpu"
    if meta > 0 and cpu == 0 and cuda == 0:
        print("[MODE] detected=meta-only (skeleton only)");     return "meta-only"
    print("[MODE] mixed/unrecognized (check PARAMS dump below)")
    return "unknown"

def main():
    # （1）基础环境
    # os.environ.setdefault("OMP_NUM_THREADS",  "8")
    # os.environ.setdefault("MALLOC_ARENA_MAX", "2")

    # # （2）WSM 行为开关（滑窗 + 回环 + 组上限等） —— 与原脚本保持一致
    # os.environ.setdefault("WSM_CPU_ROLLING_MODE",  "1")  # 滚动滑窗
    # os.environ.setdefault("WSM_CPU_RING_OFFSET",   "0")
    # os.environ.setdefault("WSM_CPU_WRAP_AROUND",   "1")  # 窗口末尾后回环到 L0
    # os.environ.setdefault("WSM_CPU_ROLL_STRIDE",   "1")
    # os.environ.setdefault("WSM_CPU_ROLL_SYNC",     "1")  # 计算线程同步推进
    # os.environ.setdefault("WSM_AGGRESSIVE_GPU_PREFETCH", "2")  # 当前层 ffn + 下一层 attn
    # os.environ.setdefault("WSM_H2D_GROUP_BACKLOG_MAX",   "1")
    # os.environ.setdefault("WSM_GPU_MAX_GROUPS",          "6")
    # os.environ.setdefault("WSM_SKIP_PRELOAD_WAIT",       "1")  # 不卡在预热等待
    # os.environ.setdefault("WSM_KV_THROTTLE_THRESHOLD",   "8")
    # os.environ.setdefault("WSM_KV_THROTTLE_MS",          "16")

    os.environ.setdefault("OMP_NUM_THREADS",  "8")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")

    # （2）WSM 行为开关（滑窗 + 回环 + 组上限等）
    os.environ.setdefault("WSM_CPU_ROLLING_MODE",  "1")  # 滚动滑窗
    os.environ.setdefault("WSM_CPU_RING_OFFSET",  "0")  
    os.environ.setdefault("WSM_CPU_WRAP_AROUND",   "1")  # 窗口末尾后回环到 L0
    os.environ.setdefault("WSM_CPU_ROLL_STRIDE",   "1")
    os.environ.setdefault("WSM_CPU_ROLL_SYNC",     "1")  # 计算线程同步推进
    os.environ.setdefault("WSM_AGGRESSIVE_GPU_PREFETCH", "2")  # 当前层 ffn + 下一层 attn
    os.environ.setdefault("WSM_H2D_GROUP_BACKLOG_MAX",   "4")
    os.environ.setdefault("WSM_GPU_MAX_GROUPS",          "10")
    os.environ.setdefault("WSM_SKIP_PRELOAD_WAIT",       "1")  # 不卡在预热等待
    os.environ.setdefault("WSM_EVICT_FINISHED",        "1")   # 组算完即踢（释放预算）
    
    os.environ.setdefault("WSM_KV_THROTTLE_THRESHOLD",       "8")
    os.environ.setdefault("WSM_KV_THROTTLE_MS",       "16")
    
    os.environ.setdefault("WSM_BALANCE_PREFETCH", "1")
    os.environ.setdefault("WSM_BALANCE_TOL",      "1")   # attn/ffn 允许相差 ≤1
    os.environ.setdefault("WSM_PAIR_AHEAD",       "2")   # 就近择层范围：同层→i+1→i+2
    os.environ.setdefault("WSM_KIND_AHEAD_CAP",   "2")   # 单一类型最大前瞻距离
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 可选：减少初期碎片/线程数
    os.environ.setdefault("OMP_NUM_THREADS",  "8")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 1) 覆盖并收敛 pinned/注册池 + KV 池
    apply_runtime_overrides()
    configure_kv_pool()
    probe("after runtime clamp")

    # 2) SSD streaming 配置（小 CPU cache + GPU 预热少量层）
    # mode_config = {
    #     "raw_device": RAW_DEV,
    #     "ssd_manifest_path": MANIFEST,
    #     "prefetch_distance": 10,
    #     "max_cached_layers": 4,
    #     "cpu_cache_layers": 30,
    #     "warmup_layers": 4,       # 预热少量层
    #     "staging_mb": 64,
    #     "verbose": True,
    # }
    mode_config = {
        "raw_device": RAW_DEV,
        "ssd_manifest_path": MANIFEST,
        "prefetch_distance": 6,
        "group_prefetch_depth": 4,       # 与下游层内预取循环匹配
        "prefetch_distance_layers": 6,   # 给 WSM 内部 cfg
        "max_cached_layers": 4,
        "cpu_cache_layers": 40,      
        "warmup_layers": 4,         # 仅 GPU 预热第 0 层
        "staging_mb": 64,
        "verbose": True,
    }

    # 3) 构建（meta + SSD 流式），注意不会把 checkpoint 全量载入 CPU
    probe("before LLaMA.build")
    print("[CHECK] calling LLaMA.build(mode='mixed', load_model=False)")
    llama = LLaMA.build(
        checkpoints_dir=CKPT_DIR,
        load_model=False,           # 关键：70B 不把 checkpoint 载入 CPU
        device=device,
        max_seq_len=2048,
        max_batch_size=32,
        topk_blk=8,
        mode="mixed",
        mode_config=mode_config
    )
    probe("after LLaMA.build")

    # 识别/打印“实际模式” + 参数分布
    mode = classify_mode(llama)
    dump_param_inventory(llama.model, f"after build ({mode})")

    # 4) 读取 prompt 并做“安全裁剪”（max_gen_len=32）
    try:
        prompt_path = PROMPT_TXT
        prompt = prompt_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        raise RuntimeError(f"无法读取 {prompt_path}: {e}")

    # —— 安全裁剪：先按 tokenizer 限制 prompt token 数，避免 total_len 溢出
    #     裁剪原则与原脚本相同，只是把 max_gen_len 从 1 改为 32
    max_gen_len = 32
    max_prompt_tokens = llama.args.max_seq_len - max_gen_len
    tok = llama.tokenizer.encode(prompt, add_special_tokens=False)
    if len(tok) > max_prompt_tokens:
        # 保留结尾 max_prompt_tokens 个 token（也可改成保留开头）
        tok = tok[-max_prompt_tokens:]
        prompt = llama.tokenizer.decode(tok)

    # 5) 真正推理（decode），temperature 与 batch_size 维持原值
    probe("before inference (decode)")
    out_tokens, out_texts = llama.text_completion(
        prompts=[prompt],
        temperature=0.0,
        max_gen_len=max_gen_len,
        batch_size=4,
    )
    probe("after inference (decode)")

    print("\n========== Generation (len=32) ==========")
    print(out_texts[0])
    print("=========================================")

if __name__ == "__main__":
    main()
