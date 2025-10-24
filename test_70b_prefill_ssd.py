#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, math
from pathlib import Path
import torch

# ===== 你项目内的模块 =====
from llama3.generator import LLaMA
from llama3.config import KVCacheArgs, load_runtime_config, runtime_config_to_dict
# === 在 import LLaMA 后立刻加（runtime 注入日志，不改库） ===
from llama3 import generator as _gen

# 取到原始 build 函数（类方法/静态方法都能用这个包装方式）
_orig_build = _gen.LLaMA.build

def _debug_build(*args, **kw):
    mode       = kw.get("mode", None)
    load_model = kw.get("load_model", None)
    mode_cfg   = kw.get("mode_config", {})
    raw_dev    = (mode_cfg or {}).get("raw_device")
    manifest   = (mode_cfg or {}).get("manifest_path")

    # 决策前日志
    print(f"[MODE-DECISION] called LLaMA.build(mode={mode}, load_model={load_model})")
    # 直接把 use_raw_ssd 的判定在这里计算一遍打印出来
    use_raw_ssd = (mode in {"ssd", "mixed"}) or (mode_cfg and mode_cfg.get("weight_source") == "raw-ssd")
    print(f"[MODE-DECISION] use_raw_ssd={use_raw_ssd} raw_device={raw_dev} manifest={manifest}")

    # 调用原始 build
    llama = _orig_build(*args, **kw)

    # 返回前再根据对象状态判别一次（如果卡在 build 过程中，你至少能看到上面的“决策前日志”）
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
        except: pass
        return out
    s = _grep("/proc/self/status", ["VmRSS","VmHWM","VmLck"])
    m = _grep("/proc/meminfo", ["MemAvailable","CommitLimit","Committed_AS","Cached","Buffers"])
    return s, m

def _gpu_mem():
    if not torch.cuda.is_available(): return {}
    dev = torch.cuda.current_device()
    st  = torch.cuda.memory_stats(dev)
    return {
        "alloc_GB": st.get("allocated_bytes.all.current", 0)/(1<<30),
        "rsrv_GB":  st.get("reserved_bytes.all.current", 0)/(1<<30),
    }

def probe(stage):
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
                if b >= (64<<20): big_cpu.append((n,b))
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
    把注册总量钳在 ≤256MiB（避免 validate 自动上调到覆盖 WEIGHT，导致 N=512），
    并把 EXTENT_BYTES 降到 1MiB，降低高阶页 order 压力。
    """
    cfg = load_runtime_config({
        "pinned": {
            "WEIGHT_PINNED_BYTES":      8  << 30,   # 可进一步降到 4 GiB 验证
            "KV_PINNED_BYTES":          6  << 30,
            "EXTENT_BYTES":             1  << 20,   # 1MiB
            "PINNED_REGISTER_CHUNK":   16  << 20,   # 16MiB
            "PINNED_REGISTER_N":            8,      # 128MiB
        },
        "regpool": {
            "REG_POOL_N_BUFFERS":           8,
            "REG_POOL_BUF_BYTES":     16 << 20,     # ~128MiB 传送带
        }
    })
    D = runtime_config_to_dict(cfg)
    p = D["pinned"]
    need  = int(p["WEIGHT_PINNED_BYTES"])
    chunk = int(p["PINNED_REGISTER_CHUNK"])
    # 把 N 限制到“低于覆盖”的目标（≤256MiB）
    target_total = min(need // 2, 256 << 20)
    newN = max(1, target_total // chunk)
    p["PINNED_REGISTER_N"] = newN
    cfg = load_runtime_config({"pinned": p})   # 二次应用，防止 validate 再上调
    print("[RuntimeConfig] pinned =", runtime_config_to_dict(cfg)["pinned"])
    return cfg

# ---------- KV 池：懒分配 + 单块 ≥ 单个 KV 块 ----------
def configure_kv_pool():
    KVCacheArgs.dram_limit_gb = 12.0
    KVCacheArgs.block_bytes   = 4 * 1024 * 1024
    KVCacheArgs.preallocate   = False
    KVCacheArgs.lazy_init     = True
    if hasattr(KVCacheArgs, "prefer_bf16"):
        KVCacheArgs.prefer_bf16 = True
    print(f"[KVArgs] dram_limit={KVCacheArgs.dram_limit_gb} GiB, block_bytes={KVCacheArgs.block_bytes//(1<<20)} MiB, prealloc={KVCacheArgs.preallocate}")

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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 可选：减少初期碎片/线程数
    os.environ.setdefault("OMP_NUM_THREADS",  "8")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")

    # 1) 覆盖并收敛 pinned/注册池 + KV 池
    apply_runtime_overrides()
    configure_kv_pool()
    probe("after runtime clamp")

    # 2) SSD streaming 配置（小 CPU cache + GPU 预热少量层）
    mode_config = {
        "raw_device": RAW_DEV,
        "ssd_manifest_path": MANIFEST,
        "prefetch_distance": 4,
        "max_cached_layers": 4,
        "cpu_cache_layers": 40,      
        "warmup_layers": 1,         # 仅 GPU 预热第 0 层
        "staging_mb": 64,
        "verbose": True,
    }

    # 3) 构建（meta + SSD 流式），注意不会 torch.load checkpoint
    probe("before LLaMA.build")
    print("[CHECK] calling LLaMA.build(mode='ssd', load_model=False)")
    llama = LLaMA.build(
        checkpoints_dir=CKPT_DIR,
        load_model=False,           # ★ 关键：70B 不把 checkpoint 载入 CPU
        device=device,
        max_seq_len=8192,
        max_batch_size=1,
        topk_blk=8,
        mode="mixed",
        mode_config=mode_config
    )
    probe("after LLaMA.build")

    # 识别/打印“实际模式” + 参数分布
    mode = classify_mode(llama)
    dump_param_inventory(llama.model, f"after build ({mode})")

    # 4) 仅 prefill（不真正 decode）
    prompt = "You are a helpful assistant.\n" + ("Lorem ipsum " * 2000)
    probe("before prefill")
    tokens, texts = llama.text_completion(
        prompts=[prompt],
        temperature=0.0,
        max_gen_len=1,   # 只做 prefill
        batch_size=1,
    )
    probe("after prefill")
    print("OK prefill.")

if __name__ == "__main__":
    main()
