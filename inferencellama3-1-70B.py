#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于 test_70b_prefill_ssd.py 的完整推理脚本：
- 维持相同的运行时配置（WSM/SSD 流式权重、KV 池、env 等）
- 将生成长度改为 max_gen_len=32，并真正 decode 输出文本
- 按需求：在计算第 i 层时，保证 i+1..i+4 的组级权重已在 GPU（事件就绪，无阻塞等待）；
  DRAM 侧维持 i+4 .. i+4+cap 的环形窗口（对 80 层取模）
"""

import os
from pathlib import Path
import torch

# ===== 项目内模块 =====
from llama3.generator import LLaMA
from llama3.config import KVCacheArgs, load_runtime_config, runtime_config_to_dict
from llama3 import generator as _gen

# ========== build() 包装：只加日志，不改库 ==========
_orig_build = _gen.LLaMA.build

def _debug_build(*args, **kw):
    mode       = kw.get("mode", None)
    load_model = kw.get("load_model", None)
    mode_cfg   = (kw.get("mode_config", {}) or {})
    raw_dev    = mode_cfg.get("raw_device")
    manifest   = mode_cfg.get("manifest_path") or mode_cfg.get("ssd_manifest_path")

    print(f"[MODE-DECISION] LLaMA.build(mode={mode}, load_model={load_model})")
    use_raw_ssd = (mode in {"ssd", "mixed"}) or (mode_cfg.get("weight_source") == "raw-ssd")
    print(f"[MODE-DECISION] use_raw_ssd={use_raw_ssd} raw_device={raw_dev} manifest={manifest}")

    llama = _orig_build(*args, **kw)

    has_wsm = hasattr(llama, "weight_streaming_manager")
    if has_wsm:
        wsm = llama.weight_streaming_manager
        ssd = bool(getattr(wsm, "ssd_enabled", False) or getattr(wsm, "ssd", None))
        print(f"[MODE-DECISION] built: WSM present, ssd_enabled={ssd}")
    else:
        print("[MODE-DECISION] built: NO WSM (可能是 full-cpu/full-gpu/旧 streaming)")
    return llama

_gen.LLaMA.build = staticmethod(_debug_build)

# ===== WSM runtime monkey-patch: strict ready + CPU stub loader =====
import types

def _patched_wait_group_ready(self, layer_idx: int, group: str, compute_stream=None):
    """
    等待 (layer_idx, group) 组就绪；事件结束后**二次校验**是否真在 GPU。
    若仍不在，则强制同步 ensure_group_on_gpu()。
    """
    kind = 'attn' if group == 'attn' else 'ffn'
    key  = (int(layer_idx), kind)

    # 0) 快路径：已驻留
    try:
        if self._group_is_resident(*key):
            return
    except Exception:
        pass

    # 1) 若有 inflight 事件：等待
    evt = self._gpu_group_inflight.get(key)
    if evt is not None:
        if compute_stream is not None:
            # 兼容 threading.Event / torch.cuda.Event
            try:
                if hasattr(evt, "wait"):  # threading.Event
                    evt.wait()
                else:
                    compute_stream.wait_event(evt)
            except Exception:
                try:
                    evt.synchronize()
                except Exception:
                    pass
        else:
            try:
                if hasattr(evt, "wait"):
                    evt.wait()
                else:
                    evt.synchronize()
            except Exception:
                pass

        # 从 inflight 转常驻
        with self._group_lock:
            self._gpu_group_inflight.pop(key, None)
            if key not in self._gpu_group_lru:
                self._gpu_group_lru.append(key)
        if getattr(self, "verbose", False):
            print(f"[WSM] H2D completed for {key}")

        # ★ 关键：事件完成后再次校验；不在就同步兜底搬运
        if not self._group_is_resident(*key, wait_for_event=True):
            if getattr(self, "verbose", False):
                print(f"[WSM] Ready event done but {key} not resident; forcing sync ensure")
            self.ensure_group_on_gpu(layer_idx, kind)
        return

    # 2) 若只记录了 CUDA 事件：把 compute_stream 挂到事件上
    cuda_evt = self._group_ready_events.get(key)
    if cuda_evt is not None:
        try:
            dev_obj = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
            s = compute_stream or torch.cuda.current_stream(dev_obj)
            s.wait_event(cuda_evt)
        except Exception:
            try:
                cuda_evt.synchronize()
            except Exception:
                pass

        # 再次校验
        if not self._group_is_resident(*key, wait_for_event=True):
            if getattr(self, "verbose", False):
                print(f"[WSM] CUDA event existed but {key} not resident; forcing sync ensure")
            self.ensure_group_on_gpu(layer_idx, kind)
        return

    # 3) 没有任何可等待对象：直接兜底同步加载
    self.ensure_group_on_gpu(layer_idx, kind)


def _patched_ensure_module_on_gpu(self, m: torch.nn.Module, layer_idx: int | None = None, module_name: str | None = None):
    """
    扩展：把 **0-size CPU stub** 当作 meta 一样处理，优先从 CPU cache 取回并上卡。
    其它情况仍复用原先的 _ensure_param_on_gpu() 路径。
    """
    params_to_replace = {}
    params_full_names = {}

    def _full_name(layer_idx: int, module_name: str, local_param_name: str) -> str:
        if module_name in ("wq", "wk", "wv", "wo"):
            parent = "attention"
        elif module_name in ("w1", "w2", "w3"):
            parent = "feed_forward"
        else:
            parent = module_name or ""
        return f"layers.{layer_idx}.{parent}.{module_name}.{local_param_name}" if parent else f"layers.{layer_idx}.{module_name}.{local_param_name}"

    def _fetch_from_cpu_cache(name: str):
        if (layer_idx is not None) and (layer_idx in self.cpu_cache):
            return self.cpu_cache[layer_idx].get(name)
        return None

    for local_param_name, p in m.named_parameters(recurse=False):
        full_name = None
        if (layer_idx is not None) and (module_name is not None):
            full_name = _full_name(layer_idx, module_name, local_param_name)

        is_meta     = (p.device.type == "meta") or getattr(p, "is_meta", False)
        is_cpu_stub = (p.device.type == "cpu")  and (p.numel() == 0)

        if (is_meta or is_cpu_stub) and self.ssd_enabled and full_name:
            # 确保本层已有 CPU cache（没有就立即加载）
            if (layer_idx not in self.cpu_cache):
                try:
                    self._load_layer_to_cpu(int(layer_idx))
                except Exception:
                    pass

            cached = _fetch_from_cpu_cache(full_name)
            # 形状修复：若 cache 的 key 与期望 shape 不配，尝试同族别名
            expected = tuple(getattr(getattr(m, local_param_name), "shape", ()))
            chosen_name, chosen_tensor = None, None

            def _try_pick(names: list[str]):
                nonlocal chosen_name, chosen_tensor
                for nm in names:
                    t = _fetch_from_cpu_cache(nm)
                    if t is not None and (not expected or tuple(t.shape) == expected):
                        chosen_name, chosen_tensor = nm, t
                        break

            if cached is not None and (not expected or tuple(cached.shape) == expected):
                chosen_name, chosen_tensor = full_name, cached
            else:
                cand = []
                if module_name in ("wq", "wk", "wv"):
                    cand = [f"layers.{layer_idx}.attention.{x}.{local_param_name}" for x in ("wq","wk","wv")]
                elif module_name in ("w1", "w2", "w3"):
                    cand = [f"layers.{layer_idx}.feed_forward.{x}.{local_param_name}" for x in ("w1","w2","w3")]
                else:
                    cand = [full_name]
                _try_pick(cand)
                if chosen_tensor is None and cached is not None:
                    chosen_name, chosen_tensor = full_name, cached  # 退而求其次

            if chosen_tensor is not None:
                with torch.cuda.stream(self._select_h2d_stream_for(module_name=module_name)):
                    p_gpu = chosen_tensor.to(self.device, non_blocking=True)
                params_to_replace[local_param_name] = torch.nn.Parameter(p_gpu, requires_grad=p.requires_grad)
                params_full_names[local_param_name] = chosen_name or full_name
                if getattr(self, "verbose", False):
                    print(f"[WSM DEBUG] ✓ Loaded {'meta' if is_meta else 'stub'} param {params_full_names[local_param_name]} to GPU: {tuple(p_gpu.shape)}")
            else:
                if getattr(self, "verbose", False):
                    print(f"[WSM WARN] CPU cache miss for {full_name} (layer {layer_idx}); will rely on ensure_group_on_gpu() later")
            continue  # 该参数处理完毕

        # 其它情况：沿用原来的 CPU→GPU 逻辑
        self._ensure_param_on_gpu(p, layer_idx, full_name)

    # 安装替换后的 Parameter，并维护 name 映射
    for pname, new_param in params_to_replace.items():
        m._parameters[pname] = new_param
        full = params_full_names.get(pname)
        if full:
            try:
                pobj = getattr(m, pname)
            except Exception:
                pobj = new_param
            self.name_to_param[full] = pobj
            self.param_owner[full]   = (m, pname)

    # buffer 维持原有策略：meta→materialize，CPU→上卡
    for b in m.buffers(recurse=True):
        if getattr(b, "is_meta", False):
            try:
                b = b.to_empty(device=self.device)
            except Exception:
                pass
        elif b.device.type == "cpu":
            with torch.cuda.stream(self._select_h2d_stream_for(module_name=module_name)):
                b_gpu = b.detach().to(self.device, non_blocking=True)
            try:
                b.data = b_gpu
            except Exception:
                pass

# ===== 路径与常量（按你的环境） =====
PROMPT_TXT = Path("/home/roger/llama3-inference/prompts/prompts_batch512_len2048.txt")
RAW_DEV    = "/dev/nvme0n1p4"
MANIFEST   = "/data1/70b-fixed.runtime_manifest.json"
CKPT_DIR   = "/home/roger/.llama/checkpoints/Llama3.1-70B"

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
        except Exception:
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
    KVCacheArgs.dram_limit_gb     = 24.0
    KVCacheArgs.dram_sizing_batch = 32
    KVCacheArgs.block_bytes       = 4 * 1024 * 1024
    KVCacheArgs.preallocate       = False
    KVCacheArgs.lazy_init         = True

    # 关闭 push 即时镜像，采用后移/聚合写（避免与权重 H2D 冲突）
    KVCacheArgs.mirror_on_push = False

    # I/O 节流与写速率配置（与权重 H2D 仲裁）
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
    并打印判据，方便确认现在到底跑的是什么。
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
        print("[MODE] detected=full-gpu"); return "full-gpu"
    if cpu  > 0 and cuda == 0 and meta == 0:
        print("[MODE] detected=full-cpu"); return "full-cpu"
    if meta > 0 and cpu == 0 and cuda == 0:
        print("[MODE] detected=meta-only"); return "meta-only"
    print("[MODE] mixed/unrecognized (check PARAMS dump below)")
    return "unknown"

def main():
    # 基础系统开销收敛
    os.environ.setdefault("OMP_NUM_THREADS",  "8")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")

    # ============================================================
    # ⭐ 组级 GPU 预取（ahead=4）+ 组预算 + 等水位调度
    # ============================================================
    GPU_AHEAD_LAYERS = 4
    # CRITICAL: Reduced from 11 to 6 to reserve ~2-3GB for activation tensors during prefill
    # 70B model: each group ~400-500MB, 6 groups = ~3GB weights, leaving ~12GB for activations
    GPU_MAX_GROUPS   = 6  # Reduced to prevent OOM during long-sequence prefill

    os.environ.setdefault("WSM_GPU_MAX_GROUPS",        str(GPU_MAX_GROUPS))
    os.environ.setdefault("WSM_GROUP_PREFETCH_DEPTH",  str(GPU_AHEAD_LAYERS))
    os.environ.setdefault("WSM_GPU_AHEAD",             str(GPU_AHEAD_LAYERS))  # 供 WSM 读取
    os.environ.setdefault("WSM_BALANCE_PREFETCH",      "1")
    os.environ.setdefault("WSM_PAIR_AHEAD",            "2")  # (i+1..i+2).ffn 顶补
    os.environ.setdefault("WSM_KIND_AHEAD_CAP",        "2")
    os.environ.setdefault("WSM_H2D_GROUP_BACKLOG_MAX", "4")

    # 计算结束立刻释放（避免组堆积）
    os.environ.setdefault("WSM_EVICT_FINISHED", "1")  # ← 修正为 1（你的草稿里误写成了 0）
    os.environ.setdefault("WSM_GRP_RETAIN_MS", "3")   # 极短保留窗口

    # 跳过预加载等待：边跑边滚动预取
    os.environ.setdefault("WSM_SKIP_PRELOAD_WAIT", "1")

    # ============================================================
    # ⭐ 环形 CPU 窗口（SSD -> pinned DRAM，80 层取模）
    # ============================================================
    # CRITICAL FIX: CPU窗口必须从 i+1 开始以覆盖GPU预取需要的层 (i+1..i+4)
    # 如果offset=4，则窗口是[i+4..i+43]，GPU需要的i+1,i+2,i+3不在窗口内！
    CPU_CAP_VALUE    = 40   # 窗口大小：40层
    CPU_RING_OFFSET  = 1    # 窗口从 i+1 起，确保GPU预取的i+1..i+4都在DRAM中
    os.environ.setdefault("WSM_CPU_RING_MODE",     "1")
    os.environ.setdefault("WSM_CPU_RING_OFFSET",   str(CPU_RING_OFFSET))
    os.environ.setdefault("WSM_CPU_CACHE_CAP_LAYERS", str(CPU_CAP_VALUE))
    os.environ.setdefault("WSM_CPU_CACHE_HWM_LAYERS", str(CPU_CAP_VALUE + 3))
    os.environ.setdefault("WSM_CPU_CACHE_LWM_LAYERS", str(max(2, CPU_CAP_VALUE - 3)))
    os.environ.setdefault("WSM_CPU_BACK_MARGIN",   "4")

    # —— H2D/KV 传输仲裁（防止两边抢带宽）——
    os.environ.setdefault("WSM_KV_THROTTLE_THRESHOLD", "2")
    os.environ.setdefault("WSM_KV_THROTTLE_MS",        "16")

    # 配置总结
    print("=" * 80)
    print("🔧 组级 GPU 预取（ahead=4）+ 环形 CPU 窗口 [FIXED VERSION]")
    print("=" * 80)
    print(f"GPU 预取距离: {GPU_AHEAD_LAYERS} 层 (预取 i+1..i+{GPU_AHEAD_LAYERS})")
    print(f"GPU 组预算:   {GPU_MAX_GROUPS} 组(attn/ffn)")
    print(f"CPU 窗口容量: {CPU_CAP_VALUE} 层 (环形，对 80 层取模)")
    print(f"CPU 环形偏移: i+{CPU_RING_OFFSET} ⭐ CRITICAL: 必须覆盖GPU预取层")
    print(f"CPU 窗口范围: [i+{CPU_RING_OFFSET} .. i+{CPU_RING_OFFSET + CPU_CAP_VALUE - 1}]")
    print("=" * 80)
    print(f"⚠️  IMPORTANT: 如果看到此消息但offset={CPU_RING_OFFSET}，说明配置已正确！")
    print(f"⚠️  如果仍有问题，请检查WSM是否真的加载了新代码")
    print("=" * 80)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 1) 覆盖 pinned/注册池 + KV 池
    apply_runtime_overrides()
    configure_kv_pool()
    probe("after runtime clamp")

    # 2) WSM（SSD 流式）构造参数：关闭整层预取，改用组级窗口
    mode_config = {
        "raw_device": RAW_DEV,
        "ssd_manifest_path": MANIFEST,
        "prefetch_distance": 0,                     # 关闭整层预取
        "group_prefetch_depth": GPU_AHEAD_LAYERS,   # 组级预取深度（=4）
        "max_cached_layers": 8,                     # 组级起主导，这里仅作保险
        "cpu_cache_layers": CPU_CAP_VALUE,          # CPU 环形容量
        "warmup_layers": 1,                         # 至少预热第 0 层到 CPU
        "staging_mb": 64,
        "verbose": True,
    }

    # 3) 构建（meta + SSD 流式），不会把 70B 权重全载入 CPU
    probe("before LLaMA.build")
    print("[CHECK] calling LLaMA.build(mode='mixed', load_model=False)")
    llama = LLaMA.build(
        checkpoints_dir=CKPT_DIR,
        load_model=False,           # 关键：不把 checkpoint 载入 CPU
        device=device,
        max_seq_len=2048,
        max_batch_size=32,
        topk_blk=8,
        mode="mixed",
        mode_config=mode_config
    )
    probe("after LLaMA.build")

    # 绑定 WSM 补丁
    wsm = getattr(llama, "weight_streaming_manager", None)
    if wsm is not None:
        wsm.wait_group_ready     = types.MethodType(_patched_wait_group_ready, wsm)
        wsm._ensure_module_on_gpu = types.MethodType(_patched_ensure_module_on_gpu, wsm)
        print("[WSM PATCH] strict group-ready + CPU stub loader enabled")

    # 识别/打印"实际模式" + 参数分布
    mode = classify_mode(llama)
    dump_param_inventory(llama.model, f"after build ({mode})")

    # 4) 读取 prompt 并做“安全裁剪”（max_gen_len=32）
    try:
        prompt_path = PROMPT_TXT
        prompt = prompt_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        raise RuntimeError(f"无法读取 {prompt_path}: {e}")

    # —— 安全裁剪：按 tokenizer 限制 prompt token 数
    max_gen_len = 32  
    max_prompt_tokens = llama.args.max_seq_len - max_gen_len
    tok = llama.tokenizer.encode(prompt, add_special_tokens=False)
    if len(tok) > max_prompt_tokens:
        tok = tok[-max_prompt_tokens:]
        prompt = llama.tokenizer.decode(tok)

    # 5) 真正推理（decode）
    probe("before inference (decode)")
    out_tokens, out_texts = llama.text_completion(
        prompts=[prompt],
        temperature=0.0,
        max_gen_len=max_gen_len,
        batch_size=1,
    )
    probe("after inference (decode)")

    print(f"\n========== Generation (len={max_gen_len}) ==========")
    print(out_texts[0])
    print("=========================================")

if __name__ == "__main__":
    main()
