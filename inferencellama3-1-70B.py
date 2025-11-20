#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Llama3.1-70B 推理 + 轻量级 Profiler（JSON/CSV 自动写入固定目录）
- 记录会影响 inference 的关键路径用时（prefill / decode per-token / e2e / FTL 近似 / 吞吐）
- 同时记录不会影响 inference 的准备/探针/日志时间（non_inference 类别）
- 对 WSM 两个关键函数（wait_group_ready / _ensure_module_on_gpu）做埋点统计
- 采用 CUDA Events 逐 token 计时，统一同步，尽量低扰动
- 生成结果固定写入 LOG_DIR，自动输出 JSON + CSV 两种格式
"""

import os
from pathlib import Path
import types
import json, csv, uuid, platform, math, time, re
from datetime import datetime, timezone
from contextlib import contextmanager, nullcontext

# 🔥 CUDA 内存分配器配置（必须在 import torch 之前）
# expandable_segments 与异步流操作可能有冲突，暂时禁用
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 限制分块大小，减少碎片

# 🔥 WSM 无兜底策略：锁定事件驱动调度，禁用同步兜底 (no-fallback)
os.environ["WSM_NO_FALLBACK"] = "1"

import torch

# ===== 你可以改这里：日志输出目录 & 运行标签（可留空） =====
LOG_DIR = Path("/home/roger/logs")   # 自动创建
RUN_TAG = ""                         # 例如 "ablation-a1"；留空则自动仅用 run_id

# ===== 项目内模块 =====
from llama3.generator import LLaMA
from llama3.config import KVCacheArgs, load_runtime_config, runtime_config_to_dict
from llama3 import generator as _gen, stream_mnt

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

# ======= 轻量级 Profiler（低扰动；CUDA Events；保存 JSON/CSV） =======
PROFILER = None  # 全局句柄

def _now_utc():
    return datetime.now(timezone.utc).isoformat()

def _flatten_extras(extras: dict):
    out = {}
    for k,v in (extras or {}).items():
        out[k] = v if (isinstance(v,(int,float,str,bool)) or v is None) else str(v)
    return out

class InferenceProfiler:
    def __init__(self, run_name: str | None = None):
        self.run_id   = run_name or f"run-{uuid.uuid4().hex[:8]}"
        self.t0_ns    = time.perf_counter_ns()
        self.timeline = []   # 墙钟阶段
        self.active   = False
        self.cuda     = torch.cuda.is_available()
        self.forward_events = []      # GPU：[(kind,batch,seqlen,start_ev,end_ev)]
        self.forward_events_cpu = []  # CPU 回退：[(kind,batch,seqlen,dt_ms)]
        self.bookkeep  = {}
        self.meta      = {
            "started_at_utc": _now_utc(),
            "python": platform.python_version(),
            "torch": getattr(torch, "__version__", "unknown"),
            "device": ("cuda" if self.cuda else "cpu"),
        }
        if self.cuda:
            try:
                self.meta["cuda_device_name"] = torch.cuda.get_device_name(0)
                self.meta["cuda_cc"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            except Exception:
                pass

    @contextmanager
    def span(self, name: str, category: str, **extras):
        s = time.perf_counter_ns()
        try:
            yield
        finally:
            e = time.perf_counter_ns()
            rec = {
                "name": name, "cat": category,
                "t_start_ms": (s - self.t0_ns) / 1e6,
                "t_end_ms":   (e - self.t0_ns) / 1e6,
                "dur_ms":     (e - s) / 1e6,
            }
            rec.update(_flatten_extras(extras))
            self.timeline.append(rec)
            if name == "inference_e2e":
                self.bookkeep["inference_s_ns"] = s
                self.bookkeep["inference_e_ns"] = e

    @contextmanager
    def inference_scope(self):
        self.active = True
        with self.span("inference_e2e", "inference"):
            yield
        self.active = False

    def wrap_model_forward(self, model):
        orig = model.forward

        def _classify_args(args, kwargs):
            cand = None
            for k in ("tokens", "input_ids"):
                t = kwargs.get(k, None)
                if torch.is_tensor(t) and t.dim() == 2:
                    cand = t; break
            if cand is None:
                for a in args:
                    if torch.is_tensor(a) and a.dtype in (torch.long, torch.int32, torch.int64) and a.dim() == 2:
                        cand = a; break
            if cand is None:
                return None, None
            B, T = int(cand.size(0)), int(cand.size(1))
            return B, T

        def wrapped(*args, **kwargs):
            if not self.active:
                return orig(*args, **kwargs)
            B, T = _classify_args(args, kwargs)
            kind = "prefill" if (T is not None and T > 1) else ("decode" if T == 1 else "unknown")

            if self.cuda:
                s_ev = torch.cuda.Event(enable_timing=True)
                e_ev = torch.cuda.Event(enable_timing=True)
                s_ev.record()
                out = orig(*args, **kwargs)
                e_ev.record()
                self.forward_events.append((kind, B, T, s_ev, e_ev))
                return out
            else:
                s = time.perf_counter_ns()
                out = orig(*args, **kwargs)
                e = time.perf_counter_ns()
                self.forward_events_cpu.append((kind, B, T, (e - s) / 1e6))
                return out

        model.forward = wrapped

    # 供 WSM 补丁使用
    def span_if_active(self, name, category, **extras):
        return self.span(name, category, **extras) if self is not None else nullcontext()

    def _compute_decode_stats(self, arr):
        if not arr:
            return {"count": 0}
        s = sorted(arr)
        q = lambda p: s[int((len(s)-1)*p)]
        return {
            "count": len(arr),
            "sum_ms": sum(arr),
            "mean_ms": sum(arr)/len(arr),
            "p50_ms": q(0.50),
            "p90_ms": q(0.90),
            "p99_ms": q(0.99),
        }

    def finalize(self, tokens_in: int | None, tokens_out: int | None, extra_meta: dict | None = None):
        if extra_meta: self.meta.update(_flatten_extras(extra_meta))
        # 统一同步后读取 CUDA Event
        decode_ms = []
        prefill_total = 0.0
        if self.cuda and self.forward_events:
            torch.cuda.synchronize()
            for kind, B, T, s_ev, e_ev in self.forward_events:
                dt = float(s_ev.elapsed_time(e_ev))  # ms
                if kind == "prefill": prefill_total += dt
                elif kind == "decode": decode_ms.append(dt)
        elif self.forward_events_cpu:
            for kind, B, T, dt in self.forward_events_cpu:
                if kind == "prefill": prefill_total += dt
                elif kind == "decode": decode_ms.append(dt)

        inf_span = next((x for x in self.timeline if x["name"]=="inference_e2e"), None)
        e2e_ms = inf_span["dur_ms"] if inf_span else None
        first_decode_ms = decode_ms[0] if decode_ms else None
        ftl_approx = (prefill_total + first_decode_ms) if (first_decode_ms is not None) else None

        # 分类聚合
        sum_cat = {}
        for ev in self.timeline:
            sum_cat.setdefault(ev["cat"], 0.0)
            sum_cat[ev["cat"]] += float(ev["dur_ms"])

        def _sum_by_name_prefix(prefix):
            items = [ev for ev in self.timeline if ev["name"].startswith(prefix)]
            return {"calls": len(items), "total_ms": sum(float(ev["dur_ms"]) for ev in items)}
        wsm_stats = {
            "wait_group_ready": {"calls": _sum_by_name_prefix("wsm.wait_group_ready")["calls"],
                                 "total_ms": _sum_by_name_prefix("wsm.wait_group_ready")["total_ms"]},
            "ensure_module_on_gpu": {"calls": _sum_by_name_prefix("wsm.ensure_module_on_gpu")["calls"],
                                     "total_ms": _sum_by_name_prefix("wsm.ensure_module_on_gpu")["total_ms"]},
        }

        # 吞吐
        prefill_tps = (float(tokens_in)/ (prefill_total/1000.0)) if (tokens_in and prefill_total>0) else None
        decode_tps  = (float(tokens_out)/ (sum(decode_ms)/1000.0)) if (tokens_out and decode_ms) else None

        self.result = {
            "run": self.meta | {"run_id": self.run_id, "finished_at_utc": _now_utc()},
            "counts": {"tokens_in": tokens_in, "tokens_out": tokens_out},
            "timings": {
                "inference_e2e_ms": e2e_ms,
                "prefill_total_ms": prefill_total if prefill_total>0 else None,
                "first_decode_forward_ms": first_decode_ms,
                "first_token_latency_ms_approx": ftl_approx,
                "decode_stats": self._compute_decode_stats(decode_ms),
                "by_category_ms": sum_cat,
            },
            "throughput": {
                "prefill_toks_per_s": prefill_tps,
                "decode_toks_per_s": decode_tps,
            },
            "wsm": wsm_stats,
            "decode_step_ms": decode_ms,   # 完整序列
            "timeline": self.timeline,     # 方便溯源
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if path.lower().endswith(".json"):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.result, f, ensure_ascii=False, indent=2)
        elif path.lower().endswith(".csv"):
            rows = []
            for ev in self.timeline:
                r = {"kind":"span","name":ev["name"],"cat":ev["cat"],
                     "t_start_ms":ev["t_start_ms"],"t_end_ms":ev["t_end_ms"],"dur_ms":ev["dur_ms"]}
                rows.append(r)
            for i,dt in enumerate(self.result.get("decode_step_ms", [])):
                rows.append({"kind":"decode_step","name":f"decode_{i:04d}","cat":"inference","t_start_ms":"", "t_end_ms":"", "dur_ms":dt})
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["kind","name","cat","t_start_ms","t_end_ms","dur_ms"])
                w.writeheader(); w.writerows(rows)
        else:
            with open(path + ".json", "w", encoding="utf-8") as f:
                json.dump(self.result, f, ensure_ascii=False, indent=2)

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
            "KV_PINNED_BYTES":          8  << 30,
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
    KVCacheArgs.dram_limit_gb     = 32.0
    KVCacheArgs.dram_sizing_batch = 32
    KVCacheArgs.block_bytes       = 1 * 1024 * 1024
    KVCacheArgs.preallocate       = False
    KVCacheArgs.lazy_init         = True

    # 关闭 push 即时镜像，采用后移/聚合写（避免与权重 H2D 冲突）
    KVCacheArgs.mirror_on_push = False

    # I/O 节流与写速率配置（与权重 H2D 仲裁）
    KVCacheArgs.IO_RAW_THROTTLE_MS     = 25
    KVCacheArgs.NVME_WRITE_TARGET_MBPS = 1500

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
    """
    m = llama.model
    if hasattr(llama, "weight_streaming_manager"):
        wsm = llama.weight_streaming_manager
        ssd = bool(getattr(wsm, "ssd_enabled", False) or getattr(wsm, "ssd", None))
        cpu_warm = getattr(wsm, "disable_cpu_warm", None)
        mode = "ssd-streaming" if ssd else "cpu-gpu-streaming"
        print(f"[MODE] detected={mode}  (has WSM, ssd={ssd}, disable_cpu_warm={cpu_warm})")
        return mode
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

# ===== WSM wait_group_ready 包装（仅添加 Profiler 计时，不改逻辑） =====
def _wrap_wait_group_ready(original_method):
    """
    包装 WSM.wait_group_ready，添加 profiler 计时埋点。
    完全异步的等待逻辑已在 WSM 主类中实现。
    """
    def wrapped(self, layer_idx: int, group: str, compute_stream=None):
        with (PROFILER.span("wsm.wait_group_ready", "wsm", layer_idx=int(layer_idx), group=str(group))
              if (globals().get("PROFILER") is not None) else nullcontext()):
            # 调用 WSM 原生的完全异步 wait_group_ready
            return original_method(layer_idx, group, compute_stream)
    return wrapped


def _patched_ensure_module_on_gpu(self, m: torch.nn.Module, layer_idx: int | None = None, module_name: str | None = None):
    """
    扩展：把 **0-size CPU stub** 当作 meta 一样处理，优先从 CPU cache 取回并上卡。
    其它情况仍复用原先的 _ensure_param_on_gpu() 路径。
    """
    with (PROFILER.span("wsm.ensure_module_on_gpu", "wsm", layer_idx=(None if layer_idx is None else int(layer_idx)), module=str(module_name))
          if (globals().get("PROFILER") is not None) else nullcontext()):
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

# ---------- 辅助：固定规则生成 JSON/CSV 路径 ----------
def _sanitize_for_filename(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"[^A-Za-z0-9_.+-]", "-", s)

def build_output_paths(log_dir: Path, run_id: str, mode: str) -> tuple[Path, Path]:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = _sanitize_for_filename(RUN_TAG)
    stem = f"{ts}_{run_id}_{mode}" if not tag else f"{ts}_{tag}_{run_id}_{mode}"
    json_path = log_dir / f"{stem}.json"
    csv_path  = log_dir / f"{stem}.csv"
    return json_path, csv_path

# ---------- 运行主流程 ----------
def main():
    global PROFILER

    # 固定目录：自动创建
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 不再使用环境变量；run_id 可附带 RUN_TAG
    base_tag = RUN_TAG.strip() or None
    PROFILER = InferenceProfiler(run_name=base_tag)

    # 基础系统开销收敛
    os.environ.setdefault("OMP_NUM_THREADS",  "8")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    # PYTORCH_CUDA_ALLOC_CONF 已在顶部设置（必须在 import torch 前）

    # ============================================================
    # ⭐ 异步滑动窗口 - RTX 5080 (16GB) + 125GB RAM 实测优化
    # ============================================================
    # 硬件容量：GPU 12GB 可用 → 11 组 | RAM 110GB 可用 → 60 层
    # 策略：异步窗口 + 适度并发 + RAM 缓存优化
    # ============================================================

    # ⭐⭐⭐ P0 修复: 增加 warmup 层数，确保完整 overlap
    # 单层计算 100ms，可以 overlap 4 组 H2D (每组 25ms)
    # Warmup 至少需要覆盖: 初始层 + 预取深度 = 12 层
    GPU_AHEAD_LAYERS = 6# 预取 6 组（3 层）- 适配 11 组容量
    GPU_MAX_GROUPS   = 12
    GPU_WARMUP_LAYERS = 6# ⭐ 6 → 12 层（24 组），确保前 12 层完全 overlap
    CPU_CACHE_LAYERS = 40# CPU 缓存 50 层（79.5GB，安全余量）

    # === H2D 并发控制（⭐⭐⭐ P0 优化：PCIe Gen5 + RTX 5080 高带宽配置） ===
    # PCIe Gen5 x16 带宽: 64GB/s (Gen4的2倍)
    # 70B模型每组权重~1.5GB → Gen5单次H2D只需~25ms（Gen4的一半）
    # 更高带宽意味着需要更高并发度才能饱和PCIe，避免流水线空隙
    os.environ.setdefault("WSM_H2D_BASE_CONCURRENCY",  "8")   # ⭐ 5→16（Gen5高带宽，基础并发）
    os.environ.setdefault("WSM_H2D_PREFILL_MULT",      "3")  # Prefill: 32 并发
    os.environ.setdefault("WSM_H2D_DECODE_MULT",       "2")  # ⭐ 1.0→1.5（Decode: 24 并发）
    os.environ.setdefault("WSM_MAX_INFLIGHT_GROUPS",   "32")   # ⭐ 16→32（Inflight 上限，匹配并发）
    os.environ.setdefault("WSM_H2D_GROUP_BACKLOG_MAX", "96")   # ⭐ 48→96（H2D 队列，Gen5需要更深队列）

    # === 异步逐出机制 ===
    os.environ.setdefault("WSM_EVICT_QUEUE_SIZE",      "96")   # 逐出队列容量
    os.environ.setdefault("WSM_BG_WORKERS",            "8")    # 后台线程池

    # === GPU 窗口配置 ===
    os.environ.setdefault("WSM_GPU_MAX_GROUPS",        str(GPU_MAX_GROUPS))
    os.environ.setdefault("WSM_GPU_AHEAD_GROUPS",      str(GPU_AHEAD_LAYERS))
    # ⭐⭐⭐ P0 修复: 预取深度必须匹配计算时间窗口
    # 单层 100ms 可 overlap 4 组 H2D，设置 8 保证充足流水线
    os.environ.setdefault("WSM_GROUP_PREFETCH_DEPTH",  "6")  # ⭐ 6 → 8
    os.environ.setdefault("WSM_GPU_AHEAD",             str(GPU_AHEAD_LAYERS))
    os.environ.setdefault("WSM_GPU_BEHIND",            "2")    # 保留最近 2 层

    # === 预取策略 ===
    os.environ.setdefault("WSM_BALANCE_PREFETCH",      "1")
    os.environ.setdefault("WSM_PAIR_AHEAD",            "2")
    os.environ.setdefault("WSM_KIND_AHEAD_CAP",        "2")
    os.environ.setdefault("WSM_EVICT_FINISHED",        "1")    # 启用完成后逐出
    os.environ.setdefault("WSM_CPU_EVICT_AFTER_USE",   "0")    # 异步模式下禁用立即逐出

    # === 调试与监控 ===
    os.environ.setdefault("WSM_GRP_RETAIN_MS",         "0")
    os.environ.setdefault("WSM_SKIP_PRELOAD_WAIT",     "1")    # 启用异步预加载
    os.environ.setdefault("WSM_DEBUG_PREFETCH",        "1")    # 启用详细日志
    os.environ.setdefault("WSM_VERBOSE_MISMATCH",      "0")    # 生产环境关闭

    # === CPU 预取优化（RAM 可容纳 60 层） ===
    os.environ.setdefault("WSM_POOLED_CPU_READ",       "1")
    os.environ.setdefault("WSM_CPU_PF_WORKERS",        "12")   # CPU 预取线程数（50% CPU）
    os.environ.setdefault("WSM_REBALANCE_SYNC",        "0")    # 异步重平衡

    # === SSD→CPU 流水线 ===
    os.environ.setdefault("WSM_CPU_PREFETCH_DISTANCE", str(CPU_CACHE_LAYERS))   # CPU 预取 50 层
    os.environ.setdefault("WSM_SSD_CONCURRENCY",       "12")    # SSD 并发读取

    # === Prefill 特定优化 ===
    os.environ.setdefault("PREFILL_CPU_LAYERS",        str(CPU_CACHE_LAYERS))   # Prefill CPU 缓存 50 层
    os.environ.setdefault("PREFILL_GPU_LAYERS",        str(GPU_WARMUP_LAYERS))  # ⭐ 6 → 12 层
    os.environ.setdefault("PREFILL_PREFETCH_DISTANCE", "16")   # ⭐ 10 → 16（更远的预取距离）
    os.environ.setdefault("WSM_WARMUP_LAYERS_GPU",     str(GPU_WARMUP_LAYERS))  # ⭐ 6 → 12 层
    os.environ.setdefault("WSM_WRAPAROUND_WARMUP",     str(GPU_WARMUP_LAYERS))  # ⭐ 6 → 12 层
    
    
  

    # ============================================================
    # CPU 窗口额外配置（复用上面的 CPU_CACHE_LAYERS）
    # ============================================================
    os.environ.setdefault("WSM_CPU_RING_MODE",     "1")
    os.environ.setdefault("WSM_CPU_RING_OFFSET",   "0")
    os.environ.setdefault("WSM_CPU_CACHE_LAYERS",  str(CPU_CACHE_LAYERS))
    os.environ.setdefault("WSM_CPU_CACHE_CAP_LAYERS", str(CPU_CACHE_LAYERS))
    os.environ.setdefault("WSM_CPU_CACHE_HWM_LAYERS", str(CPU_CACHE_LAYERS))
    os.environ.setdefault("WSM_CPU_CACHE_LWM_LAYERS", str(max(2, CPU_CACHE_LAYERS - 5)))
    os.environ.setdefault("WSM_CPU_BACK_MARGIN",   "1")
    os.environ.setdefault("WSM_KV_THROTTLE_THRESHOLD", "2")
    os.environ.setdefault("WSM_KV_THROTTLE_MS",        "16")

    # 配置总结（仅打印）
    print("=" * 80)
    print("🚀 异步滑动窗口 - RTX 5080 (16GB) + 125GB RAM 优化配置")
    print("=" * 80)
    print(f"GPU 预取深度:  {GPU_AHEAD_LAYERS} 组")
    print(f"GPU 组预算:    {GPU_MAX_GROUPS} 组 (最多 ~9GB)")
    print(f"CPU 缓存容量:  {CPU_CACHE_LAYERS} 层 (~79.5GB)")
    print(f"H2D 并发度:    Prefill 24 | Decode 16")
    print(f"异步逐出队列:  64 任务")
    print(f"后台线程池:    6 workers")
    print(f"CPU 预取线程:  10 workers")
    print("=" * 80)
    print("✅ 异步窗口特性: 逐出/预取/CPU推进 全部在后台线程执行")
    print("✅ 主线程窗口滑动延迟: <1ms (vs 同步模式 ~20ms)")
    print("=" * 80)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 1) 覆盖 pinned/注册池 + KV 池
    with PROFILER.span("apply_runtime_overrides", "setup"):
        apply_runtime_overrides()
    with PROFILER.span("configure_kv_pool", "setup"):
        configure_kv_pool()
    with PROFILER.span("probe_after_runtime_clamp", "non_inference"):
        probe("after runtime clamp")

    # 2) WSM（SSD 流式）构造参数
    PRIME_WINDOW = int(os.getenv("WSM_PRIME_WINDOW", "12"))  # 从环境变量读取，默认6
    mode_config = {
        "raw_device": RAW_DEV,
        "ssd_manifest_path": MANIFEST,
        "max_cached_layers": CPU_CACHE_LAYERS,         # ✅ 修复: 必须与 CPU_CAP_VALUE 一致
        "cpu_cache_layers": CPU_CACHE_LAYERS,          # CPU 环形容量
        "warmup_layers": max(PRIME_WINDOW, GPU_AHEAD_LAYERS + 2),  # ✅ Fix: 至少预热 GPU_AHEAD + 2 层
        "staging_mb": 64,
        "verbose": True,
    }

    # 3) 构建（meta + SSD 流式）
    with PROFILER.span("probe_before_build", "non_inference"):
        probe("before LLaMA.build")
    with PROFILER.span("LLaMA.build", "setup"):
        llama = LLaMA.build(
            checkpoints_dir=CKPT_DIR,
            load_model=False,           # 不把 checkpoint 全载入 CPU
            device=device,
            max_seq_len=4096,
            max_batch_size=32,
            topk_blk=8,
            mode="mixed",
            mode_config=mode_config
        )
        s = stream_mnt.get_streams("cuda:0")
        print("h2d_mha:", s.weight_h2d_mha)
        print("h2d_ffn:", s.weight_h2d_ffn)
        print("cmp_mha:", s.compute_mha)
        print("cmp_ffn:", s.compute_ffn)
        print("kv_h2d:", s.kv_h2d, "kv_d2h:", s.kv_d2h)
    with PROFILER.span("probe_after_build", "non_inference"):
        probe("after LLaMA.build")

    # 绑定 WSM 补丁：仅为 profiler 计时，wait_group_ready 的异步逻辑已在 WSM 主类实现
    wsm = getattr(llama, "weight_streaming_manager", None)
    if wsm is not None:
        # 包装 wait_group_ready 以添加 profiler 计时
        original_wait = wsm.wait_group_ready
        wsm.wait_group_ready = types.MethodType(_wrap_wait_group_ready(original_wait), wsm)

        # 保留 ensure_module_on_gpu 的 CPU stub loader 补丁
        wsm._ensure_module_on_gpu = types.MethodType(_patched_ensure_module_on_gpu, wsm)
        print("[WSM PATCH] Profiler wrapper + CPU stub loader enabled")

        # ⭐⭐⭐ P0 优化：GPU窗口预热（避免冷启动，前N层并行H2D）
        with PROFILER.span("gpu_window_warmup", "setup"):
            warmup_layers = GPU_WARMUP_LAYERS  # ⭐ 使用配置的 12 层
            print(f"[WSM WARMUP] Preloading first {warmup_layers} layers to GPU...")
            for layer_idx in range(min(warmup_layers, wsm.n_layers)):
                try:
                    # 异步预取attn和ffn组（不阻塞，让H2D在后台并行）
                    wsm.prefetch_group_async(layer_idx, "attn", reason="warmup")
                    wsm.prefetch_group_async(layer_idx, "ffn", reason="warmup")
                except Exception as e:
                    print(f"[WSM WARMUP] Layer {layer_idx} prefetch failed: {e}")
            print(f"[WSM WARMUP] Warmup requests sent (async), first {warmup_layers} layers (24 groups) will be ready before inference")

            # ⭐⭐⭐ 额外修复：等待 warmup 完成后，继续预取后续层建立流水线
            # 在推理开始前，预取 L12-L20，确保 L12+ 也能 overlap
            # print(f"[WSM WARMUP] Extending prefetch pipeline to L{warmup_layers + 8}...")
            # for layer_idx in range(warmup_layers, min(warmup_layers + 8, wsm.n_layers)):
            #     try:
            #         wsm.prefetch_group_async(layer_idx, "attn", reason="warmup_extend")
            #         # 不预取 ffn，节省并发槽位
            #     except Exception as e:
            #         pass
            # print(f"[WSM WARMUP] Extended pipeline ready")

    PROFILER.wrap_model_forward(llama.model)

    # 4) 读取 prompt + 安全裁剪（max_gen_len=32）
    batch_size = 1
    max_gen_len = 32

    with PROFILER.span("read_prompt_file", "prompt"):
        try:
            prompt_path = PROMPT_TXT
            file_content = prompt_path.read_text(encoding="utf-8").strip()

            # 解析多个prompts（按 "===== PROMPT XXXX =====" 分隔）
            import re
            prompt_blocks = re.split(r'=====\s*PROMPT\s+\d+\s+.*?=====\s*\n', file_content)
            # 过滤空字符串
            prompt_blocks = [p.strip() for p in prompt_blocks if p.strip()]

            # 取前batch_size个prompts
            prompts = prompt_blocks[:batch_size]
            if len(prompts) < batch_size:
                # 如果prompts不足，重复最后一个prompt来填充
                print(f"Warning: Only {len(prompts)} prompts found, padding to {batch_size}")
                while len(prompts) < batch_size:
                    prompts.append(prompts[-1])

            print(f"Loaded {len(prompts)} prompts for batch_size={batch_size}")

        except Exception as e:
            raise RuntimeError(f"无法读取 {prompt_path}: {e}")

    with PROFILER.span("tokenize_and_clip", "prompt"):
        max_prompt_tokens = llama.args.max_seq_len - max_gen_len

        # 对每个prompt进行tokenize和裁剪
        clipped_prompts = []
        for prompt in prompts:
            tok = llama.tokenizer.encode(prompt, add_special_tokens=False)
            if len(tok) > max_prompt_tokens:
                tok = tok[-max_prompt_tokens:]
                prompt = llama.tokenizer.decode(tok)
            clipped_prompts.append(prompt)

        prompts = clipped_prompts
        # 使用第一个prompt的token数作为统计（假设所有prompt长度相似）
        tokens_in_count = len(llama.tokenizer.encode(prompts[0], add_special_tokens=False))

    # 5) 真正推理（decode）
    with PROFILER.span("probe_before_infer", "non_inference"):
        probe("before inference (decode)")
    with PROFILER.inference_scope():  # 端到端推理时间
        out_tokens, out_texts = llama.text_completion(
            prompts=prompts,
            temperature=0.0,
            max_gen_len=max_gen_len,
            batch_size=batch_size,
        )
    with PROFILER.span("probe_after_infer", "non_inference"):
        probe("after inference (decode)")

    # ==== 统计 tokens_out ====
    def _count_output_tokens(out_tokens_obj):
        try:
            if isinstance(out_tokens_obj, (list, tuple)):
                if len(out_tokens_obj) > 0 and isinstance(out_tokens_obj[0], (list, tuple)):
                    return len(out_tokens_obj[0])
                return len(out_tokens_obj)
            if torch.is_tensor(out_tokens_obj):
                return int(out_tokens_obj.numel())
        except Exception:
            pass
        return None

    tokens_out_count = _count_output_tokens(out_tokens)

    # ==== 汇总与保存 ====
    mode = classify_mode(llama)
    PROFILER.finalize(
        tokens_in=tokens_in_count,
        tokens_out=tokens_out_count,
        extra_meta={"llama_mode": mode, "device_str": str(device)}
    )

    # 自动生成 JSON/CSV 路径并各保存一次
    json_path, csv_path = build_output_paths(LOG_DIR, PROFILER.run_id, mode)
    PROFILER.save(str(json_path))
    PROFILER.save(str(csv_path))
    print(f"[Profiler] JSON: {json_path}")
    print(f"[Profiler] CSV : {csv_path}")

    # ==== 输出生成文本（不影响计时）====
    print(f"\n========== Generation (batch_size={batch_size}, len={max_gen_len}) ==========")
    # 只显示前3个和最后1个，避免输出太长
    for i in [0, 1, 2, batch_size-1]:
        if i < len(out_texts):
            print(f"\n--- Batch {i} ---")
            print(out_texts[i][:200] + "..." if len(out_texts[i]) > 200 else out_texts[i])
    print("=========================================")

if __name__ == "__main__":
    main()
