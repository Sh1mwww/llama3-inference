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

import time
from dataclasses import dataclass, field

# ===================== 性能计时器（GPU/IO/Compute/阶段切分） =====================
class _PerfRecorder:
    """
    - 记录：
        * Host 总体推理耗时（wall）
        * GPU H2D 传输区间（基于 WSM 的非阻塞 copy 事件）
        * GPU 计算区间（按每层 forward 包裹 CUDA Event）
        * Host 侧 I/O 等待时间（wait_event/synchronize 的阻塞时间）
    - 给出：
        * 计算总时长、I/O 总时长
        * I/O 与计算的交叠时长（同一 GPU 时间线）
        * prefill / decode 的粗粒度时间（GPU 计算 + wall 估计）
    """
    def __init__(self, device: str | torch.device):
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        self.cuda = (dev.type == "cuda" and torch.cuda.is_available())
        self.dev  = dev
        self.reset()

    def reset(self):
        # Host 计时
        self.host_infer_start = None
        self.host_infer_end   = None
        self.host_io_wait_ms  = 0.0

        # GPU 事件
        self.t0_evt = None
        self.h2d_evt_pairs = []       # [(start_evt, end_evt, meta_dict)]
        self.compute_evt_pairs = []   # [(start_evt, end_evt, meta_dict{layer, phase, seq_len})]

        # 估计阶段分界
        self.prefill_end_ms   = None
        self.decode_start_ms  = None

        # 统计缓存
        self._final = None

    # ---------- Host 计时 ----------
    def host_start(self):
        self.host_infer_start = time.perf_counter()

    def host_end(self):
        self.host_infer_end = time.perf_counter()

    def add_host_io_wait(self, ms: float):
        self.host_io_wait_ms += float(ms)

    # ---------- GPU 计时 ----------
    def record_t0(self):
        if not self.cuda: return
        self.t0_evt = torch.cuda.Event(enable_timing=True)
        self.t0_evt.record(torch.cuda.current_stream(self.dev))

    def record_h2d_pair(self, start_evt, end_evt, meta=None):
        if not self.cuda: return
        self.h2d_evt_pairs.append((start_evt, end_evt, meta or {}))
        # Debug: print first few recordings
        if len(self.h2d_evt_pairs) <= 3:
            print(f"[PERF DEBUG] H2D event recorded: {len(self.h2d_evt_pairs)}, meta={meta}")

    def record_compute_pair(self, start_evt, end_evt, meta=None):
        if not self.cuda: return
        self.compute_evt_pairs.append((start_evt, end_evt, meta or {}))
        # Debug: print first few recordings
        if len(self.compute_evt_pairs) <= 3:
            print(f"[PERF DEBUG] Compute event recorded: {len(self.compute_evt_pairs)}, meta={meta}")

    # ---------- 工具：区间并集与交集 ----------
    @staticmethod
    def _merge(intervals):
        if not intervals: return []
        intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
        merged = []
        cs, ce = intervals[0]
        for s,e in intervals[1:]:
            if s <= ce:
                ce = max(ce, e)
            else:
                merged.append((cs, ce))
                cs, ce = s, e
        merged.append((cs, ce))
        return merged

    @staticmethod
    def _total_length(intervals):
        return sum(max(0.0, e - s) for s,e in intervals)

    @staticmethod
    def _intersect(a, b):
        i, j = 0, 0
        res = []
        while i < len(a) and j < len(b):
            s = max(a[i][0], b[j][0])
            e = min(a[i][1], b[j][1])
            if s < e:
                res.append((s,e))
            if a[i][1] < b[j][1]:
                i += 1
            else:
                j += 1
        return res

    # ---------- 完成统计 ----------
    def finalize(self):
        if self._final is not None:
            return self._final

        print(f"[PERF DEBUG] Finalizing... cuda={self.cuda}, h2d_pairs={len(self.h2d_evt_pairs)}, compute_pairs={len(self.compute_evt_pairs)}")

        if self.cuda:
            torch.cuda.synchronize(self.dev)

        # Host wall
        wall_ms = None
        if self.host_infer_start is not None and self.host_infer_end is not None:
            wall_ms = (self.host_infer_end - self.host_infer_start) * 1000.0

        # 若无 CUDA，返回最基础统计
        if not self.cuda or self.t0_evt is None:
            self._final = {
                "host_wall_ms": wall_ms,
                "host_io_wait_ms": self.host_io_wait_ms,
                "gpu_compute_ms": None,
                "gpu_io_h2d_ms": None,
                "gpu_overlap_io_compute_ms": None,
                "prefill": {"gpu_compute_ms": None, "wall_est_ms": None},
                "decode":  {"gpu_compute_ms": None, "wall_est_ms": None},
            }
            return self._final

        # 转换事件为以 t0 为原点的时间区间
        def _evt_to_ms_pair(pair):
            s_evt, e_evt = pair
            s = self.t0_evt.elapsed_time(s_evt)
            e = self.t0_evt.elapsed_time(e_evt)
            if e < s: e = s
            return (float(s), float(e))

        h2d_intervals = []
        for s_evt, e_evt, meta in self.h2d_evt_pairs:
            h2d_intervals.append(_evt_to_ms_pair((s_evt, e_evt)))

        compute_intervals = []
        prefill_intervals = []
        decode_intervals  = []
        for s_evt, e_evt, meta in self.compute_evt_pairs:
            s,e = _evt_to_ms_pair((s_evt, e_evt))
            compute_intervals.append((s,e))
            phase = meta.get("phase")
            if phase == "prefill":
                prefill_intervals.append((s,e))
            elif phase == "decode":
                decode_intervals.append((s,e))

        # 并集 & 长度
        h2d_union = self._merge(h2d_intervals)
        cmp_union = self._merge(compute_intervals)
        pre_union = self._merge(prefill_intervals) if prefill_intervals else []
        dec_union = self._merge(decode_intervals)  if decode_intervals else []

        h2d_total = self._total_length(h2d_union)
        cmp_total = self._total_length(cmp_union)
        pre_total = self._total_length(pre_union)
        dec_total = self._total_length(dec_union)

        # 交叠
        inter = self._intersect(h2d_union, cmp_union)
        inter_total = self._total_length(inter)

        # 阶段边界（wall 粗估）
        wall_prefill_ms = None
        wall_decode_ms  = None
        if pre_union:
            t0 = pre_union[0][0]
            te = pre_union[-1][1]
            wall_prefill_ms = te - t0
        if dec_union:
            t1 = dec_union[0][0]
            te = dec_union[-1][1]
            wall_decode_ms = te - t1

        self._final = {
            "host_wall_ms": wall_ms,
            "host_io_wait_ms": self.host_io_wait_ms,
            "gpu_compute_ms": cmp_total,
            "gpu_io_h2d_ms": h2d_total,
            "gpu_overlap_io_compute_ms": inter_total,
            "prefill": {"gpu_compute_ms": pre_total, "wall_est_ms": wall_prefill_ms},
            "decode":  {"gpu_compute_ms": dec_total, "wall_est_ms": wall_decode_ms},
        }
        return self._final

    def pretty_print(self, extra=None):
        R = self.finalize()
        print("\n==================== ⏱️ Inference Profiling Report ====================")
        def fmt(x):
            return f"{x:.2f} ms" if isinstance(x, (int,float)) and x is not None else str(x)
        print(f"Host Wall (overall): {fmt(R['host_wall_ms'])}")
        print(f"Host I/O wait:       {fmt(R['host_io_wait_ms'])}")
        print(f"GPU Compute total:   {fmt(R['gpu_compute_ms'])}")
        print(f"GPU H2D I/O total:   {fmt(R['gpu_io_h2d_ms'])}")
        print(f"GPU Overlap(IO∩Cmp): {fmt(R['gpu_overlap_io_compute_ms'])}")
        # 交叠占比
        if R['gpu_io_h2d_ms'] and R['gpu_compute_ms'] and R['gpu_compute_ms']>0:
            overlap_ratio = R['gpu_overlap_io_compute_ms']/min(R['gpu_compute_ms']+1e-9, R['gpu_io_h2d_ms']+1e-9)
            print(f"Overlap ratio (~ against min(comp,io)): {overlap_ratio*100:.1f}%")

        print("\n---- Phase breakdown ----")
        print(f"Prefill - GPU compute: {fmt(R['prefill']['gpu_compute_ms'])} | Wall(est): {fmt(R['prefill']['wall_est_ms'])}")
        print(f"Decoder - GPU compute: {fmt(R['decode']['gpu_compute_ms'])} | Wall(est): {fmt(R['decode']['wall_est_ms'])}")

        if isinstance(extra, dict):
            for k,v in extra.items():
                print(f"{k}: {v}")
        print("=======================================================================\n")


# 全局 recorder（在 main() 中初始化）
G_PERF: _PerfRecorder | None = None


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

    ⭐ 双保险：decoder 切换检测（从高层回到 layer 0 且还没 prime 过）
    """
    # ===== 双保险：decoder 切换检测 =====
    # 检测从"高层回到 0"且还没 prime 过 → 说明进入 decoder 阶段
    if layer_idx == 0 and not getattr(self, "_decoder_prime_done", False):
        last_layer = getattr(self, "_last_executed_layer", -1)
        if last_layer > 0 and hasattr(self, "_prime_decoder_window"):
            if getattr(self, "verbose", False):
                print(f"[WSM FAILSAFE] Detected decoder start (L{last_layer}→L0); priming now")
            try:
                self._prime_decoder_window(first_n=4)
                self._decoder_prime_done = True
            except Exception as e:
                if getattr(self, "verbose", False):
                    print(f"[WSM FAILSAFE] Failed to prime decoder: {e}")
    # ===============================================

    # 原始等待逻辑：调用原始的（未被 patch 的）wait_group_ready
    # 注意：_original_wait_group_ready 会在 patch 时保存
    return self._original_wait_group_ready(layer_idx, group, compute_stream=compute_stream)


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
                _h2d_stream = self._select_h2d_stream_for(module_name=module_name)
                with torch.cuda.stream(_h2d_stream):
                    _h2d_start = torch.cuda.Event(enable_timing=True)
                    _h2d_end   = torch.cuda.Event(enable_timing=True)
                    _h2d_start.record(_h2d_stream)
                    p_gpu = chosen_tensor.to(self.device, non_blocking=True)
                    _h2d_end.record(_h2d_stream)
                if G_PERF is not None:
                    G_PERF.record_h2d_pair(_h2d_start, _h2d_end, meta={"layer": int(layer_idx) if layer_idx is not None else None, "name": params_full_names.get(local_param_name, full_name)})
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
            _h2d_stream = self._select_h2d_stream_for(module_name=module_name)
            with torch.cuda.stream(_h2d_stream):
                _h2d_start = torch.cuda.Event(enable_timing=True)
                _h2d_end   = torch.cuda.Event(enable_timing=True)
                _h2d_start.record(_h2d_stream)
                b_gpu = b.detach().to(self.device, non_blocking=True)
                _h2d_end.record(_h2d_stream)
            if G_PERF is not None:
                G_PERF.record_h2d_pair(_h2d_start, _h2d_end, meta={"layer": int(layer_idx) if layer_idx is not None else None, "buffer": True})
            try:
                b.data = b_gpu
            except Exception:
                pass


# ---------- 层级 forward 包裹以测 GPU 计算时间，并推断 prefill/decoder ----------
def _guess_phase_from_args_kwargs(args, kwargs):
    # 依据常见签名：输入张量形状推断 seq_len，start_pos/cache_position 推断阶段
    seq_len = None
    # 尝试从 args 中找张量
    def _find_first_tensor(a):
        if isinstance(a, torch.Tensor):
            return a
        if isinstance(a, (tuple, list)) and a:
            for x in a:
                t = _find_first_tensor(x)
                if t is not None:
                    return t
        if isinstance(a, dict):
            for v in a.values():
                t = _find_first_tensor(v)
                if t is not None:
                    return t
        return None

    t = None
    for a in args:
        t = _find_first_tensor(a)
        if t is not None:
            break
    if t is None:
        for v in kwargs.values():
            t = _find_first_tensor(v)
            if t is not None:
                break

    if t is not None and hasattr(t, 'shape') and len(t.shape) >= 2:
        # 尝试 [B, T, C] 或 [T, B, C] 两种
        B,T = None,None
        s = list(t.shape)
        # 简单启发式：维度中较小的（通常 < 8）可能是 batch
        if len(s) >= 3:
            candidates = [(s[0], s[1]), (s[1], s[0])]
            # 选择 T 较大者作为 seq_len
            if candidates[0][0] <= 8:
                B,T = candidates[0]
            elif candidates[1][0] <= 8:
                B,T = candidates[1]
            else:
                # 无明显 batch，取 max 作为 T
                T = max(s[0], s[1])
        else:
            # 2D 张量，取较大维
            T = max(s[0], s[1])
        seq_len = T

    start_pos = kwargs.get("start_pos", None)
    cache_pos = kwargs.get("cache_position", None)
    position_ids = kwargs.get("position_ids", None)

    phase = None
    if seq_len is not None:
        if seq_len == 1:
            phase = "decode"
        elif seq_len > 1:
            phase = "prefill"

    # 根据 start/cache pos 微调
    for pos in (start_pos, cache_pos):
        try:
            if pos is not None:
                pos_val = int(pos) if isinstance(pos, (int, float)) else int(pos[0] if isinstance(pos, (list,tuple)) else pos.item())
                if pos_val == 0 and seq_len and seq_len > 1:
                    phase = "prefill"
                elif pos_val > 0 and seq_len == 1:
                    phase = "decode"
        except Exception:
            pass

    return phase, seq_len

def wrap_model_layers_for_timing(llama):
    if not torch.cuda.is_available():
        print("[TIMER] CUDA 不可用，跳过 GPU 计算计时包裹。")
        return

    model = getattr(llama, "model", llama)

    # 寻找可能的层列表属性
    layers = getattr(model, "layers", None)
    if layers is None and hasattr(model, "model"):
        layers = getattr(model.model, "layers", None)
    if layers is None:
        # 兜底：尝试按常见命名空间搜集子模块
        layers = [m for m in model.modules() if hasattr(m, "forward")]
        print(f"[TIMER] 未找到标准 layers 列表，fallback 包裹 {len(layers)} 个模块，可能较重。")
    else:
        print(f"[TIMER] 包裹 {len(layers)} 个 Transformer 层用于 GPU 计算计时。")

    def _wrap_one(layer, layer_idx):
        if hasattr(layer, "_orig_forward_for_timer"):
            return
        layer._orig_forward_for_timer = layer.forward
        def _timed_forward(*args, **kwargs):
            stream = torch.cuda.current_stream()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)
            start_evt.record(stream)
            out = layer._orig_forward_for_timer(*args, **kwargs)
            end_evt.record(stream)
            if G_PERF is not None:
                phase, seq_len = _guess_phase_from_args_kwargs(args, kwargs)
                G_PERF.record_compute_pair(start_evt, end_evt, meta={"layer": layer_idx, "phase": phase, "seq_len": seq_len})
            return out
        layer.forward = _timed_forward

    # 若是序列容器
    try:
        for i,layer in enumerate(layers):
            _wrap_one(layer, i)
    except Exception:
        # 尝试对所有子模块包裹（保守）
        for i,layer in enumerate(list(model.modules())):
            try:
                _wrap_one(layer, i)
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
    # GPU_MAX_GROUPS   = max(10, 2 + GPU_AHEAD_LAYERS * 2 + 1)  # ≈ 11：当前(2) + 预取(8) + 缓冲(1)
    GPU_MAX_GROUPS = 10

    os.environ.setdefault("WSM_GPU_MAX_GROUPS", str(GPU_MAX_GROUPS))
    os.environ.setdefault("WSM_GROUP_PREFETCH_DEPTH", str(GPU_AHEAD_LAYERS))
    os.environ.setdefault("WSM_BALANCE_PREFETCH", "1")
    os.environ.setdefault("WSM_PAIR_AHEAD", "2")      # 同层→i+1→i+2
    os.environ.setdefault("WSM_KIND_AHEAD_CAP", "2")  # 单一类型最大前瞻距离
    os.environ.setdefault("WSM_H2D_GROUP_BACKLOG_MAX", "12")

    # 计算结束立刻释放（避免组堆积）
    os.environ.setdefault("WSM_EVICT_FINISHED", "1")  # ← 修正为 1（你的草稿里误写成了 0）
    os.environ.setdefault("WSM_GRP_RETAIN_MS", "3")   # 极短保留窗口

    # 跳过预加载等待：边跑边滚动预取
    os.environ.setdefault("WSM_SKIP_PRELOAD_WAIT", "1")

    # ============================================================
    # ⭐ 环形 CPU 窗口（SSD -> pinned DRAM，80 层取模）
    # ============================================================
    CPU_CAP_VALUE    = 40   # i+4..i+4+cap 的 cap
    CPU_RING_OFFSET  = 4    # 窗口从 i+4 起
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
    print("🔧 组级 GPU 预取（ahead=4）+ 环形 CPU 窗口")
    print("=" * 80)
    print(f"GPU 预取距离: {GPU_AHEAD_LAYERS} 层")
    print(f"GPU 组预算:   {GPU_MAX_GROUPS} 组(attn/ffn)")
    print(f"CPU 窗口容量: {CPU_CAP_VALUE} 层 (环形，对 80 层取模)")
    print(f"CPU 环形偏移: i+{CPU_RING_OFFSET}")
    print("=" * 80)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 性能计时器初始化
    global G_PERF
    G_PERF = _PerfRecorder(device)
    print(f"[PERF] G_PERF initialized: {G_PERF}, cuda={G_PERF.cuda}")

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

    # ⭐ 将 G_PERF 注入到 llama3 模块中，让它们能够访问到全局 recorder
    import llama3.layers as layers_module
    import llama3.weight_streaming_manager as wsm_module
    layers_module.G_PERF = G_PERF
    wsm_module.G_PERF = G_PERF
    print(f"[PERF] Injected G_PERF into llama3 modules")

    # 绑定 WSM 补丁
    wsm = getattr(llama, "weight_streaming_manager", None)
    if wsm is not None:
        # ⭐ 关键：先保存原始方法，避免递归调用！
        wsm._original_wait_group_ready = wsm.wait_group_ready
        # 然后用 patch 版本替换
        wsm.wait_group_ready     = types.MethodType(_patched_wait_group_ready, wsm)
        wsm._ensure_module_on_gpu = types.MethodType(_patched_ensure_module_on_gpu, wsm)
        print("[WSM PATCH] strict group-ready + CPU stub loader enabled")

    # 包裹模型层，记录 GPU 计算事件
    try:
        wrap_model_layers_for_timing(llama)
    except Exception as e:
        print(f"[TIMER] 层包裹失败：{e}")

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
    if G_PERF is not None:
        G_PERF.host_start()
        if torch.cuda.is_available():
            G_PERF.record_t0()
    out_tokens, out_texts = llama.text_completion(
        prompts=[prompt],
        temperature=0.0,
        max_gen_len=max_gen_len,
        batch_size=1,
    )
    probe("after inference (decode)")
    if G_PERF is not None:
        G_PERF.host_end()
        # 汇总并打印报告
        extra = {}
        try:
            prompt_tok = len(tok)
        except Exception:
            prompt_tok = None
        try:
            gen_tok = len(out_tokens[0]) if isinstance(out_tokens, (list,tuple)) else None
        except Exception:
            gen_tok = None
        if prompt_tok is not None:
            extra['Prompt tokens'] = prompt_tok
        if gen_tok is not None:
            extra['Generated tokens'] = gen_tok
            if (G_PERF.host_infer_start is not None) and (G_PERF.host_infer_end is not None):
                ms = max(1e-9, (G_PERF.host_infer_end - G_PERF.host_infer_start)*1000.0)
                total_tok = (prompt_tok or 0) + (gen_tok or 0)
                extra['Throughput(est)'] = f"{total_tok / (ms/1000.0):.2f} tok/s"
        G_PERF.pretty_print(extra=extra)

    print(f"\n========== Generation (len={max_gen_len}) ==========")
    print(out_texts[0])
    print("=========================================")

if __name__ == "__main__":
    main()
