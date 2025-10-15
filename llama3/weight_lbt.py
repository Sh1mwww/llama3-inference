# -*- coding: utf-8 -*-
"""
weight_lbt.py
Layer-Byte-Table (LBT) for weights: build per-layer {MHA, FFN} copy plans from a manifest,
coalesce adjacent file segments, and expose destination byte-slices on GPU params.

Usage (typical):
    from llama3.raw_param_store import ParamStore
    from llama3.weight_lbt import build_layer_copy_plan, classify_group

    store = ParamStore("/path/to/runtime_manifest.json")
    # param_dict: {param_name: torch.nn.Parameter or torch.Tensor (CUDA, already allocated)}
    plan = build_layer_copy_plan(store, param_dict, layer_id=5, group="mha", only_stream=True)

    # Iterate copy entries:
    for ce in plan.iter_copy_entries():
        # WSM 将在恰当的 CUDA stream 上把从 Host pinned 读到的 bytes 拷到 ce.dst_u8_slice
        # pinned_src = ... (由 raw_param_store 读入 Host pinned 的对应 bytes)
        # ce.dst_u8_slice.copy_(pinned_src, non_blocking=True)
        pass

Notes:
- We do NOT perform dtype conversion here. If dtype_from != dtype_to (e.g., disk INT8 -> HBM FP16),
  you should call your dequantization kernel instead of raw byte copy.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Tuple
import re
import torch

# -----------------------------
# Helpers for group classification
# -----------------------------

_MHA_PAT = re.compile(r"\.attention\.|\.attn\.", re.IGNORECASE)
_FFN_PAT = re.compile(r"\.mlp\.|\.feed_forward\.|\.ffn\.", re.IGNORECASE)

def classify_group(param_name: str) -> str:
    """Heuristic: map param name to 'mha' or 'ffn' (fallback 'other')."""
    if _MHA_PAT.search(param_name):
        return "mha"
    if _FFN_PAT.search(param_name):
        return "ffn"
    return "other"

# -----------------------------
# Core datatypes
# -----------------------------

@dataclass
class FileSegment:
    file_offset: int
    nbytes: int
    dtype_from: str  # disk dtype str, e.g. "fp16", "bf16", "int8"
    # destination side
    dst_offset: int  # byte offset within the destination param
    dtype_to: str    # param.dtype on HBM

@dataclass
class ParamCopyPlan:
    layer_id: int
    group: str              # "mha" / "ffn" / "other"
    name: str               # full param name
    dtype_to: str
    total_bytes: int
    segments: List[FileSegment] = field(default_factory=list)

    def coalesce(self) -> None:
        """Coalesce adjacent file segments with the same dtype and contiguous dst/file ranges."""
        if not self.segments:
            return
        segs = sorted(self.segments, key=lambda s: (s.file_offset, s.dst_offset))
        merged: List[FileSegment] = []
        cur = segs[0]
        for nxt in segs[1:]:
            contiguous_file = (cur.file_offset + cur.nbytes == nxt.file_offset)
            contiguous_dst  = (cur.dst_offset  + cur.nbytes == nxt.dst_offset)
            same_dtype = (cur.dtype_from == nxt.dtype_from) and (cur.dtype_to == nxt.dtype_to)
            if contiguous_file and contiguous_dst and same_dtype:
                cur = FileSegment(
                    file_offset=cur.file_offset,
                    nbytes=cur.nbytes + nxt.nbytes,
                    dtype_from=cur.dtype_from,
                    dst_offset=cur.dst_offset,
                    dtype_to=cur.dtype_to,
                )
            else:
                merged.append(cur); cur = nxt
        merged.append(cur)
        self.segments = merged

    def iter_dst_slices(self, param_tensor: torch.Tensor) -> Iterable[Tuple[FileSegment, torch.Tensor]]:
        """
        Yield tuples (segment, dst_u8_slice) for copy.
        The dst_u8_slice is param.data.view(torch.uint8)[dst_offset : dst_offset+nbytes].
        """
        assert param_tensor.is_cuda, f"Param {self.name} must be on CUDA"
        dst_u8 = param_tensor.data.view(torch.uint8)
        for seg in self.segments:
            yield seg, dst_u8.narrow(0, seg.dst_offset, seg.nbytes)

@dataclass
class LayerCopyPlan:
    layer_id: int
    group: str
    params: Dict[str, ParamCopyPlan] = field(default_factory=dict)

    def total_bytes(self) -> int:
        return sum(p.total_bytes for p in self.params.values())

    def iter_copy_entries(self) -> Iterable[Tuple[str, FileSegment, torch.Tensor]]:
        """
        Iterate all segments for this layer/group, yielding:
        (param_name, FileSegment, dst_u8_slice)
        """
        for name, p in self.params.items():
            # param tensor must be stored somewhere accessible to this plan.
            # We store a reference to the dst tensor via an attribute injected at build time.
            dst_t = getattr(p, "_dst_tensor", None)
            if dst_t is None:
                raise RuntimeError(f"Param tensor missing for {name}.")
            for seg, dst_slice in p.iter_dst_slices(dst_t):
                yield name, seg, dst_slice

# -----------------------------
# Building from ParamStore + param tensors
# -----------------------------

def _dtype_str(t: torch.Tensor) -> str:
    # Keep simple readable dtype
    if t.dtype == torch.float16: return "fp16"
    if t.dtype == torch.bfloat16: return "bf16"
    if t.dtype == torch.float32: return "fp32"
    if t.dtype == torch.int8: return "int8"
    if t.dtype == torch.int32: return "int32"
    return str(t.dtype).replace("torch.", "")

def build_layer_copy_plan(store,    # ParamStore
                          param_tensors: Dict[str, torch.Tensor],  # CUDA tensors by full name
                          layer_id: int,
                          group: str,
                          only_stream: bool = True,
                          name_filter: Optional[re.Pattern] = None) -> LayerCopyPlan:
    """
    Build a LayerCopyPlan by joining manifest segments and the actual CUDA params.
    - coalesce adjacent pieces per-param
    - attach destination tensors (as refs) into the plan for direct dst slicing

    Args:
        store: ParamStore instance
        param_tensors: {name: CUDA tensor}
        layer_id: layer index
        group: "mha" / "ffn" / "other"
        only_stream: whether to restrict to streamable params (recommended)
        name_filter: optional regex to further filter names
    """
#     groups = store._groups_for_layer(layer_id, only_stream=only_stream, names=None)  # {name: [segments...]}

#     plan = LayerCopyPlan(layer_id=layer_id, group=group)
#     for name, segs in groups.items():
#         if classify_group(name) != group:
#             continue
#         if name_filter and not name_filter.search(name):
#             continue
#         if name not in param_tensors:
#             # skip missing params (caller may pass the subset needed)
#             continue

#         dst = param_tensors[name]
#         dtype_to = _dtype_str(dst)
#         # bytes must equal param.numel()*element_size
#         need_bytes = int(dst.numel() * dst.element_size())
#         # manifest segments may be many; we compute per-seg dst_offset by accumulation
#         cursor = 0
#         pcs = ParamCopyPlan(layer_id=layer_id, group=group, name=name,
#                             dtype_to=dtype_to, total_bytes=need_bytes)
#         for seg in segs:
#             off = int(seg["offset"])
#             nby = int(seg["nbytes"])
#             dtype_from = seg.get("dtype", dtype_to)
#             pcs.segments.append(FileSegment(
#                 file_offset=off, nbytes=nby, dtype_from=dtype_from,
#                 dst_offset=cursor, dtype_to=dtype_to
#             ))
#             cursor += nby

#         # coalesce
#         pcs.coalesce()

#         # attach dst tensor
#         setattr(pcs, "_dst_tensor", dst)
#         plan.params[name] = pcs

#         # 审计：目标总字节数与累积 nbytes 是否一致（不一致则提示但不强退）
#         seg_sum = sum(s.nbytes for s in pcs.segments)
#         if seg_sum != need_bytes:
#             print(f"[LBT][WARN] {name}: dst_bytes={need_bytes}, seg_sum={seg_sum}. "
#                   f"可能存在 padding/量化或权重拆分特殊情况；请在 WSM 的 copy 回调处处理。")

#     return plan
# def build_layer_copy_plan(store,
#                           param_tensors: Dict[str, torch.Tensor],
#                           layer_id: int,
#                           group: str,
#                           only_stream: bool = True,
#                           name_filter: Optional[re.Pattern] = None) -> LayerCopyPlan:
    groups = store._groups_for_layer(layer_id, only_stream=only_stream, names=None)

    plan = LayerCopyPlan(layer_id=layer_id, group=group)
    for name, segs in groups.items():
        if classify_group(name) != group:
            continue
        if name_filter and not name_filter.search(name):
            continue
        if name not in param_tensors:
            continue

        dst_param = param_tensors[name]
        assert dst_param.is_cuda, f"{name} must be CUDA tensor"
        dtype_to = _dtype_str(dst_param)

        # 目标（param）字节数
        dst_bytes = int(dst_param.numel() * dst_param.element_size())
        # manifest 分片总字节
        seg_sum = sum(int(s["nbytes"]) for s in segs)

        # 构建 ParamCopyPlan（先按 seg_sum 作为“需要复制的总字节数”）
        pcs = ParamCopyPlan(layer_id=layer_id, group=group, name=name,
                            dtype_to=dtype_to, total_bytes=seg_sum)

        # 计算每段的 dst_offset：按 seg 顺序紧密累加
        cursor = 0
        for seg in segs:
            off = int(seg["offset"]); nby = int(seg["nbytes"])
            dtype_from = seg.get("dtype", dtype_to)
            pcs.segments.append(FileSegment(
                file_offset=off, nbytes=nby, dtype_from=dtype_from,
                dst_offset=cursor, dtype_to=dtype_to
            ))
            cursor += nby

        pcs.coalesce()

        # [NEW] 目的地选择：尺寸一致 → 直接写入 param；不一致 → 分配“外部目的地”
        use_external = (seg_sum != dst_bytes)
        if use_external:
            # 兼容回退：分配一个外部 uint8 缓冲（在与 param 相同的 device 上）
            ext_u8 = torch.empty(seg_sum, dtype=torch.uint8, device=dst_param.device)
            setattr(pcs, "_dst_tensor", ext_u8)             # WSM 直接拷到这里
            setattr(pcs, "_dst_is_external", True)          # 标记外部
            setattr(pcs, "_external_hint_param", dst_param) # 后续解码/重排时可参考
            print(f"[LBT][INFO] {name}: seg_sum({seg_sum}) != param_bytes({dst_bytes}); "
                  f"using external dst buffer on {dst_param.device}.")
        else:
            setattr(pcs, "_dst_tensor", dst_param)          # 正常：拷到 param 本身
            setattr(pcs, "_dst_is_external", False)
            setattr(pcs, "_external_hint_param", None)

        plan.params[name] = pcs

    return plan
