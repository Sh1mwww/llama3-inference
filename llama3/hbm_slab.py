# -*- coding: utf-8 -*-
"""
hbm_slab.py
Simple HBM slab pool for per-layer {MHA, FFN} residency.

Design goals:
- Allocate at most once per (layer, group); never cudaFree at eviction, just mark as free.
- Provide uint8 view for byte-wise copy, and keep dtype info for potential dequant ops.
- Keep API small; WSM decides how to lay out params within a slab.

NOTE:
- We don't forcibly rebind param storages here (no param.data reassignment). The slab serves as
  a stable target for byte copies if you choose to pack params into it. Alternatively, you can
  copy directly into each param.data without using the slab; the slab is here to avoid allocator
  churn if you need a contiguous staging region per layer.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import time
import torch

@dataclass
class Slab:
    layer_id: int
    group: str
    dtype: torch.dtype
    nbytes: int
    device: str
    tensor_u8: torch.Tensor      # uint8 contiguous buffer on CUDA
    in_use: bool = False
    created_ts: float = 0.0
    last_checkout_ts: float = 0.0

    def u8_view(self, offset: int, nbytes: int) -> torch.Tensor:
        assert 0 <= offset < self.nbytes and nbytes >= 0 and (offset + nbytes) <= self.nbytes
        return self.tensor_u8.narrow(0, offset, nbytes)

class HBMSlabPool:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.slabs: Dict[Tuple[int, str], Slab] = {}  # key=(layer_id, group)

    def reserve(self, layer_id: int, group: str, nbytes: int,
                dtype: torch.dtype = torch.float16) -> Slab:
        """
        Ensure a slab of at least nbytes exists for (layer, group). If existing slab is smaller, grow (realloc once).
        """
        key = (int(layer_id), str(group))
        want = int(nbytes)
        s = self.slabs.get(key)
        if s is None:
            with torch.cuda.device(self.device):
                t = torch.empty(want, dtype=torch.uint8, device=self.device)
            s = Slab(layer_id=layer_id, group=group, dtype=dtype, nbytes=want,
                     device=self.device, tensor_u8=t, in_use=False,
                     created_ts=time.time())
            self.slabs[key] = s
            return s
        # grow if too small (one-time realloc)
        if want > s.nbytes:
            with torch.cuda.device(self.device):
                t = torch.empty(want, dtype=torch.uint8, device=self.device)
            s.tensor_u8 = t
            s.nbytes = want
            s.dtype = dtype
        return s

    def checkout(self, layer_id: int, group: str) -> Slab:
        key = (int(layer_id), str(group))
        s = self.slabs.get(key)
        if s is None:
            raise KeyError(f"Slab not reserved for {(layer_id, group)}")
        if s.in_use:
            #allow re-entrancy or raise
            # For safety, we allow re-entrant checkout but emit a warning.
            print(f"[HBM][WARN] slab {(layer_id, group)} already in use; re-entrant checkout.")
        s.in_use = True
        s.last_checkout_ts = time.time()
        return s

    def checkin(self, layer_id: int, group: str):
        key = (int(layer_id), str(group))
        s = self.slabs.get(key)
        if s is None:
            return
        s.in_use = False

    def free_all(self):
        """Free all slabs (explicit cudaFree). Normally not used in runtime; for teardown/tests."""
        for s in self.slabs.values():
            # Let tensors go out of scope and rely on PyTorch to free
            s.tensor_u8 = None
        self.slabs.clear()

    def stats(self):
        out = {}
        for k, s in self.slabs.items():
            out[k] = {"nbytes": s.nbytes, "in_use": s.in_use, "dtype": str(s.dtype),
                      "created": s.created_ts, "last_checkout": s.last_checkout_ts}
        return out
