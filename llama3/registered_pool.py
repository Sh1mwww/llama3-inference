# -*- coding: utf-8 -*-
"""
registered_pool.py
A small pool of pre-pinned host buffers used as a "conveyor belt" for DRAM->H2D copies.

- Fixed number of buffers (N), each BUF_BYTES (default 32 MiB).
- get(nbytes) returns a view into a buffer (<= BUF_BYTES).
- put(handle) returns it immediately.
- defer_put(handle, cuda_event) delays the return until the event completes.
- gc(limit) polls a subset of pending events to reclaim completed ones.

Thread-safe, re-entrant, minimal dependencies (torch).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
import threading
import time
import torch

@dataclass
class BufferHandle:
    buf_id: int
    tensor: torch.Tensor     # pinned uint8 base tensor
    view: torch.Tensor       # pinned uint8 view with requested length (<= base size)
    nbytes: int

class RegisteredPool:
    def __init__(self, n_buffers: int = 12, buf_bytes: int = 32<<20, name: str = "regpool"):
        assert n_buffers > 0 and buf_bytes > 0
        self.name = name
        self.buf_bytes = int(buf_bytes)
        self._lock = threading.Lock()
        self._free: List[int] = list(range(n_buffers))  # stack of buffer indices
        self._buffers: List[torch.Tensor] = [
            torch.empty(self.buf_bytes, dtype=torch.uint8, pin_memory=True) for _ in range(n_buffers)
        ]
        self._pending: List[Tuple[torch.cuda.Event, BufferHandle]] = []  # to be reclaimed via gc()

    def get(self, nbytes: int, block: bool = True, timeout_s: Optional[float] = None) -> BufferHandle:
        """
        Acquire a buffer (or view) of at most buf_bytes. If block=False and none available, raises RuntimeError.
        """
        if nbytes <= 0:
            raise ValueError("nbytes must be > 0")
        if nbytes > self.buf_bytes:
            raise ValueError(f"request {nbytes} > buffer capacity {self.buf_bytes}")

        t0 = time.time()
        while True:
            with self._lock:
                if self._free:
                    idx = self._free.pop()
                    base = self._buffers[idx]
                    view = base.narrow(0, 0, nbytes)
                    return BufferHandle(buf_id=idx, tensor=base, view=view, nbytes=nbytes)
            if not block:
                raise RuntimeError("no free registered buffers")
            if timeout_s is not None and (time.time() - t0) > timeout_s:
                raise TimeoutError("get() timeout")
            time.sleep(0.001)

    def put(self, h: BufferHandle):
        """Return a buffer immediately."""
        with self._lock:
            self._free.append(h.buf_id)

    def defer_put(self, h: BufferHandle, event: Optional[torch.cuda.Event]):
        """
        Defer the return until the CUDA event is completed.
        The caller should periodically call gc() to reclaim finished events.
        """
        if event is None or event.query():
            self.put(h)
            return
        with self._lock:
            self._pending.append((event, h))

    def gc(self, limit: int = 64) -> int:
        """Poll up to 'limit' pending events; return number of reclaimed buffers."""
        n = 0
        with self._lock:
            if not self._pending:
                return 0
            remain: List[Tuple[torch.cuda.Event, BufferHandle]] = []
            for ev, h in self._pending[:limit]:
                if ev.query():
                    self._free.append(h.buf_id)
                    n += 1
                else:
                    remain.append((ev, h))
            remain.extend(self._pending[limit:])
            self._pending = remain
        return n

    def stats(self):
        with self._lock:
            return {
                "name": self.name,
                "buf_bytes": self.buf_bytes,
                "total": len(self._buffers),
                "free": len(self._free),
                "pending": len(self._pending),
            }
