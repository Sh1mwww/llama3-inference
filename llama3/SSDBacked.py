# raw_block_kv.py
import os, numpy as np, torch, threading
from concurrent.futures import ThreadPoolExecutor

ALIGN = 4096           
def aligned_array(shape, dtype, align=ALIGN):
    """返回对齐的 numpy 数组视图(O_DIRECT 需要"""
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    buf = np.empty(nbytes + align, dtype=np.uint8)        # over-allocate
    offset = (-buf.ctypes.data) % align
    arr = buf[offset:offset+nbytes].view(dtype)
    return arr.reshape(shape)

class RawBlockKVBackend:
    def __init__(self, dev_path: str, n_layers: int,
                 blk_bytes: int, blk_per_layer: int,
                 max_concurrent_io: int = 4):
        self.fd = None
        try:
            self.fd = os.open(dev_path, os.O_RDONLY | os.O_DIRECT)
        except FileNotFoundError:
            print(f"[ERROR] Device not found: {dev_path}")
            raise

        self.blk_bytes = blk_bytes
        self.blk_pl   = blk_per_layer
        self.n_layers = n_layers
        self.pool = ThreadPoolExecutor(max_workers=max_concurrent_io,
                                       thread_name_prefix="rbd_io")

    # ---------- helpers ----------
    def _offset(self, layer, blk):
        return (layer * self.blk_pl + blk) * self.blk_bytes

    # ---------- write ----------
    def write_async(self, layer:int, blk:int, t_cpu:torch.Tensor):
        """异步写单个 block"""
        def _task():
            data = t_cpu.cpu().numpy().flatten().view(np.uint8)
            assert data.nbytes == self.blk_bytes
            buf = aligned_array(data.shape, np.uint8)      # page-aligned copy
            np.copyto(buf, data)
            os.pwrite(self.fd, buf, self._offset(layer, blk))
        return self.pool.submit(_task)

    # ---------- batch read ----------
    def batch_read(self, layer:int, blocks:list[int],
                   callback, cb_arg):
        def _task():
            out = {}
            for blk in blocks:
                buf = aligned_array((self.blk_bytes,), np.uint8)
                os.pread(self.fd, buf, self._offset(layer, blk))
                t = torch.from_numpy(buf.copy()).view(torch.float16)
                out[blk] = t
            if out:
                callback(cb_arg, out)
        return self.pool.submit(_task)

    # ---------- sync read to GPU tensor ----------
    def read(self, layer:int, blk:int, t_gpu:torch.Tensor):
        buf = aligned_array((self.blk_bytes,), np.uint8)
        os.pread(self.fd, buf, self._offset(layer, blk))
        h_tmp = torch.from_numpy(buf.copy()).view(torch.float16)
        t_gpu.copy_(h_tmp.reshape(t_gpu.shape), non_blocking=True)

    def __del__(self):
        try:
            self.pool.shutdown(wait=False)
        finally:
            os.close(self.fd)
