# SSDBacked.py
import os, numpy as np, torch, threading
from concurrent.futures import ThreadPoolExecutor

ALIGN = 4096
def aligned_array(shape, dtype, align=ALIGN):
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    buf = np.empty(nbytes + align, dtype=np.uint8)
    offset = (-buf.ctypes.data) % align
    arr = buf[offset:offset+nbytes].view(dtype)
    return arr.reshape(shape)

class RawBlockKVBackend:
    def __init__(self, dev_path: str, n_layers: int,
                 blk_bytes: int, blk_per_layer: int,
                 max_concurrent_io: int = 4):
        self.fd = None
        try:
            # 改为可写 + O_DIRECT
            self.fd = os.open(dev_path, os.O_RDWR | os.O_DIRECT)
        except FileNotFoundError:
            print(f"[ERROR] Device not found: {dev_path}")
            raise

        self.blk_bytes = blk_bytes             # 逻辑有效字节（整 token）
        self.stride    = ((blk_bytes + ALIGN - 1) // ALIGN) * ALIGN   # 物理步距
        self.blk_pl    = blk_per_layer         # 每层的槽位数量（这里= max_seq_len）
        self.n_layers  = n_layers
        self.pool = ThreadPoolExecutor(max_workers=max_concurrent_io,
                                       thread_name_prefix="rbd_io")

    def _offset(self, layer, slot):
        # slot 可以是 token_idx
        return (layer * self.blk_pl + slot) * self.stride

    # ---------- sync write ----------
    def write(self, layer:int, slot:int, t_cpu:torch.Tensor, sync:bool=False):
        """写入单个“token 单元”到 SSD（整 token 封包）"""
        data_u8 = t_cpu.detach().cpu().contiguous().view(torch.uint8).numpy()
        assert data_u8.nbytes == self.blk_bytes, f"token pack mismatch: {data_u8.nbytes} != {self.blk_bytes}"

        # 4K 对齐缓冲
        buf = aligned_array((self.stride,), np.uint8)
        # 写有效负载，尾部补零
        np.copyto(buf[:self.blk_bytes], data_u8)
        if self.stride > self.blk_bytes:
            buf[self.blk_bytes:] = 0

        os.pwrite(self.fd, buf, self._offset(layer, slot))
        if sync:
            os.fsync(self.fd)

    # ---------- async write ----------
    def write_async(self, layer:int, slot:int, t_cpu:torch.Tensor, sync:bool=False):
        def _task():
            self.write(layer, slot, t_cpu, sync=sync)
        return self.pool.submit(_task)

    # ---------- sync read to GPU tensor ----------
    def read(self, layer:int, slot:int, t_gpu:torch.Tensor):
        """读取一个 token 单元；只拷出有效 self.blk_bytes 长度"""
        buf = aligned_array((self.stride,), np.uint8)
        os.pread(self.fd, buf, self._offset(layer, slot))
        h_tmp = torch.from_numpy(buf[:self.blk_bytes].copy()).view(torch.float16)
        t_gpu.copy_(h_tmp.reshape(t_gpu.shape), non_blocking=True)

    def __del__(self):
        try:
            self.pool.shutdown(wait=False)
        finally:
            if self.fd is not None:
                os.close(self.fd)
