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
            self.fd = os.open(dev_path, os.O_RDWR | os.O_DIRECT)
        except FileNotFoundError:
            print(f"[ERROR] Device not found: {dev_path}")
            raise

        self.blk_bytes = blk_bytes             # 逻辑有效字节
        self.stride    = ((blk_bytes + ALIGN - 1) // ALIGN) * ALIGN   # 物理步距
        self.blk_pl    = blk_per_layer         # 每层的槽位数量 = max_seq_len
        self.n_layers  = n_layers
        self.pool = ThreadPoolExecutor(max_workers=max_concurrent_io,
                                       thread_name_prefix="rbd_io")

    def _offset(self, layer, slot):
        return (layer * self.blk_pl + slot) * self.stride

    # ---------- sync write ----------
    def write(self, layer:int, slot:int, t_cpu:torch.Tensor, sync:bool=False):
        """
        写入单个 block 到SSD, 一个 block 一个 slot
        """

        data_u8 = t_cpu.detach().cpu().contiguous().view(torch.uint8).numpy()
        assert data_u8.nbytes == self.blk_bytes, f"block pack mismatch: {data_u8.nbytes} != {self.blk_bytes}"

        # 4K 对齐缓冲
        buf = aligned_array((self.stride,), np.uint8)
        # 写有效负载，尾部补零 - flatten data_u8 for proper copy
        data_flat = data_u8.ravel()  # Flatten to 1D
        buf[:self.blk_bytes] = data_flat[:self.blk_bytes]
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

    # ---------- batch write ----------
    def write_batch(self, layer:int, slots:list, tensors:list, sync:bool=False):
        """
        批量写入多个blocks到SSD，利用连续I/O提高效率

        Args:
            layer: 层索引
            slots: slot索引列表 (必须是升序排列以实现顺序I/O)
            tensors: 对应的tensor列表 (每个都是cpu tensor)
            sync: 是否同步fsync
        """
        if not slots or len(slots) != len(tensors):
            return

        # 检查是否连续，如果连续则使用单次大I/O
        is_contiguous = all(slots[i] + 1 == slots[i+1] for i in range(len(slots) - 1))

        if is_contiguous and len(slots) > 1:
            # 连续写入优化：合并成一个大buffer
            total_size = self.stride * len(slots)
            merged_buf = aligned_array((total_size,), np.uint8)

            for i, (slot, tensor) in enumerate(zip(slots, tensors)):
                data_u8 = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
                assert data_u8.nbytes == self.blk_bytes, f"block pack mismatch: {data_u8.nbytes} != {self.blk_bytes}"

                offset = i * self.stride
                data_flat = data_u8.ravel()  # Flatten to 1D
                merged_buf[offset:offset + self.blk_bytes] = data_flat[:self.blk_bytes]
                if self.stride > self.blk_bytes:
                    merged_buf[offset + self.blk_bytes:offset + self.stride] = 0

            # 单次大I/O写入
            os.pwrite(self.fd, merged_buf, self._offset(layer, slots[0]))
        else:
            # 非连续则使用独立写入
            for slot, tensor in zip(slots, tensors):
                self.write(layer, slot, tensor, sync=False)

        if sync:
            os.fsync(self.fd)

    # ---------- async batch write ----------
    def write_batch_async(self, layer:int, slots:list, tensors:list, sync:bool=False):
        def _task():
            self.write_batch(layer, slots, tensors, sync=sync)
        return self.pool.submit(_task)

    # ---------- sync read to GPU tensor ----------
    def read(self, layer:int, slot:int, t_gpu:torch.Tensor):
        """
        读取一个 block 单元, 只copy有效 self.blk_bytes 长度
        """
        buf = aligned_array((self.stride,), np.uint8)
        # os.pread signature: pread(fd, length, offset)
        raw_bytes = os.pread(self.fd, self.stride, self._offset(layer, slot))
        # Copy into aligned buffer
        buf[:len(raw_bytes)] = np.frombuffer(raw_bytes, dtype=np.uint8)
        # Extract effective bytes and convert to float16 tensor
        effective_data = buf[:self.blk_bytes]
        h_tmp = torch.from_numpy(effective_data.copy()).view(torch.float16)
        t_gpu.copy_(h_tmp.reshape(t_gpu.shape), non_blocking=True)

    # ---------- batch read ----------
    def read_batch(self, layer:int, slots:list, gpu_tensors:list):
        """
        批量读取多个blocks，利用连续I/O提高效率

        Args:
            layer: 层索引
            slots: slot索引列表 (升序排列以实现顺序I/O)
            gpu_tensors: 对应的GPU tensor列表 (每个都已分配好空间)
        """
        if not slots or len(slots) != len(gpu_tensors):
            return

        # 检查是否连续
        is_contiguous = all(slots[i] + 1 == slots[i+1] for i in range(len(slots) - 1))

        if is_contiguous and len(slots) > 1:
            # 连续读取优化：单次大I/O
            total_size = self.stride * len(slots)
            merged_buf = aligned_array((total_size,), np.uint8)
            # os.pread signature: pread(fd, length, offset)
            raw_bytes = os.pread(self.fd, total_size, self._offset(layer, slots[0]))
            merged_buf[:len(raw_bytes)] = np.frombuffer(raw_bytes, dtype=np.uint8)

            # 分拆到各个tensor
            for i, t_gpu in enumerate(gpu_tensors):
                offset = i * self.stride
                h_tmp = torch.from_numpy(merged_buf[offset:offset + self.blk_bytes].copy()).view(torch.float16)
                t_gpu.copy_(h_tmp.reshape(t_gpu.shape), non_blocking=True)
        else:
            # 非连续则使用独立读取
            for slot, t_gpu in zip(slots, gpu_tensors):
                self.read(layer, slot, t_gpu)

    def __del__(self):
        try:
            if hasattr(self, 'pool') and self.pool is not None:
                self.pool.shutdown(wait=False)
        finally:
            if hasattr(self, 'fd') and self.fd is not None:
                os.close(self.fd)
