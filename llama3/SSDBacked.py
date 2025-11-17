import os, numpy as np, torch, threading, ctypes
from concurrent.futures import ThreadPoolExecutor

import ctypes.util
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
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
        
        # fixed buffers 注册信息
        self._fixed_chunks = []   # list of (np_array_view, usable_bytes)
        self._has_preadv  = hasattr(os, "preadv")
        self._has_pwritev = hasattr(os, "pwritev")
        
        
    def _offset(self, layer, slot):
        return (layer * self.blk_pl + slot) * self.stride
    
    # ---------- 注册 HostPinnedExtentPool 的大块（固定缓冲，可选） ----------
    
    def register_fixed_buffers_from_pool(self, pool) -> int:
        """
        从 pinned 池注册大块 pinned 区间为“固定缓冲”。
        说明：Python 无法直接做 io_uring REGISTER_BUFFERS，这里做严格对齐校验与视图缓存；
        若系统支持 preadv/pwritev，将可直接读/写到这些 pinned 区间。
        返回注册块数量。
        """
        if not hasattr(pool, "get_registered_chunks"):
            return 0
        chunks = pool.get_registered_chunks()
        self._fixed_chunks.clear()
        for(tensor, head_pad, usable) in chunks:
            # head_pad: offset in bytes from the start of the tensor
            # usable: number of usable bytes in the chunk
            np_view = tensor.numpy()[head_pad: head_pad + usable]
            ptr = np_view.ctypes.data
            if (ptr % ALIGN) != 0 or (usable % ALIGN) != 0:
                raise ValueError(f"Fixed chunk not aligned: ptr={hex(ptr)}, usable={usable}")
            self._fixed_chunks.append((np_view, usable))
        return len(self._fixed_chunks)    
    
    
    def read_into_pinned_aligned(self, layer:int, slot:int, dst_u8: torch.Tensor):
        """
        直接把一个 block（stride 字节）读到 pinned uint8 tensor（contiguous）。
        要求：dst_u8.numel() >= self.stride，且 data_ptr/len 均满足 4KiB 对齐。
        """
        assert dst_u8.dtype == torch.uint8 and dst_u8.is_pinned()
        assert dst_u8.is_contiguous(), "dst_u8 must be contiguous"
        arr = dst_u8.numpy()[:self.stride]
        ptr = arr.ctypes.data
        if (ptr % ALIGN) != 0:
            raise ValueError(f"dst_u8 not aligned: ptr={hex(ptr)}")
        if self._has_preadv:
            nread = os.preadv(self.fd, [arr], self._offset(layer, slot))
            if nread != self.stride:
                raise IOError(f"preadv read {nread} bytes, expected {self.stride}")
        else:
            # 回退：用 bytes 读 + 拷贝到对齐数组（你的原有路径）
            buf = aligned_array((self.stride,), np.uint8)
            raw_bytes = os.pread(self.fd, self.stride, self._offset(layer, slot))
            buf[:len(raw_bytes)] = np.frombuffer(raw_bytes, dtype=np.uint8)
            arr[:] = buf[:self.stride]
        return self.stride
        
    # ---------- 直接从 pinned 写（对齐要求：nbytes=stride, offset 对齐） ----------
    def write_from_pinned_aligned(self, layer:int, slot:int, src_u8: torch.Tensor, sync:bool=False):
        """
        直接把一个 block（stride 字节）从 pinned uint8 tensor（contiguous）写出。
        要求：src_u8.numel() >= self.stride，且 data_ptr/len 均满足 4KiB 对齐。
        """
        assert src_u8.dtype == torch.uint8 and src_u8.is_pinned(), "src must be pinned uint8"
        assert src_u8.is_contiguous(), "src_u8 must be contiguous"
        arr = src_u8.numpy()[:self.stride]
        ptr = arr.ctypes.data
        if (ptr % ALIGN) != 0:
            raise ValueError(f"src_u8 not aligned: ptr={hex(ptr)}")
        if self._has_pwritev:
            nwritten = os.pwritev(self.fd, [arr], self._offset(layer, slot))
            if nwritten != self.stride:
                raise IOError(f"pwritev wrote {nwritten} bytes, expected {self.stride}")
        else:
            # 回退：用 bytes 写（你的原有路径）
            buf = aligned_array((self.stride,), np.uint8)
            buf[:self.stride] = arr
            os.pwrite(self.fd, buf, self._offset(layer, slot))
        if sync:
            os.fsync(self.fd)
        return self.stride
    

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

    # ---------- sync read to destination buffer (zero-copy path) ----------
    def read(self, layer:int, slot:int, dst:torch.Tensor):
        """
        读取一个 block 单元到目标缓冲区（pinned uint8 tensor 或可转换的 tensor）。

        **零拷贝路径（推荐）**：
        - dst 是 pinned uint8 tensor，contiguous，满足 4KiB 对齐且 numel() >= self.stride
        - 使用 os.preadv (或回退到 pread) 直接读取到 pinned 内存
        - 调用方负责在 dst[:self.blk_bytes] 上做 dtype/shape 转换

        **兼容路径（旧API）**：
        - dst 是 GPU tensor (float16/bfloat16 等)
        - 回退到临时 aligned buffer + copy 路径

        Args:
            layer: 层索引
            slot: slot 索引
            dst: 目标 tensor
                - 零拷贝：torch.uint8 pinned, contiguous, numel() >= self.stride, 4KiB aligned
                - 兼容：任意 GPU tensor，将从临时 buffer 拷贝
        """
        # 零拷贝路径：dst 是 pinned uint8 tensor
        if dst.dtype == torch.uint8 and dst.is_pinned() and dst.is_contiguous():
            if dst.numel() >= self.stride:
                arr = dst.numpy()[:self.stride]
                ptr = arr.ctypes.data
                if (ptr % ALIGN) == 0:
                    # 直接读取到 pinned 内存，无拷贝
                    if self._has_preadv:
                        nread = os.preadv(self.fd, [arr], self._offset(layer, slot))
                        if nread != self.stride:
                            raise IOError(f"preadv read {nread} bytes, expected {self.stride}")
                    else:
                        raw_bytes = os.pread(self.fd, self.stride, self._offset(layer, slot))
                        arr[:len(raw_bytes)] = np.frombuffer(raw_bytes, dtype=np.uint8)
                    # 注意：调用方需要从 dst[:self.blk_bytes] 做 view/reshape
                    return self.stride

        # 兼容路径：dst 是 GPU tensor（旧API）
        buf = aligned_array((self.stride,), np.uint8)
        raw_bytes = os.pread(self.fd, self.stride, self._offset(layer, slot))
        buf[:len(raw_bytes)] = np.frombuffer(raw_bytes, dtype=np.uint8)
        effective_data = buf[:self.blk_bytes]
        h_tmp = torch.from_numpy(effective_data.copy()).view(torch.float16)
        dst.copy_(h_tmp.reshape(dst.shape), non_blocking=True)

    # ---------- batch read ----------
    def read_batch(self, layer:int, slots:list, dst_tensors:list):
        """
        批量读取多个blocks，利用连续I/O提高效率

        Args:
            layer: 层索引
            slots: slot索引列表 (升序排列以实现顺序I/O)
            dst_tensors: 目标 tensor 列表
                - 零拷贝：每个都是 pinned uint8 tensor (numel() >= stride, 4KiB aligned)
                - 兼容：GPU tensors (将使用临时 buffer)
        """
        if not slots or len(slots) != len(dst_tensors):
            return

        # 检查是否连续
        is_contiguous = all(slots[i] + 1 == slots[i+1] for i in range(len(slots) - 1))

        if is_contiguous and len(slots) > 1:
            # 连续读取优化：单次大I/O
            total_size = self.stride * len(slots)

            # 尝试零拷贝路径（如果所有 dst 都是 pinned uint8）
            all_pinned_u8 = all(
                t.dtype == torch.uint8 and t.is_pinned() and t.is_contiguous() and t.numel() >= self.stride
                for t in dst_tensors
            )

            if all_pinned_u8:
                # 检查是否可以用单个大 preadv（如果 tensors 在内存中连续）
                # 这里为简单起见，逐个读取（未来可优化为 preadv 多个 iovec）
                for i, (slot, dst) in enumerate(zip(slots, dst_tensors)):
                    self.read(layer, slot, dst)
            else:
                # 兼容路径：使用临时 buffer
                merged_buf = aligned_array((total_size,), np.uint8)
                raw_bytes = os.pread(self.fd, total_size, self._offset(layer, slots[0]))
                merged_buf[:len(raw_bytes)] = np.frombuffer(raw_bytes, dtype=np.uint8)

                # 分拆到各个tensor
                for i, t_gpu in enumerate(dst_tensors):
                    offset = i * self.stride
                    h_tmp = torch.from_numpy(merged_buf[offset:offset + self.blk_bytes].copy()).view(torch.float16)
                    t_gpu.copy_(h_tmp.reshape(t_gpu.shape), non_blocking=True)
        else:
            # 非连续则使用独立读取
            for slot, dst in zip(slots, dst_tensors):
                self.read(layer, slot, dst)

    def __del__(self):
        try:
            if hasattr(self, 'pool') and self.pool is not None:
                self.pool.shutdown(wait=False)
        finally:
            if hasattr(self, 'fd') and self.fd is not None:
                os.close(self.fd)

    def fdatasync(self):
            """确保已写数据持久化到设备（配合 warmup/回写场景使用）"""
            if self.fd is not None:
                os.fdatasync(self.fd)
                
    def fadvise_dontneed(self, offset: int, length: int):
        """
        提示内核丢弃页缓存（非强制）。用于“warmup 用过 buffered，推理前清缓存”的场景。
        有的平台 libc 可能没有 posix_fadvise，这里容错处理。
        """
        try:
            POSIX_FADV_DONTNEED = 4
            ret = libc.posix_fadvise(self.fd,
                                        ctypes.c_longlong(offset),
                                        ctypes.c_longlong(length),
                                        ctypes.c_int(POSIX_FADV_DONTNEED))
            if isinstance(ret, int) and ret != 0:
                raise OSError(ret, f"posix_fadvise failed: errno={ret} ({os.strerror(ret)})")
        except Exception:
            pass