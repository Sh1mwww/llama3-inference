import os
import mmap
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List 
import torch
import numpy as np  


class SSDBackedKV:
    def __init__(self, path: str, n_layers: int, blk_bytes: int, 
                 blk_per_layer: int, max_concurrent_io: int = 4):
        self.path = os.path.abspath(path)
        self.blk_bytes = blk_bytes
        self.blk_pl = blk_per_layer
        self.n_layers = n_layers
        total_bytes = n_layers * blk_per_layer * blk_bytes
        
        # 创建文件
        if not os.path.exists(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            # 使用fallocate预分配空间
            os.system(f"fallocate -l {total_bytes} {self.path}")
        
        # 使用O_DIRECT提高性能
        self.fd = os.open(self.path, os.O_RDWR | os.O_DIRECT)
        self.mm = mmap.mmap(self.fd, total_bytes, mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE)
        
        # I/O线程池
        self.io_executor = ThreadPoolExecutor(max_workers=max_concurrent_io, 
                                            thread_name_prefix="ssd_io")
        
        # 写入缓冲区管理
        self.write_buffer = {}
        self.write_lock = threading.Lock()

    def _offset(self, layer: int, blk: int) -> int:
        return (layer * self.blk_pl + blk) * self.blk_bytes

    def write_async(self, layer: int, blk: int, t_cpu: torch.Tensor):
        """异步写入"""
        def _write():
            off = self._offset(layer, blk)
            data = t_cpu.cpu().numpy()
            view = np.ndarray(data.size, dtype=np.float16,
                            buffer=self.mm, offset=off).reshape(data.shape)
            np.copyto(view, data, casting='no')
            # 强制刷新到磁盘
            self.mm.flush(off, self.blk_bytes)
        
        return self.io_executor.submit(_write)

    def batch_read(self, layer: int, blocks: List[int], 
                   callback, callback_arg):
        """批量异步读取"""
        def _batch_read():
            loaded_data = {}
            for block in blocks:
                try:
                    off = self._offset(layer, block)
                    shape = (self.blk_bytes // 2,)  # float16
                    view = np.ndarray(shape[0], dtype=np.float16,
                                    buffer=self.mm, offset=off)
                    # 复制数据避免内存映射问题
                    data_copy = torch.from_numpy(view.copy())
                    loaded_data[block] = data_copy.reshape(-1)  # 需要根据实际shape调整
                except Exception as e:
                    print(f"Failed to read block {block}: {e}")
            
            if loaded_data:
                callback(callback_arg, loaded_data)
        
        return self.io_executor.submit(_batch_read)

    def read(self, layer: int, blk: int, t_gpu: torch.Tensor):
        """同步读取"""
        off = self._offset(layer, blk)
        view = np.ndarray(t_gpu.numel(), dtype=np.float16,
                         buffer=self.mm, offset=off).reshape(t_gpu.shape)
        h_tmp = torch.from_numpy(view.copy()).clone()
        t_gpu.copy_(h_tmp, non_blocking=True)

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'io_executor'):
            self.io_executor.shutdown(wait=True)
        if hasattr(self, 'mm'):
            self.mm.close()
        if hasattr(self, 'fd'):
            os.close(self.fd)