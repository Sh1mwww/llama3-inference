# llama3/raw_param_store.py

import ctypes
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

from llama3.weights_io_ssd_dram import (
    DirectIOFile,
    alloc_pinned_aligned,
    DTYPE_MAP,
)

# numpy 映射用于 reinterpret 路径
def _get_np_dtype_map():
    """构建 torch dtype 到 numpy dtype 的映射，处理 bfloat16 兼容性"""
    dtype_map = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.int8:    np.int8,
        torch.uint8:   np.uint8,
    }
    # bfloat16 支持需要 numpy >= 1.24
    try:
        dtype_map[torch.bfloat16] = np.dtype("bfloat16")
    except TypeError:
        # 旧版本 numpy 不支持 bfloat16，跳过
        pass
    return dtype_map

NP_DTYPE = _get_np_dtype_map()

def _ceil_to(v: int, b: int) -> int:
    return ((v + b - 1) // b) * b

def _detect_concat_axis(shapes: List[Tuple[int, ...]]) -> int:
    """判断多分片能否在某一轴拼接；能则返回轴索引，否则返回 -1。"""
    if not shapes:
        return -1
    base = shapes[0]
    for ax in range(len(base)):
        same_other = True
        for s in shapes:
            if len(s) != len(base):
                return -1
            for j in range(len(base)):
                if j == ax:
                    continue
                if s[j] != base[j]:
                    same_other = False
                    break
            if not same_other:
                break
        if same_other:
            return ax
    return -1


class ParamStore:
    """
    Raw Block Device 参数仓库包装器（wrapper）。

    功能：
      - fetch_layer(layer_id, ...): 从 raw 设备按 manifest 分片 O_DIRECT 读取并合并成 pinned 张量（可选直接搬到 GPU）。
      - offload_layer(layer_id, tensors, ...): 反向把张量写回 raw 设备（需要 rw=True）。
      - sanity_check_layer(layer_id, tensors): 逐分片做头部对齐小读校验（字节级）。
      - fetch_layer_async/offload_layer_async: 异步版本，利用线程池并行 I/O。
      - fetch_layer_batch: 批量加载多层。

    改进点:
      1. ✅ 复用 staging buffer，避免频繁分配 pinned memory
      2. ✅ 异步 I/O 支持 (线程池)
      3. ✅ 批量操作优化
      4. ✅ 更详细的错误处理
      5. ✅ 与 WeightStreamingManager 集成辅助方法

    用法：
      store = ParamStore("/data1/model.runtime_manifest.json", rw=True, staging_mb=32)
      with store:
          # 同步加载
          cpu_tensors = store.fetch_layer(5, only_stream=True)

          # 异步加载
          future = store.fetch_layer_async(6, only_stream=True)
          tensors = future.result()

          # 批量加载
          batch_tensors = store.fetch_layer_batch([3, 4, 5], only_stream=True)

          # 校验与回写
          store.sanity_check_layer(5, cpu_tensors, verbose=True)
          store.offload_layer(5, cpu_tensors, only_stream=True)
    """

    def __init__(self,
                 manifest_or_path: str | dict,
                 method: str = "bytecopy",     # "bytecopy" 或 "reinterpret"
                 staging_mb: int = 16,
                 rw: bool = False,
                 max_concurrent_io: int = 4) -> None:
        if isinstance(manifest_or_path, str):
            with open(manifest_or_path, "r") as f:
                self.manifest = json.load(f)
        else:
            self.manifest = manifest_or_path

        self.raw = self.manifest["raw_device"]
        self.bsz = int(self.manifest["block_size"])
        self.method_default = method
        self.rw = rw

        # 打开 raw 设备：读或读写
        mode = "rw" if rw else "r"
        self.dio = DirectIOFile(self.raw, mode=mode, block_size=self.bsz)

        # staging：对齐 pinned 暂存缓冲（复用以减少分配）
        staging_bytes = max(1, int(staging_mb)) * 1024 * 1024
        staging_bytes = _ceil_to(staging_bytes, self.bsz)
        self.staging = alloc_pinned_aligned(max(self.bsz, staging_bytes), self.bsz)

        # 预索引：layer -> name -> [segments]
        self.index: Dict[int, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
        for p in self.manifest["params"]:
            lid = int(p["layer"])
            self.index[lid][p["name"]].append(p)

        # 异步 I/O 线程池
        self.io_pool = ThreadPoolExecutor(
            max_workers=max_concurrent_io,
            thread_name_prefix="param_io"
        )

    # 允许 with 语法
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        """关闭资源：线程池、DirectIO 文件句柄"""
        if hasattr(self, 'io_pool') and self.io_pool is not None:
            self.io_pool.shutdown(wait=True)
            self.io_pool = None

        if hasattr(self, 'dio') and self.dio is not None:
            self.dio.close()
            self.dio = None

    def _ensure_staging(self, need_bytes: int):
        """确保 staging 足够大，且按 bsz 对齐。"""
        if need_bytes <= self.staging.numel():
            return
        new_bytes = _ceil_to(need_bytes, self.bsz)
        self.staging = alloc_pinned_aligned(new_bytes, self.bsz)

    def _groups_for_layer(self, layer_id: int, only_stream: bool, names: Optional[List[str]]):
        """获取层的参数分组，支持过滤"""
        groups = self.index.get(int(layer_id), {})
        if not groups:
            return {}

        # 过滤 only_stream
        if only_stream:
            g2 = {}
            for name, segs in groups.items():
                f = [s for s in segs if s.get("policy", "resident") == "stream"]
                if f:
                    g2[name] = f
            groups = g2

        # 过滤 name 列表
        if names is not None:
            groups = {n: groups[n] for n in names if n in groups}

        return groups

    def _read_bytes_odirect(self, want_off: int, want_len: int) -> bytes:
        """
        任意 offset 小段读取：用 O_DIRECT 对齐"包读"再切片。
        改进：复用 self.staging，避免频繁分配。
        """
        aligned_off = (want_off // self.bsz) * self.bsz
        delta = want_off - aligned_off
        need = delta + want_len
        need_aligned = _ceil_to(need, self.bsz)

        self._ensure_staging(need_aligned)
        self.dio.pread_into_tensor(self.staging, need_aligned, aligned_off)
        return bytes(self.staging[delta:delta + want_len].cpu().numpy())
    
# ========  直接读入 pinned uint8 目标（READ_FIXED 优先） ========
    def _read_into_pinned_u8(self, dst_u8: torch.Tensor, offset: int, nbytes: int):
        """
        将 raw 设备上的 [offset, offset+nbytes) 读入到 dst_u8（必须是 pinned/uint8/contiguous）。
        对齐则直接 READ_FIXED；不对齐则对齐“包读”到 staging，再拷贝有效字节。
        返回写入的有效 nbytes。
        """
        assert dst_u8.dtype == torch.uint8 and dst_u8.is_pinned() and dst_u8.is_contiguous(), \
              "dst_u8 must be pinned uint8 contiguous tensor"
        if nbytes <= 0:
            return 0
        bsz = self.bsz
        aligned = (offset % bsz == 0) and (nbytes % bsz == 0)
        if aligned:
            nread = self.dio.pread_into_tensor(dst_u8, nbytes, offset)
            if nread != nbytes:
                raise RuntimeError(f"Short read into pinned u8 at offset={offset}: got {nread}, expected {nbytes}")
            return nread
        else:
            # 对齐包读到 staging
            aligned_off = (offset // bsz) * bsz
            delta = offset - aligned_off
            need = delta + nbytes
            need_aligned = _ceil_to(need, bsz)

            self._ensure_staging(need_aligned)
            nread = self.dio.pread_into_tensor(self.staging, need_aligned, aligned_off)
            if nread != need_aligned:
                raise RuntimeError(f"Short read into staging at offset={aligned_off}: got {nread}, expected {delta + nbytes}")

            # 拷贝有效字节到 dst_u8
            ctypes.memmove(dst_u8.data_ptr(), self.staging.data_ptr() + delta, nbytes)
            return nbytes
        

            
    def fetch_layer(self,
                    layer_id: int,
                    only_stream: bool = False,
                    names: Optional[List[str]] = None,
                    method: Optional[str] = None,
                    to_device: Optional[torch.device | str] = None,
                    stream: Optional[torch.cuda.Stream] = None) -> Dict[str, torch.Tensor]:
        """
        从 raw 设备 O_DIRECT 读取并合并该层的参数，返回 {name: pinned CPU tensor} 或（若 to_device）GPU tensor。

        Args:
            layer_id: 层索引
            only_stream: 仅加载 policy=stream 的权重
            names: 仅加载指定名字子集
            method: "bytecopy"（稳妥/bf16友好，默认）或 "reinterpret"（fp16/fp32更快）
            to_device: 若给定（如 "cuda:0"），会把结果搬上 GPU
            stream: 配合 to_device 使用的 CUDA stream（非阻塞传输）

        Returns:
            Dict[str, torch.Tensor]: 参数字典
        """
        try:
            method = method or self.method_default
            groups = self._groups_for_layer(layer_id, only_stream=only_stream, names=names)
            if not groups:
                return {}

            out: Dict[str, torch.Tensor] = {}

            for name, segs in groups.items():
                base_dtype = DTYPE_MAP[segs[0]["dtype"]]
                elem_size  = torch.tensor([], dtype=base_dtype).element_size()
                shapes     = [tuple(s["shape"]) for s in segs]
                concat_ax  = _detect_concat_axis(shapes)

                total_nbytes = sum(int(s["nbytes"]) for s in segs)
                total_elems  = total_nbytes // elem_size

                if concat_ax == 0:
                    base_shape = list(shapes[0])
                    base_shape[0] = sum(s[0] for s in shapes)
                    new_shape = tuple(base_shape)
                    assert np.prod(new_shape) == total_elems, \
                        f"[{name}] shape/product mismatch: {new_shape} vs {total_elems} elems"
                else:
                    new_shape = (total_elems,)

                dst_cpu = torch.empty(new_shape, dtype=base_dtype, pin_memory=True)

                cursor_bytes = 0
                cursor_elems = 0

                for seg in segs:
                    offset = int(seg["offset"])
                    stride = int(seg["stride"])
                    nbytes = int(seg["nbytes"])

                    # 对齐读取到 staging
                    self._ensure_staging(stride)
                    nread = self.dio.pread_into_tensor(self.staging, stride, offset)
                    if nread != stride:
                        raise RuntimeError(
                            f"Short read on {name} segment @offset={offset}: "
                            f"got {nread}, expected {stride}"
                        )

                    seg_elems = nbytes // elem_size

                    if method == "reinterpret" and base_dtype in NP_DTYPE and concat_ax == 0:
                        src_u8  = self.staging[:nbytes].cpu().numpy()
                        np_view = src_u8.view(NP_DTYPE[base_dtype])[:seg_elems] \
                                        .reshape((seg["shape"][0],) + tuple(shapes[0][1:]))
                        dst_slice = dst_cpu[cursor_elems : cursor_elems + seg_elems] \
                                        .reshape(np_view.shape)
                        dst_slice.copy_(torch.from_numpy(np_view), non_blocking=False)
                    else:
                        # 纯字节复制（bf16 推荐）
                        ctypes.memmove(dst_cpu.data_ptr() + cursor_bytes,
                                       self.staging.data_ptr(), nbytes)

                    cursor_bytes += nbytes
                    cursor_elems += seg_elems

                out[name] = dst_cpu

            # 可选：搬到 GPU（支持自定义 stream）
            if to_device is not None:
                dev = torch.device(to_device)
                if dev.type == "cuda":
                    if stream is None:
                        # 无显式 stream 就同步搬（仍然利用 pinned）
                        for k, v in out.items():
                            out[k] = v.to(dev, non_blocking=False)
                    else:
                        # 使用用户提供的 CUDA stream 进行非阻塞搬运
                        with torch.cuda.stream(stream):
                            for k, v in out.items():
                                out[k] = v.to(dev, non_blocking=True)
                else:
                    for k, v in out.items():
                        out[k] = v.to(dev)

            return out

        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch layer {layer_id} "
                f"(only_stream={only_stream}, names={names}): {e}"
            ) from e


 # ======== 将“单个参数”直接读入 caller 提供的 pinned uint8 ========
 
    def fetch_param_into_pinned_u8(self,
                                   layer_id: int,
                                   name: str,
                                   dst_u8: torch.Tensor) -> int:
        """
        按 manifest 的多个 segment，把 layer_id:name 的参数字节连续写入 dst_u8（pinned/uint8）。
        要求：dst_u8.numel() >= sum(seg.nbytes)。
        """
        groups = self.index.get(int(layer_id), {})
        if name not in groups:
            return 0
        segs = groups[name]
        total = sum(int(s["nbytes"]) for s in segs)
        assert dst_u8.dtype == torch.uint8 and dst_u8.is_pinned() and dst_u8.is_contiguous(), \
              "dst_u8 must be pinned uint8 contiguous tensor"
        if dst_u8.numel() < total:
            raise ValueError(f"dst_u8 too small for {layer_id}:{name}, need {total}, have {dst_u8.numel()}")
        
        cursor = 0
        for seg in segs:
            offset = int(seg["offset"])
            nbytes = int(seg["nbytes"])
            slice_u8 = dst_u8.narrow(0, cursor, nbytes)
            self._read_into_pinned_u8(slice_u8, offset, nbytes)
            cursor += nbytes
        return cursor
    
    
# ========  将“整层多参数”直接读入 caller 提供的 pinned 目标 ========        

    def fetch_layer_into_pinned(self,
                                layer_id: int,
                                plan: Dict[str, torch.Tensor],
                                only_stream: bool = True):
        """
        plan: {param_name: pinned_uint8_tensor}
        仅对 plan 覆盖的参数执行加载；默认只加载 policy=stream 的参数（与 WSM 一致）。
        返回 {name: loaded_bytes}
        """
        groups = self._groups_for_layer(layer_id, only_stream=only_stream, names=list(plan.keys()))
        loaded = {}
        for name, segs in groups.items():
            dst = plan.get(name)
            if dst is None:
                continue
            n = self.fetch_param_into_pinned_u8(layer_id, name, dst)
            loaded[name] = n
        return loaded
        
    def fetch_layer_async(self,
                          layer_id: int,
                          **kwargs) -> Future:
        """
        异步版本的 fetch_layer，返回 Future[Dict[str, torch.Tensor]]。
        注意：to_device/stream 参数在异步场景需要谨慎使用（避免跨线程 CUDA 上下文问题）。
        """
        return self.io_pool.submit(self.fetch_layer, layer_id, **kwargs)

    def fetch_layer_batch(self,
                          layer_ids: List[int],
                          **kwargs) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        批量加载多层，利用线程池并行 I/O。

        Args:
            layer_ids: 要加载的层索引列表
            **kwargs: 传递给 fetch_layer 的参数

        Returns:
            Dict[layer_id, Dict[name, tensor]]
        """
        futures = {lid: self.fetch_layer_async(lid, **kwargs) for lid in layer_ids}
        return {lid: fut.result() for lid, fut in futures.items()}

    def offload_layer(self,
                      layer_id: int,
                      tensors: Dict[str, torch.Tensor],
                      only_stream: bool = False,
                      names: Optional[List[str]] = None,
                      method: Optional[str] = None):
        """
        把 {name: tensor} 写回 raw 设备（O_DIRECT 对齐）。
        需要在 __init__ 传入 rw=True。

        Args:
            layer_id: 层索引
            tensors: {name: tensor} 字典
            only_stream: 仅回写 policy=stream 的权重
            names: 仅回写指定名字子集
            method: "bytecopy" 或 "reinterpret"
        """
        if not self.rw:
            raise PermissionError(
                "ParamStore opened as read-only; set rw=True to offload."
            )

        try:
            method = method or self.method_default
            groups = self._groups_for_layer(layer_id, only_stream=only_stream, names=names)
            if not groups:
                return

            for name, segs in groups.items():
                if name not in tensors:
                    continue

                dst_tensor = tensors[name]
                expect_dtype = DTYPE_MAP[segs[0]["dtype"]]
                if dst_tensor.dtype != expect_dtype:
                    dst_tensor = dst_tensor.to(expect_dtype)

                if not dst_tensor.is_contiguous():
                    dst_tensor = dst_tensor.contiguous()

                elem_size = dst_tensor.element_size()
                total_nbytes = dst_tensor.numel() * elem_size

                cursor_bytes = 0

                for seg in segs:
                    offset = int(seg["offset"])
                    stride = int(seg["stride"])
                    nbytes = int(seg["nbytes"])

                    if cursor_bytes + nbytes > total_nbytes:
                        raise ValueError(
                            f"[{name}] provided tensor too small for segments: "
                            f"need {cursor_bytes + nbytes}, have {total_nbytes}"
                        )

                    # 准备 staging：前 nbytes 填有效数据，padding 区域清零
                    self._ensure_staging(stride)
                    self.staging[:stride].zero_()

                    # 字节复制
                    src_ptr = dst_tensor.data_ptr() + cursor_bytes
                    ctypes.memmove(self.staging.data_ptr(), src_ptr, nbytes)

                    # 对齐写：写 stride 字节到 offset
                    nw = self.dio.pwrite_from_tensor(self.staging, stride, offset)
                    if nw != stride:
                        raise RuntimeError(
                            f"Short write on {name} segment @offset={offset}: "
                            f"wrote {nw}, expected {stride}"
                        )

                    cursor_bytes += nbytes

        except Exception as e:
            raise RuntimeError(
                f"Failed to offload layer {layer_id} "
                f"(only_stream={only_stream}, names={names}): {e}"
            ) from e

    def offload_layer_async(self,
                            layer_id: int,
                            tensors: Dict[str, torch.Tensor],
                            **kwargs) -> Future:
        """异步版本的 offload_layer"""
        return self.io_pool.submit(self.offload_layer, layer_id, tensors, **kwargs)

    def sanity_check_layer(self,
                           layer_id: int,
                           tensors: Dict[str, torch.Tensor],
                           check_bytes: int = 64,
                           verbose: bool = False) -> Tuple[int, int]:
        """
        对该层所有参数逐分片做"头部 N 字节"校验：返回 (matched_pieces, total_pieces)。

        Args:
            layer_id: 层索引
            tensors: {name: tensor} 字典
            check_bytes: 每个分片校验的字节数（默认64）
            verbose: 是否打印详细信息

        Returns:
            (matched_pieces, total_pieces): 匹配的分片数 / 总分片数
        """
        groups = self.index.get(int(layer_id), {})
        total_pieces = 0
        matched_pieces = 0

        for name, segs in groups.items():
            if name not in tensors:
                continue

            dst = tensors[name]
            elem_size = dst.element_size()
            total_nbytes = dst.numel() * elem_size

            cursor_bytes = 0
            for si, seg in enumerate(segs):
                nbytes = int(seg["nbytes"])
                need = min(check_bytes, nbytes)

                # 使用复用 staging 的方法
                src_head = self._read_bytes_odirect(int(seg["offset"]), need)

                ok = False
                if cursor_bytes + need <= total_nbytes:
                    dst_head = (ctypes.c_char * need).from_address(
                        dst.data_ptr() + cursor_bytes
                    ).raw
                    ok = (src_head == dst_head)

                if verbose:
                    status = '✅ OK' if ok else '❌ FAIL'
                    print(f"[L{layer_id}][{name}][seg {si}/{len(segs)-1}] {status}")

                matched_pieces += int(ok)
                total_pieces += 1
                cursor_bytes += nbytes

        return matched_pieces, total_pieces

    # ======== 参数分片工具 ========
    def list_param_segments(self, layer_id: int, name: str):
        """返回该参数的分片列表（按 manifest 顺序）"""
        groups = self.index.get(int(layer_id), {})
        return list(groups.get(name, []))

    def param_total_nbytes(self, layer_id: int, name: str) -> int:
        """返回该参数所有分片 nbytes 之和"""
        segs = self.list_param_segments(layer_id, name)
        if not segs:
            raise KeyError(f"param not found: L{layer_id}:{name}")
        return sum(int(s["nbytes"]) for s in segs)

    # -------- 与 WeightStreamingManager 集成辅助方法 --------

    def get_param_metadata(self, layer_id: int, param_name: str) -> Optional[dict]:
        """
        获取指定参数的元数据（为 WSM 提供 offset/stride/nbytes 等信息）。

        Args:
            layer_id: 层索引
            param_name: 参数名（如 "layers.5.attention.wq.weight"）

        Returns:
            Optional[dict]: 包含 offset, stride, nbytes, shape, dtype, policy
        """
        groups = self._groups_for_layer(layer_id, only_stream=False, names=None)
        for name, segs in groups.items():
            if param_name == name or param_name.endswith(name):
                if segs:
                    return segs[0]  # 返回第一个分片的元数据
        return None

    def list_layer_params(self, layer_id: int, only_stream: bool = True) -> List[str]:
        """
        列出指定层的所有参数名。

        Args:
            layer_id: 层索引
            only_stream: 仅列出 stream 策略的参数

        Returns:
            List[str]: 参数名列表
        """
        groups = self._groups_for_layer(layer_id, only_stream=only_stream, names=None)
        return list(groups.keys())

    def get_storage_stats(self) -> dict:
        """
        获取存储统计信息。

        Returns:
            dict: 包含 total_params, total_bytes, resident_bytes, stream_bytes 等
        """
        total_params = 0
        total_bytes = 0
        resident_bytes = 0
        stream_bytes = 0

        for p in self.manifest["params"]:
            total_params += 1
            nbytes = int(p["nbytes"])
            total_bytes += nbytes

            if p.get("policy") == "resident":
                resident_bytes += nbytes
            elif p.get("policy") == "stream":
                stream_bytes += nbytes

        return {
            "total_params": total_params,
            "total_bytes": total_bytes,
            "total_gb": total_bytes / (1024**3),
            "resident_bytes": resident_bytes,
            "resident_gb": resident_bytes / (1024**3),
            "stream_bytes": stream_bytes,
            "stream_gb": stream_bytes / (1024**3),
            "block_size": self.bsz,
            "raw_device": self.raw,
        }

    def __del__(self):
        """析构时确保资源清理"""
        try:
            self.close()
        except Exception:
            pass
