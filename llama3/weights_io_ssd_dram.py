import os
import re
import json
import ctypes
import ctypes.util
import fcntl
from pathlib import Path
from glob import glob
from typing import Iterable, Optional, Dict, List, Any, Tuple
import torch

O_DIRECT    = getattr(os, "O_DIRECT", 0o40000)
O_LARGEFILE = getattr(os, "O_LARGEFILE", 0)
BLKSSZGET   = 0x1268  # ioctl: get block device logical block size

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

# ssize_t pread(int fd, void *buf, size_t count, off_t offset);
libc.pread.argtypes  = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_longlong]
libc.pread.restype   = ctypes.c_ssize_t
# ssize_t pwrite(int fd, const void *buf, size_t count, off_t offset);
libc.pwrite.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_longlong]
libc.pwrite.restype  = ctypes.c_ssize_t

def round_up(x, a): return ((x + a - 1) // a) * a

def tensor_bytes(t: torch.Tensor) -> memoryview:
    t = t.detach().cpu().contiguous()
    return memoryview(t.view(torch.uint8).numpy())

def get_logical_block_size(fd: int):
    """ioctl 获取块设备的逻辑块大小 bytes"""
    buf = ctypes.create_string_buffer(4)
    fcntl.ioctl(fd, BLKSSZGET, buf, True)
    return int.from_bytes(buf.raw, "little", signed=False)

class DirectIOFile:
        """
        以 O_DIRECT 打开的文件/设备包装器：
        - pread_into_tensor: 直接读到 pinned CPU 张量 uint8
        - pwrite_from_tensor: 直接从 pinned CPU 张量写出 uint8
        - 三对齐检查: offset / nbytes / buffer pointer 都必须是 block_size 的倍数
        """
        def __init__(self, 
                     path: str, 
                     mode: str = "r", 
                     block_size: Optional[int] = None):
            if "r" in mode and "w" in mode:
                flags = os.O_RDWR
            elif "w" in mode:
                flags = os.O_WRONLY
            else:
                flags = os.O_RDONLY
            flags |= O_DIRECT | O_LARGEFILE
            self.fd = os.open(path, flags)
            self.path = path
            self.block_size = block_size or self._detect_block_size() or 4096
            
            
        def _detect_block_size(self) -> Optional[int]:
            try:
                return get_logical_block_size(self.fd)
            except Exception:
                return None
            

        def _check_align(self, 
                         buf_ptr: int, 
                         nbytes: int, 
                         offset: int):
            '''
            offset 必须是块大小通常 512B 或 4096B的整数倍。
            因为 direct I/O 要直接对接设备，设备只能从扇区边界读写。
            nbytes读写的长度必须是块大小的整数倍。
            设备一次 I/O 至少要读写完整的扇区，不能像 page cache 那样自动补齐。
            用户态缓冲区的地址必须是块大小的整数倍。
            否则 DMA 引擎没法直接把数据写到用户缓冲区（因为 DMA 需要物理地址对齐）。
            '''
            bsz = self.block_size
            if (offset % bsz) != 0:
                raise ValueError(f"offset {offset} not {bsz}-aligned for O_DIRECT")
            if (nbytes % bsz) != 0:
                raise ValueError(f"nbytes {nbytes} not {bsz}-aligned for O_DIRECT")
            if (buf_ptr % bsz) != 0:
                raise ValueError(f"buffer pointer {hex(buf_ptr)} not {bsz}-aligned for O_DIRECT")
            
        def pread_into_tensor(self, 
                              t: torch.Tensor, 
                              nbytes: int, 
                              offset: int) -> int:
            '''
            用 O_DIRECT 方式从文件/块设备读取数据；
            直接放进 PyTorch pinned Tensor 的底层内存，避免了额外的 copy；
            同时保证所有 O_DIRECT 约束（对齐、缓冲区属性、偏移等）
            '''
            if not t.is_pinned():
                raise ValueError("Tensor is not pinned (pin_memory=True)")
            
            if t.device.type != "cpu":
                raise ValueError("Tensor must be on CPU")
            
            if not t.is_contiguous():
                raise ValueError("Tensor must be contiguous")
            
            if t.dtype != torch.uint8:
                raise ValueError("Tensor dtype must be uint8 for raw IO")

            ptr = t.data_ptr()
            self._check_align(ptr, nbytes, offset)

            ret = libc.pread(self.fd, 
                             ctypes.c_void_p(ptr), 
                             ctypes.c_size_t(nbytes), 
                             ctypes.c_longlong(offset))
            
            if ret < 0:
                err = ctypes.get_errno()
                raise OSError(err, f"pread failed: errno={err} ({os.strerror(err)})")
            return int(ret)
        
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
            
        def pwrite_from_tensor(self, t: torch.Tensor, nbytes: int, offset: int) -> int:
            if not t.is_pinned():
                raise ValueError("Tensor is not pinned (pin_memory=True)")
            if t.device.type != "cpu":
                raise ValueError("Tensor must be on CPU")
            if not t.is_contiguous():
                raise ValueError("Tensor must be contiguous")
            if t.dtype != torch.uint8:
                raise ValueError("Tensor dtype must be uint8 for raw IO")

            ptr = t.data_ptr()
            self._check_align(ptr, nbytes, offset)

            ret = libc.pwrite(self.fd, 
                              ctypes.c_void_p(ptr), 
                              ctypes.c_size_t(nbytes), 
                              ctypes.c_longlong(offset))
            if ret < 0:
                err = ctypes.get_errno()
                raise OSError(err, f"pwrite failed: errno={err} ({os.strerror(err)})")
            return int(ret)
        
        def close(self):
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None

        def __del__(self):
            try:
                self.close()
            except Exception:
                pass
            
def alloc_pinned_aligned(nbytes: int, block_size: int = 4096) -> torch.Tensor:
    """分配一个 pinned + block_size 对齐的 uint8 张量。"""
    
    if nbytes % block_size != 0:
        raise ValueError(f"nbytes must be multiple of {block_size} for O_DIRECT")
   
    # PyTorch pinned 通常页对齐（4K），但不保证 block_size 对齐（512/4K）
    # 这里尝试多次分配，直到找到一个对齐的
    for _ in range(4):
        t = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        if (t.data_ptr() % block_size) == 0:
            return t
        del t
    raise RuntimeError("Pinned allocation is not block-aligned; consider a small C helper with posix_memalign")

# ========== 权重分类规则 ==========
RESIDENT_PATTERNS = [
    r"^embed_tokens\.",                              
    r"^norm\.",
    r"^output\.",
    r"^layers\.\d+\.(attention_norm|ffn_norm)\.",    # attention/ffn RMSNorm
    r"\.bias$",                                      
]

STREAM_PATTERNS = [
    r"^layers\.\d+\.attention\.w[qkvo]\.weight$",    # Q/K/V/O
    r"^layers\.\d+\.feed_forward\.w[123]\.weight$",  # FFN W1/W2/W3
    r"^layers\.\d+\.feed_forward\.(gate|up|down).*weight$",  # FFN W1/W2/W3 (new names)
]

def classify_policy(name: str) -> str:
    for pat in RESIDENT_PATTERNS:
        if re.search(pat, name):
            return "resident"
    for pat in STREAM_PATTERNS:
        if re.search(pat, name):
            return "stream"
    return "resident" # default to resident


# ========== 一次性：checkpoint -> raw 分区 （写 shapes_meta.json） ==========

def _layer_idx_of(name: str) -> int:
    if name.startswith("layers."):
        try:
            return int(name.split(".")[1])
        except Exception:
            return -1
    return -1  # 非分层权重（如 embed/norm/output）

def _state_dict_from_loaded(sd: dict) -> dict:
    # 兼容 {'state_dict': ...} 或 {'model': {...}} 两种包装
    if "state_dict" in sd:
        return sd["state_dict"]
    if "model" in sd and isinstance(sd["model"], dict):
        return sd["model"]
    return sd

def _iter_tensors_from_pth(ckpt_path: str) -> Iterable[Tuple[str, torch.Tensor]]:
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = _state_dict_from_loaded(sd)
    # 稳定顺序：按名字排序
    for name, t in sorted(sd.items(), key=lambda x: x[0]):
        if isinstance(t, torch.Tensor):
            yield name, t


def _iter_tensors_from_dir(ckpt_dir: str) -> Iterable[Tuple[str, torch.Tensor]]:
    shards = sorted(glob(str(Path(ckpt_dir) / "consolidated*.pth")))
    if not shards:
        raise FileNotFoundError(f"No consolidated*.pth under {ckpt_dir}")
    for sp in shards:
        sd = torch.load(sp, map_location="cpu")
        sd = _state_dict_from_loaded(sd)
        for name, t in sorted(sd.items(), key=lambda x: x[0]):
            if isinstance(t, torch.Tensor):
                yield name, t
        del sd

def pack_any_to_raw(
    ckpt_path_or_dir: str,
    raw_dev: str,
    shapes_meta_out: Optional[str] = None,
    header_reserve_bytes: int = 4 * 1024 * 1024,
) -> str:
    """
    统一 pack：
    - 输入既可以是单一 .pth，也可以是包含 consolidated.*.pth 的目录
    - 顺序写入 raw 分区（O_DIRECT + 对齐补零），从 header_reserve_bytes 开始
    - 生成 shapes_meta.json（仅形状/类型/层号/策略/nbytes，不含 offset）
    返回 shapes_meta.json 路径
    """
    src = Path(ckpt_path_or_dir)
    if src.is_dir():
        it = _iter_tensors_from_dir(str(src))
    elif src.is_file():
        it = _iter_tensors_from_pth(str(src))
    else:
        raise FileNotFoundError(f"Path not found: {ckpt_path_or_dir}")

    # 打开 raw
    fd = os.open(raw_dev, O_DIRECT | O_LARGEFILE | os.O_WRONLY)
    bsz = get_logical_block_size(fd)
    assert header_reserve_bytes % bsz == 0, "header_reserve must be block-aligned"
    cur = header_reserve_bytes

    # shapes_meta（只记录形状/类型/层号/策略/nbytes；offset 每次启动由 manifest 推导）
    meta = {
        "version": 1,
        "raw_device": raw_dev,
        "header_reserve": header_reserve_bytes,
        "params": []
    }

    for name, t in it:
        # 写入 raw（直接 bytes + padding）
        t = t.detach().cpu().contiguous()
        raw = t.view(torch.uint8).numpy().tobytes()
        nbytes = len(raw)
        stride = round_up(nbytes, bsz)

        cbuf = (ctypes.c_ubyte * nbytes).from_buffer_copy(raw)
        ret = libc.pwrite(fd, ctypes.addressof(cbuf), ctypes.c_size_t(nbytes), ctypes.c_longlong(cur))
        if ret < 0:
            err = ctypes.get_errno()
            os.close(fd)
            raise OSError(err, f"pwrite failed: errno={err} {os.strerror(err)}")

        pad = stride - nbytes
        if pad:
            zero = (ctypes.c_ubyte * pad)()
            ret = libc.pwrite(fd, ctypes.addressof(zero), ctypes.c_size_t(pad), ctypes.c_longlong(cur + nbytes))
            if ret < 0:
                err = ctypes.get_errno()
                os.close(fd)
                raise OSError(err, f"pwrite padding failed: errno={err} {os.strerror(err)}")

        # shapes_meta 条目
        meta["params"].append({
            "name": name,
            "layer": _layer_idx_of(name),
            "dtype": str(t.dtype).split(".")[-1],
            "shape": list(t.shape),
            "nbytes": nbytes,
            "policy": classify_policy(name),
        })

        cur += stride

    os.fsync(fd)
    os.close(fd)

    if shapes_meta_out is None:
        # 统一命名：如果输入是目录，用目录名；如果输入是文件，用文件名
        base = src if src.is_dir() else src.with_suffix("")
        shapes_meta_out = str(Path(base).with_suffix(".shapes_meta.json"))
    Path(shapes_meta_out).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return shapes_meta_out


# ========== 每次启动：shapes_meta -> runtime_manifest（含 offset/stride） ==========

def build_runtime_manifest(shapes_meta_path: str, manifest_out_path: str) -> str:
    """
    每次启动调用：
    - 读取 shapes_meta（它的 params 顺序 == 打包写入顺序）
    - 打开 raw 设备查询 block_size
    - 按 shapes_meta["params"] 的“原始顺序”线性推导 offset/stride（从 header_reserve 开始）
    - 写 runtime_manifest.json（含 offset/stride、block_size、device）
    """
    from pathlib import Path
    import os, json

    meta = json.loads(Path(shapes_meta_path).read_text(encoding="utf-8"))
    raw_dev   = meta["raw_device"]
    header_rs = int(meta.get("header_reserve", 0))

    # block size 以当前设备为准（pack 和 runtime 必须一致）
    fd = os.open(raw_dev, os.O_RDONLY | O_DIRECT | O_LARGEFILE)
    try:
        bsz = get_logical_block_size(fd)
    finally:
        os.close(fd)

    # **不要排序**：严格按 shapes_meta 记录的顺序累计 offset
    params_in = meta["params"]  # preserve order!

    params_out = []
    cur = header_rs
    for p in params_in:
        nbytes = int(p["nbytes"])
        stride = round_up(nbytes, bsz)
        params_out.append({
            "name":   p["name"],
            "layer":  int(p["layer"]),
            "dtype":  p["dtype"],
            "shape":  p["shape"],
            "offset": cur,
            "nbytes": nbytes,
            "stride": stride,
            "policy": p.get("policy", "resident"),
        })
        cur += stride

    manifest = {
        "version": 1,
        "raw_device": raw_dev,
        "block_size": bsz,
        "header_reserve": header_rs,
        "params": params_out,
    }
    Path(manifest_out_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_out_path


# ========== 运行期：常驻加载 + 按层流式条目查询 ==========

DTYPE_MAP: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "int8": torch.int8,
    "uint8": torch.uint8,
}

def load_resident_to_gpu(
    model: torch.nn.Module,
    manifest: Dict[str, Any],
    device: str = "cuda:0",
    staging_bytes: int = 16 * 1024 * 1024,
) -> None:
    """
    启动时一次性把 policy=resident 的权重加载到 GPU。
    - 可先串行跑通，再按需要扩展为线程池并发。
    """
    dio = DirectIOFile(manifest["raw_device"], mode="r", block_size=manifest["block_size"])
    bsz = manifest["block_size"]
    staging_bytes = (staging_bytes // bsz) * bsz
    staging = alloc_pinned_aligned(staging_bytes, bsz)

    name_to_param = dict(model.named_parameters())

    for p in manifest["params"]:
        if p["policy"] != "resident":
            continue
        name = p["name"]
        if name not in name_to_param:
            print(f"[RESIDENT] missing param in model: {name}")
            continue

        stride = p["stride"]
        if stride > staging.numel():
            staging = alloc_pinned_aligned(((stride + bsz - 1)//bsz)*bsz, bsz)

        dio.pread_into_tensor(staging, stride, p["offset"])

        dst = torch.empty(p["shape"], dtype=DTYPE_MAP[p["dtype"]], pin_memory=True)
        dst.view(-1).view(torch.uint8)[:p["nbytes"]].copy_(staging[:p["nbytes"]])

        param = name_to_param[name]
        # param.data.copy_(dst.to(device, non_blocking=True))
        dst_dev = dst.to(device, non_blocking=True)
        # 如果参数仍在 meta 上，直接“以新张量替换”完成实体化；否则走原来的 copy_ 路径
        is_meta = getattr(param, "is_meta", False) or getattr(getattr(param, "data", None), "is_meta", False) \
                  or (hasattr(param, "device") and str(param.device).startswith("meta"))
        if is_meta:
            param.data = dst_dev
        else:
            param.data.copy_(dst_dev)

    dio.close()
    print("[RESIDENT] all resident params loaded to GPU")


def streamable_entries_for_layer(manifest: Dict[str, Any], layer_id: int) -> List[Dict[str, Any]]:
    """返回某层的 policy=stream 条目（Q/K/V/O、W1/W2/W3 等大矩阵）。"""
    return [p for p in manifest["params"] if p["policy"] == "stream" and int(p["layer"]) == int(layer_id)]


# ========== raw 读吞吐自检 ==========

def bench_raw_read(manifest_path: str, rounds: int = 8, chunk_bytes: int = 64*1024*1024) -> float:
    """
    简单的 raw 读吞吐测试（O_DIRECT + pread_into_tensor），返回 MB/s。
    只读连续 chunk（不破坏数据），用于估算你的 NVMe 冷/热吞吐。
    """
    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    dio = DirectIOFile(m["raw_device"], mode="r", block_size=m["block_size"])
    bsz = m["block_size"]
    chunk_bytes = (chunk_bytes // bsz) * bsz

    buf = alloc_pinned_aligned(chunk_bytes, bsz)

    import time
    t0 = time.perf_counter()
    offset = int(m.get("header_reserve", 0))
    total = 0
    for _ in range(rounds):
        dio.pread_into_tensor(buf, chunk_bytes, offset)
        total += chunk_bytes
        offset += chunk_bytes
    t1 = time.perf_counter()
    dio.close()

    mbps = (total / (1024*1024)) / (t1 - t0)
    print(f"[BENCH] raw_read_MBps={mbps:.1f} (chunk={chunk_bytes//(1024*1024)} MiB, rounds={rounds})")
    return mbps


# ========== CLI ==========

def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Weights IO (O_DIRECT + raw device)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("pack", help="Pack checkpoint to raw device (once)")
    sp.add_argument("ckpt", type=str)
    sp.add_argument("raw", type=str, help="/dev/nvme0n1p4 etc.")
    sp.add_argument("--meta-out", type=str, default=None)
    sp.add_argument("--header-reserve", type=int, default=4*1024*1024)

    sm = sub.add_parser("manifest", help="Build runtime manifest each start")
    sm.add_argument("shapes_meta", type=str)
    sm.add_argument("--out", type=str, default="/dev/shm/runtime_manifest.json")

    sb = sub.add_parser("bench-read", help="Benchmark raw read throughput")
    sb.add_argument("manifest", type=str)
    sb.add_argument("--rounds", type=int, default=8)
    sb.add_argument("--chunk", type=int, default=64*1024*1024)

    args = ap.parse_args()
    if args.cmd == "pack":
        out = pack_any_to_raw(args.ckpt, args.raw, args.meta_out, args.header_reserve)
        print(f"[OK] shapes_meta -> {out}")
    elif args.cmd == "manifest":
        out = build_runtime_manifest(args.shapes_meta, args.out)
        print(f"[OK] runtime_manifest -> {out}")
    elif args.cmd == "bench-read":
        bench_raw_read(args.manifest, args.rounds, args.chunk)
    else:
        ap.print_help()


if __name__ == "__main__":
    _cli()