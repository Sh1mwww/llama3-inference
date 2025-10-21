# common_70b_runtime.py
import json, re, time
from pathlib import Path

import torch

from llama3.config import ModelArgs, KVCacheArgs, load_runtime_config
from llama3.model import Transformer
from llama3.stream_mnt import get_streams
from llama3.weights_io_ssd_dram import DTYPE_MAP
from llama3.raw_param_store import ParamStore
from llama3.weight_lbt import classify_group  # 仅用于名字分组
from llama3.generator import LLaMA  # 只复用若干工具方法（RoPE 重建等）

RESIDENT_PATS = [
    re.compile(r"^embed_tokens\."),
    re.compile(r"^norm\."),
    re.compile(r"^output\."),
    re.compile(r"^layers\.\d+\.(attn_norm|ffn_norm)\."),
    re.compile(r"\.bias$"),
]

def is_resident(name: str) -> bool:
    return any(p.search(name) for p in RESIDENT_PATS)

def decide_budgets(manifest_path: str,
                   hbm_total_gb: float = 16.0,
                   hbm_guard_gb: float = 2.0,
                   dram_total_gb: float = 128.0,
                   dram_kv_gb: float = 64.0,
                   dram_weight_gb: float = 40.0):
    """
    基于 manifest 粗估 stream 权重每层大小并给出 WSM/KV 的建议预算。
    """
    m = json.loads(Path(manifest_path).read_text())
    L = max(int(p["layer"]) for p in m["params"]) + 1
    per_layer_stream = [0]*L
    for p in m["params"]:
        if p.get("policy") == "stream" and p["layer"] >= 0:
            per_layer_stream[p["layer"]] += int(p["nbytes"])
    # 平均每层 stream 字节
    nonzero = [x for x in per_layer_stream if x > 0]
    avg_layer_bytes = (sum(nonzero)/len(nonzero)) if nonzero else 0

    # HBM：扣掉常驻/激活预留后的“层缓存”上限
    hbm_for_layers_gb = max(0.0, hbm_total_gb - hbm_guard_gb - 5.0)  # 常驻估 4~5GB
    max_cached_layers = 3
    if avg_layer_bytes > 0:
        max_cached_layers = max(2, min(6, int((hbm_for_layers_gb*1024**3)//avg_layer_bytes)))

    # DRAM：权重 CPU cache 可容纳层数
    cpu_cache_layers = 50
    if avg_layer_bytes > 0:
        cpu_cache_layers = max(8, min(L, int((dram_weight_gb*1024**3)//avg_layer_bytes)))

    return {
        "layers": L,
        "avg_layer_gb": avg_layer_bytes/(1024**3) if avg_layer_bytes else 0.0,
        "max_cached_layers": max_cached_layers,
        "prefetch_distance": 2,
        "warmup_layers": min(2, max_cached_layers),
        "cpu_cache_layers": cpu_cache_layers,
        "kv_dram_gb": dram_kv_gb,
    }

def materialize_residents_on_gpu(model: torch.nn.Module, manifest: dict, device: str):
    """
    仅为“resident”参数在 GPU 上分配真实存储（不拷贝权重数据，等后续从 SSD 填充）。
    避免 70B 在 CPU 上实体化导致 OOM。
    """
    name2meta = {p["name"]: p for p in manifest["params"] if p.get("policy") == "resident"}
    with torch.cuda.device(device):
        for name, p in model.named_parameters(recurse=True):
            meta = name2meta.get(name)
            if meta and is_resident(name):
                shape = tuple(meta["shape"])
                dtype = DTYPE_MAP[meta["dtype"]]
                # 直接在 GPU 上创建空存储并指派给 param.data
                try:
                    p.data = torch.empty(shape, dtype=dtype, device=device)
                except Exception as e:
                    raise RuntimeError(f"allocate resident tensor failed: {name} {shape} {dtype}: {e}")

def build_meta_streaming_model(ckpt_dir: str,
                               manifest_path: str,
                               device: str = "cuda:0"):
    """
    1) meta 设备上构建骨架；2) 仅为 resident 在 GPU 上实体化存储；
    3) 初始化 WSM（SSD 模式），预取/流式启用；4) 注入 KV 流；5) 重建 RoPE。
    """
    ckpt = Path(ckpt_dir)
    params_path = ckpt/"params.json"
    manifest = json.loads(Path(manifest_path).read_text())

    # ---- 配 KV Offloader 基本参数（类属性生效）----
    KVCacheArgs.dram_limit_gb = 64.0            # 64GB给KV热集
    KVCacheArgs.ssd_device_path = "/dev/nvme0n1p4"
    KVCacheArgs.max_concurrent_io = 4

    # ---- meta 构建 ----
    with torch.device("meta"):
        args = ModelArgs.from_json(str(params_path),
                                   max_seq_len=4096,   # 可按需改
                                   max_batch_size=4,
                                   device="meta")
        model = Transformer(args)  # ☆ 只建结构，不占用真实内存

    # ---- 只为 resident 分配 GPU 存储（不填数）----
    materialize_residents_on_gpu(model, manifest, device)

    # ---- 计算建议预算 ----
    budgets = decide_budgets(manifest_path)
    # ---- 初始化 WSM（SSD 模式）----
    from llama3.weight_streaming_manager import WeightStreamingManager
    wsm = WeightStreamingManager(
        model,
        device=device,
        prefetch_distance=budgets["prefetch_distance"],
        max_cached_layers=budgets["max_cached_layers"],
        warmup_layers=budgets["warmup_layers"],
        verbose=True,
        ssd_manifest_path=manifest_path,
        cpu_cache_layers=budgets["cpu_cache_layers"],
        staging_mb=64,
    )
    streams = get_streams(device)

    # ---- 把 WSM/streams 注入各层（复用 LLaMA 的做法）----
    if hasattr(model, "layers"):
        for lid, block in enumerate(model.layers):
            if hasattr(block, "attention"):
                block.attention.layer_id = lid
                block.attention.weight_manager = wsm
                block.attention.streams = streams
                block.attention.weight_h2d_stream = getattr(streams, "weight_h2d_mha", None)
            if hasattr(block, "feed_forward"):
                block.feed_forward.layer_id = lid
                block.feed_forward.weight_manager = wsm
                block.feed_forward.streams = streams
                block.feed_forward.weight_h2d_stream = getattr(streams, "weight_h2d_ffn", None)

    # ---- 给 KV Offloader 绑定 H2D/D2H 流 ----
    try:
        if hasattr(model, "layers"):
            first_off = None
            for layer in model.layers:
                if hasattr(layer, "attention"):
                    off = getattr(layer.attention, "offloader", None)
                    if off is not None:
                        off.h2d_stream = streams.kv_h2d
                        off.d2h_stream = streams.kv_d2h
                        if first_off is None: first_off = off
            if first_off is not None:
                wsm.kv_offloader = first_off  # 繁忙期暂停写
    except Exception as e:
        print(f"[WARN] configure KV streams failed: {e}")

    # ---- 处理 freqs_complex（把 RoPE 常量建在目标设备）----
    try:
        # 复用 LLaMA 的重建逻辑
        llama_dummy = object.__new__(LLaMA)  # 不调用其 __init__
        llama_dummy.model = model
        llama_dummy.args = args
        LLaMA._recreate_freqs_complex(llama_dummy, device)
    except Exception as e:
        print(f"[WARN] recreate freqs_complex failed: {e}")

    return model, wsm, budgets
