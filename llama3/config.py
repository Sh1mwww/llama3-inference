import json
import torch.nn as nn
from dataclasses import dataclass, fields, field
from typing import Optional, List, Dict, Any



@dataclass
class LayerInfo:
    layer_id:int
    block: Optional[nn.Module] = None   #encoderblock
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KVCacheArgs:
    
    ssd_path: str = "/mnt/kv_cache/kv_cache.bin"
    ssd_size_gb: int = 500
    dram_limit_gb: float = 0.1
    


@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: Optional[int]
    vocab_size: int
    multiple_of: int
    ffn_dim_multiplier: Optional[float]
    norm_eps: float
    rope_theta: float
    use_scaled_rope: bool
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    gqa: Optional[int] = None
    max_batch_size: int = 512
    max_seq_len: int = 2048
    device: str = "cuda"
    topk_blk: int = 8
    layer_infos: List[LayerInfo] = field(default_factory=list) 
    
    @staticmethod
    def from_json(params_path: str,
                  max_seq_len: int,
                  max_batch_size: int,
                  device: str):
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        allowed = {f.name for f in fields(ModelArgs)}
        filtered = {k: v for k, v in params.items() if k in allowed}

        args =  ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **filtered
        )

        args.layer_infos = [LayerInfo(layer_id=i) for i in range(args.n_layers)]
        return args