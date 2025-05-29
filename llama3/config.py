import json
from dataclasses import dataclass, fields
from typing import Optional


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

    @staticmethod
    def from_json(params_path: str,
                  max_seq_len: int,
                  max_batch_size: int,
                  device: str):
        """
        读取官方 `params.json`，过滤掉不用的 key，再补充推理所需字段
        """
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        allowed = {f.name for f in fields(ModelArgs)}
        filtered = {k: v for k, v in params.items() if k in allowed}

        return ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **filtered
        )
