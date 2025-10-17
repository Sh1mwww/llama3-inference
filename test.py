# tests/test_inference_infra.py
import pytest
import torch

from llama3.config import ModelArgs
from llama3.layers import SelfAttention, precompute_theta_pos_frequencies
from llama3.generator import LLaMA
from llama3.model import Transformer

# ---------- 小工具 ----------
def _mk_args(device: str, *, max_seq_len=128, max_batch_size=2):
    # 关键：use_scaled_rope 是必填（见 config.py 的 ModelArgs）
    return ModelArgs(
        dim=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=None,
        vocab_size=32000,
        multiple_of=1,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        rope_theta=10000.0,
        use_scaled_rope=False,     # ★ 必填字段
        rms_norm_eps=1e-5,         # 保持与你代码默认一致
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        device=device,
        topk_blk=1,
    )

# ---------- 测试 1：SelfAttention profiling 的“事件级同步”路径 ----------
@pytest.mark.parametrize("enable_profiling", [False, True])
def test_self_attention_profiling_paths(enable_profiling):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda:0"
    args = _mk_args(device)
    sa = SelfAttention(args)
    sa.apply_causal_mask = True            # 建议默认打开因果 mask（见 layers 实现）  :contentReference[oaicite:2]{index=2}
    sa.enable_profiling = enable_profiling

    bsz, seqlen = 1, 8
    x = torch.randn(bsz, seqlen, args.dim, device=device, dtype=torch.float16)

    head_dim = args.dim // args.n_heads
    freqs = precompute_theta_pos_frequencies(head_dim, seqlen * 2, device=device, theta=args.rope_theta)

    out = sa(x, start_pos=0, freqs_complex=freqs)
    assert out.shape == (bsz, seqlen, args.dim)

    # 行为断言：profiling 关闭 → 两个时间=0；开启 → 非负
    if enable_profiling:
        assert sa.kv_elapsed_time >= 0
        assert sa.attn_time >= 0
    else:
        assert sa.kv_elapsed_time == 0
        assert sa.attn_time == 0

# ---------- 测试 2：LLaMA “先搬后半”(move then half) 顺序 ----------
def test_llama_move_then_half_order_cpu_gpu():
    # CPU 构建不应 half（你的 generator.py 原来在 __init__ 就 half，我们已建议改到 GPU 后再 half）  :contentReference[oaicite:3]{index=3}
    args = _mk_args("cpu")
    class _Tok:
        pad_token_id = 0
        eos_token_id = 1
        def encode(self, s, add_special_tokens=False): return [1, 2, 3]
        def decode(self, ids): return "x"

    llama = LLaMA(_Tok(), checkpoint=None, args=args)
    p = next(llama.model.parameters())
    assert p.device.type == "cpu"
    assert p.dtype == torch.float32  # CPU 不做 half

    if torch.cuda.is_available():
        llama.model = llama.model.to("cuda:0").half()  # 模拟 build() 中“先搬后半”
        p = next(llama.model.parameters())
        assert p.device.type == "cuda"
        assert p.dtype == torch.float16

# ---------- 测试 3：RoPE 频率张量在目标设备上的重建路径 ----------
def test_precompute_freqs_device_sanity():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    t = precompute_theta_pos_frequencies(head_dim=8, seq_len=64, device=device, theta=10000.0)
    assert t.device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert t.dtype.is_complex
