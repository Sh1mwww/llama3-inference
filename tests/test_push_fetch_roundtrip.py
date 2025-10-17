# tests/test_push_fetch_roundtrip.py
import torch
from llama3.layers import SelfAttention
from llama3.config import ModelArgs
from llama3.kv_offload import KVOffloader, BLOCK

def mk_args(device="cpu"):
    return ModelArgs(
        dim=64, n_layers=1, n_heads=8, n_kv_heads=4,
        vocab_size=32000, multiple_of=1, ffn_dim_multiplier=None,
        norm_eps=1e-5, rope_theta=1e4, use_scaled_rope=False,
        rms_norm_eps=1e-5, max_batch_size=2, max_seq_len=BLOCK,
        device=device, topk_blk=1
    )

def test_push_then_fetch_current_block():
    device="cpu"
    args = mk_args(device)
    sa = SelfAttention(args); sa.layer_id = 0
    off = KVOffloader(layers=1, heads=args.n_kv_heads or args.n_heads,
                      dim=args.dim//args.n_heads, max_seq=args.max_seq_len,
                      max_batch=args.max_batch_size, device=device, dtype_bytes=2, streams=None)
    sa.offloader = off
    sa.apply_causal_mask = True

    bsz, seqlen = 2, 4
    x = torch.randn(bsz, seqlen, args.dim)
    from llama3.model import precompute_theta_pos_frequencies
    freqs = precompute_theta_pos_frequencies(args.dim//args.n_heads, seqlen, device=device, theta=args.rope_theta)

    # 触发 push
    _ = sa(x, start_pos=0, freqs_complex=freqs)

    # 当前 block = 0，取回
    needed = torch.tensor([0], dtype=torch.long)
    k_full, v_full = off.fetch(0, needed, batch_idx=0, bsz=bsz)  # (B, H, T, D)
    assert k_full.shape[-1] == args.dim//args.n_heads
    assert k_full.size(0) == bsz
