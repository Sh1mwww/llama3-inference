# tests/test_push_called.py
import torch
from llama3.layers import SelfAttention
from llama3.config import ModelArgs

# tests/test_push_called.py 里替换 SpyOffloader

class SpyOffloader:
    def __init__(self, heads, head_dim, block_sz, device):
        self.calls = []                      # 记录 push 调用
        self.heads = heads
        self.head_dim = head_dim
        self.block_sz = block_sz
        self.device = device
        self.importance_updates = []         # 记录 update_importances 调用

    # --- push: 按框架契约记录 (不做实际存储) ---
    def push(self, layer, blk, k, v, token_idx, batch_idx):
        # 关键：k/v 形状应是 (bsz, heads, head_dim) —— 我们只做记录
        self.calls.append((layer, blk, token_idx, batch_idx, tuple(k.shape), tuple(v.shape)))

    # --- topk_blocks: 签名与框架一致 ---
    def topk_blocks(self, layer, topk_n, batch_idx=None):
        # 返回空列表；forward 会把“当前块”补进去
        return []

    # --- fetch: 返回框架期望的形状 (bsz, seq_len, heads, head_dim) ---
    def fetch(self, layer, needed, batch_idx=None, bsz=1):
        T = int(needed.numel()) * self.block_sz
        shape = (bsz, T, self.heads, self.head_dim)   # 注意是 (B, T, H, D)
        k = torch.zeros(shape, dtype=torch.float32, device=self.device)
        v = torch.zeros_like(k)
        return k, v

    # --- update_importances: 做个 no-op，但记录参数便于断言 ---
    def update_importances(self, layer, blocks, block_scores, batch_idx=None):
        # blocks 可能是 python list 或 torch.Tensor（你的前向用 list）
        if hasattr(blocks, "tolist"):
            blocks = blocks.tolist()
        self.importance_updates.append((layer, list(blocks), list(block_scores), batch_idx))


def mk_args(device="cpu"):
    return ModelArgs(
        dim=64, n_layers=1, n_heads=8, n_kv_heads=4,
        vocab_size=32000, multiple_of=1, ffn_dim_multiplier=None,
        norm_eps=1e-5, rope_theta=1e4, use_scaled_rope=False,
        rms_norm_eps=1e-5, max_batch_size=2, max_seq_len=128,
        device=device, topk_blk=1
    )

def test_kv_push_is_called_seqlen_times():
    args = mk_args("cpu")
    sa = SelfAttention(args)
    sa.layer_id = 3
    heads = args.n_kv_heads or args.n_heads
    head_dim = args.dim // args.n_heads
    spy = SpyOffloader(heads=heads, head_dim=head_dim,
                       block_sz=sa.block_sz, device="cpu")
    sa.offloader = spy
    sa.apply_causal_mask = True

    bsz, seqlen = 2, 5
    x = torch.randn(bsz, seqlen, args.dim)
    from llama3.model import precompute_theta_pos_frequencies
    freqs = precompute_theta_pos_frequencies(head_dim, seqlen, device="cpu", theta=args.rope_theta)

    _ = sa(x, start_pos=0, freqs_complex=freqs)

    # 核心断言：push 被调用 seqlen 次，且 k/v 形状正确（无 squeeze）
    assert len(spy.calls) == seqlen
    for (layer, blk, token_idx, batch_idx, kshape, vshape) in spy.calls:
        assert layer == sa.layer_id
        assert kshape == (bsz, heads, head_dim)
        assert vshape == kshape
