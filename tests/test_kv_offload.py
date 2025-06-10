import torch
from llama3.kv_offload import KVOffloader


def test_fetch_returns_tensors():
    offloader = KVOffloader(
        layers=1,
        heads=2,
        dim=4,
        max_seq=64,
        max_batch=1,
        device="cpu",
        dtype_bytes=2,
    )

    k = torch.zeros((1, 2, 4), dtype=torch.float16)
    v = torch.zeros_like(k)
    offloader.push(0, 0, k, v)
    # Drop from hot cache to trigger CPU -> device fetch
    offloader.hot.clear()

    fetched_k, fetched_v = offloader.fetch(0, torch.tensor([0]))
    assert isinstance(fetched_k, torch.Tensor)
    assert isinstance(fetched_v, torch.Tensor)

