import torch
import numpy as np

kv_heads = 8
head_dim = 128
dtype = torch.float16
bytes_per_elem = 2
tokens_per_block_list = [16, 32, 64, 128, 256, 512, 1024]
repeat = 5

print(f"{'Tokens':>6} | {'Block KB':>8} | {'Avg µs':>8} | {'Bandwidth MB/s':>15}")
print("-" * 50)

for tokens_per_block in tokens_per_block_list:
    shape = (tokens_per_block, kv_heads, head_dim)
    block_size_bytes = torch.Size(shape).numel() * bytes_per_elem

    src = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    dst = torch.empty_like(src, device="cuda")

    for _ in range(2):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()

    times_us = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        dst.copy_(src, non_blocking=True)
        end.record()
        torch.cuda.synchronize()

        elapsed_us = start.elapsed_time(end) * 1000
        times_us.append(elapsed_us)

    avg_us = np.mean(times_us)
    bandwidth = block_size_bytes / (avg_us / 1e6) / 1e6

    print(f"{tokens_per_block:6} | {block_size_bytes/1024:8.0f} | {avg_us:8.2f} | {bandwidth:15.2f}")
