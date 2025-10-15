from llama3.SSDBacked import RawBlockKVBackend
import torch, numpy as np

kv = RawBlockKVBackend("/dev/nvme0n1", n_layers=32, blk_bytes=262144, blk_per_layer=4096)

# 读：准备一个 4KiB 对齐且 pinned 的 uint8 目标
dst = torch.empty(kv.stride, dtype=torch.uint8, pin_memory=True)
kv.read_into_pinned_aligned(layer=0, slot=0, dst_u8=dst)  # 若系统支持 preadv，走直达；否则自动回退

# 写：把 pinned 内容回写
kv.write_from_pinned_aligned(layer=0, slot=1, src_u8=dst, sync=False)
