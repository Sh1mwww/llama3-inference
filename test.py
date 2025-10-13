import torch
from llama3.stream_mnt import get_streams, record_event_on, wait_event_on, on_stream

dev = "cuda:0"
s = get_streams(dev)

# 1) 6 流是否就绪（至少不为 None）
print("compute_mha:", s.compute_mha is not None,
      "weight_h2d_mha:", s.weight_h2d_mha is not None,
      "kv_h2d:", s.kv_h2d is not None, "kv_d2h:", s.kv_d2h is not None)

# 2) on_stream 不触发默认流
with on_stream(s.compute_mha, device=dev):
    a = torch.randn(1024, 1024, device=dev)
    b = torch.mm(a, a)

# 3) 事件记录/等待/回收（简单 smoke test）
evt_store = {}
record_event_on(evt_store, "mha_ready", s.compute_mha, device=dev)
wait_event_on(s.compute_ffn, evt_store["mha_ready"])  # FFN 流等待 MHA 事件
