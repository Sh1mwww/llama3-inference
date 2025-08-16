# stream_bundle.py
import torch

class StreamBundle:
    def __init__(self, device: str = "cuda"):
        assert torch.cuda.is_available()
        self.device = device
        # 数字越小优先级越高；确保权重H2D永远优先
        self.weight_h2d = torch.cuda.Stream(device=device, priority=-1)
        self.kv_h2d     = torch.cuda.Stream(device=device, priority=0)
        self.kv_d2h     = torch.cuda.Stream(device=device, priority=+1)

    def wait_weight_ready_on_current(self):
        torch.cuda.current_stream(self.device).wait_stream(self.weight_h2d)

# _STREAMS = {}
# def get_streams(device: str = "cuda") -> StreamBundle:
#     sb = _STREAMS.get(device)
#     if sb is None:
#         sb = _STREAMS[device] = StreamBundle(device)
#     return sb
_SB = None
def get_streams(device: str = "cuda") -> StreamBundle:
    global _SB
    if _SB is None:
        _SB = StreamBundle(device)
    return _SB