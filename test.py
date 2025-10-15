# 1) LBT 构建 + 合并
from llama3.raw_param_store import ParamStore
from llama3.weight_lbt import build_layer_copy_plan, classify_group
import torch, re

store = ParamStore("/data1/llama3.1-70b.runtime_manifest.json")  # 只读
# 伪造 param 字典（真实环境从你的 nn.Module 取）
param_tensors = {
    "layers.5.attention.wq.weight": torch.empty( (4096,4096), device="cuda", dtype=torch.float16 ),
    "layers.5.attention.wk.weight": torch.empty( (512,4096), device="cuda", dtype=torch.float16 ),
}
plan = build_layer_copy_plan(store, param_tensors, layer_id=5, group="mha", only_stream=True,
                             name_filter=re.compile(r"layers\.5\.attention\.(wq|wk)\.weight"))

for pname, seg, dst_slice in plan.iter_copy_entries():
    print("COPY:", pname, seg.file_offset, seg.nbytes, "-> dst byte-off", seg.dst_offset)

# 2) 注册池
from llama3.registered_pool import RegisteredPool
pool = RegisteredPool(n_buffers=8, buf_bytes=32<<20)
h = pool.get(16<<20)
print(pool.stats())
pool.put(h)
print(pool.stats())

# 3) HBM slab
from llama3.hbm_slab import HBMSlabPool
sp = HBMSlabPool("cuda:0")
s = sp.reserve(layer_id=5, group="mha", nbytes=128<<20, dtype=torch.float16)
sp.checkout(5,"mha")
view = s.u8_view(0, 1024)  # byte view
sp.checkin(5,"mha")
print(sp.stats())
