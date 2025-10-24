import glob, torch

ckpt_dir = "/home/roger/.llama/checkpoints/Llama3.1-70B"
paths = sorted(glob.glob(f"{ckpt_dir}/consolidated.*.pth"))

emb_parts, head_parts = [], []
for p in paths:
    sd = torch.load(p, map_location="cpu")
    if "tok_embeddings.weight" in sd:
        emb_parts.append(sd["tok_embeddings.weight"])          # [16032, 8192]
    # 任选其一存在
    key = next((k for k in ("output.weight","lm_head.weight","model.lm_head.weight") if k in sd), None)
    if key:
        w = sd[key]
        # 统一为 [vocab/TP, hidden] 再拼接
        if w.shape == (8192, 16032):
            w = w.t()
        head_parts.append(w)                                    # [16032, 8192]

emb_full  = torch.cat(emb_parts,  dim=0)                        # [128256, 8192]
head_full = torch.cat(head_parts, dim=0)                        # [128256, 8192]
print(emb_full.shape, head_full.shape)
