# test_prefill_70b.py
import argparse, time
import torch
from pathlib import Path
from common_70b_runtime import build_meta_streaming_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="70B ckpt dir (has params.json, tokenizer, etc.)")
    ap.add_argument("--manifest", required=True, help="runtime_manifest.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--seqlen", type=int, default=4096)
    args = ap.parse_args()

    torch.cuda.set_device(int(args.device.split(":")[1]) if ":" in args.device else 0)
    model, wsm, budgets = build_meta_streaming_model(args.ckpt, args.manifest, device=args.device)

    print(f"[Budgets] {budgets}")

    # 构造“只做 prefill”的输入：B x T
    pad_id = 0
    tokens = torch.full((args.bsz, args.seqlen), pad_id, dtype=torch.long, device=args.device)
    # 随机化一点，模拟真实提示
    for b in range(args.bsz):
        tokens[b, : args.seqlen-1] = torch.randint(1, 32000, (args.seqlen-1,), device=args.device)
        tokens[b, 0] = 1  # BOS

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        # prefill：start_pos=0，整段前向
        logits = model(tokens, start_pos=0)
    torch.cuda.synchronize()
    dt = time.time() - t0

    tok = args.bsz * args.seqlen
    print(f"[Prefill] B={args.bsz} T={args.seqlen}  time={dt:.3f}s  tok/s={tok/dt:.0f}")

if __name__ == "__main__":
    main()
