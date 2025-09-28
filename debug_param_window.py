# debug_param_window.py (fixed)
import json, argparse
from llama3.weights_io_ssd_dram import DirectIOFile, alloc_pinned_aligned

def hexdump(b: bytes, n=32):
    return " ".join(f"{x:02x}" for x in b[:n])

def read_bytes(dio: DirectIOFile, want_off: int, want_len: int, bsz: int) -> bytes:
    """
    从任意 offset（可不对齐）读取 want_len 字节：
    - 实际以 O_DIRECT 的对齐要求向下对齐到 aligned_off
    - 读取足够覆盖 [want_off, want_off+want_len) 的对齐长度
    - 返回中间切片
    """
    aligned_off = (want_off // bsz) * bsz
    delta = want_off - aligned_off
    need = delta + want_len
    need_aligned = ((need + bsz - 1) // bsz) * bsz
    buf = alloc_pinned_aligned(max(bsz, need_aligned), bsz)
    dio.pread_into_tensor(buf, buf.numel(), aligned_off)
    return bytes(buf[delta:delta + want_len].cpu().numpy())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--name", default="layers.0.feed_forward.w1.weight")
    ap.add_argument("--window", type=int, default=1, help="show N before/after")
    args = ap.parse_args()

    m = json.load(open(args.manifest, "r"))
    raw, bsz = m["raw_device"], int(m["block_size"])
    params = m["params"]

    # 找到目标index
    idx = next((i for i,p in enumerate(params) if p["name"] == args.name), None)
    if idx is None:
        print("Param not found:", args.name); return

    dio = DirectIOFile(raw, mode="r", block_size=bsz)

    # 检查整个链条是否连贯（offset == 累计stride）
    cur = m.get("header_reserve", 0)
    ok_chain = True
    break_at = None
    for i,p in enumerate(params):
        if p["offset"] != cur:
            ok_chain = False
            break_at = (i, p["name"])
            break
        cur += p["stride"]
    print("[CHAIN]", "OK" if ok_chain else f"BREAK at #{break_at[0]} {break_at[1]}")

    start = max(0, idx - args.window)
    end   = min(len(params), idx + args.window + 1)

    for i in range(start, end):
        p = params[i]
        name, off, nb, stride = p["name"], int(p["offset"]), int(p["nbytes"]), int(p["stride"])
        print(f"\n[{i}] {name}")
        print(f"  offset={off}  nbytes={nb}  stride={stride}")

        # 读首/尾 64 字节（自动对齐 O_DIRECT）
        head_len = min(64, nb)
        head = read_bytes(dio, off, head_len, bsz)
        tail_len = min(64, nb)
        tail_off = off + (nb - tail_len)
        tail = read_bytes(dio, tail_off, tail_len, bsz)

        print("  head:", hexdump(head, 32))
        print("  tail:", hexdump(tail, 32))

        # 验证 next.offset 是否等于 off+stride
        if i+1 < len(params):
            nxt = params[i+1]
            expect = off + stride
            print("  next.offset", nxt["offset"], "==?", expect, "->", "OK" if nxt["offset"]==expect else "MISMATCH")

    dio.close()

if __name__ == "__main__":
    main()
