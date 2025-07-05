#!/usr/bin/env python
import argparse
import torch

from llama3.generator import LLaMA

def _make_perf_tracker_thread_safe():
    """
    把 llama3.layers.PERF_TRACKER.lock 换成 RLock，
    并让 add_layer_stat 在拿不到锁时直接略过这一次累加，
    以免 GPU 同步期间造成全局阻塞。
    """
    import llama3.layers as layermod
    import threading

    tracker = layermod.PERF_TRACKER
    tracker.lock = threading.RLock()          # 1. 换成可重入锁

    orig_add = tracker.add_layer_stat

    def safe_add(self, layer_id, stat_name, value):
        # 2. 尝试 0.5ms 拿锁；拿不到就放弃本次统计
        locked = self.lock.acquire(timeout=5e-4)
        if not locked:
            return
        try:
            return orig_add(layer_id, stat_name, value)
        finally:
            self.lock.release()

    # 3. 替换方法
    layermod.PerformanceTracker.add_layer_stat = safe_add

_make_perf_tracker_thread_safe()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="目录下需包含 *.pth / tokenizer.model / params.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--prompt",
        nargs="+",
        required=True,
        help="支持一次输入多条 prompt,每条用引号包住",
    )
    p.add_argument("--max-gen-len", type=int, default=64)
    args = p.parse_args()

    llama = LLaMA.build(args.model_path, load_model=True, device=args.device)
    _, outs = llama.text_completion(args.prompt, max_gen_len=args.max_gen_len)
    for prompt, out in zip(args.prompt, outs):
        print("▼ Prompt:", prompt)
        print("▲ Completion:", out)
        print("-" * 60)


if __name__ == "__main__":
    main()
