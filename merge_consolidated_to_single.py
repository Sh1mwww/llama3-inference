#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 Meta 分片 (consolidated.00.pth ... consolidated.xx.pth) 合併成單個 consolidated.pth，
將權重命名映射為本專案內部格式，並對注意力投影矩陣 + FFN 矩陣做跨分片拼接重建完整矩陣。

用法:
  python merge_consolidated_to_single.py /home/roger/.llama/checkpoints/Llama3.1-70B /data1/70b.consolidated.pth
"""

import re
import os
import sys
import glob
import json
import torch
from pathlib import Path
from typing import Dict, Tuple

# ---- 命名映射 --------------------------------------------------------------

def _hf_to_internal_name(name: str) -> str:
    n = name
    if n.startswith("model."):
        n = n[len("model."):]
    n = n.replace(".self_attn.q_proj.", ".attention.wq.")
    n = n.replace(".self_attn.k_proj.", ".attention.wk.")
    n = n.replace(".self_attn.v_proj.", ".attention.wv.")
    n = n.replace(".self_attn.o_proj.", ".attention.wo.")
    n = n.replace(".input_layernorm.", ".attention_norm.")
    n = n.replace(".post_attention_layernorm.", ".ffn_norm.")
    # FFN: gate->w1, up->w3, down->w2（與本專案一致）
    n = n.replace(".mlp.gate_proj.", ".feed_forward.w1.")
    n = n.replace(".mlp.up_proj.",   ".feed_forward.w3.")
    n = n.replace(".mlp.down_proj.", ".feed_forward.w2.")
    if n == "model.embed_tokens.weight":
        n = "embed_tokens.weight"
    if n == "lm_head.weight":
        n = "output.weight"
    return n

# ---- 讀配置，至少拿到 dim（=8192 for 70B） ------------------------------------

def _load_model_args_from_dir(root: Path) -> Dict[str, int]:
    for fname in ("params.json", "config.json"):
        p = root / fname
        if p.exists():
            js = json.loads(p.read_text(encoding="utf-8"))
            dim        = int(js.get("dim") or js.get("hidden_size"))
            n_heads    = int(js.get("n_heads") or js.get("num_attention_heads"))
            n_kv_heads = int(js.get("n_kv_heads") or js.get("num_key_value_heads") or n_heads)
            head_dim   = dim // n_heads
            # 嘗試讀出 intermediate_size；讀不到也無妨，後面會用“觀察切片+相加”推斷
            inter = js.get("intermediate_size") or js.get("ffn_hidden_size")
            inter = int(inter) if inter is not None else None
            return {
                "dim": dim, "n_heads": n_heads, "n_kv_heads": n_kv_heads,
                "head_dim": head_dim, "intermediate_size": inter
            }
    raise FileNotFoundError(f"params.json / config.json not found in {root}")

# ---- 拼接策略：按參數名決定“優先拼接維度” -------------------------------------

ATTN_TAG_RE = re.compile(r"\.attention\.(wq|wk|wv|wo)\.weight$")
FFN_TAG_RE  = re.compile(r"\.feed_forward\.(w1|w2|w3)\.weight$")

def _preferred_axis(name: str, dim: int) -> Tuple[int, bool]:
    """
    返回 (優先拼接的維度, 是否需要在判斷前轉置) 的預設策略，用於“分片常見情形”：
      - ATTN:
          wq: 優先沿 0 拼（行拼）
          wk/wv: 優先沿 0 拼
          wo: 優先沿 1 拼（列拼）
      - FFN:
          w1/w3: 優先沿 0 拼（行拼）
          w2   : 優先沿 1 拼（列拼）
    若某片與優先策略不匹配（比如 (8192, 1024) 對 wq），則會在實際累積時判斷並允許轉置。
    """
    if ATTN_TAG_RE.search(name):
        tag = ATTN_TAG_RE.search(name).group(1)
        if tag in ("wq", "wk", "wv"):
            return 0, False
        else:  # wo
            return 1, False
    if FFN_TAG_RE.search(name):
        tag = FFN_TAG_RE.search(name).group(1)
        if tag in ("w1", "w3"):
            return 0, False
        else:  # w2
            return 1, False
    # 其它權重不拼接
    return -1, False

# ---- 合併主程式 -------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()
    out_pth = Path(sys.argv[2]).resolve()
    if not root.is_dir():
        print(f"❌ Not a directory: {root}")
        sys.exit(1)

    args = _load_model_args_from_dir(root)
    DIM = args["dim"]
    print(f"📋 args: dim={args['dim']} n_heads={args['n_heads']} n_kv_heads={args['n_kv_heads']} head_dim={args['head_dim']} "
          f"intermediate_size={args['intermediate_size']}")

    shards = sorted(glob.glob(str(root / "consolidated.*.pth")))
    if not shards:
        # 兜底：*.pth（排除我們自己合併出的 consolidated.pth）
        shards = [p for p in sorted(glob.glob(str(root / "*.pth"))) if Path(p).name != "consolidated.pth"]
    if not shards:
        print(f"❌ No consolidated.*.pth found in {root}")
        sys.exit(1)

    # 桶：name -> dict(axis=?, parts=[Tensor...], need_dim_check=True/False)
    buckets: Dict[str, Dict] = {}
    merged: Dict[str, torch.Tensor] = {}

    def _accumulate_slice(name: str, t: torch.Tensor):
        # 非 2D 直接放 merged（通常是 bias 或 embedding 等）
        if t.ndim != 2:
            if name not in merged:
                merged[name] = t.detach().cpu()
            else:
                # 再次出現且形狀一樣 -> 忽略；不一樣 -> 報錯
                if tuple(merged[name].shape) != tuple(t.shape):
                    raise RuntimeError(f"Duplicate non-2D key with different shapes: {name} {merged[name].shape} vs {t.shape}")
            return

        # 僅對 ATTN/FFN 權重做拼接邏輯
        m_attn = ATTN_TAG_RE.search(name)
        m_ffn  = FFN_TAG_RE.search(name)
        if not (m_attn or m_ffn):
            # 其它 2D 權重（embed/output/LayerNorm gamma 等）直接收下（通常每片只出現在一個 shard）
            if name not in merged:
                merged[name] = t.detach().cpu()
            else:
                if tuple(merged[name].shape) != tuple(t.shape):
                    raise RuntimeError(f"Duplicate key with different shapes: {name} {merged[name].shape} vs {t.shape}")
            return

        pref_axis, _ = _preferred_axis(name, DIM)
        r, c = int(t.shape[0]), int(t.shape[1])

        # 若已存在完整矩陣（之前某片就給了完整的），直接保留
        if name in merged:
            if tuple(merged[name].shape) != (r, c):
                # 如果已經有完整，且當前又來一個切片 -> 視為重複，忽略
                if name in buckets:
                    # 正常情況不會同時既有完整又有桶，但若出現，優先完整，忽略切片
                    return
                # 否則形狀衝突
                raise RuntimeError(f"Duplicate {name} with different shapes: {merged[name].shape} vs {t.shape}")
            return

        # 初始化桶
        B = buckets.get(name)
        if B is None:
            axis = pref_axis
            need_T = False
            # 嘗試按“優先策略”對齊
            if axis == 0:
                # 期望“列數 = DIM”，若不是，試著轉置；再不行就退而求其次讓下方邏輯判斷
                if c != DIM and r == DIM:
                    need_T = True
                elif c != DIM and r != DIM:
                    # 讓後續邏輯去判斷（對某些 ATTN 切片 e.g. wo 可能優先是 axis=1）
                    pass
            elif axis == 1:
                # 期望“行數 = DIM”，若不是，試著轉置
                if r != DIM and c == DIM:
                    need_T = True

            if need_T:
                t = t.T.contiguous()
                r, c = int(t.shape[0]), int(t.shape[1])

            # 若依然不匹配“axis 所對應的固定邊 = DIM”，嘗試反轉 axis
            if axis == 0 and c != DIM:
                axis = 1
            if axis == 1 and r != DIM:
                axis = 0

            B = {"axis": axis, "parts": []}
            buckets[name] = B

        # 到這裡，若需要再做一次轉置讓“固定邊 = DIM”
        axis = B["axis"]
        if axis == 0 and c != DIM and r == DIM:
            t = t.T.contiguous(); r, c = c, r
        elif axis == 1 and r != DIM and c == DIM:
            t = t.T.contiguous(); r, c = c, r

        # 最終檢查固定邊是否是 DIM
        if axis == 0 and c != DIM:
            raise RuntimeError(f"[{name}] expect concat axis=0 with columns==DIM({DIM}), got shape {t.shape}")
        if axis == 1 and r != DIM:
            raise RuntimeError(f"[{name}] expect concat axis=1 with rows==DIM({DIM}), got shape {t.shape}")

        B["parts"].append(t.detach().cpu())

    # 逐分片累積
    for sp in shards:
        print(f"  • loading {sp}")
        sd = torch.load(sp, map_location="cpu")
        for k, v in sd.items():
            if not isinstance(v, torch.Tensor):
                continue
            name = _hf_to_internal_name(k)
            _accumulate_slice(name, v)

    # 將桶裡的切片做 torch.cat（注意力 + FFN 都會來到這裡）
    for name, meta in buckets.items():
        axis  = meta["axis"]
        parts = meta["parts"]
        if len(parts) == 1:
            merged[name] = parts[0]
            continue
        cat = torch.cat(parts, dim=axis).contiguous()
        merged[name] = cat

    # 保存
    out_pth.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, str(out_pth))
    print(f"✅ saved: {out_pth}")

    # 快速檢查前兩層 ATTN/FFN 形狀
    def shp(d, key): return tuple(merged[d].shape) if d in merged else None
    for L in [0, 1]:
        print(f"Layer {L}:",
              "wq", shp(f"layers.{L}.attention.wq.weight", ""),
              "wk", shp(f"layers.{L}.attention.wk.weight", ""),
              "wv", shp(f"layers.{L}.attention.wv.weight", ""),
              "wo", shp(f"layers.{L}.attention.wo.weight", ""))
        print(f"          ",
              "w1", shp(f"layers.{L}.feed_forward.w1.weight", ""),
              "w2", shp(f"layers.{L}.feed_forward.w2.weight", ""),
              "w3", shp(f"layers.{L}.feed_forward.w3.weight", ""))
    print("🎯 提示：對 70B 期望 wq/wo=(8192,8192) wk/wv=(1024,8192)；w1/w3=(28672,8192) w2=(8192,28672)")
if __name__ == "__main__":
    main()
