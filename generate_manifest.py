#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成runtime manifest的脚本（新增 from-shards-stream：对 Meta 分片逐层streaming合并并直接打包）

用途:
1) from-hf              : 从 HF 模型目录/ID 导出整合权重 -> 打包到 raw 设备 -> 生成 manifest
2) from-checkpoint      : 从单个 consolidated.pth/.safetensors 打包 -> 生成 manifest
3) from-meta            : 从 shapes_meta.json 生成 runtime_manifest.json
4) template             : 生成 manifest 模板
5) from-shards-stream   : ★ 推荐：从 Meta 分片目录（consolidated.00..NN.pth）逐层 streaming 合并后直接打包（不会OOM）

示例:
  # 逐层streaming（不会把全模型一次性载入内存）
  python generate_manifest.py from-shards-stream /home/roger/.llama/checkpoints/Llama3.1-70B /dev/nvme0n1p4 \
      --meta-out /data1/70b-fixed.shapes_meta.json \
      --manifest-out /data1/70b-fixed.runtime_manifest.json \
      --yes
"""

import sys
import os
import re
import json
import glob
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Iterable

import torch

# 项目内模块路径
sys.path.insert(0, str(Path(__file__).parent))

from llama3.weights_io_ssd_dram import (
    build_runtime_manifest,
    pack_any_to_raw,
    get_logical_block_size,
    O_DIRECT,
    O_LARGEFILE,
)

# -------------------------
# 通用工具
# -------------------------

def extract_model_name(ckpt_path: str) -> str:
    path = Path(ckpt_path)
    name = path.name if path.is_dir() else path.stem
    name = name.lower()
    name = re.sub(r'[^a-z0-9.-]', '-', name)
    name = re.sub(r'-+', '-', name).strip('-')
    return name or "llama-model"

def check_raw_device(device_path: str) -> None:
    print(f"\n🔍 检查设备: {device_path}")
    if not Path(device_path).exists():
        print(f"❌ 设备不存在: {device_path}")
        print(f"\n可用的块设备:")
        os.system("lsblk | grep -E 'nvme|sd'")
        sys.exit(1)
    try:
        fd = os.open(device_path, os.O_RDONLY | O_DIRECT | O_LARGEFILE)
        block_size = get_logical_block_size(fd)
        os.close(fd)
        print(f"✅ 设备可访问")
        print(f"   块大小: {block_size} bytes")
    except PermissionError:
        print(f"❌ 权限不足，请使用 sudo 运行")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 无法访问设备: {e}")
        sys.exit(1)

def _load_model_args_from_dir(path: str) -> Dict[str, int]:
    p = Path(path)
    root = p if p.is_dir() else p.parent
    for fname in ("params.json", "config.json"):
        f = root / fname
        if f.exists():
            js = json.loads(f.read_text(encoding="utf-8"))
            dim        = int(js.get("dim") or js.get("hidden_size"))
            n_layers   = int(js.get("n_layers") or js.get("num_hidden_layers"))
            n_heads    = int(js.get("n_heads") or js.get("num_attention_heads"))
            n_kv_heads = int(js.get("n_kv_heads") or js.get("num_key_value_heads") or n_heads)
            head_dim   = dim // n_heads
            inter      = js.get("intermediate_size") or js.get("ffn_hidden_size")
            inter      = int(inter) if inter is not None else None
            return {
                "dim": dim, "n_layers": n_layers, "n_heads": n_heads,
                "n_kv_heads": n_kv_heads, "head_dim": head_dim,
                "intermediate_size": inter
            }
    raise FileNotFoundError(f"params.json/config.json not found beside {path}")

def _hf_to_internal_name(name: str) -> str:
    n = name
    if n.startswith("model."): n = n[len("model."):]
    n = n.replace(".self_attn.q_proj.", ".attention.wq.")
    n = n.replace(".self_attn.k_proj.", ".attention.wk.")
    n = n.replace(".self_attn.v_proj.", ".attention.wv.")
    n = n.replace(".self_attn.o_proj.", ".attention.wo.")
    n = n.replace(".input_layernorm.", ".attention_norm.")
    n = n.replace(".post_attention_layernorm.", ".ffn_norm.")
    n = n.replace(".mlp.gate_proj.", ".feed_forward.w1.")
    n = n.replace(".mlp.up_proj.",   ".feed_forward.w3.")
    n = n.replace(".mlp.down_proj.", ".feed_forward.w2.")
    if n == "model.embed_tokens.weight": n = "embed_tokens.weight"
    if n == "lm_head.weight": n = "output.weight"
    return n

# -------------------------
# 现有功能：from-meta / from-checkpoint / from-hf / template
# （保持不变，略去已有实现）
# -------------------------

def generate_from_shapes_meta(shapes_meta_path: str, output_path: Optional[str] = None) -> None:
    if not Path(shapes_meta_path).exists():
        raise FileNotFoundError(f"❌ shapes_meta文件不存在: {shapes_meta_path}")

    if output_path is None:
        meta_filename = Path(shapes_meta_path).stem
        if meta_filename.endswith('.shapes_meta'):
            model_name = meta_filename[:-len('.shapes_meta')]
        else:
            model_name = meta_filename
        output_path = f"/data1/{model_name}.runtime_manifest.json"

    print(f"🔄 从shapes_meta生成manifest...")
    print(f"   输入: {shapes_meta_path}")
    print(f"   输出: {output_path}")
    result = build_runtime_manifest(shapes_meta_path, output_path)
    print(f"✅ Manifest生成成功: {result}")

def generate_from_checkpoint(ckpt_path: str, raw_device: str,
                             meta_out: Optional[str] = None,
                             manifest_out: Optional[str] = None,
                             output_dir: Optional[str] = None,
                             header_reserve: int = 4*1024*1024,
                             auto_confirm: bool = False) -> None:
    print(f"⚠️  警告: 这将会覆盖 {raw_device} 上的现有数据!")
    if not auto_confirm:
        confirm = input(f"确认要继续吗? (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ 操作已取消"); return

    model_name = extract_model_name(ckpt_path)
    print(f"\n🔄 从 checkpoint 打包到 raw 设备...")
    print(f"   Checkpoint: {ckpt_path}")
    print(f"   模型名字 : {model_name}")
    print(f"   Raw 设备 : {raw_device}")
    print(f"   头部保留: {header_reserve} bytes")

    if output_dir is None: output_dir = "/data1"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if meta_out is None:     meta_out = str(Path(output_dir) / f"{model_name}.shapes_meta.json")
    if manifest_out is None: manifest_out = str(Path(output_dir) / f"{model_name}.runtime_manifest.json")

    print(f"\n📦 Step 1: 打包权重到raw设备...")
    shapes_meta_path = pack_any_to_raw(
        ckpt_path, raw_device,
        shapes_meta_out=meta_out,
        header_reserve_bytes=header_reserve
    )
    print(f"✅ shapes_meta生成: {shapes_meta_path}")

    print(f"\n📝 Step 2: 生成runtime manifest...")
    build_runtime_manifest(shapes_meta_path, manifest_out)
    print(f"✅ runtime_manifest生成: {manifest_out}")

def generate_template(raw_device: str, output_path: Optional[str] = None,
                     n_layers: int = 80, model_name: str = "llama-model") -> None:
    if output_path is None:
        output_path = f"/data1/{model_name}.runtime_manifest.json"
    print(f"📝 生成manifest模板...")
    try:
        fd = os.open(raw_device, os.O_RDONLY | O_DIRECT | O_LARGEFILE)
        block_size = get_logical_block_size(fd); os.close(fd)
    except Exception:
        print(f"⚠️ 无法打开设备 {raw_device}，用默认块大小 4096"); block_size=4096
    manifest = {"version":1,"raw_device":raw_device,"block_size":block_size,"header_reserve":4*1024*1024,"params":[]}
    Path(output_path).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f"✅ 模板已生成: {output_path}")

# -------------------------
# 新功能：from-shards-stream （逐层合并 + 直接打包）
# -------------------------

ATTN_TAG_RE = re.compile(r"\.attention\.(wq|wk|wv|wo)\.weight$")
FFN_TAG_RE  = re.compile(r"\.feed_forward\.(w1|w2|w3)\.weight$")

def _iter_shards(dir_path: str) -> Iterable[str]:
    shards = sorted(glob.glob(str(Path(dir_path) / "consolidated.*.pth")))
    if not shards:
        # 兜底: *.pth（排除我们自己可能生成的 consolidated.pth）
        shards = [p for p in sorted(glob.glob(str(Path(dir_path) / "*.pth"))) if Path(p).name != "consolidated.pth"]
    if not shards:
        raise FileNotFoundError(f"在 {dir_path} 未找到 consolidated.*.pth 分片")
    return shards

def _concat_axis_hint(name: str, dim: int) -> int:
    # 返回优先拼接轴（0或1）；若非 ATT/FFN，返回 -1 交给自动判断
    m = ATTN_TAG_RE.search(name)
    if m:
        tag = m.group(1)
        if tag in ("wq", "wk", "wv"): return 0
        else: return 1  # wo
    m = FFN_TAG_RE.search(name)
    if m:
        tag = m.group(1)
        if tag in ("w1", "w3"): return 0
        else: return 1  # w2
    return -1

def _streaming_iter_from_sharded_dir(dir_path: str) -> Iterable[Tuple[str, torch.Tensor]]:
    """
    逐层 streaming：每次只合并一个“层”的所有参数到完整矩阵，然后 yield 给 pack 写 raw，随后释放内存。
    先处理全局参数（embed_tokens.weight / output.weight），再 0..n_layers-1。
    """
    args = _load_model_args_from_dir(dir_path)
    DIM, N = args["dim"], args["n_layers"]
    shards = list(_iter_shards(dir_path))

    def collect_and_yield(keys_predicate):
        """
        对满足 keys_predicate(name) 的参数，跨所有 shard 收集分片并合并后 yield。
        仅保留少数“全局”参数，数量很小，内存可控。
        """
        buckets: Dict[str, Dict] = {}  # name -> {axis, parts: [T], fixed: DIM}
        def add_piece(name, t: torch.Tensor):
            n = _hf_to_internal_name(name)
            if not keys_predicate(n): return
            if t.ndim != 2:
                # 非2D（例如 1D LN 权重），若重复出现形状一致就忽略后续
                if n not in buckets: buckets[n] = {"axis": None, "parts":[t.detach().cpu()], "fixed": None}
                return
            axis = buckets.get(n, {}).get("axis")
            fixed = DIM
            if axis is None:
                # 决定拼接轴：优先规则（若是 Attn/FFN）；否则自动：谁等于 DIM，另一维拼接
                hint = _concat_axis_hint(n, DIM)
                r, c = int(t.shape[0]), int(t.shape[1])
                if hint == 0:
                    if c == DIM: axis=0
                    elif r == DIM: t=t.T.contiguous(); axis=0
                    else: axis=0  # 尝试照0轴拼，下面再校验
                elif hint == 1:
                    if r == DIM: axis=1
                    elif c == DIM: t=t.T.contiguous(); axis=1
                    else: axis=1
                else:
                    # 自动：谁等于 DIM，另一维拼接
                    if c == DIM: axis=0
                    elif r == DIM: axis=1
                    elif r != DIM and c != DIM:
                        # 尝试转置
                        if r == DIM or c == DIM: t=t.T.contiguous(); r,c=c,r
                        if c == DIM: axis=0
                        elif r == DIM: axis=1
                        else:
                            # 看起来是完整矩阵或复制；直接当完整矩阵对待
                            buckets[n] = {"axis": None, "parts":[t.detach().cpu()], "fixed": None}
                            return
                buckets.setdefault(n, {"axis":axis, "parts":[], "fixed":fixed})
            else:
                # 如果需要，按已定轴做一次转置，保证固定边= DIM
                r, c = int(t.shape[0]), int(t.shape[1])
                if axis == 0 and c != DIM and r == DIM: t = t.T.contiguous()
                if axis == 1 and r != DIM and c == DIM: t = t.T.contiguous()
            buckets[n]["parts"].append(t.detach().cpu())

        # 收集
        for sp in shards:
            sd = torch.load(sp, map_location="cpu")
            for k, v in sd.items():
                if isinstance(v, torch.Tensor): add_piece(k, v)
            del sd

        # 合并并 yield
        for n, meta in buckets.items():
            parts, axis = meta["parts"], meta["axis"]
            if axis is None or len(parts) == 1:
                yield n, parts[0]
            else:
                cat = torch.cat(parts, dim=axis).contiguous()
                yield n, cat
        buckets.clear()

    # 1) 先处理全局（不在 layers.* 下的）
    def is_global_key(n: str) -> bool:
        return (not n.startswith("layers.")) and n.endswith(".weight")
    for name, t in collect_and_yield(is_global_key):
        yield name, t

    # 2) 逐层处理
    for L in range(N):
        prefix = f"layers.{L}."
        def is_layer_key(n: str) -> bool:
            return n.startswith(prefix) and n.endswith(".weight")
        for name, t in collect_and_yield(is_layer_key):
            yield name, t

def generate_from_shards_stream(ckpt_dir: str, raw_device: str,
                                meta_out: Optional[str] = None,
                                manifest_out: Optional[str] = None,
                                output_dir: Optional[str] = None,
                                header_reserve: int = 4*1024*1024,
                                auto_confirm: bool = False) -> None:
    """
    关键：通过 monkey-patch 将 pack_any_to_raw 的“目录迭代器”替换为我们上面的 streaming 生成器，
    这样 pack 就会按“逐层合并→立刻写 raw”的方式工作，峰值内存 ≪ 全模型。
    """
    print(f"⚠️  警告: 这将会覆盖 {raw_device} 上的现有数据!")
    if not auto_confirm:
        confirm = input(f"确认要继续吗? (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ 操作已取消"); return

    args = _load_model_args_from_dir(ckpt_dir)
    print(f"📋 模型配置: dim={args['dim']}, n_layers={args['n_layers']}, n_heads={args['n_heads']}, n_kv_heads={args['n_kv_heads']}, head_dim={args['head_dim']}")

    # monkey-patch
    import llama3.weights_io_ssd_dram as wio
    orig_iter_dir = wio._iter_tensors_from_dir
    def patched_iter_dir(dir_path: str):
        print("🩹 使用 streaming 分片迭代器（逐层合并）...")
        return _streaming_iter_from_sharded_dir(dir_path)
    wio._iter_tensors_from_dir = patched_iter_dir

    # 正常调用 pack_any_to_raw（它会从我们替换的迭代器按顺序拿 tensor 并写 raw）
    model_name = extract_model_name(ckpt_dir)
    if output_dir is None: output_dir = "/data1"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if meta_out is None:     meta_out = str(Path(output_dir) / f"{model_name}.shapes_meta.json")
    if manifest_out is None: manifest_out = str(Path(output_dir) / f"{model_name}.runtime_manifest.json")

    print(f"\n📦 Step 1: 打包权重到 raw 设备（逐层 streaming）...")
    shapes_meta_path = pack_any_to_raw(
        ckpt_dir, raw_device,
        shapes_meta_out=meta_out,
        header_reserve_bytes=header_reserve
    )
    print(f"✅ shapes_meta 生成: {shapes_meta_path}")

    print(f"\n📝 Step 2: 生成 runtime manifest ...")
    build_runtime_manifest(shapes_meta_path, manifest_out)
    print(f"✅ runtime_manifest 生成: {manifest_out}")

    # 还原原 iterator（可选）
    wio._iter_tensors_from_dir = orig_iter_dir

# -------------------------
# CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="生成LLaMA3推理系统 manifest（含 from-shards-stream：逐层合并打包）",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    subparsers = parser.add_subparsers(dest='command', help='命令', required=True)

    # from-meta
    pm = subparsers.add_parser('from-meta', help='从 shapes_meta.json 生成 runtime_manifest.json')
    pm.add_argument('shapes_meta', type=str)
    pm.add_argument('--out', type=str, default=None)

    # from-checkpoint
    pc = subparsers.add_parser('from-checkpoint', help='从单文件 checkpoint 打包并生成 manifest')
    pc.add_argument('checkpoint', type=str)
    pc.add_argument('raw_device', type=str)
    pc.add_argument('--output-dir', type=str, default=None)
    pc.add_argument('--meta-out', type=str, default=None)
    pc.add_argument('--manifest-out', type=str, default=None)
    pc.add_argument('--header-reserve', type=int, default=4*1024*1024)
    pc.add_argument('--yes', action='store_true')

    # template
    pt = subparsers.add_parser('template', help='生成 manifest 模板')
    pt.add_argument('--raw-device', type=str, required=True)
    pt.add_argument('--out', type=str, default=None)
    pt.add_argument('--model-name', type=str, default='llama-model')
    pt.add_argument('--layers', type=int, default=80)

    # from-shards-stream  ★
    pss = subparsers.add_parser('from-shards-stream', help='从 Meta 分片目录逐层 streaming 合并并打包（低内存）')
    pss.add_argument('checkpoint_dir', type=str, help='包含 consolidated.00..NN.pth 与 params.json 的目录')
    pss.add_argument('raw_device', type=str)
    pss.add_argument('--output-dir', type=str, default=None)
    pss.add_argument('--meta-out', type=str, default=None)
    pss.add_argument('--manifest-out', type=str, default=None)
    pss.add_argument('--header-reserve', type=int, default=4*1024*1024)
    pss.add_argument('--yes', action='store_true')

    args = parser.parse_args()

    try:
        if args.command == 'from-meta':
            generate_from_shapes_meta(args.shapes_meta, args.out)

        elif args.command == 'from-checkpoint':
            check_raw_device(args.raw_device)
            generate_from_checkpoint(args.checkpoint, args.raw_device,
                                     meta_out=args.meta_out, manifest_out=args.manifest_out,
                                     output_dir=args.output_dir, header_reserve=args.header_reserve,
                                     auto_confirm=args.yes)

        elif args.command == 'template':
            generate_template(args.raw_device, args.out, n_layers=args.layers, model_name=args.model_name)

        elif args.command == 'from-shards-stream':
            check_raw_device(args.raw_device)
            generate_from_shards_stream(args.checkpoint_dir, args.raw_device,
                                        meta_out=args.meta_out, manifest_out=args.manifest_out,
                                        output_dir=args.output_dir, header_reserve=args.header_reserve,
                                        auto_confirm=args.yes)

        print("\n✅ 操作完成!")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
