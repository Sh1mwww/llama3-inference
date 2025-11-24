import json
import os
import time
from pathlib import Path
from typing import List, Optional
import torch, torch.nn as nn
from tqdm import tqdm
import copy
from transformers import LlamaTokenizerFast, AutoTokenizer  
from .config import ModelArgs
from .model import Transformer

try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False

    # Fallback no-op functions to avoid sprinkling if-guards everywhere.
    class nvtx:
        @staticmethod
        def range_push(name): 
            pass

        @staticmethod
        def range_pop(): 
            pass


# ================================
# LLaMA wrapper
# - Handles tokenizer/model build, (optional) weight streaming config, and text generation
# ================================

class LLaMA:
    # def __init__(self, tokenizer, checkpoint, args: ModelArgs):
    #     """
    #     Initialize model and (optionally) load checkpoint weights.
    #     Model is first constructed on the device specified in `args.device` (may be 'cpu' or 'cuda:X').
    #     """
    #     self.tokenizer = tokenizer
    #     self.args = args

    #     # (Optional) Global tracker init (left commented — keep original code)
    #     # from .global_state_tracker import init_global_tracker, get_global_tracker
    #     # from .kv_offload import BLOCK
    #     # if get_global_tracker() is None:
    #     #     print(f"[INFO] Initializing global state tracker...")
    #     #     n_blocks = (args.max_seq_len + BLOCK - 1) // BLOCK
    #     #     tracker = init_global_tracker(
    #     #         max_batch=args.max_batch_size,
    #     #         layers=args.n_layers,
    #     #         n_blocks=n_blocks
    #     #     )
    #     #     print(f"[INFO] Global state tracker initialized, waiting for actual batch registration")

    #     # print(f"[INFO] Initializing model on device: {args.device}")
    #     # self.model = Transformer(args)

    #     # print(f"[INFO] Moving model to {args.device}...")
    #     # self.model = self.model.to(args.device)
        
    #     use_meta_init = (
    #     getattr(args, "param_init_device", None) == "meta"
    #         or getattr(args, "init_on_meta", False)
    #         or str(getattr(args, "device", "")) == "meta"
    #     )
    #     if use_meta_init:
    #         meta_args = copy.copy(args)
    #         # 避免在构造期把 freqs_complex 等建在真实设备
    #         meta_args.device = "meta"
    #         print("[INFO] Initializing model skeleton on device: meta")
    #         with torch.device("meta"):
    #             self.model = Transformer(meta_args)
    #     else:
    #         print(f"[INFO] Initializing model on device: {args.device}")
    #         self.model = Transformer(args)
    #         print(f"[INFO] Moving model to {args.device}...")
    #         self.model = self.model.to(args.device)

    #     # print(f"[INFO] Converting to half precision...")
    #     # self.model = self.model.half()

    #     import re
    #     def _remap_ckpt_keys(sd):
    #         out = {}
    #         for k, v in sd.items():
    #             nk = k
    #             nk = re.sub(r"^layers\.(\d+)\.input_layernorm\.(weight|bias)$",
    #                         r"layers.\1.attn_norm.\2", nk)
    #             nk = re.sub(r"^layers\.(\d+)\.post_attention_layernorm\.(weight|bias)$",
    #                         r"layers.\1.ffn_norm.\2", nk)
    #             nk = nk.replace("model.embed_tokens.", "embed_tokens.")
    #             nk = nk.replace("model.norm.", "norm.")
    #             out[nk] = v
    #         return out


    #     if checkpoint is not None:
    #         checkpoint = _remap_ckpt_keys(checkpoint)
    #         print(f"[INFO] Loading state dict...")

    #         has_meta = False
    #         try:
    #             has_meta = any(getattr(p, "is_meta", False) or (p.device.type == "meta")
    #                            for p in self.model.parameters())
    #         except Exception:
    #             pass

    #         if has_meta:
    #             try:
    #                 self.model = self.model.to_empty("cpu")
    #             except Exception as e:
    #                 print(f"[WARN] to_empty('cpu') failed on meta model: {e}")

    #         # 重要：assign=True 避免"copy 到 meta 是 no-op"的警告
    #         missing_keys, unexpected_keys = self.model.load_state_dict(
    #             checkpoint, strict=False, assign=has_meta
    #         )

    #         if missing_keys:
    #             print(f"[WARNING] Missing keys: {len(missing_keys)} keys")
    #         if unexpected_keys:
    #             print(f"[WARNING] Unexpected keys: {len(unexpected_keys)} keys")
    #         print(f"[INFO] Model weights loaded successfully")
            
    #     for name, p in self.model.named_parameters():
    #         if getattr(p, "is_meta", False):
    #             if name.endswith("norm.weight"):
    #                 p.data = torch.ones(p.shape, dtype=p.dtype, device="cpu")
    #             elif name.endswith("bias"):
    #                 p.data = torch.zeros(p.shape, dtype=p.dtype, device="cpu")
    #             else:
    #                 buf = torch.empty(p.shape, dtype=p.dtype, device="cpu")
    #                 nn.init.normal_(buf, mean=0.0, std=0.02)
    #                 p.data = buf

    #     for name, b in self.model.named_buffers():
    #         if getattr(b, "is_meta", False):
    #             b.data = torch.zeros(b.shape, dtype=b.dtype, device="cpu")
    
    
    def __init__(self, tokenizer, checkpoint, args: ModelArgs):
        """
        Initialize model and (optionally) load checkpoint weights.
        - 支持 meta 骨架（参数不分配真实 storage）
        - 若提供 checkpoint:
            * 若骨架为 meta: 先 to_empty(device="cpu")，再 load_state_dict(assign=True)
            * 自动 remap 关键 keys（attn_norm / ffn_norm / embed_tokens / norm）
        - 兜底：清理所有残余 meta 参数/缓冲
        """
        import copy as _copy
        import re as _re
        import torch
        import torch.nn as nn

        self.tokenizer = tokenizer
        self.args = args



        # ---------- 1) 构建模型骨架（统一在 CPU 上创建） ----------
        # 移除 meta device 支持，统一用 CPU stub 参数
        print("[INFO] Initializing model skeleton on device: cpu")
        cpu_args = _copy.copy(args)
        cpu_args.device = "cpu"
        self.model = Transformer(cpu_args)

        # ---------- 2) （可选）checkpoint key 映射 ----------
        def _remap_ckpt_keys(sd):
            out = {}
            for k, v in sd.items():
                nk = k
                nk = _re.sub(r"^layers\.(\d+)\.input_layernorm\.(weight|bias)$",
                            r"layers.\1.attention_norm.\2", nk)
                nk = _re.sub(r"^layers\.(\d+)\.post_attention_layernorm\.(weight|bias)$",
                            r"layers.\1.ffn_norm.\2", nk)
                nk = nk.replace("model.embed_tokens.", "embed_tokens.")
                nk = nk.replace("model.norm.", "norm.")
                out[nk] = v
            return out

        # ---------- 3) 加载 checkpoint（meta → CPU 空实体化 → assign=True） ----------
        # missing_keys = []
        # unexpected_keys = []

        # if checkpoint is not None:
        #     checkpoint = _remap_ckpt_keys(checkpoint)
        #     print("[INFO] Loading state dict...")

        #     # 检查是否为 meta 骨架
        #     has_meta = False
        #     try:
        #         for _n, _p in self.model.named_parameters():
        #             if getattr(_p, "is_meta", False) or (_p.device.type == "meta"):
        #                 has_meta = True
        #                 break
        #     except Exception:
        #         pass

        #     if has_meta:
        #         # ☆ 关键：必须用关键字 device=，否则就是你日志里的报错
        #         self.model = self.model.to_empty(device="cpu")

        #     # 重要：assign=True → 直接绑定 storage；meta 情况下避免 no-op
        #     missing_keys, unexpected_keys = self.model.load_state_dict(
        #         checkpoint, strict=False, assign=has_meta
        #     )

        #     if missing_keys:
        #         print(f"[WARNING] Missing keys: {len(missing_keys)} keys")
        #     if unexpected_keys:
        #         print(f"[WARNING] Unexpected keys: {len(unexpected_keys)} keys")
        #     print("[INFO] Model weights loaded successfully")
        # ---------- 3) 加载 checkpoint（如果提供且不是 raw-ssd 模式） ----------
        use_raw_ssd = (getattr(args, "weight_source", "") == "raw-ssd")
        if (checkpoint is not None) and (not use_raw_ssd):
            checkpoint = _remap_ckpt_keys(checkpoint)
            print("[INFO] Loading state dict...")
            # 模型已在 CPU 上，直接加载
            missing_keys, unexpected_keys = self.model.load_state_dict(
                checkpoint, strict=False
            )
            if missing_keys:
                print(f"[WARNING] Missing keys: {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"[WARNING] Unexpected keys: {len(unexpected_keys)} keys")
            print("[INFO] Model weights loaded successfully")
            
        # ---------- 3.5) raw-ssd 模式：将大权重替换为 0-size CPU stub ----------
        # 定义哪些是"核心模块"（必须保留完整数据，无论大小）
        _CORE_PAT = _re.compile(r"(^embed_tokens\.|^norm\.|^output\.)")
        # 定义哪些是"小参数"（保留完整数据）
        _SMALL_PAT = _re.compile(r"(norm(\.|$)|\.bias$)")
        _MAX_SAFE_NUMEL = 1_000_000  # >1M 视为大权重

        def _set_module_attr(root_mod: nn.Module, dotted: str, value):
            """根据 "a.b.c" 定位到父模块并设置属性"""
            parts = dotted.split(".")
            parent = root_mod
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], value)

        if use_raw_ssd:
            print("[INFO] raw-ssd mode: replacing large weights with 0-size CPU stubs...")
            stub_count = 0
            keep_count = 0
            core_kept = []
            with torch.no_grad():
                for name, p in list(self.model.named_parameters()):
                    # 核心模块（embed_tokens, norm, output）必须保留，无论大小
                    is_core = _CORE_PAT.search(name) is not None
                    # 小参数（norm, bias）保留
                    is_small = _SMALL_PAT.search(name) and p.numel() <= _MAX_SAFE_NUMEL

                    if is_core or is_small:
                        keep_count += 1
                        if is_core:
                            core_kept.append(f"{name}: {p.shape}")
                        continue  # 保留完整权重

                    # 大权重：替换为 0-size CPU stub
                    stub = torch.empty(0, dtype=p.dtype, device="cpu")
                    new_p = nn.Parameter(stub, requires_grad=False)
                    _set_module_attr(self.model, name, new_p)
                    stub_count += 1
            print(f"[INFO] Replaced {stub_count} large weights with CPU stubs, kept {keep_count} core/small params")
            print(f"[INFO] Core modules kept: {core_kept[:5]}")  # 只打印前5个

        # ---------- 4) 所有模型已在 CPU 上，无需 meta 兜底逻辑 ----------
        # 移除了 meta device 相关的兜底代码

    

    def _configure_preload_mode(self, preload_config: dict):
        """
        Preload all weights to GPU, then use prefetch during inference.
        """
        print("🚀 Configuring Preload Mode...")

        config = {
            'max_layers_in_gpu': 4,  # 同时在GPU中保持的层数
            'prefetch_next': True,   # 是否在计算时预取下一层
            'verbose': True,
        }
        config.update(preload_config)

        try:
            print(f"📦 Preloading {config['max_layers_in_gpu']} layers to GPU...")

            # 预加载前几层到GPU
            if hasattr(self.model, 'layer_infos') and self.model.layer_infos:
                loaded_count = 0
                for i, layer_info in enumerate(self.model.layer_infos[:config['max_layers_in_gpu']]):
                    if layer_info.block is not None:
                        print(f"  Loading layer {i} to GPU...")
                        layer_info.block = layer_info.block.to(self.args.device)
                        loaded_count += 1

                print(f"✅ Successfully preloaded {loaded_count} layers to GPU")

                # 设置预取配置
                self.model.preload_config = config

            else:
                print("⚠️  No layer_infos found, falling back to full model loading")
                self.model = self.model.to(self.args.device)

        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ GPU OOM during preloading: {e}")
            print("💡 Consider reducing max_layers_in_gpu or using weight streaming")
            raise
        
    # --- paste into llama3/generator.py (inside class LLaMA) ---

    def _configure_weight_streaming(self, streaming_config: dict):
        """
        Weight streaming：激活 6 条流，保持激活在 HBM，按层流式权重到 GPU。
        """
        print("🔧 Configuring Weight Streaming...")
        from .weight_streaming_manager import WeightStreamingManager
        from .stream_mnt import get_streams

        config = {'prefetch_distance': 4, 'max_cached_layers': 4, 'warmup_layers': 2, 'verbose': True}
        config.update(streaming_config or {})

        # 小模块常驻 HBM，并保证 RoPE 等设备/精度就位
        self._configure_core_components()

        # 6 流
        self.streams = get_streams(self.args.device)

        # 建立 WSM 并注入到层（attn/ffn）
        wsm = WeightStreamingManager(
            self.model, device=self.args.device,
            prefetch_distance=config['prefetch_distance'],
            max_cached_layers=config['max_cached_layers'],
            warmup_layers=config['warmup_layers'],
            verbose=config['verbose'],
        )
        self.weight_streaming_manager = wsm
        self._integrate_wsm_to_layers(wsm, self.streams)

        # 🔥 Warm up a few groups (non-blocking) to hide first-layer H2D
        try:
            if hasattr(wsm, "warmup_groups_prefetch"):
                wsm.warmup_groups_prefetch(layers=config.get('warmup_layers', 2), blocking_first=False)
        except Exception as _e:
            print(f"[GEN] warmup_groups_prefetch skipped: {_e}")

        # KV 流
        self._configure_kv_streams()

        # 再次检查各组件设备一致性（避免 CPU/GPU 混放）
        self._verify_and_fix_device_placement()
        print("✅ Weight streaming enabled.")
        return wsm

    def _configure_ssd_streaming(self, ssd_config: dict):
        """
        SSD → CPU(pinned) → GPU 的混合 streaming 管线；同样激活 6 条流并注入层与 KV。
        """
        print("🚀 Configuring SSD Hybrid Streaming...")
        from pathlib import Path
        from .weight_streaming_manager import WeightStreamingManager
        import llama3.stream_mnt as stream_mnt

        config = {
            'ssd_manifest_path': None, 'prefetch_distance': 2, 'max_cached_layers': 4,
            'cpu_cache_layers': 50, 'staging_mb': 64, 'warmup_layers': 2, 'verbose': True,
        }
        config.update(ssd_config or {})
        if not config['ssd_manifest_path'] or not Path(config['ssd_manifest_path']).exists():
            raise FileNotFoundError("ssd_manifest_path missing or not exists")

        # 小模块常驻 HBM
        self._configure_core_components()

        # 6 流
        self.streams = stream_mnt.get_streams(self.args.device)

        # 创建 WSM（SSD backend）
        wsm = WeightStreamingManager(
            self.model, device=self.args.device,
            prefetch_distance=config['prefetch_distance'],
            max_cached_layers=config['max_cached_layers'],
            warmup_layers=config['warmup_layers'],
            verbose=config['verbose'],
            ssd_manifest_path=config['ssd_manifest_path'],
            cpu_cache_layers=config['cpu_cache_layers'],
            staging_mb=config['staging_mb'],
        )
        self.weight_streaming_manager = wsm
        self._integrate_wsm_to_layers(wsm, self.streams)

        # 🔥 Warm up a few groups (non-blocking) to hide first-layer H2D
        try:
            if hasattr(wsm, "warmup_groups_prefetch"):
                wsm.warmup_groups_prefetch(layers=config.get('warmup_layers', 2), blocking_first=False)
        except Exception as _e:
            print(f"[GEN] warmup_groups_prefetch skipped: {_e}")
        self._configure_kv_streams()
        self._verify_and_fix_device_placement()
        print("✅ SSD Hybrid Streaming enabled.")
        return wsm

    def _configure_core_components(self):
        """
        Keep small/core modules (embeddings, output head, final norm) permanently on the target device.
        ★ 同时设置 model.device 和 model.param_dtype，让下游推断有据可依
        """
        device = self.args.device
        model = self.model

        print(f"[_configure_core_components] Target device: {device}")

        # ★ 立即规定目标设备/精度（一次性做，不会搬大权重）
        if device.startswith("cuda"):
            dev = torch.device(device)
            model.device = dev                    # ★ 让下游推断有据可依
            model.param_dtype = torch.bfloat16    # ★ 统一精度

        # Keep small modules resident in HBM (fast, avoids repeated transfers)
        # 直接将核心组件移到目标设备（无需 meta 检查）
        model.embed_tokens = model.embed_tokens.to(device)
        model.norm = model.norm.to(device)
        model.output = model.output.to(device)

        # ⭐ Sanity check：打印核心模块设备
        print(f"[_configure_core_components] embed_tokens.weight.device: {model.embed_tokens.weight.device}")
        print(f"[_configure_core_components] norm.weight.device: {model.norm.weight.device}")
        print(f"[_configure_core_components] output.weight.device: {model.output.weight.device}")

        # Handle RoPE frequencies tensor/device placement
        self._handle_freqs_complex(device)

    def _handle_freqs_complex(self, device: str):
        """
        Ensure `freqs_complex` tensor lives on the correct device.
        If a direct `.to(device)` fails (e.g. stored as different type or shape),
        attempt to re-create it from model args.
        """
        model = self.model

        if hasattr(model, "freqs_complex"):
            try:
                model.freqs_complex = model.freqs_complex.to(device)
                print(f"[_handle_freqs_complex] freqs_complex moved to {model.freqs_complex.device}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to move freqs_complex to {device}: {e}")
                self._recreate_freqs_complex(device)

    def _recreate_freqs_complex(self, device: str):
        """
        Rebuild `freqs_complex` on the requested device using model args.
        NOTE: local import by design; avoids importing `layers` at module import time.
        """
        try:
            from .layers import precompute_theta_pos_frequencies  # local import intentionally kept
            print(f"   Attempting to recreate freqs_complex on {device}...")

            dim = self.args.dim
            n_heads = self.args.n_heads
            max_seq_len = self.args.max_seq_len
            rope_theta = self.args.rope_theta

            self.model.freqs_complex = precompute_theta_pos_frequencies(
                dim // n_heads,
                max_seq_len * 2,
                device=device,
                theta=rope_theta,
            )
            print(f"   Successfully recreated freqs_complex on {device}")
        except Exception as e:
            print(f"   Failed to recreate freqs_complex: {e}")
            raise RuntimeError(f"Cannot ensure freqs_complex is on {device}") from e


    def _configure_kv_streams(self):
        """
        Configure the H2D/D2H streams for KV offloader (if any) on each layer's attention module.
        """
        try:
            from . import stream_mnt  # local import intentionally kept
            streams = stream_mnt.get_streams(self.args.device)

            if hasattr(self.model, "layers"):
                first_off = None
                for layer in self.model.layers:
                    if hasattr(layer, "attention"):
                        off = getattr(layer.attention, "offloader", None)
                        if off is not None:
                            off.h2d_stream = streams.kv_h2d
                            off.d2h_stream = streams.kv_d2h
                            if first_off is None:
                                first_off = off
                # # 将 KV Offloader 注入 WSM，启用“忙态暂停写”
                # if first_off is not None and hasattr(self, "weight_streaming_manager"):
                #     try:
                #         self.weight_streaming_manager.kv_offloader = first_off
                #     except Exception:
                #         pass
                # ★ 新增：统一为单例（跨层共享一个 KVOffloader 实例）
                
                if first_off is not None:
                    for blk in getattr(self.model, "layers", []):
                        if getattr(blk.attention, "offloader", None) is not first_off:
                            blk.attention.offloader = first_off
                print("✅ KV offloader unified across layers; KV streams configured.")
        except Exception as e:
            print(f"[WARN] failed to configure KV streams: {e}")

    def _verify_and_fix_device_placement(self):
        """
        Double-check that all key components and per-layer norms live on the intended CUDA device.
        """
        device = self.args.device
        model = self.model

        if not device.startswith("cuda"):
            return

        print("🔍 Verifying device placement before inference...")
        try:
            # Force-sync core modules to target device
            print("🔧 Synchronizing all components to target device...")
            # 直接移动模块到目标设备（无需 meta 检查）
            model.embed_tokens = model.embed_tokens.to(device)
            model.norm = model.norm.to(device)
            model.output = model.output.to(device)

            if hasattr(model, 'freqs_complex'):
                model.freqs_complex = model.freqs_complex.to(device)

            # Sync per-layer norms to GPU
            print("🔧 Synchronizing layer norms to GPU...")
            if hasattr(model, "layers"):
                for layer in model.layers:
                    if hasattr(layer, 'attention_norm'):
                        layer.attention_norm = layer.attention_norm.to(device)
                    if hasattr(layer, 'ffn_norm'):
                        layer.ffn_norm = layer.ffn_norm.to(device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # ⭐⭐⭐ 强制把 embedding / norm / output 固定在 CUDA，并做断言式校验
            print("🔒 Forcing critical modules to CUDA with assertion validation...")
            m = self.model
            dev = torch.device(device)

            # 1) 强制常驻 CUDA 的小模块（确保即使前面有失败，这里也会重试）
            m.embed_tokens = m.embed_tokens.to(dev)
            m.norm         = m.norm.to(dev)
            m.output       = m.output.to(dev)

            # 2) 断言式校验（出问题直接抛，早失败）
            assert m.embed_tokens.weight.is_cuda, \
                f"embed_tokens.weight must be on CUDA, got {m.embed_tokens.weight.device}"

            for name, mod in (("norm", m.norm), ("output", m.output)):
                for pname, p in mod.named_parameters(recurse=True):
                    assert p.is_cuda, \
                        f"{name}.{pname} must be on CUDA, got {p.device}"

            print("✅ All critical modules (embed_tokens, norm, output) are on CUDA and validated")
            print("✅ All layer components synchronized to target device")
        except Exception as e:
            print(f"⚠️ Error during device synchronization: {e}")
            raise  # Re-raise to fail-fast if critical modules can't be placed on CUDA

        # 取得真实计算设备（embed_tokens 已在 CUDA）
        gpu_dev = str(self.model.embed_tokens.weight.device)  # e.g. "cuda:0"
        self.args.device = gpu_dev            # 让后续逻辑看到一致的设备
        self.model.device = torch.device(gpu_dev)

        # 把每层与子模块的 runtime device 改成 CUDA
        if hasattr(self.model, "layers"):
            for layer in self.model.layers:
                if hasattr(layer, "device"):
                    layer.device = gpu_dev
                if hasattr(layer, "attention"):
                    layer.attention.device = gpu_dev
                    off = getattr(layer.attention, "offloader", None)
                    if off is not None:
                        off.device = gpu_dev      # ★ 关键：让 fetch() 把 KV 拉到 GPU
                        if hasattr(off, "_ssd_buffer"):
                            off._ssd_buffer = None  # 若之前按 CPU 创过，丢弃，待下一次按 CUDA 重建
                if hasattr(layer, "feed_forward"):
                    layer.feed_forward.device = gpu_dev
          
    def _prime_kv_for_first_decode_step(self, prefill_len: int, bsz: int):
        """
        在进入解码前，给每一层把 (start_pos=prefill_len, seqlen=1) 会命中的 KV blocks 预取一下。
        这样 L0.attn 的第一步就能 100% 命中 GPU 暂存。
        """
        layers = getattr(self.model, "layers", [])
        for lid, blk in enumerate(layers):
            attn = getattr(blk, "attention", None)
            off  = getattr(attn, "offloader", None)
            if off is None:
                continue
            # 计算首步需要的 blocks（默认窗口=BLOCK）
            blocks = off.plan_tail_window_blocks(prefill_len, 1)  # seqlen=1 for decode step-0
            if not blocks:
                continue
            try:
                s = getattr(self.streams, "kv_h2d", None)
                if hasattr(off, "prefetch_blocks_async"):
                    off.prefetch_blocks_async(lid, blocks, bsz=bsz, stream=s)
                else:
                    off.prefetch_async(layer=lid, blocks=blocks, bsz=bsz, device=self.args.device)
            except Exception:
                # 预取失败不应阻断推理流程
                pass
  

    # ---------- Build ----------
    @staticmethod
    def build(
        checkpoints_dir: str,
        load_model: bool = True,
        device: str = "cuda",
        mode: Optional[str] = None,   # "ssd" | "stream" | "preload" | "full"
        mode_config: Optional[dict] = None,
        enable_weight_streaming: bool = False,
        streaming_config: Optional[dict] = None,
        enable_preload_mode: bool = False,
        preload_config: Optional[dict] = None,
        enable_ssd_streaming: bool = False,
        ssd_streaming_config: Optional[dict] = None,
        topk_blk: Optional[int] = None,
        max_seq_len: int = 2048,
        max_batch_size: int = 512,
    ) -> "LLaMA":
        """
        Build a LLaMA instance:
        1) Load tokenizer from checkpoint dir.
        2) Load params.json into ModelArgs.
        3) (Optional) Load .pth weights to CPU state_dict.
        4) Weight loading strategies:
           - enable_ssd_streaming: SSD raw device + CPU cache + GPU streaming (NEW)
           - enable_preload_mode: Preload N layers to GPU, prefetch during compute
           - enable_weight_streaming: Original streaming mode
           - default: Full model to GPU
        """
        ckpt_dir = Path(checkpoints_dir)

        # ---- Tokenizer ----
        tokenizer = None
        try:
            # Directly load tokenizer from folder
            tokenizer = LlamaTokenizerFast.from_pretrained(
                pretrained_model_name_or_path=ckpt_dir, legacy=True
            )
            print(f"[INFO] Loaded tokenizer with LlamaTokenizerFast")
        except Exception as e:
            print(f"[WARNING] Failed to load tokenizer with LlamaTokenizerFast from {ckpt_dir}: {e}")
            # Keep the original lazy import pattern in the fallback for parity with user's code
            try:
                from transformers import AutoTokenizer  # local (redundant) import intentionally kept
                tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, legacy=True)
                print(f"[INFO] Loaded tokenizer with AutoTokenizer")
            except Exception as e2:
                print(f"[ERROR] Failed to load tokenizer with AutoTokenizer: {e2}")
                raise RuntimeError(f"Failed to load tokenizer: {e}, {e2}")

        if tokenizer is None:
            raise RuntimeError("Failed to initialize tokenizer")

        # ---- Load params.json ----
        params_path = ckpt_dir / "params.json"
        args = ModelArgs.from_json(
            str(params_path), max_seq_len=max_seq_len, max_batch_size=max_batch_size, device=device
        )

        # Allow overriding top-k block selection args
        if topk_blk is not None:
            args.topk_blk = topk_blk

        args.checkpoints_dir = str(ckpt_dir)

        # ---- Determine if using raw-ssd mode ----
        # use_raw_ssd = (mode in {"ssd", "mixed"}) or (mode_config and mode_config.get("weight_source") == "raw-ssd")
        use_raw_ssd = (mode in {"mixed"}) or (mode_config and mode_config.get("weight_source") == "raw-ssd")


        # ---- Load checkpoint weights to CPU (optional) ----
        checkpoint = None
        # 重要：raw-ssd 模式下不加载 checkpoint，避免把整包权重吃进 CPU
        if load_model and not use_raw_ssd:
            ckpt_file = sorted(ckpt_dir.glob("*.pth"))[0]
            print(f"[INFO] Loading checkpoint: {ckpt_file}")
            t0 = time.time()
            checkpoint = torch.load(ckpt_file, map_location="cpu")
            print(f"[INFO] Done ({time.time() - t0:.1f}s)")

        # ---- Build model on CPU (统一使用 CPU stub，不用 meta device) ----
        cpu_args = ModelArgs.from_json(
            str(params_path), max_seq_len=max_seq_len, max_batch_size=max_batch_size, device="cpu"
        )

        if use_raw_ssd:
            setattr(cpu_args, "weight_source", "raw-ssd")
            # 启用 stub 参数模式，避免在 CPU 上分配 70B 权重内存
            setattr(cpu_args, "use_stub_params", True)

        if topk_blk is not None:
            cpu_args.topk_blk = topk_blk
        cpu_args.checkpoints_dir = str(ckpt_dir)

        llama = LLaMA(tokenizer, checkpoint, cpu_args)

        # ---- raw-ssd 模式：CPU stub + SSD streaming ----
        if use_raw_ssd:
            print("[INFO] raw-ssd mode: large weights replaced with CPU stubs, streaming from SSD...")

            # 模型已在 CPU 上，无需 to_empty 物化
            # RoPE freqs_complex 已在 Transformer.__init__ 中创建，无需重新计算

            # 初始化 WSM 的 SSD 后端（传 raw block device + manifest）
            from .weight_streaming_manager import WeightStreamingManager

            # Ensure model.layers accessible
            if hasattr(llama.model, "layer_infos"):
                try:
                    blocks = [info.block for info in llama.model.layer_infos if info.block is not None]
                    if blocks and not hasattr(llama.model, "layers"):
                        llama.model.layers = blocks
                except Exception:
                    pass

            cfg = mode_config or {}
            wsm = WeightStreamingManager(
                model=llama.model,
                device=device,
                prefetch_distance=cfg.get("prefetch_distance", 1),
                max_cached_layers=cfg.get("max_cached_layers", 3),
                warmup_layers=cfg.get("warmup_layers", 1),
                verbose=cfg.get("verbose", False),
                monitor_fragmentation=False,
                ssd_manifest_path=cfg.get("ssd_manifest_path") or cfg.get("manifest_path"),
                cpu_cache_layers=cfg.get("cpu_cache_layers", 40),  # ← 默认 40
                staging_mb=cfg.get("staging_mb", 64),
            )

            llama.weight_streaming_manager = wsm

            # 用 SSD manifest 把"resident 模块"加载到目标设备（CPU->GPU 按需）
            # 这里只加载 embed/norm/output/bias；大权重按层 on-demand。
            if hasattr(wsm, "load_resident_from_ssd"):
                wsm.load_resident_from_ssd(llama.model, target_device=device)

            # Integrate WSM hooks into layers
            from .stream_mnt import get_streams
            llama.streams = get_streams(device)
            llama._integrate_wsm_to_layers(wsm, llama.streams)
            # 🔥 Initial non-blocking warmup of first few groups
            try:
                if hasattr(wsm, 'warmup_groups_prefetch'):
                    wsm.warmup_groups_prefetch(layers=cfg.get('warmup_layers', 2), blocking_first=False)
            except Exception as _e:
                print(f"[GEN] warmup_groups_prefetch skipped: {_e}")

            # Configure KV streams
            llama._configure_kv_streams()

            # Verify device placement
            llama._verify_and_fix_device_placement()
            
            cuda_dev = device if str(device).startswith("cuda") else str(llama.model.embed_tokens.weight.device)
            
            llama.args.device = str(cuda_dev)
            # import torch
            setattr(llama.model, "device", torch.device(cuda_dev))
            setattr(llama.model, "param_dtype", torch.bfloat16)
            
            # 每一层（以及其子模块）都要把 .device 设成 CUDA
            for blk in getattr(llama.model, "layers", []):
                blk.device = llama.args.device
                if hasattr(blk, "attention"):    blk.attention.device    = llama.args.device
                if hasattr(blk, "feed_forward"): blk.feed_forward.device = llama.args.device
                
            # RoPE 频率张量也要跟上
            llama._handle_freqs_complex(llama.args.device)
            
            llama._verify_and_fix_device_placement()

            print("✅ Weight streaming enabled (SSD -> CPU(pinned) -> GPU by layer)")

        # ---- Weight loading strategies (统一入口) ----
        elif mode is not None:
            m = (mode or "").lower()
            cfg = mode_config or {}
            if m == "ssd" and device.startswith("cuda"):
                llama._configure_ssd_streaming(cfg)
            elif m == "stream" and device.startswith("cuda"):
                llama._configure_weight_streaming(cfg)
            elif m == "preload" and device.startswith("cuda"):
                llama._configure_preload_mode(cfg)
            elif m == "full" and device.startswith("cuda"):
                try:
                    # ★ 先设置 device 和 param_dtype，移动核心组件
                    llama._configure_core_components()
                    # 再移动整个模型到 GPU
                    llama.model = llama.model.to(device)
                    llama.args.device = device
                    # 仅当已经在 CUDA 上，才进行半精度转换
                    llama.model = llama.model.half()
                except torch.cuda.OutOfMemoryError:
                    print("❌ CUDA OOM when moving model. Keeping on CPU...")
                    device = "cpu"
                    llama.args.device = "cpu"
            else:
                # fallback to old flags
                if enable_ssd_streaming and device.startswith("cuda"):
                    llama._configure_ssd_streaming(ssd_streaming_config or {})
                elif enable_preload_mode and device.startswith("cuda"):
                    llama._configure_preload_mode(preload_config or {})
                elif enable_weight_streaming and device.startswith("cuda"):
                    llama._configure_weight_streaming(streaming_config or {})
                elif device.startswith("cuda"):
                    try:
                        # ★ 先设置 device 和 param_dtype，移动核心组件
                        llama._configure_core_components()
                        # 再移动整个模型到 GPU
                        llama.model = llama.model.to(device)
                        llama.args.device = device
                        # 传统全量加载模式下，GPU 就绪后再 half
                        llama.model = llama.model.half()
                    except torch.cuda.OutOfMemoryError:
                        print("❌ CUDA OOM when moving model. Keeping on CPU...")
                        device = "cpu"
                        llama.args.device = "cpu"
        elif device.startswith("cuda"):

            # 传统的全量加载模式
            try:
                # ★ 先设置 device 和 param_dtype，移动核心组件
                llama._configure_core_components()
                # 再移动整个模型到 GPU
                llama.model = llama.model.to(device)
                llama.args.device = device
                # 默认路径：GPU 上再 half
                llama.model = llama.model.half()
            except torch.cuda.OutOfMemoryError:
                print("❌ CUDA OOM when moving model. Keeping on CPU...")
                device = "cpu"
                llama.args.device = "cpu"

        return llama

    # ---------- Inference ----------
    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        profile_output_dir: Optional[str] = None,
        batch_size: int = 4,
        enable_batching: bool = True,
    ):
        """
        Greedy/Top-p generation with simple batching and optional KV profiling.
        - Uses `pad_token_id` for padding (falls back to `eos_token_id` if none).
        - Tracks per-token times/bytes for KV cache (rough estimate).

        Args:
            enable_batching: If False, processes all prompts in a single batch (no "local X/Y" display)
        """
        nvtx.range_push("text_completion")

        # Wait for preload completion if streaming is enabled
        if hasattr(self, 'weight_streaming_manager'):
            wsm = self.weight_streaming_manager

            if hasattr(wsm, 'wait_for_preload_ready'):
                streaming_mode = getattr(self, '_streaming_mode', 'weight_streaming')
                print(f"[INFO] Waiting for preload completion in {streaming_mode} mode (target: {wsm.target_gpu_layers} GPU + {wsm.target_cpu_layers} CPU layers)...")
                # preload_success = wsm.wait_for_preload_ready(timeout=300.0)
                import os
                if os.getenv("WSM_SKIP_PRELOAD_WAIT", "0") == "1":
                    print(f"[INFO] Skipping WSM preload wait due to WSM_SKIP_PRELOAD_WAIT=1")
                    preload_success = True
                else:
                    preload_success = wsm.wait_for_preload_ready(timeout=300.0)
                if preload_success:
                    print(f"✅ [INFO] Preload completed successfully")
                else:
                    print(f"⚠️ [WARNING] Preload timeout - proceeding with inference anyway")
            else:
                print(f"[INFO] WSM found but no preload method available")

        # Disable batching if requested
        if not enable_batching:
            batch_size = len(prompts)

        num_batches = (len(prompts) + batch_size - 1) // batch_size

        # Try to register batches in global tracker (best-effort)
        from .global_state_tracker import get_global_tracker  # keep original code behavior
        tracker = get_global_tracker()
        if tracker:
            actual_batches = list(range(num_batches))
            # Only register if empty to avoid overwriting existing schedules
            if not tracker.future_batches:
                tracker.register_future_batch(actual_batches)
                print(
                    f"[INFO] Registered {num_batches} batches for {len(prompts)} prompts "
                    f"(batch_size={batch_size}): {actual_batches}"
                )
        else:
            print("[WARNING] Global tracker not found during batch registration")

        # Expand max_batch_size if needed to cover current request
        self.args.max_batch_size = max(self.args.max_batch_size, len(prompts))

        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1

        # ---- Tokenize ----
        nvtx.range_push("tokenization")
        prompts_tok = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        nvtx.range_pop()  # tokenization

        # Storage for outputs and KV profile
        all_out_tokens, all_out_text = [], []
        kv_profile = []

        # ---- Per-batch generation ----
        for batch_idx in range(num_batches):
            try:
                # Index range of current batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(prompts_tok))
                batch_prompts = prompts_tok[start_idx:end_idx]

                # Prefer global tracker progress display if available
                try:
                    if tracker and tracker.future_batches:
                        global_batch_idx = tracker.current_batch
                        total_global_batches = len(tracker.future_batches)
                        print(
                            f"[INFO] Processing batch {global_batch_idx + 1}/{total_global_batches} "
                            f"with {len(batch_prompts)} prompts"
                        )
                    else:
                        print(f"[INFO] Processing batch {batch_idx + 1}/{num_batches} with {len(batch_prompts)} prompts")
                except Exception:
                    print(f"[INFO] Processing batch {batch_idx + 1}/{num_batches} with {len(batch_prompts)} prompts")

                # Shape planning
                bsz = len(batch_prompts)
                max_prompt = max(len(x) for x in batch_prompts)
                total_len = min(self.args.max_seq_len, max_gen_len + max_prompt)

            except Exception as e:
                print(f"❌ Error during batch {batch_idx + 1} initialization: {e}")
                continue

            try:
                # Pad with tokenizer.pad_token_id if available; otherwise fallback to eos
                pad_id = (
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                )

                # # Input token tensor: (bsz, total_len), pre-filled with pad
                # tokens = torch.full(
                #     (bsz, total_len),
                #     pad_id,
                #     dtype=torch.long,
                #     device=self.args.device,
                # )

                # Write prompt tokens at the front of each row
                # for i, tok in enumerate(batch_prompts):
                #     tokens[i, : len(tok)] = torch.tensor(tok, device=self.args.device)

                # Masks
                # eos_mask = torch.zeros(bsz, dtype=torch.bool, device=self.args.device)  # track finished sequences
                # prompt_mask = tokens != pad_id  # True where original prompt tokens exist

                # ⭐⭐⭐ 强烈建议：tokens 在入口就上卡，避免后面 CPU tensor 走到 forward
                # 使用 embed_tokens 的实际设备（已在 _verify_and_fix_device_placement 中确保在 CUDA）
                dev = getattr(self.model, "device", None)
                if dev is None:
                    try:
                        dev = str(self.model.embed_tokens.weight.device)
                    except Exception:
                        dev = self.args.device
                dev = str(dev)

                # ⭐ 关键：直接在目标设备上创建 tokens，non_blocking=True
                tokens = torch.full(
                    size=(bsz, total_len),
                    fill_value=pad_id,
                    dtype=torch.long,
                    device=dev,  # 直接在 CUDA 上创建
                )

                # ⭐ 写入 prompt tokens 时也确保在 CUDA 上
                for i, tok in enumerate(batch_prompts):
                    # 直接在目标设备上创建 tensor，避免 CPU->GPU 拷贝
                    tokens[i, : len(tok)] = torch.tensor(tok, dtype=torch.long, device=dev)

                eos_mask = torch.zeros(bsz, dtype=torch.bool, device=dev)
                prompt_mask = tokens != pad_id

                # ⭐ 入口断言：确保 tokens 在 CUDA 上（早失败）
                if not tokens.is_cuda:
                    raise RuntimeError(
                        f"[text_completion] tokens must be on CUDA, got {tokens.device}. "
                        f"Check device placement in _verify_and_fix_device_placement."
                    )

            except torch.cuda.OutOfMemoryError as e:
                print(f"❌ CUDA OOM during batch {batch_idx + 1} tensor allocation: {e}")
                torch.cuda.empty_cache()
                continue
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"❌ CUDA error during batch {batch_idx + 1} tensor allocation: {e}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
                
            # ========= ① Prefill：一次性跑完整提示词（或至少跑到 max_prompt）=========
            prefill_len = max_prompt
            if prefill_len > 0:
                nvtx.range_push("prefill_phase")
                try:
                    with torch.no_grad():
                        # _ = self.model(tokens[:, :prefill_len], start_pos=0)
                        # tchunk = int(os.getenv("PREFILL_T_CHUNK", "0"))
                        # 仅当环境变量设为正整数时启用分块；默认一次性 prefill
                        tchunk_env = os.getenv("PREFILL_T_CHUNK", "0").strip()
                        tchunk = int(tchunk_env) if tchunk_env.isdigit() else 0
                        if tchunk and prefill_len > tchunk:
                            for s in range(0, prefill_len, tchunk):
                                e = min(prefill_len, s + tchunk)
                                _ = self.model(tokens[:, s:e], start_pos=s)
                        else:
                            _ = self.model(tokens[:, :prefill_len], start_pos=0)
                except torch.cuda.OutOfMemoryError as e:
                    print(f"❌ CUDA OOM during prefill of batch {batch_idx + 1}: {e}")
                    torch.cuda.empty_cache()
                    nvtx.range_pop()  # prefill_phase (error case)
                    raise RuntimeError("GPU out of memory during prefill") from e
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        print(f"❌ CUDA error during prefill of batch {batch_idx + 1}: {e}")
                        torch.cuda.empty_cache()
                        nvtx.range_pop()  # prefill_phase (error case)
                        raise RuntimeError("CUDA error during prefill") from e
                    else:
                        nvtx.range_pop()  # prefill_phase (error case)
                        raise
                nvtx.range_pop()  # prefill_phase

            # ========= Prefill→Decode 切换：原子切换到 decoder 模式 =========
            # 旧版实现：仅调用 prime_decode_window() 来预热解码窗口
            # 新版实现：调用 enter_decode_mode() 进行原子切换，包括：
            #   1) 切换 prefetch distance（从 prefill 阶段的值切到 decoder 阶段的值）
            #   2) 设置保护窗口（保护前 N 层，避免被 wrap-around 驱逐逻辑立刻踢出）
            #   3) 启用 CPU 环窗模式（若使用 SSD 流式，确保始终有下一批层在 CPU 中准备）
            #   4) 调用 prime_decode_window 预热解码窗口（同时启动 attn/ffn 双路 H2D）
            # 这样可避免"刚 prime 就被驱逐"或"CPU 层还在路上"导致的首 token 抖动
            if hasattr(self.model, 'weight_streaming_manager') and self.model.weight_streaming_manager is not None:
                try:
                    wsm = self.model.weight_streaming_manager
                    # 1) 优先使用新版原子切换方法
                    if hasattr(wsm, 'enter_decode_mode'):
                        import os
                        # WSM_DECODER_PROTECT_LAYERS: 保护前 N 层不被驱逐（默认 6 层）
                        protect = int(os.getenv("WSM_DECODER_PROTECT_LAYERS", "40"))
                        # WSM_PRIME_WINDOW: 预热窗口大小（默认 6 层）
                        prime_w = int(os.getenv("WSM_PRIME_WINDOW", "6"))
                        wsm.enter_decode_mode(first_layer=0, protect_layers=protect, prime_window=prime_w)
                        self._prime_kv_for_first_decode_step(prefill_len=prefill_len, bsz=int(tokens.size(0)))
                        # ⭐ 解码前的"首层屏障"：显式确保 L0 事件已就绪
                        # 这能把"偶发行程波动"变成确定性等待
                        if hasattr(wsm, 'wait_group_ready'):
                            wsm.wait_group_ready(0, 'attn')  # wait_group_ready 内部已取模
                            wsm.wait_group_ready(0, 'ffn')
                    else:
                        # 2) 兼容旧版：如果 enter_decode_mode 不存在，回退到仅调用 prime_decode_window
                        if hasattr(wsm, 'prime_decode_window'):
                            import os
                            wsm.prime_decode_window(first_layer=0, window=int(os.getenv("WSM_PRIME_WINDOW", "6")))
                            self._prime_kv_for_first_decode_step(prefill_len=prefill_len, bsz=int(tokens.size(0)))
                            # ⭐ 解码前的"首层屏障"（兼容旧版）
                            if hasattr(wsm, 'wait_group_ready'):
                                wsm.wait_group_ready(0, 'attn')
                                wsm.wait_group_ready(0, 'ffn')
                except Exception as e:
                    if getattr(self.model.weight_streaming_manager, 'verbose', False):
                        print(f"[WSM][enter_decode] failed: {e}")

            # Build progress bar description with global tracker if available
            try:
                tracker = get_global_tracker()
                if tracker and hasattr(tracker, 'current_batch') and tracker.current_batch is not None:
                    global_batch_num = tracker.current_batch + 1
                    total_global_batches = (
                        len(tracker.future_batches) if hasattr(tracker, 'future_batches') else 'Unknown'
                    )
                    desc = f"Generating tokens for batch {global_batch_num}/{total_global_batches} (local {batch_idx + 1}/{num_batches})"
                else:
                    desc = f"Generating tokens for batch {batch_idx + 1}/{num_batches}"
            except Exception:
                desc = f"Generating tokens for batch {batch_idx + 1}/{num_batches}"


            # ========= ② Decode：从 prefill_len 开始单步生成 =========
            start_decode = prefill_len  # 第一轮 decode 读的是 tokens[:, prefill_len-1:prefill_len]

            # 重置 CUDA 峰值内存统计（用于监控 batch=2 的显存峰值）
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # ---- Token-by-token decode loop ----
            for cur_pos in tqdm(range(start_decode, total_len), desc=desc):
                nvtx.range_push(f"token_{cur_pos}_generation")
                try:
                    # 1) Forward last token for each row
                    nvtx.range_push(f"token_{cur_pos}_forward")
                    with torch.no_grad():
                        logits = self.model(tokens[:, cur_pos - 1: cur_pos], cur_pos)
                    nvtx.range_pop()  # forward

                    # 2) Sampling / argmax
                    nvtx.range_push(f"token_{cur_pos}_sampling")
                    if temperature > 0:
                        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                        next_tok = self._sample_top_p(probs, top_p)
                    else:
                        next_tok = torch.argmax(logits[:, -1], dim=-1)

                    next_tok = next_tok.reshape(-1)

                    # Respect prompt region: keep original token if still in prompt
                    next_tok = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_tok)

                    # 3) Write back
                    tokens[:, cur_pos] = next_tok

                    # 4) EOS tracking (only matters outside prompt)
                    eos_mask |= (~prompt_mask[:, cur_pos]) & (next_tok == self.tokenizer.eos_token_id)
                    nvtx.range_pop()  # sampling

                    # 5) Early break if all finished
                    if eos_mask.all():
                        nvtx.range_pop()  # token_generation
                        break

                except torch.cuda.OutOfMemoryError as e:
                    print(f"❌ CUDA OOM during inference at position {cur_pos}: {e}")
                    torch.cuda.empty_cache()
                    nvtx.range_pop()  # token_generation (error case)
                    raise RuntimeError("GPU out of memory during inference") from e
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        print(f"❌ CUDA error during inference at position {cur_pos}: {e}")
                        torch.cuda.empty_cache()
                        nvtx.range_pop()  # token_generation (error case)
                        raise RuntimeError("CUDA error during inference") from e
                    else:
                        nvtx.range_pop()  # token_generation (error case)
                        raise

                nvtx.range_pop()  # token_generation

                # ---- GPU 内存峰值监控（每 10 步或首/末步输出）----
                step = cur_pos - start_decode
                if torch.cuda.is_available() and (step % 10 == 0 or step == 0 or cur_pos == total_len - 1):
                    alloc = torch.cuda.max_memory_allocated() / (1024**2)
                    reserv = torch.cuda.max_memory_reserved() / (1024**2)
                    print(f"[mem] step={step} alloc={alloc:.1f} MiB, reserved={reserv:.1f} MiB")

                # ---- KV profile (rough estimate, same as original logic) ----
                kv_re_time = sum(self.model.kv_times)
                bytes_per_token = (
                    2 * self.model.args.n_kv_heads
                    * self.model.args.dim // self.model.args.n_heads
                    * self.model.embed_tokens.weight.element_size()
                )
                kv_bytes = bytes_per_token * cur_pos * self.model.args.n_layers
                kv_profile.append(
                    {
                        "batch_idx": batch_idx,
                        "token_idx": int(cur_pos),
                        "phase": "prefill" if cur_pos < max_prompt else "decode",
                        "kv_re_ms": float(kv_re_time),
                        "kv_kb": float(kv_bytes / 1024),
                    }
                )

            # ---- Decode current batch output ----
            for row in tokens.tolist():
                if self.tokenizer.eos_token_id in row:
                    row = row[: row.index(self.tokenizer.eos_token_id)]
                all_out_tokens.append(row)
                all_out_text.append(self.tokenizer.decode(row))

        # Final results
        out_tokens, out_text = all_out_tokens, all_out_text

        # ---- Save profiling if requested ----
        if profile_output_dir:
            os.makedirs(profile_output_dir, exist_ok=True)
            save_name = os.path.join(
                profile_output_dir, f"{Path(self.args.checkpoints_dir).name}_kv_profile.json"
            )
            with open(save_name, "w", encoding="utf-8") as f:
                json.dump(kv_profile, f, indent=2)
            print(f"[INFO] KV profile saved → {save_name}")

        return out_tokens, out_text

    def _integrate_wsm_to_layers(self, wsm, streams):
        """把 weight_manager / streams 注入到每一层（attn/ffn），并设置 layer_id。"""
        if not hasattr(self.model, "layers"):
            return
        for lid, block in enumerate(self.model.layers):
            # SelfAttention
            if hasattr(block, "attention"):
                block.attention.layer_id = lid
                block.attention.weight_manager = wsm
                block.attention.streams = streams
                block.attention.weight_h2d_stream = getattr(streams, "weight_h2d_mha", None)
            # FeedForward
            if hasattr(block, "feed_forward"):
                block.feed_forward.layer_id = lid
                block.feed_forward.weight_manager = wsm
                block.feed_forward.streams = streams
                block.feed_forward.weight_h2d_stream = getattr(streams, "weight_h2d_ffn", None)

    # ---------- Utils ----------
    @staticmethod
    def _sample_top_p(probs, p):
        """
        Nucleus (top-p) sampling on the last dimension of probs tensor.
        """
        sort_probs, sort_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sort_probs, dim=-1)
        sort_probs[cumsum - sort_probs > p] = 0.0
        sort_probs.div_(sort_probs.sum(dim=-1, keepdim=True))
        next_tok = torch.multinomial(sort_probs, 1)
        return torch.gather(sort_idx, -1, next_tok)

    def encode(self, text: str):
        """Convenience helper to encode with underlying tokenizer."""
        return self.tokenizer.encode(text)

    def decode(self, ids):
        """Convenience helper to decode with underlying tokenizer."""
        return self.tokenizer.decode(ids)
