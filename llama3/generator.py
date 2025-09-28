import json
import os
import time
from pathlib import Path
from typing import List, Optional
import torch
from tqdm import tqdm
from transformers import LlamaTokenizerFast, AutoTokenizer  
from .config import ModelArgs
from .model import Transformer

# ================================
# NVTX profiling support (safe fallback)
# ================================

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
    def __init__(self, tokenizer, checkpoint, args: ModelArgs):
        """
        Initialize model and (optionally) load checkpoint weights.
        Model is first constructed on the device specified in `args.device` (may be 'cpu' or 'cuda:X').
        """
        self.tokenizer = tokenizer
        self.args = args

        # (Optional) Global tracker init (left commented — keep original code)
        # from .global_state_tracker import init_global_tracker, get_global_tracker
        # from .kv_offload import BLOCK
        # if get_global_tracker() is None:
        #     print(f"[INFO] Initializing global state tracker...")
        #     n_blocks = (args.max_seq_len + BLOCK - 1) // BLOCK
        #     tracker = init_global_tracker(
        #         max_batch=args.max_batch_size,
        #         layers=args.n_layers,
        #         n_blocks=n_blocks
        #     )
        #     print(f"[INFO] Global state tracker initialized, waiting for actual batch registration")

        print(f"[INFO] Initializing model on device: {args.device}")
        self.model = Transformer(args)

        print(f"[INFO] Moving model to {args.device}...")
        self.model = self.model.to(args.device)

        print(f"[INFO] Converting to half precision...")
        self.model = self.model.half()

        if checkpoint is not None:
            print(f"[INFO] Loading state dict...")
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint, strict=False)
            if missing_keys:
                print(f"[WARNING] Missing keys: {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"[WARNING] Unexpected keys: {len(unexpected_keys)} keys")
            print(f"[INFO] Model weights loaded successfully")

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

    def _configure_weight_streaming(self, streaming_config: dict):
        """
        Enable weight streaming (keep activations on GPU, stream per-layer weights).
        NOTE: Intentionally keeps local imports to avoid circular deps and heavy eager imports.
        """
        print("🔧 Configuring Weight Streaming...")

        # Local imports intentionally kept (avoid circular imports / heavy startup)
        from .weight_streaming_manager import WeightStreamingManager
        from . import layers
        from . import stream_mnt

        # Default streaming config (merged with user-provided overrides)
        config = {
            'prefetch_distance': 1,
            'max_cached_layers': 4,
            'warmup_layers': 1,
            'verbose': False,
        }
        config.update(streaming_config)

        # Ensure model.layers is accessible when layer_infos is present (for WSM integration)
        if hasattr(self.model, "layer_infos"):
            try:
                blocks = [info.block for info in self.model.layer_infos if info.block is not None]
                if blocks and not hasattr(self.model, "layers"):
                    self.model.layers = blocks
            except Exception:
                # Swallow silently to preserve original behavior
                pass

        # Place small/core components on target device (kept resident in HBM)
        self._configure_core_components()

        # Create and wire up the WeightStreamingManager
        wsm = WeightStreamingManager(
            self.model,
            device=self.args.device,
            prefetch_distance=config['prefetch_distance'],
            max_cached_layers=config['max_cached_layers'],
            warmup_layers=config['warmup_layers'],
            verbose=True,  # force verbose to help verify integration
        )

        # Integrate WSM hooks into layers (attn/ffn)
        self._integrate_wsm_to_layers(wsm)

        # Configure KV streams if offloaders exist
        self._configure_kv_streams()

        # Verify and fix device placements to avoid accidental CPU/GPU mismatches
        self._verify_and_fix_device_placement()

        # Optional diagnostics
        try:
            first_blk = getattr(self.model, "layers", [None])[0]
            if first_blk is not None:
                print("[CHECK] first block param device:", next(first_blk.parameters()).device)
        except Exception:
            pass

        print("✅ Weight streaming enabled (activations on GPU, weights streamed per-layer).")
        print("⚙️  Running on GPU")
        return wsm

    def _configure_ssd_streaming(self, ssd_config: dict):
        """
        Enable SSD-backed hybrid weight streaming: SSD -> CPU cache -> GPU streaming.
        """
        from pathlib import Path
        print("🚀 Configuring SSD Hybrid Streaming...")

        # Local imports intentionally kept
        from .weight_streaming_manager import WeightStreamingManager
        import llama3.layers
        from llama3 import stream_mnt

        # Default config
        config = {
            'ssd_manifest_path': None,      # required: runtime_manifest.json (or shapes_meta.json, see below)
            'prefetch_distance': 2,
            'max_cached_layers': 4,
            'cpu_cache_layers': 50,
            'staging_mb': 64,
            'warmup_layers': 1,
            'verbose': True,
            'check_dram_capacity': True,
        }
        config.update(ssd_config)

        if not str(self.args.device).startswith("cuda"):
            raise RuntimeError("SSD streaming requires a CUDA device")

        if config['ssd_manifest_path'] is None:
            raise ValueError("ssd_manifest_path is required (runtime_manifest.json or shapes_meta.json)")

        mp = str(config['ssd_manifest_path'])
        if not Path(mp).exists():
            raise FileNotFoundError(f"SSD manifest not found: {mp}")

        # Optional: if a shapes_meta.json is provided, build runtime manifest on the fly.
        if mp.endswith(".shapes_meta.json"):
            try:
                from .weights_io_ssd_dram import build_runtime_manifest
                out_path = "/dev/shm/runtime_manifest.json"
                build_runtime_manifest(mp, out_path)
                config['ssd_manifest_path'] = out_path
                print(f"[SSD] Built runtime manifest → {out_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to build runtime manifest from shapes_meta: {e}")

        # Ensure model.layers accessible
        if hasattr(self.model, "layer_infos"):
            try:
                blocks = [info.block for info in self.model.layer_infos if info.block is not None]
                if blocks and not hasattr(self.model, "layers"):
                    self.model.layers = blocks
            except Exception:
                pass

        # Keep small/core modules on HBM
        self._configure_core_components()

        # Mark streaming mode
        if getattr(self, "_streaming_mode", None) not in (None, "ssd"):
            raise RuntimeError(f"Another streaming mode already active: {self._streaming_mode}")
        self._streaming_mode = "ssd"

        # Normalize staging bytes (WSM will align to device block size)
        staging_bytes = max(1, int(config['staging_mb'])) * 1024 * 1024

        # Create WSM with SSD backend
        wsm = WeightStreamingManager(
            self.model,
            device=self.args.device,
            prefetch_distance=config['prefetch_distance'],
            max_cached_layers=config['max_cached_layers'],
            warmup_layers=config['warmup_layers'],
            verbose=config['verbose'],
            # SSD backend
            ssd_manifest_path=config['ssd_manifest_path'],
            cpu_cache_layers=config['cpu_cache_layers'],
            staging_bytes=staging_bytes,
            check_dram_capacity=config['check_dram_capacity'],
        )

        # Integrate WSM hooks into layers
        self._integrate_wsm_to_layers(wsm)

        # KV streams
        self._configure_kv_streams()

        # Verify placements
        self._verify_and_fix_device_placement()

        # Print status
        stats = wsm.get_ssd_stats()
        print("✅ SSD Hybrid Streaming enabled:")
        print(f"   📦 CPU cache: {stats.get('cpu_cache_max', config['cpu_cache_layers'])} layers")
        print(f"   🎯 GPU cache: {config['max_cached_layers']} layers")
        print(f"   🔄 Prefetch distance: {config['prefetch_distance']} layers")
        print(f"   💾 Staging buffer: {config['staging_mb']} MB")
        print("⚙️  Pipeline: SSD → CPU (pinned) → GPU (HBM)")
        return wsm

    def _configure_core_components(self):
        """
        Keep small/core modules (embeddings, output head, final norm) permanently on the target device.
        """
        device = self.args.device
        model = self.model

        # Keep small modules resident in HBM (fast, avoids repeated transfers)
        model.embed_tokens = model.embed_tokens.to(device)
        model.norm = model.norm.to(device)
        model.output = model.output.to(device)

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

    def _integrate_wsm_to_layers(self, wsm):
        """
        Attach the WeightStreamingManager to attention/FFN modules so they can stream weights.
        """
        try:
            from . import layers  # local import intentionally kept
            layers.set_weight_manager(wsm)  # set global reference for modules to pick up

            # Manual injection for already-constructed blocks
            if hasattr(self.model, "layers"):
                for i, layer in enumerate(self.model.layers):
                    if hasattr(layer, "attention"):
                        layer.attention.weight_manager = wsm
                        layer.attention.layer_id = getattr(layer, "layer_id", i)
                    if hasattr(layer, "feed_forward"):
                        layer.feed_forward.weight_manager = wsm
                        layer.feed_forward.layer_id = getattr(layer, "layer_id", i)
        except Exception as e:
            print(f"[WARN] failed to set_weight_manager on blocks: {e}")

    def _configure_kv_streams(self):
        """
        Configure the H2D/D2H streams for KV offloader (if any) on each layer's attention module.
        """
        try:
            from . import stream_mnt  # local import intentionally kept
            streams = stream_mnt.get_streams(self.args.device)

            if hasattr(self.model, "layers"):
                for layer in self.model.layers:
                    if hasattr(layer, "attention"):
                        off = getattr(layer.attention, "offloader", None)
                        if off is not None:
                            off.h2d_stream = streams.kv_h2d
                            off.d2h_stream = streams.kv_d2h
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
            model.embed_tokens = model.embed_tokens.to(device)
            model.norm = model.norm.to(device)
            model.output = model.output.to(device)
            if hasattr(model, 'freqs_complex'):
                model.freqs_complex = model.freqs_complex.to(device)

            # Sync per-layer norms to GPU
            print("🔧 Synchronizing layer norms to GPU...")
            if hasattr(model, "layers"):
                for layer in model.layers:
                    if hasattr(layer, 'attn_norm'):
                        layer.attn_norm = layer.attn_norm.to(device)
                    if hasattr(layer, 'ffn_norm'):
                        layer.ffn_norm = layer.ffn_norm.to(device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            print("✅ All layer components synchronized to target device")
        except Exception as e:
            print(f"⚠️ Error during device synchronization: {e}")

    # ---------- Build ----------
    @staticmethod
    def build(
        checkpoints_dir: str,
        load_model: bool = True,
        device: str = "cuda",
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

        # ---- Load checkpoint weights to CPU (optional) ----
        checkpoint = None
        if load_model:
            ckpt_file = sorted(ckpt_dir.glob("*.pth"))[0]
            print(f"[INFO] Loading checkpoint: {ckpt_file}")
            t0 = time.time()
            checkpoint = torch.load(ckpt_file, map_location="cpu")
            print(f"[INFO] Done ({time.time() - t0:.1f}s)")

        # ---- Build model on CPU first to avoid OOM ----
        cpu_args = ModelArgs.from_json(
            str(params_path), max_seq_len=max_seq_len, max_batch_size=max_batch_size, device="cpu"
        )
        if topk_blk is not None:
            cpu_args.topk_blk = topk_blk
        cpu_args.checkpoints_dir = str(ckpt_dir)

        llama = LLaMA(tokenizer, checkpoint, cpu_args)

        # ---- Weight loading strategies ----
        if enable_ssd_streaming and device.startswith("cuda"):
            # SSD混合流式模式：SSD -> CPU缓存 -> GPU流式传输
            llama._configure_ssd_streaming(ssd_streaming_config or {})
        elif enable_preload_mode and device.startswith("cuda"):
            # 新的预加载模式：先加载几层，然后在推理时预取
            llama._configure_preload_mode(preload_config or {})
        elif enable_weight_streaming and device.startswith("cuda"):
            # 原有的流式加载模式
            llama._configure_weight_streaming(streaming_config or {})
        elif device.startswith("cuda"):
            # 传统的全量加载模式
            try:
                llama.model = llama.model.to(device)
                llama.args.device = device
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

                # Input token tensor: (bsz, total_len), pre-filled with pad
                tokens = torch.full(
                    (bsz, total_len),
                    pad_id,
                    dtype=torch.long,
                    device=self.args.device,
                )

                # Write prompt tokens at the front of each row
                for i, tok in enumerate(batch_prompts):
                    tokens[i, : len(tok)] = torch.tensor(tok, device=self.args.device)

                # Masks
                eos_mask = torch.zeros(bsz, dtype=torch.bool, device=self.args.device)  # track finished sequences
                prompt_mask = tokens != pad_id  # True where original prompt tokens exist

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

            # ---- Token-by-token decode loop ----
            for cur_pos in tqdm(range(1, total_len), desc=desc):
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
