"""Shared Stage-2 eval loader.

Responsibilities:
  1. Load a Qwen3.5AE Stage-2 checkpoint. Auto-detect adapter-only (PEFT LoRA)
     vs. full-weight; adapter-only dirs are wrapped via
     PeftModel.from_pretrained(base, adapter_dir).
  2. Verify projector weights (audio_encoder.projector.*) round-trip from the
     raw adapter safetensors into live model parameters. Prints missing /
     unexpected key counts and exits non-zero on projector regression (any
     projector key in the raw file that fails to land in a model parameter).
  3. Apply the transformers 5.5+ `prepare_inputs_for_generation` shim from
     AudioEnc/eval_ckpts/inference_ckpt_sweep.py so audio features survive
     generate().
  4. Patch `modeling_qwen3_5AE.DynamicCache -> Qwen3NextDynamicCache` for the
     hybrid linear-attention path (same shim as evaluation/eval_testclean_wer.py).
  5. Provide a ChatML prompt builder matching omni_dataset.create_omni_processor
     so eval prompts are byte-identical to training user-turn format (modulo the
     task-prompt stem, which eval pins to a canonical phrasing).

The loader returns (model, tokenizer, config); callers do their own audio
pre-processing and .generate() call.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as _sf_load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextDynamicCache

# 48 kHz DAC-VAE inputs, 1920-sample hop -> one audio token per 40 ms.
SAMPLE_RATE = 48000
HOP_LENGTH = 1920


# ---------------------------------------------------------------------------
# checkpoint loading
# ---------------------------------------------------------------------------


def _is_adapter_only(ckpt_dir: Path) -> bool:
    has_adapter = (ckpt_dir / "adapter_config.json").exists() and (
        ckpt_dir / "adapter_model.safetensors"
    ).exists()
    has_full = (ckpt_dir / "model.safetensors.index.json").exists() or (
        ckpt_dir / "model.safetensors"
    ).exists()
    return has_adapter and not has_full


def _patch_dynamic_cache() -> None:
    """Bridge the stock-transformers cache API to the vendored modeling code.

    The vendored `modeling_qwen3_5AE.py` (loaded via trust_remote_code) was
    written against an older Qwen3Next cache API than what the current
    transformers ships. Reconciling:

    - Stock DynamicCache (used by transformers.generate) does not carry
      `conv_states` / `recurrent_states`; the linear-attn path needs those.
      Swap modeling_qwen3_5AE.DynamicCache -> Qwen3NextDynamicCache so the
      custom model instantiates the correct cache.

    - `cache.has_previous_state` is a @property in current Qwen3NextDynamicCache;
      the vendored code calls it as `cache.has_previous_state(layer_idx)`.
      Replace with a method.

    - Vendored code uses the old sub-object API:
          cache.layers[i].conv_states
          cache.layers[i].recurrent_states
          cache.update_conv_state(state, i)
          cache.update_recurrent_state(state, i)
      Provide a `layers` property returning per-layer proxies plus the two
      `update_*` methods that write back into the top-level `conv_states` /
      `recurrent_states` lists.

    All shims guarded by a sentinel so repeated loader calls are idempotent.
    """
    for mod_name, mod in list(sys.modules.items()):
        if "modeling_qwen3_5AE" in mod_name and hasattr(mod, "DynamicCache"):
            mod.DynamicCache = Qwen3NextDynamicCache

    if getattr(Qwen3NextDynamicCache, "_qwen35ae_patched", False):
        return

    def _has_previous_state(self, layer_idx=None):
        idx = layer_idx if layer_idx is not None else getattr(self, "last_linear_layer", 0)
        try:
            return self.conv_states[idx] is not None
        except (AttributeError, KeyError, IndexError, TypeError):
            return False

    def _update_conv_state(self, state, layer_idx):
        self.conv_states[layer_idx] = state
        return state

    def _update_recurrent_state(self, state, layer_idx):
        self.recurrent_states[layer_idx] = state
        return state

    class _LayerProxy:
        __slots__ = ("_cache", "_idx")

        def __init__(self, cache, idx):
            self._cache = cache
            self._idx = idx

        @property
        def conv_states(self):
            return self._cache.conv_states[self._idx]

        @conv_states.setter
        def conv_states(self, value):
            self._cache.conv_states[self._idx] = value

        @property
        def recurrent_states(self):
            return self._cache.recurrent_states[self._idx]

        @recurrent_states.setter
        def recurrent_states(self, value):
            self._cache.recurrent_states[self._idx] = value

    def _layers(self):
        return [_LayerProxy(self, i) for i in range(len(self.conv_states))]

    Qwen3NextDynamicCache.has_previous_state = _has_previous_state
    Qwen3NextDynamicCache.update_conv_state = _update_conv_state
    Qwen3NextDynamicCache.update_recurrent_state = _update_recurrent_state
    Qwen3NextDynamicCache.layers = property(_layers)
    Qwen3NextDynamicCache._qwen35ae_patched = True


def _patch_generate_prepare(model: torch.nn.Module) -> None:
    """Re-inject audio_features on the first generation step.

    transformers >= 5.5 stopped threading `cache_position` through generate(),
    so the upstream prepare_inputs_for_generation silently drops audio_features
    and the model emits generic chat instead of audio-conditioned text. Detect
    first step via is_first_iteration / cache_position / empty past_key_values
    and re-inject.
    """
    def _patched_prepare(self, *args, **kwargs):
        audio_features = kwargs.pop("audio_features", None)
        audio_lengths = kwargs.pop("audio_lengths", None)
        is_first = kwargs.get("is_first_iteration", None)
        cache_position = kwargs.get("cache_position", None)
        past_kv = kwargs.get("past_key_values", None)

        model_inputs = super(type(self), self).prepare_inputs_for_generation(*args, **kwargs)

        first_step = False
        if is_first is not None:
            first_step = bool(is_first)
        elif cache_position is not None:
            first_step = bool(cache_position[0].item() == 0)
        else:
            try:
                first_step = (past_kv is None) or (past_kv.get_seq_length() == 0)
            except Exception:
                first_step = True

        if first_step and audio_features is not None:
            model_inputs["audio_features"] = audio_features
            model_inputs["audio_lengths"] = audio_lengths
        return model_inputs

    model.prepare_inputs_for_generation = types.MethodType(_patched_prepare, model)


def _verify_projector_restore(model: torch.nn.Module, adapter_dir: Path) -> dict[str, int]:
    """Cross-check projector keys from adapter_model.safetensors against live params.

    PEFT's additional_target -> modules_to_save wraps the projector into a
    ModulesToSaveWrapper that keeps TWO copies in the live state_dict:
        * `...projector.original_module.*` — frozen pre-S2 init.
        * `...projector.modules_to_save.default.*` — trained copy loaded from
          the adapter file. This is what PEFT actually runs forward through.

    We verify (a) every raw projector key lands in a `.modules_to_save.default.`
    slot, and (b) the values are bit-identical (within dtype) so we catch silent
    adapter-load failures. We also guard against (a) succeeding while PEFT
    accidentally serves the original (frozen) copy — by checking that the
    trained copy actually differs from the original for at least one key.
    """
    raw_path = adapter_dir / "adapter_model.safetensors"
    raw = _sf_load_file(str(raw_path), device="cpu")
    proj_keys_raw = [k for k in raw if "audio_encoder.projector" in k]
    if not proj_keys_raw:
        raise RuntimeError(
            f"[_loader] adapter file {raw_path} has NO projector keys — training "
            "config likely dropped `additional_target: audio_encoder.projector`."
        )

    live_state = dict(model.state_dict())

    def _map_to_live(raw_key: str) -> str:
        """Insert `.modules_to_save.default` after `.audio_encoder.projector`."""
        head, _, tail = raw_key.partition("audio_encoder.projector")
        return f"{head}audio_encoder.projector.modules_to_save.default{tail}"

    def _map_to_original(raw_key: str) -> str:
        head, _, tail = raw_key.partition("audio_encoder.projector")
        return f"{head}audio_encoder.projector.original_module{tail}"

    missing = []
    mismatch = []
    diffs_vs_original = 0

    for k in proj_keys_raw:
        live_k = _map_to_live(k)
        if live_k not in live_state:
            missing.append(k)
            continue
        live_v = live_state[live_k]
        raw_v = raw[k]
        if live_v.shape != raw_v.shape:
            mismatch.append((k, tuple(live_v.shape), tuple(raw_v.shape), "shape"))
            continue
        if not torch.equal(live_v.cpu().to(raw_v.dtype), raw_v):
            mismatch.append((k, "value-diff", None, "value"))
            continue
        # Sanity: if a matching `original_module` copy exists and differs from
        # the trained copy, training actually moved this param.
        orig_k = _map_to_original(k)
        if orig_k in live_state:
            if not torch.equal(live_state[orig_k].cpu().to(raw_v.dtype), raw_v):
                diffs_vs_original += 1

    if missing:
        print(f"[_loader] projector MISSING {len(missing)} / {len(proj_keys_raw)}:",
              flush=True)
        for m in missing[:5]:
            print(f"  expected live key: {_map_to_live(m)}", flush=True)
        raise RuntimeError("projector did not round-trip — see missing keys.")
    if mismatch:
        for k, a, b, kind in mismatch[:5]:
            print(f"[_loader] projector {kind} mismatch: {k} live={a} raw={b}", flush=True)
        raise RuntimeError("projector value/shape mismatch — see above.")

    if diffs_vs_original == 0:
        # Every trained-copy param matches the pre-S2 frozen copy — suggests
        # training never moved the projector (or the wrong copy was loaded).
        # Not fatal (could genuinely happen early in training), but flag it.
        print("[_loader] WARNING: trained projector == original init on every key. "
              "Either S2 has not moved projector yet, or PEFT served the frozen copy.",
              flush=True)

    return {
        "projector_keys_in_adapter": len(proj_keys_raw),
        "projector_keys_verified": len(proj_keys_raw),
        "projector_keys_diff_from_original": diffs_vs_original,
    }


def load_checkpoint(
    ckpt_dir: str | Path,
    base_model_dir: str | Path | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
) -> tuple[Any, Any, Any]:
    """Load a Stage-2 checkpoint. See module docstring for the full contract.

    For adapter-only dirs: base_model_dir must be given, or adapter_config.json
    must carry a valid base_model_name_or_path.
    """
    ckpt_dir = Path(ckpt_dir)
    adapter_only = _is_adapter_only(ckpt_dir)

    if adapter_only:
        if base_model_dir is None:
            cfg_path = ckpt_dir / "adapter_config.json"
            with open(cfg_path) as f:
                acfg = json.load(f)
            base_model_dir = acfg.get("base_model_name_or_path")
            if not base_model_dir or not Path(base_model_dir).is_dir():
                raise ValueError(
                    f"{ckpt_dir} is adapter-only but base_model_dir unset and "
                    f"adapter_config.json's base_model_name_or_path "
                    f"({base_model_dir!r}) is not a valid directory."
                )
        base_model_dir = Path(base_model_dir)
        print(f"[_loader] base model <- {base_model_dir}", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        ).to(device).eval()

        from peft import PeftModel

        print(f"[_loader] adapter <- {ckpt_dir}", flush=True)
        model = PeftModel.from_pretrained(base, str(ckpt_dir)).to(device).eval()
        stats = _verify_projector_restore(model, ckpt_dir)
        print(f"[_loader] projector verify ok ({stats['projector_keys_verified']} keys)", flush=True)

        tok_src = ckpt_dir if (ckpt_dir / "tokenizer.json").exists() else base_model_dir
        tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
        # Model config (audio_pad_token_id etc.) should come from the base model,
        # not the PEFT wrapper.
        config = AutoConfig.from_pretrained(base_model_dir, trust_remote_code=True)
    else:
        print(f"[_loader] full-weight load <- {ckpt_dir}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
        config = AutoConfig.from_pretrained(ckpt_dir, trust_remote_code=True)

    _patch_dynamic_cache()
    _patch_generate_prepare(model)
    return model, tokenizer, config


# ---------------------------------------------------------------------------
# ChatML prompt construction (mirrors omni_dataset.create_omni_processor)
# ---------------------------------------------------------------------------

_SYS_USER = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
_END_USER = "<|im_end|>\n<|im_start|>assistant\n"


def build_prompt_ids(
    tokenizer,
    audio_pad_id: int,
    t_audio: int,
    user_suffix_text: str,
) -> list[int]:
    """Build ChatML token ids for an audio-conditioned user turn.

    Exactly mirrors omni_dataset.py's assembly up to the assistant boundary:
        sys_user + <|audio_start|> + [audio_pad]*t_audio + <|audio_end|>
        + user_suffix + <|im_end|> + <|im_start|>assistant\n
    """
    sys_user = tokenizer.encode(_SYS_USER, add_special_tokens=False)
    audio_start = tokenizer.encode("<|audio_start|>", add_special_tokens=False)
    audio_end = tokenizer.encode("<|audio_end|>", add_special_tokens=False)
    end_user = tokenizer.encode(_END_USER, add_special_tokens=False)
    suffix = tokenizer.encode(user_suffix_text, add_special_tokens=False)

    ids = list(sys_user)
    if t_audio > 0:
        ids.extend(audio_start)
        ids.extend([audio_pad_id] * t_audio)
        ids.extend(audio_end)
    ids.extend(suffix)
    ids.extend(end_user)
    return ids


# ---------------------------------------------------------------------------
# generation helpers
# ---------------------------------------------------------------------------


def left_pad_batch(
    prompts: list[list[int]],
    pad_id: int,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Left-pad variable-length prompts. Returns (input_ids, attention_mask, max_len)."""
    max_len = max(len(p) for p in prompts)
    B = len(prompts)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, p in enumerate(prompts):
        left = max_len - len(p)
        input_ids[i, left:] = torch.tensor(p, dtype=torch.long, device=device)
        attention_mask[i, left:] = 1
    return input_ids, attention_mask, max_len


def pack_audio_batch(
    waveforms: list[torch.Tensor],  # each (S_i,), fp32
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack variable-length waveforms into (B, 1, S_max), returning audio_lengths."""
    max_s = max(w.shape[-1] for w in waveforms)
    B = len(waveforms)
    out = torch.zeros(B, 1, max_s, dtype=dtype, device=device)
    for i, w in enumerate(waveforms):
        out[i, 0, : w.shape[-1]] = w.to(dtype).to(device)
    lengths = torch.tensor(
        [max(1, w.shape[-1] // HOP_LENGTH) for w in waveforms],
        dtype=torch.long,
        device=device,
    )
    return out, lengths


# ---------------------------------------------------------------------------
# shared greedy-generate wrapper (used by all Tier-2/3 drivers)
# ---------------------------------------------------------------------------


def generate_greedy(
    model,
    tokenizer,
    cfg,
    prompts: list[list[int]],
    waveforms: list[torch.Tensor] | None,
    *,
    max_new_tokens: int,
    use_cache: bool = True,
) -> list[str]:
    """Run left-padded greedy generation on a batch of prompts.

    `waveforms=None` triggers text-only generation: audio_features and
    audio_lengths are passed as None. Prompts must then contain no
    `<|audio_pad|>` tokens (caller's responsibility — build_prompt_ids with
    t_audio=0 guarantees this).

    `use_cache=False` bypasses KV/Conv1d cache entirely — every decode step
    re-runs the full prompt through the model. This is O(n_new * prompt_len)
    instead of O(n_new + prompt_len), but avoids the cache-API shims in
    `_patch_dynamic_cache()` and serves as ground truth for shim verification.
    """
    device = next(model.parameters()).device
    pad_id = cfg.pad_token_id
    eos_id = cfg.eos_token_id

    input_ids, attn_mask, prompt_len = left_pad_batch(prompts, pad_id, device=device)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        use_cache=use_cache,
    )
    if waveforms is not None:
        audio_features, audio_lengths = pack_audio_batch(waveforms, device, dtype=torch.bfloat16)
        gen_kwargs["audio_features"] = audio_features
        gen_kwargs["audio_lengths"] = audio_lengths

    out = model.generate(**gen_kwargs)
    texts = []
    for i in range(out.size(0)):
        gen = out[i, prompt_len:]
        texts.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
    return texts


# ---------------------------------------------------------------------------
# checkpoint discovery (copied from evaluation/eval_testclean_wer.py:find_checkpoints)
# ---------------------------------------------------------------------------


def find_checkpoints(
    root: str | Path,
    steps_filter: set[int] | None = None,
    min_age_sec: int = 60,
) -> list[Path]:
    root = Path(root)
    ckpts = []
    for p in sorted(root.iterdir()):
        m = re.match(r"checkpoint-(\d+)$", p.name)
        if not m:
            continue
        step = int(m.group(1))
        if steps_filter is not None and step not in steps_filter:
            continue
        idx = p / "model.safetensors.index.json"
        mono = p / "model.safetensors"
        adapter = p / "adapter_model.safetensors"
        adapter_cfg = p / "adapter_config.json"
        ref = None
        if idx.exists():
            ref = idx
        elif mono.exists():
            ref = mono
        elif adapter.exists() and adapter_cfg.exists():
            ref = adapter
        if ref is None:
            print(f"[_loader] skip {p.name}: no weights", flush=True)
            continue
        if min_age_sec > 0 and (time.time() - ref.stat().st_mtime) < min_age_sec:
            print(f"[_loader] skip {p.name}: save in progress", flush=True)
            continue
        ckpts.append((step, p))
    ckpts.sort(key=lambda x: x[0])
    return [p for _, p in ckpts]
