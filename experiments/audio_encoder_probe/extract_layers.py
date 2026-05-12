"""Layer-wise embedding extraction for RQ1 (5 enc + 5 proj + 5 llm = 15 layers).

Spec: docs/analysis/RQ1_layer_distance.md §2 / §6 / §7.

Per utterance, saves 25 numpy keys to {out_dir}/{family}/{utt_id}.npz:
  enc_{0..4}_mean                  (encoder layers, non-causal → mean only)
  proj_{0..3}_{mean,last}          (projector LlamaDecoderLayer × 4)
  proj_out_{mean,last}             (post output_proj, 2560-d, LLM input)
  llm_{0,8,15,23,31}_{mean,last}   (LLM transformer 5 점 subsample)

Family-specific extraction:
  - whisper_tiny / whisper_small : encoder(output_hidden_states=True) → hidden_states + last_hidden_state
  - dacvae                       : hooks on encoder.encoder.block[0,1,2,4] + post-VAE z (stable seed)
  - wavtok                       : hooks on encoder.model[0,3,6,12,15]

LLM input: inference 표준 ChatML (build_prompt_ids) — §2.4.

Usage:
  python experiments/audio_encoder_probe/extract_layers.py --family whisper_tiny [--limit 5]
"""

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor, WhisperModel

# Stage1 audio_encoder.py for whisper hardcodes attn_implementation="flash_attention_2".
# This env has a broken flash_attn .so (libc symbol mismatch), so override to sdpa.
_orig_whisper_from_pretrained = WhisperModel.from_pretrained.__func__
def _patched_whisper_from_pretrained(cls, *args, **kwargs):
    kwargs["attn_implementation"] = "sdpa"
    return _orig_whisper_from_pretrained(cls, *args, **kwargs)
WhisperModel.from_pretrained = classmethod(_patched_whisper_from_pretrained)

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from evaluation.stage2._loader import build_prompt_ids  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

LLM_LAYER_IDX = [0, 8, 15, 23, 31]

# Matches eval_iemocap_session5.py prompt.
IEMOCAP_USER_SUFFIX = (
    "What is the emotion expressed?\n"
    "A. angry\nB. happy\nC. neutral\nD. sad\n"
    "Answer with the letter."
)

FAMILY_CFG: dict[str, dict] = {
    "whisper_tiny": {
        "ckpt": REPO / "external/ckpts/Qwen3.5_whisper_tiny_Stage1/Qwen3.5_whisper_tiny_Stage1/checkpoint-13000",
        "encoder_kind": "whisper",
        "whisper_model_id": "openai/whisper-tiny.en",
        "sample_rate": 16000,
        "hop_length": 320,
        "max_samples": 30 * 16000,
        "max_frames": 1500,
        "enc_hs_idx": [0, 1, 2, 3],  # enc_0..3 from hidden_states; enc_4 = last_hidden_state
    },
    "whisper_small": {
        "ckpt": REPO / "external/ckpts/Qwen3.5_whisper_small_Stage1/Qwen3.5_whisper_small_Stage1/checkpoint-13000",
        "encoder_kind": "whisper",
        "whisper_model_id": "openai/whisper-small.en",
        "sample_rate": 16000,
        "hop_length": 320,
        "max_samples": 30 * 16000,
        "max_frames": 1500,
        "enc_hs_idx": [0, 3, 6, 9],
    },
    "dacvae": {
        "ckpt": REPO / "external/ckpts/Qwen3.5AE-4B-dacvae_ASR-Stage1",
        "encoder_kind": "dacvae",
        "sample_rate": 48000,
        "hop_length": 1920,
        "max_samples": None,
        "max_frames": None,
        "enc_block_idx": [0, 1, 2, 4],  # enc_0..3 hooks; enc_4 = post-VAE z
    },
    "wavtok": {
        "ckpt": REPO / "external/models/Qwen3.5AE-4B-wavtok-40-unify",
        "encoder_kind": "seanet",
        "sample_rate": 24000,
        "hop_length": 600,
        "max_samples": None,
        "max_frames": None,
        "enc_model_idx": [0, 3, 6, 12, 15],
    },
}


def stable_hash(s: str) -> int:
    """Deterministic 32-bit hash for per-utterance VAE seed."""
    return int.from_bytes(hashlib.sha1(s.encode()).digest()[:4], "big")


def load_audio(path: str, target_sr: int, max_samples: int | None) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if max_samples is not None and wav.shape[-1] > max_samples:
        wav = wav[:, :max_samples]
    return wav, wav.shape[-1]


def compute_t_audio(family: str, n_samples: int) -> int:
    cfg = FAMILY_CFG[family]
    raw = n_samples // cfg["hop_length"]
    return min(cfg["max_frames"], raw) if cfg["max_frames"] else raw


_whisper_fe_cache: dict[str, object] = {}


def make_audio_input(family: str, wav: torch.Tensor) -> torch.Tensor:
    """Whisper: (1, 80, 3000) mel. Others: (1, 1, S) wav."""
    cfg = FAMILY_CFG[family]
    if cfg["encoder_kind"] == "whisper":
        mid = cfg["whisper_model_id"]
        if mid not in _whisper_fe_cache:
            _whisper_fe_cache[mid] = WhisperFeatureExtractor.from_pretrained(mid)
        fe = _whisper_fe_cache[mid]
        mel = fe(wav.squeeze(0).numpy(), sampling_rate=cfg["sample_rate"], return_tensors="pt").input_features
        return mel
    return wav.unsqueeze(0)


def _make_hook(captures: dict, name: str, transpose: bool = False):
    def hook(_module, _inp, output):
        t = output[0] if isinstance(output, tuple) else output
        captures[name] = t.transpose(1, 2) if transpose else t
    return hook


@torch.no_grad()
def forward_audio_encoder(
    family: str, alm, audio_input: torch.Tensor, device: str
) -> tuple[torch.Tensor, dict]:
    """Run encoder + projector with hooks. Returns (audio_embeds, captures)."""
    cfg = FAMILY_CFG[family]
    audio_encoder = alm.model.audio_encoder
    encoder = audio_encoder.encoder
    projector = audio_encoder.projector

    captures: dict[str, torch.Tensor] = {}
    handles = []

    # Projector hooks (same for all families)
    for i, layer in enumerate(projector.layers):
        handles.append(layer.register_forward_hook(_make_hook(captures, f"proj_{i}")))
    handles.append(projector.output_proj.register_forward_hook(_make_hook(captures, "proj_out")))

    try:
        if cfg["encoder_kind"] == "whisper":
            mel = audio_input.to(device).to(torch.bfloat16)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                enc_out = encoder(mel, output_hidden_states=True)
            for i, hs_idx in enumerate(cfg["enc_hs_idx"]):
                captures[f"enc_{i}"] = enc_out.hidden_states[hs_idx]
            captures["enc_4"] = enc_out.last_hidden_state
            proj_in = enc_out.last_hidden_state.to(projector.input_proj.weight.dtype)

        elif cfg["encoder_kind"] == "dacvae":
            for i, block_idx in enumerate(cfg["enc_block_idx"]):
                handles.append(encoder.encoder.block[block_idx].register_forward_hook(
                    _make_hook(captures, f"enc_{i}", transpose=True)))
            wav = audio_input.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                encoded = encoder.encode(wav)  # (B, 128, T)
            captures["enc_4"] = encoded.transpose(1, 2)
            proj_in = encoded.transpose(1, 2).to(projector.input_proj.weight.dtype)

        elif cfg["encoder_kind"] == "seanet":
            for i, m_idx in enumerate(cfg["enc_model_idx"]):
                handles.append(encoder.model[m_idx].register_forward_hook(
                    _make_hook(captures, f"enc_{i}", transpose=True)))
            wav = audio_input.to(device).to(torch.bfloat16)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                z_e = encoder(wav)  # (B, 512, T)
            proj_in = z_e.transpose(1, 2).to(projector.input_proj.weight.dtype)
        else:
            raise ValueError(family)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            audio_embeds, _ = projector(proj_in)
    finally:
        for h in handles:
            h.remove()

    return audio_embeds, captures


@torch.no_grad()
def forward_llm(
    alm, audio_embeds: torch.Tensor, t_audio: int, prompt_ids: list[int], audio_pad_id: int, device: str
) -> dict[int, torch.Tensor]:
    """Inject audio_embeds[:t_audio] at audio_pad positions; return sliced hidden states.

    Uses forward hooks because Qwen3_5AETextModel.forward() ignores `output_hidden_states`
    and only returns last_hidden_state.
    """
    llm = alm.model.language_model
    embed_layer = alm.get_input_embeddings()

    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    inputs_embeds = embed_layer(input_ids)
    audio_mask = input_ids == audio_pad_id
    n_audio = int(audio_mask.sum().item())
    valid = audio_embeds[0, :t_audio]
    if n_audio != valid.shape[0]:
        raise RuntimeError(f"prompt audio_pad ({n_audio}) != valid audio frames ({valid.shape[0]})")
    inputs_embeds[audio_mask] = valid.to(inputs_embeds.dtype)

    captures: dict[int, torch.Tensor] = {}
    handles = [
        llm.layers[li].register_forward_hook(_make_hook(captures, str(li)))
        for li in LLM_LAYER_IDX
    ]
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            _ = llm(inputs_embeds=inputs_embeds, return_dict=True)
    finally:
        for h in handles:
            h.remove()

    audio_idx = audio_mask.squeeze(0).nonzero().squeeze(-1)
    return {li: captures[str(li)][0, audio_idx] for li in LLM_LAYER_IDX}


def pool(x: torch.Tensor, kind: str) -> np.ndarray:
    v = x.float().mean(0) if kind == "mean" else x.float()[-1]
    return v.cpu().numpy().astype(np.float32)


EXPECTED_KEYS = frozenset(
    [f"enc_{i}_mean" for i in range(5)]
    + [f"proj_{i}_{p}" for i in range(4) for p in ("mean", "last")]
    + [f"proj_out_{p}" for p in ("mean", "last")]
    + [f"llm_{li}_{p}" for li in LLM_LAYER_IDX for p in ("mean", "last")]
)


def extract_one(family: str, alm, tokenizer, audio_pad_id: int, utt_id: str, audio_path: str, device: str) -> dict:
    cfg = FAMILY_CFG[family]
    wav, n_samples = load_audio(audio_path, cfg["sample_rate"], cfg["max_samples"])
    t_audio = compute_t_audio(family, n_samples)
    if t_audio == 0:
        raise RuntimeError(f"t_audio=0 (n_samples={n_samples})")
    audio_input = make_audio_input(family, wav)
    prompt_ids = build_prompt_ids(tokenizer, audio_pad_id, t_audio, IEMOCAP_USER_SUFFIX, no_think=False)

    if family == "dacvae":
        torch.manual_seed(stable_hash(utt_id))

    audio_embeds, captures = forward_audio_encoder(family, alm, audio_input, device)
    llm_hidden = forward_llm(alm, audio_embeds, t_audio, prompt_ids, audio_pad_id, device)

    result: dict[str, np.ndarray] = {}
    for i in range(5):
        cap = captures[f"enc_{i}"]
        valid_t = min(t_audio, cap.shape[1])
        result[f"enc_{i}_mean"] = pool(cap[0, :valid_t], "mean")
    for i in range(4):
        cap = captures[f"proj_{i}"]
        valid_t = min(t_audio, cap.shape[1])
        h = cap[0, :valid_t]
        result[f"proj_{i}_mean"] = pool(h, "mean")
        result[f"proj_{i}_last"] = pool(h, "last")
    cap = captures["proj_out"]
    valid_t = min(t_audio, cap.shape[1])
    h = cap[0, :valid_t]
    result["proj_out_mean"] = pool(h, "mean")
    result["proj_out_last"] = pool(h, "last")
    for li, hs in llm_hidden.items():
        result[f"llm_{li}_mean"] = pool(hs, "mean")
        result[f"llm_{li}_last"] = pool(hs, "last")
    return result


def needs_extraction(out_path: Path) -> bool:
    if not out_path.exists():
        return True
    try:
        existing = set(np.load(out_path).files)
    except Exception:
        return True
    return existing != EXPECTED_KEYS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=list(FAMILY_CFG))
    parser.add_argument("--manifest", default=str(
        REPO / "experiments/audio_encoder_probe/manifests/iemocap_4class_balanced500.csv"))
    parser.add_argument("--out-dir", default=str(REPO / "experiments/audio_encoder_probe/embeds_layers"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = FAMILY_CFG[args.family]
    out_dir = Path(args.out_dir) / args.family
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest).sort_values("utt_id").reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    logger.info(f"Manifest: {len(df)} utterances")

    if args.overwrite:
        pending = df
    else:
        pending = df[df["utt_id"].apply(lambda u: needs_extraction(out_dir / f"{u}.npz"))]
    logger.info(f"To process: {len(pending)} (skip {len(df) - len(pending)} already-done)")

    if pending.empty:
        return

    logger.info(f"Loading {args.family} from {cfg['ckpt']}")
    alm = AutoModelForCausalLM.from_pretrained(
        str(cfg["ckpt"]), dtype=torch.bfloat16,
        trust_remote_code=True, local_files_only=True, device_map=args.device,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(cfg["ckpt"]), trust_remote_code=True)
    audio_pad_id = alm.config.audio_pad_token_id

    t0 = time.time()
    n_ok = n_fail = 0
    for n, row in enumerate(pending.itertuples(index=False), 1):
        try:
            result = extract_one(args.family, alm, tokenizer, audio_pad_id,
                                 row.utt_id, row.audio_path, args.device)
            if set(result) != EXPECTED_KEYS:
                raise RuntimeError(f"key mismatch: missing {EXPECTED_KEYS - set(result)}")
            for k, v in result.items():
                if not np.isfinite(v).all():
                    raise RuntimeError(f"NaN/Inf in {k}")
            np.savez(str(out_dir / f"{row.utt_id}.npz"), **result)
            n_ok += 1
        except Exception as e:
            logger.warning(f"FAIL {row.utt_id}: {e}")
            n_fail += 1
        if n % 50 == 0 or n == len(pending):
            rate = n / (time.time() - t0)
            eta = (len(pending) - n) / max(rate, 1e-9)
            logger.info(f"[{n}/{len(pending)}] ok={n_ok} fail={n_fail} | {rate:.1f} utt/s | ETA {eta/60:.1f} min")

    logger.info(f"Done. {n_ok} saved, {n_fail} failed → {out_dir}")


if __name__ == "__main__":
    main()
