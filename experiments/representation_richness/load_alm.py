"""
Load ALM models for layer-wise representation extraction.

Supports 5 family configs:
  whisper_tiny      Stage1 ckpt-13000  16kHz mel   encoder_dim=384
  whisper_small     Stage1 ckpt-13000  16kHz mel   encoder_dim=768
  wavtok_40_unify   Stage1 ckpt-100k   24kHz raw   encoder_dim=512
  dacvae_stage1     root model         48kHz raw   encoder_dim=128
  dacvae_stage2     LoRA on stage1     48kHz raw   encoder_dim=128

Returns model, tokenizer, and family_cfg dict used by extract.py.
"""

import logging
import math
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

CKPT_ROOT = Path(__file__).parent.parent.parent / "external" / "ckpts"
MODEL_ROOT = Path(__file__).parent.parent.parent / "external" / "models"

# Metadata used by extract.py for audio preprocessing and valid-frame masking
FAMILY_CFGS = {
    "whisper_tiny": {
        "encoder_type": "whisper",            # hook on model.audio_encoder.encoder
        "sample_rate": 16000,
        "hop_length": 320,                    # encoder-out stride in samples
        "max_frames": 1500,                   # Whisper 30s cap
        "encoder_dim": 384,
        "whisper_model_id": "openai/whisper-tiny.en",
        "ckpt_dir": CKPT_ROOT / "Qwen3.5_whisper_tiny_Stage1" / "Qwen3.5_whisper_tiny_Stage1",
        "base_model_dir": MODEL_ROOT / "Qwen3.5AE-4B-whisper-tiny",
        "is_lora": False,
    },
    "whisper_small": {
        "encoder_type": "whisper",
        "sample_rate": 16000,
        "hop_length": 320,
        "max_frames": 1500,
        "encoder_dim": 768,
        "whisper_model_id": "openai/whisper-small.en",
        "ckpt_dir": CKPT_ROOT / "Qwen3.5_whisper_small_Stage1" / "Qwen3.5_whisper_small_Stage1",
        "base_model_dir": MODEL_ROOT / "Qwen3.5AE-4B-whisper-small",
        "is_lora": False,
    },
    "wavtok_40_unify": {
        "encoder_type": "wavtok",             # hook on model.audio_encoder.encoder (SEANet)
        "sample_rate": 24000,
        "hop_length": 600,                    # 24kHz / 40fps
        "max_frames": None,
        "encoder_dim": 512,
        "whisper_model_id": None,
        "ckpt_dir": CKPT_ROOT / "Qwen3.5_wavtok_40_unify_Stage1" / "Qwen3.5_wavtok_40_unify_Stage1",
        "base_model_dir": MODEL_ROOT / "Qwen3.5AE-4B-wavtok-40-unify",
        "is_lora": False,
    },
    "dacvae_stage1": {
        "encoder_type": "dacvae",             # manual encode() call (not hookable)
        "sample_rate": 48000,
        "hop_length": 1920,                   # product of encoder_rates [2,8,10,12]
        "max_frames": None,
        "encoder_dim": 128,
        "whisper_model_id": None,
        "ckpt_dir": CKPT_ROOT / "Qwen3.5AE-4B-ASR-Stage1",
        "base_model_dir": CKPT_ROOT / "Qwen3.5AE-4B-ASR-Stage1",  # IS the base
        "is_lora": False,
    },
    "dacvae_stage2": {
        "encoder_type": "dacvae",
        "sample_rate": 48000,
        "hop_length": 1920,
        "max_frames": None,
        "encoder_dim": 128,
        "whisper_model_id": None,
        "ckpt_dir": CKPT_ROOT / "Qwen3.5AE-4B-ASR-Stage2" / "checkpoint-2000",
        "base_model_dir": CKPT_ROOT / "Qwen3.5AE-4B-ASR-Stage1",
        "is_lora": True,
    },
}


def _ensure_model_files(ckpt_path: Path, base_model_dir: Path):
    """Symlink audio_encoder.py / modeling / configuration from base if missing in ckpt."""
    needed = [
        "audio_encoder.py",
        "configuration_qwen3_5AE.py",
        "modeling_qwen3_5AE.py",
        "chat_template.jinja",
    ]
    for fname in needed:
        dst = ckpt_path / fname
        src = base_model_dir / fname
        if not dst.exists():
            if src.exists():
                os.symlink(src, dst)
                logger.info(f"  Symlinked {fname} from {base_model_dir}")
            else:
                logger.warning(f"  {fname} not found in base model dir {base_model_dir}")


def _latest_checkpoint(ckpt_dir: Path) -> Path:
    """Return the latest checkpoint-XXXXX directory, or the dir itself if no subdirs."""
    import re
    cks = [d for d in ckpt_dir.iterdir() if re.match(r"checkpoint-\d+$", d.name)]
    if not cks:
        return ckpt_dir  # the dir IS the checkpoint (e.g. dacvae_stage1)
    return sorted(cks, key=lambda p: int(p.name.split("-")[1]))[-1]


def load_alm(family: str, device: str = "cuda:0", dtype=torch.bfloat16):
    """
    Load an ALM family for extraction.

    Args:
        family: one of the keys in FAMILY_CFGS
        device: target CUDA device
        dtype: model weight dtype (bfloat16 recommended)

    Returns:
        model: loaded, eval-mode model
        tokenizer: HF tokenizer
        cfg: family_cfg dict (sample_rate, hop_length, encoder_type, …)
    """
    assert family in FAMILY_CFGS, f"Unknown family: {family}. Choose from {list(FAMILY_CFGS)}"
    cfg = FAMILY_CFGS[family]

    ckpt_path = _latest_checkpoint(cfg["ckpt_dir"])
    base_dir = cfg["base_model_dir"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Loading family: {family}")
    logger.info(f"  Checkpoint: {ckpt_path}")
    logger.info(f"  Encoder type: {cfg['encoder_type']}")
    logger.info(f"{'='*60}")

    if cfg["is_lora"]:
        model = _load_lora(ckpt_path, base_dir, device, dtype)
    else:
        _ensure_model_files(ckpt_path, base_dir)
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path),
            dtype=dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Load tokenizer from base model (has tokenizer.json + special tokens)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path), trust_remote_code=True)

    # Quick sanity: all families have model.model.audio_encoder + model.model.language_model
    inner = model.model
    assert hasattr(inner, "audio_encoder"), "model.model.audio_encoder not found"
    assert hasattr(inner, "language_model"), "model.model.language_model not found"
    n_layers = len(inner.language_model.layers)
    logger.info(f"  LLM layers: {n_layers}")
    assert n_layers == 32, f"Expected 32 LLM layers, got {n_layers}"
    n_proj = len(inner.audio_encoder.projector.layers)
    logger.info(f"  Projector layers: {n_proj}")

    logger.info(f"  Loaded successfully. audio_pad_token_id = {model.config.audio_pad_token_id}")
    return model, tokenizer, cfg


def _load_lora(lora_ckpt_path: Path, base_dir: Path, device: str, dtype):
    """Load base model + merge LoRA adapters from checkpoint."""
    from peft import PeftModel

    logger.info(f"  Loading base model from {base_dir} then applying LoRA from {lora_ckpt_path}")
    _ensure_model_files(base_dir, base_dir)

    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device)

    model = PeftModel.from_pretrained(base_model, str(lora_ckpt_path))
    model = model.merge_and_unload()
    logger.info("  LoRA adapters merged and unloaded.")
    return model


def t_audio_for_family(cfg: dict, num_samples: int) -> int:
    """Compute number of audio_pad_token slots for a given waveform length."""
    if cfg["max_frames"] is not None:
        return min(cfg["max_frames"], math.ceil(num_samples / cfg["hop_length"]))
    return num_samples // cfg["hop_length"]
