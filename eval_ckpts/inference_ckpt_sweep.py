"""
inference.py — ASR inference & WER evaluation (DAC-VAE encoder + AudioQwen)

Usage:
    # audiollm-trainer에서 stage 1 projector.pt로 평가
      python inference.py --qwen3ae --eval \
        --dacvae_path /mnt/ddn/users/sehyun/AudioEncoder/AudioEnc/dacvae \
        --split test-clean --batch_size 128 --no_repeat_ngram 3 --no_think \
        --out_dir ./results/opt12_combined


    # Single file transcription
    python inference.py --audio path/to/audio.wav

    # WER evaluation on LibriSpeech test-clean
    python inference.py --eval --split test-clean

    # Use a specific checkpoint
    LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/glibc_compat.so" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python inference.py --eval --split test-clean --max_samples 10 \
        --ckpt /mnt/ddn/users/sehyun/AudioEncoder/AudioEnc/outputs/fb_dacvae_v2/stage1/final/projector.pt \
        --out_dir ./results/stage1/final

    LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/glibc_compat.so" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python inference.py --eval --split test-clean --max_samples 10 \
        --ckpt /mnt/ddn/users/sehyun/AudioEncoder/AudioEnc/outputs/fb_dacvae_v2/stage2/step_26100 \
        --out_dir ./results/stage2/step_26100
    # (Requires full_merged_model.pt in the step dir — run merge_fsdp_shards.py first)


    # Multi-GPU evaluation (4 GPUs)
    python inference.py --eval --n_gpu 4
"""

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import math

# Allow running this script from any cwd: prepend the parent AudioEnc dir so `from config
# import get_config` and friends resolve correctly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.multiprocessing as mp
import torchaudio
import torchaudio.functional as AF
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from config import get_config
from dataset import LibriSpeechDataset
from encoders.fb_dacvae import FbDACVAEEncoder

ENCODER_NAME = "fb_dacvae"
DEFAULT_CKPT = "/mnt/tmp/cache/hf/s1_proj_fb_dacvae.pt"

LLM_MAP = {
    "2b":   "Qwen/Qwen3.5-2B",
    "4b":   "Qwen/Qwen3.5-4B",
    "7b":   "Qwen/Qwen2.5-7B-Instruct",
}

QWEN3AE_MODEL_DIR = "/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/ckpts/Qwen3AE-ASR-Stage1/checkpoint-29000"
DACVAE_PKG_PATH = "/mnt/ddn/users/sehyun/AudioEncoder/AudioEnc/dacvae"  # local dacvae package (needed by Qwen3AE audio_encoder.py)
DAC_HOP_LENGTH = 1920    # product of dac_encoder_rates [2, 8, 10, 12]
DAC_SAMPLE_RATE = 48000  # Qwen3AE DAC-VAE requires 48 kHz


# ==========================================
# 1. Model loading
# ==========================================

def load_qwen3ae_model(device: str = "cuda", model_dir: str = None, dacvae_path: str = None,
                       base_model_dir: str = None):
    """Load a Qwen3.5AE checkpoint. Auto-detects adapter-only (peft LoRA) vs full-weight
    checkpoints: if `model_dir/adapter_config.json` exists, we load `base_model_dir` as
    the full base and wrap it with `PeftModel.from_pretrained(model_dir)`.

    Args:
        model_dir: checkpoint dir. May be full-weight (has model.safetensors / shards) or
            adapter-only (has adapter_config.json + adapter_model.safetensors).
        base_model_dir: required when model_dir is adapter-only. Points to the base (e.g.
            the Stage 1 init dir used as Stage 2's model_name_or_path).
    """
    if dacvae_path:
        sys.path.insert(0, dacvae_path)
    model_dir = model_dir or QWEN3AE_MODEL_DIR

    is_adapter_only = os.path.exists(os.path.join(model_dir, "adapter_config.json")) and not os.path.exists(
        os.path.join(model_dir, "model.safetensors.index.json")
    ) and not os.path.exists(os.path.join(model_dir, "model.safetensors"))

    if is_adapter_only:
        if base_model_dir is None:
            # Try to read base_model_name_or_path from adapter_config.json
            import json as _json
            with open(os.path.join(model_dir, "adapter_config.json")) as _f:
                _cfg = _json.load(_f)
            base_model_dir = _cfg.get("base_model_name_or_path")
            if not base_model_dir or not os.path.isdir(base_model_dir):
                raise ValueError(
                    f"{model_dir} is an adapter-only checkpoint but --base_model was not "
                    f"given and adapter_config.json's base_model_name_or_path "
                    f"({base_model_dir!r}) is not a valid directory"
                )
        print(f"Loading base model from {base_model_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(device).eval()
        from peft import PeftModel
        print(f"Applying adapter from {model_dir}")
        model = PeftModel.from_pretrained(model, model_dir).to(device).eval()
        # Tokenizer lives in base dir (adapter dir usually doesn't duplicate it)
        tok_src = model_dir if os.path.exists(os.path.join(model_dir, "tokenizer.json")) else base_model_dir
        tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # --- transformers 5.5+ compat shim ---
    # The checkpoint's prepare_inputs_for_generation only injects audio_features when
    # cache_position[0] == 0. In transformers >= 5.5, `generate()` no longer passes
    # cache_position through, so audio is silently dropped and the LLM outputs generic
    # chat. Override to inject on the first iteration (detected via is_first_iteration
    # or an empty past_key_values).
    import types

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
            # Fall back to past_key_values state
            try:
                first_step = (past_kv is None) or (past_kv.get_seq_length() == 0)
            except Exception:
                first_step = True

        if first_step and audio_features is not None:
            model_inputs["audio_features"] = audio_features
            model_inputs["audio_lengths"] = audio_lengths
        return model_inputs

    model.prepare_inputs_for_generation = types.MethodType(_patched_prepare, model)

    print("Qwen3AE model loaded.\n")
    return model, tokenizer


@torch.inference_mode()
def transcribe_batch_qwen3ae(
    waveforms: torch.Tensor,
    lengths: torch.Tensor,
    model,
    tokenizer,
    device: str = "cuda",
    max_new_tokens: int = 256,
    beam_size: int = 1,
    no_repeat_ngram: int = 0,
    no_think: bool = False,
    sample_temp: float = 0.0,
) -> list[str]:
    """Batch transcription using Qwen3AEForCausalLM (ChatML + EOS training).

    Matches the training format in llamafactory/data/omni_dataset.py:
      <|im_start|>system\nYou are a helpful assistant.<|im_end|>
      <|im_start|>user\n<|audio_start|>[audio_pad]*t_audio<|audio_end|>Transcribe the audio to text.<|im_end|>
      <|im_start|>assistant\n<text><eos>
    """
    B = waveforms.shape[0]

    # 1) Resample 16 kHz -> 48 kHz per training config
    waveforms_48k, lengths_48k = [], []
    for i in range(B):
        w = waveforms[i, :lengths[i].item()]
        w_48k = AF.resample(w, orig_freq=16000, new_freq=DAC_SAMPLE_RATE)
        waveforms_48k.append(w_48k)
        lengths_48k.append(w_48k.shape[0])

    audio_lengths = torch.tensor([n // DAC_HOP_LENGTH for n in lengths_48k], dtype=torch.long)

    # 2) Pack raw 48 kHz audio as (B, 1, max_samples)
    max_samples = max(lengths_48k)
    audio_features = torch.zeros(B, 1, max_samples, dtype=torch.bfloat16, device=device)
    for i, w in enumerate(waveforms_48k):
        audio_features[i, 0, :w.shape[0]] = w.to(torch.bfloat16)

    # 3) Build ChatML-wrapped prompt — same wrapper tokens as training
    audio_pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    sys_prompt = "You are a helpful assistant." + (" /no_think" if no_think else "")
    chatml_prefix_ids = tokenizer.encode(
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n<|audio_start|>",
        add_special_tokens=False,
    )
    chatml_mid_ids = tokenizer.encode(
        "<|audio_end|>Transcribe the audio to text.<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )

    seq_lens = [len(chatml_prefix_ids) + al.item() + len(chatml_mid_ids) for al in audio_lengths]
    max_seq_len = max(seq_lens)

    input_ids = torch.full((B, max_seq_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_seq_len), dtype=torch.long, device=device)
    for i in range(B):
        n_frames = audio_lengths[i].item()
        left = max_seq_len - seq_lens[i]
        cursor = left
        input_ids[i, cursor:cursor + len(chatml_prefix_ids)] = torch.tensor(chatml_prefix_ids, device=device)
        cursor += len(chatml_prefix_ids)
        input_ids[i, cursor:cursor + n_frames] = audio_pad_id
        cursor += n_frames
        input_ids[i, cursor:cursor + len(chatml_mid_ids)] = torch.tensor(chatml_mid_ids, device=device)
        attention_mask[i, left:] = 1

    # 4) Generate — model now emits EOS naturally (trained with EOS labels)
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        audio_features=audio_features,
        audio_lengths=audio_lengths.to(device),
        max_new_tokens=max_new_tokens,
        num_beams=beam_size,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    if no_repeat_ngram > 0:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram
    if sample_temp > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = sample_temp
    else:
        gen_kwargs["do_sample"] = False
    output_ids = model.generate(**gen_kwargs)

    # 5) Strip prompt tokens and decode
    new_tokens = output_ids[:, max_seq_len:]
    texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [t.strip().lower() for t in texts]


def evaluate_qwen3ae(
    model,
    tokenizer,
    split: str = "test-clean",
    device: str = "cuda",
    max_samples: int = None,
    max_new_tokens: int = 256,
    beam_size: int = 1,
    batch_size: int = 4,
    num_workers: int = 4,
    out_dir: str = None,
    mls_sample_ratio: float = 0.2,
    no_repeat_ngram: int = 0,
    no_think: bool = False,
    sample_temp: float = 0.0,
) -> float:
    cfg = get_config(ENCODER_NAME)
    max_len = cfg["max_audio_len"]

    dataset = _load_dataset(split, cfg["data_path"], max_len, mls_sample_ratio)
    indices = list(range(min(max_samples, len(dataset)))) if max_samples else list(range(len(dataset)))
    subset = Subset(dataset, indices)

    dataloader = DataLoader(subset, batch_size=batch_size, collate_fn=collate_fn,
                            num_workers=num_workers, pin_memory=True)

    hypotheses, references = [], []
    processed_count = 0
    for waveforms, lengths, transcripts in tqdm(dataloader, desc=f"Evaluating {split}"):
        hyps = transcribe_batch_qwen3ae(
            waveforms, lengths, model, tokenizer, device=device,
            max_new_tokens=max_new_tokens, beam_size=beam_size,
            no_repeat_ngram=no_repeat_ngram, no_think=no_think, sample_temp=sample_temp,
        )
        hypotheses.extend(hyps)
        references.extend(transcripts)
        processed_count += len(hyps)

        if processed_count % (batch_size * 5) < batch_size:
            wer_so_far = compute_wer(hypotheses, references)
            print(f"  [{processed_count}/{len(indices)}] WER so far: {wer_so_far*100:.2f}%")
            print(f"    REF: {references[-1]}")
            print(f"    HYP: {hypotheses[-1]}\n")

    wer = compute_wer(hypotheses, references)
    print(f"\nFinal WER on {split}: {wer*100:.2f}%  ({len(hypotheses)} samples)")

    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    save_results(references, hypotheses, wer, split, out_dir)
    return wer


class AudioQwen(nn.Module):
    def __init__(self, llm_model_name: str, encoder_dim: int, cache_dir: str = None, pad_token_id: int = 151655, 
                 encoder: nn.Module = None):
        super().__init__()
        self.encoder = encoder        

        self.pad_token_id = pad_token_id
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, cache_dir=cache_dir, trust_remote_code=True)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|audio_correspond|>"]})
        self.llm.resize_token_embeddings(len(self.tokenizer))

        self.llm.to(torch.bfloat16)
        llm_dim = self.llm.config.hidden_size
        self.projector = nn.Sequential(
            nn.Conv1d(encoder_dim, llm_dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(llm_dim, llm_dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(llm_dim, llm_dim, kernel_size=1)
        ).to(torch.bfloat16)
        self.proj_norm = nn.LayerNorm(llm_dim).to(torch.bfloat16)
        
        for m in self.projector.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.normal_(self.projector[-1].weight, std=0.02)

    def get_audio_embeds(self, audio_features: torch.Tensor) -> torch.Tensor:
        x = audio_features.transpose(1, 2).to(self.projector[0].weight.dtype)
        x = self.projector(x)
        x = x.transpose(1, 2)
        return self.proj_norm(x) # (B, C, T)

    def get_audio_embeds_from_waveform(self, audio, audio_lengths):
        enc_dtype = next(self.encoder.parameters()).dtype
        x, enc_mask = self.encoder(audio.to(enc_dtype), audio_lengths)  # (B, T_enc, C)
        x = x.to(self.projector[0].weight.dtype)

        # 프로젝터(Conv1d 등) 연산을 위해 (B, C, T)로 변환
        x = self.projector(x.transpose(1, 2))
        
        # LayerNorm 적용 및 LLM 입력을 위해 다시 (B, T, C)로 복구
        x = x.transpose(1, 2)
        
        # 정규화된 임베딩과 마스크를 튜플 형태로 반환
        return self.proj_norm(x), enc_mask

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, audio_features: torch.Tensor, audio_lengths: torch.Tensor, attention_mask: torch.Tensor = None, position_ids: torch.Tensor = None):
        audio_embeds, _ = self.get_audio_embeds_from_waveform(audio_features, audio_lengths)
        valid_embeds = []
        for i, enc_len in enumerate(audio_lengths):
            t_proj = math.ceil(math.ceil(enc_len.item() / 2) / 2)
            valid_embeds.append(audio_embeds[i, :t_proj, :])
        audio_flat = torch.cat(valid_embeds, dim=0) if valid_embeds else torch.empty((0, audio_embeds.shape[-1]), device=audio_embeds.device, dtype=audio_embeds.dtype)
        
        inputs_embeds = self.llm.get_input_embeddings()(input_ids).clone()
        audio_mask = (input_ids == self.pad_token_id)
        mask_flat = audio_mask.reshape(-1)
        num_placeholders = mask_flat.sum().item()
        min_len = min(num_placeholders, audio_flat.shape[0])
        
        if min_len > 0:
            target_indices = mask_flat.nonzero(as_tuple=True)[0][:min_len]
            inputs_embeds.view(-1, inputs_embeds.shape[-1])[target_indices] = audio_flat[:min_len]

        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, labels=labels, use_cache=False)


def load_model(ckpt_dir: str, device: str = "cuda",
               llm_size: str = None, encoder_name: str = None,
               no_lora: bool = False) -> AudioQwen:
    print(f"Loading checkpoint from: {ckpt_dir}")

    enc = encoder_name or ENCODER_NAME
    cfg = get_config(enc)
    if llm_size and llm_size in LLM_MAP:
        cfg["llm_model"] = LLM_MAP[llm_size]
    enc_cfg   = cfg["encoder"]
    cache_dir = cfg["model_cache_dir"]

    from encoders import build_encoder
    encoder = build_encoder(enc, enc_cfg, cache_dir)
    model = AudioQwen("Qwen/Qwen3.5-2B", encoder_dim=128, encoder=encoder).to(device)

    # ── resolve checkpoint path ──
    if ckpt_dir.endswith(".pt"):
        pt_path = ckpt_dir
        sf_path = None
    else:
        merged = os.path.join(ckpt_dir, "full_merged_model.pt")
        pt_path = merged if os.path.exists(merged) else None
        sf_path = os.path.join(ckpt_dir, "model.safetensors")

    # ── load raw state dict ──
    if pt_path:
        raw = torch.load(pt_path, map_location="cpu", weights_only=False)
        state_dict = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    elif sf_path and os.path.exists(sf_path):
        from safetensors.torch import load_file
        state_dict = load_file(sf_path, device="cpu")
    else:
        raise FileNotFoundError(f"checkpoint not found: {ckpt_dir}")

    # ── Stage 2 detection ──
    is_stage2 = any("base_model.model" in k for k in state_dict)

    if is_stage2:
        from peft import LoraConfig, TaskType, get_peft_model
        print("  Detected Stage 2 checkpoint (LoRA). Applying LoRA before loading…")
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16, lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
        )
        model.llm = get_peft_model(model.llm, lora_cfg)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys   : {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    if is_stage2:
        print("  Merging LoRA weights into base model…")
        model.llm = model.llm.merge_and_unload()

    model.to(torch.bfloat16).to(device).eval()
    print("Model loaded.\n")
    return model


# ==========================================
# 2. Audio preprocessing & Dataloading
# ==========================================

def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = AF.resample(waveform, orig_freq=sr, new_freq=target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0)   # (T,)

def collate_fn(batch):
    """Collate function for batching variable-length audio."""
    waveforms = [item[0] for item in batch]
    transcripts = [item[1] for item in batch]
    lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
    max_len = lengths.max().item()
    
    padded_waveforms = torch.zeros(len(waveforms), max_len)
    for i, w in enumerate(waveforms):
        padded_waveforms[i, :w.shape[0]] = w
        
    return padded_waveforms, lengths, transcripts


# ==========================================
# 3. Batch Inference
# ==========================================

@torch.inference_mode()
def transcribe_batch(
    waveforms: torch.Tensor,
    lengths: torch.Tensor,
    model: AudioQwen,
    device: str = "cuda",
    max_new_tokens: int = 256,
    beam_size: int = 1,
) -> list[str]:
    """
    Batch transcription supporting left-padding for reliable HF generation.
    """
    waveforms = waveforms.to(device)
    lengths = lengths.to(device)
    B = waveforms.shape[0]

    # Get embeddings and valid mask lengths
    audio_embeds, audio_mask = model.get_audio_embeds_from_waveform(waveforms, lengths)

    # Prepare <|audio_correspond|> token
    audio_corr_id = model.tokenizer.convert_tokens_to_ids("<|audio_correspond|>")
    embed_layer = model.llm.get_input_embeddings()
    corr_embed = embed_layer(torch.tensor([[audio_corr_id]], device=device))[0] # (1, C)

    # Prepare pad token embedding
    pad_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
    pad_embed = embed_layer(torch.tensor([[pad_id]], device=device))[0][0] # (C,)

    batched_embeds = []
    batched_masks = []
    max_seq_len = 0
    seq_embeds_list = []

    # 1) Append prompt token directly after the valid audio embeddings
    for i in range(B):
        v_len = int(audio_mask[i].sum().item())
        valid_emb = audio_embeds[i, :v_len, :]
        seq_emb = torch.cat([valid_emb, corr_embed], dim=0) # (v_len + 1, C)
        seq_embeds_list.append(seq_emb)
        max_seq_len = max(max_seq_len, seq_emb.shape[0])

    # 2) Left-pad embeddings to max_seq_len for generation compatibility
    for seq_emb in seq_embeds_list:
        seq_len = seq_emb.shape[0]
        pad_len = max_seq_len - seq_len
        
        if pad_len > 0:
            pad_tensor = pad_embed.unsqueeze(0).expand(pad_len, -1)
            padded_emb = torch.cat([pad_tensor, seq_emb], dim=0)
            mask = torch.cat([
                torch.zeros(pad_len, dtype=torch.long, device=device), 
                torch.ones(seq_len, dtype=torch.long, device=device)
            ], dim=0)
        else:
            padded_emb = seq_emb
            mask = torch.ones(seq_len, dtype=torch.long, device=device)
            
        batched_embeds.append(padded_emb)
        batched_masks.append(mask)

    inputs_embeds = torch.stack(batched_embeds, dim=0)
    attention_mask = torch.stack(batched_masks, dim=0)

    # Force left padding in tokenizer setting (just in case)
    model.tokenizer.padding_side = "left"

    output_ids = model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=beam_size,
        do_sample=False,
        pad_token_id=model.tokenizer.pad_token_id,
        eos_token_id=model.tokenizer.eos_token_id,
        use_cache=True,
    )

    texts = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [text.strip().lower() for text in texts]


# ==========================================
# 4. WER evaluation
# ==========================================

def compute_wer(hypotheses: list[str], references: list[str]) -> float:
    import jiwer
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    return jiwer.wer(
        reference=references,
        hypothesis=hypotheses,
        reference_transform=transform,
        hypothesis_transform=transform,
    )

def save_results(references: list[str], hypotheses: list[str], wer: float,
                 split: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{split}_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "reference", "hypothesis"])
        for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
            writer.writerow([i, ref, hyp])

    txt_path = os.path.join(out_dir, f"{split}_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"WER: {wer*100:.2f}%  ({len(references)} samples)\n")
        f.write("=" * 80 + "\n\n")
        for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
            f.write(f"[{i:04d}]\nREF: {ref}\nHYP: {hyp}\n\n")

    md_path = os.path.join(out_dir, f"{split}_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# ASR Results — {split}\n\n")
        f.write(f"**WER: {wer*100:.2f}%** ({len(references)} samples)\n\n")
        f.write("| idx | Reference (GT) | Hypothesis (Pred) |\n|----:|:---------------|:------------------|\n")
        for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
            ref_escaped = ref.replace('|', '\\|')
            hyp_escaped = hyp.replace('|', '\\|')

            f.write(f"| {i} | {ref_escaped} | {hyp_escaped} |\n")

    print(f"Results saved to: {out_dir}")

MLS_TOTAL = 4_050_000

def _load_dataset(split: str, data_root: str, max_len: int, mls_sample_ratio: float = 0.2):
    cfg = get_config(ENCODER_NAME)
    if split == "mls":
        from dataset import MLSDataset
        num_samples = int(MLS_TOTAL * mls_sample_ratio)
        return MLSDataset(cache_dir=cfg["mls_data_path"], num_samples=num_samples, max_len=max_len)
    else:
        return LibriSpeechDataset(cache_dir=cfg["mls_data_path"], url=split, max_len=max_len)

def evaluate(
    model: AudioQwen,
    split: str = "test-clean",
    data_root: str = None,
    device: str = "cuda",
    max_samples: int = None,
    max_new_tokens: int = 256,
    beam_size: int = 1,
    batch_size: int = 8,
    num_workers: int = 4,
    out_dir: str = None,
    mls_sample_ratio: float = 0.2,
) -> float:
    cfg = get_config(ENCODER_NAME)
    data_root = data_root or cfg["data_path"]
    max_len = cfg["max_audio_len"]
    
    dataset = _load_dataset(split, data_root, max_len, mls_sample_ratio)
    indices = list(range(min(max_samples, len(dataset)))) if max_samples else list(range(len(dataset)))
    subset = Subset(dataset, indices)
    
    dataloader = DataLoader(
        subset, 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        num_workers=num_workers, 
        pin_memory=True
    )

    hypotheses, references = [], []
    processed_count = 0

    for waveforms, lengths, transcripts in tqdm(dataloader, desc=f"Evaluating {split}"):
        hyps = transcribe_batch(
            waveforms, lengths, model, device=device,
            max_new_tokens=max_new_tokens, beam_size=beam_size
        )
        
        hypotheses.extend(hyps)
        references.extend(transcripts)
        processed_count += len(hyps)

        if processed_count % (batch_size * 5) < batch_size:
            wer_so_far = compute_wer(hypotheses, references)
            print(f"  [{processed_count}/{len(indices)}] WER so far: {wer_so_far*100:.2f}%")
            print(f"    REF: {references[-1]}")
            print(f"    HYP: {hypotheses[-1]}\n")

    wer = compute_wer(hypotheses, references)
    print(f"\nFinal WER on {split}: {wer*100:.2f}%  ({len(hypotheses)} samples)")

    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    save_results(references, hypotheses, wer, split, out_dir)
    return wer


# ==========================================
# 5. Multi-GPU evaluation worker
# ==========================================

def _worker(rank: int, args_dict: dict, shards: list, tmp_dir: str):
    indices = shards[rank]
    device = f"cuda:{rank}"

    model = load_model(
        args_dict["ckpt"], device=device,
        llm_size=args_dict["llm"], encoder_name=args_dict["encoder"],
        no_lora=args_dict["no_lora"],
    )

    cfg = get_config(ENCODER_NAME)
    data_root = args_dict["data_root"] or cfg["data_path"]
    max_len = cfg["max_audio_len"]
    dataset = _load_dataset(args_dict["split"], data_root, max_len, args_dict["mls_sample_ratio"])
    
    subset = Subset(dataset, indices)
    dataloader = DataLoader(
        subset, 
        batch_size=args_dict["batch_size"], 
        collate_fn=collate_fn, 
        num_workers=args_dict["num_workers"], 
        pin_memory=True
    )

    results = [] 
    idx_ptr = 0
    
    for waveforms, lengths, transcripts in tqdm(dataloader, desc=f"GPU {rank}", position=rank):
        hyps = transcribe_batch(
            waveforms, lengths, model, device=device,
            max_new_tokens=args_dict["max_new_tokens"],
            beam_size=args_dict["beam_size"]
        )
        
        for hyp, ref in zip(hyps, transcripts):
            orig_idx = indices[idx_ptr]
            results.append((orig_idx, ref, hyp))
            idx_ptr += 1

    out_path = os.path.join(tmp_dir, f"rank{rank}.json")
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"[GPU {rank}] Done. Saved {len(results)} results → {out_path}")


def evaluate_multi_gpu(args) -> float:
    n_gpu = args.n_gpu
    if n_gpu <= 0:
        n_gpu = torch.cuda.device_count()
    assert n_gpu > 0, "No GPUs available"

    cfg = get_config(ENCODER_NAME)
    data_root = args.data_root or cfg["data_path"]
    max_len = cfg["max_audio_len"]
    mls_sample_ratio = getattr(args, "mls_sample_ratio", 0.2)
    dataset = _load_dataset(args.split, data_root, max_len, mls_sample_ratio)
    
    total = min(args.max_samples, len(dataset)) if args.max_samples else len(dataset)
    all_indices = list(range(total))
    shards = [all_indices[rank::n_gpu] for rank in range(n_gpu)]

    args_dict = {
        "ckpt": args.ckpt, "llm": args.llm, "encoder": args.encoder,
        "no_lora": args.no_lora, "split": args.split, "data_root": args.data_root,
        "max_new_tokens": args.max_new_tokens, "beam_size": args.beam_size,
        "mls_sample_ratio": mls_sample_ratio, "batch_size": args.batch_size,
        "num_workers": args.num_workers
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        mp.spawn(_worker, args=(args_dict, shards, tmp_dir), nprocs=n_gpu, join=True)

        merged = []
        for rank in range(n_gpu):
            with open(os.path.join(tmp_dir, f"rank{rank}.json")) as f:
                merged.extend(json.load(f))

    merged.sort(key=lambda x: x[0])
    references  = [r for _, r, _ in merged]
    hypotheses  = [h for _, _, h in merged]

    wer = compute_wer(hypotheses, references)
    print(f"\nFinal WER on {args.split}: {wer*100:.2f}%  ({len(hypotheses)} samples)")

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    save_results(references, hypotheses, wer, args.split, out_dir)
    return wer


# ==========================================
# 6. Main
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="ASR inference — DAC-VAE + AudioQwen")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_gpu", type=int, default=0)
    parser.add_argument("--llm", type=str, default=None, choices=["2b", "4b", "7b"])
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--no_lora", action="store_true")

    # Single-file mode
    parser.add_argument("--audio", type=str, default=None)

    # Evaluation mode
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--split", type=str, default="test-clean")
    parser.add_argument("--mls_sample_ratio", type=float, default=0.2)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default=None)
    
    # Batch parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")

    # Qwen3AE model
    parser.add_argument("--qwen3ae", action="store_true", help="Use Qwen3AE model")
    parser.add_argument("--qwen3ae_model_dir", type=str, default=QWEN3AE_MODEL_DIR,
                        help="Path to Qwen3AE checkpoint directory")
    parser.add_argument("--dacvae_path", type=str, default=DACVAE_PKG_PATH,
                        help="Local dacvae package path to prepend to sys.path (needed by Qwen3AE audio_encoder.py)")
    parser.add_argument("--no_repeat_ngram", type=int, default=0, help="no_repeat_ngram_size (0=off)")
    parser.add_argument("--no_think", action="store_true", help="Append /no_think to system prompt")
    parser.add_argument("--sample_temp", type=float, default=0.0, help="Sampling temperature (0=greedy)")

    # Checkpoint sweep mode: iterate over every checkpoint-* dir under --ckpt_root and
    # run --qwen3ae --eval on each, saving results into {out_dir}/{ckpt_name}/.
    parser.add_argument("--ckpt_root", type=str, default=None,
                        help="Root dir containing checkpoint-* subdirs. Implies --qwen3ae --eval loop.")
    parser.add_argument("--ckpt_filter", type=str, default=None,
                        help="Comma-separated step numbers to include (e.g. 45000,46000). Default: all.")
    parser.add_argument("--skip_if_done", action="store_true",
                        help="Skip checkpoints whose {out_dir}/{ckpt_name}/{split}_results.csv already exists.")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model dir for adapter-only (LoRA) checkpoints. If omitted, "
                             "we fall back to adapter_config.json's base_model_name_or_path.")
    parser.add_argument("--min_age_sec", type=int, default=60,
                        help="Skip checkpoints whose safetensors mtime is younger than this (save in progress).")
    return parser.parse_args()


def _find_ckpt_dirs(root: str, steps_filter=None, min_age_sec: int = 60):
    """Return checkpoint-* dirs sorted by step, filtering partial/incomplete saves."""
    import time as _time
    ckpts = []
    for name in sorted(os.listdir(root)):
        m = re.match(r"checkpoint-(\d+)$", name)
        if not m:
            continue
        step = int(m.group(1))
        if steps_filter is not None and step not in steps_filter:
            continue
        path = os.path.join(root, name)
        # Pick an index/weight file to check completeness
        idx = os.path.join(path, "model.safetensors.index.json")
        mono = os.path.join(path, "model.safetensors")
        adapter = os.path.join(path, "adapter_model.safetensors")  # LoRA / PEFT adapter-only
        adapter_cfg = os.path.join(path, "adapter_config.json")
        ref = None
        if os.path.exists(idx):
            ref = idx
        elif os.path.exists(mono):
            ref = mono
        elif os.path.exists(adapter) and os.path.exists(adapter_cfg):
            ref = adapter  # adapter-only is valid for Stage 2 LoRA sweeps
        if ref is None:
            print(f"[sweep] skip {name}: no safetensors (no model.* or adapter_model.*)", flush=True)
            continue
        age = _time.time() - os.path.getmtime(ref)
        if age < min_age_sec:
            print(f"[sweep] skip {name}: save in progress (age={age:.0f}s)", flush=True)
            continue
        ckpts.append((step, path))
    ckpts.sort(key=lambda x: x[0])
    return [p for _, p in ckpts]


def _run_sweep(args):
    """Loop every checkpoint-* under args.ckpt_root; run qwen3ae eval on each."""
    import gc
    import time as _time
    if args.split == "mls":
        raise SystemExit("--ckpt_root sweep only supports LibriSpeech-style splits, not mls")

    steps_filter = None
    if args.ckpt_filter:
        steps_filter = {int(s.strip()) for s in args.ckpt_filter.split(",") if s.strip()}

    ckpt_dirs = _find_ckpt_dirs(args.ckpt_root, steps_filter, args.min_age_sec)
    if not ckpt_dirs:
        raise SystemExit(f"[sweep] no checkpoints found under {args.ckpt_root}")

    base_out = args.out_dir or os.path.join(args.ckpt_root, f"eval_{args.split}")
    os.makedirs(base_out, exist_ok=True)
    print(f"[sweep] {len(ckpt_dirs)} checkpoints, results → {base_out}", flush=True)

    summary_path = os.path.join(base_out, "summary.jsonl")
    summary_f = open(summary_path, "a")

    for ck in ckpt_dirs:
        name = os.path.basename(ck)
        ck_out = os.path.join(base_out, name)
        os.makedirs(ck_out, exist_ok=True)
        done_marker = os.path.join(ck_out, f"{args.split}_results.csv")
        if args.skip_if_done and os.path.exists(done_marker):
            print(f"[sweep] {name}: already done, skipping", flush=True)
            continue

        print(f"\n[sweep] === {name} ===", flush=True)
        t0 = _time.time()
        model, tokenizer = load_qwen3ae_model(
            device=args.device,
            model_dir=ck,
            dacvae_path=args.dacvae_path,
            base_model_dir=args.base_model,
        )
        try:
            wer = evaluate_qwen3ae(
                model, tokenizer,
                split=args.split,
                device=args.device,
                max_samples=args.max_samples,
                max_new_tokens=args.max_new_tokens,
                beam_size=args.beam_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                out_dir=ck_out,
                mls_sample_ratio=args.mls_sample_ratio,
                no_repeat_ngram=args.no_repeat_ngram,
                no_think=args.no_think,
                sample_temp=args.sample_temp,
            )
        finally:
            # Free GPU memory between ckpts
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        dt = _time.time() - t0
        rec = {"checkpoint": name, "path": ck, "split": args.split,
               "wer": float(wer), "elapsed_sec": dt,
               "max_samples": args.max_samples, "batch_size": args.batch_size,
               "no_repeat_ngram": args.no_repeat_ngram, "no_think": args.no_think}
        summary_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        summary_f.flush()
        print(f"[sweep] {name}: WER={wer*100:.2f}%  ({dt/60:.1f} min)", flush=True)

    summary_f.close()
    print(f"\n[sweep] summary → {summary_path}", flush=True)


if __name__ == "__main__":
    args = parse_args()

    # Checkpoint sweep mode takes precedence
    if args.ckpt_root is not None:
        if not args.eval:
            raise SystemExit("--ckpt_root requires --eval")
        # --qwen3ae is implicit in sweep (only supported model path here)
        _run_sweep(args)
        sys.exit(0)

    if args.qwen3ae:
        # ── Qwen3AE-4B_expand path ──
        model, tokenizer = load_qwen3ae_model(
            device=args.device,
            model_dir=args.qwen3ae_model_dir,
            dacvae_path=args.dacvae_path,
            base_model_dir=args.base_model,
        )

        if args.audio:
            waveform = load_audio(args.audio).unsqueeze(0)  # (1, T)
            lengths = torch.tensor([waveform.shape[1]], dtype=torch.long)
            result = transcribe_batch_qwen3ae(
                waveform, lengths, model, tokenizer, device=args.device,
                max_new_tokens=args.max_new_tokens, beam_size=args.beam_size,
            )
            print(f"Transcript: {result[0]}")

        elif args.eval:
            evaluate_qwen3ae(
                model, tokenizer,
                split=args.split,
                device=args.device,
                max_samples=args.max_samples,
                max_new_tokens=args.max_new_tokens,
                beam_size=args.beam_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                out_dir=args.out_dir,
                mls_sample_ratio=args.mls_sample_ratio,
                no_repeat_ngram=args.no_repeat_ngram,
                no_think=args.no_think,
                sample_temp=args.sample_temp,
            )
        else:
            print("Specify --audio <file> for transcription or --eval for WER evaluation.")

    elif args.audio:
        model = load_model(args.ckpt, device=args.device, llm_size=args.llm,
                           encoder_name=args.encoder, no_lora=args.no_lora)
        waveform = load_audio(args.audio).unsqueeze(0)  # (1, T)
        lengths = torch.tensor([waveform.shape[1]], dtype=torch.long)

        result = transcribe_batch(waveform, lengths, model, device=args.device,
                                  max_new_tokens=args.max_new_tokens,
                                  beam_size=args.beam_size)
        print(f"Transcript: {result[0]}")

    elif args.eval:
        n_gpu = args.n_gpu if args.n_gpu > 0 else torch.cuda.device_count()
        if n_gpu > 1:
            print(f"Multi-GPU evaluation: {n_gpu} GPUs with batch size {args.batch_size}")
            evaluate_multi_gpu(args)
        else:
            model = load_model(args.ckpt, device=args.device, llm_size=args.llm,
                               encoder_name=args.encoder, no_lora=args.no_lora)
            evaluate(
                model,
                split=args.split,
                data_root=args.data_root,
                device=args.device,
                max_samples=args.max_samples,
                max_new_tokens=args.max_new_tokens,
                beam_size=args.beam_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                out_dir=args.out_dir,
            )
    else:
        print("Specify --audio <file> for transcription or --eval for WER evaluation.")