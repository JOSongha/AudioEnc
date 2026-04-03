"""
inference.py — ASR inference & WER evaluation (DAC-VAE encoder + AudioQwen)

Usage:
    # Single file transcription
    python inference.py --audio path/to/audio.wav

    # WER evaluation on LibriSpeech test-clean
    python inference.py --eval --split test-clean

    # Use a specific checkpoint
    python inference.py --eval --ckpt /mnt/ddn/users/sehyun/ckpts/best_dac_vae_ckpt

    # Multi-GPU evaluation (4 GPUs)
    python inference.py --eval --n_gpu 4
"""

import argparse
import csv
import json
import os
import sys
import tempfile

import torch
import torch.multiprocessing as mp
import torchaudio
import torchaudio.functional as AF
from tqdm import tqdm

from config import get_config
from dataset import LibriSpeechDataset
from encoders.fb_dacvae import FbDACVAEEncoder
from model import AudioQwen

ENCODER_NAME = "fb_dacvae"
DEFAULT_CKPT = "/mnt/tmp/cache/hf/s1_proj_fb_dacvae.pt"

LLM_MAP = {
    "2b":   "Qwen/Qwen3.5-2B",
    "4b":   "Qwen/Qwen3.5-4B",
    "7b":   "Qwen/Qwen2.5-7B-Instruct",
}


# ==========================================
# 1. Model loading
# ==========================================

def load_model(ckpt_dir: str, device: str = "cuda",
               llm_size: str = None, encoder_name: str = None,
               no_lora: bool = False) -> AudioQwen:
    """
    AudioEncoder + AudioQwen 구성 후 checkpoint 로드.
    --no_lora: Stage 1 projector-only checkpoint (model.safetensors 없이 .pt 파일도 지원)
    """
    print(f"Loading checkpoint from: {ckpt_dir}")

    enc = encoder_name or ENCODER_NAME
    cfg = get_config(enc)
    if llm_size and llm_size in LLM_MAP:
        cfg["llm_model"] = LLM_MAP[llm_size]
    enc_cfg   = cfg["encoder"]
    cache_dir = cfg["model_cache_dir"]

    from encoders import build_encoder
    encoder = build_encoder(enc, enc_cfg, cache_dir)
    model   = AudioQwen(encoder, cfg)
    if not no_lora:
        model.apply_lora()

    # .pt (Stage 1) 또는 model.safetensors (Stage 2) 로드
    pt_path  = ckpt_dir if ckpt_dir.endswith(".pt") else None
    sf_path  = os.path.join(ckpt_dir, "model.safetensors") if not pt_path else None

    if pt_path:
        state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
    elif sf_path and os.path.exists(sf_path):
        from safetensors.torch import load_file
        state_dict = load_file(sf_path, device="cpu")
    else:
        raise FileNotFoundError(f"checkpoint not found: {ckpt_dir}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys   : {len(missing)}")
        for k in missing[:5]:
            print(f"    {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    {k}")

    model.to(device)
    model.eval()
    print("Model loaded.\n")
    return model


# ==========================================
# 2. Audio preprocessing
# ==========================================

def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load audio file → mono waveform at 16kHz, shape (T,)."""
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = AF.resample(waveform, orig_freq=sr, new_freq=target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0)   # (T,)


# ==========================================
# 3. Inference
# ==========================================

@torch.inference_mode()
def transcribe(
    waveform: torch.Tensor,
    model: AudioQwen,
    device: str = "cuda",
    max_new_tokens: int = 256,
    beam_size: int = 1,
) -> str:
    """
    waveform: (T,) mono at 16kHz
    Returns decoded transcript string.
    """
    audio        = waveform.unsqueeze(0).to(device)                  # (1, T)
    audio_lengths = torch.tensor([waveform.shape[0]], device=device)

    audio_embeds, audio_mask = model._get_audio_embeds(audio, audio_lengths)
    # (1, T_proj, llm_dim), (1, T_proj) bool

    embed     = model.llm.get_input_embeddings()
    p1_embeds = embed(model.prompt_p1_ids)   # (1, L1, dim)
    p2_embeds = embed(model.prompt_p2_ids)   # (1, L2, dim)

    inputs_embeds = torch.cat([p1_embeds, audio_embeds, p2_embeds], dim=1)

    p1_mask      = torch.ones(1, p1_embeds.shape[1], dtype=torch.long, device=device)
    p2_mask      = torch.ones(1, p2_embeds.shape[1], dtype=torch.long, device=device)
    attention_mask = torch.cat([p1_mask, audio_mask.long(), p2_mask], dim=1)

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

    text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text.strip()


# ==========================================
# 4. WER evaluation
# ==========================================

def compute_wer(hypotheses: list[str], references: list[str]) -> float:
    """Word error rate via jiwer."""
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
    """Save REF/HYP results as CSV, TXT, and Markdown."""
    os.makedirs(out_dir, exist_ok=True)

    # --- CSV ---
    csv_path = os.path.join(out_dir, f"{split}_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "reference", "hypothesis"])
        for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
            writer.writerow([i, ref, hyp])
    print(f"Saved CSV  → {csv_path}")

    # --- TXT ---
    txt_path = os.path.join(out_dir, f"{split}_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"WER: {wer*100:.2f}%  ({len(references)} samples)\n")
        f.write("=" * 80 + "\n\n")
        for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
            f.write(f"[{i:04d}]\n")
            f.write(f"REF: {ref}\n")
            f.write(f"HYP: {hyp}\n\n")
    print(f"Saved TXT  → {txt_path}")

    # --- Markdown ---
    md_path = os.path.join(out_dir, f"{split}_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# ASR Results — {split}\n\n")
        f.write(f"**WER: {wer*100:.2f}%** ({len(references)} samples)\n\n")
        f.write("| idx | Reference (GT) | Hypothesis (Pred) |\n")
        f.write("|----:|:---------------|:------------------|\n")
        for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
            ref_esc = ref.replace("|", "\\|")
            hyp_esc = hyp.replace("|", "\\|")
            f.write(f"| {i} | {ref_esc} | {hyp_esc} |\n")
    print(f"Saved MD   → {md_path}")


LIBRISPEECH_SPLITS = [
    "test-clean", "test-other", "dev-clean", "dev-other",
    "train-clean-100", "train-clean-360", "train-other-500",
]

MLS_TOTAL = 4_050_000


def _load_dataset(split: str, data_root: str, max_len: int, mls_sample_ratio: float = 0.2):
    """Load LibriSpeech or MLS dataset depending on split."""
    if split == "mls":
        from dataset import MLSDataset
        cfg = get_config(ENCODER_NAME)
        num_samples = int(MLS_TOTAL * mls_sample_ratio)
        return MLSDataset(cache_dir=cfg["mls_data_path"], num_samples=num_samples, max_len=max_len)
    else:
        return LibriSpeechDataset(root=data_root, url=split, max_len=max_len)


def evaluate(
    model: AudioQwen,
    split: str = "test-clean",
    data_root: str = None,
    device: str = "cuda",
    max_samples: int = None,
    max_new_tokens: int = 256,
    beam_size: int = 1,
    out_dir: str = None,
    mls_sample_ratio: float = 0.2,
) -> float:
    """Run WER evaluation on a LibriSpeech split or MLS (single GPU)."""
    cfg = get_config(ENCODER_NAME)
    if data_root is None:
        data_root = cfg["data_path"]

    max_len = cfg["max_audio_len"]
    dataset = _load_dataset(split, data_root, max_len, mls_sample_ratio)
    indices = range(min(max_samples, len(dataset))) if max_samples else range(len(dataset))

    hypotheses, references = [], []

    for i in tqdm(indices, desc=f"Evaluating {split}"):
        waveform, transcript = dataset[i]
        # waveform: (T,) @ 16kHz mono, transcript: already lowercased

        hyp = transcribe(waveform, model, device=device,
                         max_new_tokens=max_new_tokens, beam_size=beam_size)
        ref = transcript

        hypotheses.append(hyp)
        references.append(ref)

        if (i + 1) % 50 == 0:
            wer_so_far = compute_wer(hypotheses, references)
            print(f"  [{i+1}/{len(indices)}] WER so far: {wer_so_far*100:.2f}%")
            print(f"    REF: {ref}")
            print(f"    HYP: {hyp}\n")

    wer = compute_wer(hypotheses, references)
    print(f"\nFinal WER on {split}: {wer*100:.2f}%  ({len(hypotheses)} samples)")

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    save_results(references, hypotheses, wer, split, out_dir)

    return wer


# ==========================================
# 5. Multi-GPU evaluation worker
# ==========================================

def _worker(rank: int, args_dict: dict, shards: list, tmp_dir: str):
    """Each GPU worker processes its shard of indices and saves results to tmp_dir."""
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

    results = []  # list of (original_idx, ref, hyp)
    for i in tqdm(indices, desc=f"GPU {rank}", position=rank):
        waveform, transcript = dataset[i]
        hyp = transcribe(waveform, model, device=device,
                         max_new_tokens=args_dict["max_new_tokens"],
                         beam_size=args_dict["beam_size"])
        results.append((i, transcript, hyp))

    out_path = os.path.join(tmp_dir, f"rank{rank}.json")
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"[GPU {rank}] Done. Saved {len(results)} results → {out_path}")


def evaluate_multi_gpu(args) -> float:
    """Distribute evaluation across multiple GPUs using multiprocessing."""
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

    # Shard indices across GPUs (round-robin keeps load balanced)
    shards = [all_indices[rank::n_gpu] for rank in range(n_gpu)]

    args_dict = {
        "ckpt": args.ckpt, "llm": args.llm, "encoder": args.encoder,
        "no_lora": args.no_lora, "split": args.split, "data_root": args.data_root,
        "max_new_tokens": args.max_new_tokens, "beam_size": args.beam_size,
        "mls_sample_ratio": mls_sample_ratio,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        mp.spawn(_worker, args=(args_dict, shards, tmp_dir), nprocs=n_gpu, join=True)

        # Merge results in original index order
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
# 5. Main
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="ASR inference — DAC-VAE + AudioQwen")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT,
                        help="Accelerate checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_gpu", type=int, default=0,
                        help="Number of GPUs for evaluation (0 = use all available)")
    parser.add_argument("--llm", type=str, default=None, choices=["2b", "4b", "7b"],
                        help="LLM 크기: 2b=Qwen3.5-2B, 4b=Qwen3.5-4B, 7b=Qwen2.5-7B-Instruct")
    parser.add_argument("--encoder", type=str, default=None,
                        help="인코더 이름 (dac_vae, mimi_semantic 등, 기본: dac_vae)")
    parser.add_argument("--no_lora", action="store_true",
                        help="Stage 1 projector-only checkpoint (LoRA 없음)")

    # Single-file mode
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to audio file for single transcription")

    # Evaluation mode
    parser.add_argument("--eval", action="store_true",
                        help="Run WER evaluation on LibriSpeech")
    parser.add_argument("--split", type=str, default="test-clean",
                        choices=["test-clean", "test-other", "dev-clean", "dev-other",
                                 "train-clean-100", "train-clean-360", "train-other-500", "mls"])
    parser.add_argument("--mls_sample_ratio", type=float, default=0.2,
                        help="MLS dataset 샘플링 비율 (기본 0.2 = 20%%)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="LibriSpeech root dir (default: config data_path)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit evaluation samples (None = full split)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--beam_size", type=int, default=1, help="1 = greedy")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Directory to save results (default: ./results)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.audio:
        model = load_model(args.ckpt, device=args.device, llm_size=args.llm,
                           encoder_name=args.encoder, no_lora=args.no_lora)
        waveform = load_audio(args.audio)
        result = transcribe(waveform, model, device=args.device,
                            max_new_tokens=args.max_new_tokens,
                            beam_size=args.beam_size)
        print(f"Transcript: {result}")

    elif args.eval:
        n_gpu = args.n_gpu if args.n_gpu > 0 else torch.cuda.device_count()
        if n_gpu > 1:
            print(f"Multi-GPU evaluation: {n_gpu} GPUs")
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
                out_dir=args.out_dir,
            )

    else:
        print("Specify --audio <file> for transcription or --eval for WER evaluation.")
        print("Example:")
        print("  python inference.py --audio sample.wav")
        print("  python inference.py --eval --split test-clean --n_gpu 4")
