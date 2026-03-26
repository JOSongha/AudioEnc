"""
LibriSpeech WER/CER 평가 스크립트.

사용법:
    python eval.py --encoder encodec --split test-clean
    python eval.py --encoder mimi_semantic --split test-clean --ckpt /path/to/model.safetensors
    python eval.py --encoder dac --split test-other --batch-size 4
"""

import argparse
import os
import warnings

import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from encoders import build_encoder
from model import AudioQwen

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==========================================
# WER / CER
# ==========================================

def _edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def _normalize(text: str) -> str:
    import string
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def compute_wer(references, hypotheses):
    try:
        import jiwer
        tr = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.Strip(),
            jiwer.RemoveMultipleSpaces(),
        ])
        return jiwer.wer(references, hypotheses,
                         truth_transform=tr, hypothesis_transform=tr)
    except ImportError:
        total_edits, total_words = 0, 0
        for ref, hyp in zip(references, hypotheses):
            r, h = _normalize(ref).split(), _normalize(hyp).split()
            total_edits += _edit_distance(r, h)
            total_words += len(r)
        return total_edits / max(total_words, 1)


def compute_cer(references, hypotheses):
    try:
        import jiwer
        tr = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.Strip(),
        ])
        return jiwer.cer(references, hypotheses,
                         truth_transform=tr, hypothesis_transform=tr)
    except ImportError:
        total_edits, total_chars = 0, 0
        for ref, hyp in zip(references, hypotheses):
            r = list(_normalize(ref).replace(" ", ""))
            h = list(_normalize(hyp).replace(" ", ""))
            total_edits += _edit_distance(r, h)
            total_chars += len(r)
        return total_edits / max(total_chars, 1)


# ==========================================
# Data
# ==========================================

def collate_fn(batch, max_audio_len: int, target_sr: int = 16000):
    """LibriSpeech: (waveform, sr, transcript, speaker_id, chapter_id, utterance_id)"""
    audios, lengths, transcripts = [], [], []
    for waveform, sr, transcript, *_ in batch:
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        waveform = waveform.squeeze(0)  # (T,)
        waveform = waveform[:max_audio_len]
        audios.append(waveform)
        lengths.append(waveform.shape[0])
        transcripts.append(transcript.lower())

    max_len = max(lengths)
    audio_padded = torch.zeros(len(audios), max_len)
    for i, a in enumerate(audios):
        audio_padded[i, :len(a)] = a

    return audio_padded, torch.tensor(lengths), transcripts


# ==========================================
# Inference
# ==========================================

@torch.no_grad()
def generate_transcripts(model, audio, audio_lengths, max_new_tokens: int = 256):
    device = next(model.parameters()).device
    audio = audio.to(device)
    audio_lengths = audio_lengths.to(device)

    audio_embeds, audio_mask = model._get_audio_embeds(audio, audio_lengths)
    B = audio_embeds.shape[0]
    embed = model.llm.get_input_embeddings()

    p1_embeds = embed(model.prompt_p1_ids).expand(B, -1, -1)
    p2_embeds = embed(model.prompt_p2_ids).expand(B, -1, -1)
    inputs_embeds = torch.cat([p1_embeds, audio_embeds, p2_embeds], dim=1)

    p1_mask   = torch.ones(B, p1_embeds.shape[1], device=device, dtype=torch.long)
    p2_mask   = torch.ones(B, p2_embeds.shape[1], device=device, dtype=torch.long)
    attn_mask = torch.cat([p1_mask, audio_mask.long(), p2_mask], dim=1)

    out_ids = model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=model.tokenizer.pad_token_id,
        eos_token_id=model.tokenizer.eos_token_id,
    )

    return [
        model.tokenizer.decode(ids, skip_special_tokens=True).strip()
        for ids in out_ids
    ]


# ==========================================
# Model loading
# ==========================================

def load_model(cfg, ckpt_path=None):
    enc_name  = cfg["encoder_name"]
    cache_dir = cfg["model_cache_dir"]

    encoder = build_encoder(enc_name, cfg["encoder"], cache_dir)
    model   = AudioQwen(encoder, cfg)
    model.apply_lora()

    if ckpt_path:
        _load_safetensors(model, ckpt_path)
    else:
        # 자동 탐색: best_{enc}_ckpt_ep*_step*/ 패턴으로 가장 최근 것 선택
        import glob
        pattern = os.path.join(cache_dir, f"best_{enc_name}_ckpt_ep*_step*", "model.safetensors")
        candidates = sorted(glob.glob(pattern))
        found = candidates[-1] if candidates else None
        if found:
            _load_safetensors(model, found)
        else:
            print("Warning: no checkpoint found, using random weights")

    return model


def _load_safetensors(model, path):
    from safetensors.torch import load_file
    state = load_file(path, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded: {path}")
    if missing:
        print(f"  Missing keys : {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True,
                        choices=["encodec", "dac", "dac_vae", "fb_dacvae",
                                 "mimi_acoustic", "mimi_semantic"])
    parser.add_argument("--split", default="test-clean",
                        choices=["test-clean", "test-other", "dev-clean", "dev-other"])
    parser.add_argument("--ckpt",           default=None,  help="safetensors 체크포인트 경로")
    parser.add_argument("--data-path",      default=None,  help="LibriSpeech 루트")
    parser.add_argument("--cache-dir",      default=None,  help="모델 캐시 경로")
    parser.add_argument("--batch-size",     type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-samples",    type=int, default=None, help="평가 샘플 수 제한")
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--llm",            default=None, choices=["4b", "2b"])
    parser.add_argument("--show-examples",  type=int, default=5, help="출력할 예시 수")
    args = parser.parse_args()

    cfg = get_config(args.encoder)
    if args.llm == "2b":   cfg["llm_model"]      = "Qwen/Qwen3.5-2B"
    if args.data_path:     cfg["data_path"]       = args.data_path
    if args.cache_dir:     cfg["model_cache_dir"] = args.cache_dir

    os.environ.setdefault("HF_HOME",    cfg["model_cache_dir"])
    os.environ.setdefault("TORCH_HOME", os.path.join(
        os.path.dirname(cfg["model_cache_dir"]), "torch"))

    # Dataset
    print(f"Loading LibriSpeech {args.split} ...")
    dataset = torchaudio.datasets.LIBRISPEECH(
        root=cfg["data_path"], url=args.split, download=True,
    )
    if args.num_samples:
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(min(args.num_samples, len(dataset))))
    print(f"  {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda b: collate_fn(b, max_audio_len=cfg["max_audio_len"]),
    )

    # Model
    model = load_model(cfg, args.ckpt)
    model.eval()
    model.to(args.device)

    # Evaluate
    all_refs, all_hyps = [], []
    for audio, audio_lengths, refs in tqdm(loader, desc=f"Eval [{args.split}]"):
        hyps = generate_transcripts(model, audio, audio_lengths, args.max_new_tokens)
        all_refs.extend(refs)
        all_hyps.extend(hyps)

    wer = compute_wer(all_refs, all_hyps)
    cer = compute_cer(all_refs, all_hyps)

    print(f"\n{'='*45}")
    print(f"Encoder  : {args.encoder}")
    print(f"Split    : {args.split}")
    print(f"Samples  : {len(all_refs)}")
    print(f"WER      : {wer * 100:.2f}%")
    print(f"CER      : {cer * 100:.2f}%")
    print(f"{'='*45}")

    if args.show_examples > 0:
        print(f"\nExamples (first {args.show_examples}):")
        for i in range(min(args.show_examples, len(all_refs))):
            print(f"  [{i+1}] REF: {all_refs[i]}")
            print(f"       HYP: {all_hyps[i]}")
            print()


if __name__ == "__main__":
    main()
