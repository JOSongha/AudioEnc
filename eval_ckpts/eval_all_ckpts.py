"""
eval_all_ckpts.py — 한 run 디렉토리 안의 모든 stage2 체크포인트에 대해
LibriSpeech dev-set WER를 측정하고 ref/hyp pair 를 JSON 으로 저장.

결과 디렉토리 구조:
    eval_ckpts/results/<run_tag>/
        summary.json                   ← 전체 ckpt WER 요약
        <ckpt_name>.json               ← ckpt 별 ref/hyp pair
        index.html                     ← 비교 UI (build_viewer.py 가 생성)

Usage:
    cd /mnt/fr20tb/wbl_residency/jos/AudioEnc
    /mnt/ddn/users/jos/miniforge3/envs/audio/bin/python eval_ckpts/eval_all_ckpts.py \
        --run-dir /mnt/tmp/cache/hf/fb_dacvae/s2_outputs_0414_1442 \
        --split dev-clean \
        --max-samples 200 \
        --gpu 0

학습이 돌고 있는 상황에서는 --gpu 로 한 GPU 만 점유하고 --max-samples 로
표본 수를 제한해 실행 시간을 짧게 유지하는 것을 권장.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import get_config  # noqa: E402
from dataset import LibriSpeechDataset  # noqa: E402
from encoders import build_encoder  # noqa: E402
from inference import compute_wer  # noqa: E402
from model import AudioQwen  # noqa: E402


# §42: train_pipeline_override.py 에서 `<|audio_correspond|>` special token
# 추가/resize_token_embeddings 를 제거했음. 따라서 eval 측도 추가 조치 불필요.
# (이전엔 학습이 vocab 을 248078 로 resize 해서 eval 측도 똑같이 맞춰줘야 했지만
# 이제는 Qwen 원본 vocab 그대로 → add_special_tokens / resize 호출 제거.)


def load_eval_model(ckpt_dir: str, device: str, encoder_name: str = "fb_dacvae"):
    """§42 이후 포맷: vocab resize 불필요, p1/p2 legacy prompt 사용."""
    print(f"Loading checkpoint from: {ckpt_dir}")
    cfg = get_config(encoder_name)
    # eval 은 단일 GPU greedy → flash_attn 불필요. 학습과 동일 process node 에서
    # flash_attn .so 의 glibc symbol(__libc_single_threaded) 충돌이 산발적이라
    # sdpa 로 강제. liger kernel 도 fused_linear_ce 만 필요해서 끔.
    cfg = dict(cfg)
    cfg["attn_implementation"] = "sdpa"
    cfg["use_liger_kernel"] = False
    enc_cfg = cfg["encoder"]
    cache_dir = cfg["model_cache_dir"]

    encoder = build_encoder(encoder_name, enc_cfg, cache_dir)
    model = AudioQwen(encoder, cfg)

    model.apply_lora()

    sf_path = os.path.join(ckpt_dir, "model.safetensors")
    if not os.path.exists(sf_path):
        raise FileNotFoundError(f"model.safetensors not found in {ckpt_dir}")
    from safetensors.torch import load_file
    state_dict = load_file(sf_path, device="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys   : {len(missing)}")
        for k in missing[:3]:
            print(f"    {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        for k in unexpected[:3]:
            print(f"    {k}")

    model.to(device)
    model.eval()
    print("Model loaded.\n")
    return model


@torch.inference_mode()
def transcribe_train_format(
    waveform: torch.Tensor,
    model,
    cfg: dict,
    device: str,
    max_new_tokens: int = 256,
) -> str:
    """§42: train_pipeline_override._greedy_batch 와 동일한 p1/p2 legacy 포맷.

    시퀀스: [p1_embeds] + [audio_embeds] + [p2_embeds] → transcript 토큰 생성.
      p1 = "Audio:\\n", p2 = "\\nTranscript:\\n"
    """
    audio = waveform.unsqueeze(0).to(device)                       # (1, T)
    audio_lengths = torch.tensor([waveform.shape[0]], device=device)

    audio_embeds = model._get_audio_embeds(audio, audio_lengths)
    if isinstance(audio_embeds, tuple):
        audio_embeds = audio_embeds[0]
    audio_embeds = audio_embeds.to(torch.bfloat16)
    if os.environ.get("EVAL_DEBUG"):
        print(f"  audio_embeds shape={tuple(audio_embeds.shape)} "
              f"norm={audio_embeds.float().norm().item():.3f} "
              f"mean={audio_embeds.float().mean().item():.4f} "
              f"std={audio_embeds.float().std().item():.4f}")

    tokenizer = model.tokenizer
    embed     = model.llm.get_input_embeddings()
    p1_ids = tokenizer.encode("Audio:\n",        add_special_tokens=False)
    p2_ids = tokenizer.encode("\nTranscript:\n", add_special_tokens=False)
    p1_tensor = torch.tensor([p1_ids], device=device)              # (1, L1)
    p2_tensor = torch.tensor([p2_ids], device=device)              # (1, L2)
    p1_embeds = embed(p1_tensor).to(torch.bfloat16)                # (1, L1, D)
    p2_embeds = embed(p2_tensor).to(torch.bfloat16)                # (1, L2, D)

    inputs_embeds = torch.cat([p1_embeds, audio_embeds, p2_embeds], dim=1).to(torch.bfloat16)
    attn_mask     = torch.ones(1, inputs_embeds.shape[1], device=device, dtype=torch.long)

    spt = cfg["samples_per_token"]
    cap = max(32, min(max_new_tokens,
                      int(waveform.shape[0] / max(spt, 1) / cfg["sample_rate"] * 7)))

    out = model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        max_new_tokens=cap,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
        no_repeat_ngram_size=4,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.strip()


def discover_checkpoints(run_dir: Path) -> list[Path]:
    """model.safetensors mtime 기준으로 시간순 정렬.

    split 번호는 §33 split_rng 셔플 결과라 이름 순 != 학습 순서.
    실제 저장된 시각으로 정렬해야 학습 진행에 따른 WER 추이가 의미 있음.
    """
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    ckpts = [
        p for p in run_dir.iterdir()
        if p.is_dir() and (p / "model.safetensors").exists()
    ]
    ckpts.sort(key=lambda p: (p / "model.safetensors").stat().st_mtime)
    return ckpts


def build_dataset(split: str, encoder_name: str):
    cfg = get_config(encoder_name)
    return LibriSpeechDataset(
        cache_dir=cfg["data_path"],
        url=split,
        max_len=cfg["max_audio_len"],
    )


def evaluate_checkpoint(
    ckpt_dir: Path,
    dataset,
    indices: list[int],
    encoder_name: str,
    device: str,
    max_new_tokens: int,
    beam_size: int,
) -> dict:
    model = load_eval_model(str(ckpt_dir), device=device, encoder_name=encoder_name)
    cfg = get_config(encoder_name)

    refs, hyps = [], []
    t0 = time.time()
    for i in tqdm(indices, desc=ckpt_dir.name, leave=False):
        waveform, transcript = dataset[i]
        hyp = transcribe_train_format(
            waveform, model, cfg, device=device,
            max_new_tokens=max_new_tokens,
        )
        refs.append(transcript)
        hyps.append(hyp)

    wer = compute_wer(hyps, refs)
    elapsed = time.time() - t0

    del model
    torch.cuda.empty_cache()

    mtime = (ckpt_dir / "model.safetensors").stat().st_mtime
    return {
        "ckpt": ckpt_dir.name,
        "ckpt_path": str(ckpt_dir),
        "saved_at": datetime.fromtimestamp(mtime).strftime("%m/%d %H:%M:%S"),
        "saved_ts": mtime,
        "wer": wer,
        "n_samples": len(refs),
        "elapsed_sec": elapsed,
        "pairs": [
            {"idx": idx, "ref": r, "hyp": h}
            for idx, r, h in zip(indices, refs, hyps)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="stage2 출력 디렉토리 (e.g. .../s2_outputs_0414_1442)")
    parser.add_argument("--encoder", type=str, default="fb_dacvae")
    parser.add_argument("--split", type=str, default="dev-clean",
                        choices=["dev-clean", "dev-other", "test-clean", "test-other"])
    parser.add_argument("--max-samples", type=int, default=200,
                        help="dev set 에서 추출할 표본 수 (0=전체)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--out-root", type=Path,
                        default=REPO_ROOT / "eval_ckpts" / "results")
    parser.add_argument("--only", type=str, nargs="*", default=None,
                        help="특정 ckpt 이름만 평가 (substring match)")
    parser.add_argument("--seed", type=int, default=0,
                        help="표본 셔플 seed (0 = 앞에서부터 순서대로)")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)

    ckpts = discover_checkpoints(args.run_dir)
    if args.only:
        ckpts = [c for c in ckpts if any(s in c.name for s in args.only)]
    if not ckpts:
        print(f"No checkpoints found in {args.run_dir}")
        return
    print(f"Found {len(ckpts)} checkpoint(s) in {args.run_dir}")
    for c in ckpts:
        print(f"  - {c.name}")

    print(f"\nLoading dataset: {args.split}")
    dataset = build_dataset(args.split, args.encoder)
    n_total = len(dataset)
    n_eval = n_total if args.max_samples == 0 else min(args.max_samples, n_total)
    if args.seed:
        import random
        rng = random.Random(args.seed)
        indices = rng.sample(range(n_total), n_eval)
        indices.sort()
    else:
        indices = list(range(n_eval))
    print(f"Evaluating {n_eval}/{n_total} samples per checkpoint")

    run_tag = args.run_dir.name
    out_dir = args.out_root / run_tag / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_dir": str(args.run_dir),
        "encoder": args.encoder,
        "split": args.split,
        "n_samples": n_eval,
        "indices": indices,
        "checkpoints": [],
    }

    for ckpt in ckpts:
        try:
            result = evaluate_checkpoint(
                ckpt, dataset, indices,
                encoder_name=args.encoder,
                device=device,
                max_new_tokens=args.max_new_tokens,
                beam_size=args.beam_size,
            )
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {ckpt.name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        ckpt_path = out_dir / f"{ckpt.name}.json"
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"  → {ckpt.name}: WER = {result['wer']*100:.2f}%  "
              f"({result['elapsed_sec']:.1f}s)  → {ckpt_path}")

        summary["checkpoints"].append({
            "ckpt": result["ckpt"],
            "saved_at": result["saved_at"],
            "saved_ts": result["saved_ts"],
            "wer": result["wer"],
            "n_samples": result["n_samples"],
            "elapsed_sec": result["elapsed_sec"],
            "json": ckpt_path.name,
        })

        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Summary: {out_dir/'summary.json'}")
    print(f"Build viewer: python eval_ckpts/build_viewer.py {out_dir}")


if __name__ == "__main__":
    main()
