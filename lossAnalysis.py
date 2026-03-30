"""
Stage 1 projector의 per-step CE loss 분석.

Teacher forcing (GT 문맥) vs 자유 생성 (모델 자신의 예측 문맥) 시
각 토큰 위치에서의 CE loss를 비교한다.

결과는 stdout 출력 + CSV 저장.

사용법:
    python lossAnalysis.py \
        --proj /mnt/ddn/users/sehyun/ckpts/proj_dac_vae/s1_proj_dac_vae_ep2_step41951_best.pt \
        --num-samples 3
"""

import argparse
import csv
import os

import torch
import torch.nn.functional as F
import torchaudio

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import get_config
from encoders import build_encoder
from model import AudioQwen


@torch.no_grad()
def compute_per_step_loss(model, waveform, audio_len, gt_token_ids, device, max_tokens):
    """
    두 가지 per-step CE loss를 계산한다.

    teacher_losses[i] : GT 앞 토큰이 전부 주어졌을 때 i번째 GT 토큰의 CE loss
    free_losses[i]    : 모델이 스스로 생성한 앞 토큰이 주어졌을 때 i번째 GT 토큰의 CE loss
    pred_tokens[i]    : 자유 생성에서 i번째 스텝에 모델이 고른 토큰
    """
    audio     = waveform.unsqueeze(0).to(device)
    audio_len = audio_len.unsqueeze(0).to(device)

    audio_embeds, audio_mask = model._get_audio_embeds(audio, audio_len)

    embed = model.llm.get_input_embeddings()
    p1_e  = embed(model.prompt_p1_ids).expand(1, -1, -1)
    p2_e  = embed(model.prompt_p2_ids).expand(1, -1, -1)

    ctx_embeds = torch.cat([p1_e, audio_embeds, p2_e], dim=1)
    p1_m       = torch.ones(1, p1_e.shape[1],   device=device, dtype=torch.long)
    p2_m       = torch.ones(1, p2_e.shape[1],   device=device, dtype=torch.long)
    ctx_mask   = torch.cat([p1_m, audio_mask.long(), p2_m], dim=1)
    ctx_len    = ctx_embeds.shape[1]

    gt_ids = gt_token_ids[:max_tokens].unsqueeze(0).to(device)  # (1, N)
    N      = gt_ids.shape[1]

    # ── Teacher-forced per-step loss (forward 1회) ──────────────────────────
    gt_embeds = embed(gt_ids)
    tf_inputs = torch.cat([ctx_embeds, gt_embeds[:, :-1, :]], dim=1)
    tf_attn   = torch.cat([ctx_mask, torch.ones(1, N - 1, device=device, dtype=torch.long)], dim=1)
    tf_logits = model.llm(
        inputs_embeds=tf_inputs, attention_mask=tf_attn, use_cache=False
    ).logits[0]

    teacher_losses = [
        F.cross_entropy(tf_logits[ctx_len - 1 + i].unsqueeze(0),
                        gt_ids[0, i].unsqueeze(0)).item()
        for i in range(N)
    ]

    # ── Free generation per-step loss (forward N회) ──────────────────────────
    free_losses = []
    pred_tokens = []
    cur_embeds  = ctx_embeds.clone()
    cur_mask    = ctx_mask.clone()

    for i in range(N):
        logit_last = model.llm(
            inputs_embeds=cur_embeds, attention_mask=cur_mask, use_cache=False
        ).logits[0, -1, :]

        free_losses.append(
            F.cross_entropy(logit_last.unsqueeze(0), gt_ids[0, i].unsqueeze(0)).item()
        )

        pred_id = logit_last.argmax().unsqueeze(0)
        pred_tokens.append(model.tokenizer.decode([pred_id.item()]))

        cur_embeds = torch.cat([cur_embeds, embed(pred_id.view(1, 1))], dim=1)
        cur_mask   = torch.cat([cur_mask, torch.ones(1, 1, device=device, dtype=torch.long)], dim=1)

    gt_words = [model.tokenizer.decode([t.item()]) for t in gt_ids[0]]
    return gt_words, teacher_losses, free_losses, pred_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj",        required=True, help="projector checkpoint 경로")
    parser.add_argument("--encoder",     default="dac_vae", help="encoder 이름 (default: dac_vae)")
    parser.add_argument("--num-samples", type=int, default=3,  help="분석할 샘플 수")
    parser.add_argument("--max-tokens",  type=int, default=30, help="샘플당 최대 토큰 수")
    parser.add_argument("--out",         default="loss_analysis.csv", help="CSV 저장 경로")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = get_config(args.encoder)

    print("Loading model...")
    encoder = build_encoder(args.encoder, cfg["encoder"], cfg["model_cache_dir"])
    model   = AudioQwen(encoder, cfg)

    print(f"Loading projector: {args.proj}")
    proj_state = torch.load(args.proj, map_location="cpu", weights_only=True)
    model.load_state_dict(proj_state, strict=False)
    model.to(device).eval()

    dataset = torchaudio.datasets.LIBRISPEECH(
        root=cfg["data_path"], url="train-clean-100", download=False
    )

    csv_rows = []

    for sample_idx in range(args.num_samples):
        waveform, sr, transcript, *_ = dataset[sample_idx]
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform  = waveform.squeeze(0)
        audio_len = torch.tensor(waveform.shape[0])

        token_ids = model.tokenizer(
            transcript.lower(), add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]
        token_ids = torch.cat([token_ids, torch.tensor([model.tokenizer.eos_token_id])])

        gt_words, tf_losses, free_losses, pred_tokens = compute_per_step_loss(
            model, waveform, audio_len, token_ids, device, args.max_tokens
        )

        # stdout
        print(f"\n{'='*65}")
        print(f"Sample [{sample_idx}]: {transcript.lower()}")
        print(f"{'='*65}")
        print(f"{'Pos':>4}  {'GT':>16}  {'Pred(free)':>16}  {'TF_loss':>9}  {'Free_loss':>9}")
        print("─" * 65)
        for i, (gt, pred, tf_l, fr_l) in enumerate(
            zip(gt_words, pred_tokens, tf_losses, free_losses)
        ):
            marker = "  ← audio only" if i == 0 else ""
            print(f"{i:>4}  {repr(gt):>16}  {repr(pred):>16}  {tf_l:>9.4f}  {fr_l:>9.4f}{marker}")
        print("─" * 65)
        print(f"{'avg':>4}  {'':>16}  {'':>16}  "
              f"{sum(tf_losses)/len(tf_losses):>9.4f}  "
              f"{sum(free_losses)/len(free_losses):>9.4f}")

        # CSV rows
        for i, (gt, pred, tf_l, fr_l) in enumerate(
            zip(gt_words, pred_tokens, tf_losses, free_losses)
        ):
            csv_rows.append({
                "sample_idx": sample_idx,
                "pos":        i,
                "gt_token":   gt,
                "pred_token": pred,
                "tf_loss":    round(tf_l, 6),
                "free_loss":  round(fr_l, 6),
                "audio_only": i == 0,
            })

    # CSV 저장
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nCSV saved → {args.out}")


if __name__ == "__main__":
    main()
