"""probe_projector.py — projector output 의 audio 별 diversity 측정.

목적: 가설 2 (projector 가 audio-specific 정보를 LLM 에 못 전달) 의 직접 검증.

측정:
  1. N 개 서로 다른 audio 를 encoder → projector 통과
  2. 각 audio 의 projector output 을 mean-pool 해서 (D,) 벡터 얻음
  3. Pairwise cosine similarity 계산
  4. Qwen token embedding 의 norm 분포와 projector output norm 비교

건강한 결과:
  - Pairwise cos sim: 0.2 ~ 0.5 (audio 별 clearly distinct)
  - Norm 이 Qwen token embed norm 수준 (~7-15 정도)

Collapsed 결과 (가설 2 확정):
  - Pairwise cos sim: 0.95+ (거의 동일)
  - Norm 이 Qwen token embed norm 대비 << 1

Usage:
    CUDA_VISIBLE_DEVICES=7 /mnt/ddn/users/jos/miniforge3/envs/audio/bin/python \\
        eval_ckpts/probe_projector.py \\
        --s1-ckpt /mnt/tmp/cache/hf/fb_dacvae/s1_outputs_0415_1631/s1_proj.pt \\
        --n-samples 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import get_config  # noqa: E402
from dataset import LibriSpeechDataset  # noqa: E402
from encoders import build_encoder  # noqa: E402
from model import AudioQwen  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1-ckpt", type=str, required=True,
                    help="Stage 1 projector ckpt (.pt) — load projector weights only")
    ap.add_argument("--encoder", default="fb_dacvae")
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"

    # Build model (no LoRA, projector-only eval)
    cfg = dict(get_config(args.encoder))
    cfg["attn_implementation"] = "sdpa"
    cfg["use_liger_kernel"] = False

    encoder = build_encoder(args.encoder, cfg["encoder"], cfg["model_cache_dir"])
    model = AudioQwen(encoder, cfg)

    # Load projector weights
    state = torch.load(args.s1_ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    n_proj_loaded = sum(1 for k in state if "projector" in k or "proj_norm" in k)
    print(f"[load] {n_proj_loaded} projector keys from {args.s1_ckpt}")
    if unexpected:
        print(f"  unexpected: {len(unexpected)}")

    model.to(device).eval()

    # Dataset
    dataset = LibriSpeechDataset(cache_dir=cfg["data_path"], url=args.split,
                                 max_len=cfg["max_audio_len"])
    import random
    random.seed(args.seed)
    indices = random.sample(range(len(dataset)), args.n_samples)
    print(f"[dataset] {args.split}: {args.n_samples}/{len(dataset)} samples "
          f"(random seed={args.seed})")

    # Extract projector outputs
    with torch.inference_mode():
        pooled_embs = []
        all_norms = []
        all_t_projs = []
        for idx in indices:
            wav, text = dataset[idx]
            wav = wav.unsqueeze(0).to(device)
            lens = torch.tensor([wav.shape[-1]], device=device)
            emb = model._get_audio_embeds(wav, lens)   # (1, T_proj, D)
            if isinstance(emb, tuple):
                emb = emb[0]
            emb = emb.float()   # fp32 for sim computation
            # Per-token norm distribution
            per_token_norm = emb.norm(dim=-1).flatten()   # (T_proj,)
            all_norms.append(per_token_norm.cpu())
            all_t_projs.append(emb.shape[1])
            # Mean-pool across time → (D,) representative vector
            pooled = emb.mean(dim=1).squeeze(0)   # (D,)
            pooled_embs.append(pooled)

        pooled = torch.stack(pooled_embs)   # (N, D)
        N = pooled.shape[0]

    # ── Projector output stats
    all_norms_cat = torch.cat(all_norms)
    print("\n=== Projector output (after proj_norm LayerNorm) ===")
    print(f"  T_proj range   : {min(all_t_projs)} ~ {max(all_t_projs)}")
    print(f"  per-token norm : mean={all_norms_cat.mean():.3f}  "
          f"std={all_norms_cat.std():.3f}  "
          f"min={all_norms_cat.min():.3f}  max={all_norms_cat.max():.3f}")

    # Pairwise cosine similarity of pooled embeddings
    pooled_norm = F.normalize(pooled, dim=-1)
    sim_matrix = pooled_norm @ pooled_norm.T   # (N, N)
    off_diag = sim_matrix[~torch.eye(N, dtype=torch.bool, device=sim_matrix.device)]
    print(f"\n=== Pairwise cosine similarity (mean-pooled, N={N}) ===")
    print(f"  off-diag mean  : {off_diag.mean():.4f}")
    print(f"  off-diag std   : {off_diag.std():.4f}")
    print(f"  off-diag min   : {off_diag.min():.4f}")
    print(f"  off-diag max   : {off_diag.max():.4f}")
    print()
    print("  Interpretation:")
    print(f"    healthy   : 0.2 ~ 0.5 (audio 별 distinct)")
    print(f"    collapsed : 0.9+ (거의 동일 → 가설 2 확정)")
    print(f"    current   : {off_diag.mean():.3f} → ", end="")
    if off_diag.mean() > 0.9:
        print("❌ COLLAPSED (가설 2 확정)")
    elif off_diag.mean() > 0.7:
        print("⚠️  near-collapsed (가설 2 강한 의심)")
    elif off_diag.mean() > 0.5:
        print("🟡 weak distinction")
    else:
        print("✅ healthy distinction")

    # Compare to Qwen token embedding distribution
    print("\n=== Qwen3.5-2B token embedding (comparison) ===")
    with torch.inference_mode():
        token_embeds = model.llm.get_input_embeddings().weight.float()   # (V, D)
        token_norms = token_embeds.norm(dim=-1)
        print(f"  vocab size     : {token_embeds.shape[0]}")
        print(f"  per-token norm : mean={token_norms.mean():.3f}  "
              f"std={token_norms.std():.3f}  "
              f"min={token_norms.min():.3f}  max={token_norms.max():.3f}")
        # Ratio
        ratio = all_norms_cat.mean() / token_norms.mean()
        print(f"\n  projector_norm / token_norm = {ratio:.3f}")
        if ratio < 0.5:
            print("    ⚠️  projector output 이 token embed 보다 훨씬 작음 — LLM attention 이 무시 가능성")
        elif ratio > 2.0:
            print("    ⚠️  projector output 이 token embed 보다 훨씬 큼 — LLM input distribution 파괴 가능성")
        else:
            print("    ✅ norm scale 은 comparable")

    # Sample a few token embeddings and compute their pairwise cos sim baseline
    # (audio projector output 과 같은 측정 방식으로 비교)
    with torch.inference_mode():
        # Random token sample
        rng = torch.Generator().manual_seed(args.seed)
        sample_idx = torch.randperm(token_embeds.shape[0], generator=rng)[:N]
        token_sample = token_embeds[sample_idx]   # (N, D)
        token_sample_norm = F.normalize(token_sample, dim=-1)
        token_sim = token_sample_norm @ token_sample_norm.T
        token_off = token_sim[~torch.eye(N, dtype=torch.bool)]
        print(f"\n=== Random Qwen token embeddings baseline (N={N}) ===")
        print(f"  off-diag cos sim: mean={token_off.mean():.4f}  "
              f"std={token_off.std():.4f}")
        print(f"  (비교: projector pairwise cos sim mean = {off_diag.mean():.4f})")


if __name__ == "__main__":
    main()
