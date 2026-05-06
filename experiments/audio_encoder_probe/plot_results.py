"""Plot linear-probe results: bar chart per dataset with all 5 encoders."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RESULTS = ROOT / "_results/probe_results.csv"
FIG_DIR = ROOT / "_results/figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS)

# Best C per (encoder, dataset)
best_C = (df.groupby(["encoder", "dataset", "C"])
            .accuracy.mean()
            .reset_index()
            .sort_values("accuracy", ascending=False)
            .groupby(["encoder", "dataset"]).head(1)
            .set_index(["encoder", "dataset"])["C"])
df["best_C"] = df.set_index(["encoder", "dataset"]).index.map(best_C)
best_df = df[df.C == df.best_C].copy()

encoders = ["whisper_tiny", "whisper_small", "wavtok_40_unify", "encodec_24k", "dacvae"]
datasets = ["iemocap_4class", "ravdess", "cremad", "esc50"]
display_enc = {
    "whisper_tiny": "Whisper-tiny\n(384d, ASR)",
    "whisper_small": "Whisper-small\n(768d, ASR)",
    "wavtok_40_unify": "WavTok-40\n(512d, recon)",
    "encodec_24k": "EnCodec-24k\n(128d, recon)",
    "dacvae": "DAC-VAE\n(128d, recon)",
}
display_ds = {
    "iemocap_4class": "IEMOCAP (4-class)\nleave-session-out",
    "ravdess": "RAVDESS (8-class)\nleave-speakers-out",
    "cremad": "CREMA-D (6-class)\nleave-speakers-out",
    "esc50": "ESC-50 (50-class)\nofficial 5-fold",
}
chance = {"iemocap_4class": 0.25, "ravdess": 0.125, "cremad": 1/6, "esc50": 0.02}
enc_colors = {
    "whisper_tiny": "#5dade2", "whisper_small": "#1f618d",
    "wavtok_40_unify": "#f4a460", "encodec_24k": "#cd853f", "dacvae": "#8b4513",
}

# ── Plot 1: 1x4 panel, accuracy bar chart ───────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), sharey=False)
for ax, ds in zip(axes, datasets):
    sub = best_df[best_df.dataset == ds]
    means = [sub[sub.encoder == e].accuracy.mean() for e in encoders]
    stds = [sub[sub.encoder == e].accuracy.std() for e in encoders]
    x = np.arange(len(encoders))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=[enc_colors[e] for e in encoders])
    ax.axhline(chance[ds], ls="--", color="red", alpha=0.6, label=f"chance={chance[ds]:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([display_enc[e] for e in encoders], fontsize=8, rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(display_ds[ds], fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    for i, m in enumerate(means):
        ax.text(i, m + 0.02, f"{m:.3f}", ha="center", fontsize=8)

plt.suptitle("Audio-encoder linear probe — 5 encoders × 4 datasets",
             y=1.02, fontsize=12)
plt.tight_layout()
out = FIG_DIR / "probe_summary_bar.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
plt.close()

# ── Plot 2: per-fold accuracy lines ─────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, ds in zip(axes, datasets):
    sub = best_df[best_df.dataset == ds]
    folds = sorted(sub.fold.unique(), key=str)
    for enc in encoders:
        sub_e = sub[sub.encoder == enc].set_index("fold").reindex(folds)
        ax.plot(range(len(folds)), sub_e.accuracy.values,
                marker="o", label=display_enc[enc].split("\n")[0],
                color=enc_colors[enc])
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels([str(f) for f in folds], fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Fold")
    ax.set_title(display_ds[ds], fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
plt.tight_layout()
out = FIG_DIR / "probe_per_fold.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
plt.close()

# ── Plot 3: C-grid sensitivity ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, ds in zip(axes, datasets):
    for enc in encoders:
        sub = (df[(df.encoder == enc) & (df.dataset == ds)]
               .groupby("C").accuracy.agg(["mean", "std"]))
        ax.errorbar(sub.index, sub["mean"], yerr=sub["std"],
                    marker="o", capsize=3,
                    label=display_enc[enc].split("\n")[0],
                    color=enc_colors[enc])
    ax.set_xscale("log")
    ax.set_xlabel("C (L2 inverse strength)")
    ax.set_ylabel("Accuracy")
    ax.set_title(display_ds[ds], fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
plt.tight_layout()
out = FIG_DIR / "probe_C_sensitivity.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
plt.close()

# ── Plot 4: heatmap of accuracy ─────────────────────────────────────────────

acc_matrix = np.zeros((len(encoders), len(datasets)))
for i, e in enumerate(encoders):
    for j, ds in enumerate(datasets):
        sub = best_df[(best_df.encoder == e) & (best_df.dataset == ds)]
        acc_matrix[i, j] = sub.accuracy.mean()

fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(acc_matrix, cmap="viridis", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(datasets)))
ax.set_xticklabels([d.split("\n")[0] for d in [display_ds[d] for d in datasets]], rotation=15, fontsize=9)
ax.set_yticks(range(len(encoders)))
ax.set_yticklabels([display_enc[e].replace("\n", " ") for e in encoders], fontsize=9)
for i in range(len(encoders)):
    for j in range(len(datasets)):
        ax.text(j, i, f"{acc_matrix[i, j]:.3f}",
                ha="center", va="center",
                color="white" if acc_matrix[i, j] < 0.55 else "black",
                fontsize=10)
plt.colorbar(im, ax=ax, label="Accuracy (5-fold mean)")
ax.set_title("Audio-encoder × Dataset accuracy heatmap")
plt.tight_layout()
out = FIG_DIR / "probe_heatmap.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
plt.close()

print(f"\nAll plots saved to {FIG_DIR}")
