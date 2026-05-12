"""
Emotion dataset stat visualization.
Output: docs/analysis/emotion_datasets_viz.png (3-panel figure)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker

# ── Data ──────────────────────────────────────────────────────────────────────

EMOTIONS = [
    "anger", "disgust", "fear", "happiness", "neutral",
    "sadness", "surprise", "amused", "excited", "frustrated",
    "calm", "anxious", "apologetic", "assertive", "concerned",
    "encouraging", "sleepy",
]

# (dataset_label, is_train_pool, total_samples, mean_duration_s, {emotion: count})
# For CMU-MOSEI: total_samples = utterance count, emotion dict = label counts (multi-label, sum > total)
DATASETS = [
    ("IEMOCAP\ntrain", True, 5882, None, {
        "anger": 933, "disgust": 2, "fear": 30, "happiness": 452,
        "neutral": 1324, "sadness": 839, "surprise": 89,
        "excited": 742, "frustrated": 1468,
    }),
    ("IEMOCAP\ntest", False, 1507, 4.3, {
        "anger": 197, "happiness": 613, "neutral": 386, "sadness": 311,
    }),
    ("MELD\ntrain", True, 9989, None, {
        "anger": 1109, "disgust": 271, "fear": 268, "happiness": 1743,
        "neutral": 4710, "sadness": 683, "surprise": 1205,
    }),
    ("MELD\ndev+test", True, 3716, 3.1, {
        "anger": 498, "disgust": 89, "fear": 90, "happiness": 564,
        "neutral": 1725, "sadness": 319, "surprise": 431,
    }),
    ("DailyTalk", True, 23773, 3.3, {
        "anger": 159, "disgust": 66, "fear": 18, "happiness": 3856,
        "neutral": 18966, "sadness": 301, "surprise": 407,
    }),
    ("EmoV-DB", True, 6893, 4.9, {
        "anger": 1268, "disgust": 1019,
        "neutral": 1568, "amused": 1317, "sleepy": 1721,
    }),
    ("RAVDESS", True, 1440, 3.7, {
        "anger": 192, "disgust": 192, "fear": 192, "happiness": 192,
        "neutral": 96, "sadness": 192, "surprise": 192, "calm": 192,
    }),
    ("MUStARD++", True, 1200, 4.7, {
        "anger": 53, "disgust": 29, "fear": 23, "happiness": 244,
        "neutral": 438, "sadness": 149, "surprise": 101,
        "excited": 115, "frustrated": 48,
    }),
    ("CREMA-D\ntrain", False, 5953, 2.5, {
        "anger": 914, "disgust": 559, "fear": 623, "happiness": 355,
        "neutral": 3199, "sadness": 303,
    }),
    ("CREMA-D\ntest", False, 1489, 2.6, {
        "anger": 239, "disgust": 125, "fear": 170, "happiness": 76,
        "neutral": 812, "sadness": 67,
    }),
    ("SAVEE", False, 480, 3.8, {
        "anger": 60, "disgust": 60, "fear": 60, "happiness": 60,
        "neutral": 120, "sadness": 60, "surprise": 60,
    }),
    ("TESS", False, 2800, 2.1, {
        "anger": 400, "disgust": 400, "fear": 400, "happiness": 400,
        "neutral": 400, "sadness": 400, "surprise": 400,
    }),
    ("ESD (EN)", False, 17500, 3.2, {
        "anger": 3500, "happiness": 3500, "neutral": 3500,
        "sadness": 3500, "surprise": 3500,
    }),
    ("JL-Corpus", False, 2400, 2.1, {
        "anger": 240, "happiness": 240, "neutral": 240, "sadness": 240,
        "anxious": 240, "apologetic": 240, "assertive": 240,
        "concerned": 240, "encouraging": 240, "excited": 240,
    }),
    # ── New datasets ──────────────────────────────────────────────────────────
    ("MSP-IMPROV", False, 8438, None, {
        "anger": 792, "happiness": 2167, "neutral": 3477, "sadness": 2002,
    }),
    # CMU-MOSEI: multi-label — emotion dict = label counts (sum > utterance total)
    ("CMU-MOSEI\n★multi-label", False, 22856, None, {
        "anger": 4600, "disgust": 3755, "fear": 1803,
        "happiness": 10752, "sadness": 5601, "surprise": 2055,
    }),
    ("EmoVoice-DB\n†synthetic", False, 22100, 6.6, {
        "anger": 3486, "disgust": 2950, "fear": 2961, "happiness": 3269,
        "neutral": 3188, "sadness": 3174, "surprise": 3072,
    }),
    ("eNTERFACE'05", False, 1166, None, {
        "anger": 194, "disgust": 195, "fear": 194, "happiness": 195,
        "sadness": 194, "surprise": 194,
    }),
]

labels = [d[0] for d in DATASETS]
is_train = [d[1] for d in DATASETS]
totals = [d[2] for d in DATASETS]
durations = [d[3] for d in DATASETS]
emotion_dicts = [d[4] for d in DATASETS]

# Build count matrix  [dataset x emotion]
mat = np.zeros((len(DATASETS), len(EMOTIONS)))
for i, ed in enumerate(emotion_dicts):
    for j, em in enumerate(EMOTIONS):
        mat[i, j] = ed.get(em, 0)

EMOTION_COLORS = {
    "neutral":     "#b0b0b0",
    "happiness":   "#f9c74f",
    "anger":       "#e63946",
    "sadness":     "#4895ef",
    "surprise":    "#f77f00",
    "fear":        "#9b5de5",
    "disgust":     "#52b788",
    "excited":     "#ffb703",
    "frustrated":  "#e07a5f",
    "calm":        "#90e0ef",
    "amused":      "#f4a261",
    "sleepy":      "#adb5bd",
    "anxious":     "#c77dff",
    "apologetic":  "#d4a5a5",
    "assertive":   "#a8dadc",
    "concerned":   "#457b9d",
    "encouraging": "#1d3557",
}

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 22))
fig.patch.set_facecolor("#fafafa")

gs = fig.add_gridspec(
    2, 2,
    height_ratios=[1.6, 1],
    hspace=0.35, wspace=0.32,
    left=0.08, right=0.97, top=0.93, bottom=0.05,
)

ax_heat = fig.add_subplot(gs[0, :])
ax_bar  = fig.add_subplot(gs[1, 0])
ax_dot  = fig.add_subplot(gs[1, 1])

# ── Panel A: Heatmap ─────────────────────────────────────────────────────────
mat_plot = mat.copy()
mat_plot[mat_plot == 0] = np.nan

emo_has_data = np.nansum(mat, axis=0) > 0
emo_cols = [e for e, h in zip(EMOTIONS, emo_has_data) if h]
mat_plot_trimmed = mat_plot[:, emo_has_data]

vmin, vmax = 10, 20000
im = ax_heat.imshow(
    mat_plot_trimmed,
    aspect="auto",
    norm=LogNorm(vmin=vmin, vmax=vmax),
    cmap="YlOrRd",
)

for i in range(len(DATASETS)):
    for j in range(len(emo_cols)):
        v = mat_plot_trimmed[i, j]
        if np.isnan(v):
            ax_heat.text(j, i, "–", ha="center", va="center",
                         fontsize=7, color="#cccccc")
        else:
            txt = f"{int(v):,}" if v >= 1000 else str(int(v))
            col = "white" if v > 2000 else "#333333"
            ax_heat.text(j, i, txt, ha="center", va="center",
                         fontsize=6.5, color=col, fontweight="bold")

ax_heat.set_xticks(range(len(emo_cols)))
ax_heat.set_xticklabels(emo_cols, rotation=35, ha="right", fontsize=9)
ax_heat.set_yticks(range(len(DATASETS)))
ax_heat.set_yticklabels(labels, fontsize=7.5)

# Shade eval-only rows
for i, train in enumerate(is_train):
    if not train:
        ax_heat.add_patch(mpatches.FancyBboxPatch(
            (-0.5, i - 0.5), len(emo_cols), 1,
            boxstyle="square,pad=0", linewidth=0,
            facecolor="#e8f4f8", alpha=0.35, zorder=0,
        ))

# Separator line
train_end = sum(is_train) - 0.5
ax_heat.axhline(train_end, color="#2196F3", linewidth=1.8, linestyle="--", alpha=0.7)
ax_heat.text(len(emo_cols) - 0.5, train_end - 0.15,
             "  ← train pool", fontsize=7.5, color="#2196F3", va="bottom", ha="right")
ax_heat.text(len(emo_cols) - 0.5, train_end + 0.15,
             "  ← eval only", fontsize=7.5, color="#2196F3", va="top", ha="right")

# Highlight CMU-MOSEI (multi-label) row with border
mosei_idx = next(i for i, d in enumerate(DATASETS) if "MOSEI" in d[0])
for j in range(len(emo_cols)):
    ax_heat.add_patch(plt.Rectangle(
        (j - 0.5, mosei_idx - 0.5), 1, 1,
        fill=False, edgecolor="#ff9800", linewidth=1.2, zorder=3,
    ))

cb = fig.colorbar(im, ax=ax_heat, fraction=0.012, pad=0.01)
cb.set_label("Sample count (log scale)\n★ CMU-MOSEI: label count (multi-label, sum > utterances)\n† EmoVoice-DB: GPT-4o synthetic", fontsize=7.5)
ax_heat.set_title(
    "A.  Dataset × Emotion Coverage  (log-scale count; – = class absent)\n"
    "★ CMU-MOSEI = multi-label emotion counts  |  † EmoVoice-DB = GPT-4o synthetic",
    fontsize=11, fontweight="bold", pad=8,
)

# ── Panel B: Stacked proportion bar ──────────────────────────────────────────
CORE7 = ["neutral", "happiness", "anger", "sadness", "surprise", "fear", "disgust"]
OTHER_LABEL = "other\nemotions"

proportions = []
for ed, tot in zip(emotion_dicts, totals):
    row = {}
    for em in CORE7:
        row[em] = ed.get(em, 0) / tot
    row[OTHER_LABEL] = sum(v for k, v in ed.items() if k not in CORE7) / tot
    proportions.append(row)

bar_cols = CORE7 + [OTHER_LABEL]
bar_colors = [EMOTION_COLORS.get(e, "#999999") for e in bar_cols]

bottoms = np.zeros(len(DATASETS))
bar_ys = np.arange(len(DATASETS))
for col, color in zip(bar_cols, bar_colors):
    vals = np.array([p[col] for p in proportions])
    ax_bar.barh(bar_ys, vals, left=bottoms, color=color,
                height=0.7, label=col, edgecolor="white", linewidth=0.3)
    bottoms += vals

# Mark CMU-MOSEI with hatching overlay (to indicate multi-label)
ax_bar.barh(mosei_idx, 1.0, left=0, height=0.7,
            fill=False, edgecolor="#ff9800", linewidth=1.5, zorder=3)

ax_bar.set_yticks(bar_ys)
ax_bar.set_yticklabels(labels, fontsize=7)
ax_bar.set_xlim(0, 1)
ax_bar.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax_bar.axvline(0.5, color="gray", linestyle=":", linewidth=0.8)
ax_bar.set_title("B.  Emotion Composition (% of dataset total)",
                 fontsize=10, fontweight="bold", pad=6)
ax_bar.set_xlabel("Proportion", fontsize=8)

for i, train in enumerate(is_train):
    marker = "▶" if train else "◆"
    color  = "#2196F3" if train else "#e63946"
    ax_bar.text(-0.02, i, marker, ha="right", va="center", fontsize=7,
                color=color, transform=ax_bar.get_yaxis_transform())

legend_patches = [
    mpatches.Patch(color=EMOTION_COLORS.get(e, "#999999"), label=e)
    for e in CORE7
] + [mpatches.Patch(color="#999999", label=OTHER_LABEL)]
ax_bar.legend(handles=legend_patches, fontsize=6.5, ncol=2,
              loc="lower right", framealpha=0.85)

# ── Panel C: Duration bubble ──────────────────────────────────────────────────
dur_x, dur_y, dur_s, dur_c, dur_lbl = [], [], [], [], []
for i, (label, train, tot, dur, _) in enumerate(DATASETS):
    if dur is None:
        continue
    dur_x.append(dur)
    dur_y.append(tot)
    dur_s.append(max(tot / 25, 30))
    dur_c.append("#2196F3" if train else "#e63946")
    dur_lbl.append(label.replace("\n", " "))

ax_dot.scatter(dur_x, dur_y, s=dur_s, c=dur_c, alpha=0.72,
               edgecolors="white", linewidths=0.8)

for x, y, lbl in zip(dur_x, dur_y, dur_lbl):
    ax_dot.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=(5, 3), fontsize=7, color="#333333")

ax_dot.set_yscale("log")
ax_dot.set_xlabel("Mean duration per clip (s)", fontsize=8)
ax_dot.set_ylabel("Total samples (log scale)", fontsize=8)
ax_dot.set_title("C.  Duration vs. Size\n(bubble=proportional to samples)",
                 fontsize=10, fontweight="bold", pad=6)
ax_dot.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax_dot.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)

train_patch = mpatches.Patch(color="#2196F3", label="train pool", alpha=0.8)
eval_patch  = mpatches.Patch(color="#e63946", label="eval only",  alpha=0.8)
ax_dot.legend(handles=[train_patch, eval_patch], fontsize=8, loc="upper right")

# ── Title ──────────────────────────────────────────────────────────────────────
n_sl  = sum(d[2] for d in DATASETS if "MOSEI" not in d[0])
n_tot = sum(d[2] for d in DATASETS)
fig.suptitle(
    f"Emotion Dataset Survey — {n_sl:,} single-label + 22,856 CMU-MOSEI utterances = {n_tot:,} total  "
    f"(18 datasets, 2026-05-10)",
    fontsize=12, fontweight="bold", y=0.975,
)

out_path = "/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/docs/analysis/emotion_datasets_viz.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
