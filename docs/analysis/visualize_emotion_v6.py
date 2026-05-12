"""
V6 emotion dataset visualization — based on actual manifest files.
Output: docs/analysis/emotion_v6_viz.png
"""
import re
import json
import glob
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

MANIFEST_DIR = Path("/mnt/ddn/users/sehyun/datasets/manifests/v6/audio_emotion")

# ── Label normalisation ───────────────────────────────────────────────────────
LETTER_PREFIX = re.compile(r"^[A-Z]\.\s+")

SEMANTIC_MAP = {
    "anger": "anger", "angry": "anger",
    "happiness": "happiness", "happy": "happiness", "joy": "happiness",
    "sadness": "sadness", "sad": "sadness",
    "surprise": "surprise", "surprised": "surprise",
    "fear": "fear", "fearful": "fear",
    "disgust": "disgust", "disgusted": "disgust",
    "neutral": "neutral",
    "no emotion": "no emotion",
    "excited": "excited", "excitement": "excited",
    "frustrated": "frustrated", "frustration": "frustrated",
    "calm": "calm",
    "amused": "amused",
    "sleepy": "sleepy",
    "other": "other",
}

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
    "no emotion":  "#dddddd",
    "other":       "#cccccc",
}

DS_ORDER = ["dailytalk", "emovdb", "iemocap_train", "meld", "mustardpp", "ravdess"]
DS_LABELS = {
    "dailytalk":    "DailyTalk",
    "emovdb":       "EmoV-DB",
    "iemocap_train":"IEMOCAP\ntrain (S1-4)",
    "meld":         "MELD\ntrain+dev",
    "mustardpp":    "MUStARD++",
    "ravdess":      "RAVDESS",
}

# ── Parse manifests ───────────────────────────────────────────────────────────
def parse_manifests():
    stats = defaultdict(lambda: {"total": 0, "raw": Counter(), "sem": Counter()})
    for fpath in sorted(MANIFEST_DIR.glob("*.jsonl")):
        parts = fpath.stem.split("_")   # emotion_<ds...>_<shard>
        ds = "_".join(parts[1:-1])
        with open(fpath) as f:
            for line in f:
                row = json.loads(line)
                choices = row.get("choices", [])
                ans = row.get("answer", "")
                if choices and isinstance(ans, str) and len(ans) == 1:
                    idx = ord(ans) - ord("A")
                    raw_label = LETTER_PREFIX.sub("", choices[idx]).strip() if 0 <= idx < len(choices) else ans
                elif isinstance(ans, int) and choices:
                    raw_label = LETTER_PREFIX.sub("", choices[ans]).strip()
                else:
                    raw_label = str(ans)
                sem_label = SEMANTIC_MAP.get(raw_label, raw_label)
                stats[ds]["total"] += 1
                stats[ds]["raw"][raw_label] += 1
                stats[ds]["sem"][sem_label] += 1
    return stats

stats = parse_manifests()

# ── Eval sets (hardcoded from actual data / eval code) ────────────────────────
# IEMOCAP S5: ang→angry(anger), hap→happy(happiness), exc→happy(happiness),
#             neu→neutral, sad→sad(sadness). Using canonical CORE_LABELS names.
IEMOCAP_S5 = {"anger": 197, "happiness": 613, "neutral": 386, "sadness": 311}  # 1507 total

# MELD test: from test_sent_emo.csv (joy → happiness)
MELD_TEST = {"neutral": 1256, "happiness": 402, "anger": 345,
             "surprise": 281, "sadness": 208, "disgust": 68, "fear": 50}  # 2610 total

# ── Collect all semantic labels that appear ───────────────────────────────────
all_sem_labels_train = set()
for ds in DS_ORDER:
    all_sem_labels_train.update(stats[ds]["sem"].keys())

# Order: neutral first, no-emotion last, others by total count
FIXED_ORDER = ["neutral", "happiness", "anger", "sadness", "surprise", "fear", "disgust",
               "excited", "frustrated", "calm", "amused", "sleepy", "no emotion", "other"]
sem_cols = [l for l in FIXED_ORDER if l in all_sem_labels_train]
# Add any remaining
for l in sorted(all_sem_labels_train):
    if l not in sem_cols:
        sem_cols.append(l)

# ── Build matrices ────────────────────────────────────────────────────────────
n_ds = len(DS_ORDER)
n_em = len(sem_cols)
mat = np.zeros((n_ds, n_em))
for i, ds in enumerate(DS_ORDER):
    for j, em in enumerate(sem_cols):
        mat[i, j] = stats[ds]["sem"].get(em, 0)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 20))
fig.patch.set_facecolor("#fafafa")

gs = fig.add_gridspec(
    3, 2,
    height_ratios=[1.4, 1.1, 1.1],
    hspace=0.42, wspace=0.30,
    left=0.08, right=0.97, top=0.93, bottom=0.05,
)

ax_heat  = fig.add_subplot(gs[0, :])
ax_bar   = fig.add_subplot(gs[1, 0])
ax_raw   = fig.add_subplot(gs[1, 1])
ax_eval  = fig.add_subplot(gs[2, :])

# ── Panel A: Heatmap (semantic labels) ───────────────────────────────────────
from matplotlib.colors import LogNorm

mat_plot = mat.copy()
mat_plot[mat_plot == 0] = np.nan
vmin, vmax = 10, 25000

im = ax_heat.imshow(mat_plot, aspect="auto",
                    norm=LogNorm(vmin=vmin, vmax=vmax), cmap="YlOrRd")

for i in range(n_ds):
    for j in range(n_em):
        v = mat_plot[i, j]
        if np.isnan(v):
            ax_heat.text(j, i, "–", ha="center", va="center", fontsize=7.5, color="#cccccc")
        else:
            txt = f"{int(v):,}" if v >= 1000 else str(int(v))
            col = "white" if v > 3000 else "#333333"
            ax_heat.text(j, i, txt, ha="center", va="center",
                         fontsize=7, color=col, fontweight="bold")

ds_tick_labels = [DS_LABELS[d] for d in DS_ORDER]
ax_heat.set_yticks(range(n_ds))
ax_heat.set_yticklabels(ds_tick_labels, fontsize=9)
ax_heat.set_xticks(range(n_em))
ax_heat.set_xticklabels(sem_cols, rotation=35, ha="right", fontsize=9)

cb = fig.colorbar(im, ax=ax_heat, fraction=0.010, pad=0.01)
cb.set_label("Sample count (log scale)", fontsize=8)

totals = [stats[ds]["total"] for ds in DS_ORDER]
grand_total = sum(totals)
ax_heat.set_title(
    f"A.  V6 Train Manifest  —  {n_ds} datasets, {grand_total:,} samples total  "
    f"(semantic label grouping; – = absent)",
    fontsize=11, fontweight="bold", pad=8,
)

# ── Panel B: Stacked % bar per dataset (semantic) ────────────────────────────
bar_colors = [EMOTION_COLORS.get(l, "#999999") for l in sem_cols]
bottoms = np.zeros(n_ds)
bar_ys = np.arange(n_ds)
for j, (col, color) in enumerate(zip(sem_cols, bar_colors)):
    vals = mat[:, j] / np.array(totals)
    ax_bar.barh(bar_ys, vals, left=bottoms, color=color, height=0.7,
                label=col, edgecolor="white", linewidth=0.3)
    bottoms += vals

ax_bar.set_yticks(bar_ys)
ax_bar.set_yticklabels(ds_tick_labels, fontsize=8.5)
ax_bar.set_xlim(0, 1)
ax_bar.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax_bar.axvline(0.5, color="gray", linestyle=":", linewidth=0.8)
ax_bar.set_title("B.  Label Composition per Dataset  (% of total)",
                 fontsize=10, fontweight="bold", pad=6)

legend_patches = [
    mpatches.Patch(color=EMOTION_COLORS.get(l, "#999999"), label=l)
    for l in sem_cols if mat[:, sem_cols.index(l)].sum() > 0
]
ax_bar.legend(handles=legend_patches, fontsize=6.5, ncol=2,
              loc="lower right", framealpha=0.85, bbox_to_anchor=(1.0, 0.0))

# ── Panel C: Raw label space per dataset (unique labels count + names) ────────
ax_raw.axis("off")
col_x = [0.01, 0.18, 0.35, 0.52, 0.69, 0.86]
header_y = 0.97

# Header
for x, ds in zip(col_x, DS_ORDER):
    n = stats[ds]["total"]
    ax_raw.text(x, header_y, f"{DS_LABELS[ds].replace(chr(10), ' ')}\n({n:,})",
                fontsize=7.5, fontweight="bold", va="top", transform=ax_raw.transAxes)

# Unique raw labels per dataset
y_start = 0.82
y_step  = 0.072
for x, ds in zip(col_x, DS_ORDER):
    raw_sorted = stats[ds]["raw"].most_common()
    for k, (lbl, cnt) in enumerate(raw_sorted[:10]):
        y = y_start - k * y_step
        if y < 0.02:
            ax_raw.text(x, y + y_step, f"  …+{len(raw_sorted)-k} more",
                        fontsize=6, color="#888888", va="top", transform=ax_raw.transAxes)
            break
        pct = 100 * cnt / stats[ds]["total"]
        col = EMOTION_COLORS.get(SEMANTIC_MAP.get(lbl, lbl), "#999999")
        ax_raw.text(x, y, f"● {lbl}  {cnt} ({pct:.0f}%)",
                    fontsize=6.5, va="top", transform=ax_raw.transAxes, color=col)

ax_raw.set_title("C.  Raw Label Inventory per Dataset  (top-10 by count)",
                 fontsize=10, fontweight="bold", pad=6)

# ── Panel D: Train combined vs Eval ──────────────────────────────────────────
# Train: sum all sem labels, exclude "no emotion" and "other" for comparison
train_combined = Counter()
for ds in DS_ORDER:
    train_combined.update(stats[ds]["sem"])

# Build comparison sets
CORE_LABELS = ["neutral", "happiness", "anger", "sadness", "surprise", "fear", "disgust",
               "excited", "frustrated", "calm", "amused", "sleepy"]

sets = {
    "Train\n(all, balanced)":    {l: train_combined.get(l, 0) for l in CORE_LABELS},
    "Train\n(excl. no-emo/other)": {l: train_combined.get(l, 0) for l in CORE_LABELS},
    "MELD\ntest":                {l: MELD_TEST.get(l, 0) for l in CORE_LABELS},
    "IEMOCAP S5\neval":          {l: IEMOCAP_S5.get(l, 0) for l in CORE_LABELS},
}
# For Train excl, strip no-emotion
train_excl_total = sum(train_combined.get(l, 0) for l in CORE_LABELS)
sets["Train\n(excl. no-emo/other)"] = {l: train_combined.get(l, 0) for l in CORE_LABELS}

set_names = list(sets.keys())
n_sets = len(set_names)
bar_ys2 = np.arange(n_sets)
bottoms2 = np.zeros(n_sets)

for em in CORE_LABELS:
    row_counts = np.array([sets[s].get(em, 0) for s in set_names], dtype=float)
    row_totals = np.array([sum(sets[s].values()) for s in set_names], dtype=float)
    row_totals = np.where(row_totals == 0, 1, row_totals)
    row_pct = row_counts / row_totals
    color = EMOTION_COLORS.get(em, "#999999")
    ax_eval.barh(bar_ys2, row_pct, left=bottoms2, color=color, height=0.65,
                 label=em, edgecolor="white", linewidth=0.3)
    bottoms2 += row_pct

# Annotate total sample count
for i, sname in enumerate(set_names):
    total = sum(sets[sname].values())
    ax_eval.text(1.01, i, f"n={total:,}", va="center", fontsize=8,
                 transform=ax_eval.get_yaxis_transform())

ax_eval.set_yticks(bar_ys2)
ax_eval.set_yticklabels(set_names, fontsize=9.5)
ax_eval.set_xlim(0, 1)
ax_eval.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax_eval.axvline(0.5, color="gray", linestyle=":", linewidth=0.8)
ax_eval.set_title(
    "D.  Train vs. Eval Emotion Distribution  (% of total; MELD joy→happiness renamed)",
    fontsize=10, fontweight="bold", pad=6,
)

legend_patches2 = [
    mpatches.Patch(color=EMOTION_COLORS.get(l, "#999999"), label=l)
    for l in CORE_LABELS if any(sets[s].get(l, 0) > 0 for s in set_names)
]
ax_eval.legend(handles=legend_patches2, fontsize=7.5, ncol=4,
               loc="lower right", framealpha=0.9)

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    f"V6 Emotion Train Manifests  —  {grand_total:,} train samples across {n_ds} datasets  "
    f"|  Eval: MELD-test (n=2,610) + IEMOCAP-S5 (n=1,507)  |  2026-05-11",
    fontsize=11, fontweight="bold", y=0.975,
)

out = "/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/docs/analysis/emotion_v6_viz.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")
