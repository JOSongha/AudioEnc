"""
Train / Test 분할 기준 emotion class 분포 시각화.

규칙:
  Train = (split 없는 데이터 전량) + (split 있는 데이터의 train portion)
  Test  = split 있는 데이터의 test portion 만

CMU-MOSEI: 다중 레이블 → Panel A(per-emotion bar)에서 제외, Panel B/C source breakdown에만 포함.

Output: docs/analysis/emotion_split_viz.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

# ── Single-label Train sources ────────────────────────────────────────────────
TRAIN_SOURCES = [
    # v6 train pool, split 있음
    ("IEMOCAP S1-4",   {"anger":933, "disgust":2, "fear":30, "happiness":452,
                        "neutral":1324, "sadness":839, "surprise":89,
                        "excited":742, "frustrated":1468}),
    ("MELD tr+dev",    {"anger":1262, "disgust":293, "fear":308, "happiness":1906,
                        "neutral":5179, "sadness":794, "surprise":1355}),
    # v6 train pool, split 없음
    ("DailyTalk",      {"anger":159, "disgust":66, "fear":18, "happiness":3856,
                        "neutral":18966, "sadness":301, "surprise":407}),
    ("EmoV-DB",        {"anger":1268, "disgust":1019, "neutral":1568,
                        "amused":1317, "sleepy":1721}),
    ("RAVDESS",        {"anger":192, "disgust":192, "fear":192, "happiness":192,
                        "neutral":96, "sadness":192, "surprise":192, "calm":192}),
    ("MUStARD++",      {"anger":53, "disgust":29, "fear":23, "happiness":244,
                        "neutral":438, "sadness":149, "surprise":101,
                        "excited":115, "frustrated":48}),
    # eval-only, split 있음 → train split
    ("CREMA-D train",  {"anger":914, "disgust":559, "fear":623, "happiness":355,
                        "neutral":3199, "sadness":303}),
    # eval-only, split 없음
    ("SAVEE",          {"anger":60, "disgust":60, "fear":60, "happiness":60,
                        "neutral":120, "sadness":60, "surprise":60}),
    ("TESS",           {"anger":400, "disgust":400, "fear":400, "happiness":400,
                        "neutral":400, "sadness":400, "surprise":400}),
    ("ESD (EN)",       {"anger":3500, "happiness":3500, "neutral":3500,
                        "sadness":3500, "surprise":3500}),
    ("JL-Corpus",      {"anger":240, "happiness":240, "neutral":240, "sadness":240,
                        "anxious":240, "apologetic":240, "assertive":240,
                        "concerned":240, "encouraging":240, "excited":240}),
    # ── 신규 eval-only, split 없음 ────────────────────────────────────────────
    ("MSP-IMPROV",     {"anger":792, "happiness":2167, "neutral":3477, "sadness":2002}),
    ("EmoVoice-DB\n†synth",
                       {"anger":3486, "disgust":2950, "fear":2961, "happiness":3269,
                        "neutral":3188, "sadness":3174, "surprise":3072}),
    ("eNTERFACE'05",   {"anger":194, "disgust":195, "fear":194, "happiness":195,
                        "sadness":194, "surprise":194}),
]

# ── Single-label Test sources ─────────────────────────────────────────────────
TEST_SOURCES = [
    ("IEMOCAP S5\n(4-class)", {"anger":197, "happiness":613, "neutral":386, "sadness":311}),
    ("MELD test",             {"anger":345, "disgust":67, "fear":50, "happiness":401,
                               "neutral":1256, "sadness":208, "surprise":281}),
    ("CREMA-D test",          {"anger":239, "disgust":125, "fear":170, "happiness":76,
                               "neutral":812, "sadness":67}),
]

# ── CMU-MOSEI multi-label (separate) ─────────────────────────────────────────
# 총 22,856 utterances, 공식 split: train 14,524 / valid 1,765 / test 4,188
MOSEI_TRAIN_N = 16289   # train + valid
MOSEI_TEST_N  = 4188
# label counts (multi-label, score >= 1)
MOSEI_EMOTIONS = {"anger":4600, "disgust":3755, "fear":1803,
                  "happiness":10752, "sadness":5601, "surprise":2055}

# ── Aggregate ─────────────────────────────────────────────────────────────────
def aggregate(sources):
    total = {}
    for _, ed in sources:
        for k, v in ed.items():
            total[k] = total.get(k, 0) + v
    return total

train_total = aggregate(TRAIN_SOURCES)
test_total  = aggregate(TEST_SOURCES)

CORE7 = ["neutral", "happiness", "anger", "sadness", "surprise", "fear", "disgust"]
all_emotions = sorted(
    set(train_total) | set(test_total),
    key=lambda e: (CORE7.index(e) if e in CORE7 else 99, e)
)

COLORS = {
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

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 18))
fig.patch.set_facecolor("#fafafa")

gs = fig.add_gridspec(
    2, 2,
    height_ratios=[1.1, 1],
    hspace=0.38, wspace=0.28,
    left=0.07, right=0.97, top=0.93, bottom=0.05,
)
ax_bar = fig.add_subplot(gs[0, :])
ax_tr  = fig.add_subplot(gs[1, 0])
ax_te  = fig.add_subplot(gs[1, 1])

# ── Panel A: Grouped bar (single-label only) ──────────────────────────────────
x = np.arange(len(all_emotions))
w = 0.35

train_vals = [train_total.get(e, 0) for e in all_emotions]
test_vals  = [test_total.get(e, 0)  for e in all_emotions]
sl_train_total = sum(train_total.values())
sl_test_total  = sum(test_total.values())

bars_tr = ax_bar.bar(x - w/2, train_vals, width=w, color="#2196F3",
                     alpha=0.82, label=f"Train SL ({sl_train_total:,})",
                     edgecolor="white", linewidth=0.4)
bars_te = ax_bar.bar(x + w/2, test_vals, width=w, color="#e63946",
                     alpha=0.82, label=f"Test SL ({sl_test_total:,})",
                     edgecolor="white", linewidth=0.4)

for bar in bars_tr:
    h = bar.get_height()
    if h > 0:
        ax_bar.text(bar.get_x() + bar.get_width()/2, h + 120,
                    f"{int(h):,}", ha="center", va="bottom", fontsize=6.5,
                    color="#1565C0", fontweight="bold")
for bar in bars_te:
    h = bar.get_height()
    if h > 0:
        ax_bar.text(bar.get_x() + bar.get_width()/2, h + 120,
                    f"{int(h):,}", ha="center", va="bottom", fontsize=6.5,
                    color="#b71c1c", fontweight="bold")

# Overlay CMU-MOSEI multi-label label counts
mosei_tr_prop = MOSEI_TRAIN_N / (MOSEI_TRAIN_N + MOSEI_TEST_N)
mosei_te_prop = MOSEI_TEST_N  / (MOSEI_TRAIN_N + MOSEI_TEST_N)
for j, em in enumerate(all_emotions):
    ml_tr = int(MOSEI_EMOTIONS.get(em, 0) * mosei_tr_prop)
    ml_te = int(MOSEI_EMOTIONS.get(em, 0) * mosei_te_prop)
    if ml_tr > 0:
        ax_bar.bar(j - w/2, ml_tr, width=w, bottom=train_vals[j],
                   color="#ff9800", alpha=0.55, edgecolor="white",
                   linewidth=0.4, hatch="//")
    if ml_te > 0:
        ax_bar.bar(j + w/2, ml_te, width=w, bottom=test_vals[j],
                   color="#ff9800", alpha=0.55, edgecolor="white",
                   linewidth=0.4, hatch="//")

ml_patch = mpatches.Patch(facecolor="#ff9800", alpha=0.6, hatch="//",
                           label=f"CMU-MOSEI ★ (ML) Tr{MOSEI_TRAIN_N:,}/Te{MOSEI_TEST_N:,}")
ax_bar.legend(handles=[
    mpatches.Patch(color="#2196F3", alpha=0.82, label=f"Train SL ({sl_train_total:,})"),
    mpatches.Patch(color="#e63946", alpha=0.82, label=f"Test SL ({sl_test_total:,})"),
    ml_patch,
], fontsize=8.5, framealpha=0.9)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(all_emotions, rotation=30, ha="right", fontsize=9.5)
ax_bar.set_ylabel("Sample count", fontsize=9)
ax_bar.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax_bar.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)

# Shade core-7
core7_xs = [i for i, e in enumerate(all_emotions) if e in CORE7]
if core7_xs:
    ax_bar.axvspan(min(core7_xs) - 0.5, max(core7_xs) + 0.5,
                   color="#e8f5e9", alpha=0.35, zorder=0)
    ax_bar.text(max(core7_xs) + 0.5, ax_bar.get_ylim()[1] * 0.97,
                " core 7", fontsize=8, color="#388e3c", va="top")

ax_bar.set_title(
    f"A.  Train vs. Test — Emotion Count\n"
    f"SL Train: {sl_train_total:,}  |  SL Test: {sl_test_total:,}  |  "
    f"★ CMU-MOSEI (multi-label): Train {MOSEI_TRAIN_N:,} / Test {MOSEI_TEST_N:,} utterances",
    fontsize=11, fontweight="bold", pad=8,
)

# ── Panel B: Train stacked bar by source ─────────────────────────────────────
bar_ys = np.arange(len(TRAIN_SOURCES) + 1)  # +1 for CMU-MOSEI
src_labels_tr = [s[0] for s in TRAIN_SOURCES] + ["CMU-MOSEI\n★ML tr+val"]
bottoms = np.zeros(len(TRAIN_SOURCES) + 1)

for em in all_emotions:
    vals = np.array([s[1].get(em, 0) for s in TRAIN_SOURCES] + [0], dtype=float)
    ax_tr.barh(bar_ys, vals, left=bottoms,
               color=COLORS.get(em, "#999999"), height=0.72,
               label=em, edgecolor="white", linewidth=0.25)
    bottoms += vals

# CMU-MOSEI as single block (multi-label, use utterance count)
mosei_row = len(TRAIN_SOURCES)
ax_tr.barh(mosei_row, MOSEI_TRAIN_N, left=0, color="#ff9800",
           height=0.72, alpha=0.7, hatch="//",
           edgecolor="white", linewidth=0.4)
ax_tr.text(MOSEI_TRAIN_N + 200, mosei_row,
           f"{MOSEI_TRAIN_N:,} utterances", va="center", fontsize=7, color="#333")

for i, (_, ed) in enumerate(TRAIN_SOURCES):
    tot = sum(ed.values())
    ax_tr.text(tot + 200, i, f"{tot:,}", va="center", fontsize=6.5, color="#333")

ax_tr.set_yticks(bar_ys)
ax_tr.set_yticklabels(src_labels_tr, fontsize=7.5)
ax_tr.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax_tr.set_xlabel("Sample count", fontsize=8)
ax_tr.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
ax_tr.set_title(
    f"B.  Train set by source\nSL total: {sl_train_total:,}  +  CMU-MOSEI {MOSEI_TRAIN_N:,} utter.",
    fontsize=10, fontweight="bold", pad=6,
)

legend_handles = [mpatches.Patch(color=COLORS.get(e, "#999999"), label=e)
                  for e in all_emotions]
legend_handles += [mpatches.Patch(facecolor="#ff9800", alpha=0.7, hatch="//",
                                  label="CMU-MOSEI (ML)")]
ax_tr.legend(handles=legend_handles, fontsize=5.5, ncol=3,
             loc="lower right", framealpha=0.85,
             bbox_to_anchor=(1.01, -0.02))

# ── Panel C: Test stacked bar by source ──────────────────────────────────────
bar_ys_te = np.arange(len(TEST_SOURCES) + 1)
src_labels_te = [s[0] for s in TEST_SOURCES] + ["CMU-MOSEI\n★ML test"]
bottoms = np.zeros(len(TEST_SOURCES) + 1)

for em in all_emotions:
    vals = np.array([s[1].get(em, 0) for s in TEST_SOURCES] + [0], dtype=float)
    if vals.sum() == 0:
        continue
    ax_te.barh(bar_ys_te, vals, left=bottoms,
               color=COLORS.get(em, "#999999"), height=0.55,
               label=em, edgecolor="white", linewidth=0.25)
    bottoms += vals

mosei_te_row = len(TEST_SOURCES)
ax_te.barh(mosei_te_row, MOSEI_TEST_N, left=0, color="#ff9800",
           height=0.55, alpha=0.7, hatch="//",
           edgecolor="white", linewidth=0.4)
ax_te.text(MOSEI_TEST_N + 50, mosei_te_row,
           f"{MOSEI_TEST_N:,} utterances", va="center", fontsize=7.5, color="#333")

for i, (_, ed) in enumerate(TEST_SOURCES):
    tot = sum(ed.values())
    ax_te.text(tot + 50, i, f"{tot:,}", va="center", fontsize=8, color="#333")

ax_te.set_yticks(bar_ys_te)
ax_te.set_yticklabels(src_labels_te, fontsize=9)
ax_te.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax_te.set_xlabel("Sample count", fontsize=8)
ax_te.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
ax_te.set_title(
    f"C.  Test set by source\nSL total: {sl_test_total:,}  +  CMU-MOSEI {MOSEI_TEST_N:,} utter.",
    fontsize=10, fontweight="bold", pad=6,
)

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    f"Emotion Dataset — Train / Test Split Distribution  "
    f"(SL {sl_train_total + sl_test_total:,} + CMU-MOSEI 22,856 utter., 2026-05-10)",
    fontsize=13, fontweight="bold", y=0.975,
)

out = "/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/docs/analysis/emotion_split_viz.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")

# ── Print summary for markdown ────────────────────────────────────────────────
print("\n=== TRAIN (single-label) ===")
print(f"{'emotion':<14} {'count':>8}  {'pct':>6}")
for e in all_emotions:
    v = train_total.get(e, 0)
    if v:
        print(f"{e:<14} {v:>8,}  {v/sl_train_total*100:>5.1f}%")
print(f"{'TOTAL':<14} {sl_train_total:>8,}  100.0%")
print(f"  + CMU-MOSEI: {MOSEI_TRAIN_N:,} utterances (multi-label)")

print("\n=== TEST (single-label) ===")
for e in all_emotions:
    v = test_total.get(e, 0)
    if v:
        print(f"{e:<14} {v:>8,}  {v/sl_test_total*100:>5.1f}%")
print(f"{'TOTAL':<14} {sl_test_total:>8,}  100.0%")
print(f"  + CMU-MOSEI: {MOSEI_TEST_N:,} utterances (multi-label)")
