#!/usr/bin/env python3
"""Word alignment statistical sanity check.

각 dataset/split Arrow 파일에서 샘플링하여 alignment 품질 통계를 출력.
"""
import json, random, sys, time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

MERGED = Path("/mnt/tmp/cache/word_alignments_merged")

TARGETS = [
    ("librispeech", "dev-clean"),
    ("librispeech", "train-clean-100"),
    ("librispeech", "train-other-500"),
    ("mls",         "train"),
    ("gigaspeech",  "train"),
    ("voxpopuli",   "train"),
]

SAMPLE_N = 5000   # 각 split에서 랜덤 샘플 수


def load_sample(arrow_path: Path, n: int) -> list[dict]:
    with open(arrow_path, "rb") as f:
        reader = ipc.open_file(f)
        total = sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))

    indices = set(random.sample(range(total), min(n, total)))

    rows = []
    with open(arrow_path, "rb") as f:
        reader = ipc.open_file(f)
        offset = 0
        for bi in range(reader.num_record_batches):
            batch = reader.get_batch(bi)
            for li in range(batch.num_rows):
                gi = offset + li
                if gi in indices:
                    row = {col: batch.column(col)[li].as_py()
                           for col in batch.schema.names}
                    rows.append(row)
            offset += batch.num_rows
    return rows


def analyze(rows: list[dict], label: str) -> dict:
    coverage_ratios = []
    wps_list = []          # words per second
    gap_list = []          # inter-word gaps (sec)
    overlap_count = 0
    oob_count = 0          # last_word_end > audio_duration + 0.1
    missing_words = 0      # transcript words != aligned words
    score_list = []        # LibriSpeech only
    total = len(rows)

    for row in rows:
        words = row.get("words") or []
        dur = row.get("audio_duration") or 0
        transcript = row.get("transcript") or ""

        if not words or dur <= 0:
            missing_words += 1
            continue

        # transcript word count vs aligned word count
        n_transcript = len(transcript.split())
        n_aligned = len(words)
        if abs(n_transcript - n_aligned) > max(1, n_transcript * 0.1):
            missing_words += 1

        starts = [w["start"] for w in words]
        ends = [w["end"] for w in words]

        # coverage
        coverage_ratios.append(ends[-1] / dur if dur > 0 else 0)

        # words per second
        span = ends[-1] - starts[0]
        if span > 0:
            wps_list.append(n_aligned / span)

        # gaps between consecutive words
        for i in range(1, len(words)):
            gap = starts[i] - ends[i-1]
            gap_list.append(gap)
            if gap < -0.02:   # 20ms tolerance
                overlap_count += 1

        # out of bounds
        if ends[-1] > dur + 0.1:
            oob_count += 1

        # scores (LibriSpeech)
        for w in words:
            if "score" in w and w["score"] is not None:
                score_list.append(w["score"])

    def pct(arr, p):
        return float(np.percentile(arr, p)) if arr else float("nan")

    result = {
        "label": label,
        "n_sampled": total,
        "coverage_ratio": {
            "mean": float(np.mean(coverage_ratios)) if coverage_ratios else float("nan"),
            "p5":   pct(coverage_ratios, 5),
            "p50":  pct(coverage_ratios, 50),
            "p95":  pct(coverage_ratios, 95),
        },
        "words_per_sec": {
            "mean": float(np.mean(wps_list)) if wps_list else float("nan"),
            "p5":   pct(wps_list, 5),
            "p50":  pct(wps_list, 50),
            "p95":  pct(wps_list, 95),
        },
        "inter_word_gap_sec": {
            "mean": float(np.mean(gap_list)) if gap_list else float("nan"),
            "p5":   pct(gap_list, 5),
            "p50":  pct(gap_list, 50),
            "p95":  pct(gap_list, 95),
        },
        "overlap_count":   overlap_count,
        "overlap_rate":    overlap_count / max(len(gap_list), 1),
        "oob_count":       oob_count,
        "oob_rate":        oob_count / max(total, 1),
        "word_count_mismatch": missing_words,
        "word_count_mismatch_rate": missing_words / max(total, 1),
    }
    if score_list:
        result["score"] = {
            "mean": float(np.mean(score_list)),
            "p5":   pct(score_list, 5),
            "p50":  pct(score_list, 50),
            "p95":  pct(score_list, 95),
        }
    return result


def fmt(r: dict) -> str:
    lines = []
    lines.append(f"\n### {r['label']}  (n={r['n_sampled']:,})")
    c = r["coverage_ratio"]
    lines.append(f"  coverage (last_end/dur)  mean={c['mean']:.3f}  p5={c['p5']:.3f}  p50={c['p50']:.3f}  p95={c['p95']:.3f}")
    w = r["words_per_sec"]
    lines.append(f"  words/sec                mean={w['mean']:.2f}  p5={w['p5']:.2f}  p50={w['p50']:.2f}  p95={w['p95']:.2f}")
    g = r["inter_word_gap_sec"]
    lines.append(f"  inter-word gap (s)       mean={g['mean']:.3f}  p5={g['p5']:.3f}  p50={g['p50']:.3f}  p95={g['p95']:.3f}")
    lines.append(f"  overlap rate             {r['overlap_rate']*100:.2f}%  ({r['overlap_count']} pairs)")
    lines.append(f"  out-of-bounds rate       {r['oob_rate']*100:.2f}%  ({r['oob_count']} utts)")
    lines.append(f"  word-count mismatch      {r['word_count_mismatch_rate']*100:.2f}%  ({r['word_count_mismatch']} utts)")
    if "score" in r:
        s = r["score"]
        lines.append(f"  CTC score                mean={s['mean']:.3f}  p5={s['p5']:.3f}  p50={s['p50']:.3f}  p95={s['p95']:.3f}")
    return "\n".join(lines)


all_results = []
for dataset, split in TARGETS:
    arrow_path = MERGED / dataset / f"{split}.arrow"
    if not arrow_path.exists():
        print(f"[SKIP] {dataset}/{split}: 파일 없음", flush=True)
        continue
    print(f"[{dataset}/{split}] 샘플링 중...", flush=True)
    t0 = time.time()
    rows = load_sample(arrow_path, SAMPLE_N)
    result = analyze(rows, f"{dataset}/{split}")
    print(fmt(result), flush=True)
    all_results.append(result)

# JSON 저장
out_json = Path("/mnt/fr20tb/wbl_residency/jos/AudioEnc/docs/quality_check_results.json")
out_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
print(f"\n\n결과 저장: {out_json}")
