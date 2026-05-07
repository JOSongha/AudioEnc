#!/usr/bin/env python
"""Build LibriTTS-R audio_asr shards directly from nubes.

Source layout on nubes (`hyperscaleai-audiollm` bucket):
    /datasets/public/libriTTS/{train-clean-100, train-clean-360,
    train-other-500}/<spk>/<chapter>/
        <spk>_<chapter>.trans.tsv          (chapter transcripts, one row per utt:
                                            utt_id \\t original \\t normalized)
        <spk>_<chapter>.book.tsv           (chapter metadata, ignored)
        <utt_id>.wav                       (audio)
        <utt_id>.{normalized,original}.txt (per-utt transcripts; the normalized
                                            file equals trans.tsv col 3 verbatim
                                            so we skip per-file fetch)

We fetch one `.trans.tsv` per chapter (not per utterance), parse each line,
and emit `{"nubes_path": "...wav", "text": "<normalized>", "modality":
"audio_asr", "source": "libritts_r"}` rows. This is ~10K HTTP fetches
(one per chapter) instead of 354,721 per-utt fetches — an order of
magnitude faster than the gigaspeech / voxpopuli pattern.

Splits ingested:
    train-clean-100   (33,232 utt, per docs/setup/datasets.md §2)
    train-clean-360   (116,454 utt)
    train-other-500   (205,035 utt)
    Total target: 354,721 utt

Held out (excluded from train, used by eval_librispeech_wer):
    dev-clean / dev-other / test-clean / test-other (not enumerated here).

The normalized text variant is used as target. LibriTTS-R is the upstream
LibriTTS reissue; the file naming with `.wav` extension is the canonical
LibriTTS layout. Runtime resampling to encoder SR happens in omni_dataset.py
regardless of source SR.

No /mnt/ddn/users/<person>/ caches. No /mnt/ddn/omni_dataset/audio/ jsonl.
Only nubes.

Usage:
    python -m scripts.manifest_builders.build_libritts_r \\
        --out /mnt/tmp/datasets/manifests/v6/audio_asr --workers 32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manifest_builders._nubes_helper import (  # noqa: E402
    fetch_text_files_parallel, list_dir,
)


SPLITS = ["train-clean-100", "train-clean-360", "train-other-500"]
PREFIX_BASE = "datasets/public/libriTTS/"
NUBES_PREFIX_FULL = "hyperscaleai-audiollm/datasets/public/libriTTS/"
OUT_DEFAULT = Path("/mnt/tmp/datasets/manifests/v6/audio_asr")


def enumerate_chapter_tsv_paths(split: str) -> list[tuple[str, str, str]]:
    """Walk <split>/<spk>/<chapter>/ and return [(split, spk, chapter), ...]."""
    out: list[tuple[str, str, str]] = []
    split_prefix = f"{PREFIX_BASE}{split}/"
    for spk_entry in list_dir(split_prefix):
        if not spk_entry.get("IsDir"):
            continue
        spk = spk_entry["Name"]
        spk_prefix = f"{split_prefix}{spk}/"
        for ch_entry in list_dir(spk_prefix):
            if not ch_entry.get("IsDir"):
                continue
            chapter = ch_entry["Name"]
            out.append((split, spk, chapter))
    return out


def parse_trans_tsv(text: str) -> list[tuple[str, str]]:
    """Return list of (utt_id, normalized_text) from a chapter trans.tsv body."""
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        utt_id, _original, normalized = parts[0], parts[1], parts[2]
        rows.append((utt_id, normalized))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--shard-size", type=int, default=88000)
    ap.add_argument("--prefix", default="libritts_r")
    ap.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    ap.add_argument("--workers", type=int, default=32,
                    help="Parallel chapter-trans.tsv fetch workers.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # 1) Enumerate (split, spk, chapter) triples.
    print(f"[libritts_r] enumerating splits {args.splits} ...", flush=True)
    chapters: list[tuple[str, str, str]] = []
    for split in args.splits:
        before = len(chapters)
        chapters.extend(enumerate_chapter_tsv_paths(split))
        print(f"  {split}: +{len(chapters) - before} chapters "
              f"(running total {len(chapters)})", flush=True)
    print(f"[libritts_r] total chapters: {len(chapters):,}", flush=True)

    # 2) Build trans.tsv path list keyed by chapter triple.
    tsv_paths_by_chapter: dict[str, tuple[str, str, str]] = {}
    for split, spk, chapter in chapters:
        tsv_path = (f"{PREFIX_BASE}{split}/{spk}/{chapter}/"
                    f"{spk}_{chapter}.trans.tsv")
        tsv_paths_by_chapter[tsv_path] = (split, spk, chapter)

    # 3) Fetch trans.tsv files in parallel and emit rows.
    shard_idx = 0
    shard_buf: list[str] = []
    n_utts = 0
    n_chapters_done = 0
    n_chapters_empty = 0
    per_split: dict[str, int] = {s: 0 for s in args.splits}

    def flush():
        nonlocal shard_idx, shard_buf
        if not shard_buf:
            return
        out_path = args.out / f"{args.prefix}_{shard_idx:04d}.jsonl"
        if args.resume and out_path.exists():
            print(f"  shard {shard_idx:04d}: skip (exists)", flush=True)
            shard_buf = []
            shard_idx += 1
            return
        if not args.dry_run:
            with open(out_path, "w") as f:
                f.writelines(shard_buf)
        print(f"  shard {shard_idx:04d}: {len(shard_buf):,} rows -> {out_path}",
              flush=True)
        shard_idx += 1
        shard_buf = []

    print(f"[libritts_r] fetching {len(tsv_paths_by_chapter):,} trans.tsv "
          f"files in parallel (workers={args.workers}) ...", flush=True)
    for tsv_path, body in fetch_text_files_parallel(
            list(tsv_paths_by_chapter.keys()), workers=args.workers):
        n_chapters_done += 1
        if not body:
            n_chapters_empty += 1
            continue
        split, spk, chapter = tsv_paths_by_chapter[tsv_path]
        for utt_id, text in parse_trans_tsv(body):
            wav_path = (f"{NUBES_PREFIX_FULL}{split}/{spk}/{chapter}/"
                        f"{utt_id}.wav")
            out_row = {
                "nubes_path": wav_path,
                "text": text,
                "modality": "audio_asr",
                "source": "libritts_r",
            }
            shard_buf.append(json.dumps(out_row, ensure_ascii=False) + "\n")
            n_utts += 1
            per_split[split] += 1
            if len(shard_buf) >= args.shard_size:
                flush()
        if n_chapters_done % 1000 == 0:
            print(f"  chapters {n_chapters_done:,}/"
                  f"{len(tsv_paths_by_chapter):,}  "
                  f"utts={n_utts:,}  empty={n_chapters_empty:,}",
                  flush=True)
    flush()

    print(f"[libritts_r] DONE: chapters={n_chapters_done:,} "
          f"empty={n_chapters_empty:,} utts={n_utts:,} "
          f"shards={shard_idx}", flush=True)
    for s in args.splits:
        print(f"  {s}: {per_split[s]:,} utts", flush=True)


if __name__ == "__main__":
    main()
