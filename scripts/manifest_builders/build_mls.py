#!/usr/bin/env python
"""Build MLS English audio_asr shards directly from nubes.

Source layout on nubes:
    /datasets/public/16kHz/mls/mls_english/train/transcripts.txt   (2.45 GB, single file)
        line format: <utt_id>\\t<text>
        utt_id format: <spk>_<book>_<seq>
    /datasets/public/16kHz/mls/mls_english/train/audio/<spk>/<book>/<utt_id>.flac

10,808,037 utterances total. Unlike GigaSpeech / VoxPopuli, MLS keeps
all transcripts in a single tab-separated file, so no per-file fetch
needed — we download transcripts.txt once and parse.

No dependency on /mnt/ddn/users/<person>/ caches. No dependency on
/mnt/ddn/omni_dataset/audio/mls_english/ jsonl. Only nubes.

Runtime: 2.45 GB single-file download + parse ~ a few minutes.

Usage:
    python -m scripts.manifest_builders.build_mls \\
        --out /mnt/tmp/datasets/manifests/v6/audio_asr
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manifest_builders._nubes_helper import BUCKET, NUBES_GATEWAY  # noqa: E402


TRANSCRIPTS_NUBES = (
    "datasets/public/16kHz/mls/mls_english/train/transcripts.txt"
)
AUDIO_PREFIX = (
    "hyperscaleai-audiollm/datasets/public/16kHz/mls/mls_english/train/audio/"
)
OUT_DEFAULT = Path("/mnt/tmp/datasets/manifests/v6/audio_asr")


def utt_to_audio_path(utt_id: str) -> str | None:
    """`<spk>_<book>_<seq>` -> nubes audio path with full bucket prefix."""
    parts = utt_id.split("_")
    if len(parts) < 3:
        return None
    spk, book = parts[0], parts[1]
    return f"{AUDIO_PREFIX}{spk}/{book}/{utt_id}.flac"


def stream_transcripts():
    """Yield (utt_id, text) pairs from the streamed transcripts.txt."""
    url = f"{NUBES_GATEWAY}/{BUCKET}/{TRANSCRIPTS_NUBES}"
    with urllib.request.urlopen(url, timeout=300.0) as r:
        # Read line-by-line. Avoid loading 2.45 GB into RAM at once.
        for raw in r:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if not line:
                continue
            tab = line.find("\t")
            if tab < 0:
                continue
            yield line[:tab], line[tab + 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--shard-size", type=int, default=88000)
    ap.add_argument("--prefix", default="mls")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    shard_buf: list[str] = []
    n_done = 0
    n_skipped = 0

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

    print(f"[mls] streaming nubes:{TRANSCRIPTS_NUBES} ...", flush=True)
    for utt_id, text in stream_transcripts():
        nubes_path = utt_to_audio_path(utt_id)
        if nubes_path is None:
            n_skipped += 1
            continue
        out_row = {
            "nubes_path": nubes_path,
            "text": text,
            "modality": "audio_asr",
            "source": "mls",
        }
        shard_buf.append(json.dumps(out_row, ensure_ascii=False) + "\n")
        n_done += 1
        if len(shard_buf) >= args.shard_size:
            flush()
        if n_done % 1_000_000 == 0:
            print(f"  parsed {n_done:,} (skipped={n_skipped:,})", flush=True)
    flush()

    print(f"[mls] DONE: parsed={n_done:,} skipped={n_skipped:,} "
          f"shards={shard_idx}", flush=True)


if __name__ == "__main__":
    main()
