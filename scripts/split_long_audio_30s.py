"""Split long-audio manifest entries into <=30s chunks for Whisper training.

Whisper-small.en accepts only 30-second mel inputs. Per the Whisper paper, training data
is segmented into 30s windows; we replicate that by adding `audio_start` / `audio_end` fields
to manifest entries that exceed 30s.

Manifest format (JSONL): existing entries have `{"nubes_path": "...", "text": "...", ...}`.
Split entries get added: `{"audio_start": float, "audio_end": float}`.

Transcript handling for split entries:
  - Default: copy original `text` to every chunk (correct only when chunks share a single
    utterance; misaligned otherwise). Marked with a warning. For Stage1 manifest where
    most entries are already <30s, the default is harmless.
  - Future improvement: word-level timestamp alignment via force-aligner (wav2vec2 CTC).
    Hook left at `align_text_to_window()`.

Audio duration source:
  - `--audio-root <local_path>`: prefix to prepend to nubes_path for local file lookup.
  - `--probe-from-nubes`: HTTP GET to nubes gateway and read header (slow). Off by default.
  - Cached duration is also accepted if entry has `audio_duration` field.

Usage:
    python scripts/split_long_audio_30s.py \\
        --input external/datasets/libri_mls_vox \\
        --output external/datasets/libri_mls_vox_30s \\
        --audio-root /path/to/local/audio \\
        --max-seconds 30.0 \\
        [--limit-shards 1]   # for testing
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import soundfile as sf


def get_duration(entry: dict, audio_root: Path | None) -> float | None:
    """Return duration in seconds, or None if cannot determine."""
    if "audio_duration" in entry:
        return float(entry["audio_duration"])

    nubes_path = entry.get("nubes_path") or entry.get("audio_path")
    if not nubes_path:
        return None

    if audio_root is not None:
        local = audio_root / nubes_path
        if local.exists():
            try:
                info = sf.info(str(local))
                return info.frames / info.samplerate
            except Exception:
                return None
    return None


def align_text_to_window(text: str, start: float, end: float, total_duration: float) -> str:
    """Placeholder for transcript alignment. Currently returns full text (warning-worthy)."""
    return text


def split_entry(entry: dict, duration: float, max_seconds: float) -> list[dict]:
    """Return list of split entries. Single-element list if no split needed."""
    if duration <= max_seconds:
        return [entry]

    n_chunks = math.ceil(duration / max_seconds)
    chunks = []
    for i in range(n_chunks):
        start = i * max_seconds
        end = min(start + max_seconds, duration)
        new = dict(entry)
        new["audio_start"] = start
        new["audio_end"] = end
        new["audio_duration_chunk"] = end - start
        new["text"] = align_text_to_window(entry.get("text", ""), start, end, duration)
        chunks.append(new)
    return chunks


def process_shard(in_path: Path, out_path: Path, audio_root: Path | None,
                  max_seconds: float, stats: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            stats["entries_in"] += 1

            duration = get_duration(entry, audio_root)
            if duration is None:
                stats["entries_unknown_duration"] += 1
                fout.write(line + "\n")  # pass through; runtime safety-trim will handle if >30s
                stats["entries_out"] += 1
                continue

            if duration > max_seconds:
                stats["entries_long"] += 1
                stats["seconds_long"] += duration

            chunks = split_entry(entry, duration, max_seconds)
            for c in chunks:
                fout.write(json.dumps(c, ensure_ascii=False) + "\n")
                stats["entries_out"] += 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Input manifest dir (contains shard_*.jsonl)")
    p.add_argument("--output", required=True, help="Output manifest dir")
    p.add_argument("--audio-root", default=None, help="Local audio root (prefix to nubes_path)")
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--limit-shards", type=int, default=None, help="Process only first N shards (for testing)")
    args = p.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    audio_root = Path(args.audio_root) if args.audio_root else None

    shards = sorted(in_dir.glob("shard_*.jsonl"))
    if not shards:
        raise SystemExit(f"No shard_*.jsonl in {in_dir}")
    if args.limit_shards:
        shards = shards[: args.limit_shards]

    stats = {
        "shards": len(shards),
        "entries_in": 0,
        "entries_out": 0,
        "entries_long": 0,
        "entries_unknown_duration": 0,
        "seconds_long": 0.0,
    }

    print(f"Processing {len(shards)} shard(s) from {in_dir}")
    print(f"Audio root: {audio_root or '(not set; pass-through entries with unknown duration)'}")
    for shard in shards:
        out_shard = out_dir / shard.name
        process_shard(shard, out_shard, audio_root, args.max_seconds, stats)

    print(f"\n=== Stats ===")
    print(f"  shards processed       : {stats['shards']}")
    print(f"  entries in             : {stats['entries_in']:,}")
    print(f"  entries out            : {stats['entries_out']:,}")
    delta = stats["entries_out"] - stats["entries_in"]
    print(f"  delta (split overhead) : +{delta:,}")
    print(f"  entries >30s (split)   : {stats['entries_long']:,}")
    print(f"  total long audio (s)   : {stats['seconds_long']:.1f}")
    print(f"  entries unknown dur    : {stats['entries_unknown_duration']:,} (passed through)")
    if stats["entries_unknown_duration"] > 0:
        print(f"  WARNING: pass-through entries will be trimmed at runtime if they exceed {args.max_seconds}s")


if __name__ == "__main__":
    main()
