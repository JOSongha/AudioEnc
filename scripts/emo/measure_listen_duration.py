"""Measure total audio duration of a LISTEN parquet directory.

WAV bytes live in the `audio.bytes` column; duration is read from the WAV header
via soundfile.info (no decode).

Usage:
  python scripts/measure_listen_duration.py [parquet_dir ...]
Defaults to the filtered + original dirs.
"""
import io
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import soundfile as sf

VARIANT_RE = re.compile(r"_\d+[A-Z]$")
strip_variant = lambda s: VARIANT_RE.sub("", s)


def measure(parquet_dir: Path, pattern: str) -> dict:
    paths = sorted(parquet_dir.glob(pattern))
    # audio dedup across question variants: same base id = same audio, count once.
    per_src_sec: dict[str, float] = defaultdict(float)
    per_src_n: dict[str, int] = defaultdict(int)
    seen: set[str] = set()
    total_rows = 0

    for p in paths:
        df = pd.read_parquet(p, columns=["id", "dataset_source", "audio"])
        total_rows += len(df)
        for _id, src, audio in zip(df["id"], df["dataset_source"], df["audio"]):
            base = strip_variant(_id)
            if base in seen:
                continue
            seen.add(base)
            info = sf.info(io.BytesIO(audio["bytes"]))
            dur = info.frames / info.samplerate
            per_src_sec[src] += dur
            per_src_n[src] += 1

    total_sec = sum(per_src_sec.values())
    total_n = sum(per_src_n.values())
    return {
        "dir": str(parquet_dir),
        "pattern": pattern,
        "rows_in_files": total_rows,
        "unique_audios": total_n,
        "total_seconds": total_sec,
        "total_hours": total_sec / 3600,
        "per_source_hours": {s: per_src_sec[s] / 3600 for s in sorted(per_src_sec)},
        "per_source_audios": dict(per_src_n),
    }


def pprint(r: dict) -> None:
    print(f"\n## {r['dir']} ({r['pattern']})")
    print(f"   rows in parquet : {r['rows_in_files']:>7,}")
    print(f"   unique audios   : {r['unique_audios']:>7,}")
    print(f"   total duration  : {r['total_hours']:.2f} h ({r['total_seconds']:.1f} s)")
    print(f"   by source (unique-audio hours):")
    for s, h in sorted(r["per_source_hours"].items(), key=lambda kv: -kv[1]):
        n = r["per_source_audios"][s]
        print(f"      {s:<15} {h:6.2f} h   ({n:,} audios)")


def main() -> None:
    if len(sys.argv) > 1:
        dirs = [Path(a) for a in sys.argv[1:]]
    else:
        dirs = [
            Path("/mnt/tmp/listen_analysis/filtered"),
            Path("/mnt/tmp/listen_analysis/data"),
        ]
    for d in dirs:
        pprint(measure(d, "train-*.parquet"))
        pprint(measure(d, "test-*.parquet"))


if __name__ == "__main__":
    main()
