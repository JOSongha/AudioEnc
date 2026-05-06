"""Build LAION Audiostock manifest by joining the LAION csv with already-downloaded mp3s.

Source CSV : /mnt/tmp/datasets/laion_csvs/Audiostock.csv  (cols: ,url,caption1)
Source mp3 : /mnt/tmp/datasets/laion_audiostock/audio/<id>.mp3   (~9k clips downloaded)
URL→ID regex (matches `audiostock\.\w+/audio/<id>/play`).

Output: /mnt/tmp/datasets/manifests/v3/laion_audiostock_<NNNN>.jsonl
        15000 rows / shard.

Spec row:
    {"modality":"audio_env_sound", "source":"laion_audiostock",
     "audio_path":"/mnt/tmp/datasets/laion_audiostock/audio/<id>.mp3",
     "captions":["<caption>"]}

Drops rows where mp3 is missing (download failed) — count is reported.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

CSV_PATH = Path("/mnt/tmp/datasets/laion_csvs/Audiostock.csv")
AUDIO_ROOT = Path("/mnt/tmp/datasets/laion_audiostock/audio")
OUT_DIR = Path("/mnt/tmp/datasets/manifests/v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ID_RE = re.compile(r"audiostock\.\w+/audio/(\d+)/play")
SHARD_ROWS = 15000


def main() -> None:
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    print(f"[audiostock] csv rows: {len(rows)}", flush=True)

    n_bad_url = 0
    n_missing = 0
    n_empty_cap = 0
    n_dup = 0
    seen_ids: set[str] = set()
    out_rows: list[dict] = []

    for r in rows:
        m = ID_RE.search(r.get("url", "") or "")
        if not m:
            n_bad_url += 1
            continue
        aid = m.group(1)
        if aid in seen_ids:
            n_dup += 1
            continue
        mp3 = AUDIO_ROOT / f"{aid}.mp3"
        if not mp3.exists() or mp3.stat().st_size <= 1024:
            n_missing += 1
            continue
        cap = (r.get("caption1") or "").strip()
        if not cap:
            n_empty_cap += 1
            continue
        seen_ids.add(aid)
        out_rows.append({
            "modality": "audio_env_sound",
            "source": "laion_audiostock",
            "audio_path": str(mp3),
            "captions": [cap],
        })

    print(
        f"[audiostock] kept={len(out_rows)} "
        f"missing_mp3={n_missing} bad_url={n_bad_url} "
        f"empty_caption={n_empty_cap} dup={n_dup}",
        flush=True,
    )

    n_shards = 0
    for i in range(0, len(out_rows), SHARD_ROWS):
        chunk = out_rows[i:i + SHARD_ROWS]
        out_path = OUT_DIR / f"laion_audiostock_{n_shards:04d}.jsonl"
        with open(out_path, "w") as f:
            for row in chunk:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[audiostock] wrote {out_path} ({len(chunk)} rows)", flush=True)
        n_shards += 1

    print(f"[audiostock] DONE: {len(out_rows)} clips → {n_shards} shards", flush=True)


if __name__ == "__main__":
    main()
