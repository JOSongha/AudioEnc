"""Build AudioCaps train+val manifests (extract FLAC bytes from parquet).

Source parquets: /mnt/tmp/datasets/audiocaps/data/{train,validation}-XXXXX-of-NNN.parquet
Schema: audiocap_id, youtube_id, start_time, caption, audio_length, audio: struct<bytes,path>

Strategy:
  - For each split (train, validation):
      - Iterate parquet rows.
      - Extract FLAC bytes once per (youtube_id, start_time) → write to laion_extracted/audiocaps/<youtube_id>_<start_time>.flac
      - Group all captions per (youtube_id, start_time) into a list.
  - Skip rows with empty caption or where FLAC bytes are missing.

Output:
    /mnt/tmp/datasets/laion_extracted/audiocaps/<youtube_id>_<start_time>.flac
    /mnt/tmp/datasets/manifests/v3/audiocaps_train_<NNNN>.jsonl
    /mnt/tmp/datasets/manifests/v3/audiocaps_val_<NNNN>.jsonl

NOTE: TEST split is intentionally skipped (held out for Stage-2 eval).
"""
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

PARQUET_DIR = Path("/mnt/tmp/datasets/audiocaps/data")
AUDIO_OUT_ROOT = Path("/mnt/tmp/datasets/laion_extracted/audiocaps")
MANIFEST_OUT = Path("/mnt/tmp/datasets/manifests/v3")
SHARD_ROWS = 15000

AUDIO_OUT_ROOT.mkdir(parents=True, exist_ok=True)
MANIFEST_OUT.mkdir(parents=True, exist_ok=True)


def process_split(split: str, manifest_prefix: str) -> tuple[int, int]:
    """Returns (n_unique_audios, n_shards)."""
    parquets = sorted(PARQUET_DIR.glob(f"{split}-*.parquet"))
    print(f"[audiocaps:{split}] parquets={len(parquets)}", flush=True)

    # key: (youtube_id, start_time) → list of captions
    captions_per_key: dict[tuple[str, int], list[str]] = defaultdict(list)
    flac_path_per_key: dict[tuple[str, int], Path] = {}

    n_rows = 0
    n_no_caption = 0
    n_no_audio = 0
    n_extract = 0

    for pq_path in parquets:
        t = pq.read_table(
            pq_path,
            columns=["youtube_id", "start_time", "caption", "audio"],
        )
        ytids = t.column("youtube_id").to_pylist()
        starts = t.column("start_time").to_pylist()
        caps = t.column("caption").to_pylist()
        audios = t.column("audio").to_pylist()

        for ytid, start, cap, audio in zip(ytids, starts, caps, audios):
            n_rows += 1
            if not cap or not str(cap).strip():
                n_no_caption += 1
                continue
            if audio is None:
                n_no_audio += 1
                continue
            audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
            if not audio_bytes:
                n_no_audio += 1
                continue

            key = (str(ytid), int(start))
            flac_path = AUDIO_OUT_ROOT / f"{ytid}_{start}.flac"

            if key not in flac_path_per_key:
                # write FLAC bytes once
                if not flac_path.exists() or flac_path.stat().st_size == 0:
                    flac_path.write_bytes(audio_bytes)
                    n_extract += 1
                flac_path_per_key[key] = flac_path

            captions_per_key[key].append(str(cap).strip())

    # build manifest rows: one per unique key
    rows: list[dict] = []
    for key, caps in captions_per_key.items():
        flac_path = flac_path_per_key[key]
        if not flac_path.exists() or flac_path.stat().st_size == 0:
            continue
        rows.append({
            "modality": "audio_env_sound",
            "source": "audiocaps",
            "audio_path": str(flac_path),
            "captions": caps,
        })

    print(
        f"[audiocaps:{split}] rows={n_rows} no_caption={n_no_caption} no_audio={n_no_audio} "
        f"unique_audios={len(rows)} extracted={n_extract}",
        flush=True,
    )

    n_shards = 0
    for i in range(0, len(rows), SHARD_ROWS):
        chunk = rows[i:i + SHARD_ROWS]
        out = MANIFEST_OUT / f"{manifest_prefix}_{n_shards:04d}.jsonl"
        with open(out, "w") as f:
            for r in chunk:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_shards += 1
    print(f"[audiocaps:{split}] wrote {n_shards} shards", flush=True)
    if rows:
        print(f"[audiocaps:{split}] sample: {json.dumps(rows[0], ensure_ascii=False)}", flush=True)
    return len(rows), n_shards


def main():
    train_n, train_s = process_split("train", "audiocaps_train")
    val_n, val_s = process_split("validation", "audiocaps_val")
    print(f"[audiocaps] DONE train={train_n} ({train_s} shards), val={val_n} ({val_s} shards)", flush=True)


if __name__ == "__main__":
    main()
