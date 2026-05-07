"""Build Clotho dev+val manifests (one row per .wav, 5 captions).

Source captions:
    /mnt/tmp/datasets/env_sound/Clotho/captions_development.csv  (3839 rows)
    /mnt/tmp/datasets/env_sound/Clotho/captions_validation.csv   (1045 rows)
Source audio:
    /mnt/tmp/datasets/env_sound/Clotho/{development,validation}/<file_name>

CSV columns: file_name, caption_1..5.

Skip rows with missing audio or any empty caption (kept = all 5 non-empty).
"""
import csv
import json
from pathlib import Path

ROOT = Path("/mnt/tmp/datasets/env_sound/Clotho")
MANIFEST_OUT = Path("/mnt/tmp/datasets/manifests/v3")
SHARD_ROWS = 15000

MANIFEST_OUT.mkdir(parents=True, exist_ok=True)


def process_split(csv_name: str, audio_dir_name: str, manifest_prefix: str) -> tuple[int, int]:
    csv_path = ROOT / csv_name
    audio_dir = ROOT / audio_dir_name

    rows: list[dict] = []
    n_seen = 0
    n_no_audio = 0
    n_empty_cap = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            n_seen += 1
            fname = r.get("file_name", "").strip()
            if not fname:
                n_no_audio += 1
                continue
            audio_path = audio_dir / fname
            if not audio_path.exists() or audio_path.stat().st_size == 0:
                n_no_audio += 1
                continue
            caps = []
            for k in ("caption_1", "caption_2", "caption_3", "caption_4", "caption_5"):
                v = r.get(k, "").strip()
                if v:
                    caps.append(v)
            if not caps:
                n_empty_cap += 1
                continue
            rows.append({
                "modality": "audio_env_sound",
                "source": "clotho",
                "audio_path": str(audio_path),
                "captions": caps,
            })

    print(
        f"[clotho:{audio_dir_name}] seen={n_seen} kept={len(rows)} "
        f"no_audio={n_no_audio} empty_cap={n_empty_cap}",
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
    print(f"[clotho:{audio_dir_name}] wrote {n_shards} shards", flush=True)
    if rows:
        print(f"[clotho:{audio_dir_name}] sample: {json.dumps(rows[0], ensure_ascii=False)}", flush=True)
    return len(rows), n_shards


def main():
    dev_n, dev_s = process_split("captions_development.csv", "development", "clotho_dev")
    val_n, val_s = process_split("captions_validation.csv", "validation", "clotho_val")
    print(f"[clotho] DONE dev={dev_n} ({dev_s} shards), val={val_n} ({val_s} shards)", flush=True)


if __name__ == "__main__":
    main()
