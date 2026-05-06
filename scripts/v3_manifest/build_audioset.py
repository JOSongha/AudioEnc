"""Build AudioSet bal_train manifest (ontology -> synthesized caption).

Sources:
    Audio: /mnt/tmp/datasets/env_sound/AudioSet/audio/<video_id>.flac
        (extracted earlier from data/bal_train/*.parquet by
        scripts/env_sound/prepare_manifest.py; here we only read labels.)
    Labels: /mnt/tmp/datasets/env_sound/AudioSet/data/bal_train/*.parquet
        cols: video_id, audio (ignored), labels (Freebase ids), human_labels (text)
    Ontology: /mnt/tmp/datasets/env_sound/AudioSet/ontology.json
        (also available under /mnt/ddn/users/jos/AudioEnc/log/tmp/...; both 632 entries.)

Output (matches v3 sound-captioning spec exactly):
    /mnt/tmp/datasets/manifests/v3/audioset_bal_train_<NNNN>.jsonl
    {"modality": "audio_env_sound", "source": "audioset",
     "audio_path": ".../<vid>.flac", "captions": ["sound of dog, bark"]}

Caption synthesis from `human_labels` (lowercased, comma-separated, "and" before
last when there are >=3 labels):
    1 label : "sound of dog"
    2 labels: "sound of dog, bark"
    3+      : "sound of dog, bark, and music"

Skip rows with empty human_labels OR missing audio file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

AUDIO_ROOT = Path("/mnt/tmp/datasets/env_sound/AudioSet/audio")
PARQUET_DIR = Path("/mnt/tmp/datasets/env_sound/AudioSet/data/bal_train")
ONTOLOGY_CANDIDATES = [
    Path("/mnt/tmp/datasets/env_sound/AudioSet/ontology.json"),
    Path("/mnt/ddn/users/jos/AudioEnc/log/tmp/datasets/env_sound/AudioSet/ontology.json"),
]
OUT_MANIFEST = Path("/mnt/tmp/datasets/manifests/v3")
OUT_MANIFEST.mkdir(parents=True, exist_ok=True)

SHARD_ROWS = 15000


def load_ontology() -> dict[str, dict]:
    for p in ONTOLOGY_CANDIDATES:
        if p.exists():
            with open(p) as f:
                entries = json.load(f)
            return {e["name"]: e for e in entries}
    raise FileNotFoundError(
        f"AudioSet ontology.json not found in any of: {ONTOLOGY_CANDIDATES}"
    )


def synthesize_caption(human_labels: list[str]) -> str | None:
    """Return single 'sound of ...' caption per v3 spec, or None if no labels."""
    labels = [str(x).strip() for x in human_labels if str(x).strip()]
    if not labels:
        return None
    lower = [x.lower() for x in labels]
    if len(lower) == 1:
        return f"sound of {lower[0]}"
    if len(lower) == 2:
        return f"sound of {lower[0]}, {lower[1]}"
    return f"sound of {', '.join(lower[:-1])}, and {lower[-1]}"


def iter_parquet_rows():
    parquets = sorted(PARQUET_DIR.glob("*.parquet"))
    for pq_path in parquets:
        t = pq.read_table(pq_path, columns=["video_id", "human_labels"])
        vids = t.column("video_id").to_pylist()
        hls = t.column("human_labels").to_pylist()
        for vid, hl in zip(vids, hls):
            yield vid, hl


def main() -> None:
    onto = load_ontology()
    print(f"[audioset] ontology: {len(onto)} entries", flush=True)

    rows: list[dict] = []
    n_seen = 0
    n_skip_no_label = 0
    n_skip_no_audio = 0

    for vid, human_labels in iter_parquet_rows():
        n_seen += 1
        if not human_labels:
            n_skip_no_label += 1
            continue
        flac_path = AUDIO_ROOT / f"{vid}.flac"
        if not flac_path.exists() or flac_path.stat().st_size == 0:
            n_skip_no_audio += 1
            continue
        cap = synthesize_caption([str(x) for x in human_labels])
        if cap is None:
            n_skip_no_label += 1
            continue
        rows.append({
            "modality": "audio_env_sound",
            "source": "audioset",
            "audio_path": str(flac_path),
            "captions": [cap],
        })

    print(
        f"[audioset] seen={n_seen} kept={len(rows)} skip_no_label={n_skip_no_label} "
        f"skip_no_audio={n_skip_no_audio}",
        flush=True,
    )

    # clear existing audioset shards so we don't leave stale ones around
    for old in OUT_MANIFEST.glob("audioset_bal_train_*.jsonl"):
        old.unlink()

    n_shards = 0
    for i in range(0, len(rows), SHARD_ROWS):
        chunk = rows[i:i + SHARD_ROWS]
        out = OUT_MANIFEST / f"audioset_bal_train_{n_shards:04d}.jsonl"
        with open(out, "w") as f:
            for r in chunk:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_shards += 1
    print(f"[audioset] wrote {n_shards} shards -> {OUT_MANIFEST}", flush=True)
    if rows:
        print(f"[audioset] sample: {json.dumps(rows[0], ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
