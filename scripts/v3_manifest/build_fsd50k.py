"""Build FSD50K dev manifest (ontology -> synthesized caption).

Sources:
    Audio: /mnt/tmp/datasets/env_sound/FSD50K/FSD50K.dev_audio/<fname>.wav
    Ground truth: /mnt/tmp/datasets/env_sound/FSD50K/FSD50K.ground_truth/dev.csv
        cols: fname, labels (comma-joined names like "Electric_guitar,Guitar,..."),
              mids (Freebase ids), split (train|val)
    Ontology: /mnt/tmp/datasets/env_sound/AudioSet/ontology.json
        FSD50K labels are an AudioSet ontology subset -> reuse same ontology.

Output (matches v3 sound-captioning spec exactly):
    /mnt/tmp/datasets/manifests/v3/fsd50k_dev_<NNNN>.jsonl
    {"modality": "audio_env_sound", "source": "fsd50k",
     "audio_path": ".../<fname>.wav", "captions": ["sound of A, B, and C"]}

FSD50K label names use underscores ("Electric_guitar") -> map to ontology display
name ("Electric guitar") via mid lookup (preferred, deterministic), with
underscore->space fallback.

Caption synthesis (lowercased, comma-separated, "and" before last when >=3):
    1 label : "sound of music"
    2 labels: "sound of guitar, music"
    3+      : "sound of electric guitar, guitar, and music"

Includes BOTH train + val sub-splits of dev.csv (the official FSD50K dev set).
Skip rows with empty labels OR missing audio file.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

AUDIO_ROOT = Path("/mnt/tmp/datasets/env_sound/FSD50K/FSD50K.dev_audio")
GT_CSV = Path("/mnt/tmp/datasets/env_sound/FSD50K/FSD50K.ground_truth/dev.csv")
ONTOLOGY_CANDIDATES = [
    Path("/mnt/tmp/datasets/env_sound/AudioSet/ontology.json"),
    Path("/mnt/ddn/users/jos/AudioEnc/log/tmp/datasets/env_sound/AudioSet/ontology.json"),
]
OUT_MANIFEST = Path("/mnt/tmp/datasets/manifests/v3")
OUT_MANIFEST.mkdir(parents=True, exist_ok=True)

SHARD_ROWS = 15000


def load_ontology() -> tuple[dict[str, dict], dict[str, dict]]:
    for p in ONTOLOGY_CANDIDATES:
        if p.exists():
            with open(p) as f:
                entries = json.load(f)
            return {e["name"]: e for e in entries}, {e["id"]: e for e in entries}
    raise FileNotFoundError(
        f"AudioSet ontology.json not found in any of: {ONTOLOGY_CANDIDATES}"
    )


def normalize_fsd_label(raw: str, mid: str, by_name: dict, by_id: dict) -> str:
    """Resolve FSD50K underscore-name to ontology display name.

    Prefer mid lookup (deterministic). Fall back to underscore->space substitution.
    """
    if mid and mid in by_id:
        return by_id[mid]["name"]
    candidate = raw.replace("_", " ").strip()
    if candidate in by_name:
        return by_name[candidate]["name"]
    return candidate


def synthesize_caption(label_names: list[str]) -> str | None:
    labels = [x for x in label_names if x and x.strip()]
    if not labels:
        return None
    lower = [x.lower() for x in labels]
    if len(lower) == 1:
        return f"sound of {lower[0]}"
    if len(lower) == 2:
        return f"sound of {lower[0]}, {lower[1]}"
    return f"sound of {', '.join(lower[:-1])}, and {lower[-1]}"


def main() -> None:
    by_name, by_id = load_ontology()
    print(f"[fsd50k] ontology: {len(by_name)} entries", flush=True)

    rows: list[dict] = []
    n_seen = 0
    n_skip_no_label = 0
    n_skip_no_audio = 0

    with open(GT_CSV) as f:
        reader = csv.DictReader(f)
        for r in reader:
            n_seen += 1
            fname = r["fname"]
            raw_labels = [x for x in r["labels"].split(",") if x.strip()]
            mids = [x for x in r["mids"].split(",") if x.strip()]
            if not raw_labels:
                n_skip_no_label += 1
                continue
            if len(mids) < len(raw_labels):
                mids = mids + [""] * (len(raw_labels) - len(mids))
            label_names = [
                normalize_fsd_label(rl, mid, by_name, by_id)
                for rl, mid in zip(raw_labels, mids)
            ]

            wav_path = AUDIO_ROOT / f"{fname}.wav"
            if not wav_path.exists() or wav_path.stat().st_size == 0:
                n_skip_no_audio += 1
                continue
            cap = synthesize_caption(label_names)
            if cap is None:
                n_skip_no_label += 1
                continue
            rows.append({
                "modality": "audio_env_sound",
                "source": "fsd50k",
                "audio_path": str(wav_path),
                "captions": [cap],
            })

    print(
        f"[fsd50k] seen={n_seen} kept={len(rows)} skip_no_label={n_skip_no_label} "
        f"skip_no_audio={n_skip_no_audio}",
        flush=True,
    )

    # clear existing fsd50k shards so we don't leave stale ones around
    for old in OUT_MANIFEST.glob("fsd50k_dev_*.jsonl"):
        old.unlink()

    n_shards = 0
    for i in range(0, len(rows), SHARD_ROWS):
        chunk = rows[i:i + SHARD_ROWS]
        out = OUT_MANIFEST / f"fsd50k_dev_{n_shards:04d}.jsonl"
        with open(out, "w") as f:
            for row in chunk:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_shards += 1
    print(f"[fsd50k] wrote {n_shards} shards -> {OUT_MANIFEST}", flush=True)
    if rows:
        print(f"[fsd50k] sample: {json.dumps(rows[0], ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
