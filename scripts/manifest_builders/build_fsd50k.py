"""Build FSD50K dev manifest (ontology description -> caption list).

Sources:
    Audio: /mnt/tmp/datasets/env_sound/FSD50K/FSD50K.dev_audio/<fname>.wav
    Ground truth: /mnt/tmp/datasets/env_sound/FSD50K/FSD50K.ground_truth/dev.csv
        cols: fname, labels (comma-joined names like "Electric_guitar,Guitar,..."),
              mids (Freebase ids), split (train|val)
    Ontology: nubes /users/jos/AudioEnc/AudioSet/ontology.json (shared with
        build_audioset.py — FSD50K labels are an AudioSet ontology subset).
        Override with env AUDIOSET_ONTOLOGY=/local/path/ontology.json if needed;
        ddn legacy paths kept as last-resort fallback.

Output (matches v3 sound-captioning spec, Clotho/AudioCaps multi-caption style):
    /mnt/tmp/datasets/manifests/v3/fsd50k_dev_<NNNN>.jsonl
    {"modality": "audio_env_sound", "source": "fsd50k",
     "audio_path": ".../<fname>.wav",
     "captions": ["The electric guitar is a guitar that requires external ...",
                  "A guitar is a stringed musical instrument ...",
                  "Music is sound that has been organized ..."]}

FSD50K label names use underscores ("Electric_guitar") -> map to ontology entry
via mid lookup (preferred) or underscore->space fallback. Each label's
ontology `description` (rich human-curated text, avg ~128 chars) becomes one
caption in the list; the loader picks one per epoch.

Includes BOTH train + val sub-splits of dev.csv (the official FSD50K dev set).
Skip rows with empty labels OR missing audio file.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manifest_builders._nubes_helper import fetch_object  # noqa: E402

AUDIO_ROOT = Path("/mnt/tmp/datasets/env_sound/FSD50K/FSD50K.dev_audio")
GT_CSV = Path("/mnt/tmp/datasets/env_sound/FSD50K/FSD50K.ground_truth/dev.csv")
ONTOLOGY_NUBES_PATH = "users/jos/AudioEnc/AudioSet/ontology.json"
ONTOLOGY_LOCAL_FALLBACKS = [
    Path("/mnt/tmp/datasets/env_sound/AudioSet/ontology.json"),
    Path("/mnt/ddn/users/jos/AudioEnc/log/tmp/datasets/env_sound/AudioSet/ontology.json"),
]
OUT_MANIFEST = Path("/mnt/tmp/datasets/manifests/v3")
OUT_MANIFEST.mkdir(parents=True, exist_ok=True)

SHARD_ROWS = 15000


def load_ontology() -> tuple[dict[str, dict], dict[str, dict]]:
    """Load AudioSet ontology: env override -> nubes -> ddn fallback."""
    override = os.environ.get("AUDIOSET_ONTOLOGY")
    if override and Path(override).is_file():
        with open(override) as f:
            entries = json.load(f)
        return {e["name"]: e for e in entries}, {e["id"]: e for e in entries}
    try:
        entries = json.loads(fetch_object(ONTOLOGY_NUBES_PATH, timeout=60).decode())
        return {e["name"]: e for e in entries}, {e["id"]: e for e in entries}
    except Exception as e:
        for p in ONTOLOGY_LOCAL_FALLBACKS:
            if p.is_file():
                print(f"[fsd50k] nubes ontology fetch failed ({e}), falling back to {p}",
                      flush=True)
                with open(p) as f:
                    entries = json.load(f)
                return {e_["name"]: e_ for e_ in entries}, {e_["id"]: e_ for e_ in entries}
        raise FileNotFoundError(
            "AudioSet ontology.json not reachable from nubes "
            f"({ONTOLOGY_NUBES_PATH}) nor ddn fallbacks ({ONTOLOGY_LOCAL_FALLBACKS}); "
            "set AUDIOSET_ONTOLOGY=/path/to/ontology.json"
        ) from e


def resolve_ontology_entry(raw: str, mid: str, by_name: dict, by_id: dict) -> dict | None:
    """Resolve FSD50K underscore-name to its ontology entry (preserves description).

    Prefer mid lookup (deterministic). Fall back to underscore->space substitution.
    Returns None when neither mid nor name match (rare; caller falls back to raw).
    """
    if mid and mid in by_id:
        return by_id[mid]
    candidate = raw.replace("_", " ").strip()
    if candidate in by_name:
        return by_name[candidate]
    return None


def descriptions_for(raw_labels: list[str], mids: list[str],
                     by_name: dict, by_id: dict) -> list[str]:
    """Return one ontology description per label (rich human-curated text).

    Falls back to the underscore-stripped lowercase name if a label is missing
    from the ontology. Empty descriptions are skipped, then deduped while
    preserving order.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw, mid in zip(raw_labels, mids):
        if not raw or not raw.strip():
            continue
        entry = resolve_ontology_entry(raw, mid, by_name, by_id)
        desc = (entry or {}).get("description", "").strip()
        cap = desc if desc else raw.replace("_", " ").lower().strip()
        if cap in seen:
            continue
        seen.add(cap)
        out.append(cap)
    return out


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

            wav_path = AUDIO_ROOT / f"{fname}.wav"
            if not wav_path.exists() or wav_path.stat().st_size == 0:
                n_skip_no_audio += 1
                continue
            caps = descriptions_for(raw_labels, mids, by_name, by_id)
            if not caps:
                n_skip_no_label += 1
                continue
            rows.append({
                "modality": "audio_env_sound",
                "source": "fsd50k",
                "audio_path": str(wav_path),
                "captions": caps,
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
