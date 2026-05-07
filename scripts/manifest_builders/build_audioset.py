"""Build AudioSet bal_train manifest (ontology description -> caption list).

Sources:
    Audio: /mnt/tmp/datasets/env_sound/AudioSet/audio/<video_id>.flac
        (extracted earlier from data/bal_train/*.parquet by
        scripts/env_sound/prepare_manifest.py; here we only read labels.)
    Labels: /mnt/tmp/datasets/env_sound/AudioSet/data/bal_train/*.parquet
        cols: video_id, audio (ignored), labels (Freebase ids), human_labels (text)
    Ontology: /mnt/tmp/datasets/env_sound/AudioSet/ontology.json
        (also available under /mnt/ddn/users/jos/AudioEnc/log/tmp/...; both 632 entries.)

Output (matches v3 sound-captioning spec, Clotho/AudioCaps multi-caption style):
    /mnt/tmp/datasets/manifests/v3/audioset_bal_train_<NNNN>.jsonl
    {"modality": "audio_env_sound", "source": "audioset",
     "audio_path": ".../<vid>.flac",
     "captions": ["The human voice consists of sound made ...",
                  "A dog is a furry mammal ..."]}

Each clip's `human_labels` is mapped to its ontology entry's `description`
(rich human-curated text, avg ~128 chars). Loader picks one caption per epoch
(omni_dataset.py's caption pool sampling). Labels with no ontology hit fall
back to their lowercase name as a one-word caption.

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


def descriptions_for(human_labels: list[str], onto: dict) -> list[str]:
    """Return one ontology description per label (rich human-curated text).

    Falls back to the lowercase label name if the label is missing from the
    ontology (rare: AudioSet's eval-set human_labels match ontology by `name`).
    Empty descriptions are skipped, then deduped while preserving order.
    """
    out: list[str] = []
    seen: set[str] = set()
    for x in human_labels:
        name = str(x).strip()
        if not name:
            continue
        entry = onto.get(name)
        desc = (entry or {}).get("description", "").strip()
        cap = desc if desc else name.lower()
        if cap in seen:
            continue
        seen.add(cap)
        out.append(cap)
    return out


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
        caps = descriptions_for([str(x) for x in human_labels], onto)
        if not caps:
            n_skip_no_label += 1
            continue
        rows.append({
            "modality": "audio_env_sound",
            "source": "audioset",
            "audio_path": str(flac_path),
            "captions": caps,
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
