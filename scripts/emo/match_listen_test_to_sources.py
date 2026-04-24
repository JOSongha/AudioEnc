"""Map each LISTEN-test ID to a file in its raw source corpus directory.

Outputs a JSON dict:
  { source_corpus: { listen_test_id: absolute_source_path, ... }, ... }

and an exclusion-manifest JSON:
  { source_corpus: [list of raw-corpus relative paths to hold out], ... }

The exclusion manifest is what downstream `build_training_manifest.py` should
consume: for each source corpus, enumerate raw files and drop anything listed.

Mapping rules per source (derived from observing LISTEN IDs in the parquet):
  IEMOCAP     IEMOCAP_Session{N}_{wav_stem}       -> Session{N}/sentences/wav/{dialog}/{wav_stem}.wav
  MELD        MELD_{split}_{dia}_{utt}            -> {split}_splits/dia{dia}_utt{utt}.mp4 (wav after extraction)
  CREMA-D     CREMA-D_{filestem}                  -> AudioWAV/{filestem}.wav
  TESS        TESS_{stem}                         -> {stem}/{stem}.wav
  ESD         Emotion-Speech_{speaker}_{utt}      -> {speaker}/<emotion>/{speaker}_{utt}.wav
  MUStARD     MUStARD_{showcode}_{n}_{id}_u       -> key in final_mustard++.csv -> video file
  MOSEI       MOSEI_Test_modified_{vid}_{st}_{en} -> {vid} segment [{st},{en}]  (segment cut needed)
  MUStARD    (ditto)
  RAVDESS/SAVEE — LISTEN uses a sequential integer. Mapping needs a hash match pass.
  OMG / PODCAST — can't recover source-split from ID alone (see audit §3.3, §3.5).

This script only produces a BEST-EFFORT mapping. Entries that cannot be resolved
are returned in the `unmapped` dict.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

LISTEN_PARQUET = Path("/mnt/tmp/listen_analysis/data/test-00000-of-00001.parquet")
RAW = Path("/mnt/tmp/datasets/emotion_raw")
OUT_DIR = Path("/mnt/tmp/listen_analysis/exclude_manifests")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANT_RE = re.compile(r"_\d+[A-Z]$")
strip_variant = lambda s: VARIANT_RE.sub("", s)


def map_iemocap(ids: list[str]) -> tuple[dict, list]:
    base = RAW / "IEMOCAP" / "data"   # user-mounted structure may differ
    mapped, missing = {}, []
    if not base.exists():
        return mapped, ids
    index = {p.stem: p for p in base.rglob("*.wav")}
    for lid in ids:
        stem = lid.split("_", 2)[2] if lid.count("_") >= 2 else None
        if stem and stem in index:
            mapped[lid] = str(index[stem])
        else:
            missing.append(lid)
    return mapped, missing


def map_meld(ids: list[str]) -> tuple[dict, list]:
    base = RAW / "MELD" / "MELD.Raw"
    mapped, missing = {}, []
    if not base.exists():
        return mapped, ids
    for lid in ids:
        m = re.match(r"MELD_(train|dev|test)_(\d+)_(\d+)$", lid)
        if not m:
            missing.append(lid); continue
        split, dia, utt = m.groups()
        mp4 = base / f"{split}_splits" / f"dia{dia}_utt{utt}.mp4"
        if mp4.exists():
            mapped[lid] = str(mp4)
        else:
            missing.append(lid)
    return mapped, missing


def map_crema_d(ids: list[str]) -> tuple[dict, list]:
    base = RAW / "CREMA-D" / "AudioWAV"
    mapped, missing = {}, []
    if not base.exists():
        return mapped, ids
    for lid in ids:
        stem = lid[len("CREMA-D_"):]
        wav = base / f"{stem}.wav"
        if wav.exists():
            mapped[lid] = str(wav)
        else:
            missing.append(lid)
    return mapped, missing


def map_tess(ids: list[str]) -> tuple[dict, list]:
    base = RAW / "TESS"
    mapped, missing = {}, []
    if not base.exists():
        return mapped, ids
    index = {p.stem: p for p in base.rglob("*.wav")}
    for lid in ids:
        stem = lid[len("TESS_"):]
        if stem in index:
            mapped[lid] = str(index[stem])
        else:
            missing.append(lid)
    return mapped, missing


def map_esd(ids: list[str]) -> tuple[dict, list]:
    base = RAW / "ESD"
    mapped, missing = {}, []
    if not base.exists():
        return mapped, ids
    # ESD layout: 0011/Angry/0011_000028.wav (speaker/emotion/speaker_utt.wav)
    index = {p.stem: p for p in base.rglob("*.wav")}
    for lid in ids:
        stem = lid[len("Emotion-Speech_"):]  # -> "0011_000028"
        if stem in index:
            mapped[lid] = str(index[stem])
        else:
            missing.append(lid)
    return mapped, missing


def map_ravdess(ids: list[str]) -> tuple[dict, list]:
    # LISTEN uses a sequential `RAVDESS_train_{idx}_{emo}_{intensity}` id.
    # Without a published index map, we cannot resolve ids to RAVDESS files
    # deterministically. An exact content-hash pass is needed (load LISTEN
    # audio bytes, hash, match against Actor_{N}/*.wav).
    return {}, list(ids)


def map_savee(ids: list[str]) -> tuple[dict, list]:
    return {}, list(ids)


def map_podcast(ids: list[str]) -> tuple[dict, list]:
    # MSP-Podcast not downloadable w/o EULA yet; id does not expose original split.
    return {}, list(ids)


def map_omg(ids: list[str]) -> tuple[dict, list]:
    return {}, list(ids)


def map_mosei(ids: list[str]) -> tuple[dict, list]:
    return {}, list(ids)


def map_mustard(ids: list[str]) -> tuple[dict, list]:
    base = RAW / "MUStARD_Plus_Plus" / "videos"
    mapped, missing = {}, []
    if not base.exists():
        return mapped, ids
    # Flat search: LISTEN id like "MUStARD_EMB_1_105_u" → a {showcode}_{id}_{n}_u.mp4
    # We search by filename stem suffix "_u" for utterance, "_c" for context.
    index = {p.stem: p for p in base.rglob("*.mp4")}
    for lid in ids:
        stem = lid[len("MUStARD_"):]
        if stem in index:
            mapped[lid] = str(index[stem])
        else:
            missing.append(lid)
    return mapped, missing


MAPPERS = {
    "IEMOCAP": map_iemocap,
    "MELD": map_meld,
    "CREMA-D": map_crema_d,
    "TESS": map_tess,
    "Emotion-Speech": map_esd,
    "RAVDESS": map_ravdess,
    "SAVEE": map_savee,
    "PODCAST": map_podcast,
    "OMG": map_omg,
    "MOSEI": map_mosei,
    "MUStARD": map_mustard,
}


def main() -> None:
    test = pd.read_parquet(LISTEN_PARQUET, columns=["id", "dataset_source"])
    # Unique base IDs per source (strip question-variant suffix)
    by_src: dict[str, list[str]] = defaultdict(list)
    for _id, src in zip(test["id"], test["dataset_source"]):
        by_src[src].append(strip_variant(_id))
    by_src = {s: sorted(set(ids)) for s, ids in by_src.items()}

    summary = {}
    full_map = {}
    unmapped = {}
    for src, ids in by_src.items():
        fn = MAPPERS.get(src)
        if fn is None:
            print(f"  no mapper for {src}")
            summary[src] = {"total": len(ids), "mapped": 0, "unmapped": len(ids)}
            unmapped[src] = ids
            continue
        m, u = fn(ids)
        full_map[src] = m
        unmapped[src] = u
        summary[src] = {"total": len(ids), "mapped": len(m), "unmapped": len(u)}
        print(f"  {src:<15} total={len(ids):4d} mapped={len(m):4d} unmapped={len(u):4d}")

    (OUT_DIR / "listen_test_source_mapping.json").write_text(
        json.dumps({"summary": summary, "mapped": full_map, "unmapped": unmapped}, indent=2)
    )
    print(f"\nwrote {OUT_DIR / 'listen_test_source_mapping.json'}")


if __name__ == "__main__":
    main()
