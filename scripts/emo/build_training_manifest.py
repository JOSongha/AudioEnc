"""Enumerate available raw emotion corpora, exclude LISTEN-test items, emit a
training manifest JSONL + per-source exclusion report.

For each supported source:
  1. list every audio file on disk
  2. drop files that appear in the LISTEN-test-source mapping (any source)
  3. emit one manifest row per remaining file: {path, source, source_relpath}

Sources supported today (anything requiring EULA / GDrive is skipped until the
raw corpus is present):
  - RAVDESS  (uses ravdess_hash_match.json for exclusions)
  - MELD     (train/dev/test extracted WAVs; exclusion via per-id mapping)

Output:
  /mnt/tmp/listen_analysis/train_manifest/train_manifest.jsonl
  /mnt/tmp/listen_analysis/train_manifest/filter_report.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

RAW = Path("/mnt/tmp/datasets/emotion_raw")
MAP_DIR = Path("/mnt/tmp/listen_analysis/exclude_manifests")
OUT_DIR = Path("/mnt/tmp/listen_analysis/train_manifest")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_exclusions() -> dict[str, set[str]]:
    """Load excluded raw-corpus paths per source (translated to manifest-side paths)."""
    paths: dict[str, set[str]] = {}
    f = MAP_DIR / "listen_test_source_mapping.json"
    if f.exists():
        data = json.loads(f.read_text())
        for src, m in data.get("mapped", {}).items():
            for lid, raw_path in m.items():
                # MELD mapping stores .mp4 paths in MELD.Raw/{dir}/... — translate to
                # audio/{split}/*.wav emitted by extract_meld_audio.sh.
                if src == "MELD" and raw_path.endswith(".mp4"):
                    p = Path(raw_path)
                    # dir name → split name: train_splits / dev_splits_complete / output_repeated_splits_test
                    split = {
                        "train_splits": "train",
                        "dev_splits_complete": "dev",
                        "output_repeated_splits_test": "test",
                    }.get(p.parent.name)
                    if split is None:
                        continue
                    wav = RAW / "MELD" / "audio" / split / (p.stem + ".wav")
                    paths.setdefault(src, set()).add(str(wav))
                else:
                    paths.setdefault(src, set()).add(raw_path)
    f = MAP_DIR / "ravdess_hash_match.json"
    if f.exists():
        data = json.loads(f.read_text())
        paths.setdefault("RAVDESS", set()).update(data.get("mapped", {}).values())
    return paths


def iter_source_files(src: str) -> Iterator[Path]:
    """Enumerate TRAIN-side audio files per corpus.

    Held-out splits (defined in stage2_eval_plan.md §8) are EXCLUDED here so the
    training manifest can never accidentally contaminate Tier-2 eval:
      MELD       : train + dev only   (test held out)
      DailyTalk  : first 95% dialogues only (last 5% held out)
      EmoV-DB    : Bea / Josh / Sam only   (Jenie held out as speaker-out eval)
      RAVDESS    : Actor_01-20 only   (Actor_21-24 held out)
      IEMOCAP    : Session1-4 only    (Session5 held out)  — re-applied here for defence-in-depth
    """
    if src == "RAVDESS":
        # RAVDESS actors 21-24 held out
        for p in sorted((RAW / "RAVDESS").rglob("*.wav")):
            actor = p.parent.name  # "Actor_NN"
            if actor.startswith("Actor_"):
                try:
                    n = int(actor.split("_")[1])
                    if n >= 21:  # 21-24 held out
                        continue
                except ValueError:
                    pass
            yield p
    elif src == "MELD":
        # MELD-test held out
        for split in ("train", "dev"):
            d = RAW / "MELD" / "audio" / split
            if d.exists():
                yield from sorted(d.glob("*.wav"))
    elif src == "CREMA-D":
        d = RAW / "CREMA-D" / "AudioWAV"
        if d.exists():
            yield from sorted(d.glob("*.wav"))
    elif src == "Emotion-Speech":
        d = RAW / "ESD"
        if d.exists():
            yield from sorted(d.rglob("*.wav"))
    elif src == "TESS":
        d = RAW / "TESS"
        if d.exists():
            yield from sorted(d.rglob("*.wav"))
    elif src == "IEMOCAP":
        # Session 5 held out for speaker-independent SER eval
        d = RAW / "IEMOCAP"
        if d.exists():
            for p in sorted(d.rglob("*.wav")):
                if "Session5" in str(p):
                    continue
                yield p
    elif src == "MUStARD":
        d = RAW / "MUStARD_Plus_Plus" / "videos"
        if d.exists():
            yield from sorted(d.rglob("*.mp4"))
    elif src == "DailyTalk":
        # Last 5% of DIALOGUES held out. Dialog id is encoded in filename as
        # "<utt>_<speaker>_d<dialog>.wav" (the `d<N>` suffix) and mirrored as
        # the parent directory name `data/<dialog>/…`. Earlier draft matched
        # the leading utterance index by mistake (yielded only 20 "IDs").
        import re
        dialog_re = re.compile(r"_d(\d+)$")
        d = RAW / "DailyTalk" / "dailytalk" / "data"
        if not d.exists():
            d = RAW / "DailyTalk"
        if d.exists():
            ids: set[int] = set()
            for p in d.rglob("*.wav"):
                m = dialog_re.match("_d" + p.stem.split("_d", 1)[-1]) if "_d" in p.stem else None
                if m:
                    ids.add(int(m.group(1)))
                    continue
                # fall back to parent-dir name
                try:
                    ids.add(int(p.parent.name))
                except ValueError:
                    pass
            ids_sorted = sorted(ids)
            cutoff = ids_sorted[int(len(ids_sorted) * 0.95)] if ids_sorted else 0
            for p in sorted(d.rglob("*.wav")):
                try:
                    dlg = int(p.parent.name)   # dialogue dir name
                except ValueError:
                    # fall back to stem suffix
                    parts = p.stem.rsplit("_d", 1)
                    if len(parts) != 2 or not parts[1].isdigit():
                        continue
                    dlg = int(parts[1])
                if dlg < cutoff:
                    yield p
    elif src == "EmoV-DB":
        # Jenie speaker held out (FR speaker doubles as cross-lingual robustness eval).
        d = RAW / "EmoV-DB"
        if d.exists():
            for p in sorted(d.rglob("*.wav")):
                if "jenie" in str(p).lower():
                    continue
                yield p
    else:
        return


def main() -> None:
    excl = load_exclusions()
    print("loaded exclusion counts:", {s: len(v) for s, v in excl.items()})

    sources = ["RAVDESS", "MELD", "CREMA-D", "Emotion-Speech", "TESS", "IEMOCAP",
               "MUStARD", "DailyTalk", "EmoV-DB"]
    manifest_path = OUT_DIR / "train_manifest.jsonl"
    report = {}

    with manifest_path.open("w") as fout:
        for src in sources:
            files = list(iter_source_files(src))
            if not files:
                report[src] = {"available": 0, "excluded": 0, "kept": 0}
                continue
            ex = excl.get(src, set())
            kept = [p for p in files if str(p) not in ex]
            for p in kept:
                fout.write(json.dumps({
                    "path": str(p),
                    "source": src,
                    "relpath": str(p.relative_to(RAW)),
                }) + "\n")
            report[src] = {"available": len(files), "excluded": len(files) - len(kept), "kept": len(kept)}
            print(f"  {src:<15} avail={len(files):>6}  excluded={len(files) - len(kept):>4}  kept={len(kept):>6}")

    (OUT_DIR / "filter_report.json").write_text(json.dumps(report, indent=2))
    total_kept = sum(r["kept"] for r in report.values())
    print(f"\ntotal kept: {total_kept}")
    print(f"manifest: {manifest_path}")
    print(f"report:   {OUT_DIR / 'filter_report.json'}")


if __name__ == "__main__":
    main()
