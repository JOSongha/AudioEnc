#!/usr/bin/env python
"""Build CMU Arctic same-transcript / different-speaker pairs.

CMU Arctic layout per speaker:
  cmu_us_<spk>_arctic/etc/txt.done.data        # ( arctic_a0001 "text" ) lines
  cmu_us_<spk>_arctic/wav/arctic_a0001.wav

We pair by sentence_id (arctic_aXXXX / arctic_bXXXX). A pair group is emitted
when >= 2 speakers have a wav for the same sentence_id.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

ARCTIC_ROOT = Path("/mnt/tmp/datasets/cmu_arctic")
OUT_PATH = Path(__file__).parent / "arctic_pairs.jsonl"

LINE_RE = re.compile(r'^\(\s*(?P<sid>arctic_[ab]\d+)\s+"(?P<text>[^"]*)"\s*\)\s*$')
SPK_DIR_RE = re.compile(r"^cmu_us_(?P<spk>[a-z]+)_arctic$")


def collect():
    by_sid = defaultdict(list)
    for spk_dir in sorted(ARCTIC_ROOT.glob("cmu_us_*_arctic")):
        m = SPK_DIR_RE.match(spk_dir.name)
        if not m:
            continue
        spk = m.group("spk")
        prompt_file = spk_dir / "etc" / "txt.done.data"
        if not prompt_file.exists():
            print(f"[arctic] WARN missing {prompt_file}")
            continue
        wav_dir = spk_dir / "wav"
        for raw in prompt_file.read_text(errors="ignore").splitlines():
            lm = LINE_RE.match(raw.strip())
            if not lm:
                continue
            sid = lm.group("sid")
            text = lm.group("text").strip()
            wav = wav_dir / f"{sid}.wav"
            if not wav.exists():
                continue
            by_sid[sid].append({
                "audio_path": str(wav),
                "text": text,
                "spk": spk,
            })
    return by_sid


def main():
    groups = collect()
    pairs = []
    for sid, records in sorted(groups.items()):
        if len({r["spk"] for r in records}) < 2:
            continue
        transcript_id = f"arctic.{sid}"
        for r in records:
            pairs.append({
                "transcript_id": transcript_id,
                "src": "arctic",
                "audio_path": r["audio_path"],
                "text": r["text"],
                "spk": r["spk"],
                "meta": {"sid": sid},
            })
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_t = defaultdict(int)
    for r in pairs:
        by_t[r["transcript_id"]] += 1
    print(f"[arctic] groups={len(by_t)} rows={len(pairs)} -> {OUT_PATH}")
    print(f"[arctic] avg speakers/group={sum(by_t.values()) / max(1, len(by_t)):.2f}")


if __name__ == "__main__":
    main()
