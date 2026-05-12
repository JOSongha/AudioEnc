#!/usr/bin/env python
"""Build IEMOCAP same-transcript / different-speaker pairs.

Idea: IEMOCAP has 3 scripted scenes (script01, script02, script03) performed
by all 10 actors across 5 sessions. The transcript filename in
`SessionX/dialog/transcriptions/<dialog_id>.txt` is

    <utt_id> [start-end]: <text>

and `dialog_id` like `Ses01F_script01_1` says "Session 1, F-led take, script 1".
Within a script, utterances are indexed by `F000`, `M000`, `F001`, ...
The line ordering is fixed by the script: line `F003` of `script01_1` is the
same text whichever session performed it.

So we group by `(script_no, take_no, utt_idx)` and emit a "pair group" for
each transcript line that has >= 2 different (session, gender) recordings.

Audio path: `SessionX/sentences/wav/<dialog_id>/<utt_id>.wav`.

Cross-gender takes: the same line F003 of script01_1 is recorded in both
`Ses01F_*` (F-led) and `Ses01M_*` (M-led) takes — both pulled from the SAME
F actress. `--match_gender` drops the cross-gender take so we never count
the same actress twice in the same pair group. Matches the existing §4.3
manifest convention (`sec43_embeddings/build_manifest.py`).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

IEMOCAP_ROOT = Path("/mnt/tmp/datasets/emotion_raw/IEMOCAP/IEMOCAP_full_release")
OUT_PATH = Path(__file__).parent / "iemocap_pairs.jsonl"

DIALOG_RE = re.compile(r"^Ses(\d+)([FM])_script(\d+)_(\d+)$")
LINE_RE = re.compile(r"^(?P<uid>Ses\d+[FM]_script\d+_\d+_[FM]\d+)\s+\[[^\]]+\]:\s*(?P<txt>.+)$")
UID_TAIL_RE = re.compile(r"_(?P<gender>[FM])(?P<idx>\d+)$")


def collect(match_gender: bool, scripts_keep: set[str] | None):
    by_line = defaultdict(list)
    for sess in range(1, 6):
        trans_dir = IEMOCAP_ROOT / f"Session{sess}" / "dialog" / "transcriptions"
        if not trans_dir.exists():
            continue
        for fp in sorted(trans_dir.glob("Ses*_script*.txt")):
            m = DIALOG_RE.match(fp.stem)
            if not m:
                continue
            sess_no, led_gender, script_no, take_no = m.groups()
            script_full = f"script{script_no}_{take_no}"
            if scripts_keep is not None and script_full not in scripts_keep:
                continue
            for raw in fp.read_text(errors="ignore").splitlines():
                lm = LINE_RE.match(raw.strip())
                if not lm:
                    continue
                uid = lm.group("uid")
                text = lm.group("txt").strip()
                tail = UID_TAIL_RE.search(uid)
                if not tail:
                    continue
                spk_gender = tail.group("gender")
                line_idx = int(tail.group("idx"))
                if match_gender and spk_gender != led_gender:
                    # drop the cross-gender take: same actor as the led-gender take
                    continue
                wav = (
                    IEMOCAP_ROOT
                    / f"Session{sess}"
                    / "sentences"
                    / "wav"
                    / fp.stem
                    / f"{uid}.wav"
                )
                if not wav.exists():
                    continue
                key = (int(script_no), int(take_no), spk_gender, line_idx)
                by_line[key].append({
                    "audio_path": str(wav),
                    "text": text,
                    "spk": f"S{sess_no}{spk_gender}",
                    "session": int(sess_no),
                    "gender": spk_gender,
                    "uid": uid,
                })
    return by_line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match_gender", action="store_true", default=True,
                    help="drop cross-gender take (same actor double-count). Default on.")
    ap.add_argument("--no_match_gender", dest="match_gender", action="store_false",
                    help="keep both F-led and M-led takes (each take = separate row).")
    ap.add_argument("--scripts", nargs="*", default=None,
                    help='limit to specific script ids, e.g. "script01_1 script02_2 script03_2"')
    ap.add_argument("--min_speakers", type=int, default=2)
    args = ap.parse_args()

    scripts_keep = set(args.scripts) if args.scripts else None
    groups = collect(match_gender=args.match_gender, scripts_keep=scripts_keep)
    pairs = []
    for (script_no, take_no, spk_gender, line_idx), records in sorted(groups.items()):
        speakers = {r["spk"] for r in records}
        if len(speakers) < args.min_speakers:
            continue
        transcript_id = f"iemocap.script{script_no:02d}_{take_no}_{spk_gender}{line_idx:03d}"
        for r in records:
            pairs.append({
                "transcript_id": transcript_id,
                "src": "iemocap",
                "audio_path": r["audio_path"],
                "text": r["text"],
                "spk": r["spk"],
                "meta": {"uid": r["uid"], "session": r["session"], "gender": r["gender"]},
            })
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_t = defaultdict(int)
    for r in pairs:
        by_t[r["transcript_id"]] += 1
    print(f"[iemocap] groups={len(by_t)} rows={len(pairs)} -> {OUT_PATH}")
    print(f"[iemocap] avg speakers/group={sum(by_t.values()) / max(1, len(by_t)):.2f}")


if __name__ == "__main__":
    main()
