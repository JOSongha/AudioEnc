"""Filter leakage from LISTEN_full train split.

Drops LISTEN-train rows where:
  (a) base audio (id with `_<N>[A-Z]` variant suffix stripped) also appears in
      LISTEN-test — same-audio-different-question contamination.
  (b) id encodes a non-train split of the source corpus:
        MELD_(test|dev)_*          → MELD held-out items
        MOSEI_(Test|Val)_*         → CMU-MOSEI held-out items
        IEMOCAP_Session5_*         → IEMOCAP conventional held-out session
  (c) LISTEN-test itself — rows with base audio in LISTEN-test are excluded via (a),
      but we also strip LISTEN-test rows whose base audio collides with a source
      corpus *held-out* item the above rules protect, so LISTEN-test stays a strict
      subset (no rows removed from LISTEN-test by default — see FILTER_TEST flag).

Sources OMG / PODCAST / MUStARD / CREMA-D / RAVDESS / TESS / SAVEE / Emotion-Speech
have no original-split string in LISTEN IDs (or no canonical source split exists),
so no corpus-level split filter is applied to them — only rule (a) cleans them.

Input:  /mnt/tmp/listen_analysis/data/{train,test}-*.parquet
Output: /mnt/tmp/listen_analysis/filtered/{train,test}-*.parquet + filter_report.json
"""
import json
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path("/mnt/tmp/listen_analysis/data")
OUT = Path("/mnt/tmp/listen_analysis/filtered")
OUT.mkdir(parents=True, exist_ok=True)

VARIANT_RE = re.compile(r"_\d+[A-Z]$")


def strip_variant(id_str: str) -> str:
    return VARIANT_RE.sub("", id_str)


def main() -> None:
    train_paths = sorted(ROOT.glob("train-*.parquet"))
    print(f"Loading {len(train_paths)} train shards + 1 test shard…", flush=True)
    train = pd.concat([pd.read_parquet(p) for p in train_paths], ignore_index=True)
    test = pd.read_parquet(ROOT / "test-00000-of-00001.parquet")
    print(f"  train rows: {len(train):,}   test rows: {len(test):,}", flush=True)

    drop = pd.Series(False, index=train.index)

    test_base = set(test["id"].map(strip_variant))
    train_base = train["id"].map(strip_variant)
    m_audio = train_base.isin(test_base)
    drop |= m_audio

    m_meld = train["id"].str.match(r"MELD_(test|dev)_")
    drop |= m_meld

    m_mosei = train["id"].str.match(r"MOSEI_(Test|Val)_")
    drop |= m_mosei

    m_iemocap_s5 = train["id"].str.match(r"IEMOCAP_Session5_")
    drop |= m_iemocap_s5

    filtered = train.loc[~drop].reset_index(drop=True)

    print("\n── drop counts (non-exclusive; union is the total filter) ──")
    print(f"  rule (a) base-audio overlap with LISTEN-test : {int(m_audio.sum()):>6,}")
    print(f"  rule (b) MELD_(test|dev)_*                  : {int(m_meld.sum()):>6,}")
    print(f"  rule (b) MOSEI_(Test|Val)_*                 : {int(m_mosei.sum()):>6,}")
    print(f"  rule (b) IEMOCAP_Session5_*                 : {int(m_iemocap_s5.sum()):>6,}")
    print(f"  union dropped                               : {int(drop.sum()):>6,}")
    print(f"  kept                                        : {len(filtered):>6,}")

    print("\n── per-source rows before → after ──")
    before = train["dataset_source"].value_counts().to_dict()
    after = filtered["dataset_source"].value_counts().to_dict()
    for s in sorted(before):
        a = after.get(s, 0)
        print(f"  {s:<15} {before[s]:>6,} → {a:>6,}   ({a / before[s] * 100:5.1f}% kept)")

    n = len(filtered)
    n_shards = 3
    shard = -(-n // n_shards)
    for i in range(n_shards):
        lo = i * shard
        hi = min(lo + shard, n)
        out_path = OUT / f"train-{i:05d}-of-{n_shards:05d}.parquet"
        filtered.iloc[lo:hi].to_parquet(out_path, index=False)
        print(f"  wrote {out_path}  ({hi - lo:,} rows)")

    shutil.copy(ROOT / "test-00000-of-00001.parquet", OUT / "test-00000-of-00001.parquet")
    print(f"  copied test-00000-of-00001.parquet ({len(test):,} rows)")

    report = {
        "input_dir": str(ROOT),
        "output_dir": str(OUT),
        "train_rows_original": int(len(train)),
        "train_rows_kept": int(len(filtered)),
        "test_rows": int(len(test)),
        "drop_rules": {
            "a_base_audio_overlap_with_listen_test": int(m_audio.sum()),
            "b_meld_test_or_dev": int(m_meld.sum()),
            "b_mosei_test_or_val": int(m_mosei.sum()),
            "b_iemocap_session5": int(m_iemocap_s5.sum()),
            "union": int(drop.sum()),
        },
        "per_source_before": {k: int(v) for k, v in before.items()},
        "per_source_after": {k: int(v) for k, v in after.items()},
    }
    with (OUT / "filter_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport: {OUT / 'filter_report.json'}")


if __name__ == "__main__":
    main()
