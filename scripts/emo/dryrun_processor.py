"""Sanity-check create_omni_processor on a few rows from the combined manifest.

Does not require GPU — loads up to 3 rows per modality, runs them through the
processor, prints input_ids/labels/audio shapes. Catches basic crashes in the
Option-C processor without burning smoke-run GPU time.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/mnt/ddn/users/jos/audiollm-trainer/src")

MANIFEST = Path("/mnt/tmp/listen_analysis/train_manifest/stage2_combined_manifest.jsonl")


def pick_rows(n_per_modality: int = 3) -> list[dict]:
    per = defaultdict(list)
    with MANIFEST.open() as f:
        for line in f:
            r = json.loads(line)
            m = r.get("modality", "audio_asr")
            if len(per[m]) < n_per_modality:
                per[m].append(r)
            if all(len(v) >= n_per_modality for v in per.values()) and len(per) >= 4:
                break
    return [r for rows in per.values() for r in rows]


def transpose(rows: list[dict]) -> dict[str, list]:
    """list of dicts → dict of lists (HF datasets batched-map convention)."""
    keys = set().union(*(r.keys() for r in rows))
    return {k: [r.get(k) for r in rows] for k in keys}


def main() -> None:
    from transformers import AutoTokenizer
    from llamafactory.data.omni_dataset import create_omni_processor

    tok = AutoTokenizer.from_pretrained(
        "/mnt/tmp/s2_init_42k", trust_remote_code=True,
    )
    audio_pad = tok.convert_tokens_to_ids("<|audio_pad|>")
    print(f"audio_pad_token_id: {audio_pad}")

    processor = create_omni_processor(
        tokenizer=tok,
        audio_pad_token_id=audio_pad,
        sample_rate=48000,
        hop_length=1920,
        max_audio_samples=1_600_000,
        load_from_nubes=True,      # ASR rows use nubes
    )

    rows = pick_rows(n_per_modality=2)
    print(f"picked {len(rows)} rows:")
    for r in rows:
        print(f"  [{r.get('modality')}] source={r.get('source')} "
              f"path={r.get('path') or r.get('nubes_path')}")

    batch = transpose(rows)
    result = processor(batch)
    print(f"\nprocessor output:")
    print(f"  input_ids:      {len(result['input_ids'])} rows")
    print(f"  labels:         {len(result['labels'])} rows")
    print(f"  audio_features: {len(result['audio_features'])} rows")
    print(f"  audio_lengths:  {result['audio_lengths']}")
    for i, (iid, lab, al) in enumerate(zip(result["input_ids"], result["labels"],
                                            result["audio_lengths"])):
        preview = tok.decode(iid[:30])
        print(f"  row {i}: len={len(iid)} audio_len={al} preview={preview!r}")


if __name__ == "__main__":
    main()
