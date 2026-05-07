"""AudioSet eval with sentence/caption prompt (matches v5/v6 Stage-1 training).

The Stage-2 ``eval_audioset_map.py`` defaults to a comma-list classification
prompt ("Speech, Music, ..."). This wrapper forces ``--score-mode sentence``
so the model is asked for a free-form caption ("Describe what you hear in
the audio.") and labels are recovered from the response by substring-matching
the AudioSet vocabulary.

⚠ **Naming caveat**: the directory is `evaluation/stage1_v4/` but the
wrapper now fits **v5 / v6** Stage-1 better than v4. v3 / v4 / v5 / v6 all
route AudioSet to the same caption-form prompt pool (`sound_caption`,
"Describe what you hear..." family — see [`omni_dataset.py:319`](../../src/llamafactory/data/omni_dataset.py#L319)),
so the *prompt* format is unchanged across versions. What v5 changed is the
**caption target text**: v3 / v4 used synthetic "sound of X, Y, Z" comma-list
captions built from labels, v5 swapped to full-sentence ontology descriptions
(see datasets.md changelog 2026-05-06 / v5 caption fix), v6 inherits v5.
Running this wrapper on a v4 Stage-1 ckpt asks the model for a sentence-style
description, but the v4 ckpt was trained to output "sound of X, Y" — F1 will
be artificially low. For v5 / v6 Stage-1 ckpts (and Stage-2 ckpts whose
Stage-1 base was v5 / v6) the response style matches training.

Everything else (model load path, AudioSet split, tokenization, F1/Jaccard
computation, output schema) is unchanged from the Stage-2 implementation.

Usage (same args as evaluation.stage2.eval_audioset_map, minus --score-mode):
    python -m evaluation.stage1_v4.eval_audioset_caption \
        --ckpt-root /mnt/tmp/Qwen3.5_dac_vae_v6_Stage1_jos/... \
        --base-model /mnt/ddn/users/jos/audiollm-trainer/external/models/Qwen3.5AE-4B \
        --out-root  .../eval_audioset_caption \
        --ckpts 68000
"""
from __future__ import annotations

import sys

from evaluation.stage2.eval_audioset_map import main as _stage2_main


def main():
    # Force sentence mode if user didn't pass it explicitly.
    if "--score-mode" not in sys.argv:
        sys.argv.extend(["--score-mode", "sentence"])
    _stage2_main()


if __name__ == "__main__":
    main()
