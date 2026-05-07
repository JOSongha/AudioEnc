"""FSD50K eval with sentence/caption prompt (matches v5/v6 Stage-1 training).

See ``eval_audioset_caption.py`` for the same naming caveat: prompt format
is caption-form across v3 / v4 / v5 / v6, but the *target text content*
differs (v3 / v4 = synthetic "sound of X, Y" comma-list, v5+ = ontology
description sentences). The wrapper's sentence-mode prompt fits the v5+
target style, so running on a v4 ckpt gives artificially low F1 (model
outputs label-list comma-strings rather than sentences).

Same wrapper pattern: forces ``--score-mode sentence`` on the Stage-2
evaluator so the prompt becomes ``Describe what you hear in this audio.
Mention every distinct sound event.``, labels are recovered by
substring-matching the FSD50K 200-class vocabulary, and F1/Jaccard are
reported over the recovered label set.

Usage (same args as evaluation.stage2.eval_fsd50k_map, minus --score-mode):
    python -m evaluation.stage1_v4.eval_fsd50k_caption \
        --ckpt-root /mnt/tmp/Qwen3.5_dac_vae_v6_Stage1_jos/... \
        --base-model /mnt/ddn/users/jos/audiollm-trainer/external/models/Qwen3.5AE-4B \
        --out-root  .../eval_fsd50k_caption \
        --ckpts 68000
"""
from __future__ import annotations

import sys

from evaluation.stage2.eval_fsd50k_map import main as _stage2_main


def main():
    if "--score-mode" not in sys.argv:
        sys.argv.extend(["--score-mode", "sentence"])
    _stage2_main()


if __name__ == "__main__":
    main()
