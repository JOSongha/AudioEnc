"""AudioSet eval with caption prompt (matches v4 Stage-1 training).

v4 Stage-1 trains the audio_env_sound modality exclusively in caption form
(``Describe what you hear in the audio.`` style stems with caption targets).
The Stage-2 ``eval_audioset_map.py`` defaults to a comma-list classification
prompt which is out-of-distribution for v4. This wrapper invokes the same
underlying evaluator with ``--score-mode sentence`` forced on, so the prompt
matches training and labels are recovered from the free-form description by
substring-matching the AudioSet vocabulary.

Everything else (model load path, AudioSet split, tokenization, F1/Jaccard
computation, output schema) is unchanged from the Stage-2 implementation.

Usage (same args as evaluation.stage2.eval_audioset_map, minus --score-mode):
    python -m evaluation.stage1_v4.eval_audioset_caption \
        --ckpt-root /mnt/tmp/Qwen3.5_dac_vae_v4_Stage1_jos/Qwen3.5AE-ASR-Stage1-dac-vae-v4 \
        --base-model /mnt/ddn/users/jos/audiollm-trainer/external/models/Qwen3.5AE-4B \
        --out-root  /mnt/tmp/Qwen3.5_dac_vae_v4_Stage1_jos/eval_v4_caption/eval_audioset_caption \
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
