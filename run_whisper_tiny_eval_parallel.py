#!/usr/bin/env python3
"""Whisper-tiny Stage-2 eval — driver-per-GPU parallel mode.

Each remaining driver runs on a single GPU and sequentially walks all 31 ckpts.
This eliminates the 8x I/O contention from multi-GPU sharding (8 procs reading
the same dataset concurrently).

Drivers already done: librispeech_wer, esc50_acc.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

CKPT_ROOT = Path("/mnt/tmp/results/Qwen3.5AE-Stage2-whisper-tiny-emoFull-asr033-env05-txt03")
OUT_ROOT = Path("/mnt/tmp/results/whisper_tiny_eval")
BASE_MODEL = "/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/ckpts/Qwen3.5_whisper_tiny_Stage1/Qwen3.5_whisper_tiny_Stage1/checkpoint-13000"
LOG_ROOT = Path("/tmp/whisper_tiny_eval_logs")

ENV_LIB = "/mnt/ddn/users/jos/miniforge3/envs/audio_lmf/lib"
LD_PRELOAD = f"{ENV_LIB}/libstdc++.so.6:{ENV_LIB}/glibc_compat.so"

ALL_CKPTS = ",".join(str(c) for c in range(1000, 32000, 1000))

# (gpu, module, extra_args)
JOBS = [
    (0, "evaluation.stage2.eval_audioset_map",    ["--batch-size", "8"]),
    (1, "evaluation.stage2.eval_fsd50k_map",      ["--batch-size", "4"]),
    (2, "evaluation.stage2.eval_clotho_caption",  ["--batch-size", "4", "--no-pycoco"]),
    (3, "evaluation.stage2.eval_listen_mcqa",     ["--batch-size", "4"]),
    (4, "evaluation.stage2.eval_listen_official", ["--batch-size", "4"]),
    (5, "evaluation.stage2.eval_source_emotion",  ["--batch-size", "4"]),
]


def main() -> None:
    print(f"[start] {time.strftime('%H:%M:%S')} jobs={len(JOBS)} (driver-per-GPU mode)",
          flush=True)

    procs = []
    for gpu, module, extra_args in JOBS:
        name = module.rsplit(".", 1)[-1].replace("eval_", "")
        out_dir = OUT_ROOT / name
        log_dir = LOG_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", module,
            "--ckpt-root", str(CKPT_ROOT),
            "--out-root", str(out_dir),
            "--base-model", BASE_MODEL,
            "--ckpts", ALL_CKPTS,
            *extra_args,
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "LD_PRELOAD": LD_PRELOAD}
        log_path = log_dir / f"gpu{gpu}_solo.log"
        log = open(log_path, "w")
        log.write(f"$ {' '.join(cmd)}\n")
        log.flush()
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((gpu, name, p, log_path))
        print(f"[launch] {name:20s} gpu{gpu} pid={p.pid} log={log_path}", flush=True)

    fail = []
    for gpu, name, p, log_path in procs:
        rc = p.wait()
        status = "OK" if rc == 0 else f"FAIL rc={rc}"
        print(f"[done]   {name:20s} gpu{gpu} {status}", flush=True)
        if rc != 0:
            fail.append((gpu, name, log_path))
    if fail:
        print(f"[warn]   {len(fail)} driver(s) failed: {fail}", flush=True)

    print(f"\n[end]   {time.strftime('%H:%M:%S')} ALL DRIVERS DONE", flush=True)


if __name__ == "__main__":
    main()
