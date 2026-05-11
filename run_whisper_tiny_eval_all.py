#!/usr/bin/env python3
"""Whisper-tiny Stage-2 evaluation orchestrator.

Runs all 9 evaluation drivers across all whisper-tiny S2 checkpoints, fanning
ckpts out across 8 GPUs in parallel within each driver. Each driver invocation
loops over its assigned ckpts internally (dataset loaded once per invocation).
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

CKPTS = list(range(1000, 32000, 1000))  # 1000..31000
N_GPUS = 8

# (module, extra_args) — sub-dir under OUT_ROOT defaults to module's last segment
DRIVERS = [
    ("evaluation.stage2.eval_librispeech_wer", ["--split", "test.clean", "--batch-size", "4"]),
    ("evaluation.stage2.eval_esc50_acc",       ["--all", "--batch-size", "8"]),
    ("evaluation.stage2.eval_audioset_map",    ["--batch-size", "8"]),
    ("evaluation.stage2.eval_fsd50k_map",      ["--batch-size", "4"]),
    ("evaluation.stage2.eval_clotho_caption",  ["--batch-size", "4", "--no-pycoco"]),
    ("evaluation.stage2.eval_listen_mcqa",     ["--batch-size", "4"]),
    ("evaluation.stage2.eval_listen_official", ["--batch-size", "4"]),
    ("evaluation.stage2.eval_source_emotion",  ["--batch-size", "4"]),
]


def shard_ckpts(ckpts: list[int], n_gpus: int) -> dict[int, list[int]]:
    """Round-robin assign ckpts to GPUs."""
    out = {g: [] for g in range(n_gpus)}
    for i, c in enumerate(ckpts):
        out[i % n_gpus].append(c)
    return out


def run_driver(driver_module: str, extra_args: list[str]) -> None:
    name = driver_module.rsplit(".", 1)[-1].replace("eval_", "")
    out_dir = OUT_ROOT / name
    log_dir = LOG_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    gpu_ckpts = shard_ckpts(CKPTS, N_GPUS)
    procs = []
    for gpu, ckpts in gpu_ckpts.items():
        if not ckpts:
            continue
        ckpt_arg = ",".join(str(c) for c in ckpts)
        cmd = [
            sys.executable, "-m", driver_module,
            "--ckpt-root", str(CKPT_ROOT),
            "--out-root", str(out_dir),
            "--base-model", BASE_MODEL,
            "--ckpts", ckpt_arg,
            *extra_args,
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "LD_PRELOAD": LD_PRELOAD}
        log_path = log_dir / f"gpu{gpu}.log"
        log = open(log_path, "w")
        log.write(f"$ {' '.join(cmd)}\n")
        log.flush()
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((gpu, p, log_path))
        print(f"[launch] {name} gpu{gpu} ckpts={ckpt_arg} pid={p.pid}", flush=True)

    fail = []
    for gpu, p, log_path in procs:
        rc = p.wait()
        status = "OK" if rc == 0 else f"FAIL rc={rc}"
        print(f"[done]   {name} gpu{gpu} {status} log={log_path}", flush=True)
        if rc != 0:
            fail.append((gpu, log_path))
    if fail:
        print(f"[warn]   {name}: {len(fail)} GPU(s) failed: {fail}", flush=True)


def main() -> None:
    print(f"[start] {time.strftime('%H:%M:%S')} ckpts={len(CKPTS)} drivers={len(DRIVERS)} gpus={N_GPUS}",
          flush=True)
    for module, args in DRIVERS:
        t0 = time.time()
        print(f"\n========== {module} ==========", flush=True)
        run_driver(module, args)
        dt = (time.time() - t0) / 60
        print(f"[driver-done] {module} elapsed={dt:.1f}m", flush=True)
    print(f"\n[end]   {time.strftime('%H:%M:%S')} ALL DRIVERS DONE", flush=True)


if __name__ == "__main__":
    main()
