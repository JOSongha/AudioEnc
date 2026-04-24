"""Offline rationale synthesis for emotion MCQA rows.

Reads emotion_mcqa_manifest.jsonl, fills `rationale: null` fields with a
20-50-token justification grounded on audio cues only, emits
emotion_mcqa_manifest_with_rationale.jsonl.

Uses an OpenAI-compatible chat-completion API. Supports three back-ends:
  1. OpenAI:       env OPENAI_API_KEY, model="gpt-4o-mini"
  2. Anthropic:    env ANTHROPIC_API_KEY, model="claude-haiku-4-5"  (via direct Anthropic SDK)
  3. Local vLLM:   env OMNI_LLM_BASE (e.g. "http://localhost:8000/v1"), OMNI_LLM_MODEL

Tune MAX_ROWS for a dry-run before committing; resume-safe (skips rows that
already have rationale != null).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

IN = Path("/mnt/tmp/listen_analysis/train_manifest/emotion_mcqa_manifest.jsonl")
OUT = Path("/mnt/tmp/listen_analysis/train_manifest/emotion_mcqa_manifest_with_rationale.jsonl")

MAX_ROWS = int(os.environ.get("OMNI_RATIONALE_MAX", 0)) or None  # 0 / unset → all
SLEEP_BETWEEN = float(os.environ.get("OMNI_RATIONALE_SLEEP", 0.05))


def make_prompt(row: dict) -> str:
    transcript = row.get("transcript") or ""
    label = row.get("label", "")
    return (
        f"Speaker says: \"{transcript.strip()}\"\n"
        f"Ground-truth emotion: {label}\n"
        "In 1-2 sentences (≤50 tokens), explain why this sounds like the "
        "given emotion using audio cues only — prosody, pitch, pacing, "
        "energy, timbre. Do NOT mention face, visual, or the literal words. "
        "Start with 'Because'."
    )


def call_openai(prompt: str) -> str:
    api_key = os.environ["OPENAI_API_KEY"]
    model = os.environ.get("OMNI_LLM_MODEL", "gpt-4o-mini")
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "max_tokens": 96, "temperature": 0.4,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def call_vllm(prompt: str) -> str:
    base = os.environ["OMNI_LLM_BASE"].rstrip("/")
    model = os.environ.get("OMNI_LLM_MODEL", "local")
    resp = requests.post(
        f"{base}/chat/completions",
        json={"model": model, "max_tokens": 96, "temperature": 0.4,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def call_anthropic(prompt: str) -> str:
    import anthropic  # local import so missing sdk doesn't blow imports
    client = anthropic.Anthropic()
    model = os.environ.get("OMNI_LLM_MODEL", "claude-haiku-4-5")
    msg = client.messages.create(
        model=model, max_tokens=96,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def pick_backend():
    if os.environ.get("OMNI_LLM_BASE"):
        return call_vllm, "vllm"
    if os.environ.get("OPENAI_API_KEY"):
        return call_openai, "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return call_anthropic, "anthropic"
    sys.exit("set one of: OMNI_LLM_BASE (vllm), OPENAI_API_KEY, ANTHROPIC_API_KEY")


def already_done(out_path: Path) -> set[str]:
    """Resume support — ids already written (by path key) to OUT."""
    seen = set()
    if not out_path.exists():
        return seen
    for line in out_path.open():
        try:
            seen.add(json.loads(line)["path"])
        except Exception:
            pass
    return seen


def main() -> None:
    call, name = pick_backend()
    print(f"backend: {name}")

    seen = already_done(OUT)
    print(f"resume: {len(seen):,} rows already in {OUT.name}")

    n_in = 0
    n_new = 0
    n_fail = 0

    with IN.open() as fin, OUT.open("a") as fout:
        for line in fin:
            row = json.loads(line)
            n_in += 1
            if MAX_ROWS and n_new >= MAX_ROWS:
                break
            if row["path"] in seen:
                continue
            if row.get("rationale"):
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            prompt = make_prompt(row)
            try:
                rationale = call(prompt)
            except Exception as e:
                n_fail += 1
                print(f"[err] {e} (path={row['path']})", flush=True)
                continue
            row["rationale"] = rationale
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_new += 1
            if n_new % 200 == 0:
                print(f"  synthesized {n_new:,}  ({n_fail} fails)", flush=True)
            time.sleep(SLEEP_BETWEEN)

    print(f"\ninput rows scanned: {n_in:,}")
    print(f"newly synthesized:  {n_new:,}")
    print(f"failed:             {n_fail:,}")
    print(f"output:             {OUT}")


if __name__ == "__main__":
    main()
