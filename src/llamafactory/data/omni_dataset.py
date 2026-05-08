# Copyright 2026 NAVER Cloud Foundation Research team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Omni dataset pipeline: IterableDataset-based speech-text multimodal training.

Flow:
    JSONL manifest (file or directory of .jsonl files)
      → load_dataset("json", streaming=True)  # HF IterableDataset
      → split_dataset_by_node()               # DDP sharding
      → .map(process_sample)                   # audio load + tokenize
      → .map(pack_samples, batched=True)       # greedy knapsack packing
      → DataLoader(batch_size=B, drop_last=True)
      → OmniCollator                           # tensor conversion + padding + 4D attention mask
"""

import bisect
import glob
import io
import os
import random
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any, Literal

import requests
import torch
import torchaudio

from ..extras.constants import IGNORE_INDEX
from .collator import batch_group_counter, prepare_4d_attention_mask

# Why: `requests` `timeout=N` is per-recv inactivity, not wall-clock; a stuck
# socket can drip bytes (or stall in connect/DNS) past it. And `with
# ThreadPoolExecutor:` blocks `__exit__` until all workers join, so a single
# hung worker freezes the whole batch — observed as 600s NCCL ALLREDUCE
# timeouts in DDP (Whisper-small v4 Stage1, 2026-05-05). We bound per-call
# with a (connect, read) tuple AND the whole batch with `as_completed(
# timeout=...)` + `shutdown(wait=False)` so leaked threads die in background.
NUBES_CONNECT_TIMEOUT = 3.0
NUBES_READ_TIMEOUT = 5.0
NUBES_BATCH_DEADLINE = 30.0

# Outer watchdog: even non-nubes layers (audio decode, mel extract, tokenize,
# packing) can hang on bad samples or syscalls. A SIGALRM-based deadline lets
# us return a partial batch instead of letting one bad row stall the rank for
# 600s and trip NCCL. Each fire is logged with the row that was active at the
# time + the running count, so a deterministic-step crash leaves a pattern.
PROCESS_BATCH_DEADLINE = 60  # seconds


class _BatchTimeout(Exception):
    """Raised by SIGALRM handler when process_samples exceeds the deadline."""


def _on_alarm(signum, frame):
    raise _BatchTimeout()


_watchdog_fire_count = [0]  # mutable holder; per-worker process


# ---------------------------------------------------------------------------
# Task-prompt pool (paraphrase bank per task).  Diversifies training-time user
# message so the model doesn't overfit a single phrasing.  Eval scripts pin a
# fixed phrasing; the first entry of each pool MUST match the eval string so
# the training distribution covers eval.
#
# Invariants:
#   - TASK_PROMPTS["asr"][0]              == eval_testclean_wer.py:31 literal.
#   - Any entry must stand alone as a user message (no assistant-role tokens).
#   - No emoji / no non-ASCII that would surprise the tokenizer.
# ---------------------------------------------------------------------------
# Per-token modality id stored alongside input_ids so the custom trainer can
# compute loss separately per task. Sentinel -1 marks padding / ignored tokens.
MODALITY_ID_MAP: dict[str, int] = {
    "audio_asr":       0,
    "audio_emotion":   1,
    "audio_env_sound": 2,
    "text":            3,
}
MODALITY_NAMES: dict[int, str] = {v: k for k, v in MODALITY_ID_MAP.items()}
MODALITY_PAD_ID: int = -1


TASK_PROMPTS: dict[str, list[str]] = {
    "asr": [
        # PINNED as [0] to match eval_testclean_wer.py:31 CHATML_MID literal.
        "Transcribe the audio to text.",
        "Transcribe this recording.",
        "Write down what is said in the audio.",
        "Please transcribe this audio clip.",
        "Provide a transcription of the spoken audio.",
        "Convert the speech in this clip into text.",
        "Type out the words spoken in the audio.",
        "Give me a verbatim transcript of the recording.",
        "What is being said in this recording?",
        "What are the exact words spoken in this audio?",
        "Can you tell me what the speaker is saying?",
        "Transcribe.",
        "Transcript, please.",
        "Audio -> text.",
        "Listen carefully and write down what you hear.",
        "Produce an accurate transcription of this utterance.",
        "Turn the following speech into written text.",
        "Render the spoken content of this clip as text.",
    ],
    "sound_caption": [
        "Describe what you hear in the audio.",
        "Describe the sounds in this clip.",
        "Describe this audio in one sentence.",
        "Briefly describe the content of this audio.",
        "Give a natural-language description of this audio.",
        "Summarize what is happening in this recording.",
        "Write a caption for this audio clip.",
        "Caption the audio based on what it contains.",
        "Provide a short caption describing this sound.",
        "What sounds are present in this audio?",
        "What can you hear in this recording?",
        "What is going on acoustically in this clip?",
        "In a sentence, what does this audio sound like?",
        "Give a one-line description of this sound.",
        "Provide a description of this sound.",
        "Listen to the clip and describe the audio scene.",
        "Tell me what this recording sounds like.",
    ],
    "sound_classify_multi": [
        "List the sound events in this audio, separated by commas.",
        "Identify all sound events audible in this clip, comma-separated.",
        "What sound events occur in this audio? List them, separated by commas.",
        "Enumerate the sounds present in this clip.",
        "Name every sound event you can hear, separated by commas.",
    ],
    "sound_classify_single": [
        "Classify this sound.",
        "What category of sound is this?",
        "Identify the class of this sound.",
        "Give a single-class label for this audio.",
        "What does this sound belong to?",
    ],
    # Sentence-form variants (sentence_form_sound_p > 0 in processor)
    "sound_describe_multi": [
        "Describe what you hear in this audio. Mention every distinct sound event.",
        "In a sentence, describe all the sounds you can identify in this clip.",
        "Tell me what sounds are present in this audio.",
        "Listen to the audio and describe each sound event in a single sentence.",
    ],
    "sound_describe_single": [
        "Describe the sound you hear in this audio in one sentence.",
        "In one sentence, what sound is this?",
        "Describe this sound briefly.",
    ],
    "emotion_classify": [
        "What emotion is being expressed in the audio?",
        "What emotion does the speaker convey?",
        "What is the speaker feeling?",
        "What mood is the speaker in?",
        "How does the speaker sound emotionally?",
        "Classify the emotion of the speaker.",
        "Label the speaker's emotional state.",
        "Identify the emotional tone of this utterance.",
        "Determine the affective state of the speaker.",
        "Which emotion best describes the speaker's state?",
        "Pick the emotion that best fits this utterance.",
        "Select the emotion expressed by the speaker.",
        "Based on the speaker's voice, what emotion are they expressing?",
        "Judging from tone and prosody, what is the speaker feeling?",
        "From the vocal cues alone, identify the speaker's emotion.",
        "Emotion?",
        "Speaker emotion:",
    ],
}


# ---------------------------------------------------------------------------
# Sentence-form helpers (sentence_form_sound_p > 0 enables in processor)
# ---------------------------------------------------------------------------


def _english_join(items: list[str]) -> str:
    """Oxford-comma join. ['a','b'] -> 'a and b'; ['a','b','c'] -> 'a, b, and c'."""
    items = [str(x) for x in items if x]
    if not items: return ""
    if len(items) == 1: return items[0]
    if len(items) == 2: return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


_SENTENCE_TEMPLATES_MULTI = [
    "I hear {x} in this audio.",
    "The audio contains {x}.",
    "This recording features {x}.",
    "{x_cap} can be heard in the audio.",
    "Sounds in the clip: {x}.",
]
_SENTENCE_TEMPLATES_SINGLE = [
    "I hear {x}.",
    "This sounds like {x}.",
    "The sound is {x}.",
    "It is {x}.",
]


def format_labels_as_sentence(labels: list[str], rng, multi: bool = True) -> str:
    """Render a label set as a natural sentence.

    multi=True   ['Horse','Dog']   -> 'I hear horse and dog in this audio.'
    multi=False  ['Bell']          -> 'The sound is bell.'
    Underscores in label names are converted to spaces and labels are
    lowercased to read naturally.
    """
    norm = [l.replace("_", " ").lower() for l in labels if l]
    if not norm: return ""
    if multi:
        joined = _english_join(norm)
        tpl = rng.choice(_SENTENCE_TEMPLATES_MULTI)
        return tpl.format(x=joined, x_cap=joined[:1].upper() + joined[1:])
    return rng.choice(_SENTENCE_TEMPLATES_SINGLE).format(x=norm[0])


# ---------------------------------------------------------------------------
# 1. Manifest path utils
# ---------------------------------------------------------------------------


def resolve_jsonl_files(manifest_path: str) -> list[str]:
    """Return a sorted list of .jsonl file paths from a file or directory."""
    if os.path.isdir(manifest_path):
        return sorted(glob.glob(os.path.join(manifest_path, "*.jsonl")))
    return [manifest_path]


# ---------------------------------------------------------------------------
# 2. Per-sample processor (used with IterableDataset.map)
# ---------------------------------------------------------------------------


def create_omni_processor(
    tokenizer,
    audio_pad_token_id: int,
    sample_rate: int = 48000,
    hop_length: int = 1920,
    max_audio_samples: int = 5760000,
    load_from_nubes: bool = False,
    nubes_gateway: str = "http://c.nubes.sto.navercorp.com:8000/v1",
    nubes_max_workers: int = 12,
    sentence_form_sound_p: float = 0.0,
):
    """Per-row multimodal ChatML builder.

    Modality drives the user-side text and assistant target:
      audio_asr        Transcribe the audio.                → assistant: transcript
      audio_emotion    MCQA: question + lettered choices.   → assistant: letter (+rationale if present)
      audio_env_sound  Clotho caption / FSD50K or ESC-50
                       classification.                       → assistant: caption | label list | class
      text             MCQA: question + lettered choices
                       (no audio).                           → assistant: letter

    Falls back to `audio_asr` if `modality` is missing (keeps Stage-1 shards
    — `{"nubes_path", "text"}` rows — working unchanged).
    """
    eos_token_id = tokenizer.eos_token_id
    rng = random.Random(20260424)

    # Encode constant ChatML fragments once.
    _sys_user = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    _end_user = "<|im_end|>\n<|im_start|>assistant\n"
    sys_user_ids = tokenizer.encode(_sys_user, add_special_tokens=False)
    audio_start_ids = tokenizer.encode("<|audio_start|>", add_special_tokens=False)
    audio_end_ids = tokenizer.encode("<|audio_end|>", add_special_tokens=False)
    end_user_ids = tokenizer.encode(_end_user, add_special_tokens=False)

    def _enc(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    # ---- per-modality user-suffix + assistant-target ------------------------
    #
    # Prompt stems are sampled from TASK_PROMPTS (module-level pool) to avoid
    # overfitting on a single phrasing. MCQA variants keep the fixed suffix
    # "Choices: ... Answer with the letter." so the parsing harness stays
    # compatible; only the stem question varies.
    def _build_prompt_targets(row: dict) -> tuple[str, str] | None:
        """Return (user_suffix_text, assistant_target_text) or None to drop row."""
        modality = row.get("modality") or "audio_asr"

        if modality == "audio_asr":
            text = row.get("text")
            if not text:
                return None
            return (rng.choice(TASK_PROMPTS["asr"]), text)

        if modality == "audio_emotion":
            q = row.get("question")
            choices = row.get("choices") or []
            ans = row.get("answer")
            if not choices or not ans:
                return None
            choices_str = " ".join(choices) if isinstance(choices, list) else str(choices)
            rationale = row.get("rationale")
            target = f"{ans}. {rationale}" if rationale else ans
            # Override the manifest-baked generic question with a paraphrase
            # stem; keep choices + letter-instruction suffix for parse compat.
            stem = rng.choice(TASK_PROMPTS["emotion_classify"])
            return (f"{stem}\nChoices: {choices_str}\nAnswer with the letter.", target)

        if modality == "audio_env_sound":
            # v3 sources (clotho, audiocaps, macs, laion_freesound, laion_bbc,
            # laion_epidemic, laion_audiostock, audioset, fsd50k) all carry a
            # `captions` list — pick one and route to sound_caption pool.
            caps = row.get("captions") or []
            if caps:
                return (rng.choice(TASK_PROMPTS["sound_caption"]), rng.choice(caps))
            # Legacy `labels`-based shards (esc50 and pre-v3 fsd50k) still
            # supported. Keep until those shards are deprecated.
            src = row.get("source", "")
            labels = row.get("labels") or []
            if not labels:
                return None
            if src == "fsd50k":
                if sentence_form_sound_p > 0 and rng.random() < sentence_form_sound_p:
                    return (rng.choice(TASK_PROMPTS["sound_describe_multi"]),
                            format_labels_as_sentence(labels, rng, multi=True))
                return (rng.choice(TASK_PROMPTS["sound_classify_multi"]),
                        ", ".join(labels))
            if src == "esc50":
                if sentence_form_sound_p > 0 and rng.random() < sentence_form_sound_p:
                    return (rng.choice(TASK_PROMPTS["sound_describe_single"]),
                            format_labels_as_sentence([labels[0]], rng, multi=False))
                return (rng.choice(TASK_PROMPTS["sound_classify_single"]),
                        str(labels[0]))
            return None

        if modality == "text":
            q = row.get("question")
            choices = row.get("choices") or []
            ans = row.get("answer")
            if not q or not choices or not ans:
                return None
            choices_str = " ".join(choices) if isinstance(choices, list) else str(choices)
            return (f"{q}\nChoices: {choices_str}\nAnswer with the letter.", ans)

        return None

    # Per-worker session (created lazily, one per DataLoader worker process)
    _session_holder: list[requests.Session | None] = [None]

    def _get_session() -> requests.Session:
        if _session_holder[0] is None:
            session = requests.Session()
            session.mount("http://", requests.adapters.HTTPAdapter(max_retries=0))
            _session_holder[0] = session
        return _session_holder[0]

    def _download_from_nubes(nubes_path: str) -> tuple[Any, int] | None:
        try:
            session = _get_session()
            resp = session.get(
                f"{nubes_gateway}/{nubes_path}",
                timeout=(NUBES_CONNECT_TIMEOUT, NUBES_READ_TIMEOUT),
            )
            if resp.status_code != 200:
                print(f"[omni] nubes HTTP {resp.status_code} for {nubes_path}", flush=True)
                return None
            waveform, sr = torchaudio.load(io.BytesIO(resp.content))
            return waveform, sr
        except Exception as e:
            print(f"[omni] nubes download error: {e} ({nubes_path})", flush=True)
            return None

    def _rows(examples: dict[str, list[Any]]) -> list[dict[str, Any]]:
        """Transpose HF batched-dict-of-lists → list-of-dicts (per-row)."""
        n = 0
        for v in examples.values():
            if isinstance(v, list):
                n = max(n, len(v))
        out = []
        for i in range(n):
            out.append({k: (v[i] if isinstance(v, list) and i < len(v) else None)
                        for k, v in examples.items()})
        return out

    def process_samples(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        all_input_ids: list[list[int]] = []
        all_labels: list[list[int]] = []
        all_audio_features: list[Any] = []
        all_audio_lengths: list[int] = []
        all_audio_modality_ids: list[int] = []
        all_modality_ids: list[list[int]] = []

        rows = _rows(examples)
        n = len(rows)
        active_stage = "init"
        active_row_info = ""

        old_handler = signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(PROCESS_BATCH_DEADLINE)
        try:
            # Pre-download nubes audio in parallel (only for rows that need it).
            active_stage = "nubes_fetch"
            needs_nubes_idx = [i for i, r in enumerate(rows)
                               if (r.get("modality") or "audio_asr") != "text"
                               and load_from_nubes
                               and r.get("nubes_path")]
            nubes_results: dict[int, Any] = {}
            if needs_nubes_idx:
                paths = [rows[i]["nubes_path"] for i in needs_nubes_idx]
                pool = ThreadPoolExecutor(max_workers=nubes_max_workers)
                futures = {
                    pool.submit(_download_from_nubes, p): (i, p)
                    for i, p in zip(needs_nubes_idx, paths)
                }
                try:
                    for fut in as_completed(futures, timeout=NUBES_BATCH_DEADLINE):
                        i, _ = futures[fut]
                        try:
                            nubes_results[i] = fut.result()
                        except Exception as e:
                            print(f"[omni] future error: {e}", flush=True)
                            nubes_results[i] = None
                except FuturesTimeoutError:
                    for fut, (i, p) in futures.items():
                        if not fut.done():
                            nubes_results[i] = None
                            fut.cancel()
                            print(
                                f"[omni] nubes batch deadline {NUBES_BATCH_DEADLINE}s exceeded; "
                                f"skipping {p}",
                                flush=True,
                            )
                finally:
                    pool.shutdown(wait=False)

            for i, row in enumerate(rows):
                modality = row.get("modality") or "audio_asr"
                active_row_info = (
                    f"i={i}/{n} mod={modality} "
                    f"src={row.get('source')} "
                    f"path={row.get('nubes_path') or row.get('path') or row.get('audio_path')}"
                )
                active_stage = "build_prompt"
                pt = _build_prompt_targets(row)
                if pt is None:
                    continue
                user_suffix_text, target_text = pt

                # ---- (A) get waveform for audio rows, skip for text -------------
                waveform = None
                if modality != "text":
                    active_stage = "load_waveform"
                    try:
                        if load_from_nubes and row.get("nubes_path"):
                            result = nubes_results.get(i)
                            if result is None:
                                continue
                            waveform, sr = result
                        elif row.get("path"):
                            waveform, sr = torchaudio.load(row["path"])
                        elif row.get("audio_path"):
                            waveform, sr = torchaudio.load(row["audio_path"])
                        else:
                            continue
                    except Exception as e:
                        print(f"[omni] Audio load error: {e} "
                              f"(nubes_path={row.get('nubes_path')}, "
                              f"path={row.get('path') or row.get('audio_path')})", flush=True)
                        continue

                    active_stage = "resample"
                    if sr != sample_rate:
                        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    if max_audio_samples is not None and waveform.shape[-1] > max_audio_samples:
                        # head-truncate: match Whisper variant behavior so long clips
                        # (LAION-BBC, Freesound, etc.) stay in the training pool.
                        waveform = waveform[..., :max_audio_samples]

                # ---- (B) token counts -------------------------------------------
                active_stage = "token_counts"
                if waveform is not None:
                    num_samples = waveform.shape[-1]
                    t_audio = num_samples // hop_length
                    if t_audio == 0:
                        continue
                else:
                    t_audio = 0

                active_stage = "tokenize"
                user_suffix_ids = _enc(user_suffix_text)
                target_ids = _enc(target_text)
                if not target_ids:
                    continue

                # ---- (C) assemble input_ids + labels ----------------------------
                active_stage = "assemble"
                input_ids = list(sys_user_ids)
                if t_audio > 0:
                    input_ids.extend(audio_start_ids)
                    input_ids.extend([audio_pad_token_id] * t_audio)
                    input_ids.extend(audio_end_ids)
                input_ids.extend(user_suffix_ids)
                input_ids.extend(end_user_ids)
                context_len = len(input_ids)
                input_ids.extend(target_ids)
                input_ids.append(eos_token_id)

                labels = [IGNORE_INDEX] * context_len + list(target_ids) + [eos_token_id]

                all_input_ids.append(input_ids)
                all_labels.append(labels)
                # Per-token modality id aligned with input_ids; used downstream for
                # per-task loss decomposition. A single row has one modality, so
                # fill the whole sequence with the row's modality id.
                mod_id = MODALITY_ID_MAP.get(modality, MODALITY_PAD_ID)
                all_modality_ids.append([mod_id] * len(input_ids))
                # Maintain 1:1 index alignment with input_ids. Text rows get a
                # length-0 placeholder — packer filters these out via the
                # `audio_lengths > 0` check, so no empty tensor ever reaches
                # the collator's stack(). audio_modality_ids parallels
                # audio_features so the encoder can gate noise aug per modality.
                if waveform is not None:
                    all_audio_features.append(waveform.squeeze(0))
                    all_audio_lengths.append(t_audio)
                    all_audio_modality_ids.append(mod_id)
                else:
                    all_audio_features.append(torch.zeros(0, dtype=torch.float32))
                    all_audio_lengths.append(0)
                    all_audio_modality_ids.append(MODALITY_PAD_ID)
        except _BatchTimeout:
            _watchdog_fire_count[0] += 1
            print(
                f"[omni-watchdog] FIRE #{_watchdog_fire_count[0]} (dac) "
                f"deadline={PROCESS_BATCH_DEADLINE}s "
                f"stage={active_stage} {active_row_info} "
                f"completed={len(all_input_ids)}/{n} pid={os.getpid()}",
                flush=True,
            )
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        return {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "audio_features": all_audio_features,
            "audio_lengths": all_audio_lengths,
            "audio_modality_ids": all_audio_modality_ids,
            "modality_ids": all_modality_ids,
        }

    return process_samples


# ---------------------------------------------------------------------------
# 3. Sequence packing (used with IterableDataset.map batched=True)
# ---------------------------------------------------------------------------


def _search_for_fit(numbers: list[int], capacity: int) -> int:
    """Find the index of largest number that fits within capacity."""
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def create_omni_packer(
    cutoff_len: int,
    pad_token_id: int,
    neat_packing: bool = True,
):
    """Return a batched map function that packs samples via greedy knapsack."""

    def pack_samples(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        n = len(examples["input_ids"])

        # Build (length, original_index) pairs, filter oversized, sort by length
        indexed_lengths = sorted(
            [(len(examples["input_ids"][i]), i) for i in range(n) if len(examples["input_ids"][i]) <= cutoff_len],
            key=lambda x: x[0],
        )
        sorted_lengths = [l for l, _ in indexed_lengths]
        sorted_indices = [i for _, i in indexed_lengths]

        # Greedy knapsack — pack indices, not just lengths
        knapsacks: list[list[int]] = []  # each knapsack = list of original indices
        while sorted_lengths:
            current_bin: list[int] = []
            remaining = cutoff_len

            while True:
                idx = _search_for_fit(sorted_lengths, remaining)
                if idx == -1:
                    break
                remaining -= sorted_lengths.pop(idx)
                current_bin.append(sorted_indices.pop(idx))

            knapsacks.append(current_bin)

        # Build packed outputs
        model_inputs: dict[str, list] = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
            "audio_features": [],  # list of lists — each packed seq's audio waveforms
            "audio_lengths": [],  # list of lists — each packed seq's audio token counts
            "audio_modality_ids": [],  # list of lists — per-audio modality id, aligned with audio_features
            "modality_ids": [],  # per-token modality id, aligned with input_ids
        }

        have_modality = "modality_ids" in examples
        have_audio_modality = "audio_modality_ids" in examples

        for knapsack in knapsacks:
            packed_input_ids = []
            packed_labels = []
            packed_attention_mask = []
            packed_audio_features = []
            packed_audio_lengths = []
            packed_audio_modality_ids = []
            packed_modality_ids = []

            for seq_idx, orig_idx in enumerate(knapsack):
                ids = examples["input_ids"][orig_idx]
                lbl = examples["labels"][orig_idx]
                packed_input_ids.extend(ids)
                packed_labels.extend(lbl)

                if have_modality:
                    packed_modality_ids.extend(examples["modality_ids"][orig_idx])
                else:
                    packed_modality_ids.extend([MODALITY_PAD_ID] * len(ids))

                if neat_packing:
                    packed_attention_mask.extend([seq_idx + 1] * len(ids))  # 1-indexed
                else:
                    packed_attention_mask.extend([1] * len(ids))

                # Skip length-0 placeholders from text-only rows (see processor).
                if examples["audio_lengths"][orig_idx] > 0:
                    packed_audio_features.append(examples["audio_features"][orig_idx])
                    packed_audio_lengths.append(examples["audio_lengths"][orig_idx])
                    if have_audio_modality:
                        packed_audio_modality_ids.append(
                            examples["audio_modality_ids"][orig_idx]
                        )
                    else:
                        packed_audio_modality_ids.append(MODALITY_PAD_ID)

            # Pad to cutoff_len
            pad_len = cutoff_len - len(packed_input_ids)
            if pad_len > 0:
                packed_input_ids.extend([pad_token_id] * pad_len)
                packed_labels.extend([IGNORE_INDEX] * pad_len)
                packed_attention_mask.extend([0] * pad_len)
                packed_modality_ids.extend([MODALITY_PAD_ID] * pad_len)

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["attention_mask"].append(packed_attention_mask)
            model_inputs["audio_features"].append(packed_audio_features)
            model_inputs["audio_lengths"].append(packed_audio_lengths)
            model_inputs["audio_modality_ids"].append(packed_audio_modality_ids)
            model_inputs["modality_ids"].append(packed_modality_ids)

        return model_inputs

    return pack_samples


# ---------------------------------------------------------------------------
# 4. Collator
# ---------------------------------------------------------------------------


@dataclass
class OmniCollator:
    """Collator for omni pipeline.

    Converts packed samples to tensors, flattens audio_features across the batch,
    and builds 4D attention mask for eager/sdpa or keeps seq indices for flash_attn.
    """

    pad_token_id: int
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: torch.dtype = torch.float32
    block_diag_attn: bool = False

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # 1. Stack input_ids, labels, attention_mask (already padded to cutoff_len by packer)
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)

        # 2. Flatten audio_features across batch → (N_audio, 1, S_max)
        all_waveforms = []
        all_audio_lengths = []
        all_audio_modality_ids = []
        for f in features:
            for wav in f["audio_features"]:
                all_waveforms.append(wav if isinstance(wav, torch.Tensor) else torch.tensor(wav, dtype=torch.float32))
            all_audio_lengths.extend(f["audio_lengths"])
            # audio_modality_ids parallels audio_features (one entry per audio).
            # Old shards may not have it; default to MODALITY_PAD_ID and let
            # the encoder fall back to legacy noise-aug-on-all behavior.
            if "audio_modality_ids" in f:
                all_audio_modality_ids.extend(f["audio_modality_ids"])
            else:
                all_audio_modality_ids.extend([MODALITY_PAD_ID] * len(f["audio_lengths"]))

        if all_waveforms:
            max_audio_len = max(w.size(0) for w in all_waveforms)
            audio_features = torch.stack(
                [torch.nn.functional.pad(w, (0, max_audio_len - w.size(0))) for w in all_waveforms]
            ).unsqueeze(1)  # (N_audio, 1, S_max)
        else:
            audio_features = torch.zeros((0, 1, 0), dtype=torch.float32)

        audio_lengths = torch.tensor(all_audio_lengths, dtype=torch.long)
        audio_modality_ids = torch.tensor(all_audio_modality_ids, dtype=torch.long)

        # 3. Build attention mask / position_ids
        # Stack per-token modality ids (same shape as input_ids). If upstream
        # packer didn't emit them (old shards), default to -1 (ignored).
        if all("modality_ids" in f for f in features):
            modality_ids = torch.tensor([f["modality_ids"] for f in features], dtype=torch.long)
        else:
            modality_ids = torch.full_like(input_ids, MODALITY_PAD_ID)

        result = {
            "input_ids": input_ids,  # (B, cutoff_len)
            "labels": labels,  # (B, cutoff_len)
            "audio_features": audio_features,  # (N_audio, 1, S_max)
            "audio_lengths": audio_lengths,  # (N_audio,)
            "audio_modality_ids": audio_modality_ids,  # (N_audio,) — gates ASR-only noise aug in encoder
            "modality_ids": modality_ids,  # (B, cutoff_len)  — consumed by OmniTrainer, popped before model call
        }

        if self.block_diag_attn:
            if self.attn_implementation != "flash_attention_2":
                # eager / sdpa → 4D block-diagonal causal mask
                result["attention_mask"] = prepare_4d_attention_mask(attention_mask, self.compute_dtype)
            else:
                # flash_attention_2 → remove padding, merge into single sequence
                non_pad = attention_mask != 0
                result["input_ids"] = input_ids[non_pad].unsqueeze(0)
                result["labels"] = labels[non_pad].unsqueeze(0)
                result["modality_ids"] = modality_ids[non_pad].unsqueeze(0)
                position_ids = batch_group_counter(attention_mask)
                position_ids = position_ids[non_pad].unsqueeze(0)
                result["labels"][position_ids == 0] = IGNORE_INDEX
                result["position_ids"] = position_ids

                # cu_seqlens for FA2 varlen + fla linear_attn: boundaries where position resets to 0.
                # int32 matches FA2's `flash_attn_varlen_func` requirement; fla's Triton kernels
                # cast internally so int32 also works on the linear_attn side.
                # max_seqlen is derived inside the model from cu_seqlens — no need to pass separately.
                flat_pos = position_ids[0]  # (N,)
                boundary_idx = (flat_pos == 0).nonzero(as_tuple=True)[0]  # sample starts
                total_len = flat_pos.numel()
                cu_seqlens = torch.cat(
                    [boundary_idx.to(torch.int32), torch.tensor([total_len], dtype=torch.int32)]
                )
                result["cu_seqlens"] = cu_seqlens

        return result
