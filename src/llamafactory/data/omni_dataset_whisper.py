"""Whisper-small variant of the omni dataset processor + collator.

Differences from `omni_dataset.py`:
  * audio loaded at 16 kHz mono (resample if needed) with optional `audio_start`/`audio_end`
  * waveform → log-mel [80, 3000] via `WhisperFeatureExtractor` (in dataloader, CPU)
  * `audio_pad_token` count = ceil(num_samples / 320), capped at 1500 (Whisper 50 fps × 30 s)
  * `audio_features` carries the mel tensor (not the waveform); collator stacks into [N, 80, 3000]

Reused from `omni_dataset.py` (no behavior change):
  * `TASK_PROMPTS`, `MODALITY_ID_MAP`, `MODALITY_PAD_ID`
  * `resolve_jsonl_files`, `create_omni_packer`
  * 4D attention mask / FA2 cu_seqlens machinery (we expose the same fields).
"""

import io
import os
import random
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any, Literal

import requests
import torch

from ..extras.constants import IGNORE_INDEX
from .audio_io import load_audio_chunk
from .collator import batch_group_counter, prepare_4d_attention_mask
from .omni_dataset import (
    MODALITY_ID_MAP,
    MODALITY_PAD_ID,
    TASK_PROMPTS,
    format_labels_as_sentence,
)
from .whisper_features import (
    WHISPER_HOP_LENGTH,
    WHISPER_MAX_FRAMES,
    WHISPER_MAX_SAMPLES,
    WHISPER_MEL_BINS,
    WHISPER_MEL_FRAMES,
    WHISPER_SAMPLE_RATE,
    audio_pad_token_count,
    extract_mel,
)

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


def create_omni_processor_whisper(
    tokenizer,
    audio_pad_token_id: int,
    whisper_model_id: str = "openai/whisper-small.en",
    max_audio_samples: int = WHISPER_MAX_SAMPLES,  # 480_000 (30 s @ 16k)
    load_from_nubes: bool = False,
    nubes_gateway: str = "http://c.nubes.sto.navercorp.com:8000/v1",
    nubes_max_workers: int = 12,
    sentence_form_sound_p: float = 0.0,
):
    """Per-row processor producing chatml input_ids + log-mel audio_features.

    Output dict (per row, list-aligned across the batch):
      input_ids       list[int]
      labels          list[int]
      audio_features  Tensor[80, 3000] for audio rows; Tensor[0] placeholder for text-only
      audio_lengths   int (= number of audio_pad_token slots; 0 for text-only)
      modality_ids    list[int] (per-token, sequence length)
    """
    eos_token_id = tokenizer.eos_token_id
    rng = random.Random(20260424)

    _sys_user = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    _end_user = "<|im_end|>\n<|im_start|>assistant\n"
    sys_user_ids = tokenizer.encode(_sys_user, add_special_tokens=False)
    audio_start_ids = tokenizer.encode("<|audio_start|>", add_special_tokens=False)
    audio_end_ids = tokenizer.encode("<|audio_end|>", add_special_tokens=False)
    end_user_ids = tokenizer.encode(_end_user, add_special_tokens=False)

    def _enc(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    def _build_prompt_targets(row: dict) -> tuple[str, str] | None:
        """Same logic as omni_dataset.create_omni_processor; copied so this module
        is self-contained and not coupled to internal helpers."""
        modality = row.get("modality") or "audio_asr"

        if modality == "audio_asr":
            text = row.get("text")
            if not text:
                return None
            return (rng.choice(TASK_PROMPTS["asr"]), text)

        if modality == "audio_emotion":
            choices = row.get("choices") or []
            ans = row.get("answer")
            if not choices or not ans:
                return None
            choices_str = " ".join(choices) if isinstance(choices, list) else str(choices)
            rationale = row.get("rationale")
            target = f"{ans}. {rationale}" if rationale else ans
            stem = rng.choice(TASK_PROMPTS["emotion_classify"])
            return (f"{stem}\nChoices: {choices_str}\nAnswer with the letter.", target)

        if modality == "audio_env_sound":
            # v3 sources (clotho, audiocaps, macs, laion_*, audioset, fsd50k)
            # all carry a `captions` list. Legacy `labels` path kept for
            # esc50 / pre-v3 fsd50k shards.
            caps = row.get("captions") or []
            if caps:
                return (rng.choice(TASK_PROMPTS["sound_caption"]), rng.choice(caps))
            src = row.get("source", "")
            labels = row.get("labels") or []
            if not labels:
                return None
            if src == "fsd50k":
                if sentence_form_sound_p > 0 and rng.random() < sentence_form_sound_p:
                    return (rng.choice(TASK_PROMPTS["sound_describe_multi"]),
                            format_labels_as_sentence(labels, rng, multi=True))
                return (rng.choice(TASK_PROMPTS["sound_classify_multi"]), ", ".join(labels))
            if src == "esc50":
                if sentence_form_sound_p > 0 and rng.random() < sentence_form_sound_p:
                    return (rng.choice(TASK_PROMPTS["sound_describe_single"]),
                            format_labels_as_sentence([labels[0]], rng, multi=False))
                return (rng.choice(TASK_PROMPTS["sound_classify_single"]), str(labels[0]))
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

    _session_holder: list[requests.Session | None] = [None]

    def _get_session() -> requests.Session:
        if _session_holder[0] is None:
            session = requests.Session()
            session.mount("http://", requests.adapters.HTTPAdapter(max_retries=0))
            _session_holder[0] = session
        return _session_holder[0]

    def _download_from_nubes(nubes_path: str) -> bytes | None:
        try:
            session = _get_session()
            resp = session.get(
                f"{nubes_gateway}/{nubes_path}",
                timeout=(NUBES_CONNECT_TIMEOUT, NUBES_READ_TIMEOUT),
            )
            if resp.status_code != 200:
                print(f"[omni-whisper] nubes HTTP {resp.status_code} for {nubes_path}", flush=True)
                return None
            return resp.content
        except Exception as e:
            print(f"[omni-whisper] nubes download error: {e} ({nubes_path})", flush=True)
            return None

    def _rows(examples: dict[str, list[Any]]) -> list[dict[str, Any]]:
        n = max((len(v) for v in examples.values() if isinstance(v, list)), default=0)
        return [
            {k: (v[i] if isinstance(v, list) and i < len(v) else None) for k, v in examples.items()}
            for i in range(n)
        ]

    def _load_waveform(row: dict, audio_bytes: bytes | None) -> torch.Tensor | None:
        """Return mono fp32 waveform [S] at 16 kHz, sliced to [audio_start, audio_end] if present."""
        audio_start = row.get("audio_start")
        audio_end = row.get("audio_end")
        try:
            if audio_bytes is not None:
                return load_audio_chunk(audio_bytes, target_sr=WHISPER_SAMPLE_RATE,
                                        audio_start=audio_start, audio_end=audio_end)
            path = row.get("path") or row.get("audio_path")
            if not path:
                return None
            return load_audio_chunk(path, target_sr=WHISPER_SAMPLE_RATE,
                                    audio_start=audio_start, audio_end=audio_end)
        except Exception as e:
            print(f"[omni-whisper] audio load error: {e} (row={row.get('nubes_path') or row.get('path')})",
                  flush=True)
            return None

    def process_samples(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        all_input_ids: list[list[int]] = []
        all_labels: list[list[int]] = []
        all_audio_features: list[Any] = []
        all_audio_lengths: list[int] = []
        all_modality_ids: list[list[int]] = []

        rows = _rows(examples)
        n = len(rows)
        # Updated as we walk through the batch so the watchdog log can name the
        # row that was active when the deadline fired.
        active_stage = "init"
        active_row_info = ""

        old_handler = signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(PROCESS_BATCH_DEADLINE)
        try:
            # Pre-fetch nubes audio bytes in parallel
            active_stage = "nubes_fetch"
            needs_nubes_idx = [i for i, r in enumerate(rows)
                               if (r.get("modality") or "audio_asr") != "text"
                               and load_from_nubes
                               and r.get("nubes_path")]
            nubes_bytes: dict[int, bytes | None] = {}
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
                            nubes_bytes[i] = fut.result()
                        except Exception as e:
                            print(f"[omni-whisper] future error: {e}", flush=True)
                            nubes_bytes[i] = None
                except FuturesTimeoutError:
                    for fut, (i, p) in futures.items():
                        if not fut.done():
                            nubes_bytes[i] = None
                            fut.cancel()
                            print(
                                f"[omni-whisper] nubes batch deadline {NUBES_BATCH_DEADLINE}s exceeded; "
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

                # ---- (A) waveform → mel ----------------------------------------
                mel = None
                t_audio = 0
                if modality != "text":
                    active_stage = "load_waveform"
                    wav = _load_waveform(row, nubes_bytes.get(i))
                    if wav is None:
                        continue
                    if max_audio_samples is not None and wav.shape[0] > max_audio_samples:
                        # Manifest split should have prevented this, but trim defensively.
                        wav = wav[:max_audio_samples]
                    t_audio = audio_pad_token_count(wav.shape[0])
                    if t_audio == 0:
                        continue
                    active_stage = "extract_mel"
                    mel = extract_mel(wav, model_id=whisper_model_id)  # [80, 3000]

                # ---- (B) tokenize ----------------------------------------------
                active_stage = "tokenize"
                user_suffix_ids = _enc(user_suffix_text)
                target_ids = _enc(target_text)
                if not target_ids:
                    continue

                # ---- (C) chatml assembly ---------------------------------------
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
                mod_id = MODALITY_ID_MAP.get(modality, MODALITY_PAD_ID)
                all_modality_ids.append([mod_id] * len(input_ids))

                if mel is not None:
                    all_audio_features.append(mel)         # [80, 3000]
                    all_audio_lengths.append(t_audio)
                else:
                    # Length-0 placeholder for text-only rows; collator skips these.
                    all_audio_features.append(torch.zeros((WHISPER_MEL_BINS, 0), dtype=torch.float32))
                    all_audio_lengths.append(0)
        except _BatchTimeout:
            _watchdog_fire_count[0] += 1
            print(
                f"[omni-watchdog] FIRE #{_watchdog_fire_count[0]} (whisper) "
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
            "modality_ids": all_modality_ids,
        }

    return process_samples


@dataclass
class WhisperOmniCollator:
    """Collator for Whisper omni pipeline.

    Differences from OmniCollator:
      * audio_features stacked as [N_audio, 80, 3000] mel tensors (no padding needed)
      * everything else (input_ids, labels, attention_mask, modality_ids, FA2 cu_seqlens)
        identical to OmniCollator behavior.
    """

    pad_token_id: int
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: torch.dtype = torch.float32
    block_diag_attn: bool = False

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)

        # Flatten + filter mel tensors across batch.  The packer leaves audio_features as
        # a list-of-lists keyed by knapsack; we collect only the non-empty mel entries.
        mels: list[torch.Tensor] = []
        all_audio_lengths: list[int] = []
        for f in features:
            for mel, t_audio in zip(f["audio_features"], f["audio_lengths"]):
                if t_audio <= 0:
                    continue
                if not isinstance(mel, torch.Tensor):
                    mel = torch.tensor(mel, dtype=torch.float32)
                if mel.shape != (WHISPER_MEL_BINS, WHISPER_MEL_FRAMES):
                    raise ValueError(
                        f"unexpected mel shape {tuple(mel.shape)}; expected "
                        f"({WHISPER_MEL_BINS}, {WHISPER_MEL_FRAMES})"
                    )
                mels.append(mel)
                all_audio_lengths.append(t_audio)

        if mels:
            audio_features = torch.stack(mels)  # [N_audio, 80, 3000]
        else:
            audio_features = torch.zeros((0, WHISPER_MEL_BINS, WHISPER_MEL_FRAMES), dtype=torch.float32)

        audio_lengths = torch.tensor(all_audio_lengths, dtype=torch.long)

        if all("modality_ids" in f for f in features):
            modality_ids = torch.tensor([f["modality_ids"] for f in features], dtype=torch.long)
        else:
            modality_ids = torch.full_like(input_ids, MODALITY_PAD_ID)

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "audio_features": audio_features,        # [N_audio, 80, 3000]
            "audio_lengths": audio_lengths,          # [N_audio]
            "modality_ids": modality_ids,
        }

        if self.block_diag_attn:
            if self.attn_implementation != "flash_attention_2":
                result["attention_mask"] = prepare_4d_attention_mask(attention_mask, self.compute_dtype)
            else:
                non_pad = attention_mask != 0
                result["input_ids"] = input_ids[non_pad].unsqueeze(0)
                result["labels"] = labels[non_pad].unsqueeze(0)
                result["modality_ids"] = modality_ids[non_pad].unsqueeze(0)
                position_ids = batch_group_counter(attention_mask)
                position_ids = position_ids[non_pad].unsqueeze(0)
                result["labels"][position_ids == 0] = IGNORE_INDEX
                result["position_ids"] = position_ids

                flat_pos = position_ids[0]
                boundary_idx = (flat_pos == 0).nonzero(as_tuple=True)[0]
                total_len = flat_pos.numel()
                cu_seqlens = torch.cat(
                    [boundary_idx.to(torch.int32), torch.tensor([total_len], dtype=torch.int32)]
                )
                result["cu_seqlens"] = cu_seqlens

        return result
