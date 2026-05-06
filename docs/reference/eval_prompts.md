# Stage-2 evaluation prompts

각 평가 모듈이 사용한 user-turn prompt 정리. 모든 모듈은 ChatML template ([`_loader.py:406–408`](../../evaluation/stage2/_loader.py#L406)) 으로 system+user wrapping 후 `build_prompt_ids` 가 audio placeholder 토큰을 splice. assistant turn 은 모델이 생성.

## ChatML wrapping (모든 task 공통)

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|audio_start|><|audio_pad|>...<|audio_pad|><|audio_end|>
{TASK_PROMPT}<|im_end|>
<|im_start|>assistant
```

- `<|audio_pad|>` 토큰 갯수 = `floor(n_samples / hop)` (encoder별 hop 다름: DAC 1920, Whisper 320, WavTok encoder별)
- text-only experiment 는 `<|audio_start|>...<|audio_end|>` 블록 생략

## Task별 user prompt

각 task 의 코드 위치 + `{TASK_PROMPT}` 자리에 들어가는 문자열.

### ASR

| task | module | prompt |
|---|---|---|
| LibriSpeech (test-clean / test-other) | [`eval_librispeech_wer.py:48`](../../evaluation/stage2/eval_librispeech_wer.py#L48) | `Transcribe the audio to text.` |

generation: greedy, max_new_tokens=128, temperature=0. WER/CER computed after Whisper-style English text normalization.

### Sound classification & captioning

| task | module | prompt |
|---|---|---|
| ESC-50 5-fold | [`eval_esc50_acc.py:48`](../../evaluation/stage2/eval_esc50_acc.py#L48) | `Classify this sound.` |
| FSD50K eval (greedy) | [`eval_fsd50k_map.py:73`](../../evaluation/stage2/eval_fsd50k_map.py#L73) | `List the sound events in this audio, separated by commas.` |
| FSD50K seq mode | (same module, `--score-mode sequence`) | `List the sound events in this audio, separated by commas.` (per-label scoring; same stem) |
| AudioSet eval (greedy) | [`eval_audioset_map.py:61`](../../evaluation/stage2/eval_audioset_map.py#L61) | `List the sound events in this audio, separated by commas.` |
| AudioSet seq mode | (same module, `--score-mode sequence`) | same stem |
| Clotho captioning | [`eval_clotho_caption.py:48`](../../evaluation/stage2/eval_clotho_caption.py#L48) | `Describe what you hear in the audio.` |

generation:
- ESC-50: greedy, max_new_tokens=8 (label-only)
- FSD50K / AudioSet greedy: max_new_tokens=64, comma-split parsing → multi-label set match against vocabulary
- Sequence mode: `score_labels_teacher_forced` ranks each candidate label by next-token logprob
- Clotho: max_new_tokens=128, BLEU/CIDEr/METEOR/ROUGE-L/SPICE on N-best caption

### Emotion classification

> ⚠ **MELD / IEMOCAP contamination warning** — LISTEN-train 에 MELD-test 881 audio + MELD-dev 361 audio 가 그대로 포함되어 있어 (cross-corpus 평가용 LISTEN-train 으로 학습한 모델 기준), MELD test split 으로 측정한 수치는 held-out 으로 해석할 수 없음. IEMOCAP 도 동일 패턴. 자세한 내용은 [`../stage2/leakage_audit.md §1`](../stage2/leakage_audit.md). clean held-out emotion 평가는 RAVDESS / EmoV-DB / DailyTalk + MMAU-speech 사용.

[`eval_source_emotion.py`](../../evaluation/stage2/eval_source_emotion.py)

prompt builder ([line 187](../../evaluation/stage2/eval_source_emotion.py#L187)):
```
{QUESTION}
Choices: A) {c1} B) {c2} ... 
Answer with the letter.
```

- `QUESTION` (line 55): `What emotion does the speaker convey?`
- 각 corpus 별 emotion list:
  - **MELD** (lines 58, 7-class): `["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]`
  - **DailyTalk** (lines 59–60, 7-class): `["no emotion", "happiness", "sadness", "anger", "surprise", "fear", "disgust"]`
  - **EmoV-DB Jenie** (line 61, 5-class): `["amused", "angry", "disgusted", "neutral", "sleepy"]`
  - **RAVDESS** (lines 62–63, 8-class): `["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprise"]`
- letters: `A, B, C, D, E, F, G, H`
- generation: max_new_tokens=96, parse first `[A-H]` letter

### LISTEN-MCQA (LISTEN benchmark MCQA aggregate)

[`eval_listen_mcqa.py:50`](../../evaluation/stage2/eval_listen_mcqa.py#L50)

- Default stem: `What emotion is being expressed in the audio?`
- option `--use-native-question` 로 dataset-native question 으로 교체 가능
- LISTEN-test parquet 의 row마다 native question + choices 사용 가능

### LISTEN-official (LISTEN paper Table 2 reproduction)

> ⚠ **Type 4 (paralinguistic) caveat** — type-4 row 들은 LISTEN-train 에도 포함되어 있어 held-out 평가가 아님 (LISTEN-test parquet 에 type-4 가 빠져 있어 train shard-2 의 975 row 를 끌어옴). Type 1/2/3 만 진정한 held-out.


[`eval_listen_official.py:187–198`](../../evaluation/stage2/eval_listen_official.py#L187)

prompt builder:
```
{question}

A. {choice_A}
B. {choice_B}
...

Respond with only the letter (A, B, C, etc.):
```

- `question` 은 LISTEN parquet 의 row-native question (각 sample별 다름)
- input_mode `text` / `audio_and_text` 시 `Transcription: {tx}\n\n` 가 prefix 로 추가
- audio mode 는 transcription 안 들어감 (audio-only test)
- choices 는 row 별 randomized (random.seed(42))
- generation: max_new_tokens=3 (letter-only)
- parse: first `[A-H]` letter (uppercase)

experiment-별 input_mode 매핑 ([line 89](../../evaluation/stage2/eval_listen_official.py#L89)):
| exp | input_mode | parquet filter |
|---|---|---|
| 1_text | text | type 1 |
| 1_audio | audio | type 1 |
| 1_audio_and_text | audio_and_text | type 1 |
| 2A | text | type 2A |
| 2B | audio | type 2B |
| 2C | audio_and_text | type 2B |
| 3A | text | type 3A |
| 3B | audio | type 3B |
| 3C | audio_and_text | type 3B |
| 4 (paralinguistic) | audio | type 4 |

### Text retention (commonsense QA, no audio)

[`eval_text_retention.py:177–181`](../../evaluation/stage2/eval_text_retention.py#L177)

prompt builder:
```
{question}
Choices: {c1} {c2} ... 
Answer with the letter.
```

- 각 row 의 native question + choices (이미 letter prefix 포함)
- 6 benchmarks: HellaSwag, WinoGrande, BoolQ, ARC-Easy, ARC-Challenge, COPA
- `t_audio = 0` (no audio_pad block in ChatML)
- generation: max_new_tokens=8, parse first `[A-F]`

---

## Generation 공통 설정

- `do_sample=False` (greedy)
- `temperature=0`
- `num_beams=1`
- `pad_token_id=tokenizer.eos_token_id`
- `use_cache=True` (default; `--no-cache` 시 KV cache off, ground-truth 비교용)

## Inference 결과 위치

각 ckpt × task 별:
```
/mnt/tmp/results/<encoder_run_dir>/<eval_task>/checkpoint-<step>/
  ├── summary.json          # aggregated metrics
  ├── summary_<exp>.json    # per-experiment (LISTEN-official only)
  └── predictions_<exp>.jsonl   # per-sample inference 결과 (정답 + 예측 + raw text)
```

predictions jsonl 의 entry:
```json
{
  "id": "...",
  "stem": "Transcribe the audio to text.",
  "expected": "ground truth",
  "predicted": "model output",
  "correct": true,
  ...
}
```

Cross-node share 시: 특정 ckpt × task 의 `predictions.jsonl` 파일을 [`docs/stage1/whisper/ckpt12k/`](../stage1/whisper/ckpt12k/) 등 shared 경로로 cp.
