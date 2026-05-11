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

> ⚠ **MELD / IEMOCAP contamination warning (Stage-2 LISTEN-mix only)** — Stage-2 의 LISTEN-train 에 MELD-test 881 + MELD-dev 361 audio 가 그대로 포함 → LISTEN-mix 학습 모델 기준 MELD test 수치는 held-out 으로 해석 불가. IEMOCAP 도 동일. 자세한 내용은 [`../stage2/leakage_audit.md §1`](../stage2/leakage_audit.md). **Stage-1 v6 ckpt** 평가는 LISTEN-mix 학습 안 했으므로 clean.
>
> **v6 emotion 평가 룰** (datasets.md § 4): canonical (공식 train/test) split 없는 source 는 통째로 학습 풀에 들어감 (DT/EmoV/RAVDESS/MUStARD), eval 시 self-held-out 또는 cross-corpus 사용. IEMOCAP 만 예외 (학계 관행 leave-session-out: Sessions 1-4 학습, Session 5 eval, 4-class 표준 프로토콜). MELD 는 공식 split 있어 train+dev 학습, test 만 eval. v5 leak-fix (DT 마지막 5% / EmoV-DB Jenie / RAVDESS Actors 21-24 held-out) 는 v6 에서 폐기.

[`eval_source_emotion.py`](../../evaluation/stage2/eval_source_emotion.py)

prompt builder:
```
{QUESTION}
Choices: A) {c1} B) {c2} ... 
Answer with the letter.
```

- `QUESTION` ([line 64](../../evaluation/stage2/eval_source_emotion.py#L64)): `What emotion does the speaker convey?`
- **v6 평가 corpus 는 MELD test 만** ([`load_meld_test()`](../../evaluation/stage2/eval_source_emotion.py#L78)). v5 의 self-held-out eval (DT 마지막 5% / EmoV-DB Jenie / RAVDESS Actors 21-24) 은 leak-fix 폐기와 함께 같이 폐기됨 — 해당 `load_dailytalk_heldout` / `load_emov_jenie` / `load_ravdess_heldout` 함수도 제거.
- **MELD** ([line 67](../../evaluation/stage2/eval_source_emotion.py#L67), 7-class): `["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]`
- letters: `A, B, C, D, E, F, G, H`
- generation: max_new_tokens=96, parse first `[A-H]` letter
### IEMOCAP Session 5 (leave-session-out)

[`eval_iemocap_session5.py`](../../evaluation/stage2/eval_iemocap_session5.py)

prompt builder ([line 130-134](../../evaluation/stage2/eval_iemocap_session5.py#L130-L134)):
```
{QUESTION}
{choices_str}
Answer with the letter.
```

- `QUESTION` ([line 63](../../evaluation/stage2/eval_iemocap_session5.py#L63)): `What is the emotion expressed?`
- **4-class 표준 프로토콜** (학계 관행):
  - `LABEL_MAP` ([line 53-59](../../evaluation/stage2/eval_iemocap_session5.py#L53-L59)): `ang→angry, hap→happy, exc→happy (exc→hap merge), neu→neutral, sad→sad`
  - choices: `["A. angry", "B. happy", "C. neutral", "D. sad"]`
  - fru/sur/fea/dis/oth/xxx 는 학습은 10-class native 로 받았으나 eval 채점 대상 X (1,650 valid utt → 1,241 utt 만 채점 입력, 409 row silently drop)
- generation: max_new_tokens (eval_source_emotion 와 동일), parse first `[A-D]` letter

**v6 룰 예외**: IEMOCAP 만 canonical split 부재인데도 leave-session-out 유지 — 학계 관행 (Sessions 1-4 학습, Session 5 eval). datasets.md § 4 참고.

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

Cross-node share 시: 특정 ckpt × task 의 `predictions.jsonl` 파일을 ad-hoc shared 경로 (`/mnt/tmp/share/<topic>/` 등) 로 cp.
