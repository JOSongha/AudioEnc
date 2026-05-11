# v6 evaluation prompts

각 평가 모듈이 사용한 user-turn prompt 정리. 모든 모듈은 ChatML template ([`_loader.py:406–408`](../../evaluation/stage2/_loader.py#L406)) 으로 system+user wrapping 후 `build_prompt_ids` 가 audio placeholder 토큰을 splice. assistant turn 은 모델이 생성.

> v6 = Stage-1 only (projector training, LLM frozen). LoRA 없음, text 학습 없음. v6 평가 대상 = ASR + sound captioning + sound ontology multi-label + emotion classification. Single-class sound classification (ESC-50) 과 text retention 은 v6 scope 외.
>
> 학습 풀에 안 들어간 source 의 cross-corpus eval 룰은 [`datasets.md § 7`](../setup/datasets.md) leak audit 표 참고.

## ChatML wrapping (모든 task 공통)

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|audio_start|><|audio_pad|>...<|audio_pad|><|audio_end|>
{TASK_PROMPT}<|im_end|>
<|im_start|>assistant
```

- `<|audio_pad|>` 토큰 갯수 = `floor(n_samples / hop)` (encoder 별 hop: DAC 1920, Whisper 320, WavTok encoder 별)
- text-only 평가는 v6 scope 에 없음 → audio block 항상 존재

## Task 별 user prompt

### ASR

| task | module | prompt |
|---|---|---|
| LibriSpeech (test-clean / test-other) | [`eval_librispeech_wer.py:48`](../../evaluation/stage2/eval_librispeech_wer.py#L48) | `Transcribe the audio to text.` |
| ASR external (LibriSpeech / MLS / VoxPopuli / GigaSpeech / CommonVoice) | [`eval_asr_external.py:55,82`](../../evaluation/stage2/eval_asr_external.py#L82) | 위 `ASR_STEM` 그대로 import |

- canonical stem = `TASK_PROMPTS["asr"][0]` 와 동일 ([`eval_librispeech_wer.py:46`](../../evaluation/stage2/eval_librispeech_wer.py#L46))
- generation: greedy, max_new_tokens=128, temperature=0
- 채점: Whisper EnglishTextNormalizer 적용 후 jiwer WER / CER
- v6 in-dist datasets: `librispeech_{clean,other}` (LibriTTS-R train.* 가 speaker/text 공유, test split 은 disjoint), `mls`, `voxpopuli`, `gigaspeech` (XL train 50 % sample, test split disjoint). `commonvoice*` 는 v6 ASR 풀 부재 → cross-corpus.

### Sound captioning

| task | module | prompt |
|---|---|---|
| Clotho captioning | [`eval_clotho_caption.py:48`](../../evaluation/stage2/eval_clotho_caption.py#L48) | `Describe what you hear in the audio.` |

- canonical stem = `TASK_PROMPTS["sound_caption"][0]` 와 동일 ([`eval_clotho_caption.py:46`](../../evaluation/stage2/eval_clotho_caption.py#L46))
- generation: greedy, max_new_tokens=128
- 채점: BLEU-1..4 / CIDEr / METEOR / ROUGE-L / SPICE (`pycocoevalcap` 설치 시; 미설치 시 BLEU-only)
- split: `--split evaluation` (Clotho eval 1 045). dev / val 은 v6 학습 풀, eval 만 held-out.

### Sound ontology multi-label

ontology 기반 multi-label 평가 (학습 / eval 양쪽에서 ontology vocabulary 사용). 두 모듈 공통으로 `--score-mode {greedy, sequence, sentence}` 3 path 제공:

| task | module | dataset / split |
|---|---|---|
| FSD50K eval | [`eval_fsd50k_map.py`](../../evaluation/stage2/eval_fsd50k_map.py) | FSD50K eval (10 231, 200-label ontology). dev 만 학습, eval held-out. |
| AudioSet eval | [`eval_audioset_map.py`](../../evaluation/stage2/eval_audioset_map.py) | AudioSet eval (527-label ontology). bal_train 만 학습, eval / unbal_train held-out. |

stem (두 모듈 line 61-62 / 74-75 동일):
| score-mode | stem | metric |
|---|---|---|
| `greedy` (default) | `List the sound events in this audio, separated by commas.` (= `TASK_PROMPTS["sound_classify_multi"][0]`) | comma-split → vocabulary 매칭 → F1-micro / F1-macro / Jaccard |
| `sequence` | (동일 stem 위에 per-label teacher-forced scoring) | 각 label log-prob → mAP-micro / mAP-macro (leaderboard-comparable) |
| `sentence` | `Describe what you hear in this audio. Mention every distinct sound event.` | free-form caption → substring match against vocabulary → F1 |

- generation: greedy / sentence path 는 `max_new_tokens=64` (greedy) / 128 (sentence). sequence 는 generation 없음 (forward 만).
- ontology 일치성: builder 의 sound_caption pool 이 FSD50K vocabulary.csv / AudioSet ontology.json 과 동일 label string 사용 → eval 시 별도 alignment 없이 vocabulary 매칭.

### Emotion classification

v6 학습 룰 (canonical split 없는 source 는 통째로 학습, IEMOCAP 만 leave-session-out 예외, MELD 는 train+dev 학습 / test eval) 에 맞춘 두 모듈:

#### MELD test

[`eval_source_emotion.py`](../../evaluation/stage2/eval_source_emotion.py)

prompt builder ([line 156](../../evaluation/stage2/eval_source_emotion.py#L156)):
```
{QUESTION}
Choices: A) {c1} B) {c2} ... 
Answer with the letter.
```

- `QUESTION` ([line 64](../../evaluation/stage2/eval_source_emotion.py#L64)): `What emotion does the speaker convey?`
- v6 평가 corpus = MELD test 만 ([`load_meld_test`](../../evaluation/stage2/eval_source_emotion.py#L78)). DT 마지막 5 % / EmoV-DB Jenie / RAVDESS Actors 21-24 self-held-out 은 v5 leak-fix 폐기와 함께 같이 폐기됨 (loader 함수도 제거).
- MELD 7-class ([line 67](../../evaluation/stage2/eval_source_emotion.py#L67)): `["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]`
- letters: `A..G` (7-class), parser 는 `A..H` 까지 허용
- generation: max_new_tokens=96, parse first `[A-H]` letter

#### IEMOCAP Session 5 (leave-session-out, 4-class)

[`eval_iemocap_session5.py`](../../evaluation/stage2/eval_iemocap_session5.py)

prompt builder ([line 130-134](../../evaluation/stage2/eval_iemocap_session5.py#L130-L134)):
```
{QUESTION}
{choices_str}
Answer with the letter.
```

- `QUESTION` ([line 63](../../evaluation/stage2/eval_iemocap_session5.py#L63)): `What is the emotion expressed?`
- 학계 표준 4-class 프로토콜:
  - `LABEL_MAP` ([line 53-59](../../evaluation/stage2/eval_iemocap_session5.py#L53-L59)): `ang→angry, hap→happy, exc→happy (exc→hap merge), neu→neutral, sad→sad`
  - choices: `["A. angry", "B. happy", "C. neutral", "D. sad"]`
  - 학습은 10-class native 로 받지만 (fru/sur/fea/dis/oth 도 신호) eval 채점은 4-class only — Session 5 의 1,650 valid utt → 1,241 utt 만 입력, fru 381 / sur 18 / fea 10 = 409 row silently drop.
- generation: max_new_tokens (eval_source_emotion 와 동일), parse first `[A-D]` letter

### Cross-corpus emotion (외부 corpus, v6 train pool 부재)

[`datasets.md § 7`](../setup/datasets.md) leak audit 의 "외부 corpus" 행 — v6 학습에 안 들어갔으므로 cross-corpus 평가용. 모두 동일한 `What is the emotion expressed?` stem + `Answer with the letter.` 형식.

| corpus | module | classes | 비고 |
|---|---|---|---|
| LISTEN-MCQA (aggregate) | [`eval_listen_mcqa.py`](../../evaluation/stage2/eval_listen_mcqa.py) | row-native (변동) | default stem `What emotion is being expressed in the audio?` ([line 50](../../evaluation/stage2/eval_listen_mcqa.py#L50)); `--use-native-question` 으로 row-native question 사용 가능 |
| LISTEN-official | [`eval_listen_official.py`](../../evaluation/stage2/eval_listen_official.py) | row-native | LISTEN paper Table 2 reproduction. row-native question + `Respond with only the letter (A, B, C, etc.):` 형식, choices random.seed(42) 셔플, max_new_tokens=3, parse `[A-H]`. exp 별 input_mode (text / audio / audio_and_text) 매핑은 [line 89](../../evaluation/stage2/eval_listen_official.py#L89) 참고 |
| MSP-Podcast | [`eval_msp_podcast.py`](../../evaluation/stage2/eval_msp_podcast.py) | 4: neutral, happy, sad, angry | balanced subsample (≤ 400 / class) |
| SAVEE | [`eval_savee.py`](../../evaluation/stage2/eval_savee.py) | 7: anger, disgust, fear, happiness, neutral, sadness, surprise | 4 male British speakers |
| JL-Corpus | [`eval_jl_corpus.py`](../../evaluation/stage2/eval_jl_corpus.py) | 7: angry, anxious, apologetic, assertive, concerned, encouraging, excited | NZ English, secondary emotion 셋 |

LISTEN-MCQA 공통 prompt builder ([line 117](../../evaluation/stage2/eval_listen_mcqa.py#L117)):
```
{stem}
Choices: {choices_str}
Answer with the letter.
```

MSP-Podcast / SAVEE / JL-Corpus 공통 ([각 모듈 line ~99-113](../../evaluation/stage2/eval_msp_podcast.py#L113)):
```
What is the emotion expressed?
{choices_str}
Answer with the letter.
```

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

predictions jsonl entry:
```json
{
  "id": "...",
  "stem": "Transcribe the audio to text.",
  "expected": "ground truth",
  "predicted": "model output",
  "correct": true
}
```

Cross-node share 시: 특정 ckpt × task 의 `predictions.jsonl` 을 ad-hoc 공유 경로 (`/mnt/tmp/share/<topic>/` 등) 로 cp.
