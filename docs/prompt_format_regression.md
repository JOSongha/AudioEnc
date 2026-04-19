# Prompt 포맷 Regression 분석 — 현행 두 trainer 가 수렴하지 않는 근본 원인

**작성일**: 2026-04-15
**상태**: 진단 완료, 재학습 전 제안 정리
**관련 문서**:
- [word_aug_collapse.md](word_aug_collapse.md) — projector collapse 의 1차 가설 (word-aug)
- [word_alignment.md](word_alignment.md) — interleaving 계획
- [dataloader_trials.md](dataloader_trials.md) §36 — Stage 2 17 ckpt WER 측정 결과

## 증상

세 학습 구현체를 비교했을 때:

| trainer | 파일 | 수렴 여부 |
|---|---|---|
| Legacy | [arXiv/scripts/train.py](../arXiv/scripts/train.py) + [model.py](../model.py) `AudioQwen` | ✓ 수렴 |
| 현행 1 | [train_pipeline_override.py](../train_pipeline_override.py) (HF Trainer 기반) | ✗ 수렴 실패, loss plateau ≈ 3.26, eval WER 160% 고착 |
| 현행 2 | [train_pipeline_arrow_torch.py](../train_pipeline_arrow_torch.py) (pure torch + FSDP) | ✗ 수렴 실패 (사용자 보고) |

현행 두 파일은 각각 완전히 다른 엔진 (HF Trainer vs bare torch) 과 파이프라인 구조를 쓰지만 **공통적으로 수렴 실패**. 즉 엔진이나 데이터 파이프라인 자체가 문제가 아니라, 둘이 공유하는 **어떤 설계 요소** 가 원인.

## 가설: **Audio ↔ text 경계를 표현하는 방식의 차이**

### Legacy 포맷 (`model.py:AudioQwen`)

```python
p1 = "Audio:\n"               # tokens: ["Audio", ":", "\n"]
p2 = "\nTranscript:\n"        # tokens: ["\n", "Transcript", ":", "\n"]
p1_ids = tokenizer.encode(p1, add_special_tokens=False)
p2_ids = tokenizer.encode(p2, add_special_tokens=False)
...
inputs_embeds = torch.cat(
    [p1_embeds, audio_embeds, p2_embeds, transcript_embeds],
    dim=1,
)
```

- 총 5-7 개의 **pre-trained Qwen 어휘 토큰** 이 audio 앞뒤를 감싼다.
- "Audio", "Transcript", ":", "\n" 모두 Qwen 이 수천억 토큰 pretraining 에서 이미 본 단어.
- LLM 은 "Transcript: 뒤에 텍스트가 온다" 를 language prior 로 이미 알고 있음.
- 특수 토큰 추가도 없고 `resize_token_embeddings` 도 안 함.

### 현행 포맷 (`train_pipeline_override.py`, `train_pipeline_arrow_torch.py`)

```python
tokenizer.add_special_tokens({"additional_special_tokens": ["<|audio_correspond|>"]})
model.llm.resize_token_embeddings(len(tokenizer))
...
corr_id = tokenizer.convert_tokens_to_ids("<|audio_correspond|>")
input_ids = [audio_pad_id] * T_proj + [corr_id] + text_ids + [eos_id]
```

- **단 하나의 새 특수 토큰** 이 audio 와 text 사이에 들어감.
- 이 토큰은 학습 시작 시 Qwen 의 "미사용 padding 슬롯" (원래 embed 248320 행 중 248077~248319 범위) 을 재활용 → 사전학습된 의미가 전혀 없고 사실상 random-init 와 다름없음.
- audio **앞** 에는 아무 토큰도 없음. LLM 은 "이게 오디오 표현이다" 를 알 길이 없음.

## 왜 이 차이가 수렴 실패로 이어지는가

### 1. LLM language prior 활용도

Legacy 의 `"Audio:\n"` / `"\nTranscript:\n"` 은 Qwen 이 수많은 데이터에서 봤던 **자연어 패턴**. `Transcript:` 뒤에는 텍스트가 오는 게 언어모델에게 자명한 continuation pattern. 그래서 frozen LLM 이 이미 "이 자리는 텍스트를 생성해야 하는 자리" 를 알고 있고, projector 는 그냥 audio → LLM embedding space 로 매핑만 잘하면 됨.

현행의 `<|audio_correspond|>` 는 LLM 이 한 번도 본 적 없는 토큰. 그 자리가 무슨 의미인지, 뒤에 뭐가 와야 하는지 전부 Stage 1 짧은 학습 안에서 배워야 함. **Stage 1 은 LLM frozen + projector 만 학습** 이라 LLM 의 이 토큰에 대한 이해는 늘어나지 않음. 오히려 projector 가 "text-like output" 자체를 만들어내야 하는 불가능한 task.

### 2. Anchor 위치

Legacy 에는 audio **앞** 과 **뒤** 둘 다 anchor 가 있음:
- `"Audio:"` → "이 다음은 오디오 표현이다" (LLM 에게 context switch 신호)
- `"\nTranscript:\n"` → "이제부터 텍스트다" (LLM 에게 generation 시작 신호)

현행에는 audio **뒤** 에만 1 개 (`<|audio_correspond|>`). audio 앞쪽은 무방비. Projector 출력이 BOS 없이 바로 LLM 에 꽂히는 구조 — LLM 입장에선 "어느 문맥 시작인지 모르는 임의 embedding" 으로 보임.

### 3. 초기 단계 gradient 품질

Legacy 의 경우, text prompt 가 있으면 LLM forward 가 "Transcript: 뒤에 텍스트 오는 패턴" 을 인식해서 어느 정도 합리적인 출력 분포를 만들어냄. 이 분포와 실제 transcript 의 차이가 bp 를 통해 projector 로 돌아감. **projector 가 처음부터 의미 있는 gradient 를 받음**.

현행의 경우, audio + corr_id 뒤에서 LLM 이 무엇을 출력해야 하는지 모름. random-ish 분포 출력 → transcript 와의 loss 가 높고 gradient 가 noisy. projector 학습이 매우 느리고 불안정.

### 4. Training loss plateau 의 해석

현재 학습 loss 가 ~3.26 에서 plateau. 이게 의미하는 건:

```
perplexity = e^3.26 ≈ 26
per-token prob ≈ 1/26 = 3.8%
```

unconditional Qwen 의 average per-token loss 가 2-3 범위 정도 (자연어 text 에 대해). 3.26 은 오히려 그보다 **나쁘거나 비슷** — 즉 projector 가 LLM 에 유의미한 audio 조건부 정보를 전혀 못 주고 있는 상태. LLM 이 audio 를 무시하고 그냥 unconditional LM 처럼 작동하고 있는 것.

### 5. Stage 2 hyp 출력 특성 (결정적 증거)

[eval_ckpts 결과](../eval_ckpts/results/s2_outputs_0414_1442/dev-clean/index.html) 를 보면 모든 ckpt 의 hyp 가 audio 와 무관한 generic continuation 으로 고정:

```
REF: he was in a fevered state of mind owing to the blight his wife's action
HYP: to the question of whether or not we should have a government in which all men are equal...
```

이건 LLM 이 "이 자리에서 고확률로 생성할 수 있는 일반적 문장" 을 뱉는 것. 즉 projector output 이 LLM 에 유효한 조건을 못 주고, LLM 은 단순히 " `<|audio_correspond|>` 뒤에 고확률 prior 로 채움" 을 하는 상태.

Legacy 의 `"Transcript:\n"` 이라면 LLM 이 자연스레 "transcript-like 짧은 문장" 을 생성하려고 했을 것. 현행의 corr_id 는 그 anchoring 이 없음.

## 2순위 요인들 (존재하지만 결정적이진 않음)

### word-aug collapse ([word_aug_collapse.md](word_aug_collapse.md))

- word-crop 방식이 projector gradient 를 word clip 에 편향시켜 sentence-level 학습 실패.
- 하지만 train_pipeline_arrow_torch.py 는 word-aug 를 안 쓰는데도 수렴 실패 → word-aug 는 **보조 요인**.

### corr_id 버그 (train_pipeline_override precomputed 경로 한정)

- `make_precomputed_processor_fn` 이 training input_ids 에 corr_id 를 안 넣는 버그 (2026-04-15 발견, [dataloader_trials.md §40](dataloader_trials.md) 에 기록 예정).
- inference 시에는 `_greedy_batch` 가 corr_id 를 넣음 → training ↔ inference 포맷 mismatch.
- 그러나 이 버그는 **inference 만** 망가뜨림. training loss plateau 는 따로 설명해야 함.
- train_pipeline_arrow_torch 에는 이 버그 없음. 그런데도 수렴 실패 → corr 버그도 **보조 요인**.

### 정밀도 / Liger / FSDP 스택 차이

- bf16 projector, Liger monkey-patch, FSDP 섞여있어서 수치 drift 가능성은 있으나 legacy 도 일부 bf16 사용 → 결정적 요인 아님.
- Liger 는 arrow_torch 는 안 씀. 근데 둘 다 수렴 실패 → 공통 원인 아님.

### num_data_splits, streaming shuffle 등

- 데이터 다양성 관련 요인은 수렴 여부가 아니라 학습 속도·안정성에 영향. 근본적 수렴 실패 설명 불가.

## 제안: 포맷 복구 방향 — 세 가지 안 (A / B / C)

### 안 A — legacy base 포맷 그대로 복구 (**채택**)

legacy `model.py:AudioQwen` 의 base 분기 와 동일:

```python
p1 = "Audio:\n"              # → Qwen tokens: ["Audio", ":", "\n"]
p2 = "\nTranscript:\n"       # → Qwen tokens: ["\n", "Transcript", ":", "\n"]

input_ids = p1_ids + [audio_pad]*T_proj + p2_ids + text_ids + [eos]
labels    = [-100]*len(p1_ids) + [-100]*T_proj + [-100]*len(p2_ids) + text_ids + [eos]
```

- `<|audio_correspond|>` 특수 토큰 완전 제거 (vocab 추가 / `resize_token_embeddings` 도 불필요)
- legacy 가 이미 수렴했던 정확한 포맷 → **수렴 재현성 보장**
- 변수 1 개만 변경 (prompt 포맷) → 실패·성공 원인 분리 명확
- interleave 분기는 단어 단위로 `p1 + audio_wi + p2 + text_wi` 를 반복하는 확장 포맷

### 안 B — base + 명시적 task instruction

base 포맷 앞에 task 설명 추가:

```python
p1 = "Transcribe the audio to text.\nAudio:\n"
p2 = "\nTranscript:\n"
```

- legacy base 가 이미 수렴한 상태에서 "instruction 을 넣으면 더 좋을까?" ablation
- 장점: 디버깅 명료성, 다국어 task 확장 여지
- 단점: 추가 보장 없음, token overhead 미세 증가
- **우선순위**: 낮음. A 가 수렴하면 굳이 바꿀 동기가 약함. 본 실험에선 **기각**.

### 안 C — ChatML instruct 포맷

legacy `model.py:AudioQwen` 의 instruct 분기:

```python
p1 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
p2 = "\nTranscribe the audio to text.<|im_end|>\n<|im_start|>assistant\n"
```

- 장점:
  - 명시적 역할 구분 (user = audio 입력 / assistant = transcript 출력)
  - instruction-following 표면적 증가 (instruct 모델 전환 시 특히)
  - 다국어 / task variation (예: "Transcribe in Korean") 으로 확장 가능
- 단점 · 숨은 변수 (**보류 이유**):
  - 현재 LLM 은 `Qwen/Qwen3.5-2B` **base** (SFT 미적용). `<|im_start|>` / `<|im_end|>` 는 vocab 에는 있지만 base pretraining 에서 "turn marker" 로 학습된 건 아님 → instruct 포맷 의도대로 동작 확신 어려움
  - ChatML 을 제대로 활용하려면 실제론 `Qwen/Qwen3.5-2B-Instruct` 로 LLM 교체 필요 → **Stage 1 projector 재학습 필수** (다른 LLM embedding space)
  - 즉 C 는 "LLM 모델 교체 + 포맷 교체 + projector 재학습" 세 변수 동시 변경 → 1 차 실험으로 부적합
- **우선순위**: 중. A 성공 후 별도 ablation 으로 시도. 두 단계로 분리:
  - C-i. Qwen3.5-2B **base** 유지 + ChatML prompt — "base 가 ChatML 을 어느 정도 처리하는가"
  - C-ii. Qwen3.5-2B-**Instruct** 교체 + ChatML — 진짜 instruct 파이프라인

### 결정 프로세스 (2026-04-15 대화 요약)

1. 수렴 실패 원인 후보: word-aug collapse, corr_id training/inference mismatch, prompt 포맷 regression.
2. `train_pipeline_arrow_torch.py` (word-aug / corr 버그 둘 다 없음) 에서도 수렴 실패 확인 → 두 요인 모두 **충분조건 아님**. 공통 요인은 **prompt 포맷**.
3. legacy 와 현행 두 파일의 diff 축소 → 유일하게 남는 공통 차이 = single `<|audio_correspond|>` token vs multi-token text prompt.
4. 안 3 개 비교:
   - A: 변수 최소 변경, legacy 재현성 보장 → **1 차 실험으로 채택**
   - B: 효과 불확실, legacy 가 없이도 수렴했던 점으로 **기각**
   - C: 숨은 변수 과다, 선후 구분 필요 → **A 성공 후 ablation 에 등록**
5. 단계적 접근: A 로 먼저 재패킹 → 학습 수렴 확인 → C-i / C-ii 분리 실험.

## 실험 계획 (갱신)

### Phase 1: 안 A 재패킹 + 학습 (현재 진행 예정)

- `half_inlv_pack_arrow.py` 의 `_emit_sentence` / `_emit_interleave` 를 legacy base 포맷으로 변경 (완료)
- `train_pipeline_override.py:make_precomputed_processor_fn._emit` 도 동일 포맷 적용 (완료)
- 4 worker parallel + 8 shards/rank equal-split → 약 3-4 시간 재패킹
- 학습: 동일 cfg (Qwen3.5-2B base, LoRA r=16, FSDP Stage 2) 로 재시작
- 기준: loss plateau 3.26 이하로 내려가는가, WER 이 legacy 수준 (< 100%) 에 근접하는가

### Phase 2: A 가 수렴하면 — ablation 으로 다음 요인 격리

1. **interleave 의 기여도** — half_inlv 의 interleave bin 을 껐다 켜서 (sentence-only vs 50/50) WER 비교
2. **C-i**: Qwen3.5-2B base + ChatML prompt — 같은 base 모델에서 prompt 포맷만 바꿈
3. **C-ii**: Qwen3.5-2B-Instruct + ChatML — 진짜 instruct pipeline, Stage 1 부터 재학습
4. **B**: base + task instruction prefix (우선순위 최저, A/C 보다 잠재력 낮음)

### Phase 3: A 가 실패하면 — 더 깊은 원인 탐색

prompt 포맷이 전부가 아니었다는 뜻. 다음 후보:
- FSDP + bf16 + LoRA 스택에서의 수치 drift
- Liger kernel manual monkey-patch 의 subtle bug
- projector 초기화 차이 (legacy 가 fp32 projector Stage 1 에서 학습했는지 확인 필요)
- gradient_accumulation × per_device_batch × n_gpu 가 legacy 와 다른 경우 effective batch 크기 차이

이 경우 engineering 관점에서 `train_pipeline_arrow_torch.py` (pure torch, 단순 구조) 를 기반으로 변수 1 개씩 추가하며 re-bisect 하는 게 빠를 수도 있음.

## 실험 계획

### 1단계: 현재 half_inlv pack run 완료 + 학습 (진행 중)

- 현재 돌고 있는 half_inlv 재패킹 (corr 버그 fix + word-crop 회피 + equal-shard resplit) 결과로 학습.
- 만약 이것도 legacy 보다 나쁘게 수렴하면 **prompt 포맷이 결정적 원인** 이라는 증거가 확정됨.
- 목적: legacy 와 비교할 만한 control 확보.

### 2단계: 포맷 복구 실험

1단계 결과 보고 결정:
- legacy 와 유사하게 수렴하면 → corr 버그 + word-aug 가 주 원인. 포맷 논의 보류.
- 여전히 수렴 실패하면 → 안 X 로 전환해서 재학습.

### 3단계: 안 Z (interleaving + legacy prompt) 검증

1단계가 실패하고 2단계 (안 X) 가 성공하면, interleaving 의 추가 supervision 효과를 1:1 ablation 으로 재확인 (interleaving + legacy prompt vs sentence-only + legacy prompt).

## 기록용 요약

- 현행 두 trainer 모두 **legacy 의 multi-token text prompt 를 1 개 새 special token 으로 교체** 하면서 수렴 실패. 원인은 LLM 의 language prior 를 못 쓰게 된 것.
- word-aug collapse 와 corr_id 버그는 **보조 요인** — 둘 다 없는 train_pipeline_arrow_torch 도 수렴 안 되는 것으로 증명.
- 가장 확실한 fix 는 "Audio:\n" / "\nTranscript:\n" 텍스트 prompt 복구.
- 현재 돌고 있는 half_inlv run 은 이 가설의 control 역할 — 결과 나오면 본격적으로 포맷 복구 방향 결정.

## 2026-04-15 코드 수정 요약 (안 A 적용)

안 A 를 최종 확정하고 `train_pipeline_override.py` 의 **학습 + 추론 경로 전부** 를 legacy p1/p2 포맷으로 통일. 기존 ckpt 재사용은 포기 (vocab 변경 포함).

### 수정 대상

| 경로 | 함수 / 위치 | 변경 내용 |
|---|---|---|
| [train_pipeline_override.py:205-210](../train_pipeline_override.py#L205) | `AudioQwen` docstring | 시퀀스 설명 `[audio_pad] + <|audio_correspond|>` → `p1 + [audio_pad] + p2` |
| [train_pipeline_override.py:308-312](../train_pipeline_override.py#L308) | `AudioQwen.__init__` | `add_special_tokens({"<|audio_correspond|>"})` / `resize_token_embeddings` **삭제**. vocab = Qwen 원본 그대로 |
| [train_pipeline_override.py:743-781](../train_pipeline_override.py#L743) | `make_processor_fn._build_one` (raw audio 경로) | `audio_correspond_id` 제거, `p1_ids + [audio_pad]*t_audio + p2_ids + text_ids + eos` |
| [train_pipeline_override.py:1272-1273](../train_pipeline_override.py#L1272) | `make_precomputed_processor_fn._emit` | 이미 4/15 오전에 안 A 로 수정됨 (재확인) |
| [train_pipeline_override.py:1753-1800](../train_pipeline_override.py#L1753) | `_greedy_batch` (WER eval) | `corr_embeds` 제거, `inputs_embeds = cat([p1_embeds, audio_embeds, p2_embeds])` |
| [train_pipeline_override.py:1854-1886](../train_pipeline_override.py#L1854) | `evaluate_val_loss` (raw audio val loss) | `corr_id` 슬롯을 `p1_ids`/`p2_ids` 로 교체, labels IGNORE 범위 동기화 |
| [train_pipeline_override.py:1928-1983](../train_pipeline_override.py#L1928) | `evaluate_val_loss_precomputed` | 이전 시퀀스는 prompt 가 **아예 없었음** (`[audio_pad]*T_proj + text + eos`) → training mismatch + prompt 부재 이중 버그. `p1 + [audio_pad]*T_proj + p2 + text + eos` 로 수정 |

### 남은 `<|audio_correspond|>` 참조

- [L1267 주석](../train_pipeline_override.py#L1267): §41 설명용 comment. 코드 경로엔 없음.

`eval_ckpts/eval_all_ckpts.py::transcribe_train_format` 은 별도 파일. 새 ckpt 로 재eval 하기 전에 동일하게 p1/p2 로 갱신 필요 (TODO).

### 패킹 시 pad/EOS 순서 — 중요 불변식

[collator `_pack_rows`](../train_pipeline_override.py#L945-L950) 가 bin 하나를 만드는 방식:

```
bin = [p1+audio+p2+text+EOS]  ← sample 1
    + [p1+audio+p2+text+EOS]  ← sample 2
    + ...
    + [p1+audio+p2+text+EOS]  ← 마지막 sample
    + pad pad pad ... pad     ← cutoff_len (16384) 까지 right-pad
```

핵심:
- **샘플 사이에는 pad 가 없다** — 다음 샘플이 바로 붙음. block-diagonal `attention_mask` 가 경계를 분리하므로 sample 간 attention leak 없음 ([L937-L938](../train_pipeline_override.py#L937): `attention_mask.extend([seq_idx+1] * len(ids))`).
- **pad 구간은 bin 의 맨 끝에만** 존재. `attn_mask=0`, `labels=IGNORE_INDEX`.
- 따라서 **모든 EOS 는 pad 보다 먼저** 나온다. 각 sample 의 마지막 토큰 = EOS, 그 다음 sample 이 있으면 바로 `p1`, 없으면 pad.
- `pad_token_id` 는 `tokenizer.pad_token_id` (Qwen 의 경우 `eos` 와 같은 id 가 될 수 있으나 attn_mask=0 이라 loss 에 영향 없음).

비-packed 경로 (`evaluate_val_loss`) 도 right-pad + EOS 먼저 규칙을 똑같이 따름 ([L1909-L1915](../train_pipeline_override.py#L1909)):
```
seq = p1 + [audio_pad]*t_audio + p2 + text_ids + EOS + pad*pad_len
```

`_greedy_batch` 는 text/EOS 없이 `[p1_embeds, audio_embeds, p2_embeds]` 만 입력하고 `generate()` 가 transcript→EOS 를 생성 (pad 없음, `attn_mask=1` 전부).

### 수정 후 검증 TODO

- [ ] Phase 3 pack 완료 후 첫 학습 step → Stage 1 loss 가 legacy baseline 수준 (< 3.26 plateau) 으로 내려가는지
- [ ] 1K step 시점 rank-0 sample eval 출력이 audio-conditioned 응답인지 (이전엔 generic LM prior)
- [ ] `eval_ckpts/eval_all_ckpts.py::transcribe_train_format` 도 p1/p2 로 맞춰 신규 ckpt eval 가능하게

### Mixed pack 의 데이터셋 혼합 구조 (재확인)

사용자 확인 요청 ("한 팩에 여러 데이터셋에서의 샘플이 들어가는 거 맞지?") 에 대한 응답을 불변식으로 기록:

[half_inlv_pack_arrow.py:636-757](../precompute/half_inlv_pack_arrow.py#L636-L757) 의 Phase 순서:

1. **Phase 1 (accumulate)**: rank 내에서 `ls100 → ls360 → ls500 → mls → gs → vp` 순차 (또는 `--num-workers N` 병렬) 로 processor 를 통과해 하나의 `tmp_rank{N}.arrow` 에 append. 이 시점엔 tmp 파일 내부가 데이터셋별로 덩어리 상태.
2. **Phase 2 (shuffle)**: `total_samples` 개 전역 인덱스를 Fisher-Yates shuffle ([L720-L726](../precompute/half_inlv_pack_arrow.py#L720-L726), seed = `42 + rank`). **이 시점에 데이터셋 경계가 완전히 해체** 됨.
3. **Phase 3 (pack)**: shuffled 순서대로 `packing_bucket_size=200` 개씩 꺼내 greedy knapsack 으로 `cutoff_len=16384` bin 에 채움. bucket 하나가 이미 6 개 데이터셋의 샘플을 섞어서 갖고 있으므로 **각 bin = 6 개 데이터셋의 샘플이 대략 비율대로 혼합**.
4. **Phase 4 (resplit)**: rank 당 하나의 `rank{N}_pack.arrow` intermediate → bin 수 기준 `--shards-per-rank 8` 등분. 각 shard 는 이미 cross-dataset mix 된 상태 그대로 slice.
5. **Rebalance (rank 간)**: `ls100/ls360/ls500/vp` 는 단일 shard (rank 0 만 데이터 보유), `mls/gs` 는 rank 별 shard → rank 구성이 조금 다름. 마지막에 `--rebalance` 로 rank 간 총 bin 수를 ±1 로 맞춤.

즉 학습 step 하나가 보는 batch (bin 들의 모음) 는 언제나 cross-dataset mixed 상태. step 당 gradient 가 특정 도메인으로 편향되지 않음.

### Stage 1 run 1 결과 요약 (2026-04-15)

**환경**: `run.sh --encoder fb_dacvae` (6 datasets, --fsdp, --liger, 8 GPU), 8 splits × 1 epoch, bs 12, LR 0.0002 constant.

**Split-end loss**: 3.790 → 3.696 → 3.711 (noise) → 3.651 → 3.650 → 3.630 → 3.632 → **3.612**

- Plateau 는 아니지만 split 당 Δ ≈ -0.02 로 매우 얕은 하강.
- Grad norm 0.02 대까지 감소 — update 거의 없음.
- 이전 `<|audio_correspond|>` run 의 training plateau 3.26 보다 **높은 자리에서 수렴**.
- **직접 비교 불가 주의**: 현재 run 은 50% word-interleave bin 포함 → interleave 는 word 경계마다 p1/p2 반복이라 intrinsic loss 가 sentence 보다 높음. 두 run 의 training loss 는 1:1 비교 의미 없음. WER 이 유일한 실질 지표.

**Stage 1 run 1 총 optimizer step = 138** (wandb 기준). 내 tee log count 131 은 부정확 (tqdm `\r` 덮어쓰기로 일부 loss line 이 다른 bar 와 섞여 `'loss'` 문자열 누락). wandb 가 authoritative.

### Stage 1 resume — §42 extension (2026-04-16)

Stage 1 을 1 epoch 더 돌려 projector 를 fully converge 시키고 싶다는 요구. 핵심 변경 3 가지:

1. **[`CumulativeWandbCallback`](../train_pipeline_override.py#L2020) 확장**: `initial_step_offset` / `initial_epoch_offset` / `cfg` 파라미터. split 경계에서 Trainer 가 재생성돼도 offset 은 `cfg["_wandb_step_offset"]` 로 persist.
2. **[`--stage1-resume-step N`](../train_pipeline_override.py#L2496) CLI arg**: 최신 `s1_outputs_*/s1_proj.pt` 자동 preload + wandb step_offset = N.
3. **[Stage 1 outer loop 재구성](../train_pipeline_override.py#L2665)**: `stage1_epochs` 바깥 loop + 매 epoch 마다 shuffled split order (Stage 2 와 동일 RNG, `run_id` 기반 deterministic, rank 공통). 기존 sequential 는 Stage 1 가 1 epoch 라서 셔플 불필요했던 가정 — 2+ epoch 돌릴 땐 shard 단위 셔플로 다양성 확보 필요.

**실행**:
```bash
bash run.sh --encoder fb_dacvae --stage 1 --stage1-resume-step 138 --stage1-epochs 1
```

- 새 `run_id` → 새 wandb run (기존 Stage 2 wandb 히스토리 보존)
- Wandb x축이 step 139+ 부터 시작해 시각적으로 run 1 과 연속
- 저장: `{cache}/fb_dacvae/s1_outputs_{new_run_id}/s1_proj_split{1..8}.pt` + `s1_proj.pt`

### eval_ckpts 업데이트 (2026-04-16)

[eval_ckpts/eval_all_ckpts.py](../eval_ckpts/eval_all_ckpts.py) 를 p1/p2 포맷에 맞춰 업데이트:

- `transcribe_train_format`: `[audio_embeds, corr_embeds]` → `[p1_embeds, audio_embeds, p2_embeds]`
- `load_eval_model`: `add_special_tokens(<|audio_correspond|>)` + `resize_token_embeddings` 삭제 (§42 에서 vocab 변경 안 함)
- 이걸로 Stage 2 run 1 의 16 개 ckpt (fe1×8 + fe2×8) WER 평가 가능

---

## S1+S2 run 1 결과 보고 (2026-04-16)

### Eval 설정

- Pack: `packed_half_inlv_16384` (50% sentence + 50% word-interleave, §39)
- Stage 1: 1 epoch, 8 splits sequential, 138 optimizer steps, LR=0.0002 constant, ended loss **3.612**
- Stage 2: 2 epochs × 8 splits shuffled, total 267 steps, FSDP + LoRA(r=16, α=32, q/k/v/o_proj, dropout=0.1)
- Dataset: 6 datasets (ls100/ls360/ls500/mls/gs/vp) × rank-balanced 7,476 bins/rank
- Evaluation: dev-clean 200 samples (first 200 indices), greedy decoding, `repetition_penalty=1.1`, `no_repeat_ngram=4`, `max_new_tokens ≤ audio_sec × 7 (min 32, cap 256)`

### WER 결과 (200 samples)

[results CSV](../eval_ckpts/results/s1_s2_0415_1631/s1_s2_wer.csv)

| stage / split | name | WER% |
|---|---|---|
| S1 split1 | s1_proj_split1.pt | 126.30 |
| **S1 split2 (best)** | s1_proj_split2.pt | **118.13** |
| S1 split3 | s1_proj_split3.pt | 120.93 |
| S1 split4 | s1_proj_split4.pt | 124.81 |
| S1 split5 (worst) | s1_proj_split5.pt | 128.25 |
| S1 split6 | s1_proj_split6.pt | 126.81 |
| S1 split7 | s1_proj_split7.pt | 123.09 |
| S1 split8 (final) | s1_proj_split8.pt | 123.45 |
| S2 fe1 range | 8 ckpts | **125.14 – 130.84** |
| S2 fe2 range | 8 ckpts | **133.13 – 135.18** |

### 핵심 관찰

1. **모든 WER > 100%** — hyp 가 ref 보다 길고, 단어 거의 전부 틀림 (insertion + substitution 지배). Hallucination.
2. **Stage 2 fe1→fe2 악화** (avg 127% → 134%). LoRA epoch 2 가 오히려 성능 떨어뜨림.
3. **Stage 1 final (123%) > Stage 2 best (125%)** — LoRA fine-tuning 이 projector-only 보다 나쁨.
4. **이전 `<|audio_correspond|>` run (160%) 대비 25-40%p 개선**. §41 fix (p1/p2 legacy prompt) 효과 있음. 그러나 실용 수준 아님.

### 수학적 진단: Perplexity 관점에서 왜 WER > 100%

Stage 1 loss 3.612 → **per-token perplexity ≈ 37**. 건강한 ASR 은 perplexity 1.5-3 (12-25배 혼란도).

15-token 문장 teacher-forced 완전 정답 확률 = `(0.027)^15 ≈ 10^-23`. Autoregressive generate 시:

- Step 1 error rate ~50% 가정 (낙관)
- Step 2: context 가 training 분포 이탈 → OOD 추론, accuracy 추가 하락
- **Error cascade** → 몇 step 뒤 완전 OOD 영역, LM prior 가 지배 → hallucination

즉 **training loss 3.6 이면 coherent generation 이 수학적으로 불가능**. Loss 2 이하로 낮춰야 의미있는 WER 가능.

### 가설 3가지

#### 가설 A: Half-interleave 포맷의 train/test mismatch (의심 1순위)

Training bins 50% 가 interleave: `[p1 + audio_w0 + p2 + text_w0][p1 + audio_w1 + p2 + text_w1]...[EOS]`.

모델이 학습한 전이 확률:
- `P("p1" | ...text_word_i)` ≈ 0.85-0.9 (interleave 에서 text 뒤에 다시 p1 반복)
- `P(EOS | ...)` ≈ 0.1-0.15

Inference 는 항상 sentence 포맷 (`p1 + audio_full + p2 → text`). 하지만 모델의 **학습된 prior 는 EOS 대신 "Audio:" 재발화**.

→ hyp 가 ref 뒤에 literal `Audio:` 토큰 + 계속 이어지는 hallucination 으로 채워져 길이 폭증 + 단어 엉킴.

**검증 방법**: full-dev eval 의 per-ckpt JSON 에서 HYP 에 `Audio` / `Transcript` 텍스트 포함 여부.

#### 가설 B: Projector 구조 / 용량 한계

Projector: Conv1d(128→2048, stride=2, k=5) × 2 + Conv1d(1×1) + LayerNorm. ~26.5M params.

Local receptive field (230 ms window) 로는 word-level (500 ms+) global context 부족. Stage 1 loss plateau 3.6 이 이 구조적 한계일 수도.

#### 가설 C: LoRA 설정이 약한 신호에 overfit

- r=16, α=32 (scaling 2.0 — 공격적)
- target_modules = **모든 attention projection** (q/k/v/o_proj)
- Projector 가 약한 audio 신호를 제공 → LoRA 가 이 불완전 신호에 overfit
- 2번째 epoch 에서 악화 (fe1 127 → fe2 134)

### 의사결정: **Sentence-only 재패킹 먼저** (가설 A 검증)

가설 A 가 확인되면 재패킹 한 번으로 대폭 개선 가능성. 가설 B/C 는 구조 변경 필요 (오래 걸림).

**실행**:
1. `half_inlv_pack_arrow.py` 에 `--sentence-only` 플래그 추가 (hash_split 결과 무시, 항상 sentence bin)
2. 출력: `packed_sentence_only_16384/` (기존 `packed_half_inlv_16384/` 보존)
3. 같은 config (LR, LoRA, epochs) 로 Stage 1+2 재학습
4. Full dev-clean eval 비교

**기대**: 
- 가설 A true → WER 80-100% 수준 가능
- 가설 A false (여전히 120%+) → 가설 B/C 로 우선순위 이동 (projector 구조 or LoRA config)

### Stage 1 resume 진행 (2026-04-16 03:44 ~ 06:45 예상)

추가 1 epoch (shuffled splits), wandb step offset 138. Loss trajectory:
- Resume start: 3.611 (projector preload 확인)
- Split 1 (sh첫)→6 end: 3.790 → 3.596 → 3.594 → 3.584 → 3.584 → 3.570 → (split 7 진행 중)
- 총 Δ ≈ -0.045 from resume start, 평균 -0.01/split

한 epoch 더 돌아도 plateau 3.5 정도 예상. WER 결정적 개선은 기대 어려움.

그러나 완료 후 eval 해서 현재 가설 강화/반증 자료로 활용.
