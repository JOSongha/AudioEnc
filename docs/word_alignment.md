# Word-Level Alignment

작성: 2026-04-06  
업데이트: 2026-04-08

## 진행 현황 (2026-04-08)

### Alignment (per-utt JSON 생성)

| 데이터셋 | Split | 완료 / 전체 | 상태 |
|---------|-------|------------|------|
| LibriSpeech | dev-clean | 2,703 / 2,703 | ✅ 완료 |
| LibriSpeech | train-clean-100 | 28,539 / 28,539 | ✅ 완료 |
| LibriSpeech | train-clean-360 | 104,014 / 104,014 | ✅ 완료 |
| LibriSpeech | train-other-500 | 148,688 / 148,688 | ✅ 완료 |
| MLS | train | 2,420,047 / 2,420,047 | ✅ 완료 |
| GigaSpeech | train | 8,282,987 / 8,282,988 | ✅ 완료 (1개 미처리) |
| VoxPopuli | train | 182,482 / 182,482 | ✅ 완료 |

**GigaSpeech 1개 누락**: `POD0000005964_S0000139` — HF 소스에 `bytes: b''` (빈 오디오). 원본 데이터 문제로 수정 불가.

### Merge (JSONL + Arrow)

출력 경로: `/mnt/tmp/cache/word_alignments_merged/{dataset}/{split}.jsonl|.arrow`

| 데이터셋 | Split | JSONL | Arrow |
|---------|-------|-------|-------|
| LibriSpeech | dev-clean | ✅ | ✅ |
| LibriSpeech | train-clean-100 | ✅ | ✅ |
| LibriSpeech | train-clean-360 | ✅ | ✅ |
| LibriSpeech | train-other-500 | ✅ | ✅ |
| MLS | train | ✅ | ✅ |
| GigaSpeech | train | ✅ | ✅ |
| VoxPopuli | train | ✅ | ✅ |

---

## 목표

전체 학습 데이터셋 각 발화에 대해 **word-level timestamp** (start, end, score)를  
사전 생성하여 JSON으로 저장.

---

## 데이터셋별 alignment 전략

| 데이터셋 | 모델 | 이유 |
|---------|------|------|
| **LibriSpeech** | wav2vec2-large CTC forced alignment | 동일 도메인(clean) → 정확도 최고, 속도 빠름 |
| **MLS / GigaSpeech / VoxPopuli** | **Qwen3-ForcedAligner-0.6B** | 다양한 도메인·억양 robust, 다국어 지원 |

### 도구 비교

| | wav2vec2-large CTC | Qwen3-ForcedAligner-0.6B |
|---|---|---|
| 아키텍처 | CTC + Viterbi (sequential) | Non-Autoregressive (parallel) |
| 파라미터 | ~300M | 600M |
| 언어 | 영어 전용 | 11개 언어 |
| 학습 도메인 | LibriSpeech (clean) | 다양 (robust) |
| 정렬 오차 | ~10ms (clean speech) | 32.4ms (평균, diverse) |
| 속도 | 빠름 | 중간 |

---

## 1. LibriSpeech — wav2vec2 CTC Forced Alignment

### 파이프라인

```
alignment/
  build_manifest.py      # LibriSpeech 전체 발화 목록 → manifest.jsonl + shard_N.jsonl
  align_worker.py        # 단일 GPU 워커 (shard 하나 처리, ThreadPoolExecutor prefetch)
  run_align.sh           # 다중 GPU × 다중 워커 런처
  merge_alignments.py    # 완료 후 per-utt JSON → per-split JSONL + 검증
```

### 병렬화

- **8 GPU × 8 workers/GPU = 64 프로세스**
- GPU 당 VRAM: ~3.2GB × 8 = ~25GB (A100 80GB의 30%)
- rank 0 먼저 모델 로드, 나머지는 `min(rank*2, 30)`초 stagger
- 출력 파일이 이미 존재하면 스킵 → 재시작 내성

### 발화 수 및 처리 시간

| Split | 발화 수 |
|-------|--------|
| train-clean-100 | 28,539 |
| train-clean-360 | 104,014 |
| train-other-500 | 148,688 |
| dev-clean | 2,703 |
| **합계** | **283,944** |

실측: ~62 utt/s → **약 75~90분** 완료. **현재 완료됨.**

### 출력

```
/mnt/fr20tb/wbl_residency/jos/AudioEnc/log/tmp/cache/word_alignments/
  {split}/{speaker}/{chapter}/{utt_id}.json   # per-utt JSON
  shard_{0..63}.jsonl                          # 워커별 JSONL (legacy)

/mnt/tmp/cache/word_alignments_merged/librispeech/
  {split}.jsonl   # split별 병합 JSONL
  {split}.arrow   # Arrow IPC
```

### merge 및 검증

```bash
python3 alignment/merge_alignments.py \
    --alignment-dir /mnt/tmp/cache/word_alignments \
    --manifest      /mnt/tmp/cache/word_alignments/manifest.jsonl \
    --output-dir    /mnt/tmp/cache/word_alignments
```

---

## 2. MLS / GigaSpeech / VoxPopuli — Qwen3-ForcedAligner

### 모델

`Qwen/Qwen3-ForcedAligner-0.6B` (FunASR 프레임워크)

- NAR 구조: 전체 발화를 한 번에 보고 경계 예측 → 다양한 도메인에서 robust
- 입력: audio path + transcript + language
- 출력: `[[word, start_sec, end_sec], ...]`

### 파이프라인

```
alignment/
  align_qwen3.py         # MLS/GigaSpeech/VoxPopuli alignment 워커
  run_align_qwen3.sh     # 다중 GPU 런처
```

### 출력 경로

```
/mnt/tmp/cache/word_alignments_qwen3/{dataset}/train/{...}/{utt_id}.json

/mnt/tmp/cache/word_alignments_merged/{dataset}/
  train.jsonl   # split별 병합 JSONL
  train.arrow   # Arrow IPC
```

### 출력 스키마

```json
{
  "utterance_id": "mls_1234567",
  "dataset": "mls",
  "split": "train",
  "audio_duration": 5.12,
  "transcript": "the quick brown fox",
  "words": [
    {"word": "the",   "start": 0.10, "end": 0.22},
    {"word": "quick", "start": 0.25, "end": 0.48}
  ],
  "alignment_model": "Qwen/Qwen3-ForcedAligner-0.6B",
  "processed_at": "2026-04-06T..."
}
```

---

## 공통 출력 JSON 스키마

```json
{
  "utterance_id": "19-198-0000",
  "split": "train-clean-100",
  "audio_duration": 4.32,
  "transcript": "he hoped there would be stew for dinner",
  "words": [
    {"word": "he",    "start": 0.18, "end": 0.30, "score": 0.991},
    {"word": "hoped", "start": 0.34, "end": 0.62, "score": 0.987}
  ],
  "alignment_model": "...",
  "processed_at": "2026-04-06T..."
}
```

`score`: CTC 기반만 존재. Qwen3 출력에는 없음 → 필드 생략.  
`score` 없는 항목은 학습 시 필터링 대상에서 제외.

---

---

## 데이터 품질 통계 (2026-04-08)

각 split Arrow에서 5,000개 랜덤 샘플링. 스크립트: `alignment/quality_check.py`  
전체 결과 JSON: `docs/quality_check_results.json`

### 지표 설명

| 지표 | 설명 |
|------|------|
| **coverage** | `last_word_end / audio_duration`. 1.0 이상 = 오디오 끝까지 정렬, <1.0 = 끝에 묵음 |
| **words/sec** | 정렬 구간 내 단어 밀도. 영어 평균 약 2.5~3.5 wps |
| **inter-word gap** | 연속 단어 사이 침묵(초). p50=0 → Qwen3 출력 특성 (contiguous) |
| **overlap rate** | 앞 단어 end > 다음 단어 start (20ms tolerance). 0%이 이상적 |
| **out-of-bounds rate** | `last_word_end > audio_duration + 0.1s`. 타임스탬프 오류 의심 |
| **word-count mismatch** | 전사 단어 수와 정렬 단어 수 차이 >10%. 정렬 누락 의심 |
| **CTC score** | WAV2VEC2 CTC 정렬 신뢰도 (LibriSpeech만 존재, 0~1) |

### 결과

| Dataset/Split | n | coverage (mean/p5/p95) | words/sec (mean) | gap p50 | overlap | OOB | mismatch | CTC score (mean) |
|---|---|---|---|---|---|---|---|---|
| librispeech/dev-clean | 2,703 | 1.004 / 1.001 / 1.008 | 2.76 | 0.060s | 0.00% | 0.00% | 0.00% | 0.949 |
| librispeech/train-clean-100 | 5,000 | 1.002 / 1.001 / 1.005 | 2.72 | 0.060s | 0.00% | 0.00% | 0.00% | 0.952 |
| librispeech/train-other-500 | 5,000 | 1.002 / 1.001 / 1.006 | 2.69 | 0.060s | 0.00% | 0.00% | 0.00% | 0.950 |
| mls/train | 5,000 | 0.967 / 0.934 / 0.988 | 2.78 | 0.000s | 0.00% | 0.04% | 0.00% | — |
| gigaspeech/train | 5,000 | 0.930 / 0.800 / 0.986 | 3.35 | 0.000s | 0.00% | 0.04% | 0.00% | — |
| voxpopuli/train | 5,000 | 0.981 / 0.928 / 1.000 | 2.80 | 0.000s | 0.00% | 0.40% | 0.00% | — |

### 해석

- **LibriSpeech (WAV2VEC2 CTC)**: coverage >1.0 은 CTC 정렬이 오디오 끝까지 확장하는 특성. CTC score 평균 0.95로 매우 높음. overlap/mismatch 없음 → 우수
- **MLS (Qwen3)**: coverage 0.967 (끝 ~3% 묵음). gap p50=0 (Qwen3는 단어를 붙여서 출력). OOB 0.04% 미미 → 우수
- **GigaSpeech (Qwen3)**: coverage 0.930으로 상대적으로 낮음 — 팟캐스트 특성상 발화 끝 묵음이 김. words/sec 3.35로 빠른 편 (팟캐스트 자연스러운 빠른 발화). 이상 없음
- **VoxPopuli (Qwen3)**: OOB 0.40% (20/5000) — 일부 발화에서 마지막 단어 타임스탬프가 오디오 길이 초과. 필터링 기준으로 활용 가능

### 결론

overlap, word-count mismatch 모두 0%. 전체적으로 alignment 품질 양호. VoxPopuli OOB 0.4%는 필터링 시 제거 권장.

---

## 활용 방안

- **데이터 품질 필터링**: score 평균 < 0.5 발화 제외 (LibriSpeech), OOB 발화 제외
- **세그먼트 단위 학습**: 단어/구 경계로 오디오를 자르는 augmentation
- **보조 alignment loss**: projector frame ↔ word timestamp 대응 손실 (장기)

---

## Interleaving vs word-crop ASR — alignment 의 올바른 사용

Word timestamp 를 확보했을 때 "이걸로 뭘 할 수 있는가" 에 대한 두 접근의 구분.

### 접근 1. Interleaving (modality 교차 삽입)

한 시퀀스 안에 text 토큰과 해당 단어의 audio 청크를 **번갈아** 배치:

```
[t_안녕하세요] [a_0.0~1.2] [t_만나서] [a_1.2~2.0] [t_반갑습니다] [a_2.0~3.0]
```

- forward pass 에서 모델이 매 단어마다 "이 텍스트 ↔ 이 오디오 구간" 대응을 직접 관찰
- cross-modal alignment 를 토큰 레벨로 강제하는 supervision
- word timestamp 가 있어야만 가능한 고유한 학습 신호

### 접근 2. Word-crop ASR (단순 분할)

timestamp 로 단어별 오디오를 잘라 각자 독립 mini-ASR 샘플로 만듦:

```
sample 1: audio[0.0~1.2] → "안녕하세요"
sample 2: audio[1.2~2.0] → "만나서"
sample 3: audio[2.0~3.0] → "반갑습니다"
```

- 모델 입력 구조는 기존 sentence ASR 과 동일 (`audio → text`)
- 한 utterance 를 N 개 짧은 샘플로 쪼개는 전처리일 뿐

### 왜 word-crop 은 의미가 적은가

1. **sentence ASR 로 이미 커버됨**. `full_wav → full_text` 학습 시 attention 이 내부적으로 word 정렬을 암묵적으로 학습. crop 해서 따로 넣는다고 모델이 새로 배우는 신호가 추가되지 않음.
2. **오히려 문맥 손실**. 단어 하나만 자르면 coarticulation, 앞뒤 prosody, 언어모델 prior 가 모두 사라짐 — ASR 난이도가 오히려 올라감. 쉬운 샘플을 인위적으로 어렵게 만드는 꼴.
3. **alignment 정보의 가치를 버림**. word timestamp 가 가진 *고유한* 가치는 "어떤 audio 구간이 어떤 text 토큰에 해당" 을 forward pass 안에 주입할 수 있다는 것. crop 은 이 정보를 전처리 단계에서 소모하고 흔적 없이 버림 → 결국 "timestamp 없이 sentence ASR 만 한 것" 과 신호량 차이가 없음.
4. **interleaving 의 장점도 얻지 못함**. 모달리티 교차로 얻는 fine-grained alignment supervision 은 crop 구조에선 발생하지 않음.

### 정리

Word alignment = **interleaving 의 재료** 이지 **crop 의 재료** 가 아님.

| 용도 | alignment 필요? | 고유 신호 추가? |
|---|---|---|
| sentence ASR (`a_full → t_full`) | ✗ | baseline |
| word-crop ASR (`a_word_i → t_word_i`) | ○ (전처리만) | **거의 없음 (오히려 열화)** |
| interleaving (`[t_1, a_1, t_2, a_2, ...]`) | ○ (forward 내) | **있음 — token-level cross-modal alignment** |

crop 할 거면 애당초 alignment 없이 sentence ASR 만 해도 되고, alignment 를 굳이 활용하려면 모달리티 섞는 interleaving 을 해야 그 정보가 gradient 에 실림. crop + ASR 은 sentence ASR 의 열화판이면서 interleaving 의 이점도 얻지 못하는 중간 지대.

### 현행 `--word-aug` 구현 점검 (2026-04-15)

기존 `precompute/pack_arrow.py` + `train_pipeline_override.make_precomputed_processor_fn` 의 `--word-aug` 모드가 실제로 무엇을 내뱉는지 확인.

**[train_pipeline_override.py:1281-1329](../train_pipeline_override.py#L1281-L1329) — `make_precomputed_processor_fn.process_samples`**

utterance 1개마다:
1. 원본 전체 문장 emit (full feats → full text)
2. alignment 의 각 word `w` 에 대해:
   - `start_frame = round(w.start × fps)`, `end_frame = round(w.end × fps)`
   - `feats_2d[start_frame:end_frame]` 슬라이스
   - `word_flat → word_text` 한 쌍으로 emit

→ **정확히 위에서 "의미 없다" 고 분류한 word-crop ASR 그 자체.** alignment 가 있음에도 forward pass 안에서 cross-modal 신호로 쓰이지 않고 전처리 단계에서 소모됨.

### packed_16384 현황 — 혼입 상태

`/mnt/ddn/users/jos/precomputed/fb_dacvae/mixed/packed_16384/` (= 현재 학습에서 in-memory 로 로드되는 shard 세트, ~190 GB).

- 빌드 시 `--word-aug` ✓ 로 빌드됨
- 한 shard 안에 **문장 bin 과 word sub-clip bin 이 섞여 pack** 됨 → bin 단위로 필터링 불가능
- 현재 fb_dacvae S2 (run id `fb_dacvae_2b_S2_0414_1442`) 가 이 데이터로 학습 → loss 3.26 plateau + dev-clean WER 160% 고착. [word_aug_collapse.md](word_aug_collapse.md) §27 collapse 가설과 일치

결론: packed_16384 는 **부분 쓰레기** (문장 부분은 유효하지만 bin 섞여서 현실적으로 재사용 불가) → **폐기 후 재빌드 대상**.

조치:
- [x] `precompute/pack_arrow.py` → [arXiv/scripts/pack_arrow.py](../arXiv/scripts/pack_arrow.py) 이동 (word-crop 구현 아카이브)
- [x] `precompute/run_pack.sh` → [arXiv/scripts/run_pack.sh](../arXiv/scripts/run_pack.sh) 이동 (archived pack_arrow 호출 wrapper 라 함께 폐기)
- [x] [arXiv/README.md](../arXiv/README.md) 에 아카이브 인덱스 추가 (각 파일의 원래 역할 / 이동 사유 / 대체재 기록)
- [ ] 새 `precompute/pack_arrow.py` + `precompute/run_pack.sh` 작성 — 아래 interleaving 방식으로 재빌드

---

## Word-interleaving packing 계획

### 목표

alignment 를 word-crop 이 아닌 **token-level cross-modal supervision** 으로 사용하는 packing 포맷을 만든다. 한 utterance 안에서 text 토큰과 해당 word 의 encoder feature 청크를 번갈아 배치해서 "이 text = 이 audio" 대응을 모델 forward 에 직접 주입.

### 기본 아이디어

utterance 한 개에 대해 **sentence 포맷과 interleaved 포맷을 1:1 로 emit**. 즉 utterance 당 bin 2개가 생성됨 (기존 word-aug 는 utterance 당 1 + N_words, 보통 10+ 배).

| 포맷 | input_ids 구조 | labels |
|---|---|---|
| **sentence** (초기 기획) | `[audio_pad]*T_full + <|audio_correspond|> + text_ids + EOS` | `[-100]*T_full + [-100] + text_ids + [eos]` |
| **interleaved** (초기 기획) | `[audio_pad]*T_w1 + <|audio_correspond|> + text_ids_w1 + [audio_pad]*T_w2 + <|audio_correspond|> + text_ids_w2 + ... + EOS` | 각 `text_ids_wi` 만 활성 (-100 이외), audio_pad / corr 은 -100 |

> **§42 주**: 위 테이블은 초기 기획안이며 §41 ([prompt_format_regression.md](prompt_format_regression.md)) 의 **안 A** 채택으로 `<|audio_correspond|>` 는 전부 legacy `p1="Audio:\n"` / `p2="\nTranscript:\n"` 로 교체됨. 실제 구현은 sentence bin = `p1 + [audio_pad]*T_full + p2 + text + eos`, interleaved bin = word 별로 `p1 + [audio_pad]*T_wi + p2 + text_wi` 반복 + `eos`. `tokenizer.add_special_tokens` / `resize_token_embeddings` 도 제거.

- audio feature 는 utterance 단위로 **한 번만** 인코더 통과 (원본 `feats_2d` 재사용). interleaved bin 은 단지 `feats_2d[start:end]` 구간들을 순서대로 concat + 각 구간 사이에 해당 단어 text 를 끼워 넣음.
- ~~`<|audio_correspond|>` 토큰을 word 경계마다 재사용.~~ → §42: word 경계마다 legacy `p1`/`p2` 텍스트 프롬프트를 재사용. Qwen 이 pretraining 에서 본 "Audio:", "Transcript:" 패턴이라 LLM language prior 를 anchor 로 활용.
- T_w_i = 각 word 구간의 encoder frame 수 (end_frame − start_frame). projector stride 에 의해 llm token 수로 다운샘플.

### 1:1 비율 제안

"word interleaving : sentence = 1:1" 을 우선 채택.

이유:
- **sentence 포맷은 여전히 필수**. 실제 inference 는 word timestamp 없이 full audio → full text. sentence 학습을 유지해야 실전 generalization 확보.
- **interleaving 은 alignment 를 활용한 추가 supervision**. 1:1 로 섞으면 두 분포에 대해 균형 학습.
- 2:1 (interleaving 더 많게) 도 가능하지만 먼저 1:1 로 baseline.
- 기존 word-aug 는 sample 수 기준 대략 1 : N_words(평균 15-20) → interleaving 이 과도하게 dominant 해지지 않도록 명시적 1:1 로 제한.

구현: packer 가 utterance 를 처리할 때 emit 횟수를 **utterance 당 정확히 2 bin** (1 sentence + 1 interleaved) 으로 제한. alignment 가 없거나 word 수 < 2 면 sentence 만 emit.

### 1:1 비율을 어떻게 실현할지 — 안 1 vs 안 2

"sentence : interleave = 1:1" 을 실제 packing 에서 구현하는 방법 두 가지:

- **안 1. utterance 당 둘 다 emit**. 매 utterance 에 sentence bin 1 + interleave bin 1 = 2 bin. 총 데이터셋 시간 환산 ~20 k h → ~40 k h.
- **안 2. utterance 단위 random split 으로 하나만 emit**. deterministic hash (`hash(utt_id) & 1`) 로 각 utterance 를 sentence 또는 interleave 중 하나에만 배정. 총 시간 ~20 k h 유지 (절반이 sentence, 절반이 interleave).

| 차원 | 안 1 (40 k h) | 안 2 (20 k h) |
|---|---|---|
| 학습 데이터 시간 | 2× | 동일 |
| 디스크 / 패킹 시간 | 2× (~380 GB) | 동일 (~190 GB) |
| 1 epoch 벽시계 | 2× | 동일 |
| utterance 당 bin | **2** (sentence + interleave) | **1** (둘 중 하나) |
| paired-view 신호 | ○ (같은 audio, 두 포맷) | ✗ |
| ablation 깔끔함 | ✗ (데이터 양 + 포맷 효과 혼재) | ○ (양 고정, 포맷 효과만 분리) |
| 실패 시 비용 | 2× 시간 날림 | 동일 시간 (현 baseline 과 1:1 비교 가능) |

### 채택: 안 2 (utterance-level deterministic random split)

**이유**

1. **가설 검증 단계** — 지금 목적은 "interleaving 이 word-crop collapse 를 해소하는가" 확인. 안 1 로 WER 이 내려가면 데이터 양 효과와 포맷 효과가 혼재해서 원인 분리 불가. 안 2 는 baseline 과 데이터 양을 고정하고 포맷 효과만 측정.
2. **빠른 턴어라운드** — 24 h vs 48 h. 가설 기각 시 리스크 최소화. 다음 pivot (projector 구조, LR, word_sep 신토큰 등) 으로 빨리 넘어갈 수 있음.
3. **이후 스케일 경로 명확** — 안 2 로 유의미한 WER 개선 나오면 그 다음 안 1 로 데이터 양 확장하는 순서. 역순은 불가 (안 1 성공 → 안 2 검증하려면 어차피 재학습 필요).
4. **utterance 단위 split 이 domain bias 차단** — 데이터셋 단위가 아니라 utterance hash 기반이라 ls100/ls360/mls/gs/vp 각 셋에서 대략 절반씩 sentence / interleave 가 나옴.

**구현 상세**

- 각 utterance `utt_id` 에 대해 `hash(utt_id) & 1` 로 format 결정. 같은 utterance 는 항상 같은 포맷으로 배정 (재빌드 시 재현성).
- 이 결정은 `processor_fn` 안에서 수행 — alignment_lookup 이 있고 split 결과가 interleave 면 interleaved bin 만 emit, 아니면 sentence bin 만 emit.
- alignment 없거나 word 수 < 2 인 utterance 는 hash 결과와 무관하게 sentence 로 fallback.
- 두 포맷이 같은 shard 에 섞여 pack 되므로 각 배치에 두 포맷 공존 → gradient 가 두 신호를 동시에 받음.

**남은 고민 (후속 실험)**

- 안 2 성공 후 안 1 (utt 당 둘 다) 으로 확장해서 데이터 양 효과 추가 측정.
- 순수 interleaving (100%, sentence 0) — 가장 강한 supervision 이나 sentence long-range context 학습은 포기. 별도 ablation.
- 1:1 이 아닌 2:1 / 3:1 (interleave dominant) — 안 2 가 성공한 뒤 ratio sweep.

### 구현 단계

1. **`precompute/half_inlv_pack_arrow.py` 작성** (완료, 2026-04-15)
   - 기존과 동일한 CLI 옵션 (`--rank`, `--mixed`, `--cutoff-len`, `--datasets`) 유지
   - `--word-aug` 플래그 폐기. hash split 은 항상 ON (mixed 모드에서).
   - feature 슬라이싱은 기존 word-aug 와 동일 로직 재사용 (`feats_2d[start:end]`) + `total_stride` 배수 end-zero-pad
   - 안 2 split: utterance 당 `md5(utt_id)[0] & 1` 결정 → 0 이면 sentence bin, 1 이면 interleave bin 만 emit. alignment 부재 / word 수 < 2 면 sentence 로 fallback.
   - `<|audio_correspond|>` 포맷 → **legacy p1/p2 text prompt 포맷** 으로 전환 (§41, [prompt_format_regression.md](prompt_format_regression.md)):
     - sentence bin: `p1 + [audio_pad]*T_proj + p2 + text_ids + [eos]`
     - interleave bin: word 별로 `p1 + [audio_pad]*T_proj_wi + p2 + text_wi` 반복 + `[eos]`
     - `p1 = "Audio:\n"`, `p2 = "\nTranscript:\n"` — Qwen pretraining 어휘 그대로 사용

2. **`train_pipeline_override.py:make_precomputed_processor_fn`** (완료, 2026-04-15)
   - `_emit` 을 legacy p1/p2 포맷으로 교체 (§41)
   - sentence bin 전용 경로로 간소화. interleave 는 offline pack 단계에서 생성되므로 training processor 는 그대로 pass-through.
   - labels 마스킹: `[-100]*len(p1) + [-100]*T_proj + [-100]*len(p2) + text + [eos]`

3. **`config.py` / `run.sh`** `--interleave` 플래그 전달 경로 추가, CLI 에 노출

4. **Smoke test**
   - 1 rank / 1 dataset / 1 shard 로 pack 해서 bin 수 ≈ utterance 수 (안 2 는 정확히 1:1 split 이므로 2× 가 아님) 확인
   - 같은 utterance 를 두 번 돌려 bin 포맷이 재현되는지 (hash 결정성) 확인
   - 한 bin 디코딩해서 text / audio_pad 배치가 의도대로인지 육안 검증
   - train_pipeline_override.forward 가 새 포맷 bin 을 OOM 없이 통과하는지 dry run

5. **전체 재빌드**
   - ls100 / ls360 / ls500 / mls / gs / vp 순서로 `--mixed --interleave`
   - 기존 `packed_16384` 는 보관 또는 삭제 (디스크 절약 시 삭제; ablation 비교 용도라면 이름 변경해 남김)

6. **학습 재개**
   - 동일 cfg 로 stage2 재시작, 새 shard 로 학습
   - WerCallback 이 FSDP 에서 skip 되므로 split 경계마다 별도 eval 돌릴지 여부 별도 결정

### 예상 효과

- utterance 당 bin 수 1+N (기존 word-aug) → **1** (안 2) 으로 줄어 packing 데이터 용량 약 **1/16 ~ 1/20 축소** (단어 평균 N=15-20 가정)
- 학습 step 수 대폭 감소 → epoch 당 벽시계 시간 단축
- baseline 대비 학습 시간·디스크 동일하게 유지한 채 포맷 효과만 검증 가능
- loss plateau 가 word-crop collapse 때문이라면 interleaving 으로 탈출 기대 (반증 시 별도 진단 필요)

### Open questions

- ~~interleaving bin 의 `<|audio_correspond|>` 토큰 반복~~ → §41 에서 legacy p1/p2 text prompt 로 전환하면서 무관해짐.
- max_text_len / cutoff_len 에 걸려 word 수가 많은 긴 utterance 가 잘리는 경우 처리 — `_emit_interleave` 안에서 누적 길이 check 로 break (mid-word truncate). sentence 는 방어 필터로 drop (실제로는 cutoff_len=16384 ≫ 최대 감당 가능 길이라 발동 안 됨).
- sentence 와 interleaved 를 같은 shard 에 섞을지, 별 shard 로 분리할지 — 섞는 쪽이 batch 다양성 확보에 유리. 같은 shard 로 진행.

### 중복 / 누락 / 50-50 비율 보장 (phase 별)

**Phase 1 (processor, utterance → temp row)**

- rank 0~7 의 per-rank precomputed arrow 는 upstream precompute 단계에서 **disjoint 파티션** 으로 생성 → 같은 utterance 가 두 rank 에서 읽히지 않음.
- processor 안 루프:
  ```python
  for utt_id, flat_feats, feat_len, text in zip(...):
      use_interleave = (hash_split(utt_id) == 1 and alignment_lookup is not None and utt_id is not None)
      emitted = False
      if use_interleave:
          words = alignment_lookup.get(utt_id)
          if words and len(words) >= 2:
              emitted = _emit_interleave(...)    # 1 row
      if not emitted:
          _emit_sentence(...)                    # 1 row
  ```
  → utterance 당 **정확히 1 row**. 중복·누락 모두 0.
- `hash_split(utt_id) = md5(utt_id.encode())[0] & 1` — md5 출력의 첫 바이트 LSB 는 결정론적이고 uniform. 즉 어떤 utt_id 를 넣든 0/1 이 고정이고, 전체 집합에 대해 **이론적으로 정확히 50%** (N 이 커지면 ±1 표본오차 수준으로 수렴).

**Phase 2 (Fisher-Yates global shuffle)**

```python
indices = np.arange(total_samples)
rng = random.Random(seed + rank)
for i in range(total_samples - 1, 0, -1):
    j = rng.randint(0, i)
    indices[i], indices[j] = indices[j], indices[i]
```
- 표준 순열 셔플: 결과 배열은 `[0..total_samples)` 의 permutation → 각 index 정확히 한 번 등장.
- 중복·누락 0.

**Phase 3 (packing)**

```python
for i in range(0, total_samples, packing_bucket_size):
    idx_batch = indices[i:i + packing_bucket_size]
    batch = _gather_rows(idx_batch)
    packed = packer_fn(batch)
```
- bucket 들은 `range(0, total, step)` 으로 **disjoint** → 각 index 가 정확히 한 bucket 에 속함.
- `_gather_rows` 는 LRU cache 로 RecordBatch 를 읽어 오지만 실제 row 접근은 한 번씩 → 중복 없음.
- `packer_fn` (greedy knapsack) 은 bucket 내부에서 bin 에 재배치할 뿐 drop 안 함 (단 아래 edge case 제외).

**Phase 4 (equal-bin resplit)**

```python
base = total_bins // num_shards
rem  = total_bins - base * num_shards
sizes = [base + (1 if i < rem else 0) for i in range(num_shards)]
assert sum(sizes) == total_bins
```
- 누적 bin 수가 shard target 에 도달할 때 writer rotate. RecordBatch slice 로 순차 write → 손실·중복 없음.
- 결과: rank 당 shard 0..N-1 이 bin 수 ±1 이내 동일.

**Edge case (누락 가능 지점)**

1. `feat_len == 0` 이거나 `text_ids` 가 비어있는 utterance → processor 가 skip. 이건 학습 불가능한 empty row 이므로 genuine skip.
2. 입력 길이가 `cutoff_len` 을 초과하는 utterance → `_emit_sentence` 마지막 방어 check 로 drop, packer 의 `if len(ids) <= cutoff_len` 필터도 함께 작동. 실제로는 `max_audio_len=20s → T_proj≤430` + `max_text_len=256` + `p1/p2` 합산 ≈ 695 토큰 ≪ 16384 이므로 발동 안 됨.
3. `_emit_interleave` 는 mid-word break 로 자기 자신을 drop 하지 않음 (뒷 단어만 잘림). 이 경우에도 최소 2 단어 조건 (`valid_word_count >= 2`) 을 만족하면 1 row emit.

**50/50 편차 원인**

1. **alignment 미존재 dataset** — 현재 모든 선정 dataset (ls100/ls360/ls500/mls/gs/vp) 은 word alignment arrow 가 있음 → 해당 없음.
2. **`valid_word_count < 2`** — 단어 1 개 이하인 utterance. mls/vp/ls 는 <0.1% 로 무시할 수준이나 **gigaspeech 만 3.15%** 로 유의미 (짧은 segment 에서 단어 1 개 이하 케이스).
3. **hash 표본오차** — N ≥ 10M 수준에선 50.0% ± 0.03% 정도.

**실측 결과 (2026-04-15, 전체 alignment arrow 기준)**:

| dataset | n | hash=0 | hash=1 | words<2 |
|---|---:|---:|---:|---:|
| ls100 | 28,539 | 50.00% | 50.00% | 0.000% |
| ls360 | 104,014 | 50.03% | 49.97% | 0.057% |
| ls500 | 148,688 | 50.28% | 49.72% | 0.036% |
| mls | 2,420,040 | 49.98% | 50.02% | 0.000% |
| gs | 8,282,987 | 49.98% | 50.02% | **3.151%** |
| vp | 182,109 | 50.08% | 49.92% | 0.100% |
| **TOTAL** | **11,166,377** | **49.99%** | **50.01%** | **2.34%** |

hash 분포는 11M 규모에서도 완벽하게 50.00/50.00 근처. 최종 emit 비율은 `words<2 fallback` 때문에 약간 skew:

- **sentence** ≈ 5,712,291 (**51.16%**)
- **interleave** ≈ 5,454,086 (**48.84%**)

2.3% 정도 sentence 쪽으로 치우침. 주 원인 = gigaspeech 의 짧은 segment 3.15%. 배치 학습 관점에선 거의 정확히 1:1 로 취급 가능.

### sentence vs interleave 실제 emit 경로 요약

| utterance 상태 | hash=0 | hash=1 |
|---|---|---|
| alignment 有 & word ≥ 2 | sentence bin | **interleave bin** |
| alignment 有 & word < 2 | sentence bin | sentence bin (fallback) |
| alignment 無 또는 utt_id None | sentence bin | sentence bin (fallback) |

즉 interleave bin 으로 가는 utterance = `hash=1 AND 유효 단어 2개 이상`. 나머지는 모두 sentence.
