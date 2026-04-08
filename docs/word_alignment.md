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
