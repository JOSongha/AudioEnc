# Dynamic Batching (arXiv Reference)

작성: 2026-04-08  
상태: 아카이브 — 현재 파이프라인은 Sequence Packing을 사용함. 이 문서는 이전 접근법(DynamicBatchSampler)과의 비교 및 arXiv 기술 설명용으로 보존.

코드 이동 (2026-04-08): `BucketBatchSampler`, `DynamicBatchSampler`, 길이 계산 유틸리티(`get_dataset_lengths` 등)를 `dataset.py`에서 `arXiv/scripts/dynamic_batching.py`로 이동.

---

## 1. 배경

스트리밍 ASR 학습에서 오디오 길이는 샘플마다 크게 다르다 (수백ms ~ 20s). 고정 크기 배치로 학습하면:

- 짧은 샘플이 섞이면 긴 샘플에 맞춰 padding → GPU utilization 저하
- 긴 샘플만 배치에 들어가면 메모리 사용이 예측 불가

이를 해결하기 위해 두 가지 접근법이 있다.

---

## 2. 구형 접근법: DynamicBatchSampler

`arXiv/scripts/train_stage1.py`에 구현된 방식.

### 알고리즘

```python
# 전처리: 길이별 bucket 생성
bucket_lengths = [length_of(sample_i) for i in range(N)]
bucket_lengths.sort()
# 학습 시:
# 1. 배치를 길이 기준으로 greedy pack: sum(lengths) <= max_batch_tokens
# 2. 같은 길이대 샘플끼리 배치 → padding 최소화
```

### 특성

| 항목 | 값 |
|---|---|
| 구현 위치 | `dataset.py:DynamicBatchSampler` |
| 동작 | 길이 정렬 → greedy bin packing |
| 기준 | `max_batch_tokens` (LLM 토큰 수 합계) |
| 전처리 | 전체 데이터셋 순회 후 `bucket_lengths_{N}.pkl` 캐시 저장 |
| 스트리밍 호환 | ✗ (전체 길이 사전 계산 필요) |

### 한계

- 스트리밍 데이터셋 불호환: 길이 사전 계산이 필요해 전체 데이터셋을 한 번 순회해야 함
- MLS(44k h), GigaSpeech(10k h) 규모에서 pkl 생성 시간이 큰 병목
- HF streaming 데이터셋에서 `len()` 없음 → 배치 수 추정 불가

---

## 3. 현재 접근법: Sequence Packing (Greedy Knapsack)

`train_pipeline_override.py:create_packer()`에 구현된 방식.

### 알고리즘

```
bucket(1000 processed samples)
  → 길이 오름차순 정렬
  → greedy knapsack:
      bin 시작 (remaining = cutoff_len)
      남은 샘플 중 remaining 이하인 가장 큰 것 선택 → bin에 추가
      더 이상 fit하는 샘플 없음 → bin 종료 (pad to cutoff_len)
      반복
```

### 특성

| 항목 | 값 |
|---|---|
| 구현 위치 | `train_pipeline_override.py:create_packer()` |
| 동작 | bucket 내 greedy knapsack |
| 기준 | `packing_cutoff_len` (= 최대 시퀀스 길이, 토큰 수) |
| 전처리 | 불필요 (on-the-fly) |
| 스트리밍 호환 | ✓ |
| 아이템 절단 | ✗ (fit 불가 아이템은 다음 bin으로) |

### Attention mask

패킹 후 각 토큰에 서브시퀀스 인덱스 부여:

```
bin:  [seq1_tok0, seq1_tok1, seq2_tok0, seq2_tok1, seq2_tok2, PAD, PAD]
mask: [     1,        1,         2,         2,         2,       0,   0  ]
```

- `0`: padding (attend 불가)
- `1, 2, 3, …`: 같은 숫자 = 같은 서브시퀀스 (cross-attend 허용)

Flash Attention 2 경로: padding 제거 → `(1, sum_nonpad)` + position_ids 서브시퀀스별 0 리셋.  
Eager/SDPA 경로: `prepare_4d_attention_mask()`로 4D block-diagonal float mask 생성.

---

## 4. 두 접근법 비교

| | DynamicBatchSampler | Sequence Packing |
|---|---|---|
| 배치 기준 | 총 토큰 수 ≤ max_batch_tokens | bin 길이 ≤ cutoff_len |
| 배치 크기 | 가변 (길이에 따라 다름) | 고정 (항상 cutoff_len) |
| padding | 배치 내 최대 길이에 맞춤 (최소화) | cutoff_len에 맞춤 (더 적음) |
| 스트리밍 | ✗ | ✓ |
| 전처리 | pkl 캐시 필요 | 불필요 |
| Flash Attn 2 | 일반 padding 제거 | varlen kernel (더 효율적) |
| 구현 복잡도 | 낮음 | 중간 |

---

## 5. arXiv 기술 메모

Dynamic batching과 sequence packing은 목적이 동일하다 — padding으로 인한 연산 낭비 최소화.  
DynamicBatchSampler는 배치 구성 단계에서 길이를 맞추는 반면,  
Sequence Packing은 여러 시퀀스를 하나의 고정 길이 시퀀스로 연결하여 근본적으로 padding을 제거한다.

Sequence Packing은:
1. 스트리밍 데이터와 호환됨
2. Flash Attention 2의 varlen kernel을 온전히 활용할 수 있음 (비연속 position_ids로 경계 인식)
3. 배치 크기가 고정이어서 memory estimator가 단순함

현재 파이프라인은 Sequence Packing만 사용.  
`max_batch_tokens`는 WER evaluation의 greedy decode 배치 크기 제한에만 남아 있음 (train에는 미사용).
