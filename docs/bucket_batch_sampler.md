# 배치 샘플러: BucketBatchSampler & DynamicBatchSampler

## 문제: 랜덤 배치에서 발생하는 패딩 낭비

`DataLoader(shuffle=True)` 는 길이가 다른 샘플을 무작위로 배치에 묶는다.
`collate_fn`에서 `pad_sequence`로 배치 내 최장 길이에 맞춰 패딩하므로,
짧은 샘플이 긴 샘플과 같은 배치에 들어가면 패딩 구간이 많아진다.

```
기존 (shuffle=True):
배치 1: [2초, 18초, 3초, 17초]  → 전부 18초로 패딩 (실제 오디오 22%, 패딩 78%)
배치 2: [4초, 1초, 15초, 2초]   → 전부 15초로 패딩
...
```

이 패딩은 encoder forward, projector Conv1d, LLM forward 세 곳에서
모두 연산량으로 포함된다. 특히 LLM attention은 O(T²)이므로
패딩이 많을수록 낭비가 급격히 커진다.

`max_audio_len = 16000 * 20` (20초)로 설정되어 있어, 배치 내에 20초짜리 샘플이
하나만 있어도 나머지 전부가 20초로 패딩되면서 문제가 더욱 심각했다.

---

## 해결책 1: BucketBatchSampler

비슷한 길이의 샘플끼리 같은 배치로 묶는다. 배치 크기는 고정.

```
1. 전체 샘플을 오디오 길이 기준으로 정렬
2. 정렬된 순서로 bucket_size(= batch_size × 100)개씩 묶음 (버킷)
3. 각 버킷 내에서 셔플 (비슷한 길이끼리지만 순서는 랜덤)
4. 버킷 내에서 batch_size개씩 배치 생성
5. 배치 순서를 전체적으로 셔플
6. DDP: rank별로 배치를 분배
```

```
변경 후 (BucketBatchSampler):
배치 1: [2초, 3초, 2초, 3초]     → 전부 3초로 패딩  (패딩 ~33%)
배치 2: [15초, 17초, 16초, 18초] → 전부 18초로 패딩 (패딩 ~17%)
```

버킷 크기(`batch_size × 100`)는 완전 정렬과 완전 랜덤 사이의 균형이다.
버킷이 너무 작으면 패딩 감소 효과가 줄고, 너무 크면 학습 순서가
길이 순서에 지나치게 고정되어 overfitting 위험이 있다.

---

## 해결책 2: DynamicBatchSampler

배치 크기를 고정하지 않고, **배치 내 총 LLM 토큰 수**를 예산으로 잡는다.

오디오 샘플 수를 예산으로 쓰면 안 되는 이유:

```
오디오 샘플 수는 같아도 LLM 토큰 수가 다를 수 있다:

20초 × 6개:  오디오 샘플 1,920,000  LLM 토큰 = 6 × (430 + 256) = 4,116
2초  × 60개: 오디오 샘플 1,920,000  LLM 토큰 = 60 × (43 + 256) = 17,940  ← OOM

VRAM은 오디오 샘플이 아니라 LLM 토큰이 결정한다.
```

`max_batch_tokens`는 `get_config()`에서 encoder 정보를 이용해 자동 계산된다.

```python
samples_per_token = hop × (16000 / tgt_sr) × prod(proj_strides)
max_audio_tokens  = max_audio_len / samples_per_token
max_batch_tokens  = 6 × (max_audio_tokens + max_text_len)
```

| encoder | samples_per_token | max_audio_tokens (20초) | max_batch_tokens |
|---|---|---|---|
| EnCodec | ~213 | ~1,501 | ~10,542 |
| DAC | ~186 | ~1,720 | ~11,856 |
| Mimi semantic | ~640 | ~500 | ~4,536 |

```
예산: max_batch_tokens (encoder별 자동 계산)

짧은 클립(2초, DAC): 토큰 = 43 + 256 = 299  → 예산/299 ≈ 40개 한 배치
긴 클립(20초, DAC):  토큰 = 430 + 256 = 686  → 예산/686 ≈ 17개 한 배치
```

---

## 두 샘플러 비교

| | BucketBatchSampler | DynamicBatchSampler |
|---|---|---|
| 배치 크기 | 고정 | 가변 (LLM 토큰 예산 기반) |
| 패딩 감소 | 비슷한 길이끼리 묶어 감소 | 예산 내에서 최대한 채워 감소 |
| VRAM 일관성 | 배치마다 다를 수 있음 | 토큰 예산으로 일정하게 유지 |
| 구현 복잡도 | 단순 | 중간 |
| 짧은 클립 활용 | 고정 배치 크기라 손해 | 자동으로 배치 크기 증가 |

---

## DDP 호환성 (공통)

기존 코드는 `accelerator.prepare(dataloader)`로 DataLoader를 넘겼다.
이 경우 accelerate가 내부적으로 `DistributedSampler`를 주입한다.

두 샘플러 모두 `num_replicas`와 `rank`를 받아 DDP 분배를 직접 처리하므로
`accelerator.prepare()`에 dataloader를 넘기지 않는다.

```python
# 변경 전
model, optimizer, train_loader, val_loader = accelerator.prepare(...)

# 변경 후
model, optimizer = accelerator.prepare(model, optimizer)
# dataloader는 sampler가 DDP 직접 처리
```

`accumulate()` 컨텍스트 매니저는 model 기준으로 동작하므로
dataloader를 prepare하지 않아도 gradient accumulation은 정상 작동한다.

에폭마다 셔플 시드를 바꾸기 위해 루프 시작 시 `set_epoch()`를 호출한다.

```python
for epoch in range(epochs):
    train_loader.batch_sampler.set_epoch(epoch)
```

### DDP에서 배치 수 불균형 문제

8개 rank가 배치를 나눌 때 총 배치 수가 8의 배수가 아니면
rank마다 배치 수가 달라진다. 이 경우 먼저 끝난 rank는
다른 rank의 `allreduce`를 기다리다 NCCL timeout으로 hang된다.

**기존 방식 (truncate)**: 나머지 배치를 버려 8의 배수로 맞춤 → 데이터 손실

**변경 후 (패딩)**: 앞 배치를 반복해 8의 배수로 맞춤 → 데이터 손실 없음

```python
remainder = len(all_batches) % self.num_replicas
if remainder:
    all_batches += all_batches[: self.num_replicas - remainder]
```

반복되는 배치는 최대 7개(= num_replicas - 1)이며,
에폭당 전체 배치 수 대비 무시할 수 있는 수준이다.

---

## 길이 계산 방법

`get_dataset_lengths(dataset, samples_per_token, max_text_len)`는
LLM 토큰 수 기준 길이를 반환한다.

```python
token_len = int(audio_samples_16k / samples_per_token) + max_text_len
```

내부적으로 raw 오디오 샘플 수를 먼저 구한 뒤 토큰으로 변환한다.
raw 샘플 수는 캐시에 저장되므로, `samples_per_token`이 바뀌어도(encoder 변경 등)
캐시를 다시 만들 필요 없다.

### LibriSpeech: `torchaudio.info()`

FLAC 파일 헤더만 읽어 `num_frames`를 얻는다. 오디오 디코딩 없음.

```python
info = torchaudio.info(path)  # 헤더만 읽음
n = int(info.num_frames * target_sr / info.sample_rate)
```

LibriSpeech 전체 (~480K 파일) 처리에 수십 초 수준.

### MLS: transcript 글자 수 → 샘플 수 변환

4,050,000개 오디오를 디코딩 없이 처리하기 위해
HuggingFace datasets의 `transcript` 컬럼을 직접 읽는다.
영어 평균 발화 속도 ~14자/초를 기준으로 샘플 수로 변환해
LibriSpeech 길이와 동일한 스케일을 맞춘다.

```python
lengths = [min(int(len(t) / 14.0 * target_sr), max_len)
           for t in selected["transcript"]]
```

### 캐싱

첫 실행 시 raw 오디오 샘플 수를 계산하고
`{model_cache_dir}/bucket_lengths_{N}.pkl`에 저장한다.
이후 실행은 캐시에서 즉시 로드한 뒤 토큰 변환만 수행한다.

---

## 변경된 파일 요약

| 파일 | 변경 내용 |
|---|---|
| `dataset.py` | `BucketBatchSampler`, `DynamicBatchSampler` 추가; `get_dataset_lengths(samples_per_token, max_text_len)` — LLM 토큰 수 반환; `_get_raw_lengths()` 분리(캐싱); `_mls_lengths` 반환값을 글자 수 → 샘플 수로 변경 |
| `config.py` | `batch_size`, `stage2_batch_size`, `max_batch_samples`, `max_batch_size` 제거; `get_config()`에서 `samples_per_token`, `max_batch_tokens` 자동 계산 |
| `train.py` | Stage 1/2 모두 `DynamicBatchSampler` 사용; `accelerator.prepare()`에서 dataloader 제거; 에폭마다 `set_epoch()` 호출 |

---

## 기대 효과

- **패딩 감소**: 배치 내 길이 분산이 줄어 실제 연산량 대비 패딩 비율 감소
- **throughput 향상**: 짧은 클립 구간에서 DynamicBatchSampler가 자동으로 더 많은 샘플을 묶어 처리
- **VRAM 일관성**: DynamicBatchSampler 사용 시 예산으로 배치당 메모리 사용량이 일정
- **데이터 손실 없음**: DDP 정렬 시 truncate 대신 패딩으로 모든 샘플이 학습에 참여
- **첫 실행 오버헤드**: 길이 계산 1회 (LibriSpeech ~수십 초, MLS ~수 분), 이후 캐시 사용
