# 인코더 피처 사전 계산 계획

작성: 2026-04-09

---

## 왜 필요한가

현재 훈련 병목은 **CPU 오디오 디코딩**이다. `nvidia-smi dmon`으로 측정하면 GPU SM 사용률이 25~50%에 불과하고, 나머지 시간은 다음 배치를 기다리며 아이들 상태다.

| 항목 | 현재 (online encoding) | 개선 후 (precomputed) |
|---|---|---|
| step당 시간 | ~13.69 s/it | ~1 s/it (예상) |
| 학습 총 시간 (Stage 1+2) | ~350 h | ~25 h |
| GPU SM 사용률 | 25~50% | >90% |

### 병목 구조

```
DataLoader worker (CPU)
  ← FLAC/OPUS 디코딩  (~0.5 s / 32 clips)
  ← 16kHz → 44kHz 리샘플링  (~0.5 s / 32 clips)
  ← 토크나이징  (빠름)
  ← Packing  (빠름)
  → GPU가 기다림 (idle ~12 s)

GPU
  ← DACVAE encoder forward  (~0.3 s / bin)
  ← Projector forward  (빠름)
  ← LLM forward × 4 grad_accum  (~0.5 s)
  ← Backward  (~0.5 s)
  = GPU 실제 연산: ~1.3 s  (vs 전체 13.69 s)
```

## 해결 방법

훈련 루프에서 인코더를 제거한다.  
모든 데이터셋에 대해 **인코더 출력 피처를 미리 계산**해 Arrow 파일로 저장한다.  
훈련 시에는 피처를 그대로 로드하고 **Projector → LLM** 만 실행한다.

```
Before:  audio bytes → decode → resample → Encoder(GPU) → Projector → LLM
After:   precomputed features → Projector → LLM
```

## 저장 형식

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `utterance_id` | string | 원본 데이터셋의 utterance ID |
| `text` | string | 정답 텍스트 |
| `features` | float32 list | shape (T_enc × out_dim,), flattened |
| `feat_len` | int32 | T_enc (인코더 출력 프레임 수) |

**Arrow** 포맷 (`.arrow`, `pyarrow` 사용). 데이터셋별·rank별로 분리 저장.

### 저장 경로

```
/mnt/fr20tb/wbl_residency/jos/ddn/precomputed/{encoder_name}/
  ls100/   rank0.arrow  rank1.arrow  ...  rank7.arrow
  ls360/   rank0.arrow  ...
  ls500/   rank0.arrow  ...
  mls/     rank0.arrow  ...
  gs/      rank0.arrow  ...
  vp/      rank0.arrow  ...
  meta.json            ← 인코더 설정, 완료 여부, 파일별 행 수
```

### 용량 추정 (fb_dacvae 기준)

| 항목 | 값 |
|---|---|
| 인코더 FPS | 86 fps (44100 Hz / 512 hop) |
| out_dim | 8 |
| float32 per frame | 32 bytes |
| 초당 바이트 | 86 × 8 × 4 = 2,752 B/s |
| 21,500 h 전체 | ~213 GB |
| float16 사용 시 | ~107 GB |

FLAC 원본 대비 **절반 이하** 크기. 로드 속도도 훨씬 빠름.

## 전처리 속도 추정

A100 80GB에서 DACVAE encoder:
- batch (64, 1, 441000) → ~0.5 s
- GPU throughput: 64 clips × 10 s / 0.5 s = **1,280 s/s per GPU**
- 8 GPU 합산: **10,240 s/s** 실시간 배속
- 21,500 h = 77.4 M 초 → 77.4M / 10,240 ≈ **7,560 s ≈ 2.1 h**

보수적 (배치 작게 잡으면): ~25 h  
공격적 (배치 크게 + CUDA stream 파이프라인): ~7 h

## GPU 내 병렬화 전략

각 GPU에서 **두 개의 CUDA stream**을 사용해 H2D 복사와 GPU 연산을 파이프라인:

```
Stream A: [H2D batch 1] → [encode batch 1] → [D2H feats 1]
Stream B:              [H2D batch 2] → [encode batch 2] → [D2H feats 2]
```

CPU 쪽에서는 **ThreadPoolExecutor**로 오디오 디코딩을 병렬 실행.

## Word-aug 처리

전처리 시 utterance 단위로만 피처를 저장한다.  
훈련 시 word 구간을 프레임 인덱스로 슬라이싱:

```python
start_frame = int(word_start_sec * tgt_sr / hop)
end_frame   = int(word_end_sec   * tgt_sr / hop)
word_feats  = features[start_frame:end_frame]   # (T_word, out_dim)
```

인코더를 word clip마다 재실행하지 않아도 되므로 현재 대비 **100× 빠름**.

## 단점 / 트레이드오프

| 단점 | 영향 |
|---|---|
| 인코더 바꾸면 전처리 재실행 필요 | encoder-swappable 목적과 충돌 |
| 전처리 디스크 공간 ~213 GB (per encoder) | /mnt/tmp 19 TB 여유, 문제없음 |
| 전처리 시간 2~25 h (1회) | 훈련 단축 효과로 충분히 회수 |
| 인코더 파인튜닝 불가 | 원래 frozen이라 해당 없음 |

인코더 비교 실험 시: 각 인코더별로 전처리를 별도 실행하면 됨.  
(encodec, dac, fb_dacvae, mimi_* 각각 ~7~25 h)

## 구현 파일

| 파일 | 역할 |
|---|---|
| `precompute/precompute_features.py` | 전처리 메인 스크립트 (single GPU) |
| `precompute/run_precompute.sh` | 8 GPU 병렬 실행 래퍼 |
| `model.py` | `_get_proj_from_precomputed()` 추가 |
| `train_pipeline_override.py` | `--precomputed` 플래그, 피처 로드 경로 |

## 실행 순서

```bash
# 1. 전처리 (약 7~25 h)
bash precompute/run_precompute.sh --encoder fb_dacvae

# 2. 전처리 완료 확인
python precompute/precompute_features.py --encoder fb_dacvae --verify

# 3. 전처리된 피처로 훈련
bash run.sh --encoder fb_dacvae --precomputed
```
