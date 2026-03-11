# 왜 RVQ 이전 피처를 쓰는가?

## audio codec의 내부 구조

EnCodec, DAC, Mimi는 모두 같은 2단계 파이프라인을 공유한다.

```
오디오 파형
    │
    ▼
┌──────────────────-┐
│  acoustic encoder │   연속적인 float 벡터  (B, C, T)
│  (CNN / ConvNet)  │   ← 이걸 쓴다
└──────────────────-┘
    │
    ▼
┌──────────────────┐
│  RVQ quantizer   │   이산 정수 코드       (B, n_codebooks, T)
│                  │   ← 쓰지 않는다
└──────────────────┘
```

RVQ(Residual Vector Quantization)는 float 벡터를 codebook 인덱스
여러 개의 합으로 근사하는 손실 압축이다. 오디오 전송·저장용으로
설계된 것이지, downstream 학습용이 아니다.

---

## pre-RVQ vs post-RVQ 비교

| | pre-RVQ (encoder 출력) | post-RVQ (quantizer 출력) |
|---|---|---|
| 형태 | float tensor | int 코드 배열 |
| 정보량 | 원본에 가까운 연속 표현 | 양자화 오차만큼 손실 |
| gradient | 흐름 ✓ | 흐르지 않음 ✗ |
| LLM 입력 | 직접 embed 가능 | 별도 codebook embed 필요 |
| 목적 | 표현 학습 | 압축·전송 |

---

## gradient가 흐르지 않는다는 것의 의미

post-RVQ 코드를 쓴다면 학습 파이프라인은 이렇게 된다:

```
오디오 → encoder → [argmin 연산] → 정수 코드 → embed → LLM
                       ↑
              미분 불가 (이산)
```

`argmin`은 미분이 정의되지 않는다. STE(Straight-Through Estimator)
같은 우회책이 있지만 이 프레임워크에서는 encoder를 frozen으로 쓰기
때문에 굳이 필요하지 않다. pre-RVQ 피처를 쓰면 문제 자체가 없어진다.

---

## 각 encoder에서 실제로 호출하는 API

```python
# EnCodec
feats = model.encoder(audio)          # (B, 128, T) @ 75fps

# DAC
feats = dac.encoder(audio)            # (B, 1024, T) @ ~86fps

# Mimi acoustic (encoder만)
feats = mimi.encoder(audio)           # (B, 512, T) @ 25fps

# Mimi semantic (encoder + transformer)
enc = mimi.encoder(audio)             # (B, 512, T)
out = mimi.encoder_transformer(enc)   # (B, T, 512)
```

반면 RVQ를 거치는 API (사용하지 않음):

```python
# EnCodec
codes = model.encode(audio)           # List[Tensor] of int codes

# DAC
z, codes, _, _, _ = dac.encode(audio) # int codes

# Mimi
codes = mimi.encode(audio)            # int codes
```

---

## 정보 보존 측면

RVQ는 근사 오차가 발생한다. 미세한 음향 특징(모음 포먼트, 억양 등)이
양자화 경계에서 뭉개질 수 있다. ASR에서는 이런 세밀한 정보가 전사
정확도에 직결되므로, 정보 손실 없는 pre-RVQ 피처가 유리하다.
