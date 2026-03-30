# DAC vs FB-DACVAE 비교

## 개요

두 encoder는 모두 DAC 기반 acoustic encoder이지만, 학습 방식과 latent 표현이 다름.

---

## DAC (Descript Audio Codec)

| 항목 | 값 |
|---|---|
| 모델 | `descript/dac-44khz` |
| 사용 부분 | encoder only (pre-RVQ) |
| Encoder 파라미터 | ~25M |
| out_dim | 1,024 |
| 입력 sr | 44kHz |
| 출력 fps | ~86 fps (hop=512 @ 44kHz) |
| Projector stride | ×4 (~215 tokens/10sec) |

- RVQ 직전의 continuous encoder output 사용
- 고차원 acoustic feature (1024-dim)
- codec 재구성 목적으로 학습된 표현

---

## FB-DACVAE (Facebook DACVAE)

| 항목 | 값 |
|---|---|
| 모델 | `facebook/dacvae-watermarked` |
| 사용 부분 | encoder + VAE bottleneck (전체 frozen) |
| out_dim | 8 (codebook_dim) |
| 입력 sr | 48kHz |
| 출력 fps | ~25 fps (hop=1920 @ 48kHz) |
| Projector stride | ×4 (~63 tokens/10sec) |

- Facebook Research의 pretrained DACVAE
- VAE bottleneck을 통해 압축된 latent (8-dim)
- watermarking 목적으로 학습된 표현 — 풍부한 semantic 정보 포함 가능성
- 낮은 fps로 LLM 입력 토큰 수가 적음 (메모리 효율적)

---

## 비교 요약

| 항목 | DAC | FB-DACVAE |
|---|---|---|
| out_dim | 1,024 | 8 |
| fps | ~86 | ~25 |
| tokens/10sec | ~215 | ~63 |
| Projector 파라미터 | ~52M | ~41M |
| Encoder 원본 목적 | 오디오 코덱 | 워터마킹 |
| Frozen 범위 | encoder | encoder + VAE |

---

## 실험 의의

DAC는 고차원·고해상도 acoustic feature를 제공하는 반면,
FB-DACVAE는 저차원으로 압축된 latent를 사용해 LLM 입력 길이가 짧음.

ASR 성능 관점에서:
- DAC: 풍부한 acoustic feature, projector가 더 많은 파라미터로 매핑
- FB-DACVAE: compact latent, 토큰 수 적어 LLM 처리 효율적

두 encoder 모두 semantic 특화 encoder(Whisper, Mimi semantic)와의 성능 비교를 통해
acoustic vs semantic representation의 ASR 기여도를 측정하는 것이 목표.
