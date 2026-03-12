# DAC vs DAC-VAE: 차이점

## 핵심 구조 비교

| 항목 | `dac` | `dac_vae` |
|---|---|---|
| 백본 | DAC encoder (frozen) | DAC encoder (frozen) |
| 출력 dim | 1024 | 256 (latent_dim) |
| bottleneck | 없음 (point estimate) | mu/logvar → sampling |
| trainable 파라미터 | 없음 (완전 frozen) | `to_mu_logvar`: Linear(1024→512) |
| 학습 중 | z = f(audio) (deterministic) | z = mu + ε·exp(0.5·logvar) |
| 평가 중 | z = f(audio) | z = mu (deterministic) |
| KL loss | 없음 | `encoder.last_kl_loss` 제공 |
| fps | ~86fps | ~86fps |
| projector 이후 | ~21.5fps (~215 tok/10s) | ~21.5fps (~215 tok/10s) |

---

## 왜 out_dim이 다른가

`dac`는 DAC encoder 출력(1024-dim)을 그대로 projector에 넘긴다.
`dac_vae`는 1024-dim → VAE bottleneck → **256-dim** latent를 projector에 넘긴다.

256-dim bottleneck의 효과:
- **정보 압축**: 1024-dim 중 ASR에 유용한 정보만 남기도록 강제
- **projector 부담 감소**: projector가 처리할 차원 수가 ¼로 줄어듦

---

## VAE bottleneck이 하는 일

```
DAC encoder (frozen)
    ↓ (B, T, 1024)
Linear: 1024 → 512  (to_mu_logvar)
    ↓ chunk
mu:     (B, T, 256)
logvar: (B, T, 256)
    ↓ reparameterization (학습 중)
z = mu + ε · exp(0.5 · logvar)
    ↓ (B, T, 256) → projector
```

**Reparameterization trick**: gradient가 mu, logvar를 통해 역전파되어 VAE head가 학습된다.
ε는 학습 중에만 추가되고, 평가 시에는 z = mu를 사용한다.

---

## KL Loss

`encoder.last_kl_loss`에 매 forward마다 계산된 값이 저장된다:

```
KL = -0.5 · mean(1 + logvar - mu² - exp(logvar))
```

현재 train.py는 KL loss를 학습 objective에 포함하지 않는다 (LM loss만 사용).
필요 시 아래처럼 추가할 수 있다:

```python
loss = outputs.loss + kl_weight * accelerator.unwrap_model(model).encoder.last_kl_loss
```

### KL regularization이 하는 일

KL을 최소화하면 `mu → 0`, `logvar → 0` (분산 → 1)으로 수렴한다.
즉, latent z가 표준정규분포 N(0, I)에 가깝도록 강제한다.

**KL 없이 reparameterization만 쓰면** 모델은 logvar를 아주 작은 음수로 만들어
(exp(logvar) ≈ 0) noise를 무력화하는 방향으로 학습된다.
결과적으로 사실상 deterministic encoder가 되어 VAE head를 추가한 의미가 없어진다.

**KL을 추가하면** logvar를 줄이면 penalty가 커지므로, 모델이 일정 수준의
uncertainty를 유지해야 한다. 그 결과 latent space가 연속적으로 구성된다
(비슷한 발음 → 가까운 z, 다른 발음 → 먼 z).

**ASR에서 이게 필요한가**: ASR은 "발음 → 텍스트"의 one-to-one 매핑이라
structured latent space의 이점이 TTS·음악 생성 대비 훨씬 작다.
KL annealing 없이 dac_vae를 쓰면 실질적으로 noisy linear projection에 가깝다.

---

## ASR에서 VAE가 불리할 수 있는 이유

현재 구현에서 `dac_vae`는 KL loss 없이 reparameterization만 적용한다.
이 상태에서는 VAE의 핵심인 latent space regularization 효과가 없고,
사실상 **"noise를 더한 linear projection"** 에 가깝다.

ASR 관점에서의 문제:

1. **KL loss 없음 → 제대로 된 VAE가 아님**
   KL regularization 없이 reparameterization만 쓰면 VAE의 이점(structured latent space)이 없다.
   오히려 학습 중 noise가 더해져 일관성이 떨어질 수 있다.

2. **ASR은 deterministic 태스크**
   같은 발음이 매 step마다 다른 z로 매핑되면, LM loss 기준으로 학습이 불안정해질 수 있다.
   stochasticity가 도움이 되는 생성 태스크(TTS, 음악 생성)와 다르다.

3. **projector가 이미 있음**
   Conv1d projector가 1024-dim → LLM dim 변환을 담당한다.
   encoder 쪽에서 1024→256 압축을 먼저 하면 projector가 배울 기회를 제한한다.

**결론**: `dac_vae`가 `dac`를 이기려면 KL annealing을 포함한 제대로 된 VAE 학습이 필요하다.
베이스라인 비교는 `dac`로 하고, VAE 실험은 KL loss를 추가한 뒤 진행하는 것을 권장한다.

---

## 언제 어떤 걸 쓸까

| 상황 | 추천 |
|---|---|
| 빠른 실험, 베이스라인 | `dac` |
| VAE 제대로 실험 (KL annealing 포함) | `dac_vae` + train.py KL loss 추가 |
| VRAM이 빡빡할 때 | `dac_vae` (out_dim 작아서 projector 가벼움) |

---

## 주의사항

- `dac_vae`는 DAC 부분만 frozen이고, `to_mu_logvar`(Linear)는 trainable이다.
  → Stage 1에서 projector와 함께 학습된다.
- 다른 encoder들과 달리 "완전 frozen" 계약을 지키지 않는다.
  → BaseAudioEncoder 문서의 "항상 frozen" 조항은 DAC 백본에만 해당.
- KL loss 없이 쓰면 VAE가 아니라 noisy linear projection이다.
