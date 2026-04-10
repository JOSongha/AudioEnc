# Mimi Acoustic vs Semantic

## Mimi의 내부 구조

Mimi는 EnCodec, DAC와 달리 encoder가 2단계로 나뉜다.

```
오디오 (24kHz)
    │
    ▼
┌─────────────────────────┐
│  encoder                │   CNN 기반 acoustic encoder
│  (ConvNet, stride=960)  │   (B, 512, T)  @ 25fps
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  encoder_transformer    │   Transformer 기반 semantic encoder
│  (Transformer)          │   (B, T, 512) @ 25fps  ← fps 변화 없음
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  RVQ quantizer          │   이산 코드 (사용 안 함)
└─────────────────────────┘
```

encoder_transformer가 있다는 점이 EnCodec, DAC와 가장 큰 차이다.
이 때문에 Mimi에서 "어디까지를 encoder로 볼 것인가"에 선택지가 생긴다.

---

## MimiAcousticEncoder

```
오디오 → mimi.encoder() → (B, 512, T)
```

CNN만 통과한다. 출력은 파형의 **저수준 음향 특징**이다:
- 스펙트럼 패턴 (포먼트, 배음 구조)
- 에너지 분포
- 운율적 특징 (피치 윤곽, 발화 리듬)

언어적 내용보다는 "어떻게 소리가 나는가"에 해당하는 정보.

---

## MimiSemanticEncoder

```
오디오 → mimi.encoder() → mimi.encoder_transformer() → (B, T, 512)
```

CNN 위에 Transformer를 추가로 통과한다. Transformer는 양방향 문맥을
보며 피처를 재구성하므로 출력은 **고수준 semantic 특징**이다:
- 음소 수준의 언어 정보
- 단어 경계에 민감한 표현
- 언어 모델과 유사한 추상화 수준

"무슨 말을 하는가"에 해당하는 정보.

---

## 두 variant의 비교

| | MimiAcoustic | MimiSemantic |
|---|---|---|
| 호출 | `encoder()` | `encoder()` + `encoder_transformer()` |
| 출력 fps | 25fps | 25fps (변화 없음) |
| 출력 dim | 512 | 512 |
| 피처 수준 | 저수준 acoustic | 고수준 semantic |
| projector | [2, 2] → ×4, 6.25fps | [2] → ×2, 12.5fps |
| 토큰/10초 | ~63 tokens | ~125 tokens |
| 파라미터 | 적음 | 많음 (transformer 추가) |
| 메모리 | 낮음 | 높음 |
| ASR 예상 성능 | 낮음 | 높음 |

---

## projector stride가 다른 이유

MimiAcoustic의 25fps 출력을 ×4 다운샘플하면 6.25fps → 63 tokens.
이 정도 압축은 acoustic 피처에서 큰 정보 손실을 유발하지 않는다
(인접 프레임 간 변화가 완만하기 때문).

MimiSemantic의 25fps 출력을 ×4 다운샘플하면 6.25fps → 63 tokens.
semantic 피처는 음소 경계, 단어 경계 같은 짧은 구간의 정보가
의미를 가지므로, ×4 압축은 과하다. ×2(125 tokens)가 적절하고,
이것이 원본 q_ming.py와 동일한 설정이다.

---

## 언제 어떤 걸 써야 하는가?

**비교 실험 목적**: 두 가지를 같은 조건에서 학습시켜 WER을 비교.
encoder_transformer가 ASR에 얼마나 기여하는지 ablation 가능.

**리소스가 충분할 때**: MimiSemantic. 예상 성능이 높고, 원본 q_ming.py와
동일한 설정이므로 재현성도 있다.

**메모리가 부족할 때**: MimiAcoustic. encoder_transformer 없이
더 가벼운 실험이 가능하다.
