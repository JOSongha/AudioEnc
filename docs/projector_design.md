# Projector 설계: 왜 Conv1d인가?

> 관련 논문: [1] Flamingo (Alayrac et al., 2022) [2] LLaSA (Shen et al., 2024)
> [3] SALMONN (Tang et al., 2024) [4] Qwen-Audio (Chu et al., 2023)

## 문제: encoder 출력이 LLM에 넣기엔 너무 길다

audio encoder는 오디오를 연속적인 float 벡터 시퀀스로 변환한다.
문제는 이 시퀀스가 너무 길다는 것이다.
(자세한 내용은 [1] Flamingo §3, [3] SALMONN §3.1 참조)

| encoder | fps | 10초 오디오 토큰 수 |
|---|---|---|
| EnCodec | 75fps | 750 tokens |
| DAC | ~86fps | 860 tokens |
| Mimi | 25fps | 250 tokens |

Qwen2.5-7B의 hidden_size는 3584. attention complexity는 O(n²)이므로
750 audio tokens + system prompt + transcript = ~1100 토큰 시퀀스를
batch_size=2, 8GPU로 학습하면 VRAM이 부족하고 step 속도가 느려진다.

**목표: 토큰 수를 줄이면서 정보를 보존한다.**

---

## 왜 Linear(MLP)가 아닌가?

가장 단순한 projector는 Linear다.
Flamingo[1]와 초기 Audio-LLM 연구들은 Linear projector를 썼지만,
이는 시퀀스 길이를 줄이지 못한다는 한계가 있다.

```python
# 차원만 맞추는 Linear
self.projector = nn.Linear(encoder.out_dim, llm_dim)
```

Linear는 각 타임스텝을 독립적으로 변환한다. 즉:
- 차원: encoder.out_dim → llm_dim ✓
- 길이: T_enc → T_enc (변화 없음) ✗

토큰 수가 그대로라 VRAM과 속도 문제가 해결되지 않는다.

---

## Conv1d stride=2의 역할

LLaSA[2]와 SALMONN[3]에서 strided convolution으로 audio 시퀀스를
다운샘플하는 방식이 효과적임을 보였다. Qwen-Audio[4]는 비슷한 목적으로
windowed attention을 활용한다.

```python
nn.Conv1d(in_dim, llm_dim, kernel_size=5, stride=2, padding=2)
```

stride=2는 **2개 프레임을 1개로 합친다**. kernel_size=5는 합칠 때
양옆 2 프레임까지 같이 보므로, 단순 평균이 아닌 학습된 가중합이다.

```
[f0, f1, f2, f3, f4, f5, f6, f7]  (T=8)
        ↓  stride=2 Conv1d
    [g0,     g1,     g2,     g3]   (T=4)
```

| encoder | projector | 출력 fps | 토큰/10초 |
|---|---|---|---|
| EnCodec (75fps) | [2, 2] → ×4 | 18.75fps | 188 tokens |
| DAC (~86fps) | [2, 2] → ×4 | ~21.5fps | ~215 tokens |
| Mimi acoustic (25fps) | [2, 2] → ×4 | 6.25fps | 63 tokens |
| Mimi semantic (25fps) | [2] → ×2 | 12.5fps | 125 tokens |

Mimi semantic만 [2]를 쓰는 이유: 이미 encoder_transformer로 압축된
고수준 피처라 ×4로 더 줄이면 정보 손실이 크다.

---

## 실제 구조

```python
# proj_strides = [2, 2] 일 때 (encodec / dac / mimi_acoustic)
nn.Conv1d(encoder.out_dim, llm_dim, kernel_size=5, stride=2, padding=2)
nn.GELU()
nn.Conv1d(llm_dim,         llm_dim, kernel_size=5, stride=2, padding=2)
nn.GELU()
nn.Conv1d(llm_dim,         llm_dim, kernel_size=1)   # 채널 믹싱 (stride=1)

# proj_strides = [2] 일 때 (mimi_semantic)
nn.Conv1d(encoder.out_dim, llm_dim, kernel_size=5, stride=2, padding=2)
nn.GELU()
nn.Conv1d(llm_dim,         llm_dim, kernel_size=1)
```

마지막 kernel_size=1 Conv1d는 stride=1이라 길이를 줄이지 않는다.
각 위치에서 채널 간 선형 변환만 수행한다 (= position-wise Linear와 동일).
LLM 입력 직전에 피처 공간을 최종 정렬하는 역할이다.

---

## 초기화 전략

```python
# 모든 Conv: Xavier uniform — fan_in/fan_out 균형 잡힌 분산
nn.init.xavier_uniform_(m.weight)

# 마지막 1×1 Conv: small-scale normal
nn.init.normal_(self.projector[-1].weight, std=0.02)
nn.init.zeros_(self.projector[-1].bias)
```

마지막 레이어를 zero-init하지 않고 std=0.02를 쓰는 이유:
zero-init이면 초기 audio embed가 LLM에 아무 신호도 주지 않아
앞쪽 Conv 레이어까지 gradient가 흐르지 않는다. std=0.02는 신호는
작게 유지하되 gradient 경로를 열어둔다.

---

## Stage 1에서 projector를 fp32로 올리는 이유

```python
# model.py: freeze_llm()
self.projector.float()
self.proj_norm.float()
```

Stage 1에서 LLM은 frozen이고 projector만 학습한다.
LLM이 fp16인 상태에서 projector를 fp16으로 두면,
gradient가 fp16 LLM을 역전파로 통과할 때 underflow가 발생할 수 있다.
projector를 fp32로 올려두면 gradient 누적이 안정적이다.

Stage 2에서는 LoRA가 적용된 LLM도 함께 학습되므로
projector를 fp32로 유지해도 되고, 명시적으로 내리지는 않는다.

---

## References

[1] Flamingo: a Visual Language Model for Few-Shot Learning.
    Alayrac et al., NeurIPS 2022. https://arxiv.org/abs/2204.14198

[2] LLaSA: Large Language and Speech Assistant.
    Shen et al., 2024. https://arxiv.org/abs/2502.04128

[3] SALMONN: Towards Generic Hearing Abilities for Large Language Models.
    Tang et al., ICLR 2024. https://arxiv.org/abs/2310.13289

[4] Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models.
    Chu et al., 2023. https://arxiv.org/abs/2311.07919
