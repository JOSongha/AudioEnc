# Stage 1 CTC: Teacher Forcing 문제의 해결

## 1. 왜 CTC를 도입하는가

### 1-1. 현재 Stage 1의 구조적 문제

현재 Stage 1은 teacher-forced CE loss로 projector를 학습한다.

```
입력:  [p1] [audio_embeds] [p2] [tok0, tok1, ..., tok{N-1}]
label:                           [tok1, tok2, ..., tok{N}, EOS]
```

HuggingFace CausalLM은 내부적으로 1칸 shift하여 다음 토큰을 예측한다.

**문제**: `tok_i`를 예측할 때 `tok0..tok{i-1}`이 GT 문맥으로 제공된다. Frozen LLM은
이미 영어 next-token prediction이 뛰어나므로, projector가 audio를 전혀 인코딩하지 않아도
텍스트 문맥으로 loss를 낮출 수 있다.

이를 `lossAnalysis.py`로 검증한 결과:

```
Teacher-forced avg loss : 4.54 ~ 4.91
Free generation avg loss: 14.24 ~ 17.78  (3~3.6배 높음)

Free gen 예측:
  Sample 0: 'The', '\n', 'The', '\n' ... (audio 무시, degeneracy)
  Sample 1: '-', '\n', '-', '\n' ...
  Sample 2: 'transcript', 'cript', ':', '\n' ... (프롬프트 패턴 반복)
```

teacher-forced loss가 낮아진 건 projector가 audio를 잘 배웠다는 신호가 아니라
LLM이 텍스트 문맥으로 답을 찾는다는 신호다.

### 1-2. Teacher Forcing 없이 학습하려면

teacher forcing을 제거하는 방법은 두 가지다:

- **순차 생성(Method 3)**: transcript를 입력에서 제거하고 토큰을 하나씩 생성.
  inference와 동일한 방식이라 exposure bias가 없지만, N개 토큰에 N번 forward pass가
  필요해서 학습 속도가 N배 느려진다. (N≈30이면 30배)

- **CTC loss**: projector 출력에 직접 CTC supervision을 가한다.
  LLM을 전혀 거치지 않고, audio frame sequence → character sequence를 직접 학습.
  forward pass 1회로 끝난다. **이것이 CTC를 선택하는 이유다.**

---

## 2. CTC가 무엇인가

### 2-1. 핵심 아이디어

CTC(Connectionist Temporal Classification)는 입력 길이(T)와 출력 길이(U)가 다를 때
alignment 정보 없이 sequence를 학습할 수 있는 loss다.

audio frame T개 → character U개로 변환할 때, 어떤 frame이 어떤 character에 대응하는지
몰라도 된다. CTC가 가능한 모든 alignment의 확률 합계를 최대화한다.

```
입력:  [f0, f1, f2, f3, f4, f5, f6, f7]  (T=8 audio frames)
출력:  [h, i]                              (U=2 characters)

가능한 alignment 예시:
  [h, h, -, h, -, i, -, -]
  [h, -, h, -, -, i, i, -]
  [-, h, -, h, i, -, i, -]
  ...
  (- = blank 토큰)
```

blank 토큰을 사용해 반복 문자와 구분하고, monotonic alignment만 허용한다
(시간 순서가 뒤집히지 않는다는 가정 — 음성에서는 항상 성립).

### 2-2. 수식

프레임 t에서 character c를 출력할 확률:

```
P(c | f_t) = softmax(Linear(f_t))[c]
```

전체 character sequence y에 대한 CTC 확률:

```
P(y | f_1..T) = Σ_{π ∈ B^{-1}(y)} Π_t P(π_t | f_t)
```

여기서 `B^{-1}(y)`는 y로 decode되는 모든 alignment π의 집합.
이 합계를 dynamic programming(forward-backward algorithm)으로 효율적으로 계산한다.

CTC loss:

```
L_CTC = -log P(y | f_1..T)
```

### 2-3. CTC의 조건

1. **T ≥ U**: frame 수가 character 수보다 많아야 한다.
   - 이 모델: T_proj ≈ 215 tokens/10sec, 30 tokens → char 약 150자 이하. 충분.

2. **Monotonic alignment**: 음성에서는 항상 성립.

3. **Conditional independence**: 각 frame의 예측이 독립적이라 가정.
   (실제로는 아니지만 CTC가 실용적으로 잘 동작함)

---

## 3. 이 모델에서 CTC가 어떻게 작동하는가

### 3-1. 전체 흐름

```
waveform (B, T_16k)
    ↓
DAC-VAE encoder
    ↓
(B, T_enc, 256)          ← encoder output
    ↓
Projector (Conv1d ×2)
    ↓
(B, T_proj, llm_dim)     ← projector output (T_proj = T_enc / 4)
    ├─────────────────────────────────────────────────
    │  [Stage 1 only]
    │  CTC head: Linear(llm_dim, char_vocab_size)
    │      ↓
    │  (B, T_proj, 40)   ← per-frame character logits
    │      ↓
    │  CTCLoss(log_softmax(logits), char_targets)
    │      ↓
    │  L_CTC
    └─────────────────────────────────────────────────
    ↓  [Stage 2]
proj_norm
    ↓
LLM (Qwen3.5-2B)
```

**핵심**: CTC loss는 LLM을 전혀 거치지 않는다. projector output에서 바로 계산된다.
따라서 "LLM이 텍스트 문맥으로 답을 찾는" 경로 자체가 없다.

### 3-2. Stage 1 학습 목표 변화

| | 현재 (CE only) | 제안 (CE + CTC) |
|---|---|---|
| Supervision 경로 | audio → projector → LLM → CE | audio → projector → **CTC** (+ optional CE) |
| LLM이 text shortcut 가능? | 가능 | CTC 경로에서는 불가 |
| Projector gradient 출처 | pos=0 + novel words (희소) | 모든 audio frame (밀집) |
| Teacher forcing 의존? | 강하게 의존 | CTC: 없음 |
| 연산량 | forward 1회 | forward 1회 + CTC (O(T·U)) |

### 3-3. Character vocabulary

Qwen tokenizer(BPE, ~150k)를 CTC에 그대로 쓰면:
- Linear(2048, 150000) → 파라미터 300M, 메모리 1.2GB
- CTC alignment 학습이 극히 어려워짐 (vocabulary가 너무 큼)

대신 character-level vocabulary를 사용한다:

```python
CHARS = list("abcdefghijklmnopqrstuvwxyz '")  # 28자
# blank 토큰(index 0) 포함 → 총 29
```

transcript를 character sequence로 변환:

```python
"chapter one" → ['c','h','a','p','t','e','r',' ','o','n','e']
```

CTC head 크기: `Linear(llm_dim, 29)` → 파라미터 약 60k. 오버헤드 무시 가능.

### 3-4. Stage 1 loss 구성

```python
loss = ce_loss + ctc_weight * ctc_loss
```

옵션 A (hybrid): CE + CTC 동시 학습
- CE가 projector를 LLM embedding space에 align하는 역할 유지
- CTC가 audio encoding을 강제

옵션 B (CTC only): CTC만 사용
- LLM alignment가 없어서 Stage 2 초기 수렴이 느릴 수 있음
- projector가 audio를 확실히 인코딩한다는 보장은 더 강함

권장: **옵션 A**. CE는 projector output을 LLM이 쓸 수 있는 space로 유지시키고,
CTC는 그 output이 실제 audio 내용을 담도록 강제한다.

---

## 4. CTC가 해결하는 것과 해결하지 못하는 것

### 해결하는 것

1. **Teacher forcing 경로 차단**: CTC 경로에는 텍스트 문맥이 없다. projector가
   audio를 인코딩하지 않으면 CTC loss가 내려가지 않는다.

2. **밀집된 gradient**: 현재 CE만 쓸 때 projector가 유의미한 gradient를 받는 위치는
   LLM이 텍스트로 예측 못하는 novel word 위치뿐이었다. CTC는 T_proj 전체 frame에서
   gradient가 생긴다.

3. **Exposure bias 없음**: CTC는 autoregressive가 아니다. 각 frame이 독립적으로
   character를 예측하므로, 앞 위치에서 틀려도 뒤 위치 학습에 cascade되지 않는다.

### 해결하지 못하는 것

1. **Stage 2의 exposure bias**: Stage 2는 여전히 teacher-forced CE를 사용한다.
   Stage 1에서 projector가 제대로 align되면 Stage 2의 수렴은 빨라지지만,
   inference와 training 간의 gap은 완전히 해소되지 않는다.

2. **Character → BPE mismatch**: Stage 1 CTC는 character 단위로 학습하고,
   Stage 2는 BPE 단위 CE loss를 쓴다. 완전히 동일한 granularity가 아니다.
   그러나 character-level alignment가 잘 되면 BPE-level prediction도 수렴이 빨라진다
   (phoneme 정보를 담은 projector라면 BPE 예측도 쉬워짐).

---

## 5. Stage 1 → Stage 2 전환

```
Stage 1:
  학습 파라미터: projector, proj_norm, CTC head
  Loss: CE + λ·CTC

Stage 2:
  CTC head 제거 (저장 불필요)
  학습 파라미터: projector, proj_norm, LoRA
  Loss: CE (teacher forcing, 기존과 동일)
```

CTC head는 Stage 1 전용 auxiliary module이다. Stage 2에서는 사용하지 않는다.
projector가 Stage 1에서 audio를 제대로 인코딩하도록 훈련되었다면,
Stage 2에서 LoRA가 "audio를 해석하는 방향"으로 빠르게 수렴한다.
