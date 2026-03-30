# Stage 1 Loss 분석: Teacher Forcing과 CE Loss의 함정

## 배경

Stage 1이 끝난 시점에서 train loss가 ~1에 가까웠음에도 WER이 193%였던 현상에 대한 분석.

---

## 1. 모델 구조 요약

입력 시퀀스:
```
[p1 prompt] [audio embeddings] [p2 prompt] [transcript tokens]
```

- `audio embeddings`: 오디오 → encoder → **projector** → proj_norm 경로
- `p1`, `p2`, `transcript`: LLM 자체 embedding layer 경유 (**projector를 거치지 않음**)

Stage 1에서 학습되는 파라미터: `projector`, `proj_norm`만 (LLM은 frozen)

---

## 2. CE Loss가 왜 낮아지는가

`model.py`의 forward에서 labels는 다음과 같이 구성된다:

```
labels = [-100 * len_ctx | tok1, tok2, ..., tokN, EOS]
```

HuggingFace CausalLM은 내부적으로 1칸 shift하여 loss를 계산한다:

```
logit[len_ctx-1]  → 예측: tok1     ← audio만 있음 (projector 필요)
logit[len_ctx+0]  → 예측: tok2     ← audio + tok1
logit[len_ctx+1]  → 예측: tok3     ← audio + tok1 + tok2
...
logit[len_ctx+N-2] → 예측: tokN    ← audio + tok1 + ... + tok(N-1)
```

**핵심 문제**: 첫 번째 토큰 예측을 제외한 나머지 N-1개 예측은 이전 transcript 토큰들을 문맥으로 사용한다. Frozen 2B LLM은 영어 텍스트 next-token prediction이 이미 뛰어나므로, projector가 audio를 전혀 인코딩하지 못해도 이 N-1개 예측에서 낮은 loss를 달성할 수 있다.

Qwen vocabulary size ~150k이면 랜덤 baseline loss ≈ ln(150000) ≈ 11.9이다. Projector가 쓰레기를 출력해서 첫 토큰 예측 loss가 11.9여도:

```
(1 × 11.9 + 49 × 0.8) / 50 ≈ 1.02
```

loss ~1이 달성된다. **이는 projector가 audio를 잘 배웠다는 신호가 아니라, LLM이 영어를 잘 한다는 신호다.**

---

## 3. Projector가 받는 Gradient가 왜 작은가

모든 위치에서 gradient가 projector로 흐르지만, `logit[len_ctx+i]`에서 오디오 임베딩이 예측에 기여하는 비중은 이전 텍스트 문맥이 길수록 0에 수렴한다. Frozen LLM은 텍스트 문맥에서 답을 거의 다 찾기 때문이다.

결과적으로 projector에 실질적으로 흐르는 gradient는 **첫 번째 토큰 예측 포지션에서 오는 것이 대부분**이고, 나머지 N-1개 포지션의 loss는 LLM의 언어 모델링 능력이 대부분 설명해버린다.

---

## 4. CE Loss ↓ ≠ WER ↓ (Exposure Bias)

**CE loss (teacher forcing)가 측정하는 것:**
> "GT 앞 토큰들이 주어졌을 때, 다음 토큰을 얼마나 잘 맞추나?"

**WER (inference)이 측정하는 것:**
> "오디오만 주어졌을 때, 처음부터 끝까지 스스로 생성한 문장이 GT와 얼마나 같나?"

```
학습 때: [audio, "chapter", "one", "missus"] → "rachel" 예측 ✓
추론 때: [audio, "The", "transcript", "of"]  → ??? (이미 틀린 문맥에서 예측)
```

추론 시 첫 토큰이 틀리면, 두 번째 토큰은 GT와 전혀 다른 분기에서 예측된다. CE loss가 낮아진 건 "GT 문맥이 주어졌을 때 다음 토큰을 잘 맞추게 됐다"는 것이지, "틀린 문맥에서도 복구할 수 있다"는 뜻이 아니다.

이를 **exposure bias**라고 부른다. 학습 중에 자신의 실수를 본 적이 없으니, 추론에서 실수가 나오면 오류가 cascade된다.

정리하면: **CE loss ↓ ≠ WER ↓**, 왜냐하면 두 개가 측정하는 조건 자체가 다르기 때문이다.

---

## 5. Stage 1이 나쁘면 Stage 2도 수렴이 나쁜 이유

Stage 2에서도 teacher forcing이 그대로 사용된다. Stage 2는 LoRA가 trainable이라서 frozen LLM보다 audio를 더 잘 무시할 수 있다. LoRA가 "텍스트 문맥으로 다음 토큰 예측"에 빠르게 적응해버리면, projector는 여전히 meaningful한 gradient를 못 받는다.

```
Stage 1: frozen LLM이 텍스트로 loss 낮춤   → projector 학습 안 됨
Stage 2: trainable LoRA가 텍스트로 loss 낮춤 → projector 학습 여전히 약함
```

Stage 1의 역할은 projector를 LLM embedding space에 align시켜 놓는 것이다. projector가 제대로 align되어 있어야 Stage 2에서 LoRA가 "audio를 해석하는 방향"으로 빠르게 수렴할 수 있는데, projector가 쓰레기를 출력하면 LoRA도 방향을 못 잡는다.

결국 두 스테이지 모두 teacher-forcing이 근본 문제다. Stage 1에서 projector가 제대로 학습되지 않으면 Stage 2도 같은 덫에 빠진다.

---

## 6. 가설 검증 결과 (lossAnalysis.py)

`lossAnalysis.py`를 사용해 `s1_proj_dac_vae_ep2_step41951_best.pt` 체크포인트에서
LibriSpeech `train-clean-100` 샘플 3개에 대한 per-step CE loss를 측정했다.
각 위치에서 teacher-forced loss와 free generation loss를 비교한다.

### 측정 결과 요약

| Sample | TF avg | Free avg | Free / TF |
|--------|--------|----------|-----------|
| 0 | 4.54 | 15.87 | 3.49× |
| 1 | 4.82 | 14.24 | 2.95× |
| 2 | 4.91 | 17.78 | 3.62× |

free generation loss가 teacher-forced loss의 약 3~3.6배다.

### 확인된 가설

**CE Loss ↓ ≠ WER ↓ (Exposure Bias, §4)**
Free generation의 예측 토큰이 일관된 degeneracy를 보인다:

- Sample 0: `'The'`, `'\n'`, `'The'`, `'\n'` 교대 반복
- Sample 1: `'-'`, `'\n'`, `'-'`, `'\n'` 교대 반복
- Sample 2: `'transcript'`, `'cript'`, `':'`, `'\n'` 교대 반복

첫 토큰이 틀리는 순간 오류가 cascade되어 무의미한 패턴으로 collapse한다.
이는 exposure bias가 실제로 작동하고 있음을 직접 보여준다.

**LLM language prior의 영향**
`p2 = "\nTranscript:\n"` 직후 첫 토큰 예측이므로, 프로젝터가 audio를 전혀 활용하지 않아도 LLM은 "영어 단어가 나올 것"이라는 prior를 갖는다.
따라서 첫 토큰 TF loss는 random baseline(ln(150000) ≈ 11.9)보다 이미 낮다(5~8).

### 수정이 필요한 가설

**"Projector gradient가 주로 pos=0에서만 온다" (§3) — 과도한 단순화**

측정 결과 pos=0의 TF loss는 5~8 수준으로, 같은 샘플 내 다른 위치와 크게 다르지 않다.

```
Sample 0:
  pos= 0 (chapter)   TF=7.97   ← audio only
  pos= 2 (miss)      TF=11.12
  pos= 3 (us)        TF=8.06
  pos= 9 (surprised) TF=7.69
  pos=10 (miss)      TF=11.62
  pos=11 (us)        TF=0.11   ← "rachel lynde" 반복, LLM이 copy
  pos=13 (achel)     TF=0.007
  pos=15 (nde)       TF=0.003
```

실제 gradient 분포는 더 복잡하다:

- **novel word 위치** (처음 등장하는 단어): TF loss 7~12 → projector에 gradient 전달
- **반복 패턴 위치** (앞서 나온 이름, 문구): TF loss ≈ 0 → projector에 gradient 거의 없음

따라서 더 정확한 표현은:

> projector에 실질적인 gradient가 흐르는 위치는 **LLM이 text context만으로 예측할 수 없는
> 모든 novel word 위치**이며, pos=0은 그 중 하나일 뿐이다.

다만 이 gradient도 여전히 약하다. 각 위치에서 audio embedding의 인과적 기여도는
텍스트 context가 길어질수록 감소하기 때문이다 (frozen LLM은 텍스트 경로를 선호).

---

## 7. 해결 방법

### 방법 1: Stage 1 없이 Stage 2만 (가장 빠른 검증)

Stage 1이 projector를 제대로 학습시키지 못한다면, Stage 1을 거치지 않고 처음부터 LoRA + projector를 함께 학습시키는 것이 더 나을 수 있다. LoRA가 trainable이면 LLM이 audio 임베딩을 무시하면 loss가 안 내려가기 때문에 projector에 더 강한 학습 신호가 전달된다.

### 방법 2: Stage 1에서 첫 토큰만 loss 계산

`model.py`의 labels에서 첫 번째 transcript 토큰 이후를 모두 -100으로 마스킹한다. 그러면 projector가 오로지 audio를 인코딩하는 방향으로만 학습된다.

단점: 첫 토큰만 loss를 계산하면, 배치 내 샘플당 gradient가 딱 1개만 생긴다. 보통 teacher forcing은 토큰 N개 × 배치 크기만큼 gradient가 흐르는데, 1/N로 줄어드는 것이다. 신호가 너무 적어서 학습이 불안정하고 느려진다.

### 방법 3: Stage 1에서 transcript를 입력에서 제거

`[p1, audio_embeds, p2]`만 입력으로 주고 transcript 전체를 label로 supervise한다. Teacher forcing이 없어지므로 projector가 audio에 온전히 의존해야 한다.

단점: transcript를 입력에서 빼면, 학습 중에 모델이 스스로 토큰을 생성해야 한다:

```
[audio] → "chapter" 생성
[audio, "chapter"] → "one" 생성
[audio, "chapter", "one"] → "missus" 생성
...
```

이게 바로 추론(inference)과 같은 방식이다. 토큰을 순서대로 하나씩 생성해야 해서 병렬 처리가 불가능하고, teacher forcing은 N개 토큰을 한 번에 처리하지만 이 방식은 N번 forward pass가 필요하다. 학습 속도가 N배 느려진다. 즉, 학습이 추론처럼 sequential해져서 실용적으로 너무 느리다.
