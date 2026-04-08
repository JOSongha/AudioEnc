# Mimi: hop 버그와 projector 불일치

원본 AudioEnc 프레임워크 초기 버전에 존재했던 두 가지 버그.
모두 수정 완료됐지만, 왜 틀렸는지 이해하기 위한 기록.

---

## 배경: Mimi의 temporal 구조

Mimi는 EnCodec, DAC와 달리 내부에 두 개의 뚜렷한 temporal stride를 가진다.

```
오디오 (24kHz, T_samples 샘플)
    │
    ▼
┌──────────────────────────────────────┐
│  acoustic encoder                    │
│  CNN 기반, stride=960                 │
│  → T_enc = ceil(T_samples / 960)     │   25fps
│  → (B, 512, T_enc)                   │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  encoder_transformer                 │
│  Transformer, sequence length 유지    │
│  → (B, T_enc, 512)  ← 동일 T_enc      │   25fps (fps 변화 없음)
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  RVQ quantizer                       │
│  stride=2 추가 다운샘플                 │
│  → T_q = ceil(T_enc / 2)             │   12.5fps
│  → (B, n_codebooks, T_q) int codes   │
└──────────────────────────────────────┘
```

중요한 사실:
- acoustic encoder와 encoder_transformer는 **동일한 temporal resolution**을 가진다 (25fps)
- RVQ quantizer에서 추가로 2× 다운샘플이 일어나 12.5fps가 된다
- 우리가 쓰는 것은 encoder_transformer 출력 (25fps), RVQ 결과(12.5fps)가 아니다

---

## 버그 1: hop=1920 오류

### hop이 하는 일

배치 학습에서 오디오를 패딩으로 맞추면, 각 샘플마다 실제 유효한 프레임이
어디까지인지 알아야 한다. `hop`은 이 계산에 쓰인다:

```python
# dataset.py / encoder forward 내부
lengths_enc = (audio_lengths_24k.float() / hop).ceil().long()
enc_mask = torch.arange(T_enc, device=device).unsqueeze(0) < lengths_enc.unsqueeze(1)
#           ↑ T_enc 길이의 인덱스 배열    ↑ 샘플별 유효 길이
# 결과: (B, T_enc) bool — True = 실제 프레임, False = 패딩
```

예시 (hop=960, 배치 내 두 샘플):
```
샘플A: 10초 → lengths_24k = 240000 → lengths_enc = ceil(240000/960) = 250
샘플B:  6초 → lengths_24k = 144000 → lengths_enc = ceil(144000/960) = 150

enc_mask:
  A: [T T T T T ... T T T]  (250개 True, 이후 False)
  B: [T T T T T ... T F F]  (150개 True, 100개 False)
```

### 왜 hop=960인가

acoustic encoder의 총 temporal stride를 계산하면:

```
Mimi acoustic encoder는 내부적으로 여러 Conv1d 레이어를 쌓는다.
각 레이어의 stride 곱 = 총 downsampling factor

예: stride [2, 2, 4, 5, 6] = 2×2×4×5×6 = 960  →  24000 / 960 = 25fps
```

encoder_transformer는 self-attention 기반이라 sequence 길이를 바꾸지 않는다.
따라서 encoder_transformer 출력도 여전히 25fps → hop=960이 맞다.

### hop=1920을 쓰면 실제로 무슨 일이 벌어지는가

```python
# 버그 있는 코드
lengths_enc = (audio_lengths_24k.float() / 1920).ceil().long()
```

10초 샘플 기준:
```
올바른 계산: ceil(240000 / 960)  = 250   실제 프레임 수
버그 계산:   ceil(240000 / 1920) = 125   ← 절반
```

enc_mask가 `True`인 구간이 실제의 절반뿐이다:
```
올바른 mask: [T T T T ... T T T T T T]  (250개 True)
버그 mask:   [T T T T ... T F F F F F]  (125개 True, 125개 False)
                              ↑ 여기서부터 패딩으로 취급
```

LLM attention 계산 시 `audio_mask.long()`이 그대로 들어가므로,
뒤쪽 125 프레임은 attention에서 완전히 무시된다.
**5초 이후 오디오는 학습에 아무 기여도 하지 않는다.**

짧은 샘플은 그나마 덜 손해지만, 긴 샘플일수록 버려지는 비율이 커진다.

### 왜 1920이라는 숫자가 등장했는가

Mimi의 `encode()` 메서드 (RVQ 포함 full pipeline)의 결과 fps가 12.5fps다:
```
24000 / 12.5 = 1920   ← RVQ quantizer까지 거친 stride
```

모델 문서나 코드를 처음 읽을 때 `encode()` 결과 기준으로 stride를 읽으면
1920이 나온다. 하지만 우리는 `encode()`가 아닌 `encoder()` + `encoder_transformer()`를
직접 호출하기 때문에 RVQ quantizer의 stride는 관계없다.

### 수정

```python
# config.py — 수정 후
"mimi_acoustic": { "hop": 960 },   # acoustic encoder stride @ 24kHz
"mimi_semantic": { "hop": 960 },   # encoder_transformer도 fps 동일
```

---

## 버그 2: 2×stride-2 projector 적용

### 각 encoder의 fps가 다르다

projector의 역할은 encoder 출력 시퀀스를 LLM이 처리하기 좋은 길이로 줄이는 것이다.
그런데 encoder마다 출발 fps가 다르다:

| encoder | 출력 fps | 10초 기준 프레임 수 |
|---|---|---|
| EnCodec | 75fps | 750 frames |
| DAC | ~86fps | ~860 frames |
| Mimi acoustic | 25fps | 250 frames |
| Mimi semantic | 25fps | 250 frames |

EnCodec, DAC는 fps가 높아 적극적으로 줄여도 여전히 정보가 충분하다.
Mimi는 이미 25fps로 낮은 상태에서 시작한다.

### 음소 단위로 생각하기

한국어/영어 평균 음소 길이: ~80ms
초당 음소 수: ~12.5개

| fps | 프레임당 음소 | 의미 |
|---|---|---|
| 75fps | 0.17개 | 한 음소에 프레임 6개 → 여유 많음 |
| 25fps | 0.5개 | 한 음소에 프레임 2개 → 이미 촘촘 |
| 12.5fps | 1.0개 | 한 음소당 프레임 1개 → 경계선 |
| 6.25fps | 2.0개 | 프레임 하나에 음소 2개 → 구분 불가 |

Mimi 25fps → projector ÷4 → 6.25fps는 음소 구분이 불가능한 수준이다.
반면 EnCodec 75fps → projector ÷4 → 18.75fps는 여전히 충분하다.

### 초기 코드의 문제

초기 model.py는 encoder 종류를 보지 않고 항상 2×stride-2를 사용했다:

```python
# 초기 model.py — 모든 encoder에 동일 적용 (잘못됨)
self.projector = nn.Sequential(
    nn.Conv1d(encoder.out_dim, llm_dim, kernel_size=5, stride=2, padding=2),
    nn.GELU(),
    nn.Conv1d(llm_dim, llm_dim, kernel_size=5, stride=2, padding=2),  # ← Mimi에는 과함
    nn.GELU(),
    nn.Conv1d(llm_dim, llm_dim, kernel_size=1),
)
```

그 결과:

| encoder | 버그 상태 토큰 수 | 올바른 토큰 수 | 차이 |
|---|---|---|---|
| EnCodec | 188 | 188 | 동일 (원본도 ÷4) |
| DAC | ~215 | ~215 | 동일 (원본도 ÷4) |
| Mimi semantic | **63** | **125** | **절반** |

mask downsampling도 연동해서 틀렸다:
```python
# 버그: projector stride=4이므로 enc_mask[:, ::4]
# → 4 프레임마다 1개만 유효로 취급 → 실제보다 훨씬 적은 토큰
```

### 원본 q_ming.py의 정상 설정

```python
# q_ming.py (원본) — Mimi는 1×stride-2
self.projector = nn.Sequential(
    nn.Conv1d(512, llm_dim, kernel_size=5, stride=2, padding=2),
    nn.GELU(),
    nn.Conv1d(llm_dim, llm_dim, kernel_size=1),  # stride=1
)
# 마스크: enc_mask[:, ::2]
# 25fps → 12.5fps → 125 tokens / 10sec
```

### 수정: encoder별 proj_strides

`config.py`에 `proj_strides` 키를 추가해 encoder마다 다르게 설정:

```python
ENCODER_REGISTRY = {
    "encodec":       { "proj_strides": [2, 2], ... },   # 75fps  → 18.75fps (188 tokens)
    "dac":           { "proj_strides": [2, 2], ... },   # ~86fps → ~21.5fps (~215 tokens)
    "mimi_acoustic": { "proj_strides": [2, 2], ... },   # 25fps  → 6.25fps  (63 tokens)
    "mimi_semantic": { "proj_strides": [2],    ... },   # 25fps  → 12.5fps  (125 tokens) ← 원본 동일
}
```

`mimi_acoustic`에 [2, 2]를 쓰는 이유: acoustic 피처는 semantic보다 밀도가 낮고
인접 프레임 간 변화가 완만하므로 ÷4 압축에 비교적 강건하다. 또한 mimi_semantic보다
절반의 토큰으로 더 빠르게 실험할 수 있다는 실용적인 이유도 있다.

`model.py`는 이 리스트를 읽어 projector를 동적으로 빌드한다:

```python
proj_strides = cfg["encoder"].get("proj_strides", [2, 2])
layers = []
in_dim = encoder.out_dim
for stride in proj_strides:
    layers.append(nn.Conv1d(in_dim, llm_dim, kernel_size=5, stride=stride, padding=2))
    layers.append(nn.GELU())
    in_dim = llm_dim
layers.append(nn.Conv1d(llm_dim, llm_dim, kernel_size=1))
self.projector = nn.Sequential(*layers)

# 총 stride 자동 계산 — mask downsampling에 사용
self._proj_stride = 1
for m in self.projector.modules():
    if isinstance(m, nn.Conv1d):
        self._proj_stride *= m.stride[0]
# proj_strides=[2,2] → self._proj_stride=4
# proj_strides=[2]   → self._proj_stride=2
```

---

## 두 버그가 동시에 작동할 때

hop=1920 + proj_strides=[2,2] (초기 버그 상태)에서 mimi_semantic을 학습하면:

| 단계 | fps | 10초 유효 토큰 |
|---|---|---|
| encoder_transformer 출력 (정상) | 25fps | 250 |
| hop=1920 버그 → enc_mask | — | 125 (절반 마스킹) |
| proj_strides=[2,2] → projector | 6.25fps | ~32 (마스킹된 125의 1/4) |
| 정상 상태 (q_ming.py 기준) | 12.5fps | 125 |

**정상 대비 약 25%만 LLM에 전달된다.**

더 심각한 것은 이 상태에서 학습이 "돌아가기는 한다"는 점이다.
loss가 내려가고 수렴하는 것처럼 보이지만, 실제로는 truncated audio에
과적합된 잘못된 모델을 만들고 있다. 디버깅 없이는 버그를 눈치채기 어렵다.

---

## 체크리스트: 새 encoder 추가 시 hop 확인 방법

```python
import torch
from transformers import MimiModel  # 또는 해당 모델

model = MimiModel.from_pretrained("kyutai/mimi")
dummy = torch.zeros(1, 1, 24000)  # 1초 @ 24kHz

with torch.no_grad():
    out = model.encoder(dummy)

if not isinstance(out, torch.Tensor):
    out = out.last_hidden_state

print(f"입력 샘플 수: {dummy.shape[-1]}")
print(f"출력 프레임 수: {out.shape[-1] if out.ndim == 3 and out.shape[1] != 512 else out.shape[1]}")
# 24000 / 출력 프레임 수 = hop
```

이 결과로 계산한 값을 `ENCODER_REGISTRY`의 `hop`에 설정한다.
