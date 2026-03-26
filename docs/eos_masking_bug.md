# EOS 마스킹 버그 수정

**작성일: 2026-03-20**

---

## 문제 요약

학습 loss가 1 이하로 떨어졌음에도 WER이 비정상적으로 높은 현상이 관찰됨.
원인은 `model.py`의 label 구성 로직에서 **EOS 토큰이 학습 대상에서 제외**되던 버그.

---

## 배경 지식

### Causal LM의 loss 계산 방식

Causal LM은 "다음 토큰 예측"으로 학습함. 입력 시퀀스를 한 칸씩 밀어서 다음 토큰을 맞추는 방식.

```
입력:  [Audio embeds] [Transcript: "hello world"]
레이블: [-100, ..., -100, "hello", "world", EOS]
```

- `-100`인 위치는 loss 계산에서 제외 (audio, prompt 구간)
- transcript 구간만 loss 계산 대상
- **EOS 토큰을 예측하도록 학습해야 추론 시 모델이 "여기서 멈춰라"를 알 수 있음**

---

## 버그 발생 경위

### 1단계: pad_token 설정

`model.py`에서 Qwen tokenizer 로드 시 pad_token이 없으면 eos_token을 pad_token으로 설정:

```python
if self.tokenizer.pad_token is None:
    self.tokenizer.pad_token = self.tokenizer.eos_token
```

이 시점부터 `pad_token_id == eos_token_id` 가 됨.

### 2단계: collate_fn에서 EOS 수동 추가

`dataset.py`의 `collate_fn`에서 transcript를 토크나이즈한 뒤 EOS를 마지막에 붙임:

```python
text_inputs = tokenizer(texts, padding=True, truncation=True, ...)
input_ids = text_inputs.input_ids  # (B, L) — 짧은 샘플은 pad_token으로 우측 패딩

eos = torch.full((B, 1), tokenizer.eos_token_id)
input_ids = torch.cat([input_ids, eos], dim=1)  # (B, L+1)
```

이 시점의 `input_ids` 구조 (배치 내 길이가 다른 두 샘플 예시):

```
샘플 1 (짧음): [t1, t2, t3, PAD, PAD, PAD, EOS]
샘플 2 (김):   [t1, t2, t3, t4,  t5,  t6,  EOS]
```

- `PAD`와 `EOS`는 **같은 token_id**
- 모든 샘플의 마지막 열(L번째)은 항상 수동으로 붙인 EOS

### 3단계: model.py에서 label 마스킹 (버그)

```python
tgt_labels = transcript_input_ids.clone()
tgt_labels[tgt_labels == self.tokenizer.pad_token_id] = -100  # ← 버그
```

`pad_token_id == eos_token_id`이므로, 이 코드는:
- ✅ 의도: 패딩 토큰을 -100으로 마스킹
- ❌ 실제: **맨 끝의 EOS까지 -100으로 마스킹**

결과적으로 모든 샘플에서 EOS를 예측하는 loss가 0이 됨.

---

## 결과

### 학습 중

- loss는 transcript 토큰들만 대상으로 줄어듦
- EOS를 예측하는 법을 한 번도 학습하지 못함

### 추론 중 (`eval.py`)

```python
out_ids = model.llm.generate(
    ...,
    eos_token_id=model.tokenizer.eos_token_id,  # EOS 나오면 멈춤
    max_new_tokens=256,
)
```

- 모델이 EOS를 생성하지 못하므로 `max_new_tokens=256`에 도달할 때까지 토큰을 계속 생성
- transcript가 끝난 뒤에도 무의미한 토큰이 256개 추가됨
- WER은 삽입 오류(insertion error)로 인해 폭증

---

## 수정

**`model.py`의 label 마스킹 로직 변경:**

```python
# 수정 전
tgt_labels[tgt_labels == self.tokenizer.pad_token_id] = -100

# 수정 후
tgt_labels[:, :-1][tgt_labels[:, :-1] == self.tokenizer.pad_token_id] = -100
```

### 왜 `[:, :-1]`인가

`input_ids`의 마지막 열(인덱스 `-1`)은 **항상 수동으로 붙인 EOS**임.
- 짧은 샘플이든 긴 샘플이든 마지막 열은 반드시 EOS
- truncation이 발생해도 EOS는 truncation 이후에 붙으므로 마지막 열은 항상 유효한 EOS

따라서 `[:, :-1]` 범위(EOS 제외)에서만 pad 마스킹을 적용하면,
- 중간 패딩 토큰: 정상적으로 -100 마스킹
- 마지막 EOS: 마스킹 없이 학습 대상으로 유지

---

## 영향 범위

| 항목 | 영향 |
|---|---|
| Stage 1 (projector alignment) | 있음 — EOS 예측 학습 안 됨 |
| Stage 2 (LoRA fine-tuning) | 있음 — 동일 버그 |
| 학습 loss | 표면상 정상 하락 (EOS 제외 상태로 수렴) |
| 추론 WER | 크게 상승 (max_new_tokens까지 쓰레기 생성) |

---

## 비고

이 버그를 피하는 다른 방법으로, EOS와 다른 토큰을 pad_token으로 설정하는 방법도 있음:

```python
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model.resize_token_embeddings(len(tokenizer))
```

그러나 이 방법은 LLM 임베딩 크기를 바꾸고 새 토큰 임베딩 초기화가 필요해서
현재 구조에서는 `[:, :-1]` 방식이 더 간단하고 안전함.
