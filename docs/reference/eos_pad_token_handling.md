# EOS / PAD 토큰 처리 방식 — Omni 파이프라인

> 대상: `src/llamafactory/data/omni_dataset.py` (Qwen3.5AE / Qwen3AE ASR Stage1 / Stage2 공통)
> 목적: ChatML 구조의 ASR 학습 시 어떤 토큰이 어디 들어가고, loss 는 어디만 계산되는지 정리.

---

## 1. 관련 토큰 4가지

| 토큰 | 값/출처 | 역할 | 정의 위치 |
|---|---|---|---|
| `eos_token_id` | `tokenizer.eos_token_id` | assistant 응답 끝 | [omni_dataset.py:238](../../src/llamafactory/data/omni_dataset.py#L238) |
| `pad_token_id` | `tokenizer.pad_token_id` | packed 시퀀스 길이 맞추기 (tail padding) | [loader.py:666](../../src/llamafactory/data/loader.py#L666) |
| `audio_pad_token_id` | 248076 — `tokenizer.audio_pad_token_id` 또는 `<\|audio_pad\|>` 토큰 ID | audio embedding 들어갈 **자리 차지용** placeholder | [loader.py:620-628](../../src/llamafactory/data/loader.py#L620-L628) |
| `IGNORE_INDEX` | `-100` (HF 관례) | loss 계산 제외 마커 | [omni_dataset.py:35](../../src/llamafactory/data/omni_dataset.py#L35), `extras/constants.py` |

`audio_pad_token_id` 는 실제 audio feature 자리에 일단 박혀 있다가, 모델 forward 안에서 projector 출력 embedding 으로 대체됨. 학습 loss 는 이 자리에서 계산되지 않아야 하므로 labels 에선 항상 `IGNORE_INDEX`.

---

## 2. 샘플별 input_ids / labels 생성

**위치**: [omni_dataset.py:422-440](../../src/llamafactory/data/omni_dataset.py#L422-L440) (`process_samples` 내부 build 루프)

### ChatML wrapper

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|audio_start|>[audio_pad × t_audio]<|audio_end|>{user_suffix_text}<|im_end|>
<|im_start|>assistant
{target_text}<eos>
```

### 실제 코드

```python
# input_ids — :422-:437
user_suffix_ids = _enc(user_suffix_text)                      # :422
target_ids      = _enc(target_text)                           # :423
...
if t_audio > 0:
    input_ids.extend(_enc(_audio_open))                       # :430 — "<|audio_start|>"
    input_ids.extend([audio_pad_token_id] * t_audio)          # :431 — audio 자리 placeholder
    input_ids.extend(_enc(_audio_close))                      # :432 — "<|audio_end|>"
input_ids.extend(user_suffix_ids)                             # :433 — "{prompt}<|im_end|>...assistant\n"

context_len = len(input_ids)                                  # :435
input_ids.extend(target_ids)                                  # :436 — 정답
input_ids.append(eos_token_id)                                # :437 — EOS

# labels — text + EOS 만 loss 기여 :439
labels = [IGNORE_INDEX] * context_len + list(target_ids) + [eos_token_id]
```

**핵심 포인트**:
- `t_audio` 는 실제 음성 길이 기준 (`num_samples // hop_length` — [:416](../../src/llamafactory/data/omni_dataset.py#L416)).
- **EOS 는 input_ids 와 labels 양쪽에 모두 포함** → loss 계산 시 "EOS 시점에서 EOS 예측하도록" 학습.
- Audio 자리 (`audio_pad_token_id × t_audio`) 은 input_ids 에만 있고, labels 엔 `IGNORE_INDEX` 로 덮여있음 → loss 기여 0.
- ChatML prefix/mid 도 전부 `IGNORE_INDEX`.

---

## 3. Packing 과 tail padding

**위치**: [omni_dataset.py:568-574](../../src/llamafactory/data/omni_dataset.py#L568-L574)

여러 샘플을 `cutoff_len` (보통 3584) 까지 greedy knapsack 으로 packing. 다 못 채우면 tail 에 pad:

```python
pad_len = cutoff_len - len(packed_input_ids)         # :569
if pad_len > 0:
    packed_input_ids.extend([pad_token_id] * pad_len)    # :571 — text pad_token 으로 채움
    packed_labels.extend([IGNORE_INDEX] * pad_len)       # :572 — loss 제외
    packed_attention_mask.extend([0] * pad_len)          # :573 — attention 무시
    packed_modality_ids.extend([MODALITY_PAD_ID] * pad_len)  # :574
```

Pad 토큰은 **text pad_token_id** 임 (`audio_pad_token_id` 가 아님) — 모델이 audio placeholder 로 착각해서 projector output 주입을 시도하면 안 되기 때문.

---

## 4. neat_packing 과 attention_mask

**위치**: [omni_dataset.py:552-555](../../src/llamafactory/data/omni_dataset.py#L552-L555) (선언은 [:488](../../src/llamafactory/data/omni_dataset.py#L488))

같은 packed sequence 안에 여러 샘플이 들어 있을 때 샘플 간 attention 을 막아야 함. 이를 위한 attention_mask 인코딩:

```python
if neat_packing:
    packed_attention_mask.extend([seq_idx + 1] * len(ids))   # :553 — 샘플마다 1, 2, 3, ...
else:
    packed_attention_mask.extend([1] * len(ids))             # :555 — 샘플 구분 없음 (단순 1)
```

| 값 | 의미 |
|---|---|
| 0 | padding (무시) |
| 1 | (neat_packing OFF) 모든 valid 토큰 |
| 1, 2, 3, ... | (neat_packing ON) 서로 다른 샘플 구분 — 같은 숫자끼리만 attend |

Tail pad 구간은 `0` ([:573](../../src/llamafactory/data/omni_dataset.py#L573)).

---

## 5. FA2 varlen 경로에서 특수처리

**위치**: [omni_dataset.py:661-682](../../src/llamafactory/data/omni_dataset.py#L661-L682) (Collator 의 `flash_attention_2` 분기)

Flash Attention 2 는 padding 을 제거하고 모든 valid 토큰을 1D 로 평탄화한 뒤 `cu_seqlens` 로 경계를 알려주는 **varlen 포맷**을 씀.

```python
non_pad = attention_mask != 0                                  # :663
result["input_ids"]  = input_ids[non_pad].unsqueeze(0)         # :664
result["labels"]     = labels[non_pad].unsqueeze(0)            # :665
result["modality_ids"] = modality_ids[non_pad].unsqueeze(0)    # :666
position_ids = batch_group_counter(attention_mask)             # :667 — 각 샘플 내에서 0, 1, 2, ...
position_ids = position_ids[non_pad].unsqueeze(0)              # :668
result["labels"][position_ids == 0] = IGNORE_INDEX             # :669 ← 각 샘플 첫 토큰 label 제거
result["position_ids"] = position_ids                          # :670

# cu_seqlens = 각 샘플 시작 위치 (FA2 의 varlen API 용)
flat_pos = position_ids[0]                                     # :676
boundary_idx = (flat_pos == 0).nonzero(as_tuple=True)[0]       # :677 — sample starts
total_len = flat_pos.numel()                                   # :678
cu_seqlens = torch.cat(                                        # :679-:681
    [boundary_idx.to(torch.int32), torch.tensor([total_len], dtype=torch.int32)]
)
result["cu_seqlens"] = cu_seqlens                              # :682
```

### `labels[position_ids == 0] = IGNORE_INDEX` 의 의미

각 샘플의 **첫 토큰 (`<|im_start|>`) 위치에서 label 을 제거**. 이유:

- Causal LM 은 `logits[i]` 로 `labels[i]` 를 예측 (HF Trainer 가 내부적으로 shift 처리).
- Packing 된 varlen 시퀀스에서 샘플 경계를 넘는 shift 는 **이전 샘플의 EOS 직후 → 다음 샘플의 `<|im_start|>` 예측** 이 되어 버림. 이건 무의미한 transition.
- 경계를 끊기 위해 다음 샘플 첫 토큰의 label 을 IGNORE_INDEX 로 바꿔서 loss 계산 제외.

---

## 6. 전체 요약 — 토큰별 역할 테이블

| 자리 | `input_ids` | `labels` | `attention_mask` | loss 기여 |
|---|---|---|---|---|
| ChatML prefix (`<|im_start|>...user\n<|audio_start|>`) | prefix tokens | `IGNORE_INDEX` | `seq_idx` | ❌ |
| Audio 자리 | `audio_pad_token_id` × `t_audio` | `IGNORE_INDEX` | `seq_idx` | ❌ (projector output 으로 대체) |
| ChatML mid (`<|audio_end|>Transcribe...assistant\n`) | mid tokens | `IGNORE_INDEX` | `seq_idx` | ❌ |
| Transcript | text_ids | text_ids | `seq_idx` | ✅ |
| **EOS** | `eos_token_id` | `eos_token_id` | `seq_idx` | ✅ **기여** |
| Sample 경계 첫 토큰 (FA2 경로) | 다음 샘플 prefix | `IGNORE_INDEX` ([:669](../../src/llamafactory/data/omni_dataset.py#L669)) | 다른 `seq_idx` | ❌ |
| Tail padding (packed < cutoff_len) | `pad_token_id` | `IGNORE_INDEX` ([:572](../../src/llamafactory/data/omni_dataset.py#L572)) | `0` | ❌ |

---

## 7. 구현 검증 체크리스트

`t_audio` 와 projector output 의 frame 수, `audio_pad_token_id` 개수 — 이 세 수가 **서로 반드시 일치** 해야 모델 입력이 정합함. 어떤 encoder 쓰든 공통 조건.

```python
# 디버그용 assert (forward 안에)
n_audio_pad_in_ids = (input_ids == audio_pad_token_id).sum()
n_proj_out_frames = sum(audio_lengths)  # 배치 내 모든 샘플의 t_audio 합
assert n_audio_pad_in_ids == n_proj_out_frames, (
    f"audio_pad_token count ({n_audio_pad_in_ids}) != projector output frames ({n_proj_out_frames})"
)
```

DACVAE 에서는 `t_audio = num_samples // hop_length` ([omni_dataset.py:416](../../src/llamafactory/data/omni_dataset.py#L416)) 로 결정. Whisper 로 바꿀 때:
- `sample_rate = 16000`
- `hop_length = 320` (50 fps)
- `t_audio = min(1500, math.ceil(num_samples / 320))` — 30s 상한 clamp
- Encoder 출력 slice 도 같은 `t_audio` 로 맞춤 (§12.1 참조).

---

## 8. 관련 파일

| 파일 | 역할 |
|---|---|
| [src/llamafactory/data/omni_dataset.py](../../src/llamafactory/data/omni_dataset.py) | processor (샘플 생성), packer (greedy knapsack), collator (tensor 변환 + FA2 unpad) |
| [src/llamafactory/data/loader.py](../../src/llamafactory/data/loader.py) | `audio_pad_token_id` / `pad_token_id` 조회, processor/packer factory 호출 |
| [src/llamafactory/extras/constants.py](../../src/llamafactory/extras/constants.py) | `IGNORE_INDEX = -100` |
