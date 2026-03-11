# AudioEnc vs PoC 원본 코드 차이 보고서

비교 대상:
- **원본**: `PoC/train/q_enc4.py`, `PoC/train/q_dac_enc.py`, `PoC/train/q_ming.py`
- **신규**: `AudioEnc/` 프레임워크 전체

차이는 **동작이 달라지는 버그/변경**과 **동작은 같지만 코드가 달라진 리팩터링**으로 구분한다.

---

## 1. 동작이 달라지는 차이 (버그 또는 의도적 변경)

### 1-1. MimiEncoder: `encoder_transformer` 누락 ⚠️ 버그

가장 큰 차이. q_ming.py는 Mimi의 acoustic encoder와 semantic transformer를 **둘 다** 통과시킨다.

**q_ming.py (원본)**
```python
enc = self.mimi.encoder(input_values)                         # acoustic: (B, 512, T_enc)
enc_t = enc.transpose(1, 2)
sem = self.mimi.encoder_transformer(enc_t).last_hidden_state  # semantic: (B, T_enc, 512)
feats = sem  # ← semantic 피처 사용
```

**AudioEnc `encoders/mimi.py` (신규)**
```python
feats = self.mimi.encoder(audio_in)   # acoustic만 호출, encoder_transformer 없음
```

결과적으로 신규 MimiEncoder는 acoustic 피처(저수준)를 출력하는 반면,
원본은 semantic 피처(고수준)를 출력한다. 동일한 Mimi 모델을 써도 표현이 다르므로
학습 결과가 달라진다.

---

### 1-2. Mimi hop 값 오류 ⚠️ 버그

프레임 마스크 계산에 사용하는 hop(encoder stride) 값이 틀렸다.

**q_ming.py (원본)**
```python
enc_stride = 960   # 25fps: 24000 / 960 = 25
# 주석: "encoder는 2x → 25fps → stride=960"
```

**AudioEnc `config.py` (신규)**
```python
"mimi": {
    "hop": 1920,   # 12.5fps — 이것은 RVQ quantizer 기준, encoder_transformer 출력은 960
```

`hop=1920`은 RVQ quantizer의 stride(12.5fps)이지, encoder_transformer 출력 기준이 아니다.
원본처럼 encoder_transformer까지 쓴다면 `hop=960`이 맞다.
encoder만 쓰더라도 acoustic encoder의 실제 stride는 960이므로 역시 틀렸다.
결과적으로 enc_mask가 모든 배치에서 잘못 계산된다.

---

### 1-3. Mimi projector 구조 차이 (stride 불일치) ⚠️ 버그

원본 q_ming.py의 projector는 **1×stride-2** (총 stride=2)이나,
신규 `model.py`는 모든 encoder에 대해 **2×stride-2** (총 stride=4)를 적용한다.

**q_ming.py (원본)**
```python
self.projector = nn.Sequential(
    nn.Conv1d(512, llm_dim, kernel_size=5, stride=2, padding=2),  # stride-2 ×1
    nn.GELU(),
    nn.Conv1d(llm_dim, llm_dim, kernel_size=1),
)
# 마스크 다운샘플: enc_mask[:, ::2]
# 25fps → 12.5fps → ~125 tokens / 10sec
```

**AudioEnc `model.py` (신규, Mimi에도 동일 적용)**
```python
self.projector = nn.Sequential(
    nn.Conv1d(encoder.out_dim, llm_dim, kernel_size=5, stride=2, padding=2),  # stride-2 ×2
    nn.GELU(),
    nn.Conv1d(llm_dim, llm_dim, kernel_size=5, stride=2, padding=2),
    nn.GELU(),
    nn.Conv1d(llm_dim, llm_dim, kernel_size=1),
)
# 마스크 다운샘플: enc_mask[:, ::4]
# 25fps → 6.25fps → ~63 tokens / 10sec (원본의 절반)
```

EnCodec, DAC는 원본도 2×stride-2를 쓰므로 차이 없음.
Mimi만 원본과 projector 구조가 다르다.

---

### 1-4. LoRA 설정 하드코딩 ⚠️ 버그

신규 `model.py`의 `apply_lora()`에서 LoRA 파라미터를 `cfg`에서 읽지 않고 하드코딩한다.

**원본 (q_enc4.py, q_dac_enc.py, q_ming.py)**
```python
lora_config = LoraConfig(
    r=CONFIG['lora_r'],
    lora_alpha=CONFIG['lora_alpha'],
    lora_dropout=CONFIG['lora_dropout'],
    target_modules=CONFIG['lora_target_modules'],
)
```

**AudioEnc `model.py` (신규)**
```python
lora_cfg = LoraConfig(
    r=16,          # 하드코딩
    lora_alpha=32, # 하드코딩
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

`config.py`의 `TRAIN_CONFIG`에서 `lora_r` 등을 바꿔도 반영되지 않는다.
`model.py`가 `cfg`를 `__init__`에서 받음에도 `apply_lora()`에서 참조하지 않는다.

---

### 1-5. Stage 2 기본 epoch 수 차이

원본별로 `stage2_epochs`가 다르지만, 신규 `config.py`는 16으로 통일했다.

| encoder | 원본 stage2_epochs | AudioEnc stage2_epochs |
|---|---|---|
| encodec | 16 (q_enc4.py) | 16 ✓ |
| dac | **8** (q_dac_enc.py) | **16** ← 다름 |
| mimi | **8** (q_ming.py) | **16** ← 다름 |

DAC, Mimi를 신규 프레임워크로 돌리면 원본보다 2배 더 학습된다.

---

## 2. 동작은 같지만 코드가 달라진 부분 (리팩터링)

### 2-1. Mimi AutoFeatureExtractor 제거

**q_ming.py (원본)**: `AutoFeatureExtractor`로 numpy 변환 후 모델 입력 생성
```python
audio_np_list = [audio_24k[i].cpu().float().numpy() for i in range(B)]
fe_out = self.feature_extractor(audio_np_list, sampling_rate=self.tgt_sr, ...)
input_values = fe_out.input_values.to(device=device, dtype=target_dtype)
```

**AudioEnc `encoders/mimi.py` (신규)**: `AF.resample()` 직접 사용
```python
audio_tgt = AF.resample(audio_waveform.float(), self.src_sr, self.tgt_sr)
audio_in  = audio_tgt.unsqueeze(1).float().contiguous()
```

AutoFeatureExtractor는 내부적으로 리샘플 + 정규화(mean 제거, std 정규화)를 할 수 있다.
원본에서 이미 padded tensor를 numpy로 변환해서 feature_extractor에 넘기므로
실제 동작은 거의 동일하다고 볼 수 있으나, 엄밀히는 동일하다고 보장하기 어렵다.
GPU 상에서 처리해 CPU 왕복이 없어진 것은 성능 개선이다.

---

### 2-2. `apply_lora()`에서 `proj_norm` trainable 설정 추가

**q_enc4.py, q_dac_enc.py (원본)**: `apply_lora()`에서 `proj_norm`을 빠뜨림
```python
def apply_lora(self):
    ...
    for param in self.projector.parameters():
        param.requires_grad = True
    # proj_norm 없음
```

**q_ming.py (원본)**: `proj_norm`도 포함
```python
for param in self.projector.parameters():
    param.requires_grad = True
for param in self.proj_norm.parameters():
    param.requires_grad = True
```

**AudioEnc `model.py` (신규)**: `proj_norm` 포함 (q_ming.py 방식)
```python
for p in self.projector.parameters():
    p.requires_grad = True
for p in self.proj_norm.parameters():
    p.requires_grad = True
```

`freeze_llm()`에서 이미 `proj_norm`을 trainable로 설정하므로
Stage 2 시작 시 `apply_lora()` 전에 새 모델을 만들면 `proj_norm`이 frozen 상태가 된다.
따라서 `apply_lora()`에서 명시적으로 켜는 것이 올바르다.
신규 코드가 q_enc4.py의 누락을 수정한 것으로 볼 수 있다.

---

### 2-3. Stage 1 이후 VRAM 해제 방식 차이

**q_enc4.py (원본)**: 기본적인 해제
```python
del model, optimizer1, train_loader1, val_loader1, scheduler1, enc
gc.collect()
torch.cuda.empty_cache()
accelerator.free_memory()
```

**q_dac_enc.py (원본)**: accelerator 내부 참조까지 명시적 해제
```python
accelerator.free_memory()
accelerator._models.clear()
accelerator._optimizers.clear()
accelerator._dataloaders.clear()
del model, ...
```

**AudioEnc `train.py` (신규)**: q_dac_enc.py 방식을 따르되 `hasattr` 체크 추가
```python
if hasattr(accelerator, "free_memory"):
    accelerator.free_memory()
for attr in ("_models", "_optimizers", "_dataloaders"):
    getattr(accelerator, attr, [None]).clear() if hasattr(accelerator, attr) else None
```

`getattr(accelerator, attr, [None]).clear()` 패턴은 attribute가 없을 때
`[None]`이라는 새 리스트를 만들어 clear하는 것이므로 의미 없는 동작이다.
다음과 같이 쓰는 것이 더 명확하다:
```python
if hasattr(accelerator, attr):
    getattr(accelerator, attr).clear()
```

---

### 2-4. `global_step` 처리 구조

**원본**: Stage 1 함수와 Stage 2 코드가 하나의 `train()` 함수 안에 있어 변수 공유
```python
# Stage 1에서 global_step 증가
# Stage 1 skip 시:
if skip_stage1:
    global_step = 0  # Stage 2 초반에 설정
```

**AudioEnc (신규)**: `run_stage1()`이 `(proj_path, global_step)`을 반환해 명시적으로 전달
```python
proj_path, step_offset = run_stage1(...)
run_stage2(..., step_offset=step_offset)
```

동작은 동일하나 변수 스코프가 명확해졌다.

---

### 2-5. `collate_fn_factory` 시그니처

**원본**: `max_text_len`을 전역 `CONFIG`에서 참조
```python
def collate_fn_factory(tokenizer):
    def collate_fn(batch):
        ...
        max_length=CONFIG['max_text_len'],
```

**AudioEnc `dataset.py` (신규)**: 명시적 파라미터로 수정
```python
def collate_fn_factory(tokenizer, max_text_len: int = 256):
```

전역 의존성 제거. 동작은 동일.

---

### 2-6. `make_scheduler` 시그니처

**원본**: 전역 `CONFIG`에서 `gradient_accumulation_steps`, `warmup_ratio` 참조
```python
def make_scheduler(optimizer, dataloader, epochs, accelerator):
    total_steps = (len(dataloader) // CONFIG['gradient_accumulation_steps']) * epochs
```

**AudioEnc (신규)**: `cfg`를 명시적으로 받음
```python
def make_scheduler(optimizer, dataloader, epochs, cfg, accelerator):
    total_steps = (len(dataloader) // cfg["gradient_accumulation_steps"]) * epochs
```

동작 동일, 전역 의존성 제거.

---

## 요약표

| # | 구분 | 파일 | 원본 동작 | 신규 동작 | 심각도 | 수정 |
|---|---|---|---|---|---|---|
| 1-1 | **버그** | `encoders/mimi.py` | encoder + encoder_transformer | encoder만 | 🔴 높음 | ✅ mimi_acoustic / mimi_semantic 분리 |
| 1-2 | **버그** | `config.py` | Mimi hop=960 | Mimi hop=1920 | 🔴 높음 | ✅ hop=960으로 수정 |
| 1-3 | **버그** | `model.py` | Mimi projector: 1×stride-2 | 모두 2×stride-2 | 🔴 높음 | ✅ proj_strides=[2] for mimi_semantic |
| 1-4 | **버그** | `model.py` | LoRA cfg에서 읽음 | 하드코딩 | 🟡 중간 | ✅ self._cfg에서 읽도록 수정 |
| 1-5 | **변경** | `config.py` | DAC/Mimi stage2_epochs=8 | 16으로 통일 | 🟡 중간 | ✅ encoder별 stage2_epochs override 추가 |
| 2-1 | 리팩터링 | `encoders/mimi.py` | AutoFeatureExtractor 경유 | AF.resample 직접 | ⚪ 낮음 | 유지 (성능 개선) |
| 2-2 | 개선 | `model.py` | q_enc4/dac: proj_norm 미포함 | proj_norm 포함 | ⚪ 낮음 | 유지 (버그 수정) |
| 2-3 | 리팩터링 | `train.py` | encoder별 VRAM 해제 방식 상이 | 통일 (단, 패턴 어색) | ⚪ 낮음 | ✅ hasattr 패턴 정리 |
| 2-4 | 리팩터링 | `train.py` | global_step 변수 공유 | 반환값으로 명시 전달 | ⚪ 낮음 | 유지 |
| 2-5 | 리팩터링 | `dataset.py` | CONFIG 전역 참조 | 파라미터로 수신 | ⚪ 낮음 | 유지 |
| 2-6 | 리팩터링 | `train.py` | CONFIG 전역 참조 | cfg 파라미터로 수신 | ⚪ 낮음 | 유지 |

**모든 항목 수정 완료.**
