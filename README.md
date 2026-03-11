# AudioEnc — Encoder-Swappable ASR Training Framework

Qwen3.5-4B + 교체 가능한 Audio Encoder로 LibriSpeech, MLS ASR을 학습하는 프레임워크.
`--encoder encodec|dac|mimi` 인자 하나로 encoder를 바꿔 동일한 학습 파이프라인을 실행한다.

---

## 디렉토리 구조

```
AudioEnc/
├── encoders/
│   ├── __init__.py       # build_encoder() 팩토리
│   ├── base.py           # BaseAudioEncoder ABC
│   ├── encodec.py        # facebook/encodec-24khz
│   ├── dac.py            # descript-audio-codec 44kHz
│   └── mimi.py           # kyutai/mimi
├── config.py             # TRAIN_CONFIG + ENCODER_REGISTRY
├── dataset.py            # LibriSpeechDataset, MLSDataset, collate_fn_factory
├── model.py              # AudioQwen (encoder-agnostic)
├── train.py              # 2-Stage 학습 루프 (argparse)
└── run.sh                # 실행 예시
```

---

## 모듈 설계

### 1. `encoders/base.py` — 인터페이스 정의

모든 encoder가 지켜야 할 계약:

```python
class BaseAudioEncoder(nn.Module, ABC):
    out_dim: int        # encoder 출력 채널 수
    src_sr:  int = 16000  # 입력 sr (dataset은 항상 16kHz)

    def forward(audio_waveform, audio_lengths) -> (feats, enc_mask):
        # feats:    (B, T_enc, out_dim)  fp32
        # enc_mask: (B, T_enc)           bool, True=real frame
```

**불변 조건:**
- 항상 frozen (`requires_grad=False`)
- 항상 eval 모드 (`train()` override로 강제)
- fp32 출력 (V100 cuDNN LSTM 호환)
- 내부 리샘플링은 encoder 책임 (dataset은 16kHz 고정)

---

### 2. `encoders/` — 구현체

| 파일 | 클래스 | 모델 | out_dim | fps | hop |
|---|---|---|---|---|---|
| `encodec.py` | `EnCodecEncoder` | facebook/encodec-24khz | 128 | 75 | 320@24kHz |
| `dac.py` | `DACEncoder` | descript/dac-44kHz | 1024 | ~86 | 512@44kHz |
| `mimi.py` | `MimiEncoder` | kyutai/mimi | 512 | 12.5 | 1920@24kHz |

각 encoder는 `__init__(cfg: dict, cache_dir: str)` 시그니처를 따른다.
`cfg`는 `config.py`의 `ENCODER_REGISTRY[name]`에서 넘어온다.

**새 encoder 추가 방법:**
1. `encoders/myenc.py`에 `BaseAudioEncoder` 상속 클래스 작성
2. `config.py`의 `ENCODER_REGISTRY`에 항목 추가
3. `encoders/__init__.py`의 `ENCODER_CLASSES`에 등록

---

### 3. `config.py` — 설정 분리

```python
TRAIN_CONFIG = { ... }          # 학습 하이퍼파라미터 (encoder 무관)

ENCODER_REGISTRY = {
    "encodec": { "model_id": ..., "out_dim": 128, "tgt_sr": 24000, "hop": 320 },
    "dac":     { "model_type": ..., "out_dim": 1024, "tgt_sr": 44100, "hop": 512 },
    "mimi":    { "model_id": ..., "out_dim": 512, "tgt_sr": 24000, "hop": 1920 },
}
```

---

### 4. `model.py` — `AudioQwen`

encoder를 주입받아 동작하는 encoder-agnostic 모델.
projector의 입력 차원을 `encoder.out_dim`으로 자동 결정한다.

```
forward 흐름:
  audio (B, T_16k)
    → encoder(audio, lengths)           # (B, T_enc, out_dim), mask
    → projector (Conv1d ×3, stride-2×2) # (B, T_proj, llm_dim)
    → proj_norm (LayerNorm)
    → [p1 embed] + [audio embed] + [p2 embed] + [transcript embed]
    → Qwen2.5-7B → loss (transcript tokens만)
```

**projector 구조** (encoder와 무관하게 동일):
```
Conv1d(out_dim → llm_dim, k=5, s=2) + GELU
Conv1d(llm_dim → llm_dim, k=5, s=2) + GELU
Conv1d(llm_dim → llm_dim, k=1)
LayerNorm
```
총 stride=4. mask downsampling도 자동 계산 (`::proj_stride`).

**메서드:**
- `freeze_llm()` — Stage 1용, projector fp32 캐스팅 포함
- `apply_lora()` — Stage 2용, peft LoraConfig 적용

---

### 5. `dataset.py` — 데이터 (encoder 무관)

**클래스**
- `LibriSpeechDataset` — 16kHz 모노, 10초 truncate, 단일 split 래퍼
- `MLSDataset` — Multilingual LibriSpeech English, 지정 샘플 수만큼 랜덤 샘플링 (seed 고정)
- `collate_fn_factory(tokenizer)` — 오디오 패딩 + 텍스트 토크나이징 + EOS 추가

**학습 데이터 구성 (`build_datasets`)**

| 데이터셋 | Split | 규모 |
|---|---|---|
| LibriSpeech | train-clean-100 | ~100h |
| LibriSpeech | train-clean-360 | ~360h |
| LibriSpeech | train-other-500 | ~500h |
| MLS English | train (샘플링) | ~9000h |
| **합계** | | **~10,000h** |

검증은 LibriSpeech `dev-clean` 사용. 데이터는 첫 실행 시 자동 다운로드 (`download=True`).

**MLS 샘플 수 추정 기준**
- MLS English train 전체 ≈ 44,500시간 / 평균 발화 ~8초
- 9,000시간 = 32,400,000초 → `num_samples ≈ 4,050,000`
- config의 `mls_num_samples` 키로 조정 (기본값 900,000)

**config 키**
```python
cfg = {
    "data_path":      "/path/to/librispeech",   # LibriSpeech 루트
    "mls_data_path":  "/path/to/mls",            # MLS 루트 (없으면 data_path 사용)
    "max_audio_len":  160000,                    # 최대 샘플 수 (10초 @ 16kHz)
    "mls_num_samples": 4_050_000,               # MLS에서 샘플링할 발화 수
}
```

encoder가 바뀌어도 dataset은 변경 없음. 항상 16kHz 출력.

---

### 6. `train.py` — 2-Stage 학습 루프

```bash
python train.py --encoder encodec
python train.py --encoder dac
python train.py --encoder mimi
```

**Stage 1: Projector Alignment** (LLM frozen)
- optimizer: AdamW (standard)
- projector fp32 강제
- checkpoint: `s1_projector_{encoder}.pt`

**Stage 2: LoRA Fine-tuning**
- optimizer: bitsandbytes AdamW8bit (없으면 standard)
- checkpoint naming: `best_{encoder}_ckpt`, `best_{encoder}_ckpt_r1`, ...
- 자동 resume: `r1 > r0 > fresh`

**encoder별로 독립적인 checkpoint 공간** → 동일 머신에서 여러 encoder 병렬 실험 가능.

---

## 실행

```bash
# 단일 encoder
bash run.sh encodec

# 여러 encoder 순차 실험
for enc in encodec dac mimi; do
    bash run.sh $enc
done
```

---

## Encoder별 특성 비교

| | EnCodec | DAC | Mimi |
|---|---|---|---|
| 모델 | facebook/encodec-24khz | descript/dac-44kHz | kyutai/mimi |
| out_dim | 128 | 1024 | 512 |
| fps (before proj) | 75 | ~86 | 12.5 |
| fps (after proj) | ~19 | ~21 | ~3 |
| 10초 토큰 수 | ~188 | ~215 | ~31 |
| 로드 방식 | HuggingFace | dac library | HuggingFace |
| torch.load 패치 | 불필요 | 필요 (audiotools) | 불필요 |
| V100 cuDNN 이슈 | LSTM → fp32 강제 | 해당 없음 | 해당 없음 |

---

## 주요 설계 원칙

1. **인터페이스 고정**: `forward(audio, lengths) → (feats, mask)` — 학습 루프가 encoder 내부를 모름
2. **설정 중앙화**: encoder별 파라미터는 `ENCODER_REGISTRY`에만 존재, 코드 중복 없음
3. **checkpoint 격리**: encoder 이름이 checkpoint 경로에 포함되어 실험이 섞이지 않음
4. **기존 PoC 코드 보존**: q_enc4.py, q_dac_enc.py와 동일한 학습 로직, 리팩터링만 수행
