# Experiment Setup

## Overview

Audio Encoder 종류에 따른 ASR 성능 비교 실험.
공통 백본(Qwen3.5-2B Base, `--llm`으로 교체 가능)과 데이터를 고정하고, audio encoder만 교체하여 비교.

---

## Data

| 키 | 데이터셋 (HuggingFace ID) | Split | 규모 | 특징 |
|---|---|---|---|---|
| `ls100` | `openslr/librispeech_asr` | train-clean-100 | ~100h | 낭독체, clean |
| `ls360` | `openslr/librispeech_asr` | train-clean-360 | ~360h | 낭독체, clean |
| `ls500` | `openslr/librispeech_asr` | train-other-500 | ~500h | 낭독체, noisy |
| `mls` | `parler-tts/mls_eng_10k` | train | ~10,000h | 다국어 낭독체 (EN), OGG-Opus |
| `gs` | `speechcolab/gigaspeech` (xl) | train | ~10,000h | 오디오북·팟캐스트·YouTube 혼합 |
| `vp` | `facebook/voxpopuli` (en) | train | ~500h | 유럽의회 연설, 자발화 |
| **학습 합계** | | | **~21,460h** | |
| — | `openslr/librispeech_asr` | dev-clean | — | 검증 (WER) |

- 전처리: 16kHz mono, 최대 **20초** truncate (초과분 버림)
- `train_pipeline_override.py`: 스트리밍 모드 — 사전 다운로드 불필요, `interleave_datasets`로 균등 혼합
- GigaSpeech: 노이즈 태그(`<NOISE>`, `<COMMA>` 등) 정규식으로 제거
- MLS / VoxPopuli: OGG-Opus 포맷 → `Audio(decode=False)` + `torchaudio.load()` (soundfile 미지원)
- 기본값: `--datasets` 미지정 시 6개 전체 사용

### Word Alignment 데이터

전체 학습 데이터셋에 대해 단어 단위 타임스탬프(start/end)가 사전 생성되어 있음.

| 데이터셋 | 도구 | 발화 수 | 경로 |
|---|---|---|---|
| LibriSpeech (ls100/360/500/dev) | wav2vec2-large CTC | 283,944 | `word_alignments_merged/librispeech/{split}.arrow` |
| MLS | Qwen3-ForcedAligner-0.6B | 2,420,047 | `word_alignments_merged/mls/train.arrow` |
| GigaSpeech | Qwen3-ForcedAligner-0.6B | 8,282,987 | `word_alignments_merged/gigaspeech/train.arrow` |
| VoxPopuli | Qwen3-ForcedAligner-0.6B | 182,482 | `word_alignments_merged/voxpopuli/train.arrow` |

기본 경로: `/mnt/tmp/cache/word_alignments_merged/{dataset}/{split}.arrow`  
자세한 내용 및 품질 통계: [`docs/word_alignment.md`](word_alignment.md)

---

## Backbone LLM

| 항목 | 값 |
|---|---|
| 모델 | `Qwen/Qwen3.5-2B` (Base, 기본값) |
| 파라미터 | **~2.0B** |
| hidden_size | 2,048 |
| num_hidden_layers | 24 |
| 학습 방식 | Stage 1: frozen / Stage 2: LoRA (r=16) |
| LoRA 학습 파라미터 | ~5M (q/k/v/o_proj × 24 layers) |
| dtype | bf16 |
| 프롬프트 포맷 | `"Audio:\n"` + audio embeds + `"\nTranscript:\n"` + transcript |

> `--llm <모델명>` 인자로 실행 시 임의의 HuggingFace 모델로 교체 가능 (예: `Qwen/Qwen3.5-0.8B`).
> config의 `llm_model` 키를 직접 수정해도 된다.

---

## Audio Encoders

모든 encoder는 **frozen** (학습 중 가중치 고정).

### Acoustic Encoders

오디오의 저수준 음향 특징을 추출. 음색·음질에 민감.

#### 1. DAC (Descript Audio Codec)

| 항목 | 값 |
|---|---|
| 모델 | `descript/dac-44khz` |
| 전체 파라미터 | ~74M |
| 사용 부분 | encoder only (pre-RVQ) |
| Encoder 파라미터 | **~25M** |
| out_dim | 1,024 |
| 입력 sr | 44kHz (내부 16→44kHz 리샘플) |
| 출력 fps | ~86 fps (hop=512 @ 44kHz) |

#### 2. EnCodec

| 항목 | 값 |
|---|---|
| 모델 | `facebook/encodec-24khz` |
| 전체 파라미터 | ~45M |
| 사용 부분 | encoder only (pre-RVQ) |
| Encoder 파라미터 | **~14M** |
| out_dim | 128 |
| 입력 sr | 24kHz (내부 16→24kHz 리샘플) |
| 출력 fps | 75 fps (hop=320 @ 24kHz) |

---

### Semantic Encoder

오디오의 언어적·음소적 표현 추출. ASR에 유리.

#### 3. Whisper Tiny

| 항목 | 값 |
|---|---|
| 모델 | `openai/whisper-tiny` |
| 전체 파라미터 | 39M |
| 사용 부분 | encoder only |
| Encoder 파라미터 | **~15M** |
| out_dim | 384 |
| 출력 fps | ~50 fps |

> **현재 미구현.** `encoders/whisper.py` 추가 필요.

---

### Hybrid Encoders

Acoustic encoder(저수준) + Semantic transformer(고수준)를 순차 통과.

#### 4. Mimi — Semantic (acoustic + encoder_transformer)

| 항목 | 값 |
|---|---|
| 모델 | `kyutai/mimi` |
| 사용 부분 | encoder + encoder_transformer |
| 전체 파라미터 | **~87M** |
| out_dim | 512 |
| 입력 sr | 24kHz (내부 16→24kHz 리샘플) |
| 출력 fps | 25 fps (hop=960 @ 24kHz) |

#### 5. Mimi — Acoustic only

| 항목 | 값 |
|---|---|
| 모델 | `kyutai/mimi` |
| 사용 부분 | encoder only (encoder_transformer 제외) |
| Encoder 파라미터 | **~20M** |
| out_dim | 512 |
| 출력 fps | 25 fps |

---

## Projector

각 encoder의 출력을 LLM hidden_size(2048)로 매핑하는 학습 가능 모듈.

```
Conv1d(in_dim → 2048, k=5, s=2) + GELU
Conv1d(2048   → 2048, k=5, s=2) + GELU   ← mimi_semantic은 stride-2 1회만
Conv1d(2048   → 2048, k=1)
LayerNorm(2048)
```

파라미터 수 상세 계산은 [docs/weight_count.md](weight_count.md) 참조.

| Encoder | in_dim | Projector 파라미터 | 총 stride | 10초 후 토큰 수 |
|---|---|---|---|---|
| DAC | 1,024 | **~35.7M** | ×4 | ~215 |
| EnCodec | 128 | **~26.5M** | ×4 | ~188 |
| Mimi semantic | 512 | **~9.4M** | ×2 | ~125 |
| Mimi acoustic | 512 | **~30.4M** | ×4 | ~63 |
| fb_dacvae | 8 | **~25.3M** | ×4 | ~215 |

---

## 최종 모델 구성 요약

| 실험 | Encoder | Encoder 크기 | Projector | LLM (LoRA) | 총 파라미터 | 학습 파라미터 (Stage 2) |
|---|---|---|---|---|---|---|
| dac | DAC acoustic | ~25M | ~35.7M | 2.0B | **~2.06B** | ~41M (projector + LoRA) |
| encodec | EnCodec acoustic | ~14M | ~26.5M | 2.0B | **~2.04B** | ~32M |
| mimi_semantic | Mimi hybrid | ~87M | ~9.4M | 2.0B | **~2.10B** | ~14M |
| mimi_acoustic | Mimi acoustic | ~20M | ~30.4M | 2.0B | **~2.05B** | ~35M |
| fb_dacvae | FB DAC-VAE | — | ~25.3M | 2.0B | **~2.03B** | ~30M |

> 학습 파라미터: projector 전체 + LoRA (~5M, r=16, q/k/v/o_proj × 24 layers). Encoder와 LLM 나머지는 frozen.

---

## Training Pipeline (2-Stage)

현재 스크립트: **`train_pipeline_override.py`** (스트리밍 + Sequence Packing + FA2 + FSDP)

```
Stage 1: Projector Alignment
  - LLM frozen, projector만 학습
  - AdamW, lr=2e-4, 2 epochs, cosine schedule (warmup_ratio=0.1)
  - Sequence Packing: cutoff_len=2048, Flash Attention 2

Stage 2: LoRA Fine-tuning
  - LLM에 LoRA(r=16) 적용, projector도 계속 학습
  - AdamW, lr=2e-5, 2 epochs, cosine schedule
  - Liger Kernel (--liger), FSDP (--fsdp) 권장
  - val_loss 기준 best checkpoint 저장
```

**실행 명령**:
```bash
LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/glibc_compat.so" \
accelerate launch train_pipeline_override.py --encoder fb_dacvae --liger --fsdp

# 특정 데이터셋만 사용
accelerate launch train_pipeline_override.py --encoder fb_dacvae --datasets ls100,mls,gs --liger --fsdp

# Stage 1만 실행
accelerate launch train_pipeline_override.py --encoder fb_dacvae --stage s1 --liger --fsdp
```

**최적화 조합** (자세한 내용: `docs/packing_fa2_liger_fsdp.md`):

| 구성 | Packing | FA2 | Liger | FSDP |
|---|---|---|---|---|
| 기본값 | ✅ 항상 ON | ✅ 항상 ON | ❌ | ❌ |
| 권장 | ✅ | ✅ | ✅ `--liger` | ✅ `--fsdp` |
| FA2 비활성화 | ✅ | ❌ `--attn-impl sdpa` | — | — |

---

## Stage 1 조기 종료

학습 중 Stage 1을 예정 epoch 전에 끊고 Stage 2로 넘어가려면:

```bash
kill -USR1 $(cat /mnt/tmp/cache/train.pid)
```

- 시그널을 받으면 **현재 epoch를 완료한 뒤** Stage 2로 진입 (mid-epoch 중단 없음)
- train.pid는 Stage 1 시작 시 rank 0 프로세스가 자동 생성, Stage 1 종료 시 삭제
- Stage 1이 이미 skip된 경우(projector 파일 존재)에는 pid 파일이 생성되지 않음

---

## Checkpoint 경로

기본 `model_cache_dir = /mnt/tmp/cache/hf`

| 단계 | 경로 | 형식 |
|---|---|---|
| Stage 1 skip 판별용 | `{cache_dir}/s1_proj_{enc}.pt` | `.pt` (state_dict) |
| Stage 1 best (기록용) | `{cache_dir}/s1_proj_{enc}_ep{N}_step{N}_best.pt` | `.pt` |
| Stage 1 step 저장 | `{cache_dir}/s1_proj_{enc}_ep{N}_step{N}.pt` | `.pt` |
| Stage 2 best | `{cache_dir}/best_{enc}_ckpt_ep{N}_step{N}/` | safetensors (accelerate) |
| Stage 2 step 저장 | `{cache_dir}/step{N}_{enc}_ckpt/` | safetensors |
| Stage 2 final | `{cache_dir}/final_{enc}_ckpt_ep{N}_step{N}/` | safetensors |

- `ep{N}`: 저장 시점의 epoch 번호, `step{N}`: 저장 시점의 global step
- Stage 1 best는 `s1_proj_{enc}.pt`(고정 이름)와 `_ep{N}_step{N}_best.pt`(기록용) 둘 다 저장됨
  - 고정 이름은 Stage 2 로딩 및 Stage 1 skip 판별에 사용
  - 기록용은 어느 epoch/step에서 best였는지 확인용

**예시 (fb_dacvae, 2번째 epoch에서 best, step=1200일 때)**
```
/mnt/tmp/cache/hf/
├── s1_proj_fb_dacvae.pt                          ← Stage 1 skip 판별 / Stage 2 로딩용
├── s1_proj_fb_dacvae_ep2_step1200_best.pt        ← Stage 1 best 기록용
├── s1_proj_fb_dacvae_ep1_step600.pt              ← step 저장 예시
├── best_fb_dacvae_ckpt_ep3_step5000/             ← Stage 2 val_loss 최선 시점
│   └── model.safetensors
├── step4000_fb_dacvae_ckpt/                      ← Stage 2 step 저장 예시
└── final_fb_dacvae_ckpt_ep16_step21000/          ← Stage 2 종료 시
    └── model.safetensors
```
