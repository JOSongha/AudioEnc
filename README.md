# AudioEnc — Encoder-Swappable ASR Training Framework

Qwen3.5-2B를 LLM backbone으로, 다양한 오디오 인코더를 교체하며 ASR 성능을 비교하는 프레임워크.

`--encoder` 인자 하나로 인코더를 바꿔 동일한 학습 파이프라인을 실행. LLM은 기본 `Qwen/Qwen3.5-2B`이며 `--llm`으로 변경 가능.

---

## 디렉토리 구조

```
AudioEnc/
├── train_pipeline_override.py  # 주 학습 스크립트 (streaming + packing + FA2 + Liger + FSDP)
├── run.sh                      # accelerate launch 래퍼
├── config.py                   # TRAIN_CONFIG + ENCODER_REGISTRY + get_config()
├── encoders/
│   ├── __init__.py             # build_encoder() 팩토리
│   ├── base.py                 # BaseAudioEncoder ABC
│   ├── encodec.py              # facebook/encodec-24khz
│   ├── dac.py                  # descript-audio-codec 44kHz
│   ├── fb_dacvae.py            # facebook/dacvae-watermarked
│   └── mimi.py                 # kyutai/mimi (acoustic / semantic)
├── model.py                    # 구형 AudioQwen (inference.py 전용)
├── dataset.py                  # 구형 데이터셋 클래스 (inference.py 전용)
├── inference.py                # 추론 스크립트
├── docs/                       # 설계 문서
└── arXiv/                      # 아카이브 (구형 학습 스크립트, 분석 코드)
```

---

## 실행

```bash
# 기본 (8 GPU, 전체 데이터셋)
bash run.sh --encoder fb_dacvae

# GPU 수 변경
bash run.sh --encoder fb_dacvae --gpus 4

# 데이터셋 선택
bash run.sh --encoder fb_dacvae --datasets ls100,ls360,mls

# Stage 1만
bash run.sh --encoder fb_dacvae --stage 1

# WandB 비활성화 (빠른 테스트)
bash run.sh --encoder fb_dacvae --wandb-mode disabled --max-steps 2 --datasets ls100

# Word-level ASR augmentation 활성화
bash run.sh --encoder fb_dacvae --word-aug
```

전체 CLI 인자 → [docs/cli_reference.md](docs/cli_reference.md)

---

## 인코더별 특성

| Key | 모델 | out_dim | fps (before proj) | 10s 토큰 수 |
|---|---|---|---|---|
| `encodec` | facebook/encodec-24khz | 128 | 75 | ~188 |
| `dac` | descript/dac-44kHz | 1024 | ~86 | ~215 |
| `fb_dacvae` | facebook/dacvae-watermarked | 8 | ~86 | ~215 |
| `mimi_acoustic` | kyutai/mimi (encoder only) | 512 | 25 | ~63 |
| `mimi_semantic` | kyutai/mimi (+ encoder_transformer) | 512 | 25 | ~125 |

---

## 모델 구조

```
Audio (16kHz)
  → [frozen encoder]           (B, T_enc, out_dim)
  → [trainable projector]      Conv1d ×2 (stride-2 각각), k=5, GELU
                                Conv1d ×1 (k=1), LayerNorm
                               (B, T_proj, llm_dim=2048)
  → placeholder token 대체
  → [Qwen3.5-2B]
  → CE loss (text tokens만)
```

projector stride = `prod(proj_strides)` (기본 ×4, mimi_semantic은 ×2).

---

## 2-Stage 학습

| | Stage 1 | Stage 2 |
|---|---|---|
| 학습 대상 | projector + proj_norm | LoRA (r=16, q/k/v/o_proj) + projector |
| LLM | frozen | LoRA trainable |
| LR | 2e-4, constant | 2e-5, cosine |
| FSDP | ✗ (DDP) | ✓ |
| 체크포인트 | `s1_outputs_{run_id}/` | `s2_outputs_{run_id}/` |

Stage 1 조기 종료: `kill -USR1 $(cat /mnt/tmp/cache/train.pid)`

---

## 최적화 스택

| 기법 | 설명 | 기본값 |
|---|---|---|
| Sequence Packing | greedy knapsack, cutoff_len 토큰으로 bin 채움, padding 최소화 | 항상 ON |
| Flash Attention 2 | varlen kernel, `(1, sum_nonpad)` flat 시퀀스 | `--attn-impl flash_attention_2` |
| Liger Kernel | fused RoPE/RMSNorm/SwiGLU/CE | `--liger` (기본 ON) |
| FSDP | Stage 2 LLM + projector 파라미터 분산 | `--fsdp` (기본 ON) |

---

## 데이터셋

| Key | 데이터셋 | 규모 |
|---|---|---|
| `ls100` | LibriSpeech train-clean-100 | 100h |
| `ls360` | LibriSpeech train-clean-360 | 360h |
| `ls500` | LibriSpeech train-other-500 | 500h |
| `mls` | Multilingual LibriSpeech English | ~10,000h |
| `gs` | GigaSpeech XL | ~10,000h |
| `vp` | VoxPopuli English | ~500h |

평가: LibriSpeech dev-clean WER + val_loss (every `eval_steps` steps).

---

## 새 인코더 추가

1. `encoders/myenc.py`에 `BaseAudioEncoder` 서브클래스 작성
   - `forward(audio, lengths) → (feats, mask)` 구현
   - fp32 출력, 내부 리샘플링, 항상 frozen/eval
2. `config.py`의 `ENCODER_REGISTRY`에 항목 추가 (`out_dim`, `tgt_sr`, `hop`, `proj_strides`)
3. `encoders/__init__.py`의 `ENCODER_CLASSES`에 등록

---

## 주요 경로

| 용도 | 경로 |
|---|---|
| HF 데이터/모델 캐시 | `/mnt/tmp/cache/hf` |
| Stage 1 출력 | `/mnt/tmp/cache/hf/{encoder}/s1_outputs_{run_id}/` |
| Stage 2 출력 | `/mnt/tmp/cache/hf/{encoder}/s2_outputs_{run_id}/` |
| best projector | `s1_outputs_{run_id}/best_s1_proj.pt` |
| Word alignment Arrow | `/mnt/tmp/cache/word_alignments_merged/{dataset}/` |

`run_id` = `MMDD_HHMM` (실행 시작 시각)

---

## 설계 문서

| 문서 | 내용 |
|---|---|
| [docs/pipeline_overview.md](docs/pipeline_overview.md) | 전체 파이프라인 동작 원리 |
| [docs/cli_reference.md](docs/cli_reference.md) | 전체 CLI 인자 레퍼런스 |
| [docs/word_alignment.md](docs/word_alignment.md) | Word-level augmentation 설계 |
| [arXiv/docs/dynamic_batching.md](arXiv/docs/dynamic_batching.md) | Dynamic batching (arXiv 참조용) |
| [docs/packing_fa2_liger_fsdp.md](docs/packing_fa2_liger_fsdp.md) | 최적화 기법 상세 |
