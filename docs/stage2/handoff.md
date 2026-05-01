# Stage 2 LoRA 학습 인계 문서

audiollm-trainer 기반으로 Qwen3.5AE Stage 2 (LoRA SFT) 학습을 처음부터 돌리는 사람이 필요한 모든 것.

작성일 2026-04-29. 본문 경로/사이즈/시간 추정은 jos 클러스터 기준 (다른 환경에서는 적당히 치환).

---

## 1. 한눈에 보기

- **목표**: Stage 1 (projector-only) 위에 LoRA + projector 풀-trainable + audio_emotion / env_sound / text-SFT 혼합 manifest로 LoRA fine-tuning
- **베이스 모델**: Stage 1 init (Whisper-small 또는 Encodec-24k 등)
- **하드웨어**: 8 × A100-80GB (16-way dataloader 기준)
- **한 번 학습 시간**: ~2-3 일 (31k step, batch 32, whisper-small 인코더)
- **체크포인트 사이즈**: ckpt당 ~41 MB (`adapter_model.safetensors`, PEFT adapter + projector via `additional_target`) → 31 ckpt ≈ 1.3 GB. Trainer state / DeepSpeed optimizer state 포함 시 ckpt 폴더 전체 ~1.8 GB / ckpt — disk 계획 시 주의.

---

## 2. 하드웨어 / OS 요구

| 항목 | 권장 |
|---|---|
| GPU | 8 × A100-80GB (또는 H100). 메모리 80 GB 미만이면 batch ↓ 또는 gradient_checkpointing 활성화 |
| CPU 코어 | ≥ 32 (16 dataloader workers/rank × 8 ranks 활용 가능 시) |
| 시스템 RAM | ≥ 256 GB (tmpfs 캐시 + 16-way dataloader prefetch) |
| 로컬 scratch | 100 GB+ (xfs 또는 ext4, **NFS/Lustre 비권장 — IO 경쟁** + 쓰기 quota 위험) |
| OS | Linux x86_64, glibc ≥ 2.28 (Whisper-small / DAC encoder는 glibc_compat shim 필요할 수 있음) |
| 시스템 gcc | **9.x 권장**. nvcc는 gcc > 11 거부 → conda gcc 13이 PATH에 있으면 deepspeed `fused_adam` JIT 빌드 실패 |

---

## 3. 소프트웨어 환경

### 3.1 conda env

```bash
# 미니콘다/미니포지 설치 후
conda create -n audio_lmf python=3.10 -y
conda activate audio_lmf

# torch 2.6 + cu124
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 핵심 의존성
pip install transformers==4.57.1 peft==0.17.1 deepspeed==0.16.9 \
    accelerate datasets jiwer pyarrow safetensors \
    sentencepiece tiktoken protobuf einops omegaconf hf-transfer

# llamafactory editable
cd /path/to/audiollm-trainer
pip install -e .
```

### 3.2 gcc / nvcc 정합성

```bash
# 컴파일러 명시적으로 시스템 gcc 9 사용
export CC=/usr/bin/gcc          # gcc 9.x
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++

# torchrun 등 env binary가 시스템보다 먼저 잡히도록
export PATH=$CONDA_PREFIX/bin:$PATH
```

### 3.3 캐시 redirect (홈 디렉토리에 30 GB 이상 못 쓰는 클러스터 대응)

```bash
export TRITON_CACHE_DIR=/mnt/tmp/cache/triton
export CUDA_CACHE_PATH=/mnt/tmp/cache/nv_compute
export HF_HOME=/mnt/tmp/cache/huggingface
export WANDB_DIR=/mnt/tmp/cache/wandb
export TMPDIR=/mnt/tmp/cache/tmp
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_HOME" "$WANDB_DIR" "$TMPDIR"
```

---

## 4. 코드 (audiollm-trainer)

```bash
git clone <internal repo> audiollm-trainer
cd audiollm-trainer
pip install -e .
```

핵심 파일:
- `configs/qwen3_5ae-asr/stage2_v2.yaml` — 권장 v2 mix (emoFull / asr033 / env05 / txt03)
- `configs/qwen3_5ae-asr/stage2_whisper_small.yaml` — whisper-small 인코더 변형
- `configs/qwen3_5ae-asr/stage2.yaml` — v1 (asr14 / emo34 / env35 / txt17)
- `scripts/qwen3_5ae-asr/run_v2_8gpu.sh` — 8 GPU torchrun 런처
- `examples/deepspeed/ds_z2_config.json` — DeepSpeed ZeRO-2

---

## 5. Stage 1 init 체크포인트

S2 LoRA의 베이스 (full-weight). 인코더 별 한 개 골라 받음.

| 인코더 | S1 ckpt | 사이즈 (single ckpt) |
|---|---|---:|
| Whisper-small | `Qwen3.5_whisper_small_Stage1/checkpoint-13000` | ~15 GB |
| Whisper-tiny | `Qwen3.5_whisper_tiny_Stage1/checkpoint-13000` | ~13 GB |
| WavTok-40 | `Qwen3.5_wavtok_40_unify_Stage1/checkpoint-N` | ~14 GB |
| Encodec-24k | `Qwen3.5_encodec_24k_Stage1/checkpoint-N` | ~13 GB |
| DAC (legacy v1) | `Qwen3.5AE-4B-ASR-Stage1/checkpoint-50000` | ~14 GB |

**전달 방법**: 해당 ckpt 디렉토리만 tar.xz로 묶어 전달 (`tar -caf init.tar.xz checkpoint-13000`). 50 MB 이하 압축률 기대 어렵고 그대로 전송 권장.

S1을 직접 학습하려면 `configs/ASR/stage1_<encoder>.yaml` + `scripts/ASR/run_stage1_<encoder>.sh` 패턴 (예: `run_stage1_encodec.sh`).

---

## 6. 데이터셋

전체 ~210 GB + ASR 슈퍼셋 60-100 GB.

### 6.1 audio_emotion (training 풀)

| 코퍼스 | 사이즈 | 라이선스 | 다운로드 출처 |
|---|---:|---|---|
| MELD train+dev | 32 GB | research | https://affective-meld.github.io/ |
| DailyTalk 95% dialogs | 12 GB | CC-BY-NC-SA 4.0 | GDrive (rate-limited) |
| EmoV-DB Bea+Josh+Sam | 7.3 GB | CC BY 4.0 | OpenSLR 115 |
| RAVDESS Actors 01-20 | 765 MB | CC-BY-NC | Zenodo |
| CREMA-D | 85 MB (partial) | ODbL | HF mirror 404 → research 직접 |
| MUStARD++ | 104 MB (partial) | research | GDrive |
| 합계 (실사용) | ~52 GB | | |

**EULA 필요 (옵션, 본 sweep 미사용)**: IEMOCAP (USC), MSP-Podcast (UTD), ESD

레퍼런스 다운로드 스크립트: `scripts/emo/download_source_corpora{,_phase2,_phase3}.sh`

### 6.2 audio_env_sound (training 풀)

| 코퍼스 | 사이즈 | 다운로드 |
|---|---:|---|
| FSD50K dev (40 966 wavs, 81.5h) | 56 GB | https://zenodo.org/record/4060432 |
| Clotho development (3 839 captions) | 18 GB | https://zenodo.org/record/4783391 |
| ESC-50 | 1.4 GB | https://github.com/karolpiczak/ESC-50 |
| AudioSet bal_train (~18 683, v2만) | ~71 GB | HF `agkphysics/AudioSet` (또는 YT 직접) |

### 6.3 text-SFT (Tier 4 guardrail)

`/mnt/tmp/datasets/text_benchmarks/<bench>/<split>.parquet`. HF에서 받기:

```bash
python -c "
from datasets import load_dataset
for name in ['hellaswag','winogrande','boolq','copa']:
    ds = load_dataset(name, cache_dir='/mnt/tmp/cache/hf_text')
    ds.save_to_disk(f'/mnt/tmp/datasets/text_benchmarks/{name}')
load_dataset('allenai/ai2_arc','ARC-Easy',cache_dir='...')   # arc_easy
load_dataset('allenai/ai2_arc','ARC-Challenge',cache_dir='...') # arc_challenge
"
```
합계 ~56 MB.

### 6.4 ASR 슈퍼셋 (S2 mix 14% 또는 v2 0.33 fraction)

S2 학습용 manifest는 audio path를 LibriSpeech / MLS-en / GigaSpeech subset에서 끌어옴. 전체:

| | 사이즈 (16 kHz mono) | 비고 |
|---|---:|---|
| LibriSpeech train-clean-100/360, train-other-500 | ~60 GB | HF `openslr/librispeech_asr` (split=train.*) |
| MLS-en subset (~50k h 중 일부) | 60-100 GB | OpenSLR 94 |
| GigaSpeech M (~10k h) | ~30 GB | speechcolab/gigaspeech (gated, 신청 필요) |

S1 학습용으로 만든 통합 manifest는 `external/datasets/libri_mls_vox/shard_*.jsonl` (3.6 GB) — 오디오는 별도 위치를 참조.

### 6.5 LISTEN-test (eval 전용)

```
/mnt/tmp/datasets/listen_analysis/data/test-00000-of-00001.parquet  (3.4 GB)
```
`VibeCheck1/LISTEN_full` HF dataset의 test split만 추출.

---

## 7. Stage 2 manifest 빌드

기존 manifest 그대로 재사용해도 OK. 새로 만들려면:

```bash
# 1. 코퍼스별 held-out 분리 + MCQA 포맷 변환
python scripts/emo/build_training_manifest.py --out /mnt/tmp/datasets/listen_analysis/train_manifest

# 2. v1 (fixed concat 70:20:10 또는 14:34:35:17)
python scripts/emo/build_combined_manifest.py \
    --asr-rows 17150 --emo-rows 39919 --env-rows 41000 --text-rows 20519 \
    --out /mnt/tmp/datasets/listen_analysis/train_manifest/stage2_combined_shards

# 3. v2 (per-epoch random subsample)
python scripts/emo/build_epoch_random_manifest.py \
    --asr-pool 40000 --asr-frac 0.33 \
    --emo-pool 39919 \
    --env-pool 65000 --env-frac 0.5 \
    --text-pool 20519 --text-frac 0.3 \
    --out /mnt/tmp/datasets/listen_analysis/train_manifest/stage2_v2_shards
```

각 row: `{audio_path, modality, prompt, target, ...}` 형식 JSONL. omni_dataset.py가 modality 별로 ChatML 분기.

---

## 8. 실행 명령어

### 8.1 Stage 2 v2 (권장)

```bash
cd /path/to/audiollm-trainer

# Single-node 8 GPU
FORCE_TORCHRUN=1 NPROC_PER_NODE=8 \
    llamafactory-cli train configs/qwen3_5ae-asr/stage2_v2.yaml
```

또는 wrapper 사용:

```bash
bash scripts/qwen3_5ae-asr/run_v2_8gpu.sh
```

### 8.2 Whisper-small 변형

```bash
bash scripts/qwen3_5ae-asr/run_whisper_small_8gpu.sh
```

### 8.3 멀티노드 (NSML 같은 분산 잡 시스템)

```bash
FORCE_TORCHRUN=1 \
    NNODES=$NSML_WORLD_SIZE NODE_RANK=$NSML_RANK \
    MASTER_ADDR=$NSML_HOST_RANK0 MASTER_PORT=21267 \
    llamafactory-cli train configs/qwen3_5ae-asr/stage2_v2.yaml
```

### 8.4 yaml에서 바꿔야 하는 경로 4개

```yaml
model_name_or_path: <S1 init ckpt 경로>
omni_manifest:      <stage2 manifest 디렉토리 (shard_*.jsonl 포함)>
output_dir:         <체크포인트 저장 위치, scratch 권장>
logging_dir:        <output_dir와 동일 권장>
```

### 8.5 Smoke test (학습 전 30초 확인)

```bash
# 모델 로드 + dummy forward
python -c "
import torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(
    '<S1 init path>', trust_remote_code=True, torch_dtype=torch.bfloat16
).cuda().eval()
print('total', sum(p.numel() for p in m.parameters())/1e9, 'B')
print('trainable', sum(p.numel() for n,p in m.named_parameters()
      if 'projector' in n)/1e6, 'M')
wav = torch.randn(1, 480000, dtype=torch.bfloat16, device='cuda')   # 30s @ 16kHz
print(m.model.audio_encoder(wav).shape)
"
```

---

## 9. 평가 (sweep)

학습 끝난 후 (또는 sliding window 중간 평가):

```bash
# 단일 ckpt × 단일 bench
python -m evaluation.stage2.eval_listen_mcqa \
    --ckpt-root <CKPTS_ROOT> --out-root <CKPTS_ROOT>/eval_listen \
    --base-model <S1 init> --ckpts 11000 --batch-size 4

# 전체 sweep (35 ckpts × 12 benches, 8 GPU 병렬, 16 worker)
# → 본 sweep에서 사용한 dispatcher 그대로 복사
bash <run_dir>/run_eval_sweep_queue.sh

# 결과 집계
python -m evaluation.stage2.aggregate_results --root <run_dir> --out <run_dir>/analysis
python -m evaluation.stage2.plot_trajectories --csv <run_dir>/analysis/results.csv \
    --out <run_dir>/analysis/trajectories.pdf
```

---

## 10. 다운로드 시간 예상

1 Gbps (≈ 100 MB/s) 가정:

| 항목 | 사이즈 | 시간 |
|---|---:|---:|
| Public HF (LibriSpeech 60 G + FSD50K 56 G + Clotho 18 G + ESC-50 1.4 G) | ~135 G | ~22 min |
| AudioSet bal+eval | ~80 G | ~13 min |
| Emotion raw (research, GDrive 일부) | ~52 G | ~9 min (rate limit 시 더) |
| LISTEN parquet | 3.4 G | ~30 s |
| Text benchmarks | 56 M | <1 s |
| S1 init ckpt | ~15 G | ~2.5 min |
| **합계** | **~285 G** | **~50 min (1 Gbps)** / **5–8 h (100 Mbps)** |

병목 가능 지점:
- **GDrive rate-limit**: DailyTalk / MUStARD 등은 일별 다운로드 제한이 있어 분할 다운로드 필요할 수 있음
- **AudioSet**: HF `agkphysics/AudioSet`는 mp3 인코딩이라 빠르지만 wav 변환 필요. YT 직접 다운로드는 yt-dlp 의존 + 수일 소요
- **HF token 필요**: GigaSpeech, MSP-Podcast 등 gated 데이터셋

---

## 11. 흔한 함정 (디버깅 단축)

| 증상 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: llamafactory` | 시스템 torchrun (`/usr/local/bin/torchrun`)이 다른 python을 잡음 | `export PATH=$CONDA_PREFIX/bin:$PATH` |
| `ninja: build stopped` (fused_adam) + nvcc gcc 11+ 거부 | conda gcc 13 PATH 우선 | `export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++` |
| `ImportError: fused_adam.so cannot open` (반복 재시도 실패) | 캐시된 빌드가 손상 | `rm -rf ~/.cache/torch_extensions/py310_cu124/fused_adam` |
| `Disk quota exceeded` (Lustre 등) | 클러스터 FS quota 한계 | output_dir + cache를 모두 로컬 scratch로 |
| dataloader CPU 포화 (load avg 수백) | 16 procs × torch 기본 멀티스레드 | `export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2` |
| 학습 시작 직후 GPU 0% 오래 지속 | 첫 epoch shard 로딩 + 오디오 디코딩 (Lustre / NFS 시 심함) | 데이터셋을 로컬 xfs로 옮기거나 `/dev/shm` 캐시 |
| `WANDB authentication failed` | offline 환경 | `export WANDB_MODE=disabled` |
| Whisper feature extractor 다운로드 실패 | 인터넷 차단 | `HF_HUB_OFFLINE=1` + 사전에 `WhisperFeatureExtractor.from_pretrained('openai/whisper-small.en')` 한 번 받아두기 |

---

## 12. 출력물 (학습 + 평가 후)

```
<output_dir>/
    checkpoint-1000/, checkpoint-2000/, ... checkpoint-Nk/
        adapter_config.json
        adapter_model.safetensors      (~50 MB, projector + LoRA)
        tokenizer*, special_tokens_map.json
        trainer_state.json, scheduler.pt, rng_state_*.pth
        global_step1000/                (deepspeed state, resume용)
    trainer_log.jsonl
    runs/Qwen3.5_*/                    (tensorboard event files)
    eval_*/checkpoint-N/summary.json   (sweep 후)
    analysis/
        results.csv, results_wide.csv
        results.md, best_per_task.md
        trajectories.pdf
```

---

## 13. 본 sweep 참고 (whisper-small v2)

- 학습 시간: 31k step → ~3 일 (8 × A100-80GB)
- 평가 시간: 35 ckpt × 12 bench = 14h 41m wall (16-way work-stealing)
- 결과 (best ckpt 일부):
  - LibriSpeech test-clean WER: **2.51 %** @ ckpt-8k
  - LibriSpeech test-other WER: **6.09 %** @ ckpt-6k
  - ESC-50: **100.0 %** @ ckpt-33k
  - FSD50K mAP-macro (seq): **0.502** @ ckpt-2k
  - Clotho BLEU-4: **0.131** @ ckpt-18k
  - EmoV-DB acc: **93.85 %** @ ckpt-3k
  - RAVDESS acc: **66.7 %** @ ckpt-7k
  - Text retention 6-bench mean: **0.912** @ ckpt-34k
- 단일 Pareto-optimum ckpt 없음 — task-stratified peaks (5–8k = ASR/LISTEN, 18k = captioning, 33–34k = ESC-50/text). 배포 ckpt는 **ckpt-13–17k** 구간이 가장 균형 좋음.

자세한 분석 표는 `<run_dir>/analysis/best_per_task.md` 참고.
