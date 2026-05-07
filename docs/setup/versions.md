# 버전 라벨 카탈로그 (v2 ~ v6)

리포 안에서 "v2", "v3", "v4", "v5", "v6" 라벨이 붙은 파일을 카테고리별로 정리. 동일한 "vN" 토큰이 문맥에 따라 다른 의미로 쓰이는 경우가 있어 § 0 에 disambiguation 먼저 정리한다.

작성 기준일: 2026-05-07. 검증은 리포 루트(`/mnt/ddn/users/jos/audiollm-trainer`)에서 `find` + `grep` 으로 수행.

## 0. 의미 disambiguation

| 라벨 | 위치 | 의미 |
|---|---|---|
| **Stage-1 vN** (N=2..6) | `configs/ASR/stage1_*_v[2-6].yaml`, `/mnt/tmp/datasets/manifests/v[3-6]/` | Stage-1 사전학습 manifest 세대. v3 → v4 → v5 → v6 진화 체인. v2 는 manifest dir 없는 초기 버전 (libri+mls+vox 만) |
| **Stage-2 v1 / v2** | `configs/qwen3_5ae-asr/stage2_v2.yaml`, `docs/stage2/analysis/v1_vs_v2.pdf` | Stage-2 LoRA 미세조정 chain 라벨. Stage-1 의 v2/v3 와 **별개**. v1 은 폐기, v2 만 사용 (project memory) |
| **speechx-v6 / voice-model-engine-v6** | `configs/speechx-v6/`, `configs/voice-model-engine-v6/` | v6 manifest 위에서 도는 SpeechX / voice-engine 학습 config 모음. Stage-1/2 라벨과 직교 |
| **v5_2** (=v5.2) | `examples/train_lora/speechx_lora_v5_2_*`, `experiments/voice_engine/run_voice_engine_v5_2_*` | voice-engine LoRA 의 v5 마이너 갱신. Stage-1 manifest v5 와 다른 축 |

> wandb run hash (`ev2lml01`, `v08wsov3`, `ydv5ln9j`, `v552uxhm` 등) 안의 "v2/v3/v5" 토큰은 **random run ID** 일부라 버전 라벨이 아님.

> `configs/speechx-v7/`, `configs/voice-model-engine-v7/`, `experiments/.../v7*` 도 존재하지만 본 문서 스코프 (v2-v6) 밖이라 생략.

## 1. Stage-1 manifest 진화 체인

```
v2 (libri+mls+vox만, ASR-only)
  ↓ +sound captioning 9개 + emotion 7개 (3-modality)
v3
  ↓ per-modality split + interleave 버그 수정
v4 (DAC-VAE / Whisper-small / Whisper-tiny 3개 인코더)
  ↓ 3× projector 확대 + AudioSet/FSD50K 캡션 포맷 + emotion leak fix
v5
  ↓ +IEMOCAP Sessions 1-4 (5,882 rows) emotion 풀에 추가
v6 (현재 정식 manifest)
```

세부 데이터 스펙은 [datasets.md](datasets.md) (v6 카탈로그) 와 [docs/stage1/v4/dac_vae_dataflow.md](../stage1/v4/dac_vae_dataflow.md) (v3 vs v4 차이) 참고.

## 2. v2 (Stage-1, DAC-VAE only)

ASR-only (LibriTTS-R + MLS + VoxPopuli). manifest 디렉터리 없이 `external/datasets/libri_mls_vox` 직접 로드.

| 카테고리 | 경로 |
|---|---|
| Config | [configs/ASR/stage1_dac_vae_v2.yaml](../../configs/ASR/stage1_dac_vae_v2.yaml) |
| Script | [scripts/ASR/archive/legacy_stage1/run_stage1_dac_vae_v2.sh](../../scripts/ASR/archive/legacy_stage1/run_stage1_dac_vae_v2.sh) (archive) |
| Ckpt 출력 | `/mnt/tmp/Qwen3.5_dac_vae_v2_Stage1_jos/` (외부) |

## 3. v3 (Stage-1, 3-modality 통합)

ASR + sound captioning + emotion 첫 통합. Stage-2 v2 의 베이스가 됨.

| 카테고리 | 경로 |
|---|---|
| Config | [configs/ASR/stage1_dac_vae_v3.yaml](../../configs/ASR/stage1_dac_vae_v3.yaml) |
| Script | [scripts/ASR/archive/legacy_stage1/run_stage1_dac_vae_v3.sh](../../scripts/ASR/archive/legacy_stage1/run_stage1_dac_vae_v3.sh) (archive) |
| Eval scripts | [scripts/ASR/archive/v3_eval/](../../scripts/ASR/archive/v3_eval/) — `eval_v3_{asr_external_gpu7,auto_dispatcher,baseline_encodec,dispatcher,overnight_fill,resume_failed,sweep}.sh` (7 파일, archive) |
| Manifest builder | [scripts/emo/build_emotion_v3_manifest.py](../../scripts/emo/build_emotion_v3_manifest.py) |
| 보조 util | [scripts/stat_utils/convert_asr_v3.py](../../scripts/stat_utils/convert_asr_v3.py) |
| Doc | [docs/v3_progress_log.md](../v3_progress_log.md), [docs/stage1/dac_vae_v3_data.md](../stage1/dac_vae_v3_data.md) |
| Manifest dir | `/mnt/tmp/datasets/manifests/v3/` (외부), `/mnt/tmp/datasets/manifests/v3_emotion_split/` (16-shard emotion 재분할) |
| Ckpt 출력 | `/mnt/tmp/Qwen3.5_dac_vae_v3_Stage1_jos/` (외부) |

## 4. v4 (Stage-1, per-modality + 3-encoder)

v3 와 데이터 동일, manifest dir 구조를 per-modality 로 분리. DAC-VAE / Whisper-small / Whisper-tiny 3개 인코더 변형 (manifest 동일, 인코더만 다름).

| 카테고리 | 경로 |
|---|---|
| Configs | [configs/ASR/stage1_dac_vae_v4.yaml](../../configs/ASR/stage1_dac_vae_v4.yaml), [stage1_whisper_small_v4.yaml](../../configs/ASR/stage1_whisper_small_v4.yaml), [stage1_whisper_tiny_v4.yaml](../../configs/ASR/stage1_whisper_tiny_v4.yaml), [stage1_whisper_tiny_v4_ft.yaml](../../configs/ASR/stage1_whisper_tiny_v4_ft.yaml) |
| Scripts | [scripts/ASR/run_stage1_dac_vae_v4.sh](../../scripts/ASR/run_stage1_dac_vae_v4.sh), [run_stage1_whisper_small_v4.sh](../../scripts/ASR/run_stage1_whisper_small_v4.sh), [run_stage1_whisper_tiny_v4.sh](../../scripts/ASR/run_stage1_whisper_tiny_v4.sh) |
| Eval modules | [evaluation/stage1_v4/](../../evaluation/stage1_v4/) — `eval_audioset_caption.py`, `eval_fsd50k_caption.py`, `eval_esc50_caption.py`, [README.md](../../evaluation/stage1_v4/README.md) |
| Docs | [docs/stage1/v4/README.md](../stage1/v4/README.md), [dac_vae_dataflow.md](../stage1/v4/dac_vae_dataflow.md), [whisper_small_step12850_hang.md](../stage1/v4/whisper_small_step12850_hang.md) |
| Manifest dir | `/mnt/tmp/datasets/manifests/v4/` (외부) |
| Ckpt 출력 | `/mnt/tmp/Qwen3.5_dac_vae_v4_Stage1_jos/` (외부, 100k step 완주) |

## 5. v5 (Stage-1, projector 확대 + caption 포맷 + emotion leak fix)

v4 위에 (1) projector 3× 확대, (2) AudioSet/FSD50K 캡션 포맷 (label-list → ontology description), (3) emotion eval split 보호 (DailyTalk/EmoV-DB/RAVDESS held-out 처리).

| 카테고리 | 경로 |
|---|---|
| Configs | [configs/ASR/stage1_dac_vae_v5.yaml](../../configs/ASR/stage1_dac_vae_v5.yaml), [stage1_whisper_small_v5.yaml](../../configs/ASR/stage1_whisper_small_v5.yaml), [stage1_whisper_tiny_v5.yaml](../../configs/ASR/stage1_whisper_tiny_v5.yaml) |
| Scripts | [scripts/ASR/run_stage1_dac_vae_v5.sh](../../scripts/ASR/run_stage1_dac_vae_v5.sh), [run_stage1_whisper_small_v5.sh](../../scripts/ASR/run_stage1_whisper_small_v5.sh), [run_stage1_whisper_tiny_v5.sh](../../scripts/ASR/run_stage1_whisper_tiny_v5.sh) |
| Setup | [scripts/setup_v5_projL_models.sh](../../scripts/setup_v5_projL_models.sh) — projL 모델 디렉터리 + config.json 초기화 |
| Doc | [docs/stage1/v5.md](../stage1/v5.md) |
| Manifest dir | `/mnt/tmp/datasets/manifests/v5/` (외부) |
| projL 모델 dir | `external/models/Qwen3.5AE-4B-projL/`, `Qwen3.5AE-4B-whisper-small-projL/`, `Qwen3.5AE-4B-whisper-tiny-projL/` (config.json 만, weight 는 학습 후 생성) |

### v5 관련 — voice engine 별도 축

v5 manifest 와 별개로 voice-engine LoRA 갱신 (v5 / v5_2 마이너):

| 경로 | 설명 |
|---|---|
| [examples/train_lora/speechx_lora_v5_voice_engine.yaml](../../examples/train_lora/speechx_lora_v5_voice_engine.yaml) | v5 voice engine LoRA |
| [examples/train_lora/speechx_lora_v5_2_voice_engine.yaml](../../examples/train_lora/speechx_lora_v5_2_voice_engine.yaml) | v5.2 |
| [examples/train_lora/speechx_lora_v5_2_voice_engine_v2.yaml](../../examples/train_lora/speechx_lora_v5_2_voice_engine_v2.yaml) | v5.2 + variant v2 |
| [experiments/voice_engine/run_voice_engine_v5_vm_v5_full.sh](../../experiments/voice_engine/run_voice_engine_v5_vm_v5_full.sh) | v5 voice engine + v5 voice model, full FT |
| [experiments/voice_engine/run_voice_engine_v5_vm_v5_lora.sh](../../experiments/voice_engine/run_voice_engine_v5_vm_v5_lora.sh) | LoRA 변형 |
| [experiments/voice_engine/run_voice_engine_v5_vm_v5_2_full.sh](../../experiments/voice_engine/run_voice_engine_v5_vm_v5_2_full.sh) | v5 engine + v5_2 model |
| [experiments/voice_engine/run_voice_engine_v5_vm_v5_2_lora.sh](../../experiments/voice_engine/run_voice_engine_v5_vm_v5_2_lora.sh) | (LoRA) |
| [experiments/voice_engine/run_voice_engine_v5_2_vm_v5_2_full.sh](../../experiments/voice_engine/run_voice_engine_v5_2_vm_v5_2_full.sh) | v5_2 engine + v5_2 model |
| [experiments/voice_engine/run_voice_engine_v5_2_vm_v5_2_lora.sh](../../experiments/voice_engine/run_voice_engine_v5_2_vm_v5_2_lora.sh) | (LoRA) |
| [experiments/voice_model/run_voice_model_v5_sft.sh](../../experiments/voice_model/run_voice_model_v5_sft.sh) | v5 voice model SFT |

## 6. v6 (Stage-1 + IEMOCAP 추가, 현재 정식)

v5 위에 IEMOCAP Sessions 1-4 (5,882 rows) 추가. emotion 39.9k → 47.05k.

### Stage-1 측

| 카테고리 | 경로 |
|---|---|
| Doc | [docs/setup/datasets.md](datasets.md) — 정식 카탈로그 |
| Manifest dir | `/mnt/tmp/datasets/manifests/v6/` (외부), `v6_emotion_split/`, `v6_raw/` (IEMOCAP 처리 전 staging) |

> v6 manifest 용 별도 `configs/ASR/stage1_*_v6.yaml` 은 **현재 없음**. v5 config 를 그대로 두고 manifest path 만 v6 로 바꿔 학습한 듯 (또는 v6 학습은 speechx-v6 / voice-model-engine-v6 config 로 통합).

### speechx-v6 (12 configs)

[configs/speechx-v6/](../../configs/speechx-v6/):
- Stage-1 변형: `speechx_v6_all_stage1_decay.yaml`, `..._greenland.yaml`, `..._greenland_stepaudio.yaml`, `..._jessica.yaml`
- Stage-2 변형: `speechx_v6_all_stage2.yaml`, `..._stage2_1.yaml`, `..._stage2_abl.yaml`
- Stage-2 spd 변형: `speechx-v6-s2-nsml-decay.yaml`, `speechx-v6-s2-spd1.yaml`, `speechx-v6-s2-spd2.yaml`, `speechx-v6-s2-spd2-decay.yaml`, `speechx-v6-s2-spd2-resume.yaml`

### voice-model-engine-v6 (10 configs)

[configs/voice-model-engine-v6/](../../configs/voice-model-engine-v6/):
- Voice model SFT: `speechx_full_v6_voice_model_sft.yaml`, `..._abl1.yaml`, `..._abl2.yaml`
- Voice engine SFT: `speechx_full_voice_engine_v6_sft.yaml`, `..._abl1_sft.yaml`, `..._abl2_sft.yaml`, `..._abl3-a_sft.yaml`, `..._abl3-f_sft.yaml`, `..._abl3-g_sft.yaml`, `..._abl3-j_sft.yaml`

### Manifest builders / scripts

| 경로 | 설명 |
|---|---|
| [scripts/manifest_builders/](../../scripts/manifest_builders/) | v5/v6 공통 빌더 (이전 `scripts/v3_manifest/` 에서 rename, datasets.md changelog) |
| [experiments/voice_model/run_voice_model_v6_sft.sh](../../experiments/voice_model/run_voice_model_v6_sft.sh) | v6 voice model SFT |
| [experiments/voice_engine/run_voice_engine_v6_full.sh](../../experiments/voice_engine/run_voice_engine_v6_full.sh) | v6 voice engine full |

## 7. Stage-2 v1 / v2 (DAC-VAE LoRA, 별도 축)

§0 에서 언급. Stage-1 의 v2 와 다른 의미. v1 폐기, v2 사용.

| 카테고리 | 경로 |
|---|---|
| Config | [configs/qwen3_5ae-asr/stage2_v2.yaml](../../configs/qwen3_5ae-asr/stage2_v2.yaml) |
| Script | [configs/qwen3_5ae-asr/run_v2_8gpu.sh](../../configs/qwen3_5ae-asr/run_v2_8gpu.sh) |
| 비교 plot | [evaluation/stage2/plot_v1_v2_compare.py](../../evaluation/stage2/plot_v1_v2_compare.py) |
| Analysis | [docs/stage2/analysis/v1_vs_v2.pdf](../stage2/analysis/v1_vs_v2.pdf), [v2_per_ckpt_full.csv](../stage2/analysis/v2_per_ckpt_full.csv) |
| Best ckpt | `/mnt/ddn/users/jos/s2_best_ckpts/dacvae_v2/` (외부) |
| Inference 결과 | `/mnt/ddn/users/jos/v2_best_inference/` (외부) |

## 8. 기타 산발 v 라벨

| 경로 | 설명 |
|---|---|
| [scripts/emo/hash_match_ravdess_v2.py](../../scripts/emo/hash_match_ravdess_v2.py) | RAVDESS hash matching 유틸 v2 (v1 → v2 마이너 갱신, 버전 chain 무관) |

## 9. 외부 경로 요약 (`/mnt/tmp/`)

| 경로 | 내용 |
|---|---|
| `/mnt/tmp/datasets/manifests/v3/` | v3 통합 manifest |
| `/mnt/tmp/datasets/manifests/v3_emotion_split/` | v3 emotion 16-shard 재분할 |
| `/mnt/tmp/datasets/manifests/v4/` | v4 per-modality |
| `/mnt/tmp/datasets/manifests/v5/` | v5 per-modality |
| `/mnt/tmp/datasets/manifests/v6/` | v6 per-modality (현재 정식) |
| `/mnt/tmp/datasets/manifests/v6_emotion_split/` | v6 emotion 재분할 |
| `/mnt/tmp/datasets/manifests/v6_raw/` | v6 IEMOCAP raw staging |
| `/mnt/tmp/Qwen3.5_dac_vae_v2_Stage1_jos/` | Stage-1 v2 ckpt |
| `/mnt/tmp/Qwen3.5_dac_vae_v3_Stage1_jos/` | Stage-1 v3 ckpt |
| `/mnt/tmp/Qwen3.5_dac_vae_v4_Stage1_jos/` | Stage-1 v4 ckpt (100k 완주) |
| `/mnt/ddn/users/jos/s2_best_ckpts/dacvae_v2/` | Stage-2 v2 best |
| `/mnt/ddn/users/jos/v2_best_inference/` | Stage-2 v2 평가 결과 9 task |

## 10. 검색 커맨드 (재계산)

```bash
# 디렉터리
find /mnt/ddn/users/jos/audiollm-trainer -maxdepth 4 -type d \
  \( -name "*v2*" -o -name "*v3*" -o -name "*v4*" -o -name "*v5*" -o -name "*v6*" \) \
  -not -path "*/.git/*" -not -path "*/__pycache__/*"

# 파일
find /mnt/ddn/users/jos/audiollm-trainer -maxdepth 6 -type f \
  \( -name "*v2*" -o -name "*v3*" -o -name "*v4*" -o -name "*v5*" -o -name "*v6*" \) \
  -not -path "*/.git/*" -not -path "*/__pycache__/*" -not -path "*/wandb/*"

# 외부 manifest
ls /mnt/tmp/datasets/manifests/
```

## 변경 이력

- 2026-05-07: 초기 작성. 리포 트리 + `/mnt/tmp/datasets/manifests/` 직접 검증 기준.
