# Qwen3.5-4B + DACVAE ASR 학습

> 본 문서에서 `{{ }}` 안에 있는 내용은 수정 후 실행

---

## 1. 모델 파일 경로

Qwen3.5-4B + DACVAE Modeling (Qwen3.5 파일 변경함, Text 성능 유지 체크 완료)

```
/mnt/fr20tb/audiollm/sanghyuk/Qwen3AE-4B
/mnt/fr20tb/audiollm/sanghyuk/Qwen3.5AE-4B
```

### Gaussian noise mixing (학습시 적용)

`modeling_qwen3_5AE.py` L197–203:

```python
if self.training:
    batch_size = audio_latents.shape[0]
    k = torch.rand(
        batch_size, 1, 1,
        device=audio_latents.device,
        dtype=audio_latents.dtype,
    ) * 0.1
    epsilon = torch.randn_like(audio_latents)
    projector_input = k.sqrt() * epsilon + (1 - k).sqrt() * audio_latents
```

---

## 2. Dataset

경로: `/mnt/fr20tb/audiollm/sanghyuk/datasets/qwen3_5_dacvae_asr_shuffled_128`

| Code name              | ratio | Train durations (hrs) |
| ---------------------- | ----- | --------------------- |
| en_CommonVoice_single  | 1     | 25,000.00             |
| en_GigaSpeech_single   | 1     | 10,000.00             |
| en_LibriTTS_R_single   | 1     | 552.42                |
| en_MLS_single          | 1     | 45,000.00             |
| en_VoxPopuli_single    | 1     | 522.00                |
| **TOTAL**              | -     | **81,074.00**         |

---

## 3. 학습 코드

- Repo: `oss.navercorp.com/HyperscaleAI/audiollm-trainer` → **acoustic** branch 사용
- `README.md`는 무시
- training config → yaml 파일로 저장

### training config (yaml)

```yaml
### model
model_name_or_path: /mnt/fr20tb/audiollm/sanghyuk/Qwen3_5-4B-DACVAE
flash_attn: fa2
enable_liger_kernel: true
disable_gradient_checkpointing: false
trust_remote_code: true

### method
stage: omni
do_train: true
do_eval: false
finetuning_type: full
deepspeed: {{PATH OF `ds_z2_config.json`}}
# choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]
compute_accuracy: false
report_to: [tensorboard, wandb]

### dataset
dataset: speechx_v9  # 의미 없음
omni_manifest: /mnt/fr20tb/audiollm/sanghyuk/datasets/qwen3_5_dacvae_asr_shuffled_128
load_from_nubes: true
omni_packing_bucket_size: 256
omni_max_audio_samples: 2160000
template: omni  # 의미 없음
cutoff_len: 3584
overwrite_cache: false
preprocessing_num_workers: 16
streaming: true
packing: true
neat_packing: true
preprocessing_batch_size: 2
predict_with_generate: false
dataloader_pin_memory: true
dataloader_num_workers: 4
dataloader_prefetch_factor: 4
dataloader_persistent_workers: false
accelerator_config:
  split_batches: false
  dispatch_batches: false
  non_blocking: true

### output
run_name: Qwen3.5AE-ASR-Stage1
output_dir: /mnt/tmp/results  # 하위에 run_name 디렉터리 안에 저장됨
logging_dir: /mnt/tmp/results  # tensorboard log 저장 위치
logging_steps: 5
save_steps: 1000
save_total_limit: 8
plot_loss: false
overwrite_output_dir: true

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 1
learning_rate: 2.0e-4
weight_decay: 0.01
max_steps: 100000
lr_scheduler_type: warmup_stable_decay  # or cosine
warmup_steps: 1000
bf16: true
ddp_timeout: 180000000

### eval
eval_dataset: speechx_v9  # 의미 없음
```

---

## 4. 학습 실행 스크립트 (NSML)

```bash
pip install flash-linear-attention
pip install causal-conv1d

export WANDB_MODE=online
export WANDB_PROJECT={{YOUR_WANDB_PROJECT_NAME}}
export WANDB_API_KEY={{YOUR_WANDB_API_KEY}}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

FORCE_TORCHRUN=1 \
NNODES=$NSML_WORLD_SIZE \
NODE_RANK=$NSML_RANK \
MASTER_ADDR=$NSML_HOST_RANK0 \
MASTER_PORT=21267 \
llamafactory-cli train {{CONFIG_YAML_FILE_PATH}}
```

---

## 5. Stage1 — Projector-only 학습

- Stage1 에서는 projector 만 학습하도록 설정 (관련 코드는 oss.navercorp.com 해당 경로 참고)
- projector size: **18,158,080 (18M)**
