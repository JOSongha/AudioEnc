# Stage-1 DAC-VAE v4 — Per-modality interleave dataflow

> Config: [`configs/ASR/stage1_dac_vae_v4.yaml`](../../configs/ASR/stage1_dac_vae_v4.yaml)
> Run script: [`scripts/ASR/run_stage1_dac_vae_v4.sh`](../../scripts/ASR/run_stage1_dac_vae_v4.sh)
> Manifest dir: yaml `omni_per_modality_manifests`에 modality별 path. 각 dir은 v3 shards로 symlink.
> Loader: [`src/llamafactory/data/loader.py:get_omni_dataset`](../../src/llamafactory/data/loader.py#L600-L812)

## v3 vs v4 — 단 한 가지 차이

데이터 자체는 동일. **mixing 방식만** 변경.

| | v3 | v4 |
|---|---|---|
| Manifest layout | 단일 dir, alphabetical concat | modality 별 dir |
| 데이터 등장 순서 | 같은 modality shards가 streaming 순으로 묶여서 등장 → 한 packed sample이 단일 modality로 dominate | 매 row 가 확률 추첨 → packed sample 안에 3 modality 섞임 |
| ASR/sound/emotion 비율 | 11.3M : 690k : 43k = 94% : 5.7% : 0.36% (rows ratio 그대로) | **0.65 : 0.25 : 0.10 (yaml 명시)** |
| emotion 노출 | 0.36% × pack 우연 → wandb 곡선 sampling noise | 매 batch에 ~10% 보장 |

## 데이터 (12.08M rows / 199 shards)

| modality | rows | shards | 비율 | 주요 source |
|---|---:|---:|---:|---|
| `audio_asr` | 11,345,224 | 128 | 0.65 | mls 10.8M + libritts-r 354k + voxpopuli 182k |
| `audio_env_sound` | 690,944 | 55 | 0.25 | laion (freesound 460k + epidemic 76k + bbc 32k + audiostock 9k) + audiocaps 46k + fsd50k 41k + audioset 19k + clotho 5k + macs 4k |
| `audio_emotion` | 43,203 | 16¹ | 0.10 | dailytalk 24k + meld 11k + emovdb 7k + ravdess 1.4k + mustardpp 1 |

¹ emotion v3 6 shards → v4 launch 시 row-level 16 shard 균등 split (HF datasets streaming `.shard()` 가 file-level 분할만 지원 → world_size=8 보다 file 적으면 IndexError 회피).

제외(`v3_quarantine`): cremad, iemocap — 외부 사용자 소유 데이터로 본 v4 학습 풀에서 제외.

## Pipeline — 비율은 어디서 적용되나

```
[A] audio_asr stream     ─┐
[B] audio_env_sound       ├─→ interleave_datasets(probs=[0.65, 0.25, 0.10])  ← 여기서만
[C] audio_emotion        ─┘     stopping_strategy=all_exhausted
                                  │
                                  │  row 단위 추첨 — 매 row 가 [A]/[B]/[C] 중 하나
                                  ▼
                          processor_fn (batched=32)
                                  │ row → (audio waveform, prompt tokens, target tokens)
                                  ▼
                          packer_fn (bucket=128, neat_packing=true)
                                  │ rows를 cutoff_len=3584 안에 concat
                                  ▼
                          per_device_train_batch_size=2 (packed samples / GPU)
                                  ▼
                          global batch = 2 × grad_accum=1 × world=8 = 16 packed samples
```
```
[A] audio_asr stream    ─┐
[B] audio_env_sound      ├─→ interleave_datasets(probs=[0.65, 0.25, 0.10])  ← 여기서 row 뽑힘
[C] audio_emotion       ─┘
                              │ row마다 [A]/[B]/[C] 중 하나 확률로 선택
                              ▼
                         processor_fn (batched=32)         ← 32 rows × 65/25/10 분포
                              ▼
                         packer_fn (bucket=128)             ← 128 row → cutoff_len(3584)로 concat
                              ▼
                         per_device_train_batch_size=2      ← 2 packed samples / GPU
                              ▼
                         global batch = 2 × 1 × 8 = 16 packed samples
```

**핵심:**

- 비율 = **row(example) level**, packing/배치 *전*에 한 번만
- Pack 한 단위 안에 여러 modality 섞임 (interleave random order 그대로 concat)
- Batch 도 동일 비율 유지 — 대수적으로 row 비율이 끝까지 보존

코드 위치: [`loader.py:_build_dataset:740-764`](../../src/llamafactory/data/loader.py#L740-L764)

```python
sub_datasets = []
for idx, mod_name in enumerate(modalities):
    sub_seed = training_args.seed + idx * 1_000_003   # modality 별 독립 shuffle
    sub_datasets.append(_load_source(per_mod_manifests[mod_name], shuffle=shuffle, ...))

ds = interleave_datasets(
    sub_datasets,
    probabilities=probs,                               # [0.65, 0.25, 0.10]
    seed=training_args.seed,
    stopping_strategy=data_args.omni_per_modality_stopping,   # all_exhausted
)
```

## 확률 정상화 / 데이터 소진

- `probs` sum이 1.0에서 ±1e-3 벗어나면 자동 normalize ([loader.py:734-738](../../src/llamafactory/data/loader.py#L734-L738))
- `stopping_strategy=all_exhausted`: emotion (43k) 가 일찍 다 돌았어도 cycle 재진입, ASR (11.3M) 가 끝까지 진행되도록 함
- 100k step (effective GBS 16, packed sample) 기준 대략 — emotion ≈ 8 epoch / ASR ≈ 1 epoch (정확값은 packed sample 내 평균 row 수에 의존)

## logging_steps 50 (v3의 5에서 변경)

v3 (5-step logging) 에서는 emotion 0.36% × 16 GBS × 5 step = 평균 0.3 emotion sample / log → wandb `audio_emotion/loss` 곡선이 sampling noise 로 dominate. v4 (50-step) + interleave 0.10 → 50 × 16 × 0.10 = 80 emotion sample / log → 안정 곡선.

## 실행

```bash
# repo root에서:
bash scripts/ASR/run_stage1_dac_vae_v4.sh
# 또는 직접:
llamafactory-cli train configs/ASR/stage1_dac_vae_v4.yaml
```

## 의존 변경 (v3 → v4)

- `configs/ASR/stage1_dac_vae_v4.yaml` — 신규. `omni_per_modality_manifests` + `omni_per_modality_probs` + `omni_per_modality_stopping`. logging_steps 5 → 50
- `scripts/ASR/run_stage1_dac_vae_v4.sh` — 신규. v3.sh 와 환경 동일, config path 만 v4
- `src/llamafactory/data/loader.py:617-621` — patch. `omni_manifest` OR `omni_per_modality_manifests` 둘 중 하나 만족하면 통과 (기존 v3 호환 유지)
- `<datasets_root>/manifests/v3_emotion_split/` — 16 jsonl. v4 `audio_emotion` symlink target. v3 6 shards의 row union 후 균등 분할 (HF datasets streaming `.shard()`가 file-level 분할만 지원, world_size=8보다 file 적으면 IndexError 회피)
