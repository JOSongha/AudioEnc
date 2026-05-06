# Stage-1 v4 — onboarding

> 동료/신규 인원이 v4 setup을 빠르게 따라잡을 수 있도록 만든 문서 모음. 처음 보는 사람 기준, 위에서 아래로 읽으면 됨.

## 1. v4가 뭐고 왜 만들었나

v3와 데이터 자체는 같음. **bug 하나만 고친 것**:

- v3는 manifest dir 안의 jsonl shard들을 alphabetical concat → streaming 순으로 같은 modality가 묶여 한 packed sequence를 dominate. ASR 11.3M rows : env_sound 690k : emotion 43k 비율이 그대로 노출돼 emotion은 0.36% 등장.
- v4는 modality별 manifest dir + row-level 확률 추첨 (HuggingFace `interleave_datasets` + `omni_per_modality_probs`)으로 매 batch에 ASR 0.65 / env_sound 0.25 / emotion 0.10 비율이 보장됨.

해당 결과: emotion 28 epoch 노출, env_sound 4.5 epoch, ASR 0.7 epoch (cutoff_len 3584 + global batch 16 + 100k step 기준).

## 2. 가장 먼저 읽을 것

1. [`dac_vae_dataflow.md`](dac_vae_dataflow.md) (~110 lines) — v3↔v4 차이 한 표에 정리, manifest 구성/숫자, modality별 source breakdown, ChatML 빌드 흐름. **여기 한 장이 v4 80%**.
2. [`../../reference/projector_block_types.md`](../../reference/projector_block_types.md) — v4에서 새로 추가된 `decoder_block_type` (`llama` 기본 / `qwen3` 선택). projector 구조 차이 + 파라미터 수 비교.

## 3. 실행해 보고 싶을 때

3개 encoder 모두 같은 v4 data wiring (`omni_per_modality_manifests` + `omni_per_modality_probs`) 공유. 차이는 encoder 가중치 + sample rate + audio window cap뿐.

| encoder | config | launcher | sr | audio cap |
|---|---|---|---:|---:|
| DAC-VAE | [`stage1_dac_vae_v4.yaml`](../../../configs/ASR/stage1_dac_vae_v4.yaml) | [`run_stage1_dac_vae_v4.sh`](../../../scripts/ASR/run_stage1_dac_vae_v4.sh) | 48 kHz | 120 s |
| Whisper-small | [`stage1_whisper_small_v4.yaml`](../../../configs/ASR/stage1_whisper_small_v4.yaml) | [`run_stage1_whisper_small_v4.sh`](../../../scripts/ASR/run_stage1_whisper_small_v4.sh) | 16 kHz | 30 s |
| Whisper-tiny | [`stage1_whisper_tiny_v4.yaml`](../../../configs/ASR/stage1_whisper_tiny_v4.yaml) | [`run_stage1_whisper_tiny_v4.sh`](../../../scripts/ASR/run_stage1_whisper_tiny_v4.sh) | 16 kHz | 30 s |

기본값: 8×A100-80GB, ZeRO-2, bf16, packing on, neat_packing on, 100k step, lr 2e-4 warmup_stable_decay 1k warmup, save_steps 1k.

## 4. 학습 상태 (2026-05-06 기준)

| encoder | run name | status |
|---|---|---|
| DAC-VAE | `Qwen3.5AE-ASR-Stage1-dac-vae-v4` | ✅ 100k step 완주, 100 ckpts 평가 완료 |
| Whisper-tiny | `Qwen3.5AE-ASR-Stage1-whisper-tiny-v4` | 🔄 진행 중 (~18k step, 12k에서 NCCL timeout 후 resume) |
| Whisper-small | `Qwen3.5AE-ASR-Stage1-whisper-small-v4` | 🔄 진행 중 (다른 노드) |

학습 산출물 위치: `/mnt/tmp/Qwen3.5_<encoder>_v4_Stage1_jos/Qwen3.5AE-ASR-Stage1-<encoder>-v4/checkpoint-N/`.

## 5. Eval 결과 (DAC-VAE v4)

100 ckpts × 9 task evaluation:

- **결과 root**: `/mnt/tmp/Qwen3.5_dac_vae_v4_Stage1_jos/eval_v4/eval_<task>/checkpoint-N/summary.json` (canonical layout)
- **canonical aggregator + plot**: `_analysis/{results,results_wide}.csv`, `results.md`, `best_per_task.md`, `trajectories.pdf`
  - 생성 명령: `python -m evaluation.stage2.aggregate_results --root <eval_v4> --out <eval_v4>/_analysis` 후 `python -m evaluation.stage2.plot_trajectories --csv ... --out ... --cols 3 --title ... --exclude "ESC-50,Text retention"`
- **method-C composite + group rank trajectories** (커스텀): `_aggregate/{results,composite_fullcov}.csv`, `trajectories{,_raw}.{pdf,png}`
- **MELD class confusion subgraphs**: `_analysis/meld_confusion.{pdf,png}` — 11 ckpt × 7×7 row-normalized heatmap. neutral over-prediction trajectory 한눈에.

**Best ckpt (method-C composite, 73 fully-covered ckpts 중)**:
- 종합 best: **ckpt-68000** (composite 0.665, top-3: 68k > 62k > 66k)
- per-group best: ASR 86k / Sound 47k / Emotion 81k / LISTEN 74k / Knowledge 1k (= LM frozen, near-constant)

## 6. 알려진 이슈 / 주의사항

- **Text retention**: Stage-1은 LM frozen이라 6 benchmark 평균이 모든 ckpt에서 ~0.71로 거의 상수. trajectory plot에서 panel 빼는 것 권장 (`--exclude "Text retention"`).
- **ESC-50 / LISTEN-MCQA / LM benchmarks**: format-following 기반인데 Stage-1은 MCQ format tuning 안 됨. 점수 낮은 게 encoder 성능 문제 아니라 Stage-1 설정의 한계. v4 column을 Stage-2 column과 직접 비교할 때 caption에 명시 필수.
- **MELD class imbalance**: gold 분포 neutral 48%, joy 15%, anger 13%, ..., fear 1.9%. 모델 prediction은 neutral 55% / joy 22% / fear 0.6% 식으로 majority bias 잔존 (ckpt-68000). majority-pure baseline (acc 48%, F1 9%)보다는 위지만 minority recall (fear 2%, disgust 4%) 약함. balanced metric 선호.
- **emotion modality 16 shards**: HF datasets streaming `.shard()`가 file-level 분할만 지원해서 world_size=8보다 file 적으면 IndexError. v3 6 shards를 v4 launch 시 row-level 16 shard로 균등 split. 자세한 건 `dac_vae_dataflow.md` 각주¹.
- **NCCL timeout**: dataloader 단일 worker가 nubes audio 다운로드에서 stall하면 600s 후 NCCL 터짐. omni dataset에 SIGALRM watchdog (30s) + nubes connect/read split timeout 추가됨 (`6cf6bf8f` 커밋). 그래도 12k 근처에서 timeout 한 번 발생.
- **dataset shard 외부 사용자 소유 일부**: `cremad` (shkim), `iemocap` (sehyun + kyudan) 데이터는 v4_quarantine으로 제외. emotion 학습 풀에 안 들어감.

## 7. Paper artifact

논문 working dir: `/mnt/ddn/users/jos/AudioEnc/log/tmp/latex_work/`. v4 관련 표:

- `tbl/encoder_comparison.tex` — Table 2에 "DAC-VAE v4" column 포함. caption에 "DAC-VAE v4 column reports the Stage-1-only multi-task model at its method-C balanced default ckpt-68000 (no Stage-2 fine-tuning, hence weaker on multiple-choice format tasks like ESC-50 and the LM benchmarks)" 명시.
