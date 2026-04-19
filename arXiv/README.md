# arXiv — 아카이브 인덱스

현재 파이프라인에서 사용하지 않는 코드·문서 보관소. 삭제하지 않고 이유 + 대체 경로를 남겨 둔다.

**원칙**
- 더 이상 import / 호출되지 않는 파일은 여기로 옮긴다.
- 각 이동은 아래 표에 한 줄로 기록 (언제 / 왜 / 현재 대체재).
- 디렉토리는 원래 역할 기준으로 `data/` `docs/` `scripts/` 분류.
- §43 (2026-04-19): `scripts/` 를 `train/`, `precompute/`, `debug/` 하위 카테고리로 재정리.

## scripts/train/legacy/

초기 2-stage 학습 루프 (raw audio + 작은 DynamicBatchSampler 배치). precomputed Arrow + Sequence Packing 경로로 전환된 후 모두 아카이브.

| 파일 | 원래 역할 | 아카이브 사유 | 대체재 |
|---|---|---|---|
| [train.py](scripts/train/legacy/train.py) | 초기 2-stage ASR 학습 루프 (raw audio 입력). `torchrun --nproc_per_node=8 train.py --encoder encodec` | precomputed Arrow + Sequence Packing 경로 (train_pipeline_override.py) 로 전환 | `train_pipeline_override.py` |
| [train_debug.py](scripts/train/legacy/train_debug.py) | train-clean-100 만으로 빠른 수렴 검증용 1-GPU 디버그 루프 | 현재는 `train_pipeline_override.py --datasets ls100` 로 대체 가능 | `train_pipeline_override.py` |
| [train_stage1.py](scripts/train/legacy/train_stage1.py) | Stage 1 projector alignment 단독 학습 (legacy raw 경로) | precomputed 경로의 Stage 1 은 `train_pipeline_override.py` 내부 `run_stage1()` 으로 통합됨 | `train_pipeline_override.py` `run_stage1()` |
| [train_stage1_eos_first.py](scripts/train/legacy/train_stage1_eos_first.py) | Stage 1 변형 — collate 시 EOS 를 pad 이전에 배치하는 실험 (EOS masking 버그 수정 관련) | 원본 버그 수정 후 필요 없어짐 ([docs/eos_masking_bug.md](docs/eos_masking_bug.md)) | 현 collate 로직 내재화 |
| [dynamic_batching.py](scripts/train/legacy/dynamic_batching.py) | 초기 DynamicBatchSampler — token 예산 기반 동적 배치 | Sequence Packing 도입 후 미사용 ([docs/dynamic_batching.md](docs/dynamic_batching.md)) | `train_pipeline_override.py` 의 packing 경로 |

## scripts/precompute/

Encoder feature precompute + offline packing 스크립트 전량. Stage 1 projector collapse 확정 이후 (§43) 모두 아카이브 — packing 자체가 collapse 해소에 기여 안 함.

| 파일 | 원래 역할 | 아카이브 사유 | 대체재 |
|---|---|---|---|
| [precompute_features.py](scripts/precompute/precompute_features.py) | HF 데이터셋 → 인코더 forward → per-sample Arrow (utterance_id, text, features, feat_len) | 현재 근본 문제가 projector collapse (docs/prompt_format_regression.md §43) 라 전략 재검토 필요. 기존 precomputed arrow 는 eval 용으로만 참조 | — (프레임워크 재설계 예정) |
| [shard_arrow.py](scripts/precompute/shard_arrow.py) | 대용량 per-rank Arrow → 동일 크기 shard 분할 (학습 측 data_split 호환) | 위와 동일 | — |
| [run_precompute.sh](scripts/precompute/run_precompute.sh) | `precompute_features.py` 8 rank 병렬 실행 wrapper | 위와 동일 | — |
| [pack_arrow.py](scripts/precompute/pack_arrow.py) | per-sample precomputed Arrow → packed Arrow 오프라인 패커. `--word-aug` 지원 | 2026-04-15: word-aug 가 word-crop ASR 이 되어 projector collapse 원인 판단 ([docs/word_alignment.md](../docs/word_alignment.md)) | 후속 half_inlv / sentence 버전 → 결국 모두 archive |
| [run_pack.sh](scripts/precompute/run_pack.sh) | 위 pack_arrow.py 8 rank 병렬 + mixed rebalance 호출 wrapper | pack_arrow.py 아카이브와 함께 이동 | — |
| [half_inlv_pack_arrow.py](scripts/precompute/half_inlv_pack_arrow.py) | word-interleave 대안 (문장 token 과 word token 을 interleave) 패커 | 2026-04-16: Stage 1 loss plateau 3.6 여전, projector collapse 동일 | — |
| [run_half_inlv_pack.sh](scripts/precompute/run_half_inlv_pack.sh) | half_inlv_pack_arrow 8 rank wrapper | 위와 함께 이동 | — |
| [sentence_pack_arrow.py](scripts/precompute/sentence_pack_arrow.py) | Sentence-only pack (interleave 완전 제거, cross-dataset shuffle, redistribute rebalance) | 2026-04-17: interleave 없이도 plateau 동일 (cos sim 0.95+ 확인). Packing 포맷이 collapse 원인 아님 확정 | — |
| [run_sentence_pack.sh](scripts/precompute/run_sentence_pack.sh) | sentence_pack_arrow 8 rank wrapper + redistribute rebalance 호출 | 위와 함께 이동 | — |
| [PLAN.md](scripts/precompute/PLAN.md) | precompute / packing 설계 노트 | 참조 자료 | — |

## scripts/debug/

디버그 유틸 + 초기 비교/분석 문서.

| 파일 | 원래 역할 | 아카이브 사유 | 대체재 |
|---|---|---|---|
| [weightCmprsn.py](scripts/debug/weightCmprsn.py) | 학습 중 step 1 / 20 weight 스냅샷 비교 유틸 | 초기 debug 기록 | 참조용 |
| [loss_analysis/](scripts/debug/loss_analysis/) | Stage 1 loss 변동 분석 스냅샷 | 참조용 | — |
| [run_debug.sh](scripts/debug/run_debug.sh) | `train_debug.py` wrapper | `train_debug.py` archived → 함께 이동 | `run.sh` |
| [plan_debug_o.md](scripts/debug/plan_debug_o.md) | `train_debug.py` 디버깅 계획 메모 | 참조 자료 | — |
| [diff_report.md](scripts/debug/diff_report.md) | AudioEnc 모듈화 초기, PoC 원본(`q_enc4.py` / `q_dac_enc.py` / `q_ming.py`) 과 신규 프레임워크 비교 보고서 | 참조 자료 | — |

## data/

| 파일 | 원래 역할 | 아카이브 사유 | 대체재 |
|---|---|---|---|
| [build_librispeech_arrow.py](data/build_librispeech_arrow.py) | LibriSpeech raw flac → HF Arrow 변환 (flac bytes 를 디코딩 없이 저장) | precompute/precompute_features.py 가 HF 데이터셋을 직접 iterable 로 사용해서 pre-stage Arrow 변환 단계 불필요 | `precompute/precompute_features.py` |
| [build_librispeech_hf_arrow.py](data/build_librispeech_hf_arrow.py) | `openslr/librispeech_asr` 전체 split 을 HF 캐시로 다운로드 | 동일 | HF 기본 캐시 자동 사용 |
| [build_gigaspeech_arrow.py](data/build_gigaspeech_arrow.py) | GigaSpeech XL parquet → HF Arrow 변환 (재다운로드 없음) | 동일 | `precompute/precompute_features.py` 에서 바로 소비 |
| [build_voxpopuli_arrow.py](data/build_voxpopuli_arrow.py) | VoxPopuli English → HF Arrow 변환 | 동일 | 동일 |
| [run_build_all.sh](data/run_build_all.sh) | 위 4개 build 스크립트 병렬 실행 wrapper | 위가 전부 아카이브 되어 함께 이동 | — |

## docs/

레거시 또는 특정 시점 스냅샷 문서. 이동 사유는 "현행 doc 으로 통합됨" 또는 "특정 기간 기록 보관".

| 파일 | 주제 | 비고 |
|---|---|---|
| [bucket_batch_sampler.md](docs/bucket_batch_sampler.md) | BucketBatchSampler / DynamicBatchSampler 디자인 | Sequence Packing 도입 후 미사용 |
| [dynamic_batching.md](docs/dynamic_batching.md) | Dynamic Batching 레퍼런스 | 동일 |
| [eos_masking_bug.md](docs/eos_masking_bug.md) | collate 시 EOS 마스킹 버그 분석·수정 기록 | 수정 완료, 참조용 |
| [mimi_hop_and_projector_bugs.md](docs/mimi_hop_and_projector_bugs.md) | Mimi 인코더 hop·projector 불일치 버그 | 수정 완료, 참조용 |
| [mimi_acoustic_vs_semantic.md](docs/mimi_acoustic_vs_semantic.md) | Mimi 두 변형 (acoustic / semantic) 비교 | 인코더 선택 근거 |
| [dac_vs_fb_dacvae.md](docs/dac_vs_fb_dacvae.md) | DAC vs facebook/dacvae-watermarked 비교 | 인코더 선택 근거 |
| [pre_rvq_features.md](docs/pre_rvq_features.md) | "왜 RVQ 이전 continuous 피처를 쓰는가" 설명 | 디자인 근거 |
| [weight_count.md](docs/weight_count.md) | 모델별 파라미터 수 집계 | 참조용 |
| [cherry_pick_log.md](docs/cherry_pick_log.md) | sehyun/fast_train_merged → main cherry-pick 로그 | 병합 완료, 기록 보관 |
| [early_packing_tuning_apr06-07.md](docs/early_packing_tuning_apr06-07.md) | Packing / bucket / cutoff_len 초기 튜닝 로그 (04-06 ~ 04-11) | 특정 기간 기록 |
| [mixed_packing_memory_evolution_apr10-12.md](docs/mixed_packing_memory_evolution_apr10-12.md) | Mixed packing 메모리 진화 (04-10 ~ 04-12) | 특정 기간 기록 |
| [stage2_fsdp_nccl_resolution_apr14.md](docs/stage2_fsdp_nccl_resolution_apr14.md) | Stage 2 FSDP+NCCL 데드락 해결 기록 (04-14) | 특정 기간 기록 |
| [quality_check_results_apr08.json](docs/quality_check_results_apr08.json) | 데이터 품질 검증 결과 스냅샷 (04-08) | 특정 시점 데이터 |
