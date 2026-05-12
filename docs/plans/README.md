# Plans — Representation Richness Analysis

각 plan 은 Haiku 가 단일 세션 내에서 실행 가능하도록 detailed step + 주의사항 + 검증 게이트 포함.

| Plan | 단계 | 산출물 |
|---|---|---|
| [Plan 1](plan_repr_richness_1_download.md) | CMU-ARCTIC 7 화자 + L2-ARCTIC 다운로드 + 매니페스트 | `/mnt/tmp/cache/cmu_arctic_7/manifest.csv`, `l2_arctic/manifest.csv` |
| [Plan 2](plan_repr_richness_2_extraction.md) | 7 화자 셋에서 3 ALM family (encodec 제외) × 37 layers × {mean,last} 임베딩 추출 | `experiments/representation_richness/cmu_arctic_7/<family>/<layer>/<utt>.npz` |
| [Plan 3](plan_repr_richness_3_analysis.md) | Disentanglement + Spectrum 분석, 리포트 | `_results/*.csv`, `_results/figs/*.png`, `docs/stage2_analysis/representation_richness_phase1.md` |

## 결정사항 요약 (Phase 1)

- **데이터**: CMU-ARCTIC 7 화자 only. L2-ARCTIC 다운로드만 (분석은 후속).
- **Family**: whisper-tiny / whisper-small / wavtok-40-unify (3 개). encodec-24k 제외 (ckpt 부재).
- **ckpt**: family 별 가장 최근 step. Plan 2 시작 시 동적 판별 + 사용자 보고.
- **HF token**: `~/.cache/huggingface/token` 표준 경로에 저장됨 (600 권한). plan / code 에 literal 금지.

## 실행 순서

각 Plan 종료 시점에 사용자 검토 / 보고. Plan 1 → 2 → 3 순. 각 Plan 의 "완료 조건" 모두 충족 전에 다음 Plan 시작 X.

## 분석 설계 single source

`docs/analysis_representation_richness.md` — 가설/메트릭/풀링 정의.
