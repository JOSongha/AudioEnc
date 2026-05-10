# FSD50K (eval + metadata 보완)

업로드 준비: 2026-05-07. nubes 업로드 대상 = `hyperscaleai-audiollm/users/jos/AudioEnc/FSD50K/`.

## 배경

nubes 의 `/datasets/public/FSD50K/` 에는 **dev audio (40,966 wav) + AF-Think_*.jsonl** 만 있고 ground_truth / metadata / eval split 모두 부재. Stage-2 평가 ([`eval_fsd50k_map.py`](../../../audiollm-trainer/evaluation/stage2/eval_fsd50k_map.py)) 가 사용하는 `eval_audio + eval.csv + vocabulary.csv` 가 없고, 표준 FSD50K distribution 의 metadata 도 누락. 본 업로드로 dev_audio 외 표준 FSD50K 컴포넌트 일괄 보완.

## 내용 (표준 Zenodo FSD50K v1.0 layout 유지)

| 디렉터리 | 파일 수 / 크기 | 설명 |
|---|---|---|
| `FSD50K.eval_audio/` | 10,231 wav / 8.3 GB | 16 kHz mono, eval split |
| `FSD50K.ground_truth/eval.csv` | 1 file / 997 KB | per-clip eval label (10,231 row, `fname,labels,mids`) |
| `FSD50K.ground_truth/dev.csv` | 1 file / 1.6 MB | per-clip dev label (40,966 row) — 학습 builder 가 nubes-direct 화 시 사용 |
| `FSD50K.ground_truth/vocabulary.csv` | 1 file / 5 KB | 200-class vocab (라벨 idx → 이름 + freebase mid) |
| `FSD50K.metadata/class_info_FSD50K.json` | 1 file | 클래스 hierarchy / parent-child |
| `FSD50K.metadata/dev_clips_info_FSD50K.json` | 1 file | dev clip 별 freesound metadata |
| `FSD50K.metadata/eval_clips_info_FSD50K.json` | 1 file | eval clip 별 freesound metadata |
| `FSD50K.metadata/pp_pnp_ratings_FSD50K.json` | 1 file | PP/PNP rating annotation |
| `FSD50K.metadata/collection/` | 폴더 | TaxonomyTree / collection metadata |
| `FSD50K.doc/{LICENSE-DATASET, README.md}` | 2 files / 20 KB | 표준 distribution doc |

총 8.3 GB + 32 MB metadata.

## Source

- 로컬 (둘 동일, 내용 diff empty): 
  - `/mnt/tmp/datasets/env_sound/FSD50K/`
  - `/mnt/ddn/users/jos/AudioEnc/log/tmp/datasets/env_sound/FSD50K/`
- 원본: [Zenodo FSD50K v1.0](https://zenodo.org/record/4060432) (CC BY 4.0, Fonseca et al. 2022)

## 사용 방법

Stage-2 eval ([`eval_fsd50k_map.py`](../../../audiollm-trainer/evaluation/stage2/eval_fsd50k_map.py)) path 변수:

```python
FSD50K_ROOT = Path("/mnt/tmp/datasets/env_sound/FSD50K")
EVAL_CSV = FSD50K_ROOT / "FSD50K.ground_truth/eval.csv"
VOCAB_CSV = FSD50K_ROOT / "FSD50K.ground_truth/vocabulary.csv"
EVAL_AUDIO_DIR = Path(os.environ.get(
    "FSD50K_EVAL_AUDIO_DIR",
    str(FSD50K_ROOT / "FSD50K.eval_audio"),
))
```

nubes-direct 사용 시 audio 다운로드 + 로컬 캐시 (`FSD50K_EVAL_AUDIO_DIR` env override) 또는 nubes-aware loader 추가 필요. 본 업로드는 데이터 보존이 목적, 코드 수정은 별개.

## Row count 검증

| 항목 | Count | 일치 |
|---|---:|---|
| eval_audio wav | 10,231 | ✓ |
| eval.csv rows | 10,231 | ✓ (audio 와 1:1) |
| dev.csv rows | 40,966 | ✓ (nubes `/datasets/public/FSD50K/audio/` 의 dev count 와 동일) |
| vocabulary.csv classes | 200 | ✓ |

## 라이선스

CC BY 4.0 (Fonseca et al. 2022). 클립별 freesound 라이선스는 `dev_clips_info / eval_clips_info` 의 `license` 필드 참조.
