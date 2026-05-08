# LAION-Freesound (audio + meta csv)

업로드 진행 중: 2026-05-08 시작 (~60min ETA). nubes 업로드 대상 = `hyperscaleai-audiollm/users/jos/AudioEnc/LAION-Freesound/`.

## 배경

audiollm-trainer 의 LAION-Freesound 사용:
- **Stage-1 학습**: 460,141 flac (datasets.md § 3 sound captioning, env_sound 의 67%)

nubes top-level 의 `/datasets/public/Freesound/audio/` 가 LAION subset 인 줄 알고 매핑 시도했으나 200-sample ID 매칭 검증 결과 **다른 dump 임 확인** (60% miss + 80 hit 도 file size 0건 일치, 같은 ID 라도 다른 encoding/quality). LAION-Audio-630k subset 매핑 불가.

→ LAION 본을 사용자 영역에 직접 업로드.

## 옵션 비교 + 결정 사유

| 옵션 | 내용 | 크기 | 결정 |
|---|---|---:|---|
| **A (선택)** | extracted flac 모두 (`/mnt/tmp/datasets/laion_extracted/freesound/`) + metadata csv 2 + README | ~607 GB | ✓ 학습 친화적 (random access flac) |
| B | source tar (`freesound_no_overlap/{train_1, train_2, test}/*.tar`) 그대로 | 609 GB | X — 학습 시 tar 풀어야 해서 비효율, 파일 수만 적음 |

**A 선택 사유**:
1. extracted flac 가 random access 효율적 (학습 dataloader 가 .flac 직접 fetch)
2. metadata csv 도 함께 보존 (caption + freesound_id + license)
3. 460k 파일 수가 많지만 dir-upload `-j 16` parallel 로 처리 가능

## 내용

| 디렉터리/파일 | 파일 수 | 크기 | 설명 |
|---|---:|---:|---|
| `audio/<id>.flac` | 460,141 | ~607 GB | flat dir, freesound_id 명명 |
| `freesound_meta.csv` | 1 | 105 MB | full LAION-Freesound caption metadata (audio_filename, caption1, caption2, freesound_id, username, freesound_url, license) |
| `freesound_no_overlap_meta.csv` | 1 | 94 MB | LAION no-overlap split metadata (train_1 / train_2 / test 분리 표시) |
| `README.md` | 1 | ~5 KB | upstream README |
| `README_upload.md` | 1 | ~3 KB | 본 문서 |

## audiollm-trainer 사용

- **Stage-1 학습**: 460,141 row, datasets.md § 3
- **Builder**: 현재 LAION-Freesound 전용 builder 부재 (legacy ddn-extracted dir 직접 사용). 본 업로드 후 `build_laion_freesound.py` 작성 가능 (csv → row 변환, nubes_path 직접 박음).
- [`rewrite_audio_paths_nubes.py`](../../scripts/manifest_builders/rewrite_audio_paths_nubes.py) `PREFIX_MAPPINGS` 에 매핑 추가 예정 (audio 업로드 완료 후):
   ```python
   "laion_freesound": (
       "/mnt/tmp/datasets/laion_extracted/freesound/",
       "hyperscaleai-audiollm/users/jos/AudioEnc/LAION-Freesound/audio/",
   ),
   ```

## 검증

- csv + README 업로드 완료
- audio 460,141 flac 진행 중 (~60min ETA, dir-upload -j 16)
- 완료 후 추가: count + sample HEAD

## Source

- 로컬 extracted: `/mnt/tmp/datasets/laion_extracted/freesound/` (607 GB, 460,141 flac)
- 로컬 source: `/mnt/tmp/datasets/laion_freesound/freesound_no_overlap/{train_1,train_2,test}/*.tar` (609 GB, 900 tar)
- upstream: [LAION-Audio-630k Freesound subset](https://laion.ai/blog/laion-audio-630k/) (Wu et al. 2023)

## 라이선스

Clip-level 다양 (CC0 / CC-BY 등). csv 의 `license` 컬럼에 per-clip 명시.

## 관련 문서

- [`../nubes_upload.md` § 12.15](../nubes_upload.md)
- [`../nubes_upload.md` § 9.3](../nubes_upload.md) (LAION-Freesound 매핑 검증 → `/Freesound/` 다른 dump 결론)
- [`../datasets.md` § 3 LAION-Freesound row](../datasets.md)
