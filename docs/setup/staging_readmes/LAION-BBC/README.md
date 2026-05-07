# LAION-BBC (audio 15,973 + v6 caption metadata)

업로드 준비: 2026-05-07. nubes 업로드 대상 = `hyperscaleai-audiollm/users/jos/AudioEnc/LAION-BBC/`.

## 배경

nubes 의 `/datasets/public/LAION-Audio-630k/bbc_sound_effects/audio/` 에는 **2,000 flac 만** 있음 (`audio/train/` 1,000 + `audio/test/` 1,000, ID 14-14367 + 14378-15972). 학습 풀 (15,968 unique audio × 2 caption variants = 31,936 row) 매핑 시 부족.

### Nubes 기존 vs 우리 staging ID 비교 (2026-05-07 검증)

| 비교 | 결과 |
|---|---|
| Nubes BBC ID 셋 (2,000) ⊆ Ours (15,973)? | ✓ True (전부 포함) |
| Nubes only | 0 |
| Ours only (superset 차이) | 13,973 |
| Byte-level 동일 (ID 14 / 7553 / 15972 spot-check, 1,884,047 / 714,890 / 1,658,236) | ✓ 모두 정확 일치 |

즉 nubes 기존 LAION-BBC 2K 는 우리 superset 15,973 의 진부분집합 (동일 LAION 공식 ID, byte 동일). 우리는 superset 통째로 사용자 영역에 업로드 — nubes 2K 와 의도적 중복 (78 GB 중 ~10 GB 정도가 nubes 와 동일 byte). 사용자 영역 단순성 우선.

## 내용

| 디렉터리 | 파일 수 / 크기 | 설명 |
|---|---|---|
| `audio/<id>.flac` | 15,973 flac / 78 GB | LAION 공식 BBC SE subset (`<id>.flac` 단순 평면 명명, 1.flac ~ 16K.flac 범위) |
| `metadata/laion_bbc_train_0000.jsonl` | 15,000 row / 2.9 MB | v6 train shard 0 (audio_path + caption + source label) |
| `metadata/laion_bbc_train_0001.jsonl` | 13,742 row / 2.7 MB | v6 train shard 1 |
| `metadata/laion_bbc_test_0000.jsonl` | 3,194 row / 622 KB | v6 test shard (학습 시 train+test 모두 사용) |

총 78 GB + 6 MB metadata.

## Source

- 로컬 audio: `/mnt/tmp/datasets/laion_extracted/bbc/` (LAION-Audio-630k 의 bbc_sound_effects subset, extracted from HF dump)
- 로컬 metadata: `/mnt/tmp/datasets/manifests/v6/audio_env_sound/laion_bbc_{train,test}_*.jsonl`
- 원본 distribution: [LAION-Audio-630k](https://github.com/LAION-AI/audio-dataset) (CC0 / clip 별 상이, 대부분 CC-BY)

## v6 manifest row 분석

- 31,936 row = **15,968 unique audio × 2 caption variant** (audio 1개당 caption 2개)
- 로컬 디스크: 15,973 flac (manifest 인용 15,968 + extra 5)

## Caption / leak 위험

- canonical split 없음 (LAION 자체 train/test 분리 — 학습 시 둘 다 사용, 학습 풀 외부 eval source 와 leak 위험 없음)
- builder 가 `audio_path: /mnt/tmp/datasets/laion_extracted/bbc/<id>.flac` 으로 인용 → nubes-direct 사용 시 `nubes_path: hyperscaleai-audiollm/users/jos/AudioEnc/LAION-BBC/audio/<id>.flac` 으로 rewrite 필요 (별도 builder 작업)

## 업로드 후 검증 항목

| 항목 | 목표 |
|---|---:|
| audio/*.flac | 15,973 |
| metadata/*.jsonl | 3 file (test 1 + train 2) |
| metadata 안 audio_path 의 unique 수 | 15,968 |

## 라이선스

LAION-Audio-630k 의 LAION 공개 라이선스 (CC0 / clip 별 상이). 원 BBC Sound Effects 라이선스는 BBC 의 [Personal, educational, or research use 한정](https://sound-effects.bbcrewind.co.uk/licensing) — 상업 배포 시 별도 검토 필요.
