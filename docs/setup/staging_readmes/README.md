# Staging READMEs (nubes 업로드 source 별 README archive)

`/mnt/tmp/staging/jos_AudioEnc/<source>/` (= `/mnt/ddn/users/jos/AudioEnc/log/tmp/staging/jos_AudioEnc/<source>/`) 의 README 들을 audiollm-trainer repo 안에 archive. 각 source 별로 staging 시 작성한 `README_upload.md` (옵션 / 결정 사유 / 검증 결과 등) 와 upstream `README.md` (있는 경우) 보존.

원본 staging dir 은 transient (cleanup 가능) 이라 repo 안 archive 가 영구 reference.

## 디렉터리 매핑

| Source | Files | § 12.x | 비고 |
|---|---|---|---|
| `AudioSet/` | README.md + README_upload.md | § 12.7 | upstream + 옵션 비교 (C 결정: bal_train + eval + ontology) |
| `EmoV-DB/` | README_upload.md + repo/README.md | § 12.5 | 4 화자 통째 학습 (v5 leak-fix Jenie 폐기) |
| `FSD50K/` | README.md + FSD50K.doc/README.md | § 12.1 | upstream FSD50K.doc + eval split 보완 |
| `IEMOCAP/` | README_upload.md | § 12.4 | 옵션 A (sentences/wav + EmoEvaluation + transcriptions) + metadata 구조 |
| `LAION-BBC/` | README.md | § 12.3 | nubes 기존 vs 우리 superset ID 비교 + 옵션 B |
| `MUStARD_Plus_Plus/` | README.md + README_upload.md | § 12.2 | upstream + staging |
| `RAVDESS/` | README_upload.md | § 12.6 | filename schema + audio-only modality (24 actors 통째) |
| `MACS/` | README_upload.md | § 12.8 | 옵션 C (yaml backup, audio 는 nubes public 이용) |
| `AudioCaps/` | README_upload.md | § 12.9 | 옵션 C2 (audio 46,506 flac + parquet 473 표준 dist) |
| `MELD-CSV/` | README_upload.md | § 12.10 | train/dev/test emotion label csv (사용자 영역) |
| `DailyTalk/` | README_upload.md | § 12.11 | utterance wav 23,773 (nubes 기존 dialogue 영역 zero-byte placeholder 대체) |
| `Clotho-v2/` | README_upload.md | § 12.12 | dev/eval/val split 별 subdir + caption csv 3종 |
| `MELD-audio-wav/` | README_upload.md | § 12.14 | mp3 → wav 전환 (libsndfile mp3 디코드 inconsistent 해결) |
| `LAION-Freesound/` | README_upload.md | § 12.15 | audio 460,141 flac + meta csv (~607 GB, 진행 중) |

## 관련 문서

업로드 자체 / nubes 매핑 / 검증 결과는 [`../nubes_upload.md`](../nubes_upload.md) § 12.x 참고.
