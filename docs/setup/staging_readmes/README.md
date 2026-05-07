# Staging READMEs (nubes 업로드 source 별 README archive)

`/mnt/tmp/staging/jos_AudioEnc/<source>/` (= `/mnt/ddn/users/jos/AudioEnc/log/tmp/staging/jos_AudioEnc/<source>/`) 의 README 들을 audiollm-trainer repo 안에 archive. 각 source 별로 staging 시 작성한 `README_upload.md` (옵션 / 결정 사유 / 검증 결과 등) 와 upstream `README.md` (있는 경우) 보존.

원본 staging dir 은 transient (cleanup 가능) 이라 repo 안 archive 가 영구 reference.

## 디렉터리 매핑

| Source | Files | 비고 |
|---|---|---|
| `AudioSet/` | README.md (staging) + README_upload.md | upstream README.md 5,195 byte / staging README_upload.md 옵션 비교 + C 결정 사유 |
| `EmoV-DB/` | README_upload.md + repo/README.md | repo/ 안에 upstream README |
| `FSD50K/` | README.md (staging) + FSD50K.doc/README.md | upstream FSD50K.doc/README + staging README |
| `IEMOCAP/` | README_upload.md | 옵션 A 결정 사유 (sentences/wav + EmoEvaluation + transcriptions) + metadata 구조 설명 |
| `LAION-BBC/` | README.md (staging) | nubes 기존 vs 우리 superset ID 비교 + 옵션 B 결정 |
| `MUStARD_Plus_Plus/` | README.md + README_upload.md | upstream + staging |
| `RAVDESS/` | README_upload.md | filename schema + audio-only modality 결정 |

## 관련 문서

업로드 자체 / nubes 매핑 / 검증 결과는 [`../nubes_upload.md`](../nubes_upload.md) § 12.x 참고.
