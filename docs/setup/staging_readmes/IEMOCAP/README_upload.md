# IEMOCAP (utterance wav + EmoEvaluation + transcriptions)

업로드 준비: 2026-05-07. nubes 업로드 대상 = `hyperscaleai-audiollm/users/jos/AudioEnc/IEMOCAP/`.

## 배경

audiollm-trainer 의 IEMOCAP 사용:
- **학습 (Stage-1)**: Sessions 1-4 utterance wav (5,882 row, datasets.md § 4)
- **평가 (Stage-2)**: Session 5 utterance wav + EmoEvaluation 4-class label (eval_iemocap_session5.py, 1,650 valid utt → exc→hap merge → 1,241 채점 입력)

nubes 기존 `/datasets/public/IEMOCAP/data/` 에 **10,039 utterance wav 가 이미 있음** (byte-level 우리 로컬과 동일 — Ses01F_impro01_F000.wav 62,302 byte / Ses05F_impro01_F000.wav 81,860 byte 일치). 다만 **session 분리 없이 통합 저장** + **EmoEvaluation 라벨 / transcriptions 부재**.

## 옵션 비교 + A 결정 사유

| 옵션 | 내용 | 크기 | Leak 위험 | 결정 |
|---|---|---:|---|---|
| **A (선택)** | Sessions 1-5 별로 sentences/wav (10,039) + dialog/EmoEvaluation (151 .txt) + dialog/transcriptions (151 .txt) | **1.4 GB** | ✓ 안전 (session 분리 보존, builder prefix filter 불필요) | ✓ |
| B | EmoEvaluation + transcriptions 만 (audio 는 nubes 기존 활용) | 20 MB | ⚠ builder 가 `Ses0[1-4]_` filename prefix filter **필수**. filter bug 시 Session 5 (eval) 가 학습에 leak | X |
| C | A + dialog wav (151 대화 wav) + ForcedAlignment (음성 정렬) | 4 GB | ✓ | X — audiollm-trainer 학습/eval 미사용 |
| D | 전체 23 GB (avi, MOCAP head/hand/rotated 등 multimodal) | 23 GB | ✓ | X — audio-only task 에 불필요 |

**옵션 A 선택 사유**:
1. **Leak 안전**: session 분리 보존 → builder 가 `Session1/` ~ `Session4/` 디렉터리로 학습 split 자동 결정. Session 5 절대 학습 풀 X. 옵션 B 의 filename prefix filter 의존성 (bug 위험) 회피.
2. **운영 단순**: nubes 기존 통합 영역 (`/datasets/public/IEMOCAP/data/`) 도 그대로 두고 사용자 영역에 session-aware 형태 별도 보존. Builder 코드 가 hardcode session path 만 보면 됨.
3. **비용 합리**: 1.4 GB 중복 비용은 작음 (audio quality / leak 안전성 trade-off 우위).
4. **EmoEvaluation / transcriptions 가 nubes 어디에도 부재**. 학습/eval 에 필수라 어떤 옵션에서든 업로드 필요.
5. 옵션 C 의 dialog wav (대화 단위 wav) 와 ForcedAlignment 는 audiollm-trainer task (utterance-level emotion classification) 와 무관. 옵션 D 는 multimodal data 라 audio-only task 에 over-spec.

## 내용

| 디렉터리 | 파일 수 | 크기 | 설명 |
|---|---:|---:|---|
| `Session{1-5}/sentences/wav/<dialog>/<utt>.wav` | 1,819 / 1,811 / 2,136 / 2,103 / 2,170 = **10,039** | 1.4 GB | utterance wav (Stage-1 학습 + Stage-2 eval 핵심) |
| `Session{1-5}/dialog/EmoEvaluation/<dialog>.txt` | 28 / 30 / 32 / 30 / 31 = **151** | < 1 MB | **합의 emotion 라벨** + annotator 4명 의견. eval_iemocap_session5.py 가 직접 파싱 |
| `Session{1-5}/dialog/EmoEvaluation/Attribute/...` | (annotator 별 V/A/D 펼친 파일들) | ~6 MB | dimensional V/A/D annotation per annotator. audiollm-trainer 미사용 |
| `Session{1-5}/dialog/EmoEvaluation/Categorical/...` | (annotator 별 categorical 펼친 파일들) | ~6 MB | categorical emotion per annotator. audiollm-trainer 미사용 |
| `Session{1-5}/dialog/EmoEvaluation/Self-evaluation/...` | (화자 본인 self-rating) | ~6 MB | speaker self-rating (V/A/D + categorical). audiollm-trainer 미사용 |
| `Session{1-5}/dialog/transcriptions/<dialog>.txt` | 151 | 1.1 MB | per-utt transcription. emotion task 미사용, ASR / multimodal 시 사용 가능 |
| `README.txt` | 1 | (small) | upstream README |
| `README_upload.md` | 1 | ~5 KB | 본 문서 |

총 1.4 GB, ~10,341 file (10,039 wav + 표준 metadata + README).

### EmoEvaluation/<dialog>.txt 형식 (eval 핵심)

```
[6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]
C-E2:	Neutral;	()                    ← annotator E2 의 categorical
C-E3:	Neutral;	()
A-E3:	val 3; act 2; dom 2;	()        ← annotator E3 의 V/A/D
...
```
첫 token (예: `neu`, `xxx`) 가 다수결 **합의 label** — eval_iemocap_session5.py 가 regex 로 추출. `xxx` = annotator 합의 못함 (drop 대상).

### sub-dirs (audiollm-trainer 미사용)

| sub-dir | 내용 | 예 |
|---|---|---|
| `EmoEvaluation/Attribute/<dialog>_<annotator>_atr.txt` | annotator 별 V/A/D | `Ses01F_impro01_F000 :act 4; :val 3; :dom 2;` |
| `EmoEvaluation/Categorical/<dialog>_<annotator>_cat.txt` | annotator 별 categorical | `Ses01F_impro01_F000 :Neutral state;` |
| `EmoEvaluation/Self-evaluation/<dialog>_<f1\|m1>_*.txt` | **화자 본인** self-rating | `Ses01F_impro01_F000 :act 4; :val 3; :dom 1;` |

`.anvil` 은 annotation tool binary, 표준 distribution 의 일부.

### transcriptions/<dialog>.txt 형식

```
Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me.
Ses01F_impro01_M000 [007.5712-010.4750]: Do you have your forms?
```

emotion eval 코드는 미사용. ASR / multimodal task 또는 contextual 분석 시 활용 가능.

## Source

- 로컬: `/mnt/tmp/datasets/emotion_raw/IEMOCAP/IEMOCAP_full_release/` (jos own download, 23 GB 전체 중 옵션 A 분만 staging 으로 cp)
- 원본: USC SAIL, Busso et al. 2008 (academic-only EULA)

## 사용 방법

`eval_iemocap_session5.py` 의 path:
```python
ROOT = Path("/mnt/tmp/datasets/emotion_raw/IEMOCAP/IEMOCAP_full_release")
SESSION = "Session5"
# audio: ROOT / SESSION / "sentences/wav/<dialog>/<utt>.wav"
# label: ROOT / SESSION / "dialog/EmoEvaluation/<dialog>.txt"
```

nubes 사용 시 ROOT 를 nubes path 로 override 또는 다운로드 후 로컬 캐시.

## 라이선스

USC SAIL IEMOCAP academic-only EULA (https://sail.usc.edu/iemocap/iemocap_release.htm). 비상업/연구 목적만, 별도 EULA 동의 후 사용.
