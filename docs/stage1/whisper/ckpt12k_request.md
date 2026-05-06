# Whisper-small ckpt-12000 추가 데이터 요청 (다른 노드)

작성: 2026-04-29 06:10
용도: 본 노드에서 작성한 encoder 비교 표 ([`tbl/encoder_comparison.tex`](/mnt/ddn/users/jos/AudioEnc/log/tmp/latex_work/tbl/encoder_comparison.tex)) 의 Whisper-small 컬럼 마무리.

---

## 배경

Encoder 비교 표를 unified-checkpoint 방식으로 작성 중 — 각 encoder column 의 모든 metric 이 **동일한 단일 ckpt** 의 값:

| Encoder | Representative ckpt | rationale |
|---|---|---|
| DAC-VAE v2 | 15,000 | docs/stage2/3model_comparison.md §1 권장 balanced default |
| Whisper-tiny | 6,000 | §2 broad sweet spot |
| **Whisper-small** | **12,000** | §3 의 13-17k balanced 권장 범위 중 raw snapshot 가능한 가장 가까운 값 |

Whisper-small 의 raw `summary.json` 파일들이 본 노드 (`/mnt/tmp/results/Qwen3.5AE-Stage2-whisper-small-emoFull-asr033-env05-txt03/`) 에는 ckpt 디렉터리만 있고 `eval_*/` 평가 결과 디렉터리가 없음 (다른 노드에서 평가됨, 결과 transcribe 만 보고서에 들어옴).

보고서 §3 의 ckpt-12k 컬럼 snapshot 에는 일부 metric 이 빠져 있음 → 이 항목들 채우려면 raw 데이터 필요.

---

## 요청

`Qwen3.5AE-Stage2-whisper-small-emoFull-asr033-env05-txt03/` Stage-2 평가 결과 중 **ckpt-12000** 의 다음 파일들 내용 공유 부탁:

### 1) AudioSet (보고서 snapshot 컬럼에 ckpt-12k 값 누락)

```
eval_audioset/checkpoint-12000/summary.json
```
필요 키: `f1_micro`, `f1_macro` (greedy 모드)

```
eval_audioset_seq/checkpoint-12000/summary.json
```
필요 키: `mAP_micro`, `mAP_macro` (sequence-mode mAP). **이 ckpt 에서 sequence-mode 가 평가 안 됐다면 그것도 알려주세요** — `--` 처리.

### 2) Text retention (보고서엔 mean 만, per-benchmark 누락)

```
eval_text_retention/checkpoint-12000/summary.json
```
필요 키: `per_benchmark` 의 6 벤치마크 각 `accuracy`:
- `hellaswag.accuracy`
- `winogrande.accuracy`
- `boolq.accuracy`
- `arc_easy.accuracy`
- `arc_challenge.accuracy`
- `copa.accuracy`

(평균 0.8979 는 본 노드에 이미 있음.)

---

## 대안

위 3 파일 (`eval_audioset/`, `eval_audioset_seq/`, `eval_text_retention/`) 의 ckpt-12000 `summary.json` 을 **그대로 cat 결과** 공유해주셔도 됩니다.

**ckpt-12000 이 그 노드에 더 이상 없으면**: ckpt-15000 (보고서 권장 balanced 13-17k 중간) 의 같은 파일 3개로 대체 가능 — 그 경우 표의 representative ckpt 를 12k → 15k 로 옮기면 됨.

---

## 본 노드에 이미 있는 항목 (참고, 다시 보낼 필요 없음)

ckpt-12k snapshot 에서 다음은 본 노드 보고서 §3 표에 transcribe 되어 있어 채워둠:
- LibriSpeech: WER/CER × test-clean/test-other (4 cells)
- Emotion: MELD/DailyTalk/EmoV/RAVDESS × acc/macro-F1 (8 cells)
- ESC-50 acc
- FSD50K: F1-mi/F1-ma/Jaccard/mAP-mi/mAP-ma (5 cells)
- Clotho: BLEU-1/BLEU-4 (2 cells)
- LISTEN-MCQA: acc/macro-F1
- LISTEN-official: F1-mean/WA-mean
- Text retention 6-bench mean (0.8979)

총 채워진 cell: 22, 결손 cell: **9** (위 §1 요청).

---

## 응답 형식 예시

```
ckpt-12000 추가:
- AudioSet greedy f1_micro=0.xxx, f1_macro=0.xxx
- AudioSet seq mAP_micro=0.xxx, mAP_macro=0.xxx (또는 "not evaluated at this ckpt")
- text_retention per_benchmark:
  - hellaswag: 0.xxx
  - winogrande: 0.xxx
  - boolq: 0.xxx
  - arc_easy: 0.xxx
  - arc_challenge: 0.xxx
  - copa: 0.xxx
```

또는 raw json 그대로 OK.
