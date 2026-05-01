# Whisper-small ckpt-12000 LISTEN-official per-experiment 요청 (B 노드)

> **Status: PENDING — 미이행 (2026-04-30 audit 기준)**. 요청 산출물 (`stage1/whisper/ckpt12k/listen_official_summary*.json`) 이 본 노드에 도착 안 함. paper LISTEN breakdown 표 WS 행은 여전히 \nodata.

작성: 2026-04-30  
용도: paper 의 LISTEN breakdown 표 ([`tbl/listen_breakdown.tex`](/mnt/ddn/users/jos/AudioEnc/log/tmp/latex_work/tbl/listen_breakdown.tex)) 의 Whisper-small 행 (현재 모두 \nodata) 채우기. LISTEN paper Table 2 형식 따라가는 중.

---

## 배경

본 노드는 v2-15k + Whisper-tiny-20k 의 LISTEN-official 9 experiments (1_text, 1_audio, 1_audio_and_text, 2A, 2B, 2C, 3A, 3B, 3C) + type 4 (paralinguistic, 975 samples) 까지 평가 중. WS-12k 는 raw eval 결과가 본 노드에 없어서 표가 비어있음.

이전 cross-node 공유 (`stage1/whisper/ckpt12k/{eval_audioset,...}_summary.json`) 에는 LISTEN-official 이 빠져 있음.

---

## 요청

`/mnt/tmp/results/Qwen3.5AE-Stage2-whisper-small-emoFull-asr033-env05-txt03/eval_listen_official/checkpoint-12000/summary.json` 파일 cat 결과를 다음 위치에 저장:

```
/mnt/ddn/users/jos/audiollm-trainer/docs/stage1/whisper/ckpt12k/listen_official_summary.json
```

또는 (간단):

```bash
cp /mnt/tmp/results/Qwen3.5AE-Stage2-whisper-small-emoFull-asr033-env05-txt03/eval_listen_official/checkpoint-12000/summary.json \
   /mnt/ddn/users/jos/audiollm-trainer/docs/stage1/whisper/ckpt12k/listen_official_summary.json
```

### 필요한 키

`per_experiment` dict 안의 각 experiment 별:
- `weighted_accuracy`
- `macro_f1`
- `micro_f1`
- `chance_baseline.expected_accuracy`
- (전체 dict 그대로 보내주셔도 됨, 더 안전)

### 가능하면 추가로 — type 4 (paralinguistic) 도

본 노드는 module 패치해서 type 4 평가 가능하게 했음 ([`evaluation/stage2/eval_listen_official.py`](../../../evaluation/stage2/eval_listen_official.py) line 89, `EXPERIMENTS["4"] = ("audio", "4")` 추가). B 노드도 같은 패치 적용 후 type 4 만 추가 평가 가능:

```bash
# (B 노드 환경에 맞춰 PY 경로 등 조정)
PY=/path/to/python  
WS=/mnt/tmp/results/Qwen3.5AE-Stage2-whisper-small-emoFull-asr033-env05-txt03
PARQUET=/mnt/tmp/listen_analysis/data/test_with_type4.parquet  # 이 파일도 본 노드에서 만들어둠

# Combined parquet (test + type 4 from train2) 안 만들었다면 한 번 만드는 코드:
# python -c "
# import pyarrow.parquet as pq
# import pyarrow as pa
# test = pq.read_table('/mnt/tmp/listen_analysis/data/test-00000-of-00001.parquet')
# train2 = pq.read_table('/mnt/tmp/listen_analysis/data/train-00002-of-00003.parquet')
# mask = pa.compute.equal(train2.column('experiment_type'), pa.scalar('4'))
# type4 = train2.filter(mask)
# pq.write_table(pa.concat_tables([test, type4]), '/mnt/tmp/listen_analysis/data/test_with_type4.parquet')
# "

$PY -u -m evaluation.stage2.eval_listen_official \
    --ckpt-root $WS --out-root $WS/eval_listen_official \
    --base-model <ws_stage1_base_path> \
    --ckpts 12000 --parquet $PARQUET \
    --experiments 4 --batch-size 4
```

위 실행 후 새로 만들어진 `summary_4.json` (type 4 만 별도 파일로 저장됨) 도 같은 디렉터리에 함께 share:
```
/mnt/ddn/users/jos/audiollm-trainer/docs/stage1/whisper/ckpt12k/listen_official_summary_4.json
```

또는 module 다시 돌려서 전체 10 experiments 의 새 summary.json 으로 덮어쓰는 것도 가능.

---

## Caveat: type 4 contamination

LISTEN-train (Stage-2 emotion 학습 pool) 에 type 4 (975 samples) 가 포함되어 있어서 — type 4 평가는 training-data-contaminated upper bound. paper 에는 footnote 로 caveat 명시 예정. WS 도 같은 모델 (Stage-2 학습 데이터 동일) 이라 같은 caveat 적용됨.

---

## 응답 형식

JSON 파일 그대로 보내주시면 됨. 어떤 키가 들어있는지 확인 후 본 노드의 paper 표 자동 업데이트 진행.
