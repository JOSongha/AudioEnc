# eval_ckpts — 체크포인트별 dev-set WER 비교

한 run 디렉토리 안의 모든 stage2 체크포인트에 대해 LibriSpeech dev set 으로
WER을 측정하고, ref ↔ hyp 를 word-level diff 로 비교할 수 있는 self-contained
HTML 뷰어를 생성한다.

## Layout

```
eval_ckpts/
  eval_all_ckpts.py   ← 각 ckpt 평가 → JSON 저장
  build_viewer.py     ← JSON → 단일 HTML 뷰어
  results/
    <run_tag>/
      <split>/
        summary.json
        checkpoint_fe1_split1.json
        ...
        index.html
```

## 사용

```bash
cd /mnt/fr20tb/wbl_residency/jos/AudioEnc

# 1) 평가 (학습 중이면 한 GPU 만, 표본 수 제한 권장)
/mnt/ddn/users/jos/miniforge3/envs/audio/bin/python eval_ckpts/eval_all_ckpts.py \
    --run-dir /mnt/tmp/cache/hf/fb_dacvae/s2_outputs_0414_1442 \
    --split dev-clean --max-samples 200 --gpu 0

# 2) HTML 뷰어 생성
/mnt/ddn/users/jos/miniforge3/envs/audio/bin/python eval_ckpts/build_viewer.py \
    eval_ckpts/results/s2_outputs_0414_1442/dev-clean/

# 3) 브라우저로 열기
xdg-open eval_ckpts/results/s2_outputs_0414_1442/dev-clean/index.html
```

## 옵션

`eval_all_ckpts.py`:
- `--split`: dev-clean | dev-other | test-clean | test-other
- `--max-samples N`: 평가 표본 수 (0 = 전체). 학습과 동시에 돌릴 땐 100~300 권장.
- `--gpu N`: 한 GPU 만 점유. 학습이 8 GPU 점유 중이라도 각 GPU 에 50GB 가량 여유.
- `--only PATTERN`: 특정 ckpt 만 평가 (substring match). 예: `--only fe2`
- `--seed N`: 표본 셔플 seed (0 = 앞에서부터)
- `--beam-size N`: greedy=1, beam=4 등

`build_viewer.py`:
- 인자: 결과 디렉토리 (summary.json 포함). `index.html` 을 같은 위치에 생성.

## 뷰어 기능

- 좌측: ckpt 목록 + WER. 최저(녹) / 최고(빨) 강조.
- 우측: 선택된 ckpt 의 ref/hyp pair 목록.
  - **word-level diff 색상**: match / substitution / deletion / insertion
  - 검색창: ref 또는 hyp 텍스트 부분 일치
  - 필터: errors only / perfect only / all
  - 정렬: index 순서 / 오답 많은 순
- 단일 HTML 파일이라 서버 없이 브라우저에서 열기만 하면 됨.

## 학습 동시 실행 안전성

- inference 는 별도 process. nccl 통신 영향 없음.
- 8 GPU 모두 학습이 점유 중이지만 GPU 당 ~50 GB 여유. eval 은 한 GPU 에 6~10 GB 정도 사용.
- 학습 GPU util 100% 인 시점엔 eval throughput 이 떨어질 수 있으므로 `--max-samples 200`
  정도로 짧게 끊어 도는 것을 권장.
