# §4.3 PCA / Distance 분석 요청 (B 노드)

작성: 2026-04-30
용도: paper sec/04_experimenet.tex §4.3 *"How Is Acoustic Variability Encoded under Fixed Semantics?"* 의 분석/figure 생성.

## 분석 질문

**같은 transcript, 다른 화자** 의 audio 가 각 encoder/LM 단계에서 어떻게 임베딩되는가?
- 화자 정보가 보존되나 (acoustic encoder 가설)? 아니면 sub-word 수준에서 합쳐지나 (semantic encoder 가설)?
- encoder 단 → LM 단 으로 갈수록 화자 분산이 어떻게 변하나?

## 비교 대상

3 encoder 모두 본 노드 + B 노드에 분산 학습됨:
- **DAC-VAE v2** (acoustic, 본 노드 ckpt-15k 권장): `/mnt/tmp/results/Qwen3.5AE-Stage2v2-emoFull-asr033-env05-txt03/checkpoint-15000`
- **Whisper-tiny** (semantic small, 본 노드 ckpt-6k 권장): `/mnt/tmp/results/Qwen3.5AE-Stage2-whisper-tiny-emoFull-asr033-env05-txt03/checkpoint-6000`
- **Whisper-small** (semantic large, B 노드 ckpt-8k 권장 — best LS-clean): `/mnt/tmp/results/Qwen3.5AE-Stage2-whisper-small-emoFull-asr033-env05-txt03/checkpoint-8000`

Stage-1 base checkpoints, projector, LM은 각자 환경에 맞춰 load (paper §3 architecture 동일).

## 데이터셋 후보

같은 transcript 다양 화자가 필요. 권장 (가용성 순):
1. **VCTK** (110 화자, parallel reading scripts) — 화자별 같은 문장 다수
2. **LibriTTS-R speaker subset** — 같은 책 공유하는 화자 그룹
3. **Custom prompt** — 사용자 직접 선정 10–20 문장 × 8–10 화자

전처리: 16 kHz mono, ≤ 10 sec, normalize.

## 분석 단계

### 1) Embedding 추출

각 encoder × 각 (transcript, speaker) 페어에 대해:
- **encoder output** (raw acoustic latent, projector 입력 직전)
- **projector output** (LM embedding-space, audio_pad 위치)
- **LM hidden states** — layer 0/4/8/12/16/20/24/28 (Qwen3.5-4B 32 layer 중 sparse)

shape: `[N_speaker × N_transcript, T_token, D]` (encoder: D = 128/384/768; projector: D = LM hidden; LM: D = 2560 — Qwen3.5)

저장: `/mnt/ddn/users/jos/AudioEnc/log/sec43_embeddings/{encoder}_{stage}.npy`

### 2) PCA 투영

각 stage 별 embedding 평균 (token dim T pooling) → PCA 2D 투영. transcript-color, speaker-marker 로 scatter.

### 3) Distance 분석

- **within-speaker distance**: 같은 화자, 다른 transcript pair 평균 cosine distance
- **across-speaker distance**: 다른 화자, 같은 transcript pair 평균 cosine distance
- **ratio = across/within**: 화자 정보 보존 척도. 1에 가까우면 transcript 가 dominant, > 1 이면 화자 정보 분리됨.

각 encoder × 각 stage 별 ratio 계산.

### 4) Figure 산출

- `figures/sec43_pca_scatter.pdf` — 3 row (encoder) × 4 col (stage) 의 PCA scatter grid
- `figures/sec43_distance_ratio.pdf` — 3 line (encoder) vs LM layer depth 의 across/within ratio
- `figures/sec43_full.pdf` — 위 둘 합친 main figure

## 출력 위치

- raw embedding npys: `/mnt/ddn/users/jos/AudioEnc/log/sec43_embeddings/`
- figures: `/mnt/ddn/users/jos/AudioEnc/log/tmp/latex_work/figures/sec43_*.pdf`
- aggregate report (markdown): `/mnt/ddn/users/jos/audiollm-trainer/docs/stage2/requests/pca_distance_results.md`
- WS 부분 raw 데이터 본 노드 plot 코드에 쓸 수 있게 cross-node share: `/mnt/ddn/users/jos/audiollm-trainer/docs/stage1/whisper/ckpt12k/sec43_ws_embeddings.npz`

## 본 노드와의 분담

- **B 노드**: §4.3 전체 (PCA + distance) — embedding 추출 + 분석 + figure (3 encoder 모두). B 노드의 GPU 자원 활용.
- **본 노드**: §4.2 CKA 분석 (encoder × projector × LM layer 간 representational similarity) — 별개 figure. 본 노드 idle GPU 활용 (GPU 0/1/2/5 등).

두 분석은 데이터 의존성 없이 독립 실행 가능. 결과 figure 만 paper LaTeX 에 합쳐짐.

CKA / PCA / distance 분석이 서로 보완적이라 cross-validate 됨.

## 기한

이번 주 (~5월 5일) 까지 figure draft 받으면 paper 통합 가능.

## 응답

raw embedding 또는 figure 둘 다 보내거나, 진행 막힐 때 (모델 load / 데이터셋 없음 등) 알려주세요.
