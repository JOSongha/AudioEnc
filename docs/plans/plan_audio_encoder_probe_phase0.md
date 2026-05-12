# Plan — Audio-encoder Linear Probe (Phase 0)

> **목적**: ALM의 audio encoder가 acoustic task에 대해 가지고 있는 정보량 측정. Projector/LLM 통과 X. 가설: reconstruction-pretrained DAC-VAE가 ASR-pretrained Whisper보다 emotion / sound classification에서 우월.

---

## Encoder 비교 대상 (Phase 0)

| Encoder | 출처 | 차원 | Frame rate | Sample rate |
|---|---|---|---|---|
| Whisper-small.en | `openai/whisper-small.en`의 `.encoder` (Stage1 frozen 동일) | 768 | 50 fps | 16 kHz |
| DAC-VAE | Stage1 ckpt `Qwen3.5AE-4B-ASR-Stage1`의 `model.model.audio_encoder.encoder` (frozen 동일) | 128 (z_e) | 25 fps | 48 kHz |

---

## Datasets (Phase 0)

| Type | Dataset | 클래스 | 규모 | Split protocol | 출처 |
|---|---|---|---|---|---|
| Emotion | **IEMOCAP** (4-class) | angry / happy(=happy+excited) / sad / neutral | ~5,500 utt | 5-fold leave-session-out | `/mnt/ddn/kyudan/IEMOCAP/data/*.wav` + `external/datasets/emotion/iemocap_shard_*.jsonl` (라벨) |
| Sound | **ESC-50** | 50 classes | 2,000 clips | Official 5-fold CV | 다운로드 from GitHub (`karolpiczak/ESC-50`) |

> Phase 1 후속: RAVDESS, CREMA-D, MELD, FSD50K로 확장.

---

## 산출물

```
experiments/audio_encoder_probe/
├── manifests/
│   ├── iemocap_4class.csv     # utt_id, audio_path, emotion, session, gender
│   └── esc50.csv              # utt_id, audio_path, label, fold
├── {encoder}/{dataset}/{utt_id}.npy   # mean-pooled embedding, float32
└── _results/
    ├── probe_results.csv       # encoder × dataset × fold × C × metric
    └── figs/                   # bar charts
docs/audio_encoder_probe_phase0.md  # 리포트
```

---

## 단계별 작업

### Phase 0.1 — Dataset prep
1. **IEMOCAP manifest** (`scripts/build_iemocap_manifest.py`)
   - jsonl 3 shard 합치기 → utt_id, response 추출
   - audio_path remap: `/mnt/ddn/kyudan/IEMOCAP/data/<basename>`
   - 4-class 필터: {angry, happy + excited→happy, sad, neutral}
   - session_id from utt_id prefix (Ses01..Ses05)
   - gender from utt_id (Ses01F vs Ses01M)
2. **ESC-50 download** (re-use `scripts/env_sound/download.sh`)
   - `meta/esc50.csv` 그대로 사용 (filename, fold, target, category)

### Phase 0.2 — Embedding extraction
**파일**: `experiments/audio_encoder_probe/extract_encoder_only.py`

```python
def extract(encoder_type, manifest_csv, out_dir):
    if encoder_type == "whisper_small":
        # WhisperModel.from_pretrained("openai/whisper-small.en").encoder
        # mel = WhisperFeatureExtractor (16kHz, padded to 30s)
        # h = encoder(mel).last_hidden_state  # (1, 1500, 768)
        # valid_frames = ceil(num_samples / 320), capped at 1500
        # mean = h[:, :valid_frames, :].mean(dim=1)   # (1, 768)
    elif encoder_type == "dacvae":
        # Stage1 ckpt → model.model.audio_encoder.encoder (DACVAE)
        # wav resampled to 48kHz
        # z, _ = encoder.encode(wav.unsqueeze(1))   # (1, 128, T)
        # mean = z.mean(dim=-1)                      # (1, 128)
```

- Output: `out_dir/{utt_id}.npy` (float32, 1D vector)
- DataLoader prefetch + batch processing
- Resume: skip existing `.npy`

### Phase 0.3 — Linear probe
**파일**: `experiments/audio_encoder_probe/probe.py`

- sklearn LogisticRegression with L2
- C grid: {0.01, 0.1, 1, 10} — pick best by inner CV on train
- IEMOCAP: 5-fold leave-session-out (Ses01..Ses05를 fold로)
- ESC-50: official 5-fold CV
- Metrics: accuracy, macro-F1, weighted-F1
- Output: `_results/probe_results.csv` (columns: encoder, dataset, fold, C, accuracy, macro_f1, weighted_f1)

### Phase 0.4 — 결과 시각화
**파일**: `experiments/audio_encoder_probe/plot_results.py`

- bar chart: encoder × dataset × accuracy (5-fold mean ± std)
- IEMOCAP per-class accuracy bar chart
- 저장: `_results/figs/`

### Phase 0.5 — 리포트
**파일**: `docs/audio_encoder_probe_phase0.md`

- Setup, dataset, encoder, results table, figure
- Findings: 객관적 관찰만, 결론은 사용자 추가

---

## 주의사항

- ❌ 학습/fine-tuning 금지 — encoder는 frozen, linear probe만
- ⚠️ Whisper의 30s padding은 valid frames mask로 처리. 30s 초과 utt는 truncate (warn).
- ⚠️ DAC-VAE는 48kHz, IEMOCAP은 16kHz wav → resample 필요
- ⚠️ IEMOCAP의 `excited` → `happy` 합치는 4-class는 표준 컨벤션 (Yoon 2018)
- ⚠️ Speaker-independent split 강제 (IEMOCAP은 session-independent로 자연스럽게 만족)

---

## 완료 조건

- [ ] manifests/iemocap_4class.csv 생성 (~5,500 rows)
- [ ] ESC-50 다운로드 + manifests/esc50.csv (2,000 rows)
- [ ] Whisper-small × IEMOCAP, ESC-50 임베딩 (총 ~7,500 .npy)
- [ ] DAC-VAE × IEMOCAP, ESC-50 임베딩 (총 ~7,500 .npy)
- [ ] probe_results.csv (2 encoder × 2 dataset × 5 fold × C grid)
- [ ] figs 생성
- [ ] 리포트 작성
