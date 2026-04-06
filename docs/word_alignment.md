# LibriSpeech Word-Level Alignment

작성: 2026-04-06

## 목표

LibriSpeech 전체 (train-clean-100 / train-clean-360 / train-other-500 / dev-clean) 각 발화에 대해  
**word-level timestamp** (start, end, score) 를 사전 생성하여 JSON으로 저장.

---

## 도구 선택

| 방법 | 품질 | 비고 |
|------|------|------|
| MFA (Montreal Forced Aligner) | ★★★★★ | conda 별도 환경, 음향모델 학습 필요 → 제외 |
| **wav2vec2-large CTC forced alignment** | ★★★★☆ | **채택** |
| wav2vec2-base CTC forced alignment | ★★★☆☆ | WhisperX 기본값 (WAV2VEC2_ASR_BASE_960H) |
| Whisper large-v3 `word_timestamps=True` | ★★☆☆☆ | attention 기반, 경계 정밀도 낮음 |

### 채택 이유

- LibriSpeech는 ground truth transcript가 이미 존재 → Whisper 재전사 불필요
- wav2vec2-large-960h-lv60-self: 60,000h LV-60 + LS-960 학습, base 대비 frame-level 정밀도 2배 향상
- 경계 오차: ~10ms (base는 ~20ms)
- VRAM: ~1.2GB per GPU (A100 80GB 기준 문제없음)
- `whisperx.align()` + `model_name` 인자로 large 모델 지정 가능 — 추가 설치 불필요

### 사용 모델

```
WAV2VEC2_ASR_LARGE_LV60K_960H
```

torchaudio 번들 이름. WhisperX `load_align_model(model_name=...)` 에 직접 전달.  
LV-60k (60,000h LibriVox) + 960h LibriSpeech fine-tuned. HF 모델 로드 없이 torchaudio가 직접 처리 → transformers torch.load CVE 차단 우회.

---

## 파이프라인 구조

```
alignment/
  build_manifest.py      # LibriSpeech 전체 발화 목록 → manifest.jsonl + shard_N.jsonl
  align_worker.py        # 단일 GPU 워커 (shard 하나 처리, ThreadPoolExecutor prefetch)
  run_align.sh           # 다중 GPU × 다중 워커 런처
  merge_alignments.py    # 완료 후 per-utt JSON → per-split JSONL + 검증
```

### 출력 경로

```
/mnt/tmp/cache/word_alignments/
  manifest.jsonl
  shard_{0..7}.jsonl
  train-clean-100/{speaker}/{chapter}/{utt_id}.json
  train-clean-360/...
  train-other-500/...
  dev-clean/...
```

---

## 출력 JSON 스키마

```json
{
  "utterance_id": "19-198-0000",
  "split": "train-clean-100",
  "audio_duration": 4.32,
  "transcript": "he hoped there would be stew for dinner",
  "words": [
    {"word": "he",    "start": 0.18, "end": 0.30, "score": 0.991},
    {"word": "hoped", "start": 0.34, "end": 0.62, "score": 0.987}
  ],
  "alignment_model": "facebook/wav2vec2-large-960h-lv60-self",
  "processed_at": "2026-04-06T..."
}
```

`score`: CTC log-prob 기반 신뢰도. 임계값(0.5) 미만 발화는 학습 시 필터링 가능.

---

## 병렬화 전략

- **8 GPU × 8 workers/GPU = 64 프로세스** (WORKERS_PER_GPU 파라미터로 조정)
- GPU 당 wav2vec2-large VRAM: ~3.2GB × 8 = ~25GB (A100 80GB의 30%)
- ThreadPoolExecutor(max_workers=4)로 flac I/O prefetch → GPU 연산과 I/O 오버랩
- GPU 간 통신 없음 (alignment는 utterance 단위로 독립)
- 출력 파일이 이미 존재하면 스킵 → 재시작 내성
- rank 0이 먼저 모델 로드, 나머지는 `min(rank*2, 30)`초 stagger (GPU 메모리 할당 충돌 방지)

### GPU 활용률 참고

실측 utilization: 5~37% (낮은 이유)
1. 여러 프로세스가 CUDA MPS 없이 GPU 공유 → time-slicing 직렬화
2. CTC forced alignment (`torchaudio.functional.forced_align`)가 CPU 연산
3. wav2vec2 forward는 짧고 빠름 → GPU idle 구간 큼

개선 방안: `nvidia-smi -c EXCLUSIVE_PROCESS + nvidia-cuda-mps-control -d` (root 권한 필요)

---

## 발화 수 및 처리 시간

| Split | 발화 수 | 평균 길이 |
|-------|--------|---------|
| train-clean-100 | 28,539 | ~14s |
| train-clean-360 | 104,014 | ~14s |
| train-other-500 | 148,688 | ~14s |
| dev-clean | 2,703 | ~14s |
| **합계** | **283,944** | |

- 64 워커 (8/GPU): shard당 4,437개
- 실측 처리속도: ~62 utt/s (전체 기준)
- **실제 소요: 약 75~90분**

### merge 및 검증

완료 후 `merge_alignments.py`로 per-utt JSON → per-split JSONL 병합:
```bash
python3 merge_alignments.py \
    --alignment-dir /mnt/tmp/cache/word_alignments \
    --manifest      /mnt/tmp/cache/word_alignments/manifest.jsonl \
    --output-dir    /mnt/tmp/cache/word_alignments
```

출력:
- `train-clean-100.jsonl`, `train-clean-360.jsonl`, `train-other-500.jsonl`, `dev-clean.jsonl`
- `merge_report.json`: split별 ok/missing/invalid/low_score 통계, coverage %

검증 기준:
- 필수 필드 누락 → `invalid`
- word `end < start` → `invalid`
- word 평균 score < 0.5 → `low_score` (JSONL에 포함, 학습 시 필터링)
- 출력 JSON 없음 → `missing` (워커 재실행 필요)

---

## LibriSpeechDataset 통합 방안 (향후)

```python
class LibriSpeechDataset(Dataset):
    def __init__(self, ..., alignment_dir=None):
        self.alignment_dir = alignment_dir

    def __getitem__(self, idx):
        waveform, sr, transcript, speaker_id, chapter_id, utt_id = self.dataset[idx]
        # ... 기존 전처리 ...
        if self.alignment_dir:
            json_path = f"{self.alignment_dir}/{self.url}/{speaker_id}/{chapter_id}/{utt_id}.json"
            alignment = json.load(open(json_path))
            return waveform_processed, transcript.lower(), alignment["words"]
        return waveform_processed, transcript.lower()
```

### 활용 방안

- **데이터 품질 필터링**: score 평균 < 0.5 발화 제외
- **세그먼트 단위 학습**: 단어/구 경계로 오디오를 자르는 augmentation
- **보조 alignment loss**: projector frame ↔ word timestamp 대응 손실 (장기)
