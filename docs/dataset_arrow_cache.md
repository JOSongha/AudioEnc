# Dataset Arrow 캐시 현황

작성: 2026-04-06

## 목표

학습에 사용하는 모든 데이터셋을 HuggingFace Arrow 형식으로 캐시해  
파일 탐색 오버헤드 없이 mmap 기반 빠른 데이터 로딩을 가능하게 한다.

---

## Arrow 형식이란

HuggingFace `datasets`가 내부적으로 사용하는 Apache Arrow IPC 파일.  
- **메모리 맵(mmap)**: 파일 전체를 RAM에 올리지 않고 필요한 청크만 접근
- **컬럼형 저장**: `audio`, `transcript`를 별도 컬럼으로 분리 → 텍스트만 읽을 때 오디오 I/O 없음
- **표준 포맷**: `dataset.load_from_disk()` / `load_dataset(cache_dir=...)` 로 바로 사용

---

## 데이터셋별 현황

### MLS (Multilingual LibriSpeech English)
| 항목 | 값 |
|------|-----|
| 소스 | `parler-tts/mls_eng_10k` |
| 상태 | **완료** |
| Arrow 경로 | `/mnt/tmp/cache/parler-tts___mls_eng_10k/default/0.0.0/<hash>/` |
| 총 크기 | 149GB |
| shard 수 | train 315개 + dev 1개 + test 1개 |

Arrow 스키마:
```
audio:          struct<bytes: binary, path: string>   # mp3 bytes 그대로 저장
transcript:     string
audio_duration: double
speaker_id:     string
book_id:        string
begin_time:     double
end_time:       double
```

---

### GigaSpeech XL (10,000h)
| 항목 | 값 |
|------|-----|
| 소스 | `speechcolab/gigaspeech`, subset `xl` |
| 상태 | **빌딩 중** (Arrow 변환 재시작 2026-04-06) |
| Parquet shard | 232/258 다운로드 완료 (HF hub blob) |
| 총 크기 | ~960GB (blobs) |
| 비고 | 2026-03-29 다운로드 시작, Arrow 빌딩 단계에서 8일 멈춤 → 프로세스 재시작 |

이전 실패 원인: `num_proc=8` 멀티프로세스 Arrow 빌딩 중 deadlock 추정.  
재시작: `/tmp/download_gigaspeech2.py`, 로그 → `/mnt/tmp/cache/gigaspeech_download2.log`

---

### VoxPopuli (English)
| 항목 | 값 |
|------|-----|
| 소스 | `facebook/voxpopuli`, lang `en` |
| 상태 | **다운로드 재시작** (2026-04-06) |
| 이전 상태 | 3.1GB blob 다운로드 후 중단 (`.incomplete`) |
| 비고 | 로그 → `/mnt/tmp/cache/voxpopuli_download.log` |

---

### LibriSpeech
| 항목 | 값 |
|------|-----|
| 소스 | 직접 다운로드 (torchaudio) |
| 상태 | raw flac + .trans.txt |
| 크기 | train-clean-100: 6.3GB / train-clean-360: 23GB / train-other-500: 30GB / dev-clean: 350MB |
| 파일 수 | 283,944개 flac |
| Arrow 변환 | 미실시 (flac 직접 읽기로 충분, word alignment 파이프라인과 분리) |

Word alignment JSON은 별도로 생성 중 (`/mnt/tmp/cache/word_alignments/`).  
완료 후 `merge_alignments.py`로 split별 JSONL로 병합.

---

## 캐시 디렉토리 구조

```
/mnt/tmp/cache/
  LibriSpeech/                     # raw flac
    train-clean-100/
    train-clean-360/
    train-other-500/
    dev-clean/
  parler-tts___mls_eng_10k/        # Arrow 완료
    default/0.0.0/<hash>/*.arrow
  speechcolab___gigaspeech/        # Arrow 빌딩 중
    xl/0.0.0/
  facebook___voxpopuli/            # 다운로드 중
    en/0.0.0/
  word_alignments/                 # LibriSpeech word-level alignment
    manifest.jsonl
    shard_0~63.jsonl
    train-clean-100/{speaker}/{chapter}/{utt}.json
    train-clean-360/...
    train-other-500/...
    dev-clean/...
  hf/
    hub/
      datasets--speechcolab--gigaspeech/blobs/  # 960GB parquet blobs
      datasets--facebook--voxpopuli/blobs/       # 다운로드 중
      datasets--parler-tts--mls_eng_10k/         # parquet (MLS hub 원본)
```

---

## Arrow 변환 완료 후 사용 방법

`dataset.py`에서 `load_dataset` 호출 시 `cache_dir` 일치하면 자동으로 캐시 사용:

```python
from datasets import load_dataset

# MLS — 이미 사용 가능
ds_mls = load_dataset(
    "parler-tts/mls_eng_10k",
    split="train",
    cache_dir="/mnt/tmp/cache",
)

# GigaSpeech — Arrow 빌딩 완료 후
ds_gs = load_dataset(
    "speechcolab/gigaspeech",
    "xl",
    split="train",
    cache_dir="/mnt/tmp/cache",
)

# VoxPopuli — 다운로드 완료 후
ds_vp = load_dataset(
    "facebook/voxpopuli",
    "en",
    split="train",
    cache_dir="/mnt/tmp/cache",
)
```

오디오 bytes 접근 (decode 없이):
```python
from datasets import Audio
ds = ds.cast_column("audio", Audio(decode=False))
sample = ds[0]
audio_bytes = sample["audio"]["bytes"]   # flac/opus/mp3 bytes
```
