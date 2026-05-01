# LAION-Audio-630K 다운로드 가이드

원본: <https://github.com/LAION-AI/audio-dataset/blob/main/laion-audio-630k/README.md>
HuggingFace 미러: <https://huggingface.co/datasets/Meranti/CLAP_freesound>

633 526 audio-text pairs / 4 325 시간. 8개 sub-source 중 **4개만 공개**, 나머지 4개(Free To Use Sounds / Sonniss / WeSoundEffects / Paramount)는 LAION이 비공개로 구매했음.

---

## 1. Sub-source 요약

| Source | 샘플 | 시간 | 다운 가능 | 채널 |
|---|---:|---:|---|---|
| **Freesound (full)** | 515 581 | 3 033 h | ✅ | HF Webdataset (`Meranti/CLAP_freesound`) |
| **Freesound (no-overlap)** | 460 801 | 2 817 h | ✅ | 같은 repo (ESC-50 / FSD50K test 항목 제외) |
| **BBC Sound Effects** | 15 973 | 463 h | ✅ | HF 미러 `marianna13/BBCSoundEffects` (166 GB, 66 tar). LAION csv는 사라졌고 official URL은 BBC Rewind SPA로 redirect되어 직접 다운 불가. §3.1 |
| **Epidemic Sound** | 75 645 | 220 h | ✅ | HF 미러 `CLAPv2/epidemic_sound_effects_t5_debiased` (72 GB, audio 임베드 parquet). 원본 cloudfront URL은 unreachable이지만 미러로 우회 가능. §3.2 |
| **Audiostock** | 10 000 | 453 h | ⚠️ | csv 작동, `audiostock.net` → `audiostock.jp` 도메인 치환 시 sample mp3 (~25 KB)만 가용. §3.3 |
| Free To Use Sounds | 6 370 | 175 h | ❌ | 비공개 (LAION 구매) |
| Sonniss Game Effects | 5 049 | 84 h | ❌ | 비공개 |
| We Sound Effects | 488 | 12 h | ❌ | 비공개 |
| Paramount Motion Sound | 4 420 | 19 h | ❌ | 비공개 |
| **합계** | **633 526** | **4 325 h** | | |

CC-라이선스 분포 (Freesound):
- CC0 260 134, CC-BY 4.0 97 090, CC-BY 3.0 89 337, CC-BY-NC 58 416, CC Sampling+ 일부

### 1.1 이 환경 가용 시간 합계

| 소스 | 시간 | 비고 |
|---|---:|---|
| Freesound (no-overlap) | 2 817 h | HF 미러로 그대로 회수 |
| Freesound (full) | 3 033 h | no-overlap 슈퍼셋 (둘 중 하나 선택) |
| BBC Sound Effects | 463 h | HF 미러 (`marianna13/BBCSoundEffects`) |
| Epidemic Sound | 220 h | CLAPv2 HF 미러 (audio 임베드 parquet) |
| Audiostock samples | ~4 h ⚠️ | 풀 트랙 453 h 중 sample preview (1.5 s × 10 k)만 가용 |
| Free To Use / Sonniss / WeSoundEffects / Paramount | 290 h | 비공개 (LAION 구매), 회수 불가 |

| 회수 시나리오 | 합계 |
|---|---:|
| Freesound **no-overlap** + BBC + Epidemic + Audiostock samples | **~3 500 h** (LAION 4 325 h의 81 %) |
| Freesound **full** + BBC + Epidemic + Audiostock samples | **~3 720 h** (86 %) |

손실 (비공개 4 source): 290 h. Audiostock은 풀 트랙 453 h 중 ~4 h만 sample 형태로 받히므로 실질 -449 h 누락.

---

## 2. Freesound 다운로드 (HF Webdataset, 권장 경로)

### 2.1 repo 구조 (1907 파일, **총 1.35 TB**)

```
freesound/
    train_1/   {0..459}.tar    317.7 GB   (460 shards)
    train_2/   {0..446}.tar    308.1 GB   (447 shards)
    test/      {0..101}.tar     68.9 GB   (102 shards)
    sizes.json
freesound_no_overlap/         ←  ESC-50 / FSD50K test 클립 제거 버전
    train_1/   {0..459}.tar    334.4 GB
    train_2/   {0..348}.tar    253.1 GB
    test/      {0..90}.tar      64.6 GB
    sizes.json
freesound_meta.csv             ~70 MB
freesound_no_overlap_meta.csv  ~70 MB
README.md
```

평균 tar 크기 **~700 MB**, Webdataset 포맷 (안에 `<key>.flac` + `<key>.json` 페어).

### 2.2 다운로드 명령

```bash
# 환경
pip install -U "huggingface_hub[cli]" hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1   # Rust 백엔드, 5-10× 빠름
export HF_HOME=/mnt/tmp/cache/huggingface

# (a) huggingface-cli — 부분 다운로드 가능
huggingface-cli download Meranti/CLAP_freesound \
    --repo-type dataset \
    --local-dir /mnt/tmp/datasets/laion_freesound \
    --include "freesound_no_overlap/*" "*.csv"   # no-overlap만 받기 (~650 GB)
```

```python
# (b) Python: snapshot_download
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Meranti/CLAP_freesound",
    repo_type="dataset",
    local_dir="/mnt/tmp/datasets/laion_freesound",
    allow_patterns=[
        "freesound_no_overlap/*",   # 또는 "freesound/*" (full)
        "*.csv",
        "README.md",
    ],
    max_workers=16,
)
```

### 2.3 다운로드 시간 (1 Gbps ≈ 100 MB/s 가정)

| 선택 | 사이즈 | 시간 |
|---|---:|---:|
| `freesound_no_overlap/*` (train_1+2+test) | **652 GB** | ~1.8 h |
| `freesound/*` (full) | 695 GB | ~1.9 h |
| 둘 다 | **1 347 GB** | ~3.7 h |

`HF_HUB_ENABLE_HF_TRANSFER=1` + `max_workers=16`이면 보통 link 한도까지 포화.

### 2.4 사용 (Webdataset)

```python
import webdataset as wds

url = "/mnt/tmp/datasets/laion_freesound/freesound_no_overlap/train_1/{0..459}.tar"
ds = (wds.WebDataset(url, shardshuffle=True)
        .decode("torchaudio")           # FLAC → (waveform, sr)
        .to_tuple("flac", "json")
        .map_tuple(lambda wav_sr: wav_sr[0],
                   lambda j: j["text"]))   # 캡션은 "text" 필드

for waveform, caption in ds:
    print(waveform.shape, caption)
    break
```

각 sample의 JSON 메타:
- `text` — 캡션 (보통 1-2개)
- `original_data` — Freesound API 응답 일부 (id, username, license, …)
- `tag` — 태그 리스트

### 2.5 메타 CSV 컬럼 (별도 파일)

`freesound_meta.csv` / `freesound_no_overlap_meta.csv`:
- `audio_filename` — 예 `2394.flac`
- `caption_1`, `caption_2` (있으면)
- `freesound_id`, `username`, `freesound_url`, `license`

---

## 3. CSV-only sources — 실제 가용 상태 (2026-04-30 점검)

| Source | LAION csv URL | 우회 경로 | 상태 |
|---|---|---|---|
| **BBC** | csv 제거됨 | HF mirror `marianna13/BBCSoundEffects` (166 GB, 66 tar webdataset) | ✅ 사용 가능 |
| **Epidemic** | GDrive csv 75k rows | csv의 `dkihjuum4jcjr.cloudfront.net` 호스트 | ❌ DNS 미해결 (이 네트워크에서) |
| **Audiostock** | GDrive csv 10k rows | csv의 `audiostock.net` URL은 404 → `audiostock.jp`로 패치하면 302 → 25 KB sample mp3 | ⚠️ sample만 가용 |

### 3.1 BBC — HF 미러 (권장)

LAION README의 공식 csv는 사라졌고 (`http://bbcsfx.acropolis.org.uk/assets/*.wav` → BBC Rewind SPA HTML로 redirect, scraper 무용지물), 다행히 커뮤니티 미러가 있음:

- **`marianna13/BBCSoundEffects`** (HF dataset, public)
- 166 GB / 66 tar shards (train 58 + test 8)
- Webdataset 포맷: `<id>.flac` + `<id>.json` (캡션 메타) 페어
- ~15 973 클립 (LAION 카운트와 일치 추정)

```bash
huggingface-cli download marianna13/BBCSoundEffects \
    --repo-type dataset \
    --local-dir /mnt/tmp/datasets/laion_bbc_hf
```

```python
# 또는 Python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="marianna13/BBCSoundEffects",
    repo_type="dataset",
    local_dir="/mnt/tmp/datasets/laion_bbc_hf",
    max_workers=16,
)
```

다운 시간 (1 Gbps 가정): 166 GB / 100 MB/s ≈ **28 min**.

tar 안 디렉토리 prefix가 길게 박혀 있음 (`mnt/audio_clip/processed_datasets/BBCSoundEffects/train/...`) — webdataset로 읽을 때 key는 그 끝에 붙은 `<id>` 부분.

### 3.2 Epidemic — `CLAPv2/...` HF 미러에 오디오 임베드 (권장)

처음에는 LAION csv / Chr0my parquet의 `dkihjuum4jcjr.cloudfront.net` URL이 이 노드에서 DNS 미해결이라 단념했지만, **CLAPv2 조직이 처리된 오디오 본체를 parquet에 임베드해서 HF에 미러링해 둠**. cloudfront 의존 없이 바로 받을 수 있음.

| 미러 | 사이즈 | rows | 비고 |
|---|---:|---:|---|
| `CLAPv2/epidemic_sound_effects` | **17 GB** / 611 parquets | ~18k | 처리 단계 일부 / 캡션 1개 |
| `CLAPv2/epidemic_sound_effects_t5_debiased` | **72 GB** / 2524 parquets | ~75k | full corpus + T5 debiased 캡션 (`text` 단일) |

각 parquet 스키마:
```
index: string                                  # mnt_epidemic_sound_effects_split_<split>_<id>
datasetname: string                            # 'epidemic_sound_effects'
audio: struct<bytes: binary, path: string>     # FLAC bytes (preprocessed, 48 kHz)
audio_len: float                               # seconds
text: string                                   # main caption (T5 debiased version)
raw_text: list<string>                         # ['Title: ...', 'Id: ...', 'Added: ...', ...]
```

```bash
huggingface-cli download CLAPv2/epidemic_sound_effects_t5_debiased \
    --repo-type dataset --local-dir /mnt/tmp/datasets/laion_epidemic_clapv2
```

**이전에 받은 메타 전용 미러 (`Chr0my/Epidemic_sounds` 45 parquets × 2 MB)는 더 이상 불필요** — 위 CLAPv2 미러가 메타+오디오 둘 다 포함.

원본 cloudfront URL 의존 경로:

| 출처 | 파일 | 상태 |
|---|---|---|
| LAION GDrive csv (`1vo0NslkCTJHI03FbBSHLRztP6v2XkYNW`, 20 MB, 75k rows) | `url` 컬럼 → `dkihjuum4jcjr.cloudfront.net` | ❌ 이 네트워크 unreachable |
| `Chr0my/Epidemic_sounds` (HF, 45 parquet × 2 MB, 메타만) | `url` 컬럼 → 동일 host | ❌ unreachable |

→ CLAPv2 미러 사용으로 우회.

### 3.3 Audiostock — `.jp` URL로 패치하면 sample mp3만 가용

LAION csv (GDrive `1FnOcrb6fREIDBzB2lknJnszVn-yNCPp6`, 836 KB) 안의 URL `https://audiostock.net/audio/<id>/play`는 모두 404. `audiostock.net` → `audiostock.jp`로 도메인 치환하면 `audiostock.jp/audio/<id>/play` → 302 → cloudfront `cf-audiostock-public-files.audiostock.jp/audio-sample-128kbps/<id>-<hash>.mp3`로 작동.

**주의**: 받히는 파일은 **sample preview** (~25 KB, 128 kbps) — 풀 트랙 아님. LAION이 처음부터 sample만 처리한 것으로 추정.

총 사이즈: 10 001 × ~25 KB ≈ **250 MB**. 다운 시간 (concurrency 8): 약 **20 분**.

```python
# /tmp/audiostock_dl.py — concurrency 8, .jp 패치, output: /mnt/tmp/datasets/laion_audiostock/audio/<id>.mp3
import csv, re, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CSV = "/mnt/tmp/datasets/laion_csvs/Audiostock.csv"
OUT = Path("/mnt/tmp/datasets/laion_audiostock/audio")
OUT.mkdir(parents=True, exist_ok=True)
ID_RE = re.compile(r"audiostock\.\w+/audio/(\d+)/play")

def fetch(row):
    m = ID_RE.search(row.get("url", ""))
    if not m: return ("?", "BAD_URL")
    aid = m.group(1)
    out = OUT / f"{aid}.mp3"
    if out.exists() and out.stat().st_size > 1024:
        return (aid, "skip")
    try:
        r = requests.get(f"https://audiostock.jp/audio/{aid}/play",
                         allow_redirects=True, timeout=30)
        if r.status_code != 200:
            return (aid, f"HTTP_{r.status_code}")
        out.write_bytes(r.content)
        return (aid, "ok")
    except Exception as e:
        return (aid, f"ERR:{type(e).__name__}")

rows = list(csv.DictReader(open(CSV)))
with ThreadPoolExecutor(max_workers=8) as ex:
    for i, f in enumerate(as_completed([ex.submit(fetch, r) for r in rows])):
        if i % 200 == 0: print(i, "/", len(rows), f.result())
```

### 3.4 GDrive csv 받는 방법 (Epidemic / Audiostock 공통)

```bash
pip install gdown
python -c "
import gdown
gdown.download('https://drive.google.com/uc?id=1vo0NslkCTJHI03FbBSHLRztP6v2XkYNW',
               '/mnt/tmp/datasets/laion_csvs/Epidemic_all_debiased.csv')
gdown.download('https://drive.google.com/uc?id=1FnOcrb6fREIDBzB2lknJnszVn-yNCPp6',
               '/mnt/tmp/datasets/laion_csvs/Audiostock.csv')
"
```

Epidemic csv는 헤더 형식이 깨져 있음 (첫 줄에 column 이름과 첫 row 데이터가 합쳐져 있음). pandas 기본 파서 실패 → `pd.read_csv(..., engine="python", on_bad_lines="skip")` 또는 직접 파싱 필요.
        time.sleep(0.05)   # backoff
```

---

## 4. 라이선스 / 사용 약관

원문에서 인용:
> By downloading audios through the links provided in the csv files, you agree that you will use the audios for research purposes only, unless you get the permission from owners of the Datasource.

- **연구 용도 한정** — csv 다운로드 시점에 자동 동의
- **Freesound 개별 클립**: 클립별 CC 라이선스 (위 분포 표 참조). Webdataset 안 JSON에 `original_data.license` 있음
- **상업 사용**: `frederic.font@upf.edu` 별도 문의

---

## 5. 권장 시작 절차 (이 클러스터 환경)

1. **Freesound no_overlap** (HF, ~650 GB, ~2 h)
   ```bash
   huggingface-cli download Meranti/CLAP_freesound \
       --repo-type dataset --local-dir /mnt/tmp/datasets/laion_freesound \
       --include "freesound_no_overlap/*" "*.csv"
   ```
2. **BBC SoundEffects** (HF mirror, ~166 GB, ~28 min) — §3.1
   ```bash
   huggingface-cli download marianna13/BBCSoundEffects \
       --repo-type dataset --local-dir /mnt/tmp/datasets/laion_bbc_hf
   ```
3. **Epidemic Sound** (CLAPv2 HF mirror with embedded audio, ~72 GB, ~12 min) — §3.2
   ```bash
   huggingface-cli download CLAPv2/epidemic_sound_effects_t5_debiased \
       --repo-type dataset --local-dir /mnt/tmp/datasets/laion_epidemic_clapv2
   ```
4. **Audiostock samples** (sample mp3, ~250 MB, ~20 min) — §3.3 스크립트
5. 메타 통합: `freesound_no_overlap_meta.csv` + 각 tar 내 `<id>.json` (Freesound, BBC) + parquet `text`/`raw_text` 컬럼 (Epidemic)
6. 학습 manifest에 흡수 (LICENSE 필드 필터링: CC0/CC-BY 사용 가능, CC-BY-NC는 비상업 한정)

총 가용 사이즈 (이 환경): **~888 GB** (Freesound no_overlap 650 G + BBC HF 166 G + Epidemic 72 G + Audiostock 0.25 G), 총 다운 ~3 h

---

## 6. 메타데이터 점검 sanity 코드

```python
import pandas as pd
df = pd.read_csv("/mnt/tmp/datasets/laion_freesound/freesound_no_overlap_meta.csv")
print(f"rows: {len(df):,}")
print(f"unique audio: {df['audio_filename'].nunique():,}")
print("\nlicense distribution:")
print(df["license"].value_counts().head(10))
print("\ncaption presence:")
for col in ("caption_1", "caption_2"):
    if col in df.columns:
        print(f"  {col}: {df[col].notna().sum():,} / {len(df):,}")
```
