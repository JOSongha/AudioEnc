# Plan 1 — CMU-ARCTIC / L2-ARCTIC Download (Representation Richness, Phase 1)

> 실행자: **Haiku**. 본 plan 은 데이터 다운로드까지만 다룬다. 분석은 Plan 2, 3 에서 수행.

---

## 목표

본 plan 의 다운로드 범위:

- **CMU-ARCTIC 7 화자** ✅ 다운로드 + 매니페스트 (분석 대상)
- **L2-ARCTIC** ✅ 다운로드 + 매니페스트 (사용자가 등록 완료, 이번 plan 에서 받아둠. 분석은 후속 phase)
- **CMU-ARCTIC 18 화자** ⏸️ 본 plan 에서는 제외. 후속 plan 에서 다룬다.

산출물: `/mnt/tmp/cache/cmu_arctic_7/manifest.csv`, `/mnt/tmp/cache/l2_arctic/manifest.csv`. Plan 2 가 7 화자 매니페스트를 입력으로 임베딩을 추출한다.

### HF 인증

HF token 은 표준 경로 `~/.cache/huggingface/token` 에 이미 저장되어 있음 (600 권한). `huggingface_hub` / `datasets` / `huggingface-cli` 가 이 경로를 자동 인식. **Plan 파일이나 코드에 literal token 을 적지 말 것.** 추가 인증 불필요.

---

## 선행 조건 (시작 전 검증)

다음을 한 줄씩 확인. 실패 시 즉시 보고하고 중단.

```bash
# 1. 디스크 공간 — /mnt/tmp 가용량 ≥ 30 GB
df -h /mnt/tmp | tail -1

# 2. python + 필수 패키지
python -c "import datasets, huggingface_hub, soundfile, librosa; print('ok')"
# 부재 시: pip install datasets huggingface_hub soundfile librosa  (기존 venv 가 어떤건지 확인 후)

# 3. 네트워크
curl -sI https://huggingface.co/ | head -1
curl -sI http://festvox.org/ | head -1
```

`/mnt/tmp/cache` 는 이미 있음 (`hf/`, `openslr___librispeech_asr/` 등). HF cache 는 그 아래 그대로 사용.

---

## 절대 금지 / 주의사항

- ❌ **기존 cache 디렉토리 (`/mnt/tmp/cache/hf`, `openslr___librispeech_asr` 등) 건드리지 말 것.** 다른 프로젝트가 사용 중일 수 있음.
- ❌ `git add -A` / 전체 커밋 금지. 본 plan 은 코드 수정 없음 — 데이터만 받음.
- ❌ 다운로드 도중 Ctrl-C 후 같은 디렉토리 재시도 시 부분 파일이 남을 수 있음 → **재시도 전 해당 화자 디렉토리 삭제 후 다시**.
- ❌ Haiku 가 임의로 다른 데이터셋 (예: VCTK, EXPRESSO) 까지 받지 말 것. 본 plan 은 ARCTIC 만.
- ⚠️ 작업 중 새 터미널을 열어야 할 일이 생기면 사용자에게 보고. Plan 은 단일 세션 내에서 완수 가능하게 짜여 있음.
- ⚠️ HF 다운로드가 5 분 이상 진행 없음 → 네트워크 문제 의심, 보고하고 중단. 무한 대기 금지.
- ⚠️ 각 Phase 끝에 **검증 단계** 가 있음. 검증 실패 시 다음 Phase 로 넘어가지 말고 보고.

---

## TaskCreate 트래킹

시작 시 다음 task 들을 등록:

```
1. 환경 점검 (HF token 인식 포함)
2. 디렉토리 구조 생성
3. CMU-ARCTIC 7 화자 다운로드
4. CMU-ARCTIC 7 화자 매니페스트 생성 + 검증
5. L2-ARCTIC 다운로드
6. L2-ARCTIC 매니페스트 생성 + 검증
7. 최종 산출물 점검
```

각 task 시작 시 `in_progress`, 완료 시 즉시 `completed` 로 업데이트.

---

## Phase 1.0 — 디렉토리 구조 생성

```bash
mkdir -p /mnt/tmp/cache/cmu_arctic_7
mkdir -p /mnt/tmp/cache/l2_arctic
mkdir -p /mnt/tmp/cache/_downloads_tmp   # tar.bz2 등 임시 보관
```

산출물 디렉토리 규약 (Plan 2 가 의존):

```
/mnt/tmp/cache/cmu_arctic_7/
├── manifest.csv
├── bdl/
│   ├── arctic_a0001.wav
│   └── ...
├── slt/
└── ... (총 7 명: bdl, slt, jmk, awb, rms, clb, ksp)

/mnt/tmp/cache/l2_arctic/
├── manifest.csv
└── (24 화자)
```

---

## Phase 1.1 — CMU-ARCTIC 7 화자 다운로드

7 화자: **bdl, slt, jmk, awb, rms, clb, ksp**.

### 1.1.A — HF 시도 (primary)

```python
# 검색
from huggingface_hub import HfApi
api = HfApi()
results = list(api.list_datasets(search="cmu_arctic", limit=20))
for r in results: print(r.id)
results = list(api.list_datasets(search="cmu-arctic", limit=20))
for r in results: print(r.id)
```

후보가 보이면 README / data card 를 확인 (raw audio + transcript 동봉이 조건). 후보 없거나 모두 xvector / embedding 만 있는 미러면 1.1.B 로.

> **⚠️ Haiku 주의**: 여기서 너무 오래 헤매지 말 것. HF 후보 3 개까지만 시도, 그 이상은 1.1.B 로 직행.

HF 데이터셋이 적절하면:

```python
from datasets import load_dataset
ds = load_dataset(dataset_id, cache_dir="/mnt/tmp/cache/hf")
# 화자 필터링은 dataset schema 확인 후
```

다운로드 후 화자별 wav + transcript 를 §1.0 의 디렉토리 규약대로 정렬. 파일 형식은 16 kHz / 16-bit / mono WAV 로 통일 (필요 시 librosa resample).

### 1.1.B — Festvox 직접 다운로드 (fallback, 권장 경로)

다음 스크립트 그대로 실행. URL pattern 검증됨.

```bash
cd /mnt/tmp/cache/_downloads_tmp

for s in bdl slt jmk awb rms clb ksp; do
  url="http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_${s}_arctic-0.95-release.tar.bz2"
  echo "=== ${s} ==="
  if [ -f "cmu_us_${s}_arctic-0.95-release.tar.bz2" ]; then
    echo "already downloaded, skip"
  else
    wget --tries=3 --timeout=60 "${url}"
  fi
done
```

압축 풀고 표준 디렉토리로 이동:

```bash
cd /mnt/tmp/cache/_downloads_tmp
for s in bdl slt jmk awb rms clb ksp; do
  tar -xjf "cmu_us_${s}_arctic-0.95-release.tar.bz2"
  # 압축 해제 결과: cmu_us_${s}_arctic/ 디렉토리에 wav/, etc/ 있음
  mkdir -p /mnt/tmp/cache/cmu_arctic_7/${s}
  cp cmu_us_${s}_arctic/wav/*.wav /mnt/tmp/cache/cmu_arctic_7/${s}/
  # transcript: etc/txt.done.data — 매니페스트 단계에서 파싱
  cp cmu_us_${s}_arctic/etc/txt.done.data /mnt/tmp/cache/cmu_arctic_7/${s}/_txt.done.data
done
```

> **⚠️ wav 형식**: festvox 의 ARCTIC 은 16 kHz / 16-bit / mono. resample 불필요. 파일이 다른 sr 이면 (예: 32 kHz mirror) 보고하고 중단.

### 1.1.C — 검증 (필수)

```bash
# 화자별 utterance 수 — 모두 1132 여야 함
for s in bdl slt jmk awb rms clb ksp; do
  n=$(ls /mnt/tmp/cache/cmu_arctic_7/${s}/*.wav 2>/dev/null | wc -l)
  echo "${s}: ${n} utts"
done
# 기대: 모두 1132 (또는 ±2 정도. clb/jmk 은 일부 누락된 케이스 알려져 있음)
```

```bash
# wav format 1 개 샘플 확인
soxi /mnt/tmp/cache/cmu_arctic_7/bdl/arctic_a0001.wav 2>&1 | head -5
# 또는
python -c "import soundfile as sf; info=sf.info('/mnt/tmp/cache/cmu_arctic_7/bdl/arctic_a0001.wav'); print(info)"
# 기대: samplerate=16000, channels=1, subtype=PCM_16
```

검증 실패 시 보고하고 중단.

---

## Phase 1.2 — 7 화자 매니페스트 생성

**스크립트**: `experiments/representation_richness/build_manifest_arctic.py` 신규 작성.

```python
"""
CMU-ARCTIC manifest builder.

Reads:  /mnt/tmp/cache/cmu_arctic_<set>/<speaker>/_txt.done.data
                                                  /<utt_id>.wav
Writes: /mnt/tmp/cache/cmu_arctic_<set>/manifest.csv

Columns: audio_path, speaker_id, utt_id, transcript_id, transcript, duration_s
- transcript_id == utt_id (e.g. "arctic_a0001"); 화자 간 같은 utt_id == 같은 prompt
- transcript: txt.done.data 한 줄을 parsing (`( arctic_a0001 "<text>" )` 형식)
"""
```

매니페스트 스펙:

| Column | 예시 | 비고 |
|---|---|---|
| audio_path | `/mnt/tmp/cache/cmu_arctic_7/bdl/arctic_a0001.wav` | 절대경로 |
| speaker_id | `bdl` | |
| utt_id | `arctic_a0001` | |
| transcript_id | `arctic_a0001` | utt_id 와 동일 (CMU-ARCTIC 정의) |
| transcript | `Author of the danger trail, Philip Steels, etc.` | |
| duration_s | `2.83` | soundfile 로 계산 |

txt.done.data 파싱 정규식: `^\( (\w+) "(.*)" \)$`

검증 단계:

```python
# 1. 행 수: 7 × 1132 = 7924 (±오차 허용)
# 2. transcript_id 별 화자 분포: 7 화자 모두 있어야 (대다수 transcript_id 에서)
# 3. duration: max 30 초 미만 (Whisper 30s padding 대비)
# 4. NaN/empty transcript 없음
```

검증 실패 시: 어떤 항목 실패했는지 보고. transcript_id 누락이 일부 (예: clb 가 12 utt 부족) 면 그건 알려진 issue 이므로 그대로 진행 OK. 50 개 이상 누락이면 중단.

---

## Phase 1.3 — L2-ARCTIC

**Source**: `https://psi.engr.tamu.edu/l2-arctic-corpus/` (공식). HF mirror 가 있을 수도 있고 없을 수도 있음.

사용자가 L2-ARCTIC 등록 완료. HF token 도 표준 경로에 저장됨.

### 1.3.A — HF 시도 (primary)

```python
from huggingface_hub import HfApi
api = HfApi()  # ~/.cache/huggingface/token 자동 인식
for q in ["l2_arctic", "l2-arctic", "L2-ARCTIC"]:
    for r in api.list_datasets(search=q, limit=10):
        print(r.id)
```

후보가 보이면 `datasets.load_dataset(repo_id, cache_dir="/mnt/tmp/cache/hf", token=True)` 시도. raw audio + transcript 동봉이 조건. gated dataset 인 경우 token 으로 자동 인증.

### 1.3.B — 공식 페이지 (fallback)

L2-ARCTIC 공식: `https://psi.engr.tamu.edu/l2-arctic-corpus/`. 등록 후 다운로드 링크가 이메일/페이지로 제공.

1. 사용자가 등록 후 받은 zip 링크 / 직접 URL 이 있다면 사용자에게 확인 후 wget. **Haiku 가 페이지에서 임의로 링크 추측하지 말 것.**
2. URL 알 수 없으면 사용자에게 "L2-ARCTIC 다운로드 URL 공유 부탁드립니다" 보고하고 §1.3 STOP.
3. zip 받으면 `/mnt/tmp/cache/_downloads_tmp/l2arctic_release_*.zip` 으로 저장 후 unzip.

### 1.3.C — 표준 디렉토리 정렬

L2-ARCTIC 24 화자 (예: `ABA, ASI, BWC, EBVS, ERMS, HJK, HKK, HQTV, LXC, MBMPS, NCC, NJS, PNV, RRBI, SKA, SVBI, THV, TLV, TNI, TXHC, YBAA, YDCK, YKWK, ZHAA`) 중 실제 release 에 포함된 화자만. 화자별 `wav/` 디렉토리에 16 kHz mono PCM 으로 통일.

```
/mnt/tmp/cache/l2_arctic/
├── ABA/
│   ├── arctic_a0001.wav
│   └── ...
├── ASI/
└── ... (실제 다운로드된 화자만)
```

L2-ARCTIC 의 prompt 는 CMU-ARCTIC 과 동일 (`arctic_a0001` ~ `arctic_b0539`). transcript 는 화자별 `transcript/` 하위 .txt 또는 별도 prompt 파일에 있음. 압축 해제 후 실제 구조 확인하고 매니페스트 빌더에 반영.

### 1.3.D — 매니페스트 + 검증

`build_manifest_arctic.py` 를 L2 모드로 호출:

- audio_path, speaker_id, utt_id, transcript_id (utt_id 와 동일), transcript, duration_s
- 검증: 화자 수 (≥ 20), 화자별 utt 수 (수백 단위), wav format 16 kHz mono PCM_16
- duration: max 30 초 미만 (대부분 OK)

---

## Phase 1.4 — 최종 점검

```bash
echo "== CMU-ARCTIC 7 화자 =="
wc -l /mnt/tmp/cache/cmu_arctic_7/manifest.csv
du -sh /mnt/tmp/cache/cmu_arctic_7

echo "== L2-ARCTIC =="
wc -l /mnt/tmp/cache/l2_arctic/manifest.csv 2>/dev/null || echo "L2 not downloaded"
du -sh /mnt/tmp/cache/l2_arctic 2>/dev/null

echo "== _downloads_tmp 정리 가능 =="
du -sh /mnt/tmp/cache/_downloads_tmp
```

`_downloads_tmp` (tar.bz2 / zip 원본) 은 사용자 확인 후 삭제. 자동 삭제 X.

---

## 완료 조건

다음이 모두 충족될 때만 Plan 1 완료:

- [ ] `/mnt/tmp/cache/cmu_arctic_7/manifest.csv` 가 약 7920 행 (7 화자 × 1132 ± 누락) 을 가짐.
- [ ] `cmu_arctic_7` 의 모든 wav 가 16 kHz / mono / PCM_16.
- [ ] L2-ARCTIC: 다운로드 성공 시 manifest 완성, URL 미공유 등으로 실패 시 명확한 사유 보고.
- [ ] `experiments/representation_richness/build_manifest_arctic.py` 스크립트가 git 에 staged 되지 않음 (사용자 직접 커밋).

---

## Plan 2 로 넘기는 산출물

```
/mnt/tmp/cache/cmu_arctic_7/manifest.csv     ← Plan 2 가 이 파일을 입력으로 씀
/mnt/tmp/cache/l2_arctic/manifest.csv        ← (후속 plan 에서 사용)
experiments/representation_richness/build_manifest_arctic.py
```

Plan 1 끝에서 위 파일들의 절대경로와 행 수, 디스크 사용량을 한 번 더 출력해서 사용자에게 보고.
