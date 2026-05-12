# Temp TODOs — 2026-05-09

출처: 5/8 팀 슬랙 + 분석 2차 계획

---

## 이세현 담당

### 학습
- [ ] **WavTok v6 학습 전환**
  - v6 manifest: `/mnt/ddn/users/jos/_share/manifests_v6`
  - LAION_Freesound 누베스 업로드 중 (~7~8h) → 로컬 맵핑으로 우선 실행
  - 송하가 DAC-VAE/Whisper-tiny 이미 v6로 돌리는 중 (참고)
  - 현재 WavTok v4 Stage1 42k/100k step 진행 중 → 병렬 실행 or 완료 후 교체 결정

### 분석 2차

- [ ] **RQ1 — IEMOCAP distance 분석**
  - Axis 1 (intra-encoder): 동일 화자·발화, encoder layer 1 ↔ layer mid / late → 레이어별 거리 heatmap
  - Axis 2 (cross-encoder): encoder A layer 1 ↔ encoder B layer 1 → encoder 간 표상 비교
  - 사용 데이터: IEMOCAP Session 5, 파일 위치 `/mnt/ddn/kyudan/IEMOCAP/data/`

- [ ] **IEMOCAP, MELD (test) class 카운팅/분포 확인** ← 아래 결과 참고

- [ ] **추가 emotion eval 셋 후보 조사** ← 아래 결과 참고

---

## 조송하 담당 (참고)

- [ ] v6 정리 + 누베스 업로드 완료 + 1차 PR
- [ ] RQ2 — 동일 텍스트 다른 화자 (IEMOCAP, CMU-Arctic) → encoder layer별 distance 분석

---

## 전체 emotion 데이터셋 분포 + duration 조사 결과

### IEMOCAP (leave-session-out: S1-4=train / S5=test)

| Split | 감정 | 샘플 수 | 비율 | mean sec |
|-------|------|--------:|-----:|---------:|
| TRAIN S1-4 | frustrated | 1,468 | 25.0% | N/A (nubes) |
| | neutral | 1,324 | 22.5% | |
| | angry | 933 | 15.9% | |
| | sad | 839 | 14.3% | |
| | excited | 742 | 12.6% | |
| | happy | 452 | 7.7% | |
| | surprise | 89 | 1.5% | |
| | fear | 30 | 0.5% | |
| | other / disgust | 5 | 0.1% | |
| | **합계** | **5,882** | | |
| TEST S5 (4-class) | happy | 613 | 40.7% | 4.2s |
| | neutral | 386 | 25.6% | 3.8s |
| | sad | 311 | 20.6% | 5.0s |
| | angry | 197 | 13.1% | 4.5s |
| | **합계** | **1,507** | | **4.3s** |

**주요 관찰:**
- ⚠️ Train에서 최다 class인 `frustrated`(25%)가 eval 4-class에 **포함 안 됨** (silently drop). 학습 신호 대비 평가 프로토콜 불일치.
- Train `excited`(12.6%) → eval에서 `happy`로 merge. Train `happy`(7.7%)만 보면 test `happy`(40.7%)와 비율 역전.
- `sad`가 5.0s로 가장 긺 — 감정 표현 시 발화가 늘어지는 경향.
- S5 WAV 전부 로컬 존재: `/mnt/ddn/kyudan/IEMOCAP/data/`

---

### MELD (official train / dev / test split)

| Split | 감정 | 샘플 수 | 비율 | mean sec |
|-------|------|--------:|-----:|---------:|
| TRAIN | neutral | 4,710 | 47.2% | — |
| | joy | 1,743 | 17.4% | — |
| | surprise | 1,205 | 12.1% | — |
| | anger | 1,109 | 11.1% | — |
| | sadness | 683 | 6.8% | — |
| | disgust | 271 | 2.7% | — |
| | fear | 268 | 2.7% | — |
| | **합계** | **9,989** | | — |
| DEV | neutral | 469 | 42.3% | 3.0s |
| | joy | 163 | 14.7% | 3.1s |
| | anger | 153 | 13.8% | 3.5s |
| | surprise | 150 | 13.5% | 2.7s |
| | sadness | 111 | 10.0% | 4.0s |
| | fear | 40 | 3.6% | 2.3s |
| | disgust | 22 | 2.0% | 3.1s |
| | **합계** | **1,108** | | **3.0s** |
| TEST | neutral | 1,256 | 48.2% | 3.1s |
| | joy | 401 | 15.4% | 3.2s |
| | anger | 345 | 13.2% | 3.4s |
| | surprise | 281 | 10.8% | 2.7s |
| | sadness | 208 | 8.0% | 3.7s |
| | disgust | 67 | 2.6% | 3.9s |
| | fear | 50 | 1.9% | 4.0s |
| | **합계** | **2,610** | | **3.2s** |

> joy 1개, disgust 1개는 ffprobe 읽기 실패(305s 이상치 포함 가능) → duration 계산에서 제외

**주요 관찰:**
- neutral 극단 편향 (47~48%) — 전 split에서 일관됨
- disgust/fear 소수 class (test 2.6%, 1.9%) → 개별 class 정확도 불안정 예상
- surprise가 2.7s로 가장 짧고, sadness가 3.7s로 가장 긺
- MP4 전부 추출 완료: `/mnt/tmp/datasets/emotion_raw/MELD/MELD.Raw/output_repeated_splits_test/`

---

### DailyTalk (canonical split 없음 — v6 전량 train)

| 감정 | 샘플 수 | 비율 | mean sec |
|------|--------:|-----:|---------:|
| no emotion | 18,966 | 79.8% | 3.3s |
| happiness | 3,856 | 16.2% | 3.3s |
| surprise | 407 | 1.7% | 3.3s |
| sadness | 301 | 1.3% | 4.2s |
| anger | 159 | 0.7% | 4.2s |
| disgust | 66 | 0.3% | 3.9s |
| fear | 18 | 0.1% | 4.0s |
| **합계** | **23,773** | | **3.3s** |

**주의**: `no emotion` 79.8%로 압도적 → 학습 시 다른 emotion label에 대한 신호 희석 가능성  
sadness/anger는 3.3s 대비 4.2s로 약간 긺.

---

### EmoV-DB (canonical split 없음 — v6 전량 train)

| 감정 | 샘플 수 | 비율 | mean sec |
|------|--------:|-----:|---------:|
| sleepy | 1,721 | 25.0% | 5.7s |
| neutral | 1,568 | 22.7% | 3.9s |
| amused | 1,317 | 19.1% | 5.2s |
| angry | 1,268 | 18.4% | 4.4s |
| disgusted | 1,019 | 14.8% | 5.8s |
| **합계** | **6,893** | | **4.9s** |

비교적 균형 잡힌 5-class. sleepy/disgusted가 5.7-5.8s로 길고 neutral이 3.9s로 짧음.

---

### RAVDESS (canonical split 없음 — v6 전량 train)

| 감정 | 샘플 수 | 비율 | mean sec |
|------|--------:|-----:|---------:|
| calm | 192 | 13.3% | 3.8s |
| happy | 192 | 13.3% | 3.6s |
| sad | 192 | 13.3% | 3.7s |
| angry | 192 | 13.3% | 3.9s |
| fearful | 192 | 13.3% | 3.6s |
| disgust | 192 | 13.3% | 3.9s |
| surprised | 192 | 13.3% | 3.5s |
| neutral | 96 | 6.7% | 3.5s |
| **합계** | **1,440** | | **3.7s** |

완벽하게 균형 잡힘 (설계상). 발화 길이도 3.5~3.9s로 매우 균일.

---

### MUStARD++ (canonical split 없음 — v6 전량 train)

| 감정 | 샘플 수 | 비율 | mean sec |
|------|--------:|-----:|---------:|
| neutral | 438 | 36.5% | N/A |
| happiness | 244 | 20.3% | N/A |
| sadness | 149 | 12.4% | N/A |
| excitement | 115 | 9.6% | N/A |
| surprise | 101 | 8.4% | N/A |
| anger | 53 | 4.4% | N/A |
| frustration | 48 | 4.0% | N/A |
| disgust | 29 | 2.4% | N/A |
| fear | 23 | 1.9% | N/A |
| **합계** | **1,200** | | **~4.7s** (논문 기준) |

nubes-only → 로컬 duration 측정 불가. 논문 기재 평균 4.7s.

---

## 추가 emotion eval 셋 후보 조사 결과

### CREMA-D (MahiA/CREMA-D — train/test split 있음)

> v6 학습 풀 **제외** (영구 제외). 로컬 HF cache: `/mnt/ddn/.cache/huggingface/hub/datasets--MahiA--CREMA-D`  
> 91명 (남48/여43), 20~74세, 다인종 (African American / Asian / Caucasian / Hispanic / Unspecified)

| Split | 감정 | 샘플 수 | 비율 | mean sec |
|-------|------|--------:|-----:|---------:|
| TRAIN | neutral | 3,199 | 53.7% | 2.5s |
| | anger | 914 | 15.4% | 2.6s |
| | fear | 623 | 10.5% | 2.5s |
| | disgust | 559 | 9.4% | 2.8s |
| | happy | 355 | 6.0% | 2.4s |
| | sad | 303 | 5.1% | 2.8s |
| | **합계** | **5,953** | | **2.5s** |
| TEST | neutral | 812 | 54.5% | 2.5s |
| | anger | 239 | 16.1% | 2.7s |
| | fear | 170 | 11.4% | 2.6s |
| | disgust | 125 | 8.4% | 2.8s |
| | happy | 76 | 5.1% | 2.3s |
| | sad | 67 | 4.5% | 2.9s |
| | **합계** | **1,489** | | **2.6s** |

**주요 관찰:**
- neutral 극단 편향 (53-54%) — train/test 일관됨
- 전체 발화 매우 짧음 (mean ~2.5s) — 단문 스크립트 기반 (scripted speech)
- happy/sad 소수 class (각 5~6%) → class imbalance 심함

---

### SAVEE (AbstractTTS/SAVEE — split 없음)

> v6 학습 풀 **미포함**. 남성 화자 4명, 27~31세, 영국 영어.  
> 발화당 15 TIMIT 문장: 3 공통 + 2 감정별 + 10 일반

| Split | 감정 | 샘플 수 | 비율 | mean sec |
|-------|------|--------:|-----:|---------:|
| ALL (no split) | neutral | 120 | 25.0% | 3.6s |
| | anger | 60 | 12.5% | 3.7s |
| | disgust | 60 | 12.5% | 4.0s |
| | fear | 60 | 12.5% | 3.7s |
| | happiness | 60 | 12.5% | 3.8s |
| | sadness | 60 | 12.5% | 4.5s |
| | surprise | 60 | 12.5% | 3.8s |
| | **합계** | **480** | | **3.8s** |

**주요 관찰:**
- neutral 2배 (120 vs 60) — 설계상. 나머지 6 class 완벽 균형
- sadness 4.5s로 유독 길고, neutral 3.6s로 짧음
- 소규모 (480개) — 단독 eval 지표보다 다른 데이터셋과 함께 사용 권장

---

### TESS (Bill13579/TESS-mirror — split 없음)

> v6 학습 풀 **미포함**. 여성 화자 2명 (OAF 64세, YAF 26세), 캐나다 영어.  
> 200개 target word × "Say the word _" 구조

| Split | 감정 | 샘플 수 | 비율 | mean sec |
|-------|------|--------:|-----:|---------:|
| ALL (no split) | fear | 400 | 14.3% | 1.7s |
| | angry | 400 | 14.3% | 1.8s |
| | happy | 400 | 14.3% | 2.0s |
| | neutral | 400 | 14.3% | 2.1s |
| | pleasant surprise | 400 | 14.3% | 2.0s |
| | sad | 400 | 14.3% | 2.4s |
| | disgust | 400 | 14.3% | 2.4s |
| | **합계** | **2,800** | | **2.1s** |

> `pleasant surprise`와 `pleasant surprised`가 OAF/YAF 표기 차이 — 동일 class로 merge됨

**주요 관찰:**
- 완벽 균형 (각 400개, 14.3%)
- 발화 매우 짧음 (mean 2.1s) — 단어 1개 읽기 구조
- fear/angry가 1.7-1.8s로 짧고, sad/disgust가 2.4s로 긺

---

### ESD — Emotional Speech Dataset (sonchuate/ESD_dataset)

> v6 학습 풀 **미포함**. 10 영어 + 10 중국어 화자, 각 5감정 × 350발화.

| Split | 화자그룹 | 감정 | 샘플 수 | 비율 | mean sec |
|-------|---------|------|--------:|-----:|---------:|
| ALL (no split) | English (01-10) | Angry | 3,500 | 20.0% | 2.7s |
| | | Happy | 3,500 | 20.0% | 2.8s |
| | | Neutral | 3,500 | 20.0% | 3.2s |
| | | Sad | 3,500 | 20.0% | 4.0s |
| | | Surprise | 3,500 | 20.0% | 3.3s |
| | | **합계** | **17,500** | | **3.2s** |
| | All (EN+ZH) | Angry | 7,000 | 20.0% | 2.7s |
| | | Happy | 7,000 | 20.0% | 2.8s |
| | | Neutral | 7,000 | 20.0% | 2.9s |
| | | Sad | 7,000 | 20.0% | 3.5s |
| | | Surprise | 7,000 | 20.0% | 3.0s |
| | | **합계** | **35,000** | | **3.0s** |

**주요 관찰:**
- 완벽 균형 (각 20%), scripted parallel utterances
- Sad가 4.0s로 가장 길고, Angry가 2.7s로 짧음
- English only 17,500 / All 35,000 — eval 시 영어 화자만 필터 권장
- 화자 필터링: speaker 0001~0010 = English

---

### JL-Corpus (CLAPv2/JL-Corpus — split 없음)

> v6 학습 풀 **미포함**. New Zealand English, 남성/여성 화자.  
> 5 primary + 5 secondary emotions = 10-class

| Split | 감정 | 샘플 수 | 비율 | mean sec |
|-------|------|--------:|-----:|---------:|
| ALL (no split) | angry | 240 | 10.0% | 2.0s |
| | anxious | 240 | 10.0% | 2.2s |
| | apologetic | 240 | 10.0% | 2.1s |
| | assertive | 240 | 10.0% | 2.1s |
| | concerned | 240 | 10.0% | 2.2s |
| | encouraging | 240 | 10.0% | 2.0s |
| | excited | 240 | 10.0% | 2.1s |
| | happy | 240 | 10.0% | 2.0s |
| | neutral | 240 | 10.0% | 2.1s |
| | sad | 240 | 10.0% | 2.2s |
| | **합계** | **2,400** | | **2.1s** |

**주요 관찰:**
- 완벽 균형 (각 240개, 10%)
- 발화 짧음 (mean 2.1s), class 간 차이 거의 없음
- anxious/apologetic/assertive/concerned/encouraging 등 미세 감정 포함 — 타 데이터셋과 class 매핑 주의

---

### MLEnd (spoken numerals — 조사 보류)

> `pip install mlend` 패키지 존재하나 오디오 데이터는 별도 GitHub 다운로드 필요.  
> 감정 class가 Neutral/Bored/Excited/Question으로 **억양(intonation) 레이블** — 일반 emotion eval과 성격 다름.  
> → emotion benchmark 목적으로는 우선순위 낮음. 추후 prosody/intonation 분석 시 재검토.

---

### 데이터셋 요약 비교표

| 데이터셋 | v6 train 포함 | split | 총 샘플 | classes | mean sec | eval 사용 가능 |
|---------|:------------:|-------|--------:|--------:|---------:|:------------:|
| IEMOCAP | ✓ (S1-4) | S5=test | 5,882+1,507 | 10→4 eval | 4.3s(test) | ✓ |
| MELD | ✓ (train+dev) | official | 9,989+1,109+2,610 | 7 | 3.0-3.2s | ✓ |
| DailyTalk | ✓ (all) | — | 23,773 | 7 | 3.3s | ✗ |
| EmoV-DB | ✓ (all) | — | 6,893 | 5 | 4.9s | ✗ |
| RAVDESS | ✓ (all) | — | 1,440 | 8 | 3.7s | ✗ |
| MUStARD++ | ✓ (all) | — | 1,200 | 9 | ~4.7s | ✗ |
| **CREMA-D** | ✗ | train/test | 5,953+1,489 | 6 | 2.5s | **✓** |
| **SAVEE** | ✗ | — | 480 | 7 | 3.8s | **✓** |
| **TESS** | ✗ | — | 2,800 | 7 | 2.1s | **✓** |
| **ESD** | ✗ | — | 17,500(EN) | 5 | 3.2s | **✓** |
| **JL-Corpus** | ✗ | — | 2,400 | 10 | 2.1s | **✓** |
| MLEnd | ✗ | — | 32,654 | 4 (intonation) | — | △ (억양) |

---

## 전체 데이터셋 × 감정 class 교차표

> Row = 정규화된 감정 class (core 7개 먼저, 데이터셋 고유 class 이후)  
> Column = 데이터셋 (train pool → eval sets 순)  
> 값 = 샘플 수 (`-` = 해당 class 없음)  
> MELD tr+dev = train 9,989 + dev 1,108 합산. ESD = 영어 화자 10명 only.

| emotion | IEMOCAP (tr-S1-4) | MELD (tr+dev) | DailyTalk (tr-all) | EmoV-DB (tr-all) | RAVDESS (tr-all) | MUStARD++ (tr-all) | IEMOCAP (te-S5) | MELD (te) | CREMA-D (tr) | CREMA-D (te) | SAVEE (all) | TESS (all) | ESD-EN (all) | JL-Corpus (all) | **TOTAL** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **anger** | 933 | 1,262 | 159 | 1,268 | 192 | 53 | 197 | 345 | 914 | 239 | 60 | 400 | 3,500 | 240 | **9,762** |
| **disgust** | 2 | 293 | 66 | 1,019 | 192 | 29 | - | 67 | 559 | 125 | 60 | 400 | - | - | **2,812** |
| **fear** | 30 | 308 | 18 | - | 192 | 23 | - | 50 | 623 | 170 | 60 | 400 | - | - | **1,874** |
| **happiness** | 452 | 1,906 | 3,856 | - | 192 | 244 | 613 | 401 | 355 | 76 | 60 | 400 | 3,500 | 240 | **12,295** |
| **neutral** | 1,324 | 5,179 | 18,966 | 1,568 | 96 | 438 | 386 | 1,256 | 3,199 | 812 | 120 | 400 | 3,500 | 240 | **37,484** |
| **sadness** | 839 | 794 | 301 | - | 192 | 149 | 311 | 208 | 303 | 67 | 60 | 400 | 3,500 | 240 | **7,364** |
| **surprise** | 89 | 1,355 | 407 | - | 192 | 101 | - | 281 | - | - | 60 | 400 | 3,500 | - | **6,385** |
| amused | - | - | - | 1,317 | - | - | - | - | - | - | - | - | - | - | 1,317 |
| calm | - | - | - | - | 192 | - | - | - | - | - | - | - | - | - | 192 |
| excited | 742 | - | - | - | - | 115 | - | - | - | - | - | - | - | 240 | 1,097 |
| frustrated | 1,468 | - | - | - | - | 48 | - | - | - | - | - | - | - | - | 1,516 |
| sleepy | - | - | - | 1,721 | - | - | - | - | - | - | - | - | - | - | 1,721 |
| anxious | - | - | - | - | - | - | - | - | - | - | - | - | - | 240 | 240 |
| apologetic | - | - | - | - | - | - | - | - | - | - | - | - | - | 240 | 240 |
| assertive | - | - | - | - | - | - | - | - | - | - | - | - | - | 240 | 240 |
| concerned | - | - | - | - | - | - | - | - | - | - | - | - | - | 240 | 240 |
| encouraging | - | - | - | - | - | - | - | - | - | - | - | - | - | 240 | 240 |
| other | 3 | - | - | - | - | - | - | - | - | - | - | - | - | - | 3 |
| **TOTAL** | **5,882** | **11,097** | **23,773** | **6,893** | **1,440** | **1,200** | **1,507** | **2,608** | **5,953** | **1,489** | **480** | **2,800** | **17,500** | **2,400** | **85,022** |

**주요 관찰:**
- `neutral` 독보적 최다 (37,484 = 전체의 **44.1%**) — DailyTalk no-emotion 18,966이 단독으로 22%
- `fear` 가장 희귀 (1,874 = 2.2%) — CREMA-D와 TESS가 절반 이상 담당
- `disgust` 두 번째 희귀 (2,812 = 3.3%) — EmoV-DB 1,019가 36% 차지
- Core 7 class 합계: 77,976 (91.7%). 나머지 8.3%는 데이터셋 고유 class (amused/sleepy/excited/frustrated 등)
- ESD-EN (17,500)과 DailyTalk (23,773)이 전체의 48% — 이 두 데이터셋이 분포를 지배함
- eval 전용 세트 (IEMOCAP S5 + MELD te + CREMA-D + SAVEE + TESS + ESD + JL) 합계: 34,237 (40.3%)
