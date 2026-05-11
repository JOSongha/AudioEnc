# Emotion Datasets — v6 실제 구성 분석

---

## 핵심 특이사항 (docs/emotion_datasets.md 분석과의 차이)

1. **매니페스트 단계에서 class-balanced sampling 적용** — raw 데이터셋의 neutral 44% 편향 등은 실제 학습 분포와 무관
2. **데이터셋별 독립 label set** — 통합 7-class 분류가 아닌 각 데이터셋 고유 MCQ 형식
3. **IEMOCAP train(10-class) ↔ eval(4-class) 불일치** — train에서 학습한 6개 class가 eval에 미등장
4. **eval 매니페스트 없음** — MELD test + IEMOCAP S5 test는 외부 eval 파이프라인으로 별도 처리

---

## Train 구성

**총 50,284 샘플, 6개 데이터셋**

| 데이터셋 | 샘플 수 | class 수 | 분포 형태 | 파일 |
|---------|-------:|:-------:|:--------:|------|
| IEMOCAP train (S1-4) | 5,882 | 10 | 균등 (~10%) | `emotion_iemocap_train_000{0,1}.jsonl` |
| MELD tr+dev | 11,096 | 7 | 균등 (~14%) | `emotion_meld_0000.jsonl` |
| DailyTalk | 23,773 | 7 | 균등 (~14%) | `emotion_dailytalk_000{0,1}.jsonl` |
| EmoV-DB | 6,893 | 5 | 균등 (~20%) | `emotion_emovdb_0000.jsonl` |
| RAVDESS | 1,440 | 8 | 균등 (~12%) | `emotion_ravdess_0000.jsonl` |
| MUStARD++ | 1,200 | 9 | 균등 (~11%) | `emotion_mustardpp_0000.jsonl` |
| **합계** | **50,284** | | | |

### 데이터셋별 실제 class 분포

#### IEMOCAP train (5,882 / 10-class)

| 감정 | 샘플 수 | 비율 |
|------|--------:|-----:|
| neutral | 627 | 10.7% |
| surprise | 619 | 10.5% |
| angry | 607 | 10.3% |
| other | 607 | 10.3% |
| disgust | 587 | 10.0% |
| excited | 584 | 9.9% |
| sad | 575 | 9.8% |
| happy | 568 | 9.7% |
| frustrated | 566 | 9.6% |
| fear | 542 | 9.2% |

#### MELD tr+dev (11,096 / 7-class)

| 감정 | 샘플 수 | 비율 |
|------|--------:|-----:|
| sadness | 1,645 | 14.8% |
| disgust | 1,643 | 14.8% |
| anger | 1,586 | 14.3% |
| joy | 1,571 | 14.2% |
| surprise | 1,567 | 14.1% |
| neutral | 1,548 | 14.0% |
| fear | 1,536 | 13.8% |

> raw 분포(neutral 47%)와 달리 균등 샘플링 적용됨

#### DailyTalk (23,773 / 7-class)

| 감정 | 샘플 수 | 비율 |
|------|--------:|-----:|
| sadness | 3,421 | 14.4% |
| happiness | 3,414 | 14.4% |
| anger | 3,408 | 14.3% |
| fear | 3,400 | 14.3% |
| surprise | 3,384 | 14.2% |
| no emotion | 3,374 | 14.2% |
| disgust | 3,372 | 14.2% |

> raw 분포(no emotion 79.8%)와 달리 균등 샘플링 적용됨  
> ⚠️ "no emotion" ≠ "neutral" — DailyTalk 고유 레이블, eval의 neutral과 이질적

#### EmoV-DB (6,893 / 5-class)

| 감정 | 샘플 수 | 비율 |
|------|--------:|-----:|
| sleepy | 1,407 | 20.4% |
| amused | 1,380 | 20.0% |
| neutral | 1,378 | 20.0% |
| disgusted | 1,373 | 19.9% |
| angry | 1,355 | 19.7% |

> ⚠️ sleepy / amused — eval에 존재하지 않는 label

#### RAVDESS (1,440 / 8-class)

| 감정 | 샘플 수 | 비율 |
|------|--------:|-----:|
| happy | 205 | 14.2% |
| disgust | 202 | 14.0% |
| sad | 176 | 12.2% |
| angry | 175 | 12.2% |
| surprised | 175 | 12.2% |
| neutral | 172 | 11.9% |
| calm | 171 | 11.9% |
| fearful | 164 | 11.4% |

> ⚠️ calm — eval에 존재하지 않는 label

#### MUStARD++ (1,200 / 9-class)

| 감정 | 샘플 수 | 비율 |
|------|--------:|-----:|
| surprise | 148 | 12.3% |
| fear | 140 | 11.7% |
| sadness | 140 | 11.7% |
| happiness | 137 | 11.4% |
| neutral | 136 | 11.3% |
| frustration | 135 | 11.2% |
| anger | 129 | 10.8% |
| excitement | 126 | 10.5% |
| disgust | 109 | 9.1% |

> ⚠️ frustration / excitement — eval에 존재하지 않는 label

---

## Eval 구성

**eval 매니페스트 없음** — 외부 평가 파이프라인에서 직접 처리.  
실제 eval 기준: **MELD test + IEMOCAP S5 test**

### MELD test (2,608 / 7-class) — raw 분포

| 감정 | 샘플 수 | 비율 |
|------|--------:|-----:|
| neutral | 1,256 | 48.2% |
| joy | 401 | 15.4% |
| anger | 345 | 13.2% |
| surprise | 281 | 10.8% |
| sadness | 208 | 8.0% |
| disgust | 67 | 2.6% |
| fear | 50 | 1.9% |
| **합계** | **2,608** | |

### IEMOCAP test S5 (1,507 / 4-class) — raw 분포

eval 프로토콜: excited → happy 병합, frustrated / surprise / fear / disgust / other 제외

| 감정 | 샘플 수 | 비율 |
|------|--------:|-----:|
| happy | 613 | 40.7% |
| neutral | 386 | 25.6% |
| sad | 311 | 20.6% |
| angry | 197 | 13.1% |
| **합계** | **1,507** | |

---

## Train-Eval 구조 분석

### 1. 분포 불일치 (Train balanced vs Eval raw)

| | Train (balanced) | MELD test | IEMOCAP test |
|-|:----------------:|:---------:|:------------:|
| neutral | ~10–14% per dataset | **48.2%** | 25.6% |
| happy/joy | ~10–14% | 15.4% | **40.7%** |
| anger | ~10–14% | 13.2% | 13.1% |
| sadness | ~10–14% | 8.0% | 20.6% |
| surprise | ~10–14% | 10.8% | — |
| disgust | ~10–14% | 2.6% | — |
| fear | ~10–14% | 1.9% | — |

모델은 **균등 분포**로 학습했으나 **편향된 분포**로 평가받음.  
특히 MELD test의 neutral 48% — train에서 14%만 본 패턴을 48% 빈도로 맞춰야 함.

### 2. Label set 불일치 (Train 10-class → Eval 4-class)

IEMOCAP의 경우:

| 구분 | label set |
|------|-----------|
| Train | angry, disgust, **excited**, **fear**, **frustrated**, happy, neutral, **other**, sad, **surprise** |
| Eval | angry, happy, neutral, sad |

- **excited** (584건 학습) → eval에서 happy로 병합되나 train MCQ에는 별도 class로 존재
- **frustrated** (566건), **surprise** (619건), **fear** (542건), **disgust** (587건), **other** (607건) → eval에 미등장
- train에서 학습한 5,882건 중 **2,886건(49%)은 eval에 존재하지 않는 class**

### 3. 데이터셋별 label name 불통일

eval에 등장하는 감정을 train에서 어떻게 표현하는지:

| 감정 (eval) | IEMOCAP | MELD | DailyTalk | RAVDESS | MUStARD++ |
|------------|:-------:|:----:|:---------:|:-------:|:---------:|
| neutral | neutral | neutral | no emotion⚠️ | neutral | neutral |
| happy | happy | joy⚠️ | happiness | happy | happiness |
| anger | angry | anger | anger | angry | anger |
| sad | sad | sadness | sadness | sad | sadness |
| surprise | surprise | surprise | surprise | surprised | surprise |
| disgust | disgust | disgust | disgust | disgust | disgust |
| fear | fear | fear | fear | fearful⚠️ | fear |

> ⚠️ 표기 차이: 모델이 MCQ 선택지 텍스트 레벨에서 이를 같은 개념으로 학습하는지 불명확

### 4. Eval에 없는 Train-only class 정리

아래 class는 train에서 학습되지만 현재 eval에서 성능을 측정하지 않음:

| class | 등장 데이터셋 | 학습 샘플 수 |
|-------|-------------|------------:|
| excited | IEMOCAP | 584 |
| frustrated | IEMOCAP | 566 |
| other | IEMOCAP | 607 |
| no emotion | DailyTalk | 3,374 |
| sleepy | EmoV-DB | 1,407 |
| amused | EmoV-DB | 1,380 |
| disgusted | EmoV-DB | 1,373 |
| calm | RAVDESS | 171 |
| frustration | MUStARD++ | 135 |
| excitement | MUStARD++ | 126 |

---

## Label 전수조사

> 소스: v6 audio_emotion 매니페스트 전체 파일 직접 파싱  
> 고유 label 수: **23개** / 총 샘플: **50,284**

### 1. 전체 label — count 순

| label | count | pct | cum% | 등장 데이터셋 | eval? |
|-------|------:|----:|-----:|--------------|:-----:|
| disgust | 5,902 | 11.7% | 11.7% | DailyTalk · IEMOCAP · MELD · MUStARD++ · RAVDESS | ✓ |
| surprise | 5,706 | 11.3% | 23.1% | DailyTalk · IEMOCAP · MELD · MUStARD++ | ✓ |
| fear | 5,624 | 11.2% | 34.3% | DailyTalk · IEMOCAP · MELD · MUStARD++ | ✓ |
| sadness | 5,184 | 10.3% | 44.6% | DailyTalk · MELD · MUStARD++ | ✓ |
| anger | 5,123 | 10.2% | 54.8% | DailyTalk · MELD · MUStARD++ | ✓ |
| neutral | 3,831 | 7.6% | 62.4% | EmoV-DB · IEMOCAP · MELD · MUStARD++ · RAVDESS | ✓ |
| happiness | 3,562 | 7.1% | 69.5% | DailyTalk · MUStARD++ | ✓ |
| no emotion | 3,416 | 6.8% | 76.3% | DailyTalk | △ |
| angry | 2,140 | 4.3% | 80.5% | EmoV-DB · IEMOCAP · RAVDESS | ✓ (anger) |
| joy | 1,571 | 3.1% | 83.6% | MELD | ✓ (happiness) |
| sleepy | 1,407 | 2.8% | 86.4% | EmoV-DB | ✗ |
| amused | 1,380 | 2.7% | 89.2% | EmoV-DB | ✗ |
| disgusted | 1,373 | 2.7% | 91.9% | EmoV-DB | ✓ (disgust) |
| happy | 757 | 1.5% | 93.4% | IEMOCAP · RAVDESS | ✓ (happiness) |
| sad | 756 | 1.5% | 94.9% | IEMOCAP · RAVDESS | ✓ (sadness) |
| other | 609 | 1.2% | 96.1% | IEMOCAP | ✗ |
| excited | 588 | 1.2% | 97.3% | IEMOCAP | ✗ |
| frustrated | 584 | 1.2% | 98.5% | IEMOCAP | ✗ |
| surprised | 175 | 0.3% | 98.8% | RAVDESS | ✓ (surprise) |
| calm | 171 | 0.3% | 99.2% | RAVDESS | ✗ |
| fearful | 164 | 0.3% | 99.5% | RAVDESS | ✓ (fear) |
| frustration | 135 | 0.3% | 99.7% | MUStARD++ | ✗ |
| excitement | 126 | 0.3% | 100.0% | MUStARD++ | ✗ |

### 2. label × dataset 교차표

| label | DailyTalk | EmoV-DB | IEMOCAP | MELD | MUStARD++ | RAVDESS | TOTAL |
|-------|----------:|--------:|--------:|-----:|----------:|--------:|------:|
| disgust | 3,373 | — | 575 | 1,643 | 109 | 202 | **5,902** |
| surprise | 3,372 | — | 619 | 1,567 | 148 | — | **5,706** |
| fear | 3,380 | — | 568 | 1,536 | 140 | — | **5,624** |
| sadness | 3,399 | — | — | 1,645 | 140 | — | **5,184** |
| anger | 3,408 | — | — | 1,586 | 129 | — | **5,123** |
| neutral | — | 1,378 | 597 | 1,548 | 136 | 172 | **3,831** |
| happiness | 3,425 | — | — | — | 137 | — | **3,562** |
| no emotion | 3,416 | — | — | — | — | — | **3,416** |
| angry | — | 1,355 | 610 | — | — | 175 | **2,140** |
| joy | — | — | — | 1,571 | — | — | **1,571** |
| sleepy | — | 1,407 | — | — | — | — | **1,407** |
| amused | — | 1,380 | — | — | — | — | **1,380** |
| disgusted | — | 1,373 | — | — | — | — | **1,373** |
| happy | — | — | 552 | — | — | 205 | **757** |
| sad | — | — | 580 | — | — | 176 | **756** |
| other | — | — | 609 | — | — | — | **609** |
| excited | — | — | 588 | — | — | — | **588** |
| frustrated | — | — | 584 | — | — | — | **584** |
| surprised | — | — | — | — | — | 175 | **175** |
| calm | — | — | — | — | — | 171 | **171** |
| fearful | — | — | — | — | — | 164 | **164** |
| frustration | — | — | — | — | 135 | — | **135** |
| excitement | — | — | — | — | 126 | — | **126** |
| **TOTAL** | **23,773** | **6,893** | **5,882** | **11,096** | **1,200** | **1,440** | **50,284** |

### 3. 개념 병합 후 분포

동의어 병합 시 **14개 개념** (23 label → 14 concept):

| 개념 | 병합 label | count | pct | eval |
|------|-----------|------:|----:|:----:|
| disgust | disgust · disgusted | 7,275 | 14.5% | ✓ |
| anger | anger · angry | 7,263 | 14.4% | ✓ |
| sadness | sadness · sad | 5,940 | 11.8% | ✓ |
| happiness | happiness · happy · joy | 5,890 | 11.7% | ✓ |
| surprise | surprise · surprised | 5,881 | 11.7% | ✓ |
| fear | fear · fearful | 5,788 | 11.5% | ✓ |
| neutral | neutral | 3,831 | 7.6% | ✓ |
| no emotion | no emotion | 3,416 | 6.8% | △ |
| sleepy | sleepy | 1,407 | 2.8% | ✗ |
| amused | amused | 1,380 | 2.7% | ✗ |
| frustrated | frustrated · frustration | 719 | 1.4% | ✗ |
| excited | excited · excitement | 714 | 1.4% | ✗ |
| other | other | 609 | 1.2% | ✗ |
| calm | calm | 171 | 0.3% | ✗ |
| **합계** | | **50,284** | **100%** | |

### 4. eval 대응 요약

| 구분 | 샘플 수 | 비율 |
|------|--------:|-----:|
| eval 7-class 대응 (anger/disgust/fear/happiness/neutral/sadness/surprise) | 45,284 | 90.1% |
| no emotion (neutral 인접, 불명확) | 3,416 | 6.8% |
| eval 미등장 (amused/calm/excited/excitement/frustrated/frustration/other/sleepy) | 3,584 | 7.1% |
| **합계** | **50,284** | |

**주요 관찰:**
- disgust(14.5%)·surprise(11.7%)·fear(11.5%) 등 희귀 class가 balanced sampling으로 과대표집 — eval raw 분포(MELD: fear 1.9%, disgust 2.6%)와 큰 차이
- happiness 개념이 3가지 표기(happiness/happy/joy)로 분산 — 모델이 동의어임을 텍스트 레벨에서 학습하는지 불명확
- no emotion(DailyTalk 전용, 6.8%) — neutral과 다른 개념 가능성; eval에 직접 대응 class 없음
- eval 미등장 class 7.1% — 학습 신호 분산, eval 성능에 간접적 영향

---

## 요약

| 항목 | 내용 |
|------|------|
| Train 총 샘플 | 50,284 |
| Train 분포 | **class-balanced** (dataset 내 균등) |
| Train label 체계 | dataset별 독립 MCQ (통합 taxonomy 없음) |
| Eval 총 샘플 | 4,115 (MELD 2,608 + IEMOCAP 1,507) |
| Eval 분포 | **raw** (MELD neutral 48%, IEMOCAP happy 41%) |
| 공통 eval class | angry, happy/joy, neutral, sad (4개) |
| IEMOCAP train→eval class 축소 | 10-class → 4-class (49% 학습 샘플이 eval 미대응) |
| 주요 위험 | balanced train → skewed eval 분포 괴리, label name 불통일 |
