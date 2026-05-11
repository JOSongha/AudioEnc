# Emotion Dataset Survey

---

## 전체 교차표 (Dataset × Emotion Class)

> 값: `count (비율%)`. 비율은 **해당 row(dataset+split)의 전체 대비**.  
> Total row 비율은 85,022 전체 대비.  
> ESD = 영어 화자 10명 only (spk 01-10). MELD tr+dev 분리 기재.  
> JL-Corpus fine-grained class (anxious/apologetic/assertive/concerned/encouraging) 포함.

| Dataset | Split | Anger | Disgust | Fear | Happiness | Neutral | Sadness | Surprise | Amused | Anxious | Apologetic | Assertive | Calm | Concerned | Encouraging | Excited | Frustrated | Other | Sleepy | **Total** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IEMOCAP | train (S1-4) | 933 (15.9%) | 2 (0.0%) | 30 (0.5%) | 452 (7.7%) | 1,324 (22.5%) | 839 (14.3%) | 89 (1.5%) | - | - | - | - | - | - | - | 742 (12.6%) | 1,468 (25.0%) | 3 (0.1%) | - | **5,882** |
| IEMOCAP | test (S5, 4-class) | 197 (13.1%) | - | - | 613 (40.7%) | 386 (25.6%) | 311 (20.6%) | - | - | - | - | - | - | - | - | - | - | - | - | **1,507** |
| MELD | train | 1,109 (11.1%) | 271 (2.7%) | 268 (2.7%) | 1,743 (17.4%) | 4,710 (47.2%) | 683 (6.8%) | 1,205 (12.1%) | - | - | - | - | - | - | - | - | - | - | - | **9,989** |
| MELD | dev | 153 (13.8%) | 22 (2.0%) | 40 (3.6%) | 163 (14.7%) | 469 (42.3%) | 111 (10.0%) | 150 (13.5%) | - | - | - | - | - | - | - | - | - | - | - | **1,108** |
| MELD | test | 345 (13.2%) | 67 (2.6%) | 50 (1.9%) | 401 (15.4%) | 1,256 (48.2%) | 208 (8.0%) | 281 (10.8%) | - | - | - | - | - | - | - | - | - | - | - | **2,608** |
| DailyTalk | train (all) | 159 (0.7%) | 66 (0.3%) | 18 (0.1%) | 3,856 (16.2%) | 18,966 (79.8%) | 301 (1.3%) | 407 (1.7%) | - | - | - | - | - | - | - | - | - | - | - | **23,773** |
| EmoV-DB | train (all) | 1,268 (18.4%) | 1,019 (14.8%) | - | - | 1,568 (22.7%) | - | - | 1,317 (19.1%) | - | - | - | - | - | - | - | - | - | 1,721 (25.0%) | **6,893** |
| RAVDESS | train (all) | 192 (13.3%) | 192 (13.3%) | 192 (13.3%) | 192 (13.3%) | 96 (6.7%) | 192 (13.3%) | 192 (13.3%) | - | - | - | - | 192 (13.3%) | - | - | - | - | - | - | **1,440** |
| MUStARD++ | train (all) | 53 (4.4%) | 29 (2.4%) | 23 (1.9%) | 244 (20.3%) | 438 (36.5%) | 149 (12.4%) | 101 (8.4%) | - | - | - | - | - | - | - | 115 (9.6%) | 48 (4.0%) | - | - | **1,200** |
| CREMA-D | train | 914 (15.4%) | 559 (9.4%) | 623 (10.5%) | 355 (6.0%) | 3,199 (53.7%) | 303 (5.1%) | - | - | - | - | - | - | - | - | - | - | - | - | **5,953** |
| CREMA-D | test | 239 (16.1%) | 125 (8.4%) | 170 (11.4%) | 76 (5.1%) | 812 (54.5%) | 67 (4.5%) | - | - | - | - | - | - | - | - | - | - | - | - | **1,489** |
| SAVEE | all (no split) | 60 (12.5%) | 60 (12.5%) | 60 (12.5%) | 60 (12.5%) | 120 (25.0%) | 60 (12.5%) | 60 (12.5%) | - | - | - | - | - | - | - | - | - | - | - | **480** |
| TESS | all (no split) | 400 (14.3%) | 400 (14.3%) | 400 (14.3%) | 400 (14.3%) | 400 (14.3%) | 400 (14.3%) | 400 (14.3%) | - | - | - | - | - | - | - | - | - | - | - | **2,800** |
| ESD | all / EN only (spk 01-10) | 3,500 (20.0%) | - | - | 3,500 (20.0%) | 3,500 (20.0%) | 3,500 (20.0%) | 3,500 (20.0%) | - | - | - | - | - | - | - | - | - | - | - | **17,500** |
| JL-Corpus | all (no split) | 240 (10.0%) | - | - | 240 (10.0%) | 240 (10.0%) | 240 (10.0%) | - | - | 240 (10.0%) | 240 (10.0%) | 240 (10.0%) | - | 240 (10.0%) | 240 (10.0%) | 240 (10.0%) | - | - | - | **2,400** |
| MSP-IMPROV | all (no split) | 792 (9.4%) | - | - | 2,167 (25.7%) | 3,477 (41.2%) | 2,002 (23.7%) | - | - | - | - | - | - | - | - | - | - | - | - | **8,438** |
| CMU-MOSEI ★ | all (22,856 utter.) | 4,600★ | 3,755★ | 1,803★ | 10,752★ | - | 5,601★ | 2,055★ | - | - | - | - | - | - | - | - | - | - | - | **22,856★** |
| EmoVoice-DB † | all (no split) | 3,486 (15.8%) | 2,950 (13.3%) | 2,961 (13.4%) | 3,269 (14.8%) | 3,188 (14.4%) | 3,174 (14.4%) | 3,072 (13.9%) | - | - | - | - | - | - | - | - | - | - | - | **22,100** |
| eNTERFACE'05 | all (no split) | 194 (16.6%) | 195 (16.7%) | 194 (16.6%) | 195 (16.7%) | - | 194 (16.6%) | 194 (16.6%) | - | - | - | - | - | - | - | - | - | - | - | **1,166** |
| **Total (SL)** | | **14,234 (12.2%)** | **5,957 (5.1%)** | **5,029 (4.3%)** | **17,926 (15.4%)** | **44,149 (37.8%)** | **12,734 (10.9%)** | **9,651 (8.3%)** | 1,317 (1.1%) | 240 (0.2%) | 240 (0.2%) | 240 (0.2%) | 192 (0.2%) | 240 (0.2%) | 240 (0.2%) | 1,097 (0.9%) | 1,516 (1.3%) | 3 (0.0%) | 1,721 (1.5%) | **116,726** |

> ★ CMU-MOSEI: 다중 레이블 (한 발화에 복수 감정), 값 = label count (합계 > utterance 수), Total SL에 미포함.  
> † EmoVoice-DB: GPT-4o-audio 합성 데이터.

**주요 관찰 (18 datasets, SL 116,726 + CMU-MOSEI 22,856 utter.):**
- `neutral` 37.8% (44,149) — DailyTalk 18,966이 단독으로 전체 16% 차지 (신규 데이터 합산으로 비중 감소)
- `fear` 4.3% (5,029) — EmoVoice-DB 2,961 + CREMA-D 623이 전체 fear의 71% 담당
- `disgust` 5.1% (5,957) — EmoVoice-DB 2,950이 49% 담당
- Core 7 class = 109,441 (93.8%)
- EmoVoice-DB(22,100) + DailyTalk(23,773) + ESD-EN(17,500) = SL 전체의 54%
- CMU-MOSEI happiness label 10,752★ — 전체 happiness 관련 최대 단일 소스

---

## 데이터셋별 상세

### v6 학습 풀 포함

#### IEMOCAP — leave-session-out: S1-4 train / S5 test

- HF: 별도 다운로드, `/mnt/tmp/datasets/emotion_raw/IEMOCAP/` (jos), `/mnt/ddn/kyudan/IEMOCAP/` (세현)
- 10 actors × 5 sessions × improvisational + scripted
- **eval 프로토콜**: 4-class (angry/happy/neutral/sad), excited→happy merge, frustrated/surprise/fear silently drop
- ⚠️ Train 최다 class `frustrated` (25.0%)가 eval에서 제외 — 학습·평가 프로토콜 불일치

| Split | 감정 | 샘플 수 | 비율 | mean sec |
|-------|------|--------:|-----:|---------:|
| train S1-4 | frustrated | 1,468 | 25.0% | N/A (nubes) |
| | neutral | 1,324 | 22.5% | |
| | angry | 933 | 15.9% | |
| | sad | 839 | 14.3% | |
| | excited | 742 | 12.6% | |
| | happy | 452 | 7.7% | |
| | surprise | 89 | 1.5% | |
| | fear | 30 | 0.5% | |
| | other/disgust | 5 | 0.1% | |
| | **합계** | **5,882** | | |
| test S5 | happy | 613 | 40.7% | 4.2s |
| | neutral | 386 | 25.6% | 3.8s |
| | sad | 311 | 20.6% | 5.0s |
| | angry | 197 | 13.1% | 4.5s |
| | **합계** | **1,507** | | **4.3s** |

---

#### MELD — official train / dev / test

- HF: `TwinkStart/MELD`, 로컬 `/mnt/tmp/datasets/emotion_raw/MELD/`
- Friends TV 대화 1400+ dialogues, 13,000+ utterances
- v6: train+dev 학습, test held-out
- ⚠️ neutral 47-48%로 전 split 일관 편향. MELD test 305s 이상치 MP4 1개 존재

| Split | 감정 | 샘플 수 | 비율 | mean sec |
|-------|------|--------:|-----:|---------:|
| train | neutral | 4,710 | 47.2% | — |
| | joy | 1,743 | 17.4% | |
| | surprise | 1,205 | 12.1% | |
| | anger | 1,109 | 11.1% | |
| | sadness | 683 | 6.8% | |
| | disgust | 271 | 2.7% | |
| | fear | 268 | 2.7% | |
| | **합계** | **9,989** | | |
| dev | neutral | 469 | 42.3% | 3.0s |
| | joy | 163 | 14.7% | 3.1s |
| | anger | 153 | 13.8% | 3.5s |
| | surprise | 150 | 13.5% | 2.7s |
| | sadness | 111 | 10.0% | 4.0s |
| | fear | 40 | 3.6% | 2.3s |
| | disgust | 22 | 2.0% | 3.1s |
| | **합계** | **1,108** | | **3.0s** |
| test | neutral | 1,256 | 48.2% | 3.1s |
| | joy | 401 | 15.4% | 3.2s |
| | anger | 345 | 13.2% | 3.4s |
| | surprise | 281 | 10.8% | 2.7s |
| | sadness | 208 | 8.0% | 3.7s |
| | disgust | 67 | 2.6% | 3.9s |
| | fear | 50 | 1.9% | 4.0s |
| | **합계** | **2,608** | | **3.2s** |

MP4 경로: `/mnt/tmp/datasets/emotion_raw/MELD/MELD.Raw/output_repeated_splits_test/`

---

#### DailyTalk — no canonical split, v6 전량 train

- HF: `MYJOKERML/dailytalk-dialogue-preference`, 로컬 추출 완료
- Male/Female 화자. Mean duration 3.2s/clip
- ⚠️ `no emotion` 79.8% 압도적 → 다른 class 학습 신호 희석 우려

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

---

#### EmoV-DB — no canonical split, v6 전량 train

- HF: `CLAPv2/EmoV_DB`, 로컬 추출 완료
- 화자 5명 (Bea/Jenie/Josh/Sam/Noel), 일부 감정 미보유

| 감정 | 샘플 수 | 비율 | mean sec |
|------|--------:|-----:|---------:|
| sleepy | 1,721 | 25.0% | 5.7s |
| neutral | 1,568 | 22.7% | 3.9s |
| amused | 1,317 | 19.1% | 5.2s |
| angry | 1,268 | 18.4% | 4.4s |
| disgusted | 1,019 | 14.8% | 5.8s |
| **합계** | **6,893** | | **4.9s** |

---

#### RAVDESS — no canonical split, v6 전량 train

- HF: `xbgoose/ravdess`, 로컬 추출 완료
- 24 professional actors (남12/여12), 두 문장 neutral North American accent
- 완벽 균형 설계, 발화 길이 매우 균일 (3.5-3.9s)

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

---

#### MUStARD++ — no canonical split, v6 전량 train

- nubes-only (로컬 파일 없음). Sarcasm corpus.
- neutral 36.5% 지배, 희귀 class (fear 1.9%, disgust 2.4%) 포함

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
| **합계** | **1,200** | | ~4.7s (논문) |

---

### Eval 전용 (학습 풀 미포함)

#### CREMA-D — train/test split 있음

- HF: `AbstractTTS/CREMA-D`, 로컬 cache: `/mnt/ddn/.cache/huggingface/hub/datasets--MahiA--CREMA-D`
- 91명 (남48/여43), 20-74세, 다인종
- ⚠️ neutral 53-54% 편향. scripted speech → 발화 매우 짧음 (~2.5s)

| Split | 감정 | 샘플 수 | 비율 | mean sec |
|-------|------|--------:|-----:|---------:|
| train | neutral | 3,199 | 53.7% | 2.5s |
| | anger | 914 | 15.4% | 2.6s |
| | fear | 623 | 10.5% | 2.5s |
| | disgust | 559 | 9.4% | 2.8s |
| | happy | 355 | 6.0% | 2.4s |
| | sad | 303 | 5.1% | 2.8s |
| | **합계** | **5,953** | | **2.5s** |
| test | neutral | 812 | 54.5% | 2.5s |
| | anger | 239 | 16.1% | 2.7s |
| | fear | 170 | 11.4% | 2.6s |
| | disgust | 125 | 8.4% | 2.8s |
| | happy | 76 | 5.1% | 2.3s |
| | sad | 67 | 4.5% | 2.9s |
| | **합계** | **1,489** | | **2.6s** |

---

#### SAVEE — no canonical split

- HF: `AbstractTTS/SAVEE`, 캐시 `/mnt/tmp/cache/huggingface`
- 남성 4명, 27-31세, 영국 영어. neutral 2배 설계 (120 vs 60)

| 감정 | 샘플 수 | 비율 | mean sec |
|------|--------:|-----:|---------:|
| neutral | 120 | 25.0% | 3.6s |
| anger | 60 | 12.5% | 3.7s |
| disgust | 60 | 12.5% | 4.0s |
| fear | 60 | 12.5% | 3.7s |
| happiness | 60 | 12.5% | 3.8s |
| sadness | 60 | 12.5% | 4.5s |
| surprise | 60 | 12.5% | 3.8s |
| **합계** | **480** | | **3.8s** |

---

#### TESS — no canonical split

- HF: `Bill13579/TESS-mirror`, 캐시 `/mnt/tmp/cache/huggingface`
- 여성 2명 (OAF 64세, YAF 26세), 캐나다 영어. 단어 1개 읽기 구조 → 매우 짧음

| 감정 | 샘플 수 | 비율 | mean sec |
|------|--------:|-----:|---------:|
| anger | 400 | 14.3% | 1.8s |
| disgust | 400 | 14.3% | 2.4s |
| fear | 400 | 14.3% | 1.7s |
| happiness | 400 | 14.3% | 2.0s |
| neutral | 400 | 14.3% | 2.1s |
| pleasant surprise | 400 | 14.3% | 2.0s |
| sadness | 400 | 14.3% | 2.4s |
| **합계** | **2,800** | | **2.1s** |

---

#### ESD — no canonical split (English only)

- HF: `sonchuate/ESD_dataset`, ZIP 내부 접근, 캐시 `/mnt/tmp/cache/huggingface`
- 영어 화자 10명 (spk 0001-0010), 중국어 10명 (spk 0011-0020)
- 완벽 균형. Sad 4.0s로 가장 길고 Angry 2.7s로 가장 짧음

| 감정 | 샘플 수 (EN) | 비율 | mean sec (EN) |
|------|------------:|-----:|--------------:|
| Angry | 3,500 | 20.0% | 2.7s |
| Happy | 3,500 | 20.0% | 2.8s |
| Neutral | 3,500 | 20.0% | 3.2s |
| Sad | 3,500 | 20.0% | 4.0s |
| Surprise | 3,500 | 20.0% | 3.3s |
| **합계 (EN)** | **17,500** | | **3.2s** |
| **합계 (EN+ZH)** | **35,000** | | **3.0s** |

---

#### JL-Corpus — no canonical split

- HF: `CLAPv2/JL-Corpus`, 캐시 `/mnt/tmp/cache/huggingface`
- New Zealand English, 5 primary + 5 secondary emotions = 10-class
- ⚠️ anxious/apologetic/assertive/concerned/encouraging 등 미세 감정 → 타 데이터셋과 class 매핑 주의

| 감정 | 샘플 수 | 비율 | mean sec |
|------|--------:|-----:|---------:|
| angry | 240 | 10.0% | 2.0s |
| anxious | 240 | 10.0% | 2.2s |
| apologetic | 240 | 10.0% | 2.1s |
| assertive | 240 | 10.0% | 2.1s |
| concerned | 240 | 10.0% | 2.2s |
| encouraging | 240 | 10.0% | 2.0s |
| excited | 240 | 10.0% | 2.1s |
| happy | 240 | 10.0% | 2.0s |
| neutral | 240 | 10.0% | 2.1s |
| sad | 240 | 10.0% | 2.2s |
| **합계** | **2,400** | | **2.1s** |

---

#### MSP-IMPROV — no canonical split

- 출처: lab-msp.com (학술 등록 후 다운로드, HF 미등록)
- 12 actors (6M/6F), 652 target sentences × dyadic interaction, 총 8,438 발화
- ⚠️ neutral 41.2% 다수. 공식 고정 split 없음 → leave-speaker-out 권장
- ⚠️ acted dyadic 구조 (시나리오 기반) → IEMOCAP과 유사 특성

| 감정 | 샘플 수 | 비율 |
|------|--------:|-----:|
| neutral | 3,477 | 41.2% |
| happiness | 2,167 | 25.7% |
| sadness | 2,002 | 23.7% |
| anger | 792 | 9.4% |
| **합계** | **8,438** | |

---

#### CMU-MOSEI — official train/valid/test split ★ 다중 레이블

- 출처: CMU Multicomp Lab (multicomp.cs.cmu.edu, HF 비공식 미러 존재)
- 1,000명 YouTube 스피커, 3,228 비디오, 멀티모달 (audio+video+text)
- **공식 split**: train 14,524 / valid 1,765 / test 4,188 (총 20,477 annotated)
- ⚠️ **다중 레이블**: 한 발화에 복수 감정 태그, label count 합 > utterance count
- ⚠️ 감정 점수 0-3 continuous → binary 변환(>=1) 시 아래 카운트
- ⚠️ neutral 없음 — 6 Ekman emotions만 주석

| 감정 | label count (all) | 비율 (★ sum > 100%) |
|------|------------------:|------:|
| happiness | 10,752 | 47.1%★ |
| sadness | 5,601 | 24.5%★ |
| anger | 4,600 | 20.1%★ |
| disgust | 3,755 | 16.4%★ |
| surprise | 2,055 | 9.0%★ |
| fear | 1,803 | 7.9%★ |
| **총 utterances** | **22,856** | |

---

#### EmoVoice-DB — no canonical split  † GPT-4o 합성

- HF: `yhaha/EmoVoice-DB`, MIT License
- ⚠️ **GPT-4o-audio 생성 합성 데이터** — 실제 인간 화자 녹음 아님
- 완벽 균형 설계. 총 40.45h → mean ~6.6s/clip (타 데이터셋 대비 긺)
- eval 사용 시 train-test 분포 동질성 주의 (동일 생성 분포)

| 감정 | 샘플 수 | 비율 | mean sec |
|------|--------:|-----:|---------:|
| angry | 3,486 | 15.8% | ~6.6s |
| neutral | 3,188 | 14.4% | ~6.6s |
| happy | 3,269 | 14.8% | ~6.6s |
| sad | 3,174 | 14.4% | ~6.6s |
| surprised | 3,072 | 13.9% | ~6.6s |
| fearful | 2,961 | 13.4% | ~6.6s |
| disgusted | 2,950 | 13.3% | ~6.6s |
| **합계** | **22,100** | | **~6.6s** |

---

#### eNTERFACE'05 — no canonical split

- 출처: enterface.net (공개 연구용, HF 미등록)
- 42명 (14개국, 34남/8녀), 6가지 시나리오 영상 기반 acted
- 48kHz 스테레오 16-bit 오디오 (AVI 컨테이너)
- 균형 설계 (~194-195/class), 1,260 중 1,166 유효 (94개 품질 불합격)

| 감정 | 샘플 수 | 비율 |
|------|--------:|-----:|
| anger | 194 | 16.6% |
| disgust | 195 | 16.7% |
| fear | 194 | 16.6% |
| happiness | 195 | 16.7% |
| sadness | 194 | 16.6% |
| surprise | 194 | 16.6% |
| **합계** | **1,166** | |

---

#### MSP-Podcast — 조사 보류

- 출처: lab-msp.com (등록 필요, ~409시간 podcast naturalistic speech)
- 9 discrete classes + continuous (valence/activation/dominance), 100K+ utterances
- 규모 최대급이나 접근 절차 복잡 → 향후 대규모 학습 시 재검토

---

#### MLEnd — 조사 보류

- `pip install mlend` 패키지 필요, 오디오는 GitHub 별도 다운로드
- 감정 class가 Neutral/Bored/Excited/Question → **억양(intonation) 레이블**
- 일반 emotion benchmark 목적과 성격 다름 → 추후 prosody 분석 시 재검토

---

## 데이터셋 요약

| 데이터셋 | v6 train | split | 총 샘플 | classes | mean sec | eval 가능 |
|---------|:--------:|-------|--------:|--------:|---------:|:--------:|
| IEMOCAP | ✓ S1-4 | S5=test | 5,882 / 1,507 | 10→4 eval | 4.3s (test) | ✓ |
| MELD | ✓ tr+dev | official | 9,989 / 1,108 / 2,608 | 7 | 3.0-3.2s | ✓ |
| DailyTalk | ✓ all | — | 23,773 | 7 | 3.3s | ✗ |
| EmoV-DB | ✓ all | — | 6,893 | 5 | 4.9s | ✗ |
| RAVDESS | ✓ all | — | 1,440 | 8 | 3.7s | ✗ |
| MUStARD++ | ✓ all | — | 1,200 | 9 | ~4.7s | ✗ |
| **CREMA-D** | ✗ | tr/te | 5,953 / 1,489 | 6 | 2.5s | **✓** |
| **SAVEE** | ✗ | — | 480 | 7 | 3.8s | **✓** |
| **TESS** | ✗ | — | 2,800 | 7 | 2.1s | **✓** |
| **ESD** | ✗ | — | 17,500 (EN) | 5 | 3.2s | **✓** |
| **JL-Corpus** | ✗ | — | 2,400 | 10 | 2.1s | **✓** |
| **MSP-IMPROV** | ✗ | — (no fixed) | 8,438 | 4 | N/A | **✓** |
| **CMU-MOSEI ★** | ✗ | tr/va/te | 22,856 utter. | 6 multi-label | N/A | **✓** |
| **EmoVoice-DB †** | ✗ | — | 22,100 | 7 (synthetic) | 6.6s | △ |
| **eNTERFACE'05** | ✗ | — | 1,166 | 6 | N/A | **✓** |
| MSP-Podcast | ✗ | — | ~100K+ | 9+cont. | varies | 조사 보류 |
| MLEnd | ✗ | — | ~32,654 | 4 (억양) | — | △ |

> ★ CMU-MOSEI: 다중 레이블. † EmoVoice-DB: GPT-4o 합성.

---

## Train / Test 분할 기준 분포

> 시각화: `docs/analysis/emotion_split_viz.png` (2026-05-10 기준, 18 datasets)

**분할 규칙**
- **Train** = (split 없는 데이터 전량) + (split 있는 데이터의 train portion)  
- **Test** = split 있는 데이터의 test portion 만  
- CMU-MOSEI(★ multi-label)는 단일 레이블과 성격이 달라 별도 집계  

| 그룹 | 구성 (단일 레이블) | 구성 (multi-label 별도) |
|------|------|------|
| Train SL | IEMOCAP S1-4 · MELD tr+dev · DailyTalk · EmoV-DB · RAVDESS · MUStARD++ · CREMA-D train · SAVEE · TESS · ESD(EN) · JL-Corpus · **MSP-IMPROV · EmoVoice-DB† · eNTERFACE'05** | CMU-MOSEI tr+val 16,289 utter. |
| Test SL | IEMOCAP S5 (4-class) · MELD test · CREMA-D test | CMU-MOSEI test 4,188 utter. |

### Train — 단일 레이블 (총 111,119)

| 감정 | 샘플 수 | 비율 |
|------|--------:|-----:|
| neutral | 41,695 | 37.5% |
| happiness | 16,836 | 15.2% |
| anger | 13,453 | 12.1% |
| sadness | 12,148 | 10.9% |
| surprise | 9,370 | 8.4% |
| disgust | 5,765 | 5.2% |
| fear | 4,809 | 4.3% |
| sleepy | 1,721 | 1.5% |
| frustrated | 1,516 | 1.4% |
| amused | 1,317 | 1.2% |
| excited | 1,097 | 1.0% |
| anxious | 240 | 0.2% |
| apologetic | 240 | 0.2% |
| assertive | 240 | 0.2% |
| concerned | 240 | 0.2% |
| encouraging | 240 | 0.2% |
| calm | 192 | 0.2% |
| **합계** | **111,119** | |

> + CMU-MOSEI (★ multi-label): train+val 16,289 utterances  
> (happy 8,548 / sad 4,453 / anger 3,657 / disgust 2,985 / surprise 1,634 / fear 1,434 label counts)

**주요 관찰 (train)**
- neutral 37.5% — 신규 데이터 추가로 44.1% → 37.5%로 편향 완화
- fear 4.3% (4,809) — EmoVoice-DB 2,961이 가장 큰 기여 (61.6%)
- disgust 5.2% (5,765) — EmoVoice-DB 2,950이 51.2% 담당
- core 7 = 104,076 (93.7%)
- 비-core 7 클래스 7,043개는 전량 train에만 존재 (JL-Corpus fine-grained 포함)

### Test — 단일 레이블 (총 5,604) — core 7 only

| 감정 | 샘플 수 | 비율 | 출처 |
|------|--------:|-----:|------|
| neutral | 2,454 | 43.8% | MELD 1,256 · IEMOCAP 386 · CREMA-D 812 |
| happiness | 1,090 | 19.5% | IEMOCAP 613 · MELD 401 · CREMA-D 76 |
| anger | 781 | 13.9% | MELD 345 · CREMA-D 239 · IEMOCAP 197 |
| sadness | 586 | 10.5% | MELD 208 · IEMOCAP 311 · CREMA-D 67 |
| surprise | 281 | 5.0% | MELD 281 (IEMOCAP·CREMA-D 없음) |
| fear | 220 | 3.9% | CREMA-D 170 · MELD 50 (IEMOCAP 없음) |
| disgust | 192 | 3.4% | CREMA-D 125 · MELD 67 (IEMOCAP 없음) |
| **합계** | **5,604** | | |

> + CMU-MOSEI (★ multi-label): test 4,188 utterances  
> (happy 1,970 / sad 1,026 / anger 843 / disgust 688 / surprise 377 / fear 330 label counts)

**주요 관찰 (test)**
- Test set 자체는 변경 없음 (신규 datasets에 공식 test split 없거나 별도 집계)
- train SL 대비 test SL 비율 ≈ 20 : 1 (기존 14:1 → train 규모 증가로 확대)
- surprise·fear·disgust는 여전히 MELD·CREMA-D 의존도 집중

---

## Train → Test 분포 매칭 옵션 분석

> ⚠️ **주의: 이 섹션은 raw 데이터셋 통계 기준 분석임**  
> 실제 v6 학습 매니페스트(`/mnt/ddn/users/sehyun/datasets/manifests/v6/audio_emotion/`)를 확인한 결과,  
> v6 train은 **dataset 내 class-balanced sampling**이 적용되어 있어 아래 raw 분포 분석과 다름.  
> v6 실제 구성 분석은 **`docs/emotion_datasets_v6.md`** 참조.

### 현재 Train(core-7) vs Test 분포 괴리

> Train core-7 합계: 104,076 (전체 SL 111,119에서 non-core-7 7,043 제외)  
> ※ 아래 수치는 raw 데이터셋 기준 (실제 학습 분포 아님)

| 감정 | Train core-7 | Test | 차이 | 방향 |
|------|-------------:|-----:|-----:|:----:|
| neutral | 41,695 (40.1%) | 43.8% | −3.7% | 부족 |
| happiness | 16,836 (16.2%) | 19.5% | −3.3% | **부족** |
| anger | 13,453 (12.9%) | 13.9% | −1.0% | 부족 |
| sadness | 12,148 (11.7%) | 10.5% | +1.2% | 미미 과잉 |
| surprise | 9,370 (9.0%) | 5.0% | **+4.0%** | ⚠️ 과잉 |
| disgust | 5,765 (5.5%) | 3.4% | **+2.1%** | ⚠️ 과잉 |
| fear | 4,809 (4.6%) | 3.9% | +0.7% | 미미 과잉 |

**구조적 문제:** surprise (+4%), disgust (+2.1%) 과잉 / happiness, neutral 비율 부족  
원인: ESD(surprise 3,500), EmoVoice-DB(surprise 3,072 · disgust 2,950)가 두 class를 집중 팽창

---

### Option A — Hard Undersample

각 class에서 `T × p_test ≤ N_available` 을 만족하는 최대 T 산출:

```
neutral  : 41,695 / 0.438 =  95,194
happiness: 16,836 / 0.195 =  86,338  ← bottleneck (happiness 전량 소진)
anger    : 13,453 / 0.139 =  96,784
sadness  : 12,148 / 0.105 = 115,695
surprise :  9,370 / 0.050 = 187,400
fear     :  4,809 / 0.039 = 123,308
disgust  :  5,765 / 0.034 = 169,559

T_max = 86,338
```

| 감정 | 사용량 | 폐기량 | 사용률 |
|------|-------:|-------:|------:|
| neutral | 37,816 | 3,879 | 90.7% |
| happiness | 16,836 | 0 | 100% |
| anger | 12,001 | 1,452 | 89.2% |
| sadness | 9,066 | 3,082 | 74.6% |
| surprise | 4,317 | 5,053 | 46.1% |
| fear | 3,367 | 1,442 | 70.0% |
| disgust | 2,936 | 2,829 | 50.9% |
| **합계** | **86,338** | **17,738** | **83.0%** |

- ✅ test 분포와 정확히 일치
- ❌ core-7에서 17,738개(17.1%) 폐기, surprise·disgust 절반 버림
- ❌ non-core-7 7,043개는 별도 처리 필요

---

### Option B — Class-Weighted Loss

데이터 전량(111,119) 유지, loss에 `weight = p_test / p_train` 적용:

| 감정 | p_test | p_train | weight |
|------|-------:|--------:|-------:|
| happiness | 0.195 | 0.162 | **1.20** |
| neutral | 0.438 | 0.401 | 1.09 |
| anger | 0.139 | 0.129 | 1.08 |
| fear | 0.039 | 0.046 | 0.85 |
| sadness | 0.105 | 0.117 | 0.90 |
| disgust | 0.034 | 0.055 | 0.62 |
| surprise | 0.050 | 0.090 | **0.56** |

max/min 비율 = 1.20 / 0.56 = **2.16** → 비율 완만, 학습 안정성 양호

- ✅ 데이터 손실 없음, 구현 간단 (`torch.nn.CrossEntropyLoss(weight=...)`)
- ❌ 분포 매칭이 soft — surprise 과잉이 그라디언트에 잔존

---

### Option C — EmoVoice-DB 제외 (synthetic 정제)

합성 데이터 제거 후 A 적용 시 효과:

```
surprise: 9,370 − 3,072 = 6,298  (test 비율에 근접)
disgust : 5,765 − 2,950 = 2,815  ≈ 목표치 달성
fear    : 4,809 − 2,961 = 1,848  ← 새 bottleneck

새 T_max = 1,848 / 0.039 = 47,385
```

- ⚠️ 합성 제거 대가로 train 크기 86K → 47K (-45%)
- ⚠️ EmoVoice-DB가 fear의 61.6%를 공급 — 제거 시 fear가 새 병목으로 전환
- **결론: EmoVoice-DB는 합성이지만 fear 공급원으로서 실질적 기여, 제거 비권장**

---

### Option D — A + non-core-7 분리 (권장 후보)

1. core-7 classes: Hard undersample (T = 86,338)
2. non-core-7 classes (7,043): 별도 auxiliary head 또는 multi-task loss

총 학습 데이터: 86,338 (core-7, test-matched) + 7,043 (non-core-7 별도)

- ✅ test 분포 정확 매칭
- ✅ non-core-7 정보 보존
- ❌ 모델 구조 복잡도 증가

---

### Option E — Curriculum Sampling

- **1단계** (warmup): test-matched 분포 (T = 86,338)로 수렴
- **2단계** (fine-tune): 전체 111K로 long-tail class 추가 학습
- non-core-7 class는 2단계에만 노출

- ✅ 데이터 손실 없음, 분포 편향 완화 효과
- ❌ 스케줄 튜닝 필요, 재현성 관리 복잡

---

### 옵션 비교 요약

| Option | train 크기 | 분포 정확도 | 데이터 손실 | 구현 복잡도 |
|--------|----------:|:---------:|:----------:|:----------:|
| A: Hard undersample | 86,338 | 정확 | 17.1% | 낮음 |
| B: Weighted loss | 111,119 | 근사 | 없음 | 낮음 |
| C: EmoVoice 제외 + A | 47,385 | 정확 | 54.4% | 낮음 |
| D: A + non-core 분리 | 86,338 + 7,043 | 정확 | core만 | 중간 |
| E: Curriculum | 86K→111K | 단계적 | 없음 | 높음 |

**핵심 제약:** happiness(16,836개)가 병목 — 이 수치를 늘리지 않는 한 test-matched train 상한 = **86,338**  
**현실적 권장:** 실험 우선순위 B(구현 즉시) → A(정확 기준선) → E(best effort)
