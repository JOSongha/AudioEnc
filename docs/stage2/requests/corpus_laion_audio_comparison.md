# LAION-Audio-630K vs LAION-Audio-300M — 코퍼스 질 비교 보고서

작성: 2026-04-30
용도: Stage 2 v3+ env-sound / general-audio mix 후보 검토. 두 LAION 공개 audio-text 코퍼스 중 어느 쪽을 (또는 어떤 비율로) 활용할지 결정 근거.

출처:
- LAION-Audio-630K: <https://github.com/LAION-AI/audio-dataset/blob/main/laion-audio-630k/README.md>
- LAION-Audio-300M: <https://huggingface.co/datasets/laion/LAION-Audio-300M>

---

## 1. 한눈에 보기

| 차원 | **630K** | **300M** |
|---|---|---|
| Pair 수 | 633,526 | ~229,000,000 |
| 총 시간 | 4,325 h | (미공개, 29 TB) |
| 주 출처 | SFX 라이브러리 10종 (curated) | YouTube segment (web-crawl) |
| 캡션 작성 | 인간 / filename / 일부 T5 | 자동 (모델 미명시) |
| 캡션–오디오 정합 | 강함 (업로더 태그) | 약함 (auto-segment + auto-cap) |
| 도메인 | SFX + 환경음 위주 | 음악 + speech + ambient 혼합 |
| 라이선스 | sub-dataset별 (CC-BY/-NC, CC0, 상용) | Apache 2.0 (단, YouTube 출처) |
| Doc/QC pipeline | 공개 | 사실상 미공개 |

**결론 한 줄**: 질은 **630K 압승**, 양은 **300M 압승**, 본 audiollm-trainer 의 env-sound supervision 에는 **630K 단독 또는 630K-anchor + 300M-augment** 추천.

---

## 2. 배경 — 본 프로젝트에서 왜 비교가 필요한가

[`design.md §6.1`](../design.md) 의 v1/v2 mix 에서 env-sound 가 **35% 비중** 으로 학습:

- v1: FSD50K dev 35,884 + Clotho dev 3,356 + ESC-50 1,760 = **41,000 rows**
- v2: + AudioSet bal_train 18,683 추가 = **65,488 rows**

이 env-sound pool 의 한계:
- 모두 **classification / multi-label tagging** 형식 (FSD50K, AudioSet) 이거나 **짧은 caption** (Clotho ~5,000 only).
- Caption-style supervision 양 자체가 부족 → Clotho BLEU-4 v2 best 0.087 정체.
- 음향 reasoning / open-domain audio understanding 약함.

차기 stage 검토 시 **caption-rich 외부 corpus** 가 필요할 수 있고, LAION 의 두 후보가 자연스러운 대안.

---

## 3. 비교 방법론

다음 6 차원을 정량 + 정성 평가:

1. **캡션 source** — 인간/auto/scrape, alignment 방식
2. **오디오 품질** — studio / 압축 / 노이즈
3. **도메인 분포** — SFX / music / speech / ambient
4. **데이터 위생** — dedup, QC, 명세 공개도
5. **라이선스 / 법적 명확성**
6. **다운스트림 적합도** — CLAP, captioning, classification, MMAU-style reasoning

---

## 4. 차원별 평가

### 4.1 캡션 source 와 alignment

**630K**:
- Freesound (460K~515K): 업로더가 단 1–2개 description (인간).
- BBC SFX (16K), Epidemic (75K), Audiostock (10K + 251K raw): 1–2개 인간 caption.
- 4 purchased SFX libraries (~16K 합): filename 기반.
- Epidemic 일부: T5 모델로 **de-biased caption** 생성 (gendered term → neutral).
- 업로더가 의도적으로 단 캡션이라 **audio ↔ caption 정합 강함**.

**300M**:
- 캡션은 전부 자동 생성. 모델/파이프라인 명시 안 됨.
- 메타 필드 (`channel_follower_count`, `like_count`, `title`, `segment_filename`, `start_time_ms`) → **YouTube 영상 자동 segmentation** 후 segment 별 자동 caption.
- 영상 한 부분에 대한 caption 이 audio segment 와 일치한다는 보장 없음 (예: 영상 후반의 narration 이 앞부분 SFX segment 와 무관).
- 결과: alignment **noise 큼**.

**판정**: 정확한 supervision signal 측면에서 630K 가 **수~수십 배** 우수.

### 4.2 오디오 품질

**630K**:
- Studio-grade SFX 라이브러리 (BBC, Epidemic, Audiostock, Sonniss Game Effects 등) 비중 큼.
- Freesound 도 대부분 의도적으로 녹음/제작된 sample.
- compression artifacts 적고 SNR 높음.

**300M**:
- YouTube 압축 (AAC/Opus 변환된 mp3) → spectral cliff 위 음향 정보 손실.
- 영상 BGM/voiceover/현장 노이즈 혼재.
- Scale 로 평균화는 가능하나 개별 sample quality 는 낮음.

### 4.3 도메인 분포

**630K**:
- ~80% SFX + 환경음 (Freesound + BBC + 구매 SFX 합산).
- 음악 일부 (Audiostock 포함).
- **Speech / dialog 거의 없음** — CLAP 가 전통적으로 음성 task 약한 이유.

**300M**:
- YouTube 분포 그대로. 추정: 음악 (가장 큰 비중) + 강의/대화/내레이션 + game/ambient.
- 멀티링구얼 (Spanish, Hindi 등 caption 에서 관측됨).
- speech 비중 높지만 **align 안 된 narration** 일 가능성 — speech recog 학습엔 부적합.

### 4.4 데이터 위생

**630K**:
- "no overlap" variant 명시 (ESC50/FSD50K/Urbansound8K/Clotho 제거). 평가 leakage 방지 가능.
- sub-dataset 별 row count, 라이선스, 출처 모두 README 에 공개.

**300M**:
- README 사실상 비어있음 (license 한 줄).
- de-dup, near-dup 처리, language 분포, audio 길이 분포 모두 미공개.
- segment 추출 룰, 캡션 모델 비공개 → reproducibility ↓.

### 4.5 라이선스

**630K**:
- Freesound: CC-BY / CC-BY-NC / CC0 / CC Sampling+ 혼재 — sample 단위 메타데이터 추적 필요.
- 구매 SFX (Audiostock, Sonniss 등): 라이선스 unclear, 학습 사용 가능성 회색지대 가능.
- BBC: 비상업 사용 권장.

**300M**:
- 표기 Apache 2.0 (코퍼스 메타) — 그러나 audio 자체는 **YouTube 콘텐츠** → 학습 사용은 회색지대.
- 상업화 모델 학습 시 법무 검토 필수.

→ **둘 다 회색지대 있음**. 학술 사용 한정 시 둘 다 OK, commercial deploy 는 sub-dataset 별 audit 필요.

### 4.6 다운스트림 적합도

| Use case | 630K 적합도 | 300M 적합도 | 코멘트 |
|---|---:|---:|---|
| CLAP / contrastive | ⭐⭐⭐⭐ | ⭐⭐ | 630K 가 caption-noise 적어 batch 작아도 수렴 빠름 |
| Audio captioning (Clotho/AudioCaps eval) | ⭐⭐⭐⭐ | ⭐⭐ | 300M caption noise 가 BLEU/CIDEr loss 시 직접 해 |
| Sound classification (ESC-50 / FSD50K / AS) | ⭐⭐⭐⭐ | ⭐⭐⭐ | 630K SFX 비중이 직결, 300M 은 weak supervision |
| Speech / paralinguistic | ⭐ | ⭐⭐ | 둘 다 부적합. LibriHeavy / Whisper-corpus 권장 |
| Open-domain audio understanding (MMAU 등) | ⭐⭐⭐ | ⭐⭐⭐⭐ | 300M scale 효과 큼, noise 는 large-batch + denoising |
| Scaling pretrain (CLAP-XL 등) | ⭐⭐ | ⭐⭐⭐⭐ | 630K 만으로는 모델 크기 한계, 300M 가 필수 |

---

## 5. 본 프로젝트 적용 시나리오

### 시나리오 A — env-sound supervision 보강 (가장 가능성 큼)

목표: Stage 2 의 env-sound 35% pool 을 더 풍부한 caption-style supervision 으로 강화 → Clotho BLEU-4 / FSD50K mAP 상승 노림.

권장: **630K (no-overlap variant)** 부분 추가.
- ESC-50/FSD50K/Urbansound8K/Clotho 와 dedup 된 460,801 sample 만 추출 가능.
- 본 노드 디스크 추정: ~600–800 GB (4,325 h × 16 kHz mp3 ≈ 200–300 GB compressed; 48 kHz wav 변환 시 1 TB+).
- AudioSet bal_train (18,683 rows) 대신 또는 추가로 도입.
- Caption format 은 omni manifest `caption` 필드로 그대로 사용 가능 ([`scripts/emo/build_combined_manifest.py`](../../../scripts/emo/build_combined_manifest.py) 호환).

### 시나리오 B — 차세대 (Stage 3 / SSL pretrain) 후보

목표: scale-up 으로 일반 음향 understanding 을 키우는 후속 단계.

권장: **300M scale** 을 SSL / contrastive pretrain 에만 활용.
- 29 TB 다운로드는 본 클러스터 ddn 마운트 capacity 초과 가능 — streaming 만 현실적.
- HF `datasets` + webdataset 모드 + `streaming=True` 로 lazy load.
- Caption noise 처리: SigLIP-loss / hard-negative mining / temperature tuning 검토.
- Speech 비중 때문에 supervised fine-tune 직전엔 630K 로 anchor.

### 시나리오 C — Mixed (630K-anchor + 300M-augment)

목표: 본 프로젝트 한정 risk-balanced 옵션.

- Stage 2 v3 mix: env-sound 부분을 `630K-no-overlap : 300M-subset = 4 : 1` 비율 (rough).
- 300M 은 "high-confidence caption" filter 적용 후만 사용 (CLAP score ≥ threshold 같은 후처리).
- 학술 publishing 시 두 셋 별로 ablation 보고.

---

## 6. 위험 요인 / 미확인 사항

1. **300M README 부재** — segment 길이 분포, 캡션 모델 (Whisper? Qwen-Audio? 자체 학습?), language 비율 모두 모름. 실제 활용 전 raw sample 수백 개 inspect 필수.
2. **Caption 언어** — 300M 에 비영어 caption 비율 추정 불가. Stage 2 모델은 영어 SFT 만 했으므로 비영어 caption 은 노이즈.
3. **Copyright cascade** — YouTube 출처 audio 학습이 미국/한국 fair-use 범위 내인지 법무 자문 권장 (특히 상업 deploy 시).
4. **Disk** — 29 TB 는 본 cluster 에 풀 다운로드 불가. streaming pipeline 검증 필요.
5. **Eval contamination** — 300M 에 AudioCaps / Clotho / ESC-50 의 audio 가 YouTube 통해 들어가 있을 가능성 — eval 전 audio-hash dedup 검토.

---

## 7. 권장 다음 액션

| 우선순위 | 액션 | 난이도 |
|---|---|---|
| P1 | 630K-no-overlap (460,801 rows) 다운로드 + manifest 빌드 + ESC-50/FSD50K/Clotho 와 dedup 검증 | 중 (~1일) |
| P2 | 300M 에서 1,000 random sample 다운로드 → caption 언어 / quality / alignment 수동 검수 | 저 (~반일) |
| P3 | Stage 2 v3 smoke run 에 630K 일부 (~10K rows) 섞어서 Clotho BLEU 변화 측정 | 중 (~2일) |
| P4 | 결과 좋으면 v3 full training mix 에 시나리오 C 비율 적용 | 중 |
| P5 | 300M streaming 파이프라인 prototype (HF datasets + webdataset) | 중 |

---

## 8. References

- LAION-CLAP paper (Wu et al. 2023): "Large-scale Contrastive Language-Audio Pretraining" — 630K 설계 의도와 ablation
- LAION-Audio-630K README: <https://github.com/LAION-AI/audio-dataset/blob/main/laion-audio-630k/README.md>
- LAION-Audio-300M card: <https://huggingface.co/datasets/laion/LAION-Audio-300M>
- 본 프로젝트 비교 대상 corpus 정의: [`design.md §6.1`](../design.md), [`leakage_audit.md §6.2`](../leakage_audit.md)
