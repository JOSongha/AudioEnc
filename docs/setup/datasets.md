# Stage-1 학습 데이터 카탈로그 — 시간 (hours) 정리

> **Manifest source**: `/mnt/tmp/datasets/manifests/v6/` (16,216,222 rows total, GigaSpeech 추가 후). v3 / v4 / v5 와 audio_env_sound 풀은 동일, audio_emotion 은 IEMOCAP Sessions 1-4 (5,882 rows) 추가로 47,054 rows, audio_asr 은 GigaSpeech XL 50% random subsample (~4,133,000 rows, 5,000 h target) 추가로 15,478,224 rows. **GigaSpeech sampling 은 planned (manifest rebuild 진행 중)**, § 2 / 변경 이력 참고.
> **방법**: local audio (`audio_path` 존재) → 500 샘플 `soundfile.info()` 평균 × row count. nubes-only (`audio_path: null`, MLS/LibriTTS-R/VoxPopuli) → 공식 논문 / 데이터셋 카드 인용. GigaSpeech 는 HF cache (`/mnt/tmp/cache/gigaspeech/`) 에서 sample probe 또는 published 인용.
> **double-check**: 표준 published 값과 empirical 추정을 양쪽 표시. 일치 안 하는 항목은 § 3 footnote 에 사유.

## 1. 합계

| Modality | Rows | Hours (raw) | Hours (training-exposed, 30 s cap) | 비율 (cap 기준) |
|---|---:|---:|---:|---:|
| `audio_asr` (speech recognition) | 15,478,224 | **~50,787 h** | ~50,787 h | 95.7 % |
| `audio_env_sound` (sound captioning) | 690,944 | **~4,303 h** | **~2,215 h** | 4.2 % |
| `audio_emotion` (emotion MCQA) | 47,054 | **~48 h** | ~48 h | 0.09 % |
| **합계 (Stage-1 v6 학습 풀, GigaSpeech 포함)** | **16,216,222** | **~55,138 h** | **~53,050 h** | |

> Training-exposed = `omni_max_audio_samples` cap (30 s) 적용 후. ASR utterance 평균 짧음 (MLS / LibriTTS-R / VoxPopuli 14.5 s, GigaSpeech XL 4.3 s) 이라 cap 영향 없음 (cap == raw). Emotion utterance 도 짧음 (DT 3.2 s, MELD 3.2 s, IEMOCAP 4.8 s, EmoV-DB 4.9 s, RAVDESS 3.7 s, MUStARD++ 4.7 s 평균. sample 200 / source, seed=11). Sound captioning 만 long ambient (BBC, Freesound 일부) 가 잘려 raw 4,303 h → effective 2,215 h. § 5 참고.

## 2. ASR (audio_asr) — 15,478,224 rows ≈ 50,787 h

| Source | Rows | Hours | Split 여부 | Train | Test | 출처 / 비고 |
|---|---:|---:|---|---|---|---|
| MLS English | 10,808,037 | **44,659 h** | ✓ train / dev / test (canonical) — 현재 train 만 사용 | T | F | Pratap et al. 2020 (MLS paper Table 1, English split) |
| LibriTTS-R | 354,721 | **585 h** | ✓ train.clean.100 / train.clean.360 / train.other.500 / dev.clean / dev.other / test.clean / test.other (corpus 전체). v6 학습은 **train.\* 3 splits 만** (clean-100 33,232 + clean-360 116,454 + other-500 205,035). dev.\* / test.\* 는 LibriSpeech eval 보호 위해 제외 | T | T(LibriSpeech Test-clean/other) | Koizumi et al. 2023 (LibriTTS-R paper, "unknown_asr" 이라는 source 라벨로 라우팅됨) |
| VoxPopuli (English transcribed) | 182,466 | **543 h** | ✓ train / dev / test (canonical) — 현재 train 만 사용 | T | F | Wang et al. 2021 Table 1 (en transcribed split) |
| GigaSpeech (en, XL → 5,000 h sampled) | **4,129,334** (실측) | **~5,000 h** | ✓ XL train / dev / test (canonical). v6 학습은 **XL train 의 seed=11 random 50% subsample** (8,256,276 → 4,129,334 rows, ratio 0.5001). dev (~12 h) / test (~40 h) 는 GigaSpeech-test eval 보호 위해 제외. utterance 평균 4.3 s 라 30 s cap 영향 없음. 빌더: [`build_gigaspeech.py`](../../scripts/manifest_builders/build_gigaspeech.py) (nubes-direct) | T | T(GigaSpeech test, [`eval_asr_external.py`](../../evaluation/stage2/eval_asr_external.py)) | Chen et al. 2021 Table 2 (XL = 10,000 h, 8,266,041 utt). 5,000 h = XL 의 50% random sample (4) |

audio 가 nubes (`hyperscaleai-audiollm/datasets/public/16kHz/...`) 에 있어 직접 probe 안 함 (MLS / LibriTTS-R / VoxPopuli). GigaSpeech 는 HF (`speechcolab/gigaspeech` XL) 다운로드 후 `/mnt/tmp/cache/gigaspeech/` 캐시, sampling 결과 shard `audio_asr/gigaspeech_*.jsonl` 로 v6 에 추가. row 수는 manifest 에서 직접 카운트, hours 는 공식 published × sample ratio (5,000 / 10,000 = 0.5).

4) **GigaSpeech 5,000 h 산정**: XL 공식 10,000 h × 50% sample = 5,000 h (target). row 수 추정 = 8,266,041 × 0.5 = 4,133,020. utterance 평균 4.36 s (paper Table 4) 로 30 s cap 영향 없음, raw == cap30. dev / test 는 canonical 분리 (~12 h dev, ~40 h test) 라 sampling 대상 아님, eval 에서만 사용.

## 3. Sound captioning (audio_env_sound) — 690,944 rows

`audio_path` local probe (500 sample, mean × N) 기반. raw 와 30 s cap 별도 산출.

| Source | Rows | Empirical raw h¹ | Empirical cap-30 h¹ | Published h² | Split 여부 | Train | Test | 비고 |
|---|---:|---:|---:|---:|---|---|---|---|
| LAION-Freesound | 460,141 | 2,737 h | 1,543 h | 2,805 h | ✓ train_1 / train_2 / test (LAION 자체 분할) | T | T | train_1+train_2+test 통합 사용 |
| LAION-BBC | 31,936 | 998 h | 226 h | 463 h | ✓ train / test (LAION 자체 분할) | T | T | train+test 통합 사용 (3) |
| LAION-Epidemic | 75,645 | 231 h | 132 h | 220 h | ✓ train / test (LAION 자체 분할) | T | T | train+test 통합 사용 |
| LAION-Audiostock | 9,139 | 38 h | 15 h | 46 h | ✗ canonical split 없음 (단일 set) | T | F | 전체 사용 |
| AudioCaps | 45,623 | 125 h | 125 h | ~140 h | ✓ train / val / test | T | T | train + val 사용 (test 미사용) |
| FSD50K | 40,966 | 81 h | 81 h | ~80 h | ✓ dev / eval | T | T | dev 만 사용 (eval = Stage-2 평가) |
| AudioSet | 18,683 | 52 h | 52 h | ~54 h | ✓ bal_train / unbal_train / eval | T | T | bal_train 만 사용 (eval = Stage-2 평가) |
| Clotho | 4,881 | 30 h | 30 h | ~30 h | ✓ dev / val / eval | T | T | dev + val 사용 (eval = Stage-2 평가) |
| MACS | 3,930 | 11 h | 11 h | ~10 h | ✗ canonical split 없음 (TAU2019 base) | T | F | 전체 사용 |
| **subtotal** | **690,944** | **~4,303 h** | **~2,215 h** | | | | | |

1) `soundfile.info(path).frames / samplerate` 500 샘플 mean × N. seed=11.
2) LAION 4 splits: LAION-Audio-630K paper Table 1 (Wu et al. 2023). 나머지: 각 데이터셋 paper 의 dataset stats.
3) **LAION-BBC 2배 차이 사유**: 논문 (463 h) vs 실측 (998 h) — 우리 manifest 31,936 rows 가 LAION 공식 31,201 clips 보다 ~700 더 많고, 매우 긴 ambient clip 들 (p90 287 s, p99+ 1000 s+) 이 mean 을 끌어올림. paper 가 어떤 clip-duration cap 을 썼는지 미공개. **학습 시 30 s cap** 적용한 effective 는 226 h (논문 numbers 와 같은 자릿수).

## 4. Emotion MCQA (audio_emotion) — 47,054 rows ≈ 48 h

각 데이터셋 native 클래스 셋 유지. v5 leak-fix (DT 마지막 5 % dialogue / EmoV-DB Jenie / RAVDESS Actors 21-24 held-out) 유지. v6 추가 = IEMOCAP Sessions 1-4 (Session 5 Leave-session-out held-out, jos own download `/mnt/tmp/datasets/emotion_raw/IEMOCAP/IEMOCAP_full_release/`). hours empirical = sample 200 / source, seed=11, mean × N.

| Source | Rows | Empirical h | 클래스 (native) | Split 여부 | Train | Test |
|---|---:|---:|---|---|---|---|
| DailyTalk | 22,573 | 20.2 h | 7: anger, disgust, fear, happiness, no emotion, sadness, surprise | ✗ canonical split 없음. v5 leak-fix 로 마지막 5 % dialogue 를 eval 용으로 held-out | T | F |
| MELD (train+dev) | 11,096 | 9.8 h | 7: anger, disgust, fear, joy, neutral, sadness, surprise | ✓ train / dev / test. train+dev 만 학습, test 는 Stage-2 평가 | T | T |
| IEMOCAP (Sessions 1-4) | 5,882 | 7.9 h | 10: angry, disgust, excited, fear, frustrated, happy, neutral, other, sad, surprise | ✗ canonical split 없음 (5 sessions × 2 speakers). v6 leave-session-out: Sessions 1-4 train (10-class native), Session 5 eval. Session 5 = 2,170 utt → xxx (no-consensus) 520 제외 후 10-class valid 1,650. **eval 은 표준 4-class 프로토콜** ([`eval_iemocap_session5.py`](../../evaluation/stage2/eval_iemocap_session5.py)) 로 ang/hap/exc/neu/sad 만 채점, exc→hap merge → **1,241 utt** 만 입력 (fru 381 / sur 18 / fea 10 = 409 row 평가에서 silently drop). 학습은 10-class 라 fru/sur/fea/dis/oth 도 학습 신호 받지만 eval 점수는 안 잡힘. xxx (annotator no-consensus, 본 노드 Sessions 1-4 에서 1,987) 는 학습 풀에서도 제외 | T | T(Leave-session-out, 4-class) |
| EmoV-DB | 5,103 | 7.0 h | 5: amused, angry, disgusted, neutral, sleepy | ✗ canonical split 없음 (4 화자: Bea/Jenie/Josh/Sam). v5 leak-fix 로 Jenie 화자를 eval 용으로 held-out | T | F |
| RAVDESS | 1,200 | 1.2 h | 8: angry, calm, disgust, fearful, happy, neutral, sad, surprised | ✗ canonical split 없음 (24 actors). v5 leak-fix 로 Actors 21-24 를 eval 용으로 held-out | T | F |
| MUStARD++ | 1,200 | 1.6 h | 9: anger, disgust, excitement, fear, frustration, happiness, neutral, sadness, surprise | ✗ canonical split 없음 (소규모 sarcasm corpus) | T | F |
| **합계 (학습 풀)** | **47,054** | **~47.7 h** | | | | |

> **제외**: CREMA-D (path 가 다른 user 소유, 영구 제외), IEMOCAP Session 5 (Leave-session-out eval, 2,170 utt held-out), MELD test split (Stage-2 eval 과 일치, leakage).

## 5. 학습 시 노출 (training-exposed) ≠ 데이터 raw hours

`omni_max_audio_samples` cap (30 s, DAC 48 kHz = 1,440,000 samples / Whisper 16 kHz = 480,000 samples) 으로 clip 당 ≤ 30 s 만 학습에 들어감.

**왜 30 s 인가**: Whisper architecture 의 하드 제약 (positional embedding + conv stack 이 30 s log-mel 3,000 frames 에 고정 — pre-training 시점부터). DAC-VAE 는 architectural 한계는 아니지만 `cutoff_len = 3,584` token 안에 audio (~50 token/s × 30 s = 1,500) + text prompt + answer 가 들어가야 해서 token budget 으로 동일한 cap 이 자연스러움. 두 인코더 코드 한 줄 (`omni_max_audio_samples`) 로 통일.

**cap 초과 처리**: head-truncate (앞 30 s 사용, 뒤는 버림). Whisper variant 는 처음부터 head-truncate 였고, DAC variant 는 v6 이전까지 long clip rows 를 silently skip 하는 버그가 있어서 LAION-BBC 64 % / Freesound 21 % / Epidemic 9 % rows 가 학습에서 빠졌음. 2026-05-07 에 [`omni_dataset.py:480`](../../src/llamafactory/data/omni_dataset.py#L480) 의 `continue` 를 `wav[:max_audio_samples]` 로 fix 해서 두 variant 동작 통일 — 이제 모든 cap 초과 row 가 첫 30 s 만 학습에 들어가고 row 자체는 풀에 남음. random-crop (epoch 마다 다른 window) 안 한 이유: env_sound 0.25 ratio × 100k step 환경에서 한 row 평균 1-2 회 visit 이라 head-truncate 와 효과 차이 marginal, deterministic 이 디버깅 편함.

| Modality | Raw h | Cap-30 h | 차이 사유 |
|---|---:|---:|---|
| ASR | 50,787 | 50,787 | utterance 평균 짧음 (MLS / LibriTTS-R / VoxPopuli 14.5 s, GigaSpeech 4.3 s), cap 거의 안 걸림 |
| Sound captioning | 4,303 | 2,215 | 30 s 넘는 clip 의 뒷부분이 학습에 안 들어감. raw_h - cap30_h 비율로 측정. LAION-Freesound (raw 2,737 → cap30 1,543 = **43 % 손실**, foley/soundscape long take 다수) + LAION-BBC (raw 998 → cap30 226 = **77 % 손실**, p99+ 1,000 s 이상 ambient 라 평균 길이 >> 30 s) 가 주범. AudioCaps (10 s YouTube) / AudioSet (10 s) / Clotho (15-30 s) / MACS (10 s TAU) 는 raw 평균 ≤ 30 s 라 cap 영향 0 |
| Emotion | 47.7 | 47.7 | utterance 모두 < 30 s (IEMOCAP 최대 ~33 s 일부 제외) |

## 6. 라이선스 (요약)

| Source | License |
|---|---|
| MLS / LibriTTS-R | CC BY 4.0 |
| VoxPopuli | CC0 (transcripts), 다양 (audio) |
| GigaSpeech | apache-2.0 (metadata + alignments). audio 는 source 별 상이 (audiobook 다양, podcast 다양, YouTube 원본 라이선스). 상업 배포 시 audio 라이선스 별도 검토 |
| LibriSpeech (eval-only) | CC BY 4.0 |
| LAION-Audio-630K (Freesound/BBC/Epidemic/Audiostock) | LAION 공개 (clip 별 라이선스 상이; CC0/CC-BY 등) |
| AudioCaps | MIT |
| Clotho | Tampere Univ. dataset license (research only) |
| MACS | CC BY 4.0 (TAU2019 base) |
| AudioSet | CC BY 4.0 (annotations) / 원본 YouTube |
| FSD50K | CC BY 4.0 |
| DailyTalk / MELD / EmoV-DB / RAVDESS / MUStARD++ | 각 academic-only EULA |

상업 배포 시 academic-only 클립 분리 필요.

## 7. Leak audit (v6 vs spec)

held-out / eval 분리 룰 + builder ↔ eval 코드 일치성 전수 점검 (2026-05-07).

| 카테고리 | spec | v6 builder | eval 코드 | 결과 |
|---|---|---|---|---|
| **emotion** | | | | |
| IEMOCAP Session 5 | held-out (LSO eval, 4-class 표준 프로토콜) | `TRAIN_SESSIONS = [1,2,3,4]`, Session 5 skip, 학습은 10-class native | `eval_iemocap_session5`: `SESSION = "Session5"`, `LABEL_MAP = {ang/hap/exc→hap/neu/sad}` 4-class, fru/sur/fea silently drop (1,650 → 1,241 utt) | ✓ (의도된 비대칭 — § 4 IEMOCAP 비고 참고) |
| RAVDESS Actors 21-24 | held-out | `HELD_OUT_ACTORS = {21,22,23,24}` 제외 | `load_ravdess_heldout` → `Actor_21..24/` 만 | ✓ (1:1 동일 cutoff) |
| EmoV-DB Jenie 화자 | held-out | `HELD_OUT_SPEAKERS = {"jenie"}` 제외 | `load_emov_jenie` → `EmoV-DB/jenie/` 만 | ✓ |
| DailyTalk last 5 % dialogue | held-out (cutoff id 2,413) | `cutoff = ids_sorted[int(len*0.95)]` | `load_dailytalk_heldout` 동일 cutoff 식 | ✓ (1:1 동일 cutoff) |
| MELD test split | 제외 (Stage-2 eval 일치) | `SPLITS = ["train","dev"]` 만 (test csv 미참조) | `load_meld_test` → `MELD/audio/test/` + `test_sent_emo.csv` | ✓ |
| CREMA-D | 영구 제외 | builder 미실행 (`build_emotion_cremad.py` 잠금) | (해당 없음) | ✓ |
| **env_sound** | | | | |
| AudioCaps test | 제외 | `audiocaps_train + audiocaps_val` 5 shards 만 | (AudioCaps eval 코드 없음. test 미사용) | ✓ |
| FSD50K eval | 제외 (Stage-2 eval) | `fsd50k_dev_*` 3 shards 만 | `eval_fsd50k_map`: `FSD50K.ground_truth/eval.csv` | ✓ |
| AudioSet eval / unbal_train | 제외 (Stage-2 eval) | `audioset_bal_train_*` 2 shards 만 | `eval_audioset_map`: `AUDIOSET_ROOT/data/eval` | ✓ |
| Clotho eval | 제외 (Stage-2 eval) | `clotho_dev + clotho_val` 만 | `eval_clotho_caption --split evaluation` | ✓ |
| **ASR** | | | | |
| MLS dev/test | 제외 | `train` 100 % (50K 샘플 검증) | (MLS eval 코드 없음. LibriSpeech 으로 대체) | ✓ |
| VoxPopuli dev/test | 제외 | `train` 100 % (182,466 row) | (VoxPopuli eval 코드 없음) | ✓ |
| LibriTTS-R dev/test | 제외 (LibriSpeech eval 보호) | top-level 100 % `train/` (clean-100 / clean-360 / other-500) | `eval_librispeech_wer`: HF `openslr/librispeech_asr` `test.clean`/`test.other` | ✓ (LibriTTS-R train ↔ LibriSpeech test 절대 X) |
| GigaSpeech dev/test | 제외 (GigaSpeech eval 보호) | XL `train` 의 50% random subsample (seed=11), dev / test 미참조 | `eval_asr_external --datasets gigaspeech`: HF `speechcolab/gigaspeech` `test` split | ✓ (XL train ∩ test = ∅, sampling 도 train 안에서만) |
| **외부 corpus (cross-corpus eval)** | | | | |
| SAVEE / JL-Corpus / MSP-Podcast / ESC-50 / LISTEN | v6 train pool 부재 | (해당 없음) | 각 eval 스크립트가 별도 corpus 직접 로드 | ✓ |

검증 명령:

```bash
python3 - <<'EOF'
import json, os, re
from collections import Counter
ROOT = "/mnt/tmp/datasets/manifests/v6"
emo = [json.loads(l) for f in os.listdir(f"{ROOT}/audio_emotion")
       for l in open(f"{ROOT}/audio_emotion/{f}")]
checks = {
    "IEMOCAP Session5":  sum(1 for r in emo if r["source"]=="iemocap" and "/Session5/" in r["audio_path"]),
    "RAVDESS Act 21-24": sum(1 for r in emo if r["source"]=="ravdess"
                             and (m:=re.search(r"/Actor_(\d+)/", r["audio_path"])) and int(m.group(1))>20),
    "EmoV-DB Jenie":     sum(1 for r in emo if r["source"]=="emovdb" and "/jenie/" in r["audio_path"].lower()),
    "MELD test path":    sum(1 for r in emo if r["source"]=="meld" and "/test/" in r["audio_path"]),
    "CREMA-D source":    sum(1 for r in emo if r["source"]=="cremad"),
}
for k, v in checks.items(): print(f"  {k}: {v} (expect 0)  {'✓' if v==0 else '✗ LEAK'}")
EOF
```

## 8. 재계산 명령

manifest row 수 + hours empirical 재추정:

```bash
python3 - <<'EOF'
import json, random, os
import soundfile as sf
M = "/mnt/tmp/datasets/manifests/v6_raw"   # v3 + IEMOCAP Sessions 1-4 (= v6 emotion 풀 입력)
per_src = {}; counts = {}
for f in sorted(os.listdir(M)):
    if not f.endswith(".jsonl"): continue
    with open(f"{M}/{f}") as fh:
        for line in fh:
            d = json.loads(line); s = d.get("source", "?")
            counts[s] = counts.get(s, 0) + 1
            if d.get("audio_path"):
                per_src.setdefault(s, []).append(d["audio_path"])
for s, n in sorted(counts.items(), key=lambda x: -x[1]):
    if s in per_src:
        rng = random.Random(11)
        sm = rng.sample(per_src[s], min(500, len(per_src[s])))
        durs = [sf.info(p).frames / sf.info(p).samplerate for p in sm]
        h = n * (sum(durs)/len(durs)) / 3600
        print(f"{s:28s} rows={n:>10d} h={h:8.1f}")
    else:
        print(f"{s:28s} rows={n:>10d} h=(nubes-only, see published)")
EOF
```

## 변경 이력

- 2026-05-06: 작성. ASR published, sound/emotion empirical (500 sample/source, seed=11).
- 2026-05-07: v6 추가. MUStARD++ 1 → 1,200 row 재빌드 (다른 노드 audio_wav handoff). IEMOCAP Sessions 1-4 (5,882 row) jos own download 으로 학습 풀 포함, Session 5 (1,650 row valid) Leave-session-out held-out. emotion 합계 41,172 → 47,054, 전체 합계 12,077,340 → 12,083,222. manifest dir `/mnt/tmp/datasets/manifests/v6/` 신설. 빌더 dir 도 `scripts/v3_manifest/` → `scripts/manifest_builders/` 로 일괄 rename.
- 2026-05-07 (later): emotion hours 도 v6 시점에 sample 200 / source / seed=11 로 재추정. DT 18→20.2, MELD 10→9.8, EmoV 6→7.0, MUStARD 1.4→1.6 등 +8 % drift 반영. 전체 합계 50,134 → 50,138 h (raw), 48,046 → 48,050 h (cap30). §5 sound captioning 손실 비율 의미 풀어씀.
- 2026-05-07 (DAC cap fix): `omni_dataset.py:480` 의 long-clip skip (`continue`) 을 head-truncate (`wav[:max_audio_samples]`) 로 변경, Whisper variant 와 동작 통일. v6 이전 DAC-VAE 학습 풀에서 silently 빠지던 LAION-BBC ~20K rows / Freesound ~95K rows / Epidemic ~7K rows / Audiostock ~530 rows (총 ~123K rows, sound captioning 의 ~18%) 가 다시 학습에 들어옴. row 수 manifest 는 동일, 동작만 변경. §5 에 사유 + cap 의 architectural 근거 추가.
- 2026-05-07 (leak audit): § 7 leak audit 표 추가. 13 개 leak vector 전수 점검 결과 모두 통과. § 2 LibriTTS-R 행에 "v6 학습은 train.\* 3 splits 만 (dev/test 는 LibriSpeech eval 보호 위해 제외)" 명시. 기존 § 7 재계산 명령 → § 8 로 번호 이동.
- 2026-05-07 (eval-side audit + MELD builder fix): § 7 표에 eval 코드 컬럼 추가 — builder 측 held-out 룰 ↔ eval 측 로드 경로 1:1 매칭 검증. DT/EmoV/RAVDESS 는 builder 와 eval 이 동일 cutoff 식 / actor 셋 / speaker 셋 사용. **MELD builder sehyun-free 화**: `build_emotion_meld.py` 가 `/mnt/ddn/users/sehyun/.../meld_{train,dev}_shard_*.jsonl` 의존하던 것을 jos own `MELD.Raw/{train,dev}_sent_emo.csv` 직접 파싱으로 변경. row count (9,988 + 1,108 = 11,096) 동일 보존, sehyun 경로 의존 0. v6_emotion_split 16 shard 도 새 MELD content 로 재머지 (다른 5 source 는 변경 없음).
- 2026-05-07 (GigaSpeech 추가, planned): v6 audio_asr 풀에 GigaSpeech XL (`speechcolab/gigaspeech`) seed=11 random 50% subsample (~5,000 h, ~4.13M rows) 추가 결정. ASR rows 11,345,224 → 15,478,224, hours 45,787 → 50,787. 전체 합계 12,083,222 → 16,216,222 row, raw 50,138 → 55,138 h, cap-30 48,050 → 53,050 h (utterance 평균 4.3 s 라 cap 영향 거의 없음). dev (~12 h) / test (~40 h) 는 GigaSpeech-test eval ([`eval_asr_external.py`](../../evaluation/stage2/eval_asr_external.py)) 보호 위해 sampling 대상에서 제외. **manifest rebuild + 빌더 작성 진행 중** — 현재 본 문서는 planning entry, 실제 shard (`audio_asr/gigaspeech_*.jsonl`) 는 아직 v6 dir 에 없음. 추가 후 row 수 / hours empirical 재추정 필요 (§ 8 재계산 명령). 부수 작업: [`eval_asr_external.py:65-82`](../../evaluation/stage2/eval_asr_external.py#L65-L82) `DATASETS` 의 GigaSpeech license_note 가 docstring (OOD) 과 모순하는 점도 정리 필요 — sampling 후엔 in-dist 가 맞으니 docstring 쪽을 갱신.
- 2026-05-07 (GigaSpeech 빌드 완료 + 빌더 nubes-direct 화): v6 audio_asr/gigaspeech_*.jsonl 47 shards / **4,129,334 rows 실측** 완료 (target 4,128,138, ratio 0.5001). 초기 빌드는 sehyun 의 `qwen3_5_dacvae_asr_shuffled_128/` 셔드에서 추출했으나 그 후 **빌더 4개 nubes-direct 로 재작성**: [`build_mls.py`](../../scripts/manifest_builders/build_mls.py) (transcripts.txt 단일 파일 stream parse), [`build_voxpopuli.py`](../../scripts/manifest_builders/build_voxpopuli.py) (recursive list + per-file `.txt` fetch), [`build_gigaspeech.py`](../../scripts/manifest_builders/build_gigaspeech.py) (recursive list + 50 % sample + per-file `.txt` fetch), [`build_libritts_r.py`](../../scripts/manifest_builders/build_libritts_r.py) (stub, nubes `/libriTTS/` 구조 추가 분석 필요). 공통 helper [`_nubes_helper.py`](../../scripts/manifest_builders/_nubes_helper.py): gateway `http://c.nubes.sto.navercorp.com:8000/v1` + bucket `hyperscaleai-audiollm`, list pagination 은 `X-Continuation-Token` 헤더 기반. 기존 [`convert_libri_mls_vox.py`](../../scripts/manifest_builders/convert_libri_mls_vox.py) 는 deprecation 마크 (sehyun cache 의존). v6 audio_asr/ 셔드 자체는 그대로 (재빌드 안 함, nubes_path 정상). [`eval_asr_external.py`](../../evaluation/stage2/eval_asr_external.py) docstring + DATASETS 도 GigaSpeech "Stage-1 in-dist" 로 갱신.
- 2026-05-07 (eval docstring 정정 4 건): 다른 세션 audit 가 발견한 metadata / docstring layer 정합 이슈 처리. (1) **`eval_asr_external.py`**: GigaSpeech / CommonVoice 의 `license_note` 가 docstring "OOD" 와 모순돼 "in-dist (Stage-2 mix)" 였음 → "Stage-1 OOD | Stage-2 in-dist" 형식으로 양 stage 명시. librispeech_clean 의 "(LibriTTS-R/MLS subset)" 도 "Stage-1 in-dist (LibriTTS-R train.* shares speakers/text, disjoint test split)" 로 풀어씀. (2) **IEMOCAP 4-class vs 10-class**: § 4 IEMOCAP 행 + § 7 표에 "학습 10-class native, eval 4-class 표준 프로토콜 (exc→hap merge), Session 5 1,650 → 1,241 utt 입력" 명시. fru/sur/fea 409 row 가 eval 점수에서 silently drop 되는 의도된 비대칭 기록. (3) **`eval_source_emotion.py` docstring**: Stage-2 LISTEN-mix 가 MELD-test / MOSEI-test 끌어와 contamination 발생 가능 — Stage-1 v6 ckpt 평가는 클린이지만 Stage-2 LoRA + LISTEN ckpt 평가 시 MELD 만 영향 받음 (DT / EmoV / RAVDESS 는 무관) 경고 추가. (4) **`eval_audioset_caption.py` / `eval_fsd50k_caption.py` / `evaluation/stage1_v4/README.md`**: prompt format 은 v3 이래 caption-form 통일 (sound_caption pool) 이지만 caption target text 가 v5 에서 "sound of X, Y" 합성 → ontology description 로 갈아엎음. 따라서 sentence-mode wrapper 는 v5/v6 ckpt 에 fit, v4 ckpt 평가 시 F1 underestimate 발생 — 디렉터리 명 `stage1_v4/` 은 legacy 임을 명시.
