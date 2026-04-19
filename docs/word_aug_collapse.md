# Word-aug 로 인한 Projector Collapse

**작성일**: 2026-04-13
**업데이트**: 2026-04-15 (Stage 2 다중 ckpt 평가 결과 + 해결책 선택)
**상태**: 원인 확정 (word-aug = word-crop) — 재패킹 (half-interleave) 진행 중

## 증상

1. **Loss 정체**: Stage 1 학습 중 loss 가 ~3.45~3.55 에서 plateau. pre-word-aug baseline (10k hour, whole-only) 에서는 금방 loss 1 까지 내려간 전례 있음.

2. **WER val 개선 없음**: split 1/2 에 걸쳐 WER val 이 100% ~ 151% 사이에서 진동, 감소 trend 없음.

3. **HYP 출력이 동일 legal-speak 로 수렴** (결정적 증거):
   ```
   [val][100] REF: I WHIRLED ROUND AND THERE ON ONE OF THOSE DRY GRAVEL BEDS
                   WAS THE BIGGEST SNAKE I HAD EVER SEEN
              HYP:  between the two parties is not a question of law but of fact
                    and it is therefore impossible to determine whether or not
                    there has been an agreement by which the plaintiff
   [val][200] REF: MOST OF MY LIKENESSES DO LOOK UNAMIABLE BUT THE VERY
                   SUFFICIENT REASON I FANCY IS BECAUSE THE ORIGINALS ARE SO
              HYP:  between the two parties is not a question of law but one
                    of fact and it is for this reason that the court has no
                    jurisdiction to hear the case in its
   ```
   REF 가 완전히 다른 utterance 인데 HYP 가 **거의 동일**. 즉 projector 가 audio 내용에 따라
   구분 가능한 embedding 을 못 만들어내고 있음. LLM 은 audio prefix 가 우의미한 신호를 주지
   않으므로 "unconditional prior 중 고확률 continuation (legal-speak)" 로 fallback.

## 진단: Projector Collapse

Projector 출력이 audio 입력에 무관하게 유사한 값으로 수렴하는 상태.
- Symptom: 서로 다른 audio 에 대해 동일/유사 hypothesis 생성
- Cause: projector weight 가 audio discriminative 정보를 학습하지 못함

## 가설

### 가설 A: Word-aug 의 과도한 믹스 비율 (유력)

§20 word-aug 구현 결과, bin 내 sample 분포:
- whole utterance : word sub-clip ≈ 1 : 9
- ls100 per-rank: 4,077 → 38,352 sample (9.4x 증가) — word clip 이 압도적

이로 인한 문제:
1. **Gradient 90%+ 가 word clip 에서 옴** → projector 가 "짧은 audio → 단일 word" 문제를
   최적화하도록 편향됨. sentence-level context 를 반영하는 능력 희생.

2. **Word clip task 의 본질적 어려움**:
   - 0.2~0.5 초 짜리 audio 로 단일 word 분류
   - context 없음, 비슷한 word 구분 불가 (has/had, there/their)
   - Qwen 의 strong word-sequence prior 를 활용 못함
   - per-sample loss floor 가 높음 (~3~4)

3. **평균 희석 효과**:
   ```
   total_loss = (1 × L_whole + 9 × L_word) / 10
   L_whole → 1, L_word → 4 라면 total → 3.7
   ```
   total loss plateau (3.45~3.55) 가 이 공식과 일치.

4. **실제로 whole utterance 쪽이 학습되고 있는지 불확실**: HYP 가 audio 무관한 collapse
   형태를 보임 → sentence level 도 학습 실패하고 있을 가능성.

### 가설 B: LR 과다

- `stage1_lr = 2e-4`
- 이전 baseline (10k h) 에서도 동일 값 사용했을 가능성 높지만, word-aug 로 gradient 노이즈가
  증가하면서 2e-4 가 projector collapse 유발했을 수도
- 일반 추세: Conv1d projector + frozen LLM 구성에서 LR 1e-4 ~ 5e-4 가 흔한데 2e-4 는 그 중간
- 유력도: 낮음 (baseline 에서 잘 돌던 값)

### 가설 C: Packed bin 내 mixed clip 의 position/attention mask 결함

- word clip 과 whole clip 을 같은 bin 에 pack 할 때 position_id / attention_mask 처리가
  잘못되어 cross-contamination 발생 가능성
- Flash Attention 2 block-diagonal mask 가 clip 경계를 제대로 분리 못하면 gradient 가
  inter-clip leak
- 유력도: 중간 (collator 로직 재검증 필요)

## 검증 계획

### 실험 1: Runtime word-clip 필터 (빠른 검증, ~1.5h)

목적: word clip 을 training 시점에 스킵하여 whole-only 로 효과적 전환, WER trajectory 비교.

구현:
- `_project_precomputed` 또는 collator 에서 `audio_lengths < threshold` 인 clip drop
- fb_dacvae 기준 1 초 미만 (~86 frame) 을 word clip 으로 간주
- packed Arrow 그대로 사용, 재-pack 불필요
- 단점: bin utilization ~30% 감소, step time 증가

성공 기준:
- WER val 이 step 35/70 에서 의미있게 하락 (예: 150% → 80% 이하)
- HYP 출력이 audio 에 따라 달라짐 (collapse 해제)

### 실험 2: 전용 whole-only 재-pack + 학습 (엄밀, ~15h)

- `precompute/run_pack.sh` 에 `--word-aug` 빼고 재실행 → 기존 mixed/ 덮어쓰기
- Stage 1 4h 정도로 돌려서 loss/WER 확인
- 시간 cost 크지만 실험 1 이 애매한 결과면 여기로

### 실험 3: Collator attention mask 검증

- `OmniCollator.__call__` 의 position_id / attention_mask 빌드 로직이 clip 경계에서
  제대로 block 을 분리하는지 단위 테스트
- word clip 여러 개 + whole 1 개가 섞인 mock bin 으로 forward 후 audio embedding 상관계수 확인

## 관련 문서

- [§20 word-aug offline pre-pack](dataloader_trials.md#20-word-aug-offline-pre-pack--문장--단어-레벨-혼합-2026-04-1213) — 구현 경위
- [§21 _project_precomputed pad-stack OOM fix](dataloader_trials.md#21-_project_precomputed-clip-by-clip-loop-2026-04-13-word-aug-oom-근본-수정) — 메모리 최적화
- [packing_fa2_liger_fsdp.md](packing_fa2_liger_fsdp.md) — packing 내부 구조

## 2026-04-15 후속 조치

### 관찰: Stage 2 학습도 WER ~160% 고착 (가설 A 재확인)

fb_dacvae Stage 2 run `s2_outputs_0414_1442` (2 full-epoch × 8 data-split, word-aug 로
빌드된 `packed_16384` 사용) 에서 16 checkpoint (8 split × 2 epoch) + HF auto-save 1 개
= 17 ckpt 를 LibriSpeech dev-clean 200 samples 로 병렬 평가.

| saved | ckpt | WER |
|---|---|---|
| 04/14 16:00 | fe1_split1 | **134.39%** ← 유일한 저점 |
| 04/14 17:01 | fe1_split4 | 154.85% |
| 04/14 18:02 | fe1_split7 | 159.76% |
| 04/14 18:26 | checkpoint-25 | 160.37% |
| 04/14 18:26 | fe1_split8 | 161.33% |
| 04/14 19:27 | fe1_split2 | 161.09% |
| 04/14 20:29 | fe1_split3 | 161.50% |
| 04/14 21:30 | fe1_split6 | 160.76% |
| 04/14 22:31 | fe1_split5 | 161.25% |
| 04/14 23:32 | fe2_split2 | 161.25% |
| 04/15 00:33 | fe2_split7 | 161.58% |
| 04/15 01:34 | fe2_split4 | 163.17% |
| 04/15 02:35 | fe2_split5 | 162.87% |
| 04/15 03:36 | fe2_split6 | 162.58% |
| 04/15 04:35 | checkpoint-60 | 162.51% |
| 04/15 04:39 | checkpoint-63 | 162.63% |
| 04/15 04:39 | fe2_split3 | 162.76% |

**핵심 관찰**: fe1_split1 (첫 1 시간) 에서 134% 로 상대적 저점 찍고 이후 전 기간 160-163% 로
**퇴화**. 학습 진행으로 전혀 개선 없음, 오히려 초기 ckpt 가 best. Stage 1 에서 관찰된 "loss plateau +
HYP 내용 고착" 증상과 정확히 동일 (단 Stage 2 의 HYP 는 "to the question of whether or not we
should have a government..." 같은 다른 고확률 prior 로 수렴).

평가 방법·뷰어: [eval_ckpts/README.md](../eval_ckpts/README.md) 참조.

### 원인 확정: word-aug = word-crop ASR

word-aug 구현 [train_pipeline_override.py:1281-1329](../train_pipeline_override.py#L1281-L1329)
의 `make_precomputed_processor_fn.process_samples` 를 역추적한 결과, utterance 당:
1. 원본 전체 문장 emit (whole)
2. alignment 의 각 word `w` 에 대해 `feats_2d[start:end]` 슬라이스 + 단어 텍스트 한 쌍 emit

→ 정확히 **word-crop ASR** (alignment 를 전처리 단계에서 소모하고 forward 안 에서 cross-modal
신호로 활용 안 함) 이고, 이 구조가 가설 A 에서 "word clip 이 압도적, gradient 90%+ 가 word 에서
옴" 으로 예측한 그대로임.

이론적 배경: [docs/word_alignment.md](word_alignment.md) "Interleaving vs word-crop ASR" 섹션.
결론: word-crop 은 sentence ASR 의 **열화판** (문맥 손실) 이면서 alignment 의 고유 가치
(token-level cross-modal supervision) 를 얻지 못하는 중간 지대. Interleaving 만이 alignment 를
제대로 활용.

### 조치: 재패킹 (half-interleave, 안 2)

- `precompute/pack_arrow.py` → [arXiv/scripts/pack_arrow.py](../arXiv/scripts/pack_arrow.py) 아카이브
- `packed_16384` (~190 GB, 문장 bin + word sub-clip bin 혼입) → 폐기 결정
- 신규 [precompute/half_inlv_pack_arrow.py](../precompute/half_inlv_pack_arrow.py) 작성:
  utt-level hash split (`md5(utt_id) & 1`) 로 50/50 sentence / interleave emit.
- Interleave bin 포맷: `[audio_pad]*T_w0 + <|audio_correspond|> + text_w0 + [audio_pad]*T_w1 + ... + EOS` —
  alignment 를 forward pass 내부 token-level supervision 으로 사용.
- 데이터셋 총 시간 유지 (1 utt → 1 bin), 포맷 효과만 분리 → baseline 과 clean 비교 가능.
- 상세 설계/안 1 vs 안 2 논의: [docs/word_alignment.md](word_alignment.md) "Word-interleaving packing 계획".

### 검증 (진행 중)

half_inlv 재패킹 후 동일 cfg 로 Stage 2 재학습 → WER 수렴 양상 비교 예정.
이전의 실험 1 (runtime word-clip 필터) / 실험 2 (whole-only 재-pack) 는 interleaving 쪽이
더 상위 개념이라 skip — interleaving 이 실패하면 그때 whole-only 로 후퇴.

### 후속 진단: word-aug 만으로는 설명 부족, 더 큰 원인 존재

2026-04-15 오후 remote 에서 pull 된 [train_pipeline_arrow_torch.py](../train_pipeline_arrow_torch.py)
(pure torch + FSDP 재작성 버전) 가 **word-aug 없이도 수렴하지 않음** 이 확인. word-aug 하나로는
두 현행 trainer 의 공통 수렴 실패를 설명 못함. 두 파일이 공유하는 더 근본적 차이는:

- legacy (`model.py:AudioQwen`) 는 `"Audio:\n"` / `"\nTranscript:\n"` **multi-token 텍스트 prompt**
  로 audio 앞뒤를 감쌈 — Qwen 사전학습 어휘 그대로 사용
- 현행 둘 (override, arrow_torch) 은 **단일 새 특수 토큰 `<|audio_correspond|>`** 으로 교체 +
  `resize_token_embeddings` — LLM 이 한 번도 본 적 없는 토큰, 학습 전체 책임이 projector 로 이관

즉 word-aug 는 **보조 요인**, prompt 포맷 regression 이 **주 원인** 일 가능성. 상세 분석은
[prompt_format_regression.md](prompt_format_regression.md).

현재 half_inlv 재패킹 run 은 이 가설의 control:
- word-aug 제거 (안 2 hash split) + corr 버그 fix 까지 모두 적용
- 여전히 legacy 수렴 수준 못 따라가면 → prompt 포맷이 결정적 원인 확정 → 안 X/Y/Z 로 전환
- legacy 수준으로 수렴하면 → word-aug + corr 버그가 주 원인, prompt 논의 보류
