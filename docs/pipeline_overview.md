# Pipeline Overview — train_pipeline_override.py

작성: 2026-04-08

---

## 1. 전체 구조

```
[run.sh]
  └─ accelerate launch train_pipeline_override.py
        ├─ Stage 1: Projector Alignment (LLM frozen)
        └─ Stage 2: LoRA Fine-tuning (LLM + projector)
```

### 최적화 스택 (항상 활성화)

| 기법 | 설명 | 활성화 조건 |
|---|---|---|
| Sequence Packing | 여러 샘플을 cutoff_len 토큰으로 묶어 padding 최소화 | 항상 ON |
| Flash Attention 2 | varlen kernel, padding 완전 제거 `(1, sum_nonpad)` | 기본 ON (`--attn-impl` 변경 가능) |
| Liger Kernel | fused RoPE / RMSNorm / SwiGLU / CE loss (vocab 메모리 절감) | `--liger` (기본 ON) |
| FSDP (Stage 1) | LLM 메모리 1/8 절감 이론상이나 activation이 dominant → 실제 절감 없음. 속도 2.2배 느림 (gradient_checkpointing 충돌) | `--fsdp-stage1` (기본 OFF, **사용 비권장**) |
| FSDP (Stage 2) | LLM + projector LoRA 파라미터 분산 | `--fsdp` (기본 ON) |

---

## 2. 데이터 파이프라인

### 2-1. 전체 흐름

```
HF Streaming Dataset (6개)
  │  ls100, ls360, ls500 (LibriSpeech) — Arrow cache: /mnt/tmp/cache/hf
  │  mls (MLS English) — OGG-Opus
  │  gs (GigaSpeech XL) — 2072 shards
  │  vp (VoxPopuli en)
  │
  ├─ shard(num_shards=num_gpus, index=rank)   ← GPU별 1/N 슬라이스
  ├─ rename_column → ["audio", "text"] 통일
  │   (word_aug=True: + "utterance_id" 유지)
  ├─ interleave_datasets(seed=42)             ← 6개 균등 혼합
  ├─ shuffle(buffer_size=10000)
  │
  ├─ map(process_samples, batched=True, batch_size=4)
  │     └─ 각 (audio, text) → input_ids / labels / audio_features / audio_lengths
  │         word_aug=True: + word-level 서브샘플 생성 (§3 참조)
  │
  └─ map(pack_samples, batched=True, batch_size=200)
        └─ greedy knapsack: cutoff_len 안에 샘플 최대 밀집
            attention_mask: 서브시퀀스 인덱스 (1, 2, 3…) → block-diagonal 마스킹용
```

### 2-2. Sequence Packing 세부

**greedy knapsack 알고리즘**:
1. 배치 내 모든 샘플을 길이 오름차순 정렬
2. cutoff_len 초과 샘플은 drop (단일 아이템도 fit 불가)
3. 새 bin 시작 → remaining = cutoff_len
4. 남은 샘플 중 remaining 이하인 가장 큰 것 선택 → bin에 추가 → remaining 감소
5. fit하는 샘플이 없으면 bin 종료 → pad to cutoff_len
6. 모든 샘플 소진 시까지 반복

**중요**: 아이템은 절대 mid-truncate 되지 않는다. fit하면 통째로, 안 되면 다음 bin으로.

**attention_mask 값의 의미**:
- `0`: padding
- `1, 2, 3, …`: 서브시퀀스 인덱스 (같은 숫자 → 같은 샘플 → cross-attend 허용)

### 2-2b. Precomputed / Pre-packed 경로

`--precomputed-dir` 지정 시 HF streaming 대신 사전 계산된 Arrow 파일을 사용한다.

```
{precomputed_dir}/{encoder}/{dataset}/
  ├─ packed_{cutoff_len}/rank{N}.arrow   ← Pre-packed (우선, map 완전 스킵)
  ├─ rank{N}_s*.arrow                     ← Per-sample sharded
  └─ rank{N}.arrow                        ← Per-sample single
```

**Pre-packed 모드** (권장):
```
packed Arrow → pyarrow.ipc.read_all() → HF Dataset (in-memory)
  → .to_iterable_dataset() → interleave → DataLoader → 학습
```
processor_fn, packer_fn map 단계 없음. 데이터 로딩 수 초.

**Per-sample 모드** (pre-packed 없을 때 fallback):
```
per-sample Arrow → pyarrow.ipc.read_all() → HF Dataset (in-memory)
  → .map(processor_fn, num_proc=16) → .map(packer_fn, num_proc=16)
  → .to_iterable_dataset() → interleave → DataLoader → 학습
```

Pre-pack 생성: `bash precompute/run_pack.sh --encoder fb_dacvae`

**Data splits** (대용량 데이터셋 OOM 방지):

mls(55GB/rank)+gs(~55GB/rank) 등 대용량 데이터셋은 8 rank 동시 로드 시 ~948GB → cgroup OOM.
`num_data_splits=2` 설정 시 각 rank 파일의 bins를 N등분, split마다 1/N만 로드 후 학습, 해제를 반복한다.

```
num_data_splits=2:
  for split in [0, 1]:
    각 rank 파일에서 해당 split의 bins만 slice → 학습 → del + gc.collect()
```

자세한 내용: `dataloader_trials.md` §13 참조.

### 2-3. OmniCollator

| FA2 경로 | Eager/SDPA 경로 |
|---|---|
| padding 제거 → `(1, sum_nonpad)` | `(B, cutoff_len)` 유지 |
| position_ids: 샘플마다 0부터 리셋 | 4D block-diagonal mask 생성 |
| FA2 varlen kernel이 position 불연속성으로 경계 인식 | `prepare_4d_attention_mask`로 블록 마스크 생성 |

---

## 3. Word-level Augmentation (`--word-aug`)

### 3-1. 아이디어

기존 학습: `audio[0–10s]` → `"chapter one"` (문장 단위)

Word-aug 추가:
```
audio[0.0–0.82s]  → "chapter"   (단어 단위 서브샘플)
audio[0.98–1.06s] → "one"
audio[1.22–1.78s] → "missus"
…
```

각 단어의 acoustic signal과 텍스트 표현 사이의 직접적인 매핑을 학습하므로:
- 희귀 단어에도 gradient 발생 (문장 단위에서는 LLM context로 예측 가능해서 기여 적음)
- 발음-텍스트 alignment가 더 명확한 supervision

### 3-2. 데이터 소스

| 데이터셋 | Arrow 경로 | utterance_id 필드 | 총 단어 수 |
|---|---|---|---|
| LibriSpeech (ls100/360/500) | `word_alignments_merged/librispeech/{split}.arrow` | `id` (HF) | 283,944 발화 |
| MLS | `word_alignments_merged/mls/train.arrow` | `original_path` (.opus 제거) | 2,420,047 발화 |
| GigaSpeech | `word_alignments_merged/gigaspeech/train.arrow` | `segment_id` | 8,282,987 발화 |
| VoxPopuli | `word_alignments_merged/voxpopuli/train.arrow` | `audio_id` | 182,482 발화 |

Arrow 스키마: `utterance_id, split, audio_duration, transcript, words[{word, start, end, score?}]`

### 3-3. 메모리

`AlignmentLookup`: Arrow 파일을 memory-mapped pyarrow로 로드.
- **index dict** (`utterance_id → row_idx`): rank당 GigaSpeech ~660MB, 전체 ~1.2GB
- **words 실데이터**: OS demand paging으로 필요 시 로드 (랜덤 접근 → page fault 발생하나 허용)
- DataLoader workers는 부모 프로세스 메모리를 fork 후 COW 공유 → 물리 메모리 복제 최소화

GigaSpeech words는 uppercase (e.g., `"AND"`) → processor에서 `.lower()` 적용.

### 3-4. 서브샘플 생성 규칙

```python
min_word_samples = 1600  # 100ms @ 16kHz
for w in words:
    audio_slice = waveform[int(w.start * 16000) : int(w.end * 16000)]
    if len(audio_slice) < 1600:
        continue  # 100ms 미만 단어 skip
    emit(audio_slice, w.word.lower())
```

발화당 평균 ~15–20개 단어 → word_aug ON 시 샘플 수 약 16–21배 증가.
processor 통계 로그 (`pid=…`)에서 `sentence / word / ratio` 확인 가능.

### 3-5. 학습에서 문장 vs 단어 샘플의 혼합

두 타입 모두 같은 형식:
```
input_ids: [audio_pad]*t + [audio_correspond] + text_ids + [EOS]
labels:    [IGNORE]*(t+1)                     + text_ids + [EOS]
```

Sequence Packing이 두 타입을 자동으로 섞어서 하나의 bin에 넣음.
의도적으로 분리하지 않아도 된다 — 짧은 단어 샘플과 긴 문장 샘플이 같은 bin에 공존.

---

## 4. 모델 구조

```
Audio (16kHz, mono)
  → [frozen encoder]   예: fb_dacvae (8-dim), dac (1024-dim), encodec (128-dim) …
  → (B, T_enc, out_dim)
  → [trainable projector]   Conv1d ×2 (stride-2 각각), k=5, GELU + LayerNorm
  → (B, T_proj, llm_dim=2048)
  → LLM embedding lookup으로 placeholder token 대체
  → [Qwen3.5-2B/4B]   Stage 1: frozen / Stage 2: LoRA (r=16)
  → CE loss (packed 경로: Liger fused linear CE)
```

**samples_per_token 계산** (config.py에서 자동):
```python
samples_per_token = hop * (16000/tgt_sr) * prod(proj_strides)
# 예: fb_dacvae: hop=512, tgt_sr=44100, strides=[2,2] → ~956 samples/token
```

---

## 5. 2-Stage 학습

### Stage 1: Projector Alignment

| 항목 | 값 |
|---|---|
| 학습 파라미터 | projector + proj_norm |
| LLM | frozen |
| LR | 2e-4, cosine, warmup_ratio=0.1 |
| Epochs | 2 (기본) |
| 평가 | WER + val_loss every `eval_steps`(기본 500) steps |
| 저장 | `{cache_dir}/{encoder}/s1_outputs_{run_id}/` |
| best projector | `{cache_dir}/{encoder}/s1_outputs_{run_id}/best_s1_proj.pt` |
| FSDP | ✗ DDP 기본. `--fsdp-stage1` 사용 비권장 (속도 2.2배 느림, 메모리 절감 없음 — 검증됨) |

**Stage 1 조기 종료**: `kill -USR1 $(cat /mnt/tmp/cache/train.pid)`

### Stage 2: LoRA Fine-tuning

| 항목 | 값 |
|---|---|
| 학습 파라미터 | LoRA (r=16, q/k/v/o_proj × 24 layers) + projector |
| LLM LoRA 파라미터 | ~5M |
| LR | 2e-5, cosine, warmup_ratio=0.03 |
| Epochs | 2 (기본) |
| 평가 | WER + val_loss every `eval_steps` steps |
| 저장 주기 | `save_steps`(기본 5000) steps, `save_total_limit=3` |
| 저장 경로 | `{cache_dir}/{encoder}/s2_outputs_{run_id}/` |
| FSDP | ✓ (full_shard + auto_wrap, encoder excluded) |

---

## 6. Startup 확인 로그

`train_pipeline_override.py` 시작 시 아래와 같은 요약 출력:

```
════════════════════════════════════════════════════════
  PIPELINE CONFIG
════════════════════════════════════════════════════════
  Encoder       : fb_dacvae
  LLM           : Qwen/Qwen3.5-2B
  Stage(s)      : all
  Datasets      : ls100, ls360, ls500, mls, gs, vp
  Est. hours    : 21460h
  Cutoff len    : 16384 tokens
  ── Optimizations ──────────────────────────────────
  Seq Packing   : ✓ (always on, cutoff=16384)
  Flash Attn 2  : ✓
  Liger Kernel  : ✓
  FSDP (Stage2) : ✓
  ── Word Augmentation ──────────────────────────────
  Word-aug      : ✓
    ls100    : ✓ librispeech/train-clean-100.arrow
    ls360    : ✓ librispeech/train-clean-360.arrow
    ls500    : ✓ librispeech/train-other-500.arrow
    mls      : ✓ mls/train.arrow
    gs       : ✓ gigaspeech/train.arrow
    vp       : ✓ voxpopuli/train.arrow
  ── Training Schedule ──────────────────────────────
  Stage1 epochs : 2
  Stage2 epochs : 2
  Eval steps    : 500
  Save steps    : 5000
════════════════════════════════════════════════════════
```

런타임 중 worker 통계 로그 (every 2000 sentence samples per worker):
```
[Processor pid=12345] sentence=2,000 word=34,821 (ratio=17.4x)
```

---

## 7. 실행 방법

```bash
# 전체 데이터, word-aug ON
bash run.sh --encoder fb_dacvae --word-aug

# word-aug 없이 기본 학습
bash run.sh --encoder dac

# Stage 1만
bash run.sh --encoder fb_dacvae --stage 1 --word-aug

# 빠른 기능 테스트 (2 steps)
bash run.sh --encoder fb_dacvae \
    --wandb-mode disabled --max-steps 2 --datasets ls100 --stage all

# word-aug 기능 테스트
bash run.sh --encoder fb_dacvae \
    --wandb-mode disabled --max-steps 2 --datasets ls100 --stage 1 --word-aug
```

---

## 8. 체크포인트 경로

```
/mnt/tmp/cache/hf/{encoder}/
  s1_outputs_{run_id}/          ← Stage 1 Trainer outputs
    best_s1_proj.pt             ← best WER projector (Stage 2 로딩용)
    checkpoint-{N}/             ← Trainer step 체크포인트
  s2_outputs_{run_id}/          ← Stage 2 Trainer outputs
    checkpoint-{N}/             ← FSDP sharded 체크포인트
```

`run_id` = `MMDD_HHMM` (실행 시작 시각)
