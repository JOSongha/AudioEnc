# Plan 2 — Layer-wise Embedding Extraction (Representation Richness, Phase 1)

> 실행자: **Haiku**. Plan 1 산출물 (`cmu_arctic_7/manifest.csv`) 을 입력으로, **3 ALM family** (whisper-tiny, whisper-small, wavtok-40-unify) × 37 layers × 2 풀링 임베딩을 추출한다. **encodec-24k 는 Stage 1 ckpt 부재로 본 plan 에서 제외**. 분석은 Plan 3 에서.

---

## 목표

`cmu_arctic_7` 의 ~7,920 utterance 에 대해, 3 ALM family 의 **각 family 별 가장 최근 step 의 Stage 1 ckpt** 로 forward 하여:

- 각 family 당 **encoder out (1) + projector L1–L4 (4) + LLM L1–L32 (32) = 37 layers** 의 hidden_states 추출
- 풀링 2 종: **utt-mean** (모든 stage), **utt-last** (projector & LLM 만)
- frame-level (downsampled) 도 별도 저장 (Plan 3 의 spectrum 분석용)

산출물 경로:

```
experiments/representation_richness/cmu_arctic_7/
├── whisper_tiny/
│   ├── encoder_out/           {utt_id}.npz   (mean, frames)
│   ├── projector_L1/          {utt_id}.npz   (mean, last, frames)
│   ├── ...
│   ├── projector_L4/
│   ├── llm_L01/
│   ├── ...
│   └── llm_L32/
├── whisper_small/...
└── wavtok_40_unify/...
```

`encodec_24k/` 디렉토리는 만들지 않음 (ckpt 부재).

---

## 선행 조건

```bash
# Plan 1 완료 확인
test -f /mnt/tmp/cache/cmu_arctic_7/manifest.csv && wc -l /mnt/tmp/cache/cmu_arctic_7/manifest.csv

# Stage 1 ckpt 위치 — family 별 확인
for f in Qwen3.5_whisper_tiny_Stage1 Qwen3.5_whisper_small_Stage1 Qwen3.5_encodec_24k_Stage1 Qwen3.5_wavtok_40_unify_Stage1; do
  echo "=== $f ==="
  ls /mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/ckpts/$f/$f 2>/dev/null | grep -E "checkpoint-|safetensors|config.json" | head
done

# GPU 확인 — 1 장 이상 가용
nvidia-smi --query-gpu=index,name,memory.free --format=csv

# python 환경 — 학습 시 사용한 venv (audiollm-trainer 의 install.sh 가 만든 것) 와 동일해야
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

### Ckpt 위치 — 동적 판별

각 family 의 **가장 최근 step 의 checkpoint 디렉토리** 를 사용. Haiku 는 다음 로직으로 자동 결정:

```python
import os, re
def latest_ckpt(family_dir):
    # family_dir 예: external/ckpts/Qwen3.5_whisper_tiny_Stage1/Qwen3.5_whisper_tiny_Stage1
    cks = [d for d in os.listdir(family_dir) if re.match(r"checkpoint-\d+$", d)]
    if not cks: return None
    return os.path.join(family_dir, sorted(cks, key=lambda x: int(x.split("-")[1]))[-1])
```

현재 시점 스냅샷 (검증 후 그대로면 사용, 다른 step 으로 늘어났으면 그 step 사용):

| Family | Ckpt root | 현재 최신 step |
|---|---|---|
| whisper-tiny | `external/ckpts/Qwen3.5_whisper_tiny_Stage1/Qwen3.5_whisper_tiny_Stage1` | `checkpoint-13000` |
| whisper-small | `external/ckpts/Qwen3.5_whisper_small_Stage1/Qwen3.5_whisper_small_Stage1` | `checkpoint-13000` |
| wavtok-40 | `external/ckpts/Qwen3.5_wavtok_40_unify_Stage1/Qwen3.5_wavtok_40_unify_Stage1` | `checkpoint-100000` |
| encodec-24k | — | **N/A — Stage 1 ckpt 부재. 본 plan 에서 제외.** |

> **⚠️ Haiku 행동 지침**:
> 1. 시작 시 위 3 family 의 latest_ckpt() 결과를 **출력해서 사용자에게 보고**. 사용자 승인 (또는 사용자가 다른 step 지정) 후 진행.
> 2. encodec-24k 는 ckpt 가 추가됐는지 확인만 하고, 없으면 그대로 skip. 추출 스크립트 / 분석에서 encodec 제외.

---

## 절대 금지 / 주의사항

- ❌ ckpt 파일을 수정하거나 이동하지 말 것. 읽기 전용으로만 다룸.
- ❌ `model.train()` 호출 금지 — 추출은 `model.eval()` + `torch.no_grad()` 만.
- ❌ projector 만 학습된 Stage 1 ckpt 이므로, encoder/LLM 가중치는 base model (`external/models/Qwen3.5AE-4B-{family}`) 의 것과 동일해야 함. 로딩 시 mismatch 가 보이면 보고.
- ❌ 멀티 family 동시 GPU 로딩 금지. 4B + projector + audio encoder 가 fp16/bf16 로도 ~10 GB. **한 family 끝나면 메모리 비우고 다음**.
- ⚠️ 풀 추출은 utterance 7,924 × forward 시간으로 family 당 30 분~수시간. **반드시 smoke test 먼저** (§Phase 2.3).
- ⚠️ Frame-level 저장은 디스크 비용 큼 — 다음 §2.4.E 의 quantization/downsample 규약 준수.
- ⚠️ 진행률은 매 100 utterance 마다 stdout. tqdm 권장.
- ⚠️ utterance 별로 즉시 disk 에 저장 (in-memory 누적 금지). OOM / 크래시 시 작업 손실 방지.

---

## TaskCreate 트래킹

```
1. ckpt 위치 검증 (latest step 결정 → 사용자 보고)
2. 모델 로딩 코드 작성 (load_alm.py)
3. hook 기반 layer hidden 추출 코드 (extract.py)
4. Smoke test (2 화자 × 2 utt, whisper-tiny 로)
5. whisper-tiny 전체 추출
6. whisper-small 전체 추출
7. wavtok-40 전체 추출
8. 산출물 정합성 검증
```

---

## Phase 2.1 — 모델 로딩 코드

**파일**: `experiments/representation_richness/load_alm.py` (신규)

```python
"""
Load Stage 1 ALM (encoder + projector + frozen LLM) for any family.

Usage:
    from load_alm import load_alm
    model, processor = load_alm("whisper_tiny", device="cuda:0", dtype=torch.bfloat16)
"""
```

핵심 사항:

- `transformers.AutoModelForCausalLM.from_pretrained(ckpt_path, trust_remote_code=True, torch_dtype=torch.bfloat16)` — auto_map 이 이미 config 에 있어서 custom class 가 자동 로드됨.
- ckpt_path 는 base model 디렉토리가 아니라 **Stage 1 의 checkpoint-XXXXX** 경로. base model 의 `audio_encoder.py`, `configuration_qwen3_5AE.py`, `modeling_qwen3_5AE.py` 가 ckpt 디렉토리에 함께 있는지 확인. 없으면 base 디렉토리에서 복사 (Trainer 가 안 복사한 케이스).
  - whisper-tiny/small ckpt 는 `audio_encoder.py` 가 없을 가능성 — 확인 후 base 에서 symlink:
    ```bash
    ckpt=external/ckpts/Qwen3.5_whisper_tiny_Stage1/Qwen3.5_whisper_tiny_Stage1/checkpoint-13000
    base=external/models/Qwen3.5AE-4B-whisper-tiny
    for f in audio_encoder.py configuration_qwen3_5AE.py modeling_qwen3_5AE.py chat_template.jinja; do
      [ -f "$ckpt/$f" ] || ln -sf "$base/$f" "$ckpt/$f"
    done
    ```
- 로딩 후 `model.eval()`. requires_grad 모두 끔.
- `model` 의 sub-module 위치 (encoder / projector / LLM 본체) 를 확인하고 `family_arch.json` 같은 곳에 매핑 기록 (hook 부착에 필요). 일반적으로:
  - `model.audio_encoder.encoder` — 실제 encoder (Whisper/EnCodec/...)
  - `model.audio_encoder.projector.layers[i]` — projector LlamaDecoderLayer i (i=0..3)
  - `model.model.layers[i]` 또는 `model.language_model.layers[i]` — LLM transformer layer i (i=0..31)
  - 정확한 attribute name 은 `print(model)` 1 회 출력 후 확인.

---

## Phase 2.2 — Hook 기반 layer 추출

**파일**: `experiments/representation_richness/extract.py` (신규)

표준 HF `output_hidden_states=True` 가 audio_encoder + LLM 통합 forward 에서 일관되게 동작한다는 보장이 없음 → **forward hook** 으로 안전하게 추출.

### 2.2.A — Hook 부착

```python
# 의사코드
captured = {}  # layer_name -> tensor (B, T, d) on cpu/fp16
def make_hook(name):
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        captured[name] = h.detach().to(torch.float16).cpu()
    return hook

# encoder out: encoder 모듈의 forward output
hooks = []
hooks.append(model.audio_encoder.encoder.register_forward_hook(make_hook("encoder_out")))
for i, layer in enumerate(model.audio_encoder.projector.layers):
    hooks.append(layer.register_forward_hook(make_hook(f"projector_L{i+1}")))
for i, layer in enumerate(model.model.layers):  # 정확한 path 는 §2.1 에서 확인
    hooks.append(layer.register_forward_hook(make_hook(f"llm_L{i+1:02d}")))
```

> **⚠️ Hook 의 output 형식 검증**: `LlamaDecoderLayer` 의 forward 는 tuple 반환. `output[0]` 이 hidden_states. 첫 utterance forward 후 captured 의 모든 entry shape 와 dtype 을 print 해서 사용자에게 한 번 보여줄 것.

### 2.2.B — Forward 호출 (audio-only 입력)

```python
# 입력: audio waveform 만. text/prompt 없음.
# Stage 1 학습 시 omni dataset 의 처리 방식을 따라가야 — load_alm 에서 함께 제공하는 processor 사용.
# audio span position 추적: audio_embeds 가 들어간 token index range 를 모델이 알려주거나, processor 에서 명시적으로.
audio = load_audio_16k(path)        # (S,) numpy
inputs = processor(audio, return_tensors="pt").to(device)
with torch.no_grad():
    _ = model(**inputs, output_hidden_states=False)  # hidden_states 는 hook 으로 회수
```

> **❗ 학습 코드 참조 필수**: `src/llamafactory` 의 omni dataset / collator / template 에서 audio-only forward 를 어떻게 만드는지 확인하고 동일한 pre-processing 적용. 임의 추측 금지. 막히면 보고.

### 2.2.C — Audio span hidden 슬라이싱 (LLM only)

LLM hidden 은 (B, T_total, 2560) 이고 T_total 에는 BOS, audio span, (audio_only 이므로 EOS 정도까지). audio span 의 [start, end) index 를 알아야 함:

- 모델/template 이 audio token 위치를 attribute (e.g. `audio_token_indices`) 로 제공한다면 그것 사용.
- 없다면 input_ids 에서 `<|audio_start|>` / `<|audio_end|>` token id 를 찾아 그 사이 인덱스 사용. tokenizer 에서 이 special token id 확인 1 회.
- 첫 utterance forward 후 audio span 길이가 T_audio_frame 과 일치하는지 검증 (encoder out 의 시간 길이와 같아야 함, projector 통과 후 길이 변화 없음).

### 2.2.D — 풀링

```python
# valid frames mask 만들어 두기 (Whisper 30s padding 고려)
valid_frames = ...  # bool (T,)
def pool_mean(h, mask):
    return (h[mask].mean(dim=0)).numpy()  # (d,)
def pool_last(h, mask):
    last_idx = mask.nonzero()[-1].item()
    return h[last_idx].numpy()  # (d,)
```

- encoder_out: mean only (non-causal)
- projector_L*, llm_L*: mean + last 둘 다

### 2.2.E — Frame 저장 (downsample)

원본 frame-level 을 그대로 저장하면 disk 비용 폭발. 다음 규약:

- 시간축 downsample: max 256 frames per utterance (긴 utterance 는 균일 간격 sampling).
- dtype: float16.
- 저장 형식: `np.savez_compressed`.
- 한 utterance × 한 layer 당 파일 1 개:
  ```
  {layer_dir}/{utt_id}.npz
    keys:
      mean:  (d,) float32
      last:  (d,) float32   # encoder_out 은 없음
      frames: (T_ds, d) float16  # T_ds <= 256
      n_valid: int           # 원래 valid frame 수
  ```

> **디스크 추산**: utt 당 layer 평균 파일 ~50 KB × 37 layers × 7920 utts × 3 family ≈ 45 GB. /mnt/tmp 3 TB 에 충분.

---

## Phase 2.3 — Smoke test (필수, 풀 추출 전)

```bash
# 2 화자 × 2 utt 만 추출
python experiments/representation_richness/extract.py \
    --family whisper_tiny \
    --manifest /mnt/tmp/cache/cmu_arctic_7/manifest.csv \
    --speakers bdl,slt \
    --max_utt_per_speaker 2 \
    --out_dir experiments/representation_richness/_smoke/whisper_tiny
```

**검증**:

- 4 utterance × 37 layers = 148 npz 파일 생성됐는지
- 임의 1 개 파일 로드해서 shape 확인:
  - encoder_out: mean.shape = (d_enc,), n_valid > 0
  - projector_L4: mean.shape = (512,), last.shape = (512,)
  - llm_L32: mean.shape = (2560,), last.shape = (2560,)
- NaN / Inf 검사 (`np.isfinite(arr).all()`)

검증 통과 시에만 다음 Phase 진행. 실패 시 보고.

---

## Phase 2.4 — Family 별 풀 추출

순서: whisper_tiny → whisper_small → wavtok_40_unify. (encodec_24k 제외)

각 family 당 명령:

```bash
python experiments/representation_richness/extract.py \
    --family <family> \
    --manifest /mnt/tmp/cache/cmu_arctic_7/manifest.csv \
    --out_dir experiments/representation_richness/cmu_arctic_7/<family> \
    --batch_size 1     # OOM 위험 시 1 부터, 안전 시 4 까지
```

**모니터링**:

- 매 500 utterance 마다 진행률 + ETA 출력
- 1 family 끝날 때마다 GPU memory 비우고 (`del model; torch.cuda.empty_cache()`) 다음 family 로
- 1 family 끝난 직후: `du -sh out_dir` 로 디스크 사용량 보고

**실패 시**:

- OOM: batch_size 낮추고 재시작. resume 지원 (이미 있는 .npz 는 skip).
- 특정 utt 에서 NaN 발생: 그 utt 만 skip 하고 로그에 기록, 진행 계속.
- Hook 에서 layer 못 찾음: `print(model)` 결과 첨부해서 사용자에게 보고.

---

## Phase 2.5 — 산출물 정합성 검증

```bash
# 1. family × layer 디렉토리 모두 존재
for fam in whisper_tiny whisper_small wavtok_40_unify; do
  for lyr in encoder_out projector_L1 projector_L2 projector_L3 projector_L4 \
             llm_L01 llm_L02 ... llm_L32; do
    n=$(ls experiments/representation_richness/cmu_arctic_7/${fam}/${lyr}/*.npz 2>/dev/null | wc -l)
    echo "${fam}/${lyr}: ${n}"
  done
done
# 기대: 모두 ~7920 (utt 수). 한 layer 라도 0 이면 실패.

# 2. 임의 sampled npz 50 개 NaN 검사
python experiments/representation_richness/_verify_npz.py
```

`_verify_npz.py` 는 다음을 확인:

- 모든 npz 가 mean key 보유, encoder_out 외에는 last key 도 보유
- 모든 array 가 finite
- frame array 의 시간 길이가 256 이하

---

## 완료 조건

- [ ] Smoke test 통과
- [ ] 3 family (whisper-tiny, whisper-small, wavtok-40-unify) 의 모든 layer 에 대해 ~7920 npz 생성
- [ ] `_verify_npz.py` 통과
- [ ] 사용자에게 사용된 ckpt step 이 family 별로 명시 보고됨

---

## Plan 3 로 넘기는 산출물

```
experiments/representation_richness/cmu_arctic_7/{family}/{layer}/{utt_id}.npz
experiments/representation_richness/load_alm.py
experiments/representation_richness/extract.py
experiments/representation_richness/_verify_npz.py
```

추출 시간 / 디스크 사용량 / family × layer 별 utt count 표 한 번 더 보고.
