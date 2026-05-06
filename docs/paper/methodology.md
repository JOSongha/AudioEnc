# Methodology — Qwen3.5AE-4B ASR Stage 1 (Projector Alignment)

> 현재 `acoustic` branch 의 `stage1_projector.yaml` + `omni_dataset.py` 기반 실제 코드 경로를 reference 로 기술.
> 인용은 §8 References 에 BibTeX 키로 매핑 ([foo24] 형식). 확정되지 않은 출처는 `[?]` 로 표기.

---

## 1. Overview

우리는 frozen large language model (LLM) 에 raw waveform-level audio representation 을 투사하는 **encoder-projector-LLM** 구조를 채택한다 [FLM25, SLAM24]. Stage 1 (본 문서) 에서는 **projector 만** 학습하여 audio encoder 의 continuous latent 를 LLM 의 text embedding 공간에 정렬한다. Encoder 와 LLM 의 모든 파라미터는 동결된다.

파이프라인:

```
raw waveform (48 kHz)
   → DACVAE.encode()            [frozen, ~27.55M active params]
   → (B, T_audio, 128)          @ 25 fps
   → AudioProjector             [trainable, 18.16M params]
   → (B, T_audio, 2560)
   → inject at <|audio_pad|> positions
   → Qwen3.5-4B LLM             [frozen]
   → cross-entropy loss on transcript + EOS
```

총 학습 파라미터는 **18,158,080** (≈ 0.45 % of Qwen3.5-4B LLM 기준).

---

## 2. Audio encoder — DACVAE (frozen)

Encoder 는 `facebook/dacvae-watermarked` 체크포인트 [DAC23, AS24] 의 `encode()` 경로를 그대로 사용한다. 원본 DACVAE 는 VAE variant 의 Descript Audio Codec [DAC23] 을 acoustic watermarking extension [AS24] 과 함께 재학습한 모델이다. 구체적 forward 경로:

1. 48 kHz mono waveform 을 `encoder` (SEANet-style conv stack [SS22] with Snake1d activation [BDS20]) 로 downsample. encoder stride `[2, 8, 10, 12]` 로 1920× downsampling → **25 fps**.
2. VAE bottleneck 의 `in_proj` (1024 → 2×128) 가 mean/log-scale 으로 분기 후 reparameterize → **128-dim continuous latent** `z`.
3. Decoder, watermarker, `out_proj` 는 forward 에서 호출되지 않음 (체크포인트에만 상주, Stage 1 메모리 점유 ~80 M dead weight).

실제 forward 파라미터 수:

| 구성 | 파라미터 | 비고 |
|---|---|---|
| encoder | 27,288,704 | conv + Snake1d |
| quantizer.in_proj | 262,656 | VAE 의 mean/log-scale 투사 |
| **encode-path total** | **27,551,360** | |

---

## 3. AudioProjector (trainable)

Projector 는 4-layer causal Llama decoder adapter [Ll23] 로 구성된다:

```
input_proj:   Linear(128 → 512)         65,536 params
layers × 4:   LlamaDecoderLayer (hidden=512, heads=8, ffn=2048)  16,781,312 params
final_norm:   LlamaRMSNorm(512)         512 params
output_proj:  Linear(512 → 2560)        1,310,720 params
TOTAL                                   18,158,080 params
```

각 `LlamaDecoderLayer` 는 pre-norm RMSNorm [ZL20], SwiGLU feed-forward [Sh20], RoPE positional encoding [SP23], causal multi-head attention 을 포함한다. Attention 은 FlashAttention-2 [Dao23] varlen 구현을 사용한다.

---

## 4. LLM — Qwen3.5-4B (frozen)

Backbone 은 sanghyuk 이 Qwen3.5-4B 을 기반으로 수정한 **Qwen3.5AE** 아키텍처 [QW24, ?] 로, 32-layer mixed attention (linear attention × 24 + full attention × 8), hidden size 2560, vocabulary 248,320, bf16 weight. Text-only 성능은 원 Qwen3.5-4B 대비 유의미한 손실 없음을 사전 검증했다 [sanghyuk-internal].

Audio embedding 은 tokenizer 의 special token `<|audio_pad|>` (id = 248,076) 위치에 주입된다. 이는 Qwen2-Audio [QA24] 의 `<|audio_start|>` / `<|audio_end|>` convention 을 따른다.

---

## 5. Input Format & Tokenization

ChatML [OAI23] template 으로 각 샘플을 래핑:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|audio_start|>[<|audio_pad|>] × t_audio <|audio_end|>Transcribe the audio to text.<|im_end|>
<|im_start|>assistant
<transcript><eos>
```

where `t_audio = floor(num_samples / hop_length)` with `hop_length = 1920` (DACVAE encoder stride product). 구현: [omni_dataset.py:422-440](../../src/llamafactory/data/omni_dataset.py#L422-L440).

### Label masking (loss computation)

Training objective 는 transcript + EOS 위치에서의 causal language modeling cross-entropy loss 이다. 이외 모든 위치 (system prompt, audio placeholder, user instruction, assistant prefix) 는 HuggingFace 관례인 `IGNORE_INDEX = -100` [HF23] 로 마스킹된다.

| Position | `input_ids` | `labels` | loss contribution |
|---|---|---|---|
| ChatML prefix | prefix tokens | `IGNORE_INDEX` | ❌ |
| Audio placeholder | `<|audio_pad|>` × `t_audio` | `IGNORE_INDEX` | ❌ (embedding 은 projector output 으로 대체) |
| ChatML mid | mid tokens | `IGNORE_INDEX` | ❌ |
| Transcript | text_ids | text_ids | ✅ |
| EOS | `eos_token_id` | `eos_token_id` | ✅ |

세부 처리 규약은 [`eos_pad_token_handling.md`](../reference/eos_pad_token_handling.md) 에 정리.

---

## 6. Sequence Packing

학습 효율성을 위해 여러 샘플을 greedy knapsack [HF-packing] 으로 `cutoff_len = 3,584` 토큰까지 묶는 **neat packing** [Kundu24] 을 사용한다.

### 6.1 Packing ([omni_dataset.py:485-586](../../src/llamafactory/data/omni_dataset.py#L485-L586))

1. 샘플을 token 길이 기준 정렬.
2. 빈 knapsack 을 하나 열고, 남은 capacity 에 들어갈 수 있는 가장 큰 샘플을 bisect 로 찾아 집어넣음 (first-fit decreasing 변형).
3. Knapsack 이 더 못 받으면 새 bin 오픈.
4. Tail pad 로 `cutoff_len` 맞춤 — pad token 은 **tokenizer.pad_token_id** (audio_pad 아님).

### 6.2 Intra-packed attention isolation

`neat_packing = True` 조건에서 `attention_mask` 값을 `[1, 2, 3, ...]` (샘플 index 기반 labeling) 으로 설정하여, 같은 label 을 가진 토큰끼리만 attend 하게 한다 ([omni_dataset.py:552-555](../../src/llamafactory/data/omni_dataset.py#L552-L555)). Pad 구간은 `0`. 이는 Packed-Attention [Kr23, Kundu24] 의 block-diagonal causal mask 구현이다.

### 6.3 FlashAttention-2 varlen unpadding

FA2 backend 에서는 전체 padding 을 제거하고 모든 valid token 을 1D 로 평탄화한 뒤 `cu_seqlens` 로 샘플 경계를 전달한다 ([omni_dataset.py:661-682](../../src/llamafactory/data/omni_dataset.py#L661-L682)) [Dao23]. 이때 각 샘플의 첫 토큰 (`position_ids == 0`) 위치의 label 을 `IGNORE_INDEX` 로 덮어 샘플 경계를 넘는 loss shift 를 방지한다.

---

## 7. Training Configuration

### 7.1 Frozen vs trainable parameters

Stage 1 workflow ([workflow.py:67-71](../../src/llamafactory/train/omni/workflow.py#L67-L71)) 는 `audio_encoder.projector` 또는 LoRA 어댑터에 해당하는 파라미터만 `requires_grad=True` 로 두고 나머지는 모두 freeze (Stage 1 은 projector 만, Stage 2 는 projector + LoRA):

```python
for name, param in model.named_parameters():
    require_grad = ("audio_encoder.projector" in name) or ("lora_" in name)
    param.requires_grad = require_grad
```

학습 대상: 18,158,080 parameters (AudioProjector 전체).

### 7.2 Optimizer & schedule

| 항목 | 값 | 근거 |
|---|---|---|
| Optimizer | AdamW [LH19], FusedAdam kernel [DS20] | 표준 Transformer 학습 |
| β1, β2 | 0.9, 0.999 (HF default) | — |
| weight_decay | 0.01 | HF default |
| learning_rate | 2.0 × 10⁻⁴ | projector adapter 학습의 통상 range [SLAM24, FLM25] |
| Scheduler | `warmup_stable_decay` [HF23] | 안정 플래토 확보 |
| Warmup steps | 1,000 | — |
| Max steps | 100,000 | ≈ 1.8 epochs on 11.35 M entries |
| Precision | bf16 [Ka20] | Qwen3.5 는 fp16 에서 SSM activation NaN 발생 (내부 검증) |

### 7.3 Distributed training

- **DeepSpeed ZeRO Stage 2** [Ra20, DS20] with optimizer state & gradient sharding.
- **Gradient checkpointing** ON (default `disable_gradient_checkpointing: false`) — projector activation memory 절약.
- `NNODES × 8 GPUs` × `per_device_train_batch_size=3` × `gradient_accumulation_steps=1` → **24 packed sequences per step** per node.
- Node-group NCCL with `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800`.

### 7.4 System-level optimizations

- **FlashAttention-2** [Dao23] for attention (varlen 포맷, packed sequence 호환).
- **Liger Kernel** [Ho24] for fused SwiGLU + RMSNorm + RoPE operations, enabled via `enable_liger_kernel: true`.
- **CUDA allocator** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to mitigate fragmentation under long packed sequences.
- **`LD_PRELOAD` glibc shim** for flash_attn compatibility on host glibc 2.31 (flash_attn 2.8.3 wheel 은 2.32+ 요구).

---

## 8. Data

### 8.1 Sources

Stage 1 training set 은 공개 영어 ASR 코퍼스 3종의 혼합:

| Dataset | Hours | Reference |
|---|---|---|
| LibriTTS-R | 552.42 | [Ko23] |
| Multilingual LibriSpeech (English subset) | 45,000.00 | [Pr20] |
| VoxPopuli (English) | 522.00 | [Wa21] |
| **Total** | **46,074** | |

Nubes [?] 오브젝트 스토리지 (`hyperscaleai-audiollm/datasets/public/16kHz/...`) 에서 lazy streaming 으로 로드하며, 16 kHz 로 저장된 경우 dataloader 에서 48 kHz 로 on-the-fly resample 한다 ([omni_dataset.py:406-407](../../src/llamafactory/data/omni_dataset.py#L406-L407)).

### 8.2 Manifest

[qwen3_5_dacvae_asr_shuffled_128 — sanghyuk] 의 pre-shuffled 128-shard JSONL manifest (원본 20.67 M entries; CommonVoice + GigaSpeech 포함) 에서 **Libri + MLS + VoxPopuli 만 필터링** 하여 파생 manifest 를 생성했다 (11,345,224 entries, 54.9% retained). 필터 스크립트는 [`filter_libri_mls_vox.py`](../../../AudioEnc/log/tmp/datasets/filter_libri_mls_vox.py) 참조.

각 샘플은 `{"nubes_path": ..., "text": ...}` JSONL 한 줄. `text` 는 원본 transcript (normalization 미적용 — Qwen3.5 tokenizer 가 casing/punctuation 모두 수용).

### 8.3 Streaming pipeline

- HuggingFace `IterableDataset` + `.map(process_samples)` + `.map(pack_samples, batched=True)` [HF23].
- `split_dataset_by_node(num_shards=128)` → 8 GPU 에 16 shards 씩 distribute.
- `omni_max_audio_samples = 2,160,000` (= 45 s @ 48 kHz) 로 샘플 길이 상한.
- `omni_packing_bucket_size = 128` entries 버퍼 단위로 packing.
- Audio download concurrency: ThreadPoolExecutor(`max_workers=12`).

---

## 9. Validation of Data-Model Consistency

학습 안정성 조건 (Stage 1 전 단계에서 모든 batch 에서 반드시 성립):

> `(input_ids == audio_pad_token_id).sum()` == `sum(audio_lengths)` == `projector_output.shape[0]`

즉 placeholder 토큰 개수 = 실제 audio frame 개수 = projector output frame 개수. Whisper 등 30 초 zero-pad 인코더 교체 시 이 조건이 깨질 수 있어 encoder output 을 `t_audio` 기준 slice 해야 한다 (Whisper migration 계획: [`plan.md`](../stage1/whisper/plan.md) §12.1).

---

## 10. References

인용은 다음 BibTeX 키로 구분 (`[?]` 는 출처 확정 필요):

```bibtex
% -- Audio encoders --
@inproceedings{DAC23,
  author    = {Kumar, Rithesh and Seetharaman, Prem and Luebs, Alejandro and Kumar, Ishaan and Kumar, Kundan},
  title     = {High-Fidelity Audio Compression with Improved {RVQGAN}},
  booktitle = {NeurIPS},
  year      = {2023}
}

@inproceedings{AS24,
  author    = {{San Roman}, Robin and D{\'e}fossez, Alexandre and Adi, Yossi and Synnaeve, Gabriel},
  title     = {{AudioSeal}: Proactive Detection of Voice Cloning with Localized Watermarking},
  booktitle = {ICML},
  year      = {2024}
}

@article{Df22,
  author    = {D{\'e}fossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
  title     = {High Fidelity Neural Audio Compression},
  journal   = {TMLR},
  year      = {2023}
}

@article{Wh23,
  author    = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  title     = {Robust Speech Recognition via Large-Scale Weak Supervision},
  journal   = {ICML},
  year      = {2023}
}

@article{Mi24,
  author    = {D{\'e}fossez, Alexandre and others},
  title     = {Moshi: A Speech-Text Foundation Model for Real-Time Dialogue},
  journal   = {arXiv preprint arXiv:2410.00037},
  year      = {2024}
}

% -- LLM backbones --
@article{QW24,
  author    = {Qwen Team},
  title     = {Qwen Technical Report},
  journal   = {arXiv preprint arXiv:2412.15115},
  year      = {2024}
}

@article{QA24,
  author    = {Chu, Yunfei and others},
  title     = {{Qwen2-Audio} Technical Report},
  journal   = {arXiv preprint arXiv:2407.10759},
  year      = {2024}
}

@article{Ll23,
  author    = {Touvron, Hugo and others},
  title     = {{LLaMA}: Open and Efficient Foundation Language Models},
  journal   = {arXiv preprint arXiv:2302.13971},
  year      = {2023}
}

% -- Related speech-LLM architectures --
@inproceedings{FLM25,
  author    = {others},
  title     = {Frozen Large Language Models Can Perceive Paralinguistic Aspects of Speech},
  booktitle = {Interspeech},
  year      = {2025}
}

@article{SLAM24,
  author    = {Ma, Ziyang and others},
  title     = {An Embarrassingly Simple Approach for {LLM} with Strong {ASR} Capacity},
  journal   = {arXiv preprint arXiv:2402.08846},
  year      = {2024},
  note      = {SLAM-ASR}
}

% -- Datasets --
@inproceedings{Pa15,
  author    = {Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  title     = {{LibriSpeech}: An {ASR} Corpus based on Public Domain Audio Books},
  booktitle = {ICASSP},
  year      = {2015}
}

@inproceedings{Ko23,
  author    = {Koizumi, Yuma and others},
  title     = {{LibriTTS-R}: A Restored Multi-Speaker Text-to-Speech Corpus},
  booktitle = {Interspeech},
  year      = {2023}
}

@inproceedings{Pr20,
  author    = {Pratap, Vineel and Xu, Qiantong and Sriram, Anuroop and Synnaeve, Gabriel and Collobert, Ronan},
  title     = {{MLS}: A Large-Scale Multilingual Dataset for Speech Research},
  booktitle = {Interspeech},
  year      = {2020}
}

@inproceedings{Wa21,
  author    = {Wang, Changhan and Rivi{\`e}re, Morgane and Lee, Ann and Wu, Anne and Talnikar, Chaitanya and Haziza, Daniel and Williamson, Mary and Pino, Juan and Dupoux, Emmanuel},
  title     = {{VoxPopuli}: A Large-Scale Multilingual Speech Corpus for Representation Learning, Semi-Supervised Learning and Interpretation},
  booktitle = {ACL},
  year      = {2021}
}

% -- Training infrastructure --
@inproceedings{Dao23,
  author    = {Dao, Tri},
  title     = {{FlashAttention-2}: Faster Attention with Better Parallelism and Work Partitioning},
  booktitle = {ICLR},
  year      = {2024}
}

@inproceedings{Ra20,
  author    = {Rasley, Jeff and Rajbhandari, Samyam and Ruwase, Olatunji and He, Yuxiong},
  title     = {{DeepSpeed}: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters},
  booktitle = {KDD},
  year      = {2020}
}

@misc{DS20,
  author       = {{DeepSpeed Team}},
  title        = {{FusedAdam} — {DeepSpeed} CUDA fused Adam optimizer},
  howpublished = {\url{https://github.com/deepspeedai/DeepSpeed}},
  year         = {2020}
}

@misc{Ho24,
  author       = {Hsu, Byron and others},
  title        = {{Liger Kernel}: Efficient {T}riton Kernels for {LLM} Training},
  howpublished = {\url{https://github.com/linkedin/Liger-Kernel}},
  year         = {2024}
}

@misc{HF23,
  author       = {{Hugging Face}},
  title        = {{transformers} Library — Trainer, Dataset, Tokenizer APIs},
  howpublished = {\url{https://github.com/huggingface/transformers}},
  year         = {2019--}
}

@inproceedings{Kundu24,
  author    = {Kundu, Achintya and others},
  title     = {Enhancing Training Efficiency Using Packing with Flash Attention},
  booktitle = {arXiv:2407.09105},
  year      = {2024}
}

@inproceedings{Kr23,
  author    = {Krell, Mario and others},
  title     = {Efficient Sequence Packing without Cross-Contamination},
  booktitle = {arXiv:2107.02027},
  year      = {2021}
}

% -- Core building blocks --
@article{Va17,
  author    = {Vaswani, Ashish and others},
  title     = {Attention Is All You Need},
  journal   = {NeurIPS},
  year      = {2017}
}

@inproceedings{LH19,
  author    = {Loshchilov, Ilya and Hutter, Frank},
  title     = {Decoupled Weight Decay Regularization},
  booktitle = {ICLR},
  year      = {2019}
}

@inproceedings{Ka20,
  author    = {Kalamkar, Dhiraj and others},
  title     = {A Study of {BFLOAT16} for Deep Learning Training},
  booktitle = {arXiv:1905.12322},
  year      = {2019}
}

@article{ZL20,
  author    = {Zhang, Biao and Sennrich, Rico},
  title     = {Root Mean Square Layer Normalization},
  journal   = {NeurIPS},
  year      = {2019}
}

@article{Sh20,
  author    = {Shazeer, Noam},
  title     = {{GLU} Variants Improve Transformer},
  journal   = {arXiv preprint arXiv:2002.05202},
  year      = {2020}
}

@article{SP23,
  author    = {Su, Jianlin and Lu, Yu and Pan, Shengfeng and Wen, Bo and Liu, Yunfeng},
  title     = {{RoFormer}: Enhanced Transformer with Rotary Position Embedding},
  journal   = {Neurocomputing},
  year      = {2024}
}

@inproceedings{BDS20,
  author    = {Ziyin, Liu and Hartwig, Tilman and Ueda, Masahito},
  title     = {Neural Networks Fail to Learn Periodic Functions and How to Fix It ({Snake} activation)},
  booktitle = {NeurIPS},
  year      = {2020}
}

@inproceedings{SS22,
  author    = {D{\'e}fossez, Alexandre and others},
  title     = {{SEANet}: A Multi-modal Speech Enhancement Network},
  booktitle = {INTERSPEECH},
  year      = {2022},
  note      = {SEANet conv backbone is reused by EnCodec and DACVAE}
}

@misc{OAI23,
  author       = {{OpenAI}},
  title        = {{ChatML} — Chat Markup Language},
  howpublished = {\url{https://github.com/openai/openai-python/blob/main/chatml.md}},
  year         = {2023}
}
```

### 확정 필요 citations

- `[?]` (Nubes storage) — Naver 내부 시스템, 외부 인용 불가. 논문 작성 시 "internal object storage" 로 기술하고 인용 제거.
- `[sanghyuk-internal]` — sanghyuk 의 Qwen3.5 → Qwen3.5AE 수정 작업 자체에 대한 인용. 내부 문서나 기술 노트 있으면 그걸로 교체.
- `[Sh20]` SwiGLU 의 원 citation 은 `GLU Variants Improve Transformer` (Shazeer 2020). SwiGLU 가 LLaMA 에서 채택되어 표준화된 후 [Ll23] 로도 인용 가능.
- `[BDS20]` Snake1d activation 은 원 저자 정보 확인 필요 — BigVGAN [Lee23] 에서 재조명된 케이스 인용이 일반적일 수 있음.
- `[SS22]` SEANet 원 논문 — SoundStream [Zh21] 기반이며 EnCodec 과 DACVAE 모두 차용. 정확한 SEANet 원 논문 확인 필요.
- `[FLM25]` 저자 명단 채우기 — 사용자 선행 연구. 정확한 저자 리스트로 교체.
- `[QW24]` Qwen3.5-4B 에 대응하는 technical report 최신본. 3.5 전용 리포트가 없으면 Qwen2 report [QA24] + 3.5 model card 로 대체.
- `[Dao23]` FlashAttention-2 는 ICLR 2024 기재가 정확 — 키 이름과 연도 맞추기.

---

## 11. Appendix — Hyperparameter table

| Category | Key | Value |
|---|---|---|
| Model | Encoder | `facebook/dacvae-watermarked` (frozen) |
| Model | LLM | Qwen3.5-4B based `Qwen3.5AE-4B` (frozen) |
| Model | Projector | 4-layer Llama decoder (h=512, heads=8, ffn=2048) |
| Data | Datasets | LibriTTS-R + MLS-en + VoxPopuli-en (46,074 h) |
| Data | Manifest | `libri_mls_vox_shuffled_128` (11.35 M entries, 128 shards) |
| Data | Sample rate | 48,000 Hz |
| Data | Max audio duration | 45 s (`omni_max_audio_samples = 2,160,000`) |
| Data | Audio fps | 25 (hop 1920) |
| Training | Precision | bfloat16 |
| Training | Optimizer | FusedAdam (DeepSpeed) |
| Training | Learning rate | 2 × 10⁻⁴ |
| Training | Weight decay | 0.01 |
| Training | Scheduler | `warmup_stable_decay` |
| Training | Warmup steps | 1,000 |
| Training | Max steps | 100,000 |
| Training | Per-device batch size | 3 |
| Training | Grad accumulation | 1 |
| Training | Packed sequence length | 3,584 tokens |
| Training | Packing bucket size | 128 |
| Training | Distributed | DeepSpeed ZeRO-2, 8 GPUs × N nodes |
| Training | Gradient ckpt | enabled |
| System | Attention | FlashAttention-2 (varlen) |
| System | Kernel fusion | Liger |
| System | CUDA allocator | `expandable_segments:True` |
