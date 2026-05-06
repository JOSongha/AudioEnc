import math
import os
import re


def _infer_llm_family(llm_model: str) -> str:
    """`cfg["llm_model"]` → "Qwen3.5" | "Qwen3". (확장 시 여기만 건드리면 됨)"""
    s = (llm_model or "")
    if "Qwen3.5" in s or "qwen3_5" in s.lower():
        return "Qwen3.5"
    if "Qwen3" in s or "qwen3" in s.lower():
        return "Qwen3"
    # Fallback: 원본 문자열 그대로 두되 wandb 라벨만 ambiguous.
    return s.split("/")[-1].split("-")[0] or "LLM"


def _infer_llm_tag(llm_model: str) -> str:
    """`Qwen/Qwen3-1.7B` → "1.7b" / `Qwen/Qwen3.5-2B` → "2b" 처럼 사이즈 토큰 추출.

    Qwen family 의 실제 size 토큰(`-?(\\d+(?:\\.\\d+)?)[Bb]`)을 정규식으로 잡는다.
    """
    m = re.search(r"(\d+(?:\.\d+)?)[Bb]\b", llm_model or "")
    if m:
        return f"{m.group(1)}b"
    return "unk"

# ==========================================
# 학습 하이퍼파라미터 (encoder 무관)
# ==========================================
TRAIN_CONFIG = {
    # llm_type: "instruct" → ChatML 프롬프트, "base" → 단순 prefix
    "llm_type":  "base",
    "llm_model": "Qwen/Qwen3.5-2B",

    "gradient_accumulation_steps": 4,

    "stage1_lr": 2e-4,   # baseline
    "stage1_epochs": 3,

    # ── Projector collapse 대응 옵션 (§42+) — §43.5 baseline 로 되돌림 ───────
    # proj_norm_mode: "ln" (default, γ=1) | "ln_small_gamma" | "none"
    # 이전 "none" / "ln_small_gamma" 는 γ lock-in 방어용 실험. 기본값 "ln" 로 복원.
    "proj_norm_mode":        "ln",
    # stage1_diversity_reg: 0.0 = off
    "stage1_diversity_reg":  0.0,
    # §43.5 stage1 LR scheduler: §42 이전 legacy "constant" 로 복원 (cosine+warmup 제거).
    "stage1_lr_scheduler_type": "constant",

    "stage2_lr": 2e-5,
    "stage2_epochs": 3,

    "max_grad_norm": 1.0,
    "warmup_ratio": 0.0,

    "max_audio_len": 16000 * 20,
    "max_text_len": 256,

    # DynamicBatchSampler 설정
    # max_batch_tokens: 배치 내 총 LLM 토큰 수 상한 (오디오+텍스트 토큰 합계)
    #   get_config()에서 encoder hop/sr/stride 기준으로 자동 계산됨
    #   기본값 = 6 × (최대 오디오 토큰 + max_text_len)

    "data_path": "/mnt/tmp/cache",
    "mls_data_path": "/mnt/tmp/cache",
    "mls_num_samples": None,           # None = 전체 사용
    # Stage 1 서브샘플: LibriSpeech ~200h (~58k utterances)
    "stage1_librispeech_num_samples": None,
    "stage1_mls_num_samples": None,    # None = 전체 사용
    "model_cache_dir": "/mnt/tmp/cache/hf",
    "wandb_mode": "online",

    # "eval_steps":35,
    # "save_steps": 35,
    "eval_steps":60,
    "save_steps": 60,

    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],

    # Stage 2에서 projector도 함께 학습할지 여부
    # True: LoRA + projector 동시 학습 (기본)
    # False: LoRA만 학습, projector frozen
    "stage2_train_projector": True,

    # ── Sequence packing ──────────────────────────────────────────────────
    # packing_cutoff_len : 하나의 packed bin(=모델에 들어가는 시퀀스) 최대 토큰 수.
    #   이 길이를 초과하는 원시 시퀀스는 packer에서 버려짐(cutoff_len 이하만 패킹 대상).
    #   클수록 GPU utilization↑, 메모리↑, attention 연산량 O(T²)↑.
    #   (Flash Attention 2: 메모리 O(T), gradient checkpointing 병행 시 실측 스케일 ~1.7×/2×bin)
    #   실측: 2048→13GB/24%util, 4096→20GB/46%util, 8192→35GB/82%util (fb_dacvae, 8×A100-80GB)
    "packing_cutoff_len":  16384,
    # "packing_cutoff_len":  65536,

    # packing_bucket_size : packer(greedy knapsack)가 한 번에 받는 processed 샘플 수.
    #   greedy knapsack: bucket 내 샘플을 길이 내림차순 정렬 후 각 bin에 남은 공간에
    #   들어가는 가장 큰 샘플을 bisect로 탐색하여 채움. bucket이 클수록 탐색 풀이 넓어져
    #   bin 충전율(fill ratio)↑ → step당 유효 토큰↑ → GPU utilization 간접 향상.
    #   precomputed 모드: 오디오 디코딩 없어 CPU 부담 낮음 → 1000~2000 권장.
    #   raw audio 모드: 오디오 디코딩이 CPU 병목 → 50~200 권장.
    "packing_bucket_size": 200,

    # process_batch_size : processor_fn(토크나이징)을 한 번에 처리할 샘플 수.
    #   precomputed 모드: Arrow 피처 로드 + 토크나이징만 수행 → 128~256 권장.
    #   raw audio 모드: 오디오 디코딩 포함 → 32 권장.
    "process_batch_size":  32,
    # ──────────────────────────────────────────────────────────────────────

    "attn_implementation": "flash_attention_2",   # "eager" | "flash_attention_2" | sdpa
    "use_liger_kernel":    True,
    "use_fsdp":            False,
    "log_every":           1,

    # ── Data splits (메모리 절약) ─────────────────────────────────────────
    # num_data_splits: pre-packed 데이터를 N등분하여 epoch마다 1/N만 로드.
    #   1 = 전체 로드 (기본), 2 = 절반씩 2회, 4 = 1/4씩 4회.
    #   mls+gs 등 대용량 데이터셋에서 OOM 방지용.
    # §42+ shards-per-rank=4 로 packing 한 경우 (packed_sentence_16384) 와 매칭.
    "num_data_splits":     4,

    # ── Batch (단일 source of truth) ─────────────────────────────────────
    # stage1/stage2 모두 동일 값 사용. build_precomputed_pipeline 가 max_steps 계산 시
    # 이 값을 읽어 _table.num_rows 와 함께 사용 (§19).
    "per_device_train_batch_size": 14,
}

# ==========================================
# Encoder별 설정
#
# proj_strides: projector의 Conv1d stride 목록
#   [2, 2] → 2×stride-2, total ×4 다운샘플
#   [2]    → 1×stride-2, total ×2 다운샘플
#   각 stride마다 Conv1d(k=5) + GELU, 마지막에 Conv1d(k=1) 추가
#
# 새 encoder 추가: 이 dict에 항목 추가 후 encoders/ 에 구현체 작성
# ==========================================
ENCODER_REGISTRY = {
    "encodec": {
        # facebook/encodec-24khz — pre-RVQ acoustic latent
        # 16kHz → 24kHz, model.encoder() 직접 호출
        # out: (B, T_enc, 128) @ 75fps (hop=320 @ 24kHz)
        # projector: 75fps → 18.75fps (~188 tokens/10sec)
        "model_id":    "facebook/encodec_24khz",
        "out_dim":     128,
        "tgt_sr":      24000,
        "hop":         320,
        "proj_strides": [2, 2],
    },
    "dac": {
        # descript-audio-codec 44kHz — pre-RVQ acoustic latent
        # 16kHz → 44kHz, dac.encoder() 직접 호출
        # out: (B, T_enc, 1024) @ ~86fps (hop=512 @ 44kHz)
        # projector: ~86fps → ~21.5fps (~215 tokens/10sec)
        "model_type":  "44khz",
        "out_dim":     1024,
        "tgt_sr":      44100,
        "hop":         512,
        "proj_strides": [2, 2],
        "stage2_epochs": 8,   # 원본 q_dac_enc.py 기준
    },

    "fb_dacvae": {
        # facebookresearch/dacvae — pretrained DACVAE continuous latent
        # 16kHz → 44kHz, DACVAE.encode() 호출 (encoder + VAEBottleneck 전체 frozen)
        # out: (B, T_enc, codebook_dim) @ ~86fps (hop=512 @ 44kHz)
        # projector: ~86fps → ~21.5fps (~215 tokens/10sec)
        "model_id":    "facebook/dacvae-watermarked",
        "out_dim":     8,       # default codebook_dim (실제 로드 후 model.quantizer.codebook_dim)
        "tgt_sr":      44100,
        "hop":         512,
        "proj_strides": [2, 2],
    },
    "mimi_acoustic": {
        # kyutai/mimi — acoustic encoder만 (encoder_transformer 없음)
        # 저수준 피처. 비교 실험용.
        # out: (B, T_enc, 512) @ 25fps (hop=960 @ 24kHz)
        # projector: 25fps → 6.25fps (~63 tokens/10sec)
        "model_id":    "kyutai/mimi",
        "out_dim":     512,
        "tgt_sr":      24000,
        "hop":         960,     # 24000 / 25fps = 960
        "proj_strides": [2, 2],
    },
    "mimi_semantic": {
        # kyutai/mimi — acoustic encoder + encoder_transformer (semantic)
        # 고수준 semantic 피처. q_ming.py와 동일한 동작.
        # out: (B, T_enc, 512) @ 25fps (hop=960 @ 24kHz)
        # projector: 25fps → 12.5fps (~125 tokens/10sec) ← q_ming.py 원본과 동일
        "model_id":    "kyutai/mimi",
        "out_dim":     512,
        "tgt_sr":      24000,
        "hop":         960,     # encoder_transformer도 fps를 바꾸지 않음
        "proj_strides": [2],    # 1×stride-2 (q_ming.py 원본과 동일)
        "stage2_epochs": 8,     # 원본 q_ming.py 기준
    },
}


def get_config(encoder_name: str) -> dict:
    """TRAIN_CONFIG + encoder cfg를 합쳐서 반환. encoder별 오버라이드 적용."""
    if encoder_name not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder: '{encoder_name}'. "
            f"Available: {list(ENCODER_REGISTRY)}"
        )
    enc_cfg = ENCODER_REGISTRY[encoder_name]
    cfg = dict(TRAIN_CONFIG)

    # encoder별 stage2_epochs 오버라이드 (없으면 TRAIN_CONFIG 기본값 유지)
    if "stage2_epochs" in enc_cfg:
        cfg["stage2_epochs"] = enc_cfg["stage2_epochs"]

    # 16kHz 오디오 샘플 1개당 LLM 토큰 수 변환 계수
    # samples_per_token = hop_tgt × (16000 / tgt_sr) × prod(proj_strides)
    cfg["sample_rate"] = 16000
    samples_per_token = (
        enc_cfg["hop"]
        * (cfg["sample_rate"] / enc_cfg["tgt_sr"])
        * math.prod(enc_cfg["proj_strides"])
    )
    cfg["samples_per_token"] = samples_per_token
    max_audio_tokens = int(cfg["max_audio_len"] / samples_per_token)
    # 기본 예산: 6클립 × (최대 오디오 토큰 + 텍스트 토큰)
    # numClips4DAC = 3.7
    # numClips4DACVAE = 3.7
    # numClips = numClips4DACVAE
    # cfg["max_batch_tokens"] = int(numClips * (max_audio_tokens + cfg["max_text_len"]))
    cfg["max_batch_tokens"] = 8000

    cfg["encoder_name"] = encoder_name
    cfg["encoder"]      = enc_cfg
    # Qwen3.5 와 Qwen3 family 를 같이 지원하므로 family / size 둘 다 분기.
    # llm_tag 예: "2b" (Qwen3.5-2B), "1.7b" (Qwen3-1.7B), "4b" (Qwen3.5-4B or Qwen3-4B)
    llm_tag    = _infer_llm_tag(cfg["llm_model"])
    llm_family = _infer_llm_family(cfg["llm_model"])   # "Qwen3.5" | "Qwen3"
    cfg["project_name"] = f"{llm_family}-{llm_tag}-ASR-{encoder_name}"

    # Qwen2.5 <|image_pad|> (id=151655) — 미학습 슬롯, audio placeholder로 재사용
    # (새 special token 추가 불필요, resize_token_embeddings 불필요)
    cfg["audio_pad_token_id"] = 151655

    return cfg
