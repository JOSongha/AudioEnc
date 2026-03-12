import os

# ==========================================
# 학습 하이퍼파라미터 (encoder 무관)
# ==========================================
TRAIN_CONFIG = {
    # llm_type: "instruct" → ChatML 프롬프트, "base" → 단순 prefix
    "llm_type":  "base",
    "llm_model": "Qwen/Qwen3.5-4B",

    "batch_size": 6,
    "stage2_batch_size": 4,
    "gradient_accumulation_steps": 4,

    "stage1_lr": 5e-5,
    "stage1_epochs": 3,

    "stage2_lr": 2e-5,
    "stage2_epochs": 16,

    "stage2_resume_lr": 1e-5,
    "stage2_resume_epochs": 30,

    "max_grad_norm": 1.0,
    "warmup_ratio": 0.1,

    "max_audio_len": 16000 * 20,
    "max_text_len": 256,

    "data_path": "/mnt/tmp/cache",
    "mls_data_path": "/mnt/tmp/cache",
    "mls_num_samples": 4_050_000,
    "model_cache_dir": "/mnt/tmp/cache/hf",
    "wandb_mode": "online",

    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
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
    "dac_vae": {
        # descript-audio-codec 44kHz + trainable VAE bottleneck
        # DAC encoder (frozen, 1024-dim) → Linear → (mu, logvar) → z (256-dim)
        # 학습 중 reparameterization, 평가 중 mu 사용
        # out: (B, T_enc, 256) @ ~86fps (hop=512 @ 44kHz)
        # projector: ~86fps → ~21.5fps (~215 tokens/10sec)
        "model_type":  "44khz",
        "out_dim":     256,      # latent_dim (after VAE bottleneck)
        "latent_dim":  256,
        "tgt_sr":      44100,
        "hop":         512,
        "proj_strides": [2, 2],
        "stage2_epochs": 8,
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

    cfg["encoder_name"] = encoder_name
    cfg["encoder"]      = enc_cfg
    cfg["project_name"] = f"Qwen2.5-ASR-{encoder_name}"
    return cfg
