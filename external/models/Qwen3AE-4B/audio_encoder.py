import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from dacvae import DACVAE


class AudioProjector(nn.Module):
    def __init__(self, config):
        """
        AudioConfig를 사용하는 4-layer causal transformer adapter

        Args:
            config: AudioConfig 객체
                - audio_hidden_size: DAC-VAE 출력 차원
                - adapter_hidden_size: adapter transformer 차원
                - llm_embed_size: LLM hidden 차원
                - num_adapter_layers: adapter 레이어 수
                - num_attention_heads: attention head 수
                - 기타 Llama decoder 관련 설정들
        """
        super().__init__()
        self.config = config

        # 1. Llama 스타일 설정을 위한 Config 정의 (AudioConfig 파라미터 사용)
        self.llama_config = LlamaConfig(
            hidden_size=config.adapter_hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            num_hidden_layers=config.num_adapter_layers,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            hidden_act=config.hidden_act,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
        )
        self.llama_config._attn_implementation = getattr(config, "_attn_implementation", "eager")

        # RoPE 파라미터 설정
        self.llama_config.rope_theta = config.rope_theta
        if config.rope_scaling is not None:
            self.llama_config.rope_scaling = config.rope_scaling

        # 2. DAC 출력 차원을 adapter 차원으로 맞추는 초기 Linear
        self.input_proj = nn.Linear(config.audio_hidden_size, config.adapter_hidden_size, bias=False)

        # 3. N개의 Llama Decoder Layer (RoPE, RMSNorm, SwiGLU 내장)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(self.llama_config, layer_idx=i) for i in range(config.num_adapter_layers)]
        )

        # 4. 최종 출력을 위한 RMSNorm
        self.final_norm = LlamaRMSNorm(config.adapter_hidden_size, eps=config.rms_norm_eps)

        # 5. RoPE (Rotary Position Embedding) 초기화
        self.rotary_emb = LlamaRotaryEmbedding(config=self.llama_config)

        # 6. adapter_hidden_size와 llm_embed_size가 다르면 output projection 추가
        if config.adapter_hidden_size != config.llm_embed_size:
            self.output_proj = nn.Linear(config.adapter_hidden_size, config.llm_embed_size, bias=False)
        else:
            self.output_proj = None

    def _prepare_causal_mask(self, seq_len, device):
        # 미래 토큰을 가리는 인과적 마스크 생성
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        # LlamaDecoderLayer는 [batch, 1, q_len, k_len] 형태의 4D mask를 기대함
        return mask[None, None, :, :]

    def forward(self, x, use_cache=False, past_key_values=None):
        """
        Args:
            x: [Batch, Time_Steps, Audio_Dim] (transpose된 DAC 특징)
            use_cache: KV cache 사용 여부 (inference 최적화용)
            past_key_values: KV cache (DynamicCache) for autoregressive generation

        Returns:
            hidden_states: [Batch, Time_Steps, LLM_Hidden_Size] 형태의 오디오 임베딩
            past_key_values: updated KV cache (if use_cache=True, else None)
        """
        # A. 초기 차원 투사: audio_hidden_size -> adapter_hidden_size
        hidden_states = self.input_proj(x)

        # B. KV cache 초기화 (필요 시)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        # C. Position IDs 계산 (KV cache 고려)
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(
            past_seen_tokens, past_seen_tokens + seq_len, device=x.device, dtype=torch.long
        ).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # D. Causal Mask 생성
        if self.llama_config._attn_implementation in ("flash_attention_2", "sdpa"):
            causal_mask = None
        else:
            total_len = past_seen_tokens + seq_len
            causal_mask = self._prepare_causal_mask(total_len, x.device)
            causal_mask = causal_mask[:, :, past_seen_tokens:, :]

        # E. Llama Layers 통과
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        # F. 최종 정규화
        hidden_states = self.final_norm(hidden_states)

        # G. adapter_hidden_size -> llm_embed_size (필요시)
        if self.output_proj is not None:
            hidden_states = self.output_proj(hidden_states)

        return hidden_states, past_key_values


class AudioEncoder(nn.Module):
    def __init__(self, config):
        """
        Args:
            config: AudioConfig 객체
                - dac_* : DAC-VAE hyperparameters
                - audio_hidden_size: DAC-VAE 출력 차원
        """
        super().__init__()
        # 1. DAC-VAE (weights are loaded via from_pretrained from combined checkpoint)
        self.encoder = DACVAE(
            encoder_dim=config.dac_encoder_dim,
            encoder_rates=config.dac_encoder_rates,
            latent_dim=config.dac_latent_dim,
            decoder_dim=config.dac_decoder_dim,
            decoder_rates=config.dac_decoder_rates,
            n_codebooks=config.dac_n_codebooks,
            codebook_size=config.dac_codebook_size,
            codebook_dim=config.dac_codebook_dim,
            sample_rate=config.dac_sample_rate,
        )
        # Remove weight_norm so state_dict uses plain "weight" keys
        # (legacy weight_norm's weight_g/weight_v breaks accelerate meta tensor loading)
        for module in self.encoder.modules():
            try:
                nn.utils.remove_weight_norm(module)
            except ValueError:
                pass

        # DAC-VAE의 latent 차원 (AudioConfig에서 가져옴)
        self.audio_dim = config.audio_hidden_size

        # 2. Adapter (Projector)
        # DAC의 출력을 LLM의 Hidden Size로 변환
        self.projector = AudioProjector(config)

    def forward(self, audio_values, use_cache=False):
        """
        Args:
            audio_values: [Batch, 1, Samples] (Raw Audio)
            use_cache: KV cache 사용 여부 (inference 최적화용)

        Returns:
            audio_embeds: [Batch, Time_Steps, LLM_Hidden_Size] 형태의 오디오 임베딩
            audio_latents: [Batch, Time_Steps, Audio_Dim] DAC-VAE latent (consistency target)
        """
        # A. DAC Encoding
        # encoded의 예상 shape: [Batch, Audio_Dim, Time_Steps]
        with torch.no_grad():  # Encoder Freeze
            encoded = self.encoder.encode(audio_values)

        # B. Dimension Shuffling
        # LLM은 [Batch, Seq, Hidden]을 기대하므로
        # [B, C, T] -> [B, T, C]로 차원을 맞춥니다.
        audio_latents = encoded.transpose(1, 2)

        # C. Noise augmentation (training only)
        # x̃ = √k · ε + √(1-k) · x
        # k ~ U(0, 0.1) sampled per-utterance; ε sampled per-frame.
        # Per-utterance k keeps the difficulty level consistent within a clip
        # (stable curriculum / diffusion-style timestep sampling), while per-frame ε
        # preserves representational diversity so the projector sees many
        # realizations of the same latent under a fixed perturbation strength.
        projector_input = audio_latents
        if self.training and getattr(self, "noise_enabled", False):
            batch_size = audio_latents.shape[0]
            k = torch.rand(batch_size, 1, 1, device=audio_latents.device, dtype=audio_latents.dtype) * 0.1
            epsilon = torch.randn_like(audio_latents)
            projector_input = k.sqrt() * epsilon + (1 - k).sqrt() * audio_latents

        # D. Projection
        # [Batch, Time_Steps, LLM_Hidden_Size]
        audio_embeds, _ = self.projector(projector_input, use_cache=use_cache)

        return audio_embeds
