"""EnCodec-24k encoder (SEANet + RVQ pretraining) + 4-layer Llama causal projector.

Stage1: EnCodec encoder is frozen, projector is trained.
Input contract: `audio_features` is raw mono waveform [B, 1, S] or [B, S] @ 24kHz
(will be cast to bf16 inside forward).

EnCodec is loaded from facebook/encodec_24khz via transformers (standard HF dependency).
No inlining required — encoder hyperparameters are in config (encodec_* fields).
"""

import math
import typing as tp

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations
from transformers import EncodecConfig, EncodecModel
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


class AudioProjector(nn.Module):
    """4-layer causal Llama-style adapter projecting EnCodec z_e -> LLM embed dim."""

    def __init__(self, config):
        super().__init__()
        self.config = config

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
        self.llama_config._attn_implementation = getattr(config, "_attn_implementation", None) or "eager"
        self.llama_config.rope_theta = config.rope_theta
        if config.rope_scaling is not None:
            self.llama_config.rope_scaling = config.rope_scaling

        self.input_proj = nn.Linear(config.audio_hidden_size, config.adapter_hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(self.llama_config, layer_idx=i) for i in range(config.num_adapter_layers)]
        )
        self.final_norm = LlamaRMSNorm(config.adapter_hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.llama_config)

        if config.adapter_hidden_size != config.llm_embed_size:
            self.output_proj = nn.Linear(config.adapter_hidden_size, config.llm_embed_size, bias=False)
        else:
            self.output_proj = None

    def _prepare_causal_mask(self, seq_len, device):
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask[None, None, :, :]

    def forward(self, x, use_cache=False, past_key_values=None):
        hidden_states = self.input_proj(x)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(
            past_seen_tokens, past_seen_tokens + seq_len, device=x.device, dtype=torch.long
        ).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if self.llama_config._attn_implementation in ("flash_attention_2", "sdpa"):
            causal_mask = None
        else:
            total_len = past_seen_tokens + seq_len
            causal_mask = self._prepare_causal_mask(total_len, x.device)
            causal_mask = causal_mask[:, :, past_seen_tokens:, :]

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        hidden_states = self.final_norm(hidden_states)
        if self.output_proj is not None:
            hidden_states = self.output_proj(hidden_states)
        return hidden_states, past_key_values


class AudioEncoder(nn.Module):
    """EnCodec encoder (frozen pre-RVQ tap) + AudioProjector (trainable in Stage1)."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        encodec_cfg = EncodecConfig(
            target_bandwidths=config.encodec_target_bandwidths,
            sampling_rate=config.encodec_sampling_rate,
            audio_channels=config.encodec_audio_channels,
            normalize=config.encodec_normalize,
            chunk_length_s=config.encodec_chunk_length_s,
            overlap=config.encodec_overlap,
            hidden_size=config.encodec_hidden_size,
            num_filters=config.encodec_num_filters,
            num_residual_layers=config.encodec_num_residual_layers,
            upsampling_ratios=list(config.encodec_upsampling_ratios),
            kernel_size=config.encodec_kernel_size,
            last_kernel_size=config.encodec_last_kernel_size,
            residual_kernel_size=config.encodec_residual_kernel_size,
            dilation_growth_rate=config.encodec_dilation_growth_rate,
            use_causal_conv=config.encodec_use_causal_conv,
            norm_type=config.encodec_norm_type,
            pad_mode=config.encodec_pad_mode,
            compress=config.encodec_compress,
            num_lstm_layers=config.encodec_num_lstm_layers,
            trim_right_ratio=config.encodec_trim_right_ratio,
            codebook_size=config.encodec_codebook_size,
            codebook_dim=config.encodec_codebook_dim,
            use_conv_shortcut=config.encodec_use_conv_shortcut,
        )

        full = EncodecModel(encodec_cfg)
        self.encoder = full.encoder
        del full

        for module in self.encoder.modules():
            if is_parametrized(module, "weight"):
                remove_parametrizations(module, "weight", leave_parametrized=True)

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.audio_dim = config.audio_hidden_size
        self.projector = AudioProjector(config)

    def forward(self, audio_features, use_cache=False):
        """
        Args:
            audio_features: [B, 1, S] or [B, S] raw waveform @ 24kHz.
        Returns:
            audio_embeds: [B, T, llm_embed_size] where T = S // 320 (75 fps)
        """
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(1)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                z_e = self.encoder(audio_features.to(torch.bfloat16))  # [B, 128, T]

        audio_latents = z_e.transpose(1, 2).to(self.projector.input_proj.weight.dtype)
        audio_embeds, _ = self.projector(audio_latents, use_cache=use_cache)
        return audio_embeds
