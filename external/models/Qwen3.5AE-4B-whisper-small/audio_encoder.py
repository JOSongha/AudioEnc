"""Whisper-small.en + 4-layer Llama causal projector audio encoder.

Stage1: Whisper encoder is frozen, projector is trained.
Input contract: `audio_features` is precomputed log-mel [N, 80, 3000] (fp32 from
WhisperFeatureExtractor; cast to bf16 inside forward).
"""

import torch
import torch.nn as nn
from transformers import WhisperModel
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


class AudioProjector(nn.Module):
    """4-layer causal Llama-style adapter projecting Whisper hidden -> LLM embed dim."""

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
        # `getattr(..., "eager")` is not enough — PretrainedConfig sets `_attn_implementation = None`
        # by default, so the missing-default arm never triggers. Explicitly fall back when None.
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
    """Whisper-small.en encoder (frozen) + AudioProjector (trainable in Stage1)."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Whisper encoder loaded from HF (pretrained weights baked in via convert script).
        whisper = WhisperModel.from_pretrained(
            config.whisper_model_id,
            dtype=torch.bfloat16,
        )
        self.encoder = whisper.encoder
        del whisper  # decoder unused

        # Stage1: encoder frozen
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.audio_dim = config.audio_hidden_size  # 768
        self.projector = AudioProjector(config)

    def forward(self, audio_features, use_cache=False):
        """
        Args:
            audio_features: [N, 80, 3000] log-mel spectrogram (fp32 OK, will be cast to bf16).
            use_cache: KV cache for autoregressive decoding (rarely used in training).

        Returns:
            audio_embeds: [N, 1500, llm_embed_size]
        """
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                hidden = self.encoder(audio_features.to(torch.bfloat16)).last_hidden_state
        # hidden: [N, 1500, 768] in bf16

        audio_embeds, _ = self.projector(hidden, use_cache=use_cache)
        return audio_embeds
