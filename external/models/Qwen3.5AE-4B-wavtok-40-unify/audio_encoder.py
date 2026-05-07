"""WavTokenizer SEANetEncoder (pre-VQ tap, frozen) + 4-layer Llama causal projector.

Stage1: SEANetEncoder is frozen, projector is trained.
Input contract: `audio_features` is raw mono waveform [B, 1, S] or [B, S] @ 24kHz
(will be cast to bf16 inside forward).

SEANetEncoder is inlined here (from WavTokenizer/encoder/modules) so that HF
trust_remote_code works without subdirectory imports.
Source: https://github.com/jishengpeng/WavTokenizer/tree/main/encoder/modules
"""

import math
import typing as tp
import warnings

import einops
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


# ---------------------------------------------------------------------------
# SEANetEncoder (inlined from WavTokenizer encoder/modules)
# ---------------------------------------------------------------------------

class ConvLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, 'b ... t -> b t ...')
        x = super().forward(x)
        x = einops.rearrange(x, 'b t ... -> b ... t')
        return x


class SLSTM(nn.Module):
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, x):
        x1 = x.permute(2, 0, 1)
        y, _ = self.lstm(x1)
        y = y.permute(1, 2, 0)
        if self.skip:
            y = y + x
        return y


_CONV_NORMS = frozenset(['none', 'weight_norm', 'spectral_norm',
                         'time_layer_norm', 'layer_norm', 'time_group_norm'])


def _apply_norm(module, norm='none'):
    if norm == 'weight_norm':
        return weight_norm(module)
    elif norm == 'spectral_norm':
        return spectral_norm(module)
    return module


def _get_norm_module(module, causal=False, norm='none', **kw):
    if norm == 'layer_norm':
        return ConvLayerNorm(module.out_channels, **kw)
    elif norm == 'time_group_norm':
        return nn.GroupNorm(1, module.out_channels, **kw)
    return nn.Identity()


def _extra_pad(x, kernel_size, stride, padding_total=0):
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal - length


def _pad1d(x, paddings, mode='zero', value=0.):
    length = x.shape[-1]
    pl, pr = paddings
    if mode == 'reflect':
        max_pad = max(pl, pr)
        extra = max(0, max_pad - length + 1)
        if extra:
            x = F.pad(x, (0, extra))
        padded = F.pad(x, paddings, mode, value)
        return padded[..., :padded.shape[-1] - extra]
    return F.pad(x, paddings, mode, value)


def _unpad1d(x, paddings):
    pl, pr = paddings
    return x[..., pl: x.shape[-1] - pr]


class NormConv1d(nn.Module):
    def __init__(self, *args, causal=False, norm='none', norm_kwargs={}, **kwargs):
        super().__init__()
        self.conv = _apply_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = _get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        return self.norm(self.conv(x))


class NormConvTranspose1d(nn.Module):
    def __init__(self, *args, causal=False, norm='none', norm_kwargs={}, **kwargs):
        super().__init__()
        self.convtr = _apply_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm = _get_norm_module(self.convtr, causal, norm, **norm_kwargs)

    def forward(self, x):
        return self.norm(self.convtr(x))


class SConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1,
                 groups=1, bias=True, causal=False, norm='none', norm_kwargs={},
                 pad_mode='reflect'):
        super().__init__()
        if stride > 1 and dilation > 1:
            warnings.warn(f'SConv1d stride={stride} dilation={dilation}')
        self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride,
                               dilation=dilation, groups=groups, bias=bias,
                               causal=causal, norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x):
        B, C, T = x.shape
        ks = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dil = self.conv.conv.dilation[0]
        ks = (ks - 1) * dil + 1
        padding_total = ks - stride
        extra = _extra_pad(x, ks, stride, padding_total)
        if self.causal:
            x = _pad1d(x, (padding_total, extra), mode=self.pad_mode)
        else:
            pr = padding_total // 2
            pl = padding_total - pr
            x = _pad1d(x, (pl, pr + extra), mode=self.pad_mode)
        return self.conv(x)


class SConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, causal=False,
                 norm='none', trim_right_ratio=1., norm_kwargs={}):
        super().__init__()
        self.convtr = NormConvTranspose1d(in_channels, out_channels, kernel_size, stride,
                                          causal=causal, norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio

    def forward(self, x):
        ks = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = ks - stride
        y = self.convtr(x)
        if self.causal:
            pr = math.ceil(padding_total * self.trim_right_ratio)
            pl = padding_total - pr
        else:
            pr = padding_total // 2
            pl = padding_total - pr
        return _unpad1d(y, (pl, pr))


class SEANetResnetBlock(nn.Module):
    def __init__(self, dim, kernel_sizes=[3, 1], dilations=[1, 1],
                 activation='ELU', activation_params={'alpha': 1.0},
                 norm='weight_norm', norm_params={}, causal=False, pad_mode='reflect',
                 compress=2, true_skip=True):
        super().__init__()
        act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (ks, dil) in enumerate(zip(kernel_sizes, dilations)):
            ic = dim if i == 0 else hidden
            oc = dim if i == len(kernel_sizes) - 1 else hidden
            block += [act(**activation_params),
                      SConv1d(ic, oc, kernel_size=ks, dilation=dil, norm=norm,
                              norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module = (nn.Identity() if true_skip else
                                    SConv1d(dim, dim, kernel_size=1, norm=norm,
                                            norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode))

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class SEANetEncoder(nn.Module):
    """SEANet encoder (WavTokenizer variant, pre-VQ z_e output)."""
    def __init__(self, channels=1, dimension=128, n_filters=32, n_residual_layers=1,
                 ratios=[8, 5, 4, 2], activation='ELU', activation_params={'alpha': 1.0},
                 norm='weight_norm', norm_params={}, kernel_size=7, last_kernel_size=7,
                 residual_kernel_size=3, dilation_base=2, causal=False, pad_mode='reflect',
                 true_skip=False, compress=2, lstm=2):
        super().__init__()
        self.ratios = list(reversed(ratios))
        self.hop_length = np.prod(self.ratios)
        act = getattr(nn, activation)
        mult = 1
        model: tp.List[nn.Module] = [
            SConv1d(channels, mult * n_filters, kernel_size, norm=norm,
                    norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]
        for ratio in self.ratios:
            for j in range(n_residual_layers):
                model += [SEANetResnetBlock(
                    mult * n_filters, kernel_sizes=[residual_kernel_size, 1],
                    dilations=[dilation_base ** j, 1], norm=norm, norm_params=norm_params,
                    activation=activation, activation_params=activation_params,
                    causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip)]
            model += [act(**activation_params),
                      SConv1d(mult * n_filters, mult * n_filters * 2,
                              kernel_size=ratio * 2, stride=ratio,
                              norm=norm, norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)]
            mult *= 2
        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm)]
        model += [act(**activation_params),
                  SConv1d(mult * n_filters, dimension, last_kernel_size, norm=norm,
                          norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# AudioProjector + AudioEncoder
# ---------------------------------------------------------------------------

class AudioProjector(nn.Module):
    """4-layer causal Llama-style adapter projecting SEANetEncoder z_e -> LLM embed dim."""

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
    """WavTokenizer SEANetEncoder (frozen pre-VQ tap) + AudioProjector (trainable in Stage1)."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = SEANetEncoder(
            channels=1,
            dimension=config.wavtok_dimension,
            n_filters=config.wavtok_n_filters,
            ratios=list(config.wavtok_ratios),
            lstm=config.wavtok_lstm,
            norm="weight_norm",
        )

        # Strip weight_norm: state_dict cleanliness + accelerate meta-tensor compat
        for module in self.encoder.modules():
            try:
                nn.utils.remove_weight_norm(module)
            except ValueError:
                pass

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.audio_dim = config.audio_hidden_size
        self.projector = AudioProjector(config)

    def forward(self, audio_features, use_cache=False):
        """
        Args:
            audio_features: [B, 1, S] or [B, S] raw waveform @ 24kHz.
        Returns:
            audio_embeds: [B, T, llm_embed_size] where T = S // 600
        """
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(1)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                z_e = self.encoder(audio_features.to(torch.bfloat16))  # [B, 512, T]

        audio_latents = z_e.transpose(1, 2).to(self.projector.input_proj.weight.dtype)
        audio_embeds, _ = self.projector(audio_latents, use_cache=use_cache)
        return audio_embeds
