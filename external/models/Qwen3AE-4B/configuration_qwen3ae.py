# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3 model configuration"""

from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

logger = logging.get_logger(__name__)


class AudioConfig(PretrainedConfig):
    r"""
    Configuration class for the audio adapter module.

    This config defines a 4-layer causal transformer adapter based on Llama decoder layers
    that connects the DAC-VAE audio encoder to the LLM. The adapter processes audio features
    through causal self-attention layers to align with the LLM's hidden space.

    Args:
        audio_hidden_size (`int`, *optional*, defaults to 1024):
            Hidden size of the DAC-VAE audio encoder output (input dimension to adapter).
        adapter_hidden_size (`int`, *optional*, defaults to 4096):
            Hidden size of the adapter transformer layers.
        llm_embed_size (`int`, *optional*, defaults to 4096):
            Hidden size of the LLM (output dimension of adapter).
        num_adapter_layers (`int`, *optional*, defaults to 4):
            Number of causal transformer layers in the adapter (based on Llama decoder).
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads in each adapter layer.
        num_key_value_heads (`int`, *optional*):
            Number of key-value heads for grouped query attention. If not specified,
            defaults to `num_attention_heads` (MHA).
        head_dim (`int`, *optional*):
            Dimension of each attention head. If not specified,
            defaults to `adapter_hidden_size // num_attention_heads`.
        intermediate_size (`int`, *optional*):
            Dimension of the FFN intermediate layer. If not specified,
            defaults to `4 * adapter_hidden_size`.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation function for the FFN layers.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            Maximum sequence length for positional embeddings in the adapter.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for RMSNorm layers.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention projection layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use KV cache for faster inference.
        dac_encoder_dim (`int`, *optional*, defaults to 64):
            DAC-VAE encoder base dimension.
        dac_encoder_rates (`List[int]`, *optional*, defaults to [2, 8, 10, 12]):
            Downsampling rates for DAC-VAE encoder stages.
        dac_latent_dim (`int`, *optional*, defaults to 1024):
            DAC-VAE latent dimension.
        dac_decoder_dim (`int`, *optional*, defaults to 1536):
            DAC-VAE decoder base dimension.
        dac_decoder_rates (`List[int]`, *optional*, defaults to [12, 10, 8, 2]):
            Upsampling rates for DAC-VAE decoder stages.
        dac_n_codebooks (`int`, *optional*, defaults to 16):
            Number of residual vector quantization codebooks.
        dac_codebook_size (`int`, *optional*, defaults to 1024):
            Size of each codebook.
        dac_codebook_dim (`int`, *optional*, defaults to 128):
            Dimension of each codebook entry.
        dac_sample_rate (`int`, *optional*, defaults to 48000):
            Sample rate the DAC-VAE was trained on.
    """

    model_type = "audio_adapter"

    def __init__(
        self,
        audio_hidden_size=128,
        adapter_hidden_size=512,
        llm_embed_size=1024,
        num_adapter_layers=4,
        num_attention_heads=32,
        num_key_value_heads=None,
        head_dim=None,
        intermediate_size=None,
        hidden_act="silu",
        max_position_embeddings=8192,
        rope_theta=10000.0,
        rope_scaling=None,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        initializer_range=0.02,
        use_cache=True,
        # DAC-VAE hyperparameters
        dac_encoder_dim=64,
        dac_encoder_rates=[2, 8, 10, 12],
        dac_latent_dim=1024,
        dac_decoder_dim=1536,
        dac_decoder_rates=[12, 10, 8, 2],
        dac_n_codebooks=16,
        dac_codebook_size=1024,
        dac_codebook_dim=128,
        dac_sample_rate=48000,
        **kwargs,
    ):
        # DAC-VAE hyperparameters
        self.dac_encoder_dim = dac_encoder_dim
        self.dac_encoder_rates = dac_encoder_rates
        self.dac_latent_dim = dac_latent_dim
        self.dac_decoder_dim = dac_decoder_dim
        self.dac_decoder_rates = dac_decoder_rates
        self.dac_n_codebooks = dac_n_codebooks
        self.dac_codebook_size = dac_codebook_size
        self.dac_codebook_dim = dac_codebook_dim
        self.dac_sample_rate = dac_sample_rate
        self.audio_hidden_size = audio_hidden_size
        self.adapter_hidden_size = adapter_hidden_size
        self.llm_embed_size = llm_embed_size
        self.num_adapter_layers = num_adapter_layers
        self.num_attention_heads = num_attention_heads

        # Default to MHA if not specified
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads

        # Default head_dim based on adapter_hidden_size
        self.head_dim = head_dim if head_dim is not None else adapter_hidden_size // num_attention_heads

        # Default intermediate size to 4x hidden size
        self.intermediate_size = intermediate_size if intermediate_size is not None else 4 * adapter_hidden_size

        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        # Validate RoPE config
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(**kwargs)


class Qwen3AEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3Model`]. It is used to instantiate a
    Qwen3 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen3-8B [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen3 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen3Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers using full attention. The first `max_window_layers` layers will use full attention, while any
            additional layer afterwards will use SWA (Sliding Window Attention).
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Qwen3Model, Qwen3Config

    >>> # Initializing a Qwen3 style configuration
    >>> configuration = Qwen3Config()

    >>> # Initializing a model from the Qwen3-8B style configuration
    >>> model = Qwen3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_ae"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen3`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0.0,
        audio_pad_token_id=151671,
        audio_config=None,
        **kwargs,
    ):
        self.audio_pad_token_id = audio_pad_token_id
        if isinstance(audio_config, dict):
            audio_config = AudioConfig(**audio_config)
        self.audio_config = audio_config
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["Qwen3AEConfig", "AudioConfig"]
