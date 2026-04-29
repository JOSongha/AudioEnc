# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
"""Qwen3.5AE model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

logger = logging.get_logger(__name__)


class AudioConfig(PretrainedConfig):
    r"""
    Configuration class for the audio adapter module (WavTokenizer SEANetEncoder variant).

    Defines a causal transformer adapter based on Llama decoder layers that connects the
    WavTokenizer pre-VQ encoder output to the LLM.
    """

    model_type = "audio_adapter"
    base_config_key = "audio_config"

    def __init__(
        self,
        audio_hidden_size=512,           # WavTokenizer z_e dimension
        adapter_hidden_size=512,
        llm_embed_size=4096,
        num_adapter_layers=4,
        num_attention_heads=8,
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
        # WavTokenizer encoder hyperparameters
        wavtok_ckpt_path="/mnt/tmp/hf_cache/wavtokenizer/wavtokenizer_large_unify_600_24k.ckpt",
        wavtok_sample_rate=24000,
        wavtok_fps=40,                   # encoder output frame rate
        wavtok_hop_length=600,           # samples per output frame (24000 / 40)
        wavtok_dimension=512,            # z_e dimension
        wavtok_n_filters=32,             # SEANetEncoder n_filters
        wavtok_ratios=[6, 5, 5, 4],      # downsampling ratios
        wavtok_lstm=2,                   # LSTM layers
        **kwargs,
    ):
        self.audio_hidden_size = audio_hidden_size
        self.adapter_hidden_size = adapter_hidden_size
        self.llm_embed_size = llm_embed_size
        self.num_adapter_layers = num_adapter_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = head_dim if head_dim is not None else adapter_hidden_size // num_attention_heads
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

        self.wavtok_ckpt_path = wavtok_ckpt_path
        self.wavtok_sample_rate = wavtok_sample_rate
        self.wavtok_fps = wavtok_fps
        self.wavtok_hop_length = wavtok_hop_length
        self.wavtok_dimension = wavtok_dimension
        self.wavtok_n_filters = wavtok_n_filters
        self.wavtok_ratios = wavtok_ratios
        self.wavtok_lstm = wavtok_lstm

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(**kwargs)


class Qwen3_5AETextConfig(PretrainedConfig):
    r"""
    Text decoder configuration for Qwen3.5AE. Mirrors Qwen3_5 text config with linear attention
    interleaved layers.
    """

    model_type = "qwen3_5_ae_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_config_key = "text_config"

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
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
        vocab_size=248320,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        layer_types=None,
        full_attention_interval=4,
        partial_rotary_factor=0.25,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = rope_parameters
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.partial_rotary_factor = partial_rotary_factor

        if layer_types is None:
            layer_types = [
                "linear_attention" if bool((i + 1) % full_attention_interval) else "full_attention"
                for i in range(num_hidden_layers)
            ]
        self.layer_types = layer_types

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class Qwen3_5AEConfig(PretrainedConfig):
    r"""
    Composite configuration for Qwen3.5AE: text decoder + audio encoder adapter.
    """

    model_type = "qwen3_5_ae"
    sub_configs = {"audio_config": AudioConfig, "text_config": Qwen3_5AETextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        audio_config=None,
        audio_pad_token_id=248076,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(audio_config, dict):
            audio_config = self.sub_configs["audio_config"](**audio_config)
        elif audio_config is None:
            audio_config = self.sub_configs["audio_config"]()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            text_config = self.sub_configs["text_config"]()
        self.text_config = text_config

        self.audio_pad_token_id = audio_pad_token_id

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["Qwen3_5AEConfig", "Qwen3_5AETextConfig", "AudioConfig"]
