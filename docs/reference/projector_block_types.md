# Projector decoder block: `llama` vs `qwen3`

`AudioProjector` (in each model dir's `audio_encoder.py`) supports two interchangeable
decoder-block recipes via `AudioConfig.decoder_block_type ∈ {"llama", "qwen3"}`. Both
use the LLaMA-family recipe (pre-norm RMSNorm, RoPE, SwiGLU, bias-free linears).
The only structural difference is that Qwen3 inserts QK-norm inside attention.

## Architectural diff

Per attention layer:

| Component                   | `llama`                           | `qwen3`                                    |
|-----------------------------|-----------------------------------|--------------------------------------------|
| pre-attention norm          | `RMSNorm(hidden_size)`            | `RMSNorm(hidden_size)`                     |
| q/k/v/o projections         | bias-free Linear                  | bias-free Linear                           |
| **QK-norm**                 | —                                 | `RMSNorm(head_dim)` on q and k before RoPE |
| RoPE                        | applied to q,k                    | applied to q,k (after QK-norm)             |
| FFN                         | SwiGLU (gate/up/down, bias-free)  | SwiGLU (gate/up/down, bias-free)           |
| post-FFN norm               | `RMSNorm(hidden_size)`            | `RMSNorm(hidden_size)`                     |

Everything else (input_proj 128→512, 4 stacked decoder layers, final RMSNorm,
output_proj 512→2560) is identical.

## Parameter count

With our adapter dims (`hidden=512, intermediate=2048, heads=8, head_dim=64,
num_kv_heads=8, num_layers=4`) and `audio_hidden_size=128, llm_embed_size=2560`:

| block   | total params | per-layer attn | per-layer mlp | per-layer norms |
|---------|-------------:|---------------:|--------------:|----------------:|
| `llama` |   18,158,080 |      1,048,576 |     3,145,728 |             1,024 |
| `qwen3` |   18,158,592 |      1,048,576 |     3,145,728 |             1,536 |

The +512 delta comes from 4 layers × (q_norm 64 + k_norm 64) = 512 extra QK-norm
params. Both fit comfortably within the paper's ±2% / 18.2M projector budget.

Smoke test (auto-verified):

```
llama : block_type=llama, layers=4, params=18,158,080, forward OK
qwen3 : block_type=qwen3, layers=4, params=18,158,592, forward OK
```

## Forward signature

`LlamaDecoderLayer` and `Qwen3DecoderLayer` accept the same kwargs in modern
transformers (≥4.51): `hidden_states, attention_mask, position_ids,
position_embeddings, past_key_values, use_cache`. The projector's forward pass is
identical for both — only the layer class swap differs.

`LlamaRMSNorm/Qwen3RMSNorm` and `LlamaRotaryEmbedding/Qwen3RotaryEmbedding` share
the same constructor signature, so the swap is a drop-in via the registry.

## When to pick which

- **`llama` (default)** — backward compatible with all existing Stage-1 / Stage-2
  checkpoints. State dict keys: `layers.{i}.self_attn.{q,k,v,o}_proj.weight`,
  `layers.{i}.{input,post_attention}_layernorm.weight`,
  `layers.{i}.mlp.{gate,up,down}_proj.weight`.
- **`qwen3`** — adds QK-norm. This can stabilize attention logits at fp16/bf16 and
  may help when training the projector on long audio sequences. Adds two extra
  state-dict keys per layer: `layers.{i}.self_attn.q_norm.weight`,
  `layers.{i}.self_attn.k_norm.weight`. Not loadable from a `llama`-trained
  checkpoint without a partial-load (the q_norm/k_norm weights initialize fresh).

For the paper experiments in [03_method.tex](../../AudioEnc/log/tmp/latex_work/sec/03_method.tex)
the projector recipe is described as "pre-norm Transformer decoder following the
recipe shared by LLaMA and Qwen3" — both options realize that description.

## Switching

Add or update the field in the model dir's `config.json`:

```json
{
  "audio_config": {
    "decoder_block_type": "qwen3",
    "...": "..."
  }
}
```

Or pass it programmatically:

```python
from configuration_qwen3_5AE import AudioConfig
cfg = AudioConfig(decoder_block_type="qwen3", ...)
projector = AudioProjector(cfg)
```

Default (no field set / older configs) resolves to `"llama"`. The `AudioProjector`
falls back to `getattr(config, "decoder_block_type", "llama")`, so legacy configs
that predate this field still load.

## Affected files

The branch is mirrored across all model dirs that ship a projector:

- `audiollm-trainer/external/models/Qwen3.5AE-4B/{audio_encoder,configuration_qwen3_5AE}.py` (DAC/EnCodec/WavTokenizer base)
- `audiollm-trainer/external/models/Qwen3.5AE-4B-whisper-tiny/{audio_encoder,configuration_qwen3_5AE}.py`
- `audiollm-trainer/external/models/Qwen3.5AE-4B-whisper-small/{audio_encoder,configuration_qwen3_5AE}.py`
- `models/Qwen3.5AE-4B-s2/{audio_encoder,configuration_qwen3_5AE}.py` (Stage-2 active ckpt dir)

The legacy `Qwen3AE-4B/` (DAC-VAE v1, deprecated) is intentionally not patched.
