# AudioLLM Trainer
This repository contains a AudioLLM trainer for pre-training based on LLaMA-Factory.
rebased on LLaMA-Factory 0.9.3

To read original README -> [LLaMA-Factory README](https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/README.md)

## Environment setup
```bash
bash scripts/install.sh # This includes installing requirements.
# or 
# pip3 install -e .
```

## Pre-training (Speech X)

### Vocab expand (for audio tokens)

This script adds audio tokens to vocaublary, which expands the (tokenizer, embedding, lm_head) of the models. 

- **model_path** (`str`) - Huggingface model and tokenizer path.
- **num_units** (`int`) - Number of speech unit tokens to be expanded.
- **unit_format** (`str`) - Unit tokens format. (Default: `<|audio{:04d}|>`)
- **special_tokens** (`list`) - Special tokens to expand. (Default: `<|audio_correspond|> <|audio_continue|> <|audio_start|> <|audio_end|>`)
- **from_pretrained** (`str`, Optional) - (Pretrained Embedding weights) Numpy weight path to initialize new model embeddings. The size of first axis should be same with `num_units`.
- **std** (`[str, float]`, default: `inherit`)
  - "inherit" -> embeddings will be initialized from a multivariate normal distribution of the previous embeddings. 
  - "model_config" -> embeddings will be initialized from the model.config.initializer_range of `config.json`
  - "float value (e.g. 0.02)" -> custom float value
- **save_path** (`str`) - Path to save vocab-extended model and tokenizer.
- **overwrite** - Overwrite the given model_path. Ignored if `save_path` is given.

**Example**
```bash
python exapnd_vocab.py \
  --model_path={your_model_path} \
  --num_units=6561 \
  --from_pretrained=./models/kmeans_10k.npy \
  --save_path={save_model_path}
```

### Setting Run Name (for results path and logging)
config file 이 환경변수보다 우선권을 가집니다.

**config.yaml**
```yaml
run_name: my_run_name
```
or
```bash
export RUN_NAME=my_run_name
```

### Set WANDB configuration (Optional)
```bash
export WANDB_MODE=online
export WANDB_PROJECT={WANDB_PROJECT_NAME}
export WANDB_API_KEY={YOUR_WANDB_API_KEY}
```

### Train Unit Embedding Weights Only (Stage1)

확장된 audio unit embedding 및 lm_head parameter 만 update 하도록 학습합니다. (Text embedding 및 Decoder layer freeze)
```bash
export FREEZE_WEIGHT=True
```

## Configurations

### Prepare datasets
1. Preprare raw json or jsonl datasets
```json
# Example jsonl
{"messages":[{"role":"system","content":"너는 계산기야."},{"role":"user","content":"1+1="},{"role":"assistant","content":"2"}]}
```

2. (Optional) Register Template in [template.py](src/llamafactory/data/template.py).
```python
# Example
register_template(
    name="example_template",
    format_system=StringFormatter(slots=["<|im_start|>user\n{{content}}"]),
    format_user=StringFormatter(slots=["{{content}}<|im_end|>"]),
    format_assistant=StringFormatter(slots=["<|im_start|>assistant\n{{content}}<|im_end|>"]),
    stop_words=["<|endoftext|>"]
)
```
3. set dataset_info.json
```json
{    
    "file1": {
        "file_name": "{{raw_dataset_path}}",
        "formatting": "example_template",
        "columns": {
            "messages": "messages"
        },
        "tags": {
            "system_tag": "system",
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant"
        }
    },
    ...
}
```

## Train on NSML
```bash
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
FORCE_TORCHRUN=1 NNODES=$NSML_WORLD_SIZE NODE_RANK=$NSML_RANK MASTER_ADDR=$NSML_HOST_RANK0 MASTER_PORT=21267 llamafactory-cli train {{config_path}}
```
## Train on MLX
```bash
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
FORCE_TORCHRUN=1 NNODES=$PET_NNODES NPROC_PER_NODE=$PET_NPROC_PER_NODE RDZV_ID=$PET_RDZV_ID RDZV_ENDPOINT=$PET_RDZV_ENDPOINT llamafactory-cli train {{config_path}}
```

## Train on Superpod2
[spd2/README.md](spd2/README.md) 참조

## Resume Training

To resume training from the **last checkpoint** and resume logging on the existing Wandb run, follow below steps.

Enable the `resume_from_checkpoint` option in the training config file.

```yaml
resume_from_checkpoint: true
or
resume_from_checkpoint: checkpoint-12345 # or checkpoint path
```

Set the Wandb environment variables, such that the trainer points to the same run id.

```bash
# to resume a run from,
$ export WANDB_RESUME_FROM=cts8xs0t?_step=12345
# to fork a run from,
$ export WANDB_FORK_FROM=cts8xs0t?_step=12345
```
Run IDs can be found in the url. e.g., https://wandb.ai/naver-clova/Speech%20X/runs/cts8xs0t -> `cts8xs0t`. Run IDs are 
set automatically when starting a new run and can be retrieved from the url. Alternatively, you could set the Run ID 
before the initial execution of the training script. 


# Contact
- Hyperscale AI AudioLLM Mission / Foundation Research
- Main contributor: (sanghyuk.choi@navercorp.com)