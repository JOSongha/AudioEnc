#/bin/bash
source /home/nsml/miniforge3/etc/profile.d/conda.sh
conda activate venv

export WANDB_ENTITY="naver-clova"
export WANDB_PROJECT="AudioLLM"
export WANDB_API_KEY="4265de1bc549df326197a21ef197c755f57310ea"

export HF_HOME=/mnt/fr20tb/audiollm/datasets/.cache
export HF_DATASETS_CACHE=/mnt/fr20tb/audiollm/datasets/.cache
# export RUN_NAME=voice_engine_v6
# export RUN_NAME=voice_engine_v6_abl1
export RUN_NAME=voice_engine_v6_abl2

# When resuming training, uncomment the following WANDB-related comment and run it together with the training command.
# export WANDB_RUN_ID={RUN_ID}
# export WANDB_RESUME=must

# When performing data packing, uncomment the 'CUDA_VISIBLE_DEVICES=0' line and run the command immediately below.
# export CUDA_VISIBLE_DEVICES=0
# FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 MASTER_ADDR=10.169.39.24 MASTER_PORT=10881 llamafactory-cli train examples/train_full/speechx_full_voice_engine_v6_sft.yaml
# FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 MASTER_ADDR=10.169.39.24 MASTER_PORT=10881 llamafactory-cli train examples/train_full/speechx_full_voice_engine_v6_abl1_sft.yaml

# When performing training, please execute the following command:
# FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 MASTER_ADDR=10.169.39.24 MASTER_PORT=10881 llamafactory-cli train examples/train_full/speechx_full_voice_engine_v6_sft.yaml
# FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 MASTER_ADDR=10.169.39.24 MASTER_PORT=10881 llamafactory-cli train examples/train_full/speechx_full_voice_engine_v6_abl1_sft.yaml
FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 MASTER_ADDR=10.169.39.24 MASTER_PORT=10881 llamafactory-cli train examples/train_full/speechx_full_voice_engine_v6_abl2_sft.yaml