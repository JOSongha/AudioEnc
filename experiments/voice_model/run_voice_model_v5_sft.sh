#/bin/bash
source /home/nsml/miniforge3/etc/profile.d/conda.sh
conda activate venv

export WANDB_ENTITY="naver-clova"
export WANDB_PROJECT="AudioLLM"
export WANDB_API_KEY="0fce21600b4f5a95e2d3177a68fc2ce51a0f4614"

export HF_HOME=/mnt/fr20tb/audiollm/datasets/.cache
export HF_DATASETS_CACHE=/mnt/fr20tb/audiollm/datasets/.cache
export RUN_NAME=voice_model_v5

# When resuming training, uncomment the following WANDB-related comment and run it together with the training command.
# export WANDB_RUN_ID={RUN_ID}
# export WANDB_RESUME=must

# When performing data packing, uncomment the 'CUDA_VISIBLE_DEVICES=0' line and run the command immediately below.
# export CUDA_VISIBLE_DEVICES=0
# FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 MASTER_ADDR={RANK0_ADDR} MASTER_PORT={RANK0_PORT} llamafactory-cli train examples/train_full/speechx_full_v5_voice_model_sft.yaml

# When performing training, please execute the following command:
FORCE_TORCHRUN=1 NNODES=$NSML_WORLD_SIZE NODE_RANK=$NSML_RANK MASTER_ADDR=$NSML_HOST_RANK0 MASTER_PORT=23457 llamafactory-cli train examples/train_full/speechx_full_v5_voice_model_sft.yaml