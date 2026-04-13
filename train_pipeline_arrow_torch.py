"""
Audio-LLM Training Script (FSDP + Sharded Checkpoint)

[실행 커맨드 예시]
LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/glibc_compat.so" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
torchrun --nproc_per_node=8 train_pipeline_arrow_torch.py \
    --data_dir "/mnt/tmp/cache/precomputed/fb_dacvae" \
    --output_dir "./outputs/fb_dacvae" \
    --max_steps_s1 15000 \
    --max_steps_s2 75000 \
    --warmup_steps_s2 1000 \
    --batch_size 4 \
    --grad_accum 4 \
    --log_steps 100 \
    --save_steps 100 \
    --step_offset_s1 0 \
    --step_offset_s2 0

[인자 설명]
--data_dir: .arrow 데이터셋 경로
--output_dir: 결과물 저장 경로
--max_steps_s1/s2: 각 스테이지별 총 학습 스텝
--warmup_steps_s2: Stage 2 코사인 스케줄러 웜업 단계
--batch_size: GPU당 물리 배치 사이즈
--grad_accum: 그래디언트 누적 스텝 (Global Batch = GPU수 * batch_size * grad_accum)
--log_steps: Wandb 로깅 주기 (Global 스텝 기준)
--save_steps: 체크포인트 저장 주기 (Local 스텝 기준)
--step_offset_s1/s2: 학습 재개 시 데이터/스케줄러 Fast-forward용 오프셋
"""

import os
# DataLoader 워커 Fork 시 Segfault 방지용 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import glob
import logging
import argparse
import contextlib
import functools
from typing import Iterator, Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
# FSDP 모델/옵티마이저 상태 추출 임포트
from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict, StateDictOptions

from datasets import load_dataset, interleave_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model
import wandb
from tqdm import tqdm

# Arrow 멀티스레딩 충돌 방지용 설정
import pyarrow as pa
pa.set_cpu_count(1)
pa.set_io_thread_count(1)

# ==============================================================================
# Logging & Setup
# ==============================================================================
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# [분산 학습 설정]
# NCCL 기반 GPU 통신 초기화 및 local_rank 할당
# ------------------------------------------------------------------------------
def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

# ==============================================================================
# Model Architecture
# ==============================================================================
# ------------------------------------------------------------------------------
# [모델 아키텍처: AudioQwen]
# 오디오 피처 변환 및 LLM 임베딩 주입 모듈
# ------------------------------------------------------------------------------
class AudioQwen(nn.Module):
    def __init__(self, llm_model_name: str, encoder_dim: int, cache_dir: str = None, pad_token_id: int = 151655):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )
        
        llm_dim = self.llm.config.hidden_size
        self.projector = nn.Sequential(
            nn.Conv1d(encoder_dim, llm_dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(llm_dim, llm_dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(llm_dim, llm_dim, kernel_size=1)
        ).to(torch.bfloat16)
        self.proj_norm = nn.LayerNorm(llm_dim).to(torch.bfloat16)
        
        for m in self.projector.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.projector[-1].weight, std=0.02)

    def get_audio_embeds(self, audio_features: torch.Tensor) -> torch.Tensor:
        x = audio_features.transpose(1, 2).to(self.projector[0].weight.dtype)
        x = self.projector(x)
        x = x.transpose(1, 2)
        return self.proj_norm(x)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, audio_features: torch.Tensor, audio_lengths: torch.Tensor, attention_mask: torch.Tensor = None, position_ids: torch.Tensor = None):
        audio_embeds = self.get_audio_embeds(audio_features)
        
        valid_embeds = []
        for i, enc_len in enumerate(audio_lengths):
            t_proj = math.ceil(math.ceil(enc_len.item() / 2) / 2)
            valid_embeds.append(audio_embeds[i, :t_proj, :])
            
        if valid_embeds:
            audio_flat = torch.cat(valid_embeds, dim=0)
        else:
            audio_flat = torch.empty((0, audio_embeds.shape[-1]), device=audio_embeds.device, dtype=audio_embeds.dtype)
        
        inputs_embeds = self.llm.get_input_embeddings()(input_ids).clone()
        audio_mask = (input_ids == self.pad_token_id)
        mask_flat = audio_mask.reshape(-1)
        
        num_placeholders = mask_flat.sum().item()
        num_audio_tokens = audio_flat.shape[0]
        min_len = min(num_placeholders, num_audio_tokens)
        
        if min_len > 0:
            target_indices = mask_flat.nonzero(as_tuple=True)[0][:min_len]
            inputs_embeds.view(-1, inputs_embeds.shape[-1])[target_indices] = audio_flat[:min_len]

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            use_cache=False
        )
        return outputs

# ==============================================================================
# Data Pipeline
# ==============================================================================
# ------------------------------------------------------------------------------
# [데이터 파이프라인: 스트리밍 및 시퀀스 패킹]
# ------------------------------------------------------------------------------
class PackedStreamingDataset(IterableDataset):
    def __init__(self, arrow_paths: List[str], tokenizer, cutoff_len: int = 2048, bucket_size: int = 1000, pad_id: int = 151655):
        self.arrow_paths = arrow_paths
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.bucket_size = bucket_size
        self.pad_id = pad_id
        self.audio_corr_id = tokenizer.convert_tokens_to_ids("<|audio_correspond|>")
        self.ignore_id = -100

    def _generate_processed_samples(self) -> Iterator[Dict]:
        # 워커별 파일 분배 (데이터 중복 방지)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            per_worker = int(math.ceil(len(self.arrow_paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.arrow_paths))
            current_paths = self.arrow_paths[iter_start:iter_end]
        else:
            current_paths = self.arrow_paths

        if not current_paths:
            return

        dataset = load_dataset("arrow", data_files=current_paths, split="train", streaming=True)
        dataset = dataset.shuffle(buffer_size=10000, seed=42)
        
        for row in dataset:
            feat_len = int(row["feat_len"])
            flat_features = row["features"]
            text = row.get("text", "")
            if feat_len <= 0 or not flat_features or not text: continue
            feat_dim = len(flat_features) // feat_len
            audio_feat = torch.tensor(flat_features, dtype=torch.float32).view(feat_len, feat_dim)
            t_audio = math.ceil(math.ceil(feat_len / 2) / 2) 
            text_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not text_ids: continue
            input_ids = [self.pad_id] * t_audio + [self.audio_corr_id] + text_ids + [self.tokenizer.eos_token_id]
            labels = [self.ignore_id] * (t_audio + 1) + text_ids + [self.tokenizer.eos_token_id]
            if len(input_ids) <= self.cutoff_len:
                yield {"input_ids": input_ids, "labels": labels, "audio_features": audio_feat}

    def _pack_samples(self, sample_iterator: Iterator[Dict]) -> Iterator[Dict]:
        buffer = []
        for sample in sample_iterator:
            buffer.append(sample)
            if len(buffer) >= self.bucket_size:
                yield from self._flush_buffer(buffer)
                buffer = []
        if buffer: yield from self._flush_buffer(buffer)

    def _flush_buffer(self, buffer: List[Dict]) -> Iterator[Dict]:
        buffer.sort(key=lambda x: len(x["input_ids"]), reverse=True)
        bins = []
        for sample in buffer:
            placed = False
            for b in bins:
                if b["current_len"] + len(sample["input_ids"]) <= self.cutoff_len:
                    b["input_ids"].extend(sample["input_ids"])
                    b["labels"].extend(sample["labels"])
                    b["attention_mask"].extend([b["seq_count"]] * len(sample["input_ids"]))
                    b["audio_features"].append(sample["audio_features"])
                    b["current_len"] += len(sample["input_ids"])
                    b["seq_count"] += 1
                    placed = True
                    break
            if not placed:
                bins.append({"input_ids": sample["input_ids"][:], "labels": sample["labels"][:], "attention_mask": [1] * len(sample["input_ids"]), "audio_features": [sample["audio_features"]], "current_len": len(sample["input_ids"]), "seq_count": 2})
        for b in bins:
            pad_len = self.cutoff_len - b["current_len"]
            if pad_len > 0:
                b["input_ids"].extend([self.tokenizer.pad_token_id] * pad_len)
                b["labels"].extend([self.ignore_id] * pad_len)
                b["attention_mask"].extend([0] * pad_len) 
            audio_lengths = [f.shape[0] for f in b["audio_features"]]
            max_audio_len = max(audio_lengths)
            feat_dim = b["audio_features"][0].shape[1]
            padded_audio = torch.zeros(len(b["audio_features"]), max_audio_len, feat_dim)
            for i, f in enumerate(b["audio_features"]): padded_audio[i, :f.shape[0], :] = f
            yield {"input_ids": torch.tensor(b["input_ids"], dtype=torch.long), "labels": torch.tensor(b["labels"], dtype=torch.long), "attention_mask": torch.tensor(b["attention_mask"], dtype=torch.long), "audio_features": padded_audio, "audio_lengths": torch.tensor(audio_lengths, dtype=torch.long)}

    def __iter__(self):
        while True:
            iterator = self._generate_processed_samples()
            yield from self._pack_samples(iterator)

def collate_fn(batch: List[Dict]) -> Dict:
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    seq_ids = torch.stack([b["attention_mask"] for b in batch]) 
    attention_mask_2d = (seq_ids != 0).long()
    B, S = seq_ids.shape
    position_ids = torch.zeros((B, S), dtype=torch.long)
    for i in range(B):
        pos = 0; curr_id = -1
        for j in range(S):
            val = seq_ids[i, j].item()
            if val == 0: position_ids[i, j] = 0
            else:
                if val != curr_id: pos = 0; curr_id = val
                position_ids[i, j] = pos; pos += 1
    max_a_len = max(b["audio_features"].shape[1] for b in batch)
    total_audio_items = sum(b["audio_features"].shape[0] for b in batch)
    feat_dim = batch[0]["audio_features"].shape[2]
    batched_audio = torch.zeros(total_audio_items, max_a_len, feat_dim)
    batched_audio_lengths = []
    idx = 0
    for b in batch:
        n_items = b["audio_features"].shape[0]
        batched_audio[idx:idx+n_items, :b["audio_features"].shape[1], :] = b["audio_features"]
        batched_audio_lengths.append(b["audio_lengths"])
        idx += n_items
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask_2d, "position_ids": position_ids, "audio_features": batched_audio, "audio_lengths": torch.cat(batched_audio_lengths, dim=0)}

# ==============================================================================
# Training Engine & Sharded Checkpoint
# ==============================================================================
# ------------------------------------------------------------------------------
# [체크포인트 저장: FSDP 병렬 저장]
# ------------------------------------------------------------------------------
def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[torch.optim.lr_scheduler.LRScheduler], path: str, stage_name: str):
    os.makedirs(path, exist_ok=True)
    sched_state = scheduler.state_dict() if scheduler is not None else None
    
    if isinstance(model, FSDP):
        if stage_name == "Stage1":
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            model_state = get_model_state_dict(model, options=options)
            optim_state = get_optimizer_state_dict(model, optimizer, options=options)
            if is_main_process():
                proj_state = {k: v.cpu() for k, v in model_state.items() if "projector" in k or "proj_norm" in k}
                torch.save({'model': proj_state, 'optimizer': optim_state, 'scheduler': sched_state}, os.path.join(path, "projector.pt"))
        else:
            options = StateDictOptions(full_state_dict=False, cpu_offload=True)
            model_state = get_model_state_dict(model, options=options)
            optim_state = get_optimizer_state_dict(model, optimizer, options=options)
            save_path = os.path.join(path, f"sharded_rank{dist.get_rank()}.pt")
            torch.save({'model': model_state, 'optimizer': optim_state, 'scheduler': sched_state}, save_path)
    else:
        if is_main_process():
            unwrapped_model = model.module if hasattr(model, "module") else model
            model_state = {k: v.cpu() for k, v in unwrapped_model.state_dict().items() if "projector" in k or "proj_norm" in k}
            optim_state = optimizer.state_dict()
            torch.save({'model': model_state, 'optimizer': optim_state, 'scheduler': sched_state}, os.path.join(path, "projector.pt"))

    if is_main_process():
        logger.info(f"[{stage_name}] Checkpoint saved: {path}")

# ------------------------------------------------------------------------------
# [학습 루프]
# ------------------------------------------------------------------------------
def train_loop(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    max_steps: int, 
    grad_accum_steps: int, 
    stage_name: str, 
    device: torch.device, 
    save_dir: str, 
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    step_offset: int = 0,
    global_step_offset: int = 0,
    log_steps: int = 10,
    save_steps: int = 1000
):
    model.train()
    step = step_offset
    accum_loss = 0.0
    progress_bar = tqdm(total=max_steps, initial=step_offset, disable=not is_main_process(), desc=f"Training {stage_name}")
    
    loader_iter = iter(dataloader)
    
    if step_offset > 0:
        if is_main_process():
            logger.info(f"Fast-forwarding to local step {step_offset}")
        if scheduler is not None:
            for _ in range(step_offset): scheduler.step()
        batches_to_skip = step_offset * grad_accum_steps
        for _ in range(batches_to_skip):
            try: next(loader_iter)
            except StopIteration: break
                
    for i, batch in enumerate(loader_iter):
        input_ids = batch["input_ids"].to(device); labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device); position_ids = batch["position_ids"].to(device)
        audio_features = batch["audio_features"].to(device); audio_lengths = batch["audio_lengths"].to(device)
        
        is_accumulating = (i + 1) % grad_accum_steps != 0
        sync_context = model.no_sync() if is_accumulating and hasattr(model, "no_sync") else contextlib.nullcontext()
        
        with sync_context:
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask, position_ids=position_ids, audio_features=audio_features, audio_lengths=audio_lengths)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            
        accum_loss += loss.item()
        
        if not is_accumulating:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None: scheduler.step()
            optimizer.zero_grad()
            
            step += 1
            global_step = global_step_offset + step
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": accum_loss, "global": global_step})
            
            if is_main_process() and wandb.run is not None and step % log_steps == 0:
                lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
                wandb.log({f"{stage_name}/loss": accum_loss, f"{stage_name}/lr": lr}, step=global_step)
            
            accum_loss = 0.0
            if step % save_steps == 0: save_checkpoint(model, optimizer, scheduler, os.path.join(save_dir, f"step_{global_step}"), stage_name)
            if step >= max_steps: break
            
    progress_bar.close()
    save_checkpoint(model, optimizer, scheduler, os.path.join(save_dir, "final"), stage_name)

# ==============================================================================
# Main Execution
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--max_steps_s1", type=int, default=5000)
    parser.add_argument("--max_steps_s2", type=int, default=20000)
    parser.add_argument("--step_offset_s1", type=int, default=0)
    parser.add_argument("--step_offset_s2", type=int, default=0)
    parser.add_argument("--warmup_steps_s2", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    args = parser.parse_args()

    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if is_main_process():
        wandb.init(project="audio-llm-scratch", config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.llm, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|audio_correspond|>"]})

    arrow_files = glob.glob(os.path.join(args.data_dir, "*", "*.arrow"))
    dataset = PackedStreamingDataset(arrow_files, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=8, prefetch_factor=2, pin_memory=True)

    # [Stage 1]
    if is_main_process(): logger.info("=== Stage 1: Alignment ===")
    model = AudioQwen(args.llm, encoder_dim=128).to(device)
    model.llm.resize_token_embeddings(len(tokenizer))
    model.to(torch.bfloat16)
    for p in model.llm.parameters(): p.requires_grad = False
    
    qwen_layer_cls = type(model.llm.model.layers[0])
    fsdp_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={qwen_layer_cls})
    model = FSDP(model, auto_wrap_policy=fsdp_policy, mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16), sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=local_rank, use_orig_params=True, sync_module_states=True)
    
    optimizer_s1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005)
    train_loop(model, dataloader, optimizer_s1, args.max_steps_s1, args.grad_accum, "Stage1", device, os.path.join(args.output_dir, "stage1"), step_offset=args.step_offset_s1, global_step_offset=0, log_steps=args.log_steps, save_steps=args.save_steps)
    
    # [Stage 2]
    dist.barrier()
    if is_main_process(): logger.info("=== Stage 2: LoRA Fine-tuning ===")
    del model, optimizer_s1
    torch.cuda.empty_cache()
    
    model = AudioQwen(args.llm, encoder_dim=128)
    model.llm.resize_token_embeddings(len(tokenizer))
    if is_main_process():
        proj_path = os.path.join(args.output_dir, "stage1", "final", "projector.pt")
        if os.path.exists(proj_path):
            ckpt = torch.load(proj_path)
            model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=False)
            
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model.llm = get_peft_model(model.llm, lora_config)
    model.to(torch.bfloat16)
    
    qwen_layer_cls = type(model.llm.base_model.model.model.layers[0])
    fsdp_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={qwen_layer_cls})
    model = FSDP(model, auto_wrap_policy=fsdp_policy, mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16), sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=local_rank, use_orig_params=True, sync_module_states=True)
    
    optimizer_s2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    scheduler_s2 = get_cosine_schedule_with_warmup(optimizer_s2, num_warmup_steps=args.warmup_steps_s2, num_training_steps=args.max_steps_s2)
    train_loop(model, dataloader, optimizer_s2, args.max_steps_s2, args.grad_accum, "Stage2", device, os.path.join(args.output_dir, "stage2"), scheduler=scheduler_s2, step_offset=args.step_offset_s2, global_step_offset=args.max_steps_s1, log_steps=args.log_steps, save_steps=args.save_steps)
    
    cleanup_distributed()
    if is_main_process() and wandb.run is not None: wandb.finish()

if __name__ == "__main__":
    main()