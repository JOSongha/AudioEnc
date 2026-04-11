"""
2-Stage ASR 학습 루프.

사용법:
    torchrun --nproc_per_node=8 train.py --encoder encodec
    torchrun --nproc_per_node=8 train.py --encoder dac
    torchrun --nproc_per_node=8 train.py --encoder fb_dacvae
    torchrun --nproc_per_node=8 train.py --encoder mimi_acoustic
    torchrun --nproc_per_node=8 train.py --encoder mimi_semantic

최적화 플래그 (기본값 모두 off):
    --flash-attn          Flash Attention 2 (flash-attn 이미 설치됨)
    --liger               Liger fused kernels (pip install liger-kernel)
    --packing             Sequence packing (PackedDataset + PackedCollator)
    --cutoff-len N        Packing 시퀀스 최대 길이 (기본 2048)
    --fsdp                FSDP (DDP 대체, 4B+ 모델 권장)
"""

import argparse
import gc
import os
import signal
import warnings
from datetime import timedelta

import torch
import torchaudio
import wandb
from accelerate import Accelerator, InitProcessGroupKwargs
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from config import get_config
from dataset import (
    build_datasets, collate_fn_factory, get_dataset_lengths, DynamicBatchSampler,
    PackedDataset, PackedCollator, build_packed_processor,
)
from encoders import build_encoder
from model import AudioQwen

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"]   = "false"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "600"
os.environ["NCCL_DEBUG"] = "WARN"

_stop_stage1 = False

def _handle_sigusr1(signum, frame):
    global _stop_stage1
    _stop_stage1 = True


# ==========================================
# Utilities
# ==========================================

def make_scheduler(optimizer, dataloader, epochs, cfg, accelerator):
    total_steps  = (len(dataloader) // cfg["gradient_accumulation_steps"]) * epochs
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)


@torch.no_grad()
def run_validation(model, val_loader, accelerator):
    model.eval()
    total_loss, n = 0.0, 0
    for audio, audio_lengths, transcript_ids in val_loader:
        audio          = audio.to(accelerator.device)
        audio_lengths  = audio_lengths.to(accelerator.device)
        transcript_ids = transcript_ids.to(accelerator.device)
        outputs = model(audio, audio_lengths=audio_lengths, transcript_input_ids=transcript_ids)
        loss = accelerator.gather(outputs.loss.detach().unsqueeze(0)).mean()
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


def build_accelerator(cfg):
    """cfg 플래그에 따라 DDP 또는 FSDP Accelerator 생성."""
    # FA2 사용 시 bf16 autocast 활성화 (모델은 이미 bf16으로 로드됨)
    mp = "bf16" if cfg.get("attn_implementation") == "flash_attention_2" else "no"
    kwargs = dict(
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        mixed_precision=mp,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=7200))],
    )
    if cfg.get("use_fsdp", False):
        import torch
        from accelerate.utils import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp import FullStateDictConfig, FullOptimStateDictConfig
        fsdp_plugin = FullyShardedDataParallelPlugin(
            use_orig_params=True,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        )
        kwargs["fsdp_plugin"] = fsdp_plugin
        print("FSDP enabled (use_orig_params=True)")
    return Accelerator(**kwargs)


def build_model(cfg, accelerator):
    """rank-0 먼저 로드해서 HF 캐시 생성, 나머지는 캐시에서 로드."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    enc_cfg    = cfg["encoder"]
    cache_dir  = cfg["model_cache_dir"]
    enc_name   = cfg["encoder_name"]

    if local_rank == 0:
        encoder = build_encoder(enc_name, enc_cfg, cache_dir)
        model   = AudioQwen(encoder, cfg)
    accelerator.wait_for_everyone()
    if local_rank != 0:
        encoder = build_encoder(enc_name, enc_cfg, cache_dir)
        model   = AudioQwen(encoder, cfg)
    accelerator.wait_for_everyone()
    return model



# ==========================================
# Stage 1: Projector Alignment
# ==========================================

def run_stage1(cfg, accelerator, train_dataset, val_dataset, debug=""):
    enc_name      = cfg["encoder_name"]
    proj_path     = os.path.join(cfg["model_cache_dir"], f"s1_proj_{enc_name}.pt")

    if os.path.exists(proj_path):
        if accelerator.is_main_process:
            print(f"\n[Stage 1 Skip] checkpoint found: {proj_path}")
        return proj_path, 0  # global_step=0 (Stage 2에서 초기화)

    pid_path = os.path.join(cfg["model_cache_dir"], "train.pid")
    if accelerator.is_main_process:
        signal.signal(signal.SIGUSR1, _handle_sigusr1)
        with open(pid_path, "w") as f:
            f.write(str(os.getpid()))
        print(f"\n{'='*45}")
        print(f"Stage 1: Projector Alignment  LR={cfg['stage1_lr']}")
        print(f"  Early stop: kill -USR1 $(cat {pid_path})")
        print(f"{'='*45}\n")

    model     = build_model(cfg, accelerator)
    tokenizer = model.tokenizer
    cache_dir = cfg["model_cache_dir"]

    use_packing  = cfg.get("use_packing", False)
    spt = cfg["samples_per_token"]
    mt  = cfg["max_text_len"]
    mbt = cfg["max_batch_tokens"]
    num_replicas = accelerator.num_processes
    rank         = accelerator.process_index

    if use_packing:
        # ── packing 경로 ──────────────────────────────────────────────────
        if accelerator.is_main_process:
            print("Computing token lengths for sequence packing (Stage 1)...")
        token_lengths = get_dataset_lengths(train_dataset, spt, mt, cache_dir=cache_dir)
        processor     = build_packed_processor(tokenizer, cfg)
        train_packed  = PackedDataset(
            source_dataset=train_dataset,
            processor_fn=processor,
            token_lengths=token_lengths,
            cutoff_len=cfg["packing_cutoff_len"],
            pad_token_id=tokenizer.pad_token_id,
        )
        collator        = PackedCollator(
            pad_token_id=tokenizer.pad_token_id,
            attn_implementation=cfg.get("attn_implementation", "eager"),
        )
        train_dist_sampler = DistributedSampler(
            train_packed, num_replicas=num_replicas, rank=rank, shuffle=True,
        )
        train_loader = DataLoader(
            train_packed, batch_size=4, sampler=train_dist_sampler,
            collate_fn=collator, num_workers=4, pin_memory=True, persistent_workers=True,
        )
        # val: 기존 방식 유지 (DynamicBatchSampler)
        collate_val = collate_fn_factory(tokenizer, cfg["max_text_len"])
        val_lengths = get_dataset_lengths(val_dataset, spt, mt, cache_dir=cache_dir)
        val_sampler = DynamicBatchSampler(
            val_lengths, mbt, num_replicas=num_replicas, rank=rank,
        )
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                                collate_fn=collate_val, num_workers=2, pin_memory=True)
    else:
        # ── 기존 경로 ────────────────────────────────────────────────────
        collate = collate_fn_factory(tokenizer, cfg["max_text_len"])
        if accelerator.is_main_process:
            print("Computing dataset lengths for bucket sampler...")
        train_lengths = get_dataset_lengths(train_dataset, spt, mt, cache_dir=cache_dir)
        val_lengths   = get_dataset_lengths(val_dataset,   spt, mt, cache_dir=cache_dir)
        train_sampler = DynamicBatchSampler(train_lengths, mbt, num_replicas=num_replicas, rank=rank)
        val_sampler   = DynamicBatchSampler(val_lengths,   mbt, num_replicas=num_replicas, rank=rank)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                  collate_fn=collate, num_workers=4, pin_memory=True,
                                  persistent_workers=True, prefetch_factor=2)
        val_loader   = DataLoader(val_dataset,   batch_sampler=val_sampler,
                                  collate_fn=collate, num_workers=2, pin_memory=True)

    model.freeze_llm()
    if "c" in debug:
        model.init_ctc_head()

    # FSDP requires uniform dtype across all parameters — cast entire model to bf16
    if cfg.get("use_fsdp", False):
        model.bfloat16()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["stage1_lr"],
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    scheduler = make_scheduler(optimizer, train_loader, cfg["stage1_epochs"], cfg, accelerator)

    global_step = 0
    best_val_loss = float("inf")
    save_steps = cfg.get("save_steps", 0)
    cache_dir  = cfg["model_cache_dir"]
    for epoch in range(cfg["stage1_epochs"]):
        if use_packing:
            train_dist_sampler.set_epoch(epoch)
        else:
            train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        progress = tqdm(train_loader, desc=f"Stage1 Epoch {epoch+1}",
                        disable=not accelerator.is_main_process)
        accum_loss_buf = torch.zeros(1, device=accelerator.device)
        accum_count, accum_bsz = 0, 0
        log_every = cfg.get("log_every", 1)

        for batch in progress:
            with accelerator.accumulate(model):
                if use_packing:
                    # ── packed 경로 ──────────────────────────────────────
                    batch = {k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    outputs = model(
                        input_ids=batch["input_ids"],
                        labels=batch["labels"],
                        audio_features=batch["audio_features"],
                        audio_feat_lengths=batch["audio_lengths"],
                        attention_mask=batch.get("attention_mask"),
                        position_ids=batch.get("position_ids"),
                    )
                    if accelerator.is_main_process:
                        accum_bsz += batch["input_ids"].shape[0]
                else:
                    # ── 기존 경로 ────────────────────────────────────────
                    audio, audio_lengths, transcript_ids = batch
                    audio_lengths  = audio_lengths.to(accelerator.device)
                    transcript_ids = transcript_ids.to(accelerator.device)
                    if "c" in debug:
                        texts = tokenizer.batch_decode(transcript_ids, skip_special_tokens=True)
                        ctc_tgt, ctc_tgt_len = _text_to_ctc_targets(texts)
                        outputs = model(audio, audio_lengths=audio_lengths,
                                        transcript_input_ids=transcript_ids,
                                        ctc_targets=ctc_tgt, ctc_target_lengths=ctc_tgt_len)
                    else:
                        outputs = model(audio, audio_lengths=audio_lengths,
                                        transcript_input_ids=transcript_ids)
                    if accelerator.is_main_process:
                        accum_bsz += audio.shape[0]
                accelerator.backward(outputs.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    # 모든 rank가 동일한 step 수를 밟으므로 broadcast 불필요
                    global_step += 1
                    if accelerator.is_main_process:
                        lr_now = scheduler.get_last_lr()[0]
                        accum_loss_buf += outputs.loss.detach()
                        accum_count += 1
                        if global_step % log_every == 0:
                            avg_loss = (accum_loss_buf / accum_count).item()
                            accum_loss_buf.zero_()
                            accum_count = 0
                            wandb.log({"stage": 1, "train/loss": avg_loss,
                                       "train/lr": lr_now, "train/batch_size": accum_bsz},
                                      step=global_step)
                            progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr_now:.2e}",
                                                 bsz=accum_bsz)
                        accum_bsz = 0
                    if "w" in debug and global_step in (1, 20):
                        debug_dir = "/mnt/tmp/cache/weightCmprsn"
                        os.makedirs(debug_dir, exist_ok=True)
                        unwrapped  = accelerator.unwrap_model(model)
                        debug_path = os.path.join(debug_dir, f"full_weights_step{global_step}.pt")
                        torch.save({k: v.cpu() for k, v in unwrapped.state_dict().items()}, debug_path)
                        print(f"  [DEBUG] Full weights saved → {debug_path}")
                        if global_step == 20:
                            print("  [DEBUG] step 20 reached — exiting")
                            raise SystemExit(0)
                    if save_steps and global_step % save_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            unwrapped  = accelerator.unwrap_model(model)
                            proj_state = {k: v.cpu().half() for k, v in unwrapped.state_dict().items()
                                          if "projector" in k or "proj_norm" in k}
                            step_proj_path = os.path.join(
                                cache_dir, f"s1_proj_{enc_name}_ep{epoch+1}_step{global_step}.pt"
                            )
                            torch.save(proj_state, step_proj_path)
                            print(f"  [Step {global_step}] Projector saved → {step_proj_path}")

        val_loss = run_validation(model, val_loader, accelerator)
        if accelerator.is_main_process:
            print(f"  [Stage1 Epoch {epoch+1}] val_loss={val_loss:.4f}")
            wandb.log({"stage": 1, "val/loss": val_loss, "epoch": epoch + 1},
                      step=global_step)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped  = accelerator.unwrap_model(model)
                proj_state = {k: v.cpu().half() for k, v in unwrapped.state_dict().items()
                              if "projector" in k or "proj_norm" in k}
                named_proj_path = os.path.join(
                    cache_dir, f"s1_proj_{enc_name}_ep{epoch+1}_step{global_step}_best.pt"
                )
                torch.save(proj_state, named_proj_path)
                torch.save(proj_state, proj_path)
                print(f"  Projector saved (ep={epoch+1}, step={global_step}, val_loss={val_loss:.4f}) → {named_proj_path}")

        # SIGUSR1 수신 시 조기 종료 (모든 rank 동기화)
        stop_flag = torch.zeros(1, device=accelerator.device)
        if accelerator.is_main_process and _stop_stage1:
            stop_flag[0] = 1.0
            print(f"  [Stage 1] SIGUSR1 received — stopping early after epoch {epoch+1}")
        torch.distributed.broadcast(stop_flag, src=0)
        if stop_flag[0].item() == 1.0:
            break

    # PID 파일 정리
    if accelerator.is_main_process and os.path.exists(pid_path):
        os.remove(pid_path)

    # VRAM 해제
    accelerator.wait_for_everyone()
    if hasattr(accelerator, "free_memory"):
        accelerator.free_memory()
    for attr in ("_models", "_optimizers", "_dataloaders"):
        if hasattr(accelerator, attr):
            getattr(accelerator, attr).clear()
    del model, optimizer, train_loader, val_loader, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    return proj_path, global_step


# ==========================================
# Stage 2: LoRA Fine-tuning (+ Resume)
# ==========================================

def run_stage2(cfg, accelerator, train_dataset, val_dataset, proj_path, step_offset):
    enc_name  = cfg["encoder_name"]
    cache_dir = cfg["model_cache_dir"]

    s2_lr     = cfg["stage2_lr"]
    s2_epochs = cfg["stage2_epochs"]

    if accelerator.is_main_process:
        print(f"\n{'='*45}")
        print(f"Stage 2: LoRA Fine-tuning  LR={s2_lr}  epochs={s2_epochs}")
        print(f"{'='*45}\n")

    model = build_model(cfg, accelerator)
    model.apply_lora()

    if proj_path is None:
        proj_path = os.path.join(cache_dir, f"s1_proj_{enc_name}.pt")
    if accelerator.is_main_process:
        print(f"Loading Stage 1 projector weights from {proj_path}...")
    proj_state = torch.load(proj_path, map_location="cpu", weights_only=True)
    model.load_state_dict(proj_state, strict=False)

    # Optimizer
    try:
        import bitsandbytes as bnb
        opt_cls = bnb.optim.AdamW8bit
        if accelerator.is_main_process:
            print("Using bitsandbytes 8-bit AdamW")
    except ImportError:
        opt_cls = torch.optim.AdamW
        if accelerator.is_main_process:
            print("bitsandbytes not found, using standard AdamW")

    optimizer = opt_cls(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=s2_lr, weight_decay=0.01,
    )

    tokenizer    = model.tokenizer
    use_packing  = cfg.get("use_packing", False)
    num_replicas = accelerator.num_processes
    rank         = accelerator.process_index
    cache_dir    = cfg["model_cache_dir"]
    spt  = cfg["samples_per_token"]
    mt   = cfg["max_text_len"]
    mbt  = cfg["max_batch_tokens"]

    if use_packing:
        # ── packing 경로 ──────────────────────────────────────────────────
        if accelerator.is_main_process:
            print("Computing token lengths for sequence packing (Stage 2)...")
        token_lengths = get_dataset_lengths(train_dataset, spt, mt, cache_dir=cache_dir)
        processor     = build_packed_processor(tokenizer, cfg)
        train_packed  = PackedDataset(
            source_dataset=train_dataset,
            processor_fn=processor,
            token_lengths=token_lengths,
            cutoff_len=cfg["packing_cutoff_len"],
            pad_token_id=tokenizer.pad_token_id,
        )
        collator           = PackedCollator(
            pad_token_id=tokenizer.pad_token_id,
            attn_implementation=cfg.get("attn_implementation", "eager"),
        )
        train_dist_sampler = DistributedSampler(
            train_packed, num_replicas=num_replicas, rank=rank, shuffle=True,
        )
        train_loader = DataLoader(
            train_packed, batch_size=4, sampler=train_dist_sampler,
            collate_fn=collator, num_workers=1, pin_memory=True, persistent_workers=True,
        )
        collate_val = collate_fn_factory(tokenizer, cfg["max_text_len"])
        val_lengths = get_dataset_lengths(val_dataset, spt, mt, cache_dir=cache_dir)
        val_sampler = DynamicBatchSampler(val_lengths, mbt, num_replicas=num_replicas, rank=rank)
        val_loader  = DataLoader(val_dataset, batch_sampler=val_sampler,
                                 collate_fn=collate_val, num_workers=2, pin_memory=True)
    else:
        # ── 기존 경로 ────────────────────────────────────────────────────
        collate       = collate_fn_factory(tokenizer, cfg["max_text_len"])
        train_lengths = get_dataset_lengths(train_dataset, spt, mt, cache_dir=cache_dir)
        val_lengths   = get_dataset_lengths(val_dataset,   spt, mt, cache_dir=cache_dir)
        train_sampler = DynamicBatchSampler(train_lengths, mbt, num_replicas=num_replicas, rank=rank)
        val_sampler   = DynamicBatchSampler(val_lengths,   mbt, num_replicas=num_replicas, rank=rank)
        train_loader  = DataLoader(train_dataset, batch_sampler=train_sampler,
                                   collate_fn=collate, num_workers=8, pin_memory=True)
        val_loader    = DataLoader(val_dataset,   batch_sampler=val_sampler,
                                   collate_fn=collate, num_workers=8, pin_memory=True)

    model, optimizer = accelerator.prepare(model, optimizer)
    scheduler = make_scheduler(optimizer, train_loader, s2_epochs, cfg, accelerator)


    best_val_loss = float("inf")
    epoch_offset  = cfg["stage1_epochs"] + cfg["stage2_epochs"]
    stage_key     = 2
    global_step   = step_offset
    save_steps    = cfg.get("save_steps", 0)

    for epoch in range(s2_epochs):
        bs = getattr(train_loader, "batch_sampler", None)
        if hasattr(bs, "set_epoch"):
            bs.set_epoch(epoch)
        model.train()
        progress = tqdm(train_loader, desc=f"Stage{stage_key} Epoch {epoch+1}",
                        disable=not accelerator.is_main_process)
        accum_loss_buf = torch.zeros(1, device=accelerator.device)
        accum_count, accum_bsz = 0, 0
        log_every = cfg.get("log_every", 1)

        for batch in progress:
            with accelerator.accumulate(model):
                if use_packing:
                    # ── packed 경로 ──────────────────────────────────────
                    batch = {k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    outputs = model(
                        input_ids=batch["input_ids"],
                        labels=batch["labels"],
                        audio_features=batch["audio_features"],
                        audio_feat_lengths=batch["audio_lengths"],
                        attention_mask=batch.get("attention_mask"),
                        position_ids=batch.get("position_ids"),
                    )
                    if accelerator.is_main_process:
                        accum_bsz += batch["input_ids"].shape[0]
                else:
                    # ── 기존 경로 ────────────────────────────────────────
                    audio, audio_lengths, transcript_ids = batch
                    audio_lengths  = audio_lengths.to(accelerator.device)
                    transcript_ids = transcript_ids.to(accelerator.device)
                    outputs = model(audio, audio_lengths=audio_lengths,
                                    transcript_input_ids=transcript_ids)
                    if accelerator.is_main_process:
                        accum_bsz += audio.shape[0]
                accelerator.backward(outputs.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    # 모든 rank가 동일한 step 수를 밟으므로 broadcast 불필요
                    global_step += 1
                    if accelerator.is_main_process:
                        lr_now = scheduler.get_last_lr()[0]
                        accum_loss_buf += outputs.loss.detach()
                        accum_count += 1
                        if global_step % log_every == 0:
                            avg_loss = (accum_loss_buf / accum_count).item()
                            accum_loss_buf.zero_()
                            accum_count = 0
                            wandb.log({"stage": stage_key, "train/loss": avg_loss,
                                       "train/lr": lr_now, "train/batch_size": accum_bsz},
                                      step=global_step)
                            progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr_now:.2e}",
                                                 bsz=accum_bsz)
                        accum_bsz = 0
                    if save_steps and global_step % save_steps == 0:
                        accelerator.wait_for_everyone()
                        step_dir = os.path.join(
                            cache_dir, f"step{global_step}_{enc_name}_ckpt",
                        )
                        accelerator.save_state(step_dir)
                        if accelerator.is_main_process:
                            print(f"  [Step {global_step}] Checkpoint saved → {step_dir}")

        val_loss = run_validation(model, val_loader, accelerator)
        if accelerator.is_main_process:
            print(f"  [Stage{stage_key} Epoch {epoch+1}] val_loss={val_loss:.4f}")
            wandb.log({"stage": stage_key, "val/loss": val_loss,
                       "epoch": epoch + 1 + epoch_offset}, step=global_step)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            accelerator.wait_for_everyone()
            best_dir = os.path.join(
                cache_dir,
                f"best_{enc_name}_ckpt_ep{epoch+1}_step{global_step}",
            )
            accelerator.save_state(best_dir)
            if accelerator.is_main_process:
                print(f"  Best checkpoint saved (epoch={epoch+1}, step={global_step}, val_loss={val_loss:.4f}) → {best_dir}")

    # 최종 checkpoint
    accelerator.wait_for_everyone()
    final_dir = os.path.join(cache_dir, f"final_{enc_name}_ckpt_ep{s2_epochs}_step{global_step}")
    accelerator.save_state(final_dir)
    if accelerator.is_main_process:
        print(f"Final checkpoint saved → {final_dir}")


# ==========================================
# Entry point
# ==========================================

# CTC 문자 변환 (blank=0, a-z=1-26, space=27)
_CTC_CHAR_TO_IDX = {c: i + 1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
_CTC_CHAR_TO_IDX[" "] = 27


def _text_to_ctc_targets(texts: list[str]):
    seqs = []
    for text in texts:
        seq = [_CTC_CHAR_TO_IDX[c] for c in text.lower() if c in _CTC_CHAR_TO_IDX]
        seqs.append(seq if seq else [27])
    target_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    targets = torch.zeros(len(seqs), max(len(s) for s in seqs), dtype=torch.long)
    for i, s in enumerate(seqs):
        targets[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return targets, target_lengths


def _parse_datasets(s: str) -> list:
    valid = {"ls100", "ls360", "ls500", "mls", "gs", "vp"}
    items = [x.strip() for x in s.split(",") if x.strip()]
    unknown = set(items) - valid
    if unknown:
        raise argparse.ArgumentTypeError(
            f"알 수 없는 dataset: {unknown}. 선택 가능: {valid}"
        )
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True,
                        choices=["encodec", "dac", "fb_dacvae", "mimi_acoustic", "mimi_semantic"],
                        help="사용할 audio encoder")
    parser.add_argument("--llm",        default=None,
                        help="LLM 모델 이름 (예: Qwen/Qwen3.5-0.8B). 기본: config 값")
    parser.add_argument("--data-path",  default=None, help="데이터 루트 경로 (기본: config 값)")
    parser.add_argument("--cache-dir",  default=None, help="모델 캐시 경로 (기본: config 값)")
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--debug",      default="",   help="디버그 옵션 (w: step 50에 full weight 저장)")

    # 데이터셋 선택 및 샘플 수
    parser.add_argument("--datasets",
                        default=None, type=_parse_datasets,
                        metavar="ls100,ls360,ls500,mls,gs,vp",
                        help="사용할 데이터셋 (쉼표 구분). 기본: ls100,ls360,ls500,mls")
    parser.add_argument("--ls-samples",     default=None, type=int,
                        metavar="N", help="LibriSpeech 서브샘플 수 (Stage 2, 기본: 전체)")
    parser.add_argument("--mls-samples",    default=None, type=int,
                        metavar="N", help="MLS 샘플 수 (Stage 2, 기본: 전체)")
    parser.add_argument("--gs-subset",      default="xl",
                        choices=["xs", "s", "m", "l", "xl"],
                        help="GigaSpeech subset (기본: l=2500h)")
    parser.add_argument("--gs-samples",     default=None, type=int,
                        metavar="N", help="GigaSpeech 샘플 수 (기본: 전체)")
    parser.add_argument("--vp-samples",     default=None, type=int,
                        metavar="N", help="VoxPopuli 샘플 수 (Stage 2, 기본: 전체)")
    parser.add_argument("--s1-datasets",
                        default=None, type=_parse_datasets,
                        metavar="ls100,ls360,mls,vp",
                        help="Stage 1 전용 데이터셋. 미지정 시 --datasets 사용")
    parser.add_argument("--s1-ls-samples",  default=None, type=int,
                        metavar="N", help="Stage 1 LibriSpeech 서브샘플 수 (기본: --ls-samples)")
    parser.add_argument("--s1-mls-samples", default=None, type=int,
                        metavar="N", help="Stage 1 MLS 샘플 수 (기본: --mls-samples)")
    parser.add_argument("--s1-gs-samples",  default=None, type=int,
                        metavar="N", help="Stage 1 GigaSpeech 샘플 수 (기본: --gs-samples)")
    parser.add_argument("--s1-vp-samples",  default=None, type=int,
                        metavar="N", help="Stage 1 VoxPopuli 샘플 수 (기본: --vp-samples)")

    # ── 최적화 플래그 ──────────────────────────────────────────────────────
    parser.add_argument("--packing",    action="store_true",
                        help="Sequence packing (PackedDataset + PackedCollator)")
    parser.add_argument("--flash-attn", action="store_true",
                        help="Flash Attention 2 (flash-attn 이미 설치됨)")
    parser.add_argument("--liger",      action="store_true",
                        help="Liger fused kernels (pip install liger-kernel 필요)")
    parser.add_argument("--fsdp",       action="store_true",
                        help="FSDP (DDP 대체, 4B+ 모델 권장)")
    parser.add_argument("--cutoff-len", default=2048, type=int,
                        metavar="N", help="Packing 시퀀스 최대 길이 (기본: 2048)")
    parser.add_argument("--stage1-epochs", type=int, default=None,
                        metavar="N", help="Stage 1 epoch 수 (기본: config 값)")
    parser.add_argument("--stage2-epochs", type=int, default=None,
                        metavar="N", help="Stage 2 epoch 수 (기본: config 값)")
    parser.add_argument("--stage", choices=["1", "2", "all"], default="all",
                        help="실행할 stage 선택 (기본: all)")
    args = parser.parse_args()

    cfg = get_config(args.encoder)
    if args.llm:            cfg["llm_model"] = args.llm
    if args.data_path:      cfg["data_path"]       = args.data_path
    if args.cache_dir:      cfg["model_cache_dir"] = args.cache_dir
    if args.wandb_mode:     cfg["wandb_mode"]      = args.wandb_mode
    if args.datasets:       cfg["datasets"]        = args.datasets
    if args.ls_samples  is not None: cfg["librispeech_num_samples"] = args.ls_samples
    if args.mls_samples is not None: cfg["mls_num_samples"]         = args.mls_samples
    cfg["gs_subset"]     = args.gs_subset
    if args.gs_samples  is not None: cfg["gs_num_samples"]          = args.gs_samples
    if args.vp_samples  is not None: cfg["vp_num_samples"]          = args.vp_samples

    # 최적화 플래그 반영
    if args.packing:    cfg["use_packing"]         = True
    if args.flash_attn: cfg["attn_implementation"] = "flash_attention_2"
    if args.liger:      cfg["use_liger_kernel"]    = True
    if args.fsdp:       cfg["use_fsdp"]            = True
    cfg["packing_cutoff_len"] = args.cutoff_len
    if args.stage1_epochs is not None: cfg["stage1_epochs"] = args.stage1_epochs
    if args.stage2_epochs is not None: cfg["stage2_epochs"] = args.stage2_epochs

    os.makedirs(cfg["model_cache_dir"], exist_ok=True)
    os.environ.setdefault("HF_HOME",    cfg["model_cache_dir"])
    os.environ.setdefault("TORCH_HOME", os.path.join(os.path.dirname(cfg["model_cache_dir"]), "torch"))

    accelerator = build_accelerator(cfg)

    if accelerator.is_main_process:
        import datetime
        llm_tag  = "2b" if "2B" in cfg["llm_model"] else "4b"
        run_name = f"{args.encoder}_{llm_tag}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
        wandb.init(project=cfg["project_name"], config=cfg,
                   name=run_name, mode=cfg["wandb_mode"])

    # 데이터셋 다운로드 (rank 0만, 선택된 LibriSpeech split만)
    all_datasets = cfg.get("datasets", ["ls100", "ls360", "ls500", "mls"])
    s1_datasets  = args.s1_datasets or all_datasets
    _ls_url_map  = {"ls100": "train-clean-100", "ls360": "train-clean-360", "ls500": "train-other-500"}
    need_splits  = {"dev-clean"}
    for ds_list in (all_datasets, s1_datasets):
        for name in ds_list:
            if name in _ls_url_map:
                need_splits.add(_ls_url_map[name])
    if accelerator.is_main_process:
        for split in sorted(need_splits):
            torchaudio.datasets.LIBRISPEECH(root=cfg["data_path"], url=split, download=True)
    accelerator.wait_for_everyone()

    # Stage 1 cfg
    stage1_cfg = dict(cfg)
    stage1_cfg["datasets"] = s1_datasets
    stage1_cfg["librispeech_num_samples"] = (
        args.s1_ls_samples  if args.s1_ls_samples  is not None else cfg.get("librispeech_num_samples")
    )
    stage1_cfg["mls_num_samples"] = (
        args.s1_mls_samples if args.s1_mls_samples is not None else cfg.get("mls_num_samples")
    )
    stage1_cfg["gs_num_samples"] = (
        args.s1_gs_samples  if args.s1_gs_samples  is not None else cfg.get("gs_num_samples")
    )
    stage1_cfg["vp_num_samples"] = (
        args.s1_vp_samples  if args.s1_vp_samples  is not None else cfg.get("vp_num_samples")
    )

    stage1_train_dataset, val_dataset = build_datasets(stage1_cfg)
    stage2_train_dataset, _           = build_datasets(cfg)

    proj_path, step_offset = None, 0
    if args.stage in ("all", "1"):
        proj_path, step_offset = run_stage1(cfg, accelerator, stage1_train_dataset, val_dataset,
                                            debug=args.debug)
    if args.stage in ("all", "2"):
        run_stage2(cfg, accelerator, stage2_train_dataset, val_dataset, proj_path, step_offset)

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
