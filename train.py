"""
2-Stage ASR 학습 루프.

사용법:
    torchrun --nproc_per_node=8 train.py --encoder encodec
    torchrun --nproc_per_node=8 train.py --encoder dac
    torchrun --nproc_per_node=8 train.py --encoder dac_vae
    torchrun --nproc_per_node=8 train.py --encoder mimi_acoustic
    torchrun --nproc_per_node=8 train.py --encoder mimi_semantic
"""

import argparse
import gc
import os
import warnings
from datetime import timedelta

import torch
import torchaudio
import wandb
from accelerate import Accelerator, InitProcessGroupKwargs
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from config import get_config
from dataset import build_datasets, collate_fn_factory, get_dataset_lengths, DynamicBatchSampler
from encoders import build_encoder
from model import AudioQwen

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"]   = "false"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "600"
os.environ["NCCL_DEBUG"] = "WARN"


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


def ckpt_name(base: str, resume_round: int, suffix: str = "") -> str:
    """체크포인트 디렉토리 이름 생성."""
    r = f"_r{resume_round}" if resume_round > 0 else ""
    s = f"_{suffix}"        if suffix         else ""
    return f"{base}{r}{s}"


# ==========================================
# Stage 1: Projector Alignment
# ==========================================

def run_stage1(cfg, accelerator, train_dataset, val_dataset):
    enc_name      = cfg["encoder_name"]
    proj_path     = os.path.join(cfg["model_cache_dir"], f"s1_projector_{enc_name}.pt")

    if os.path.exists(proj_path):
        if accelerator.is_main_process:
            print(f"\n[Stage 1 Skip] checkpoint found: {proj_path}")
        return proj_path, 0  # global_step=0 (Stage 2에서 초기화)

    if accelerator.is_main_process:
        print(f"\n{'='*45}")
        print(f"Stage 1: Projector Alignment  LR={cfg['stage1_lr']}")
        print(f"{'='*45}\n")

    model    = build_model(cfg, accelerator)
    tokenizer = model.tokenizer
    collate  = collate_fn_factory(tokenizer, cfg["max_text_len"])

    num_replicas = accelerator.num_processes
    rank         = accelerator.process_index
    cache_dir    = cfg["model_cache_dir"]
    if accelerator.is_main_process:
        print("Computing dataset lengths for bucket sampler...")
    spt  = cfg["samples_per_token"]
    mt   = cfg["max_text_len"]
    mbt  = cfg["max_batch_tokens"]
    train_lengths = get_dataset_lengths(train_dataset, spt, mt, cache_dir=cache_dir)
    val_lengths   = get_dataset_lengths(val_dataset,   spt, mt, cache_dir=cache_dir)

    train_sampler = DynamicBatchSampler(train_lengths, mbt, num_replicas=num_replicas, rank=rank)
    val_sampler   = DynamicBatchSampler(val_lengths,   mbt, num_replicas=num_replicas, rank=rank)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              collate_fn=collate, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_sampler=val_sampler,
                              collate_fn=collate, num_workers=2, pin_memory=True)

    model.freeze_llm()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["stage1_lr"],
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    scheduler = make_scheduler(optimizer, train_loader, cfg["stage1_epochs"], cfg, accelerator)

    global_step = 0
    best_val_loss = float("inf")
    for epoch in range(cfg["stage1_epochs"]):
        train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        progress = tqdm(train_loader, desc=f"Stage1 Epoch {epoch+1}",
                        disable=not accelerator.is_main_process)
        accum_loss, accum_count, accum_bsz = 0.0, 0, 0

        for audio, audio_lengths, transcript_ids in progress:
            audio_lengths  = audio_lengths.to(accelerator.device)
            transcript_ids = transcript_ids.to(accelerator.device)
            with accelerator.accumulate(model):
                outputs = model(audio, audio_lengths=audio_lengths,
                                transcript_input_ids=transcript_ids)
                accelerator.backward(outputs.loss)
                if accelerator.is_main_process:
                    accum_bsz += audio.shape[0]
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    if accelerator.is_main_process:
                        global_step += 1
                        lr_now = scheduler.get_last_lr()[0]
                        accum_loss  += outputs.loss.item()
                        accum_count += 1
                        avg_loss = accum_loss / accum_count
                        wandb.log({"stage": 1, "train/loss": avg_loss,
                                   "train/lr": lr_now, "train/batch_size": accum_bsz},
                                  step=global_step)
                        progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr_now:.2e}",
                                             bsz=accum_bsz)
                        accum_bsz = 0

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
                torch.save(proj_state, proj_path)
                print(f"  Projector saved (val_loss={val_loss:.4f}) → {proj_path}")

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

    # 최신 resume checkpoint 자동 탐색 (높은 round 먼저)
    # rN → rN-1 → ... → r1 → fresh 순으로 탐색, 최대 10회 resume까지 지원
    MAX_RESUME = 10
    resume_candidates = [
        (os.path.join(cache_dir, ckpt_name(f"best_{enc_name}_ckpt", r)), r + 1)
        for r in range(MAX_RESUME, 0, -1)
    ] + [(os.path.join(cache_dir, f"best_{enc_name}_ckpt"), 1)]
    best_ckpt, resume_round = next(
        ((p, r) for p, r in resume_candidates
         if os.path.exists(os.path.join(p, "model.safetensors"))),
        (None, 0),
    )
    is_resume = best_ckpt is not None
    s2_lr     = cfg["stage2_resume_lr"]    if is_resume else cfg["stage2_lr"]
    s2_epochs = cfg["stage2_resume_epochs"] if is_resume else cfg["stage2_epochs"]

    if accelerator.is_main_process:
        print(f"\n{'='*45}")
        if is_resume:
            print(f"Stage 2 Resume (r{resume_round})  LR={s2_lr}  epochs={s2_epochs}")
            print(f"  Loading from: {best_ckpt}")
        else:
            print(f"Stage 2: LoRA Fine-tuning  LR={s2_lr}  epochs={s2_epochs}")
        print(f"{'='*45}\n")

    model = build_model(cfg, accelerator)
    model.apply_lora()

    # 가중치 로드
    if is_resume:
        from safetensors.torch import load_file
        state = load_file(os.path.join(best_ckpt, "model.safetensors"), device="cpu")
        model.load_state_dict(state, strict=False)
    else:
        if accelerator.is_main_process:
            print("Loading Stage 1 projector weights...")
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

    collate  = collate_fn_factory(model.tokenizer, cfg["max_text_len"])
    num_replicas = accelerator.num_processes
    rank         = accelerator.process_index
    cache_dir    = cfg["model_cache_dir"]

    spt  = cfg["samples_per_token"]
    mt   = cfg["max_text_len"]
    mbt  = cfg["max_batch_tokens"]
    train_lengths = get_dataset_lengths(train_dataset, spt, mt, cache_dir=cache_dir)
    val_lengths   = get_dataset_lengths(val_dataset,   spt, mt, cache_dir=cache_dir)

    train_sampler = DynamicBatchSampler(train_lengths, mbt, num_replicas=num_replicas, rank=rank)
    val_sampler   = DynamicBatchSampler(val_lengths,   mbt, num_replicas=num_replicas, rank=rank)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              collate_fn=collate, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_sampler=val_sampler,
                              collate_fn=collate, num_workers=2, pin_memory=True)

    model, optimizer = accelerator.prepare(model, optimizer)
    scheduler = make_scheduler(optimizer, train_loader, s2_epochs, cfg, accelerator)

    # Resume 시 baseline val_loss 설정 (진짜 개선 시에만 저장)
    best_val_loss = (
        run_validation(model, val_loader, accelerator) if is_resume else float("inf")
    )
    if is_resume and accelerator.is_main_process:
        print(f"  [Resume baseline] val_loss={best_val_loss:.4f}")

    prior_resume_epochs = (resume_round - 1) * cfg["stage2_resume_epochs"] if is_resume else 0
    epoch_offset = cfg["stage1_epochs"] + cfg["stage2_epochs"] + prior_resume_epochs
    stage_key    = 3 if is_resume else 2
    global_step  = step_offset

    for epoch in range(s2_epochs):
        train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        progress = tqdm(train_loader, desc=f"Stage{stage_key} Epoch {epoch+1}",
                        disable=not accelerator.is_main_process)
        accum_loss, accum_count, accum_bsz = 0.0, 0, 0

        for audio, audio_lengths, transcript_ids in progress:
            audio_lengths  = audio_lengths.to(accelerator.device)
            transcript_ids = transcript_ids.to(accelerator.device)
            with accelerator.accumulate(model):
                outputs = model(audio, audio_lengths=audio_lengths,
                                transcript_input_ids=transcript_ids)
                accelerator.backward(outputs.loss)
                if accelerator.is_main_process:
                    accum_bsz += audio.shape[0]
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    if accelerator.is_main_process:
                        global_step += 1
                        lr_now = scheduler.get_last_lr()[0]
                        accum_loss  += outputs.loss.item()
                        accum_count += 1
                        avg_loss = accum_loss / accum_count
                        wandb.log({"stage": stage_key, "train/loss": avg_loss,
                                   "train/lr": lr_now, "train/batch_size": accum_bsz},
                                  step=global_step)
                        progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr_now:.2e}",
                                             bsz=accum_bsz)
                        accum_bsz = 0

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
                ckpt_name(f"best_{enc_name}_ckpt", resume_round),
            )
            accelerator.save_state(best_dir)
            if accelerator.is_main_process:
                print(f"  Best checkpoint saved (val_loss={val_loss:.4f}) → {best_dir}")

    # 최종 checkpoint
    accelerator.wait_for_everyone()
    final_dir = os.path.join(
        cache_dir,
        ckpt_name(f"final_{enc_name}_ckpt", resume_round),
    )
    accelerator.save_state(final_dir)
    if accelerator.is_main_process:
        print(f"Final checkpoint saved → {final_dir}")


# ==========================================
# Entry point
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True,
                        choices=["encodec", "dac", "dac_vae", "mimi_acoustic", "mimi_semantic"],
                        help="사용할 audio encoder")
    parser.add_argument("--data-path",  default=None, help="데이터 루트 경로 (기본: config 값)")
    parser.add_argument("--cache-dir",  default=None, help="모델 캐시 경로 (기본: config 값)")
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    cfg = get_config(args.encoder)
    if args.data_path:  cfg["data_path"]       = args.data_path
    if args.cache_dir:  cfg["model_cache_dir"] = args.cache_dir
    if args.wandb_mode: cfg["wandb_mode"]      = args.wandb_mode

    os.makedirs(cfg["model_cache_dir"], exist_ok=True)
    os.environ.setdefault("HF_HOME",    cfg["model_cache_dir"])
    os.environ.setdefault("TORCH_HOME", os.path.join(os.path.dirname(cfg["model_cache_dir"]), "torch"))

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        mixed_precision="no",  # fp16 모델을 직접 로드하므로 "no"
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=7200))],
    )

    if accelerator.is_main_process:
        import datetime
        run_name = f"{args.encoder}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
        wandb.init(project=cfg["project_name"], config=cfg,
                   name=run_name, mode=cfg["wandb_mode"])

    # 데이터셋 다운로드 (rank 0만)
    if accelerator.is_main_process:
        for split in ["train-clean-100", "train-clean-360", "dev-clean"]:
            torchaudio.datasets.LIBRISPEECH(root=cfg["data_path"], url=split, download=True)
    accelerator.wait_for_everyone()

    train_dataset, val_dataset = build_datasets(cfg)

    proj_path, step_offset = run_stage1(cfg, accelerator, train_dataset, val_dataset)
    run_stage2(cfg, accelerator, train_dataset, val_dataset, proj_path, step_offset)

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
