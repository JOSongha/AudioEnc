"""
Stage 1 (Projector Alignment) 단독 학습 루프.

사용법:
    torchrun --nproc_per_node=8 train_stage1.py --encoder encodec
    torchrun --nproc_per_node=8 train_stage1.py --encoder dac
    torchrun --nproc_per_node=8 train_stage1.py --encoder fb_dacvae
    torchrun --nproc_per_node=8 train_stage1.py --encoder mimi_acoustic
    torchrun --nproc_per_node=8 train_stage1.py --encoder mimi_semantic
"""

import argparse
import gc
import json
import os
import signal
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

def _save_resume_state(accelerator, model, optimizer, scheduler,
                        cache_dir, enc_name,
                        epoch, global_step, best_val_loss, batches_to_skip=0):
    """accelerator.save_state()로 전체 학습 상태 저장 + 메타데이터 JSON 기록."""
    resume_dir = os.path.join(cache_dir, f"s1_resume_{enc_name}")
    meta_path  = os.path.join(cache_dir, f"s1_resume_{enc_name}_meta.json")
    accelerator.wait_for_everyone()
    accelerator.save_state(resume_dir)
    if accelerator.is_main_process:
        meta = {
            "epoch":           epoch,
            "global_step":     global_step,
            "best_val_loss":   best_val_loss,
            "batches_to_skip": batches_to_skip,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [Resume ckpt] ep={epoch} step={global_step} "
              f"skip={batches_to_skip} → {resume_dir}")


def run_stage1(cfg, accelerator, train_dataset, val_dataset, debug=""):
    enc_name      = cfg["encoder_name"]
    cache_dir     = cfg["model_cache_dir"]
    proj_path     = os.path.join(cache_dir, f"s1_proj_{enc_name}.pt")
    resume_dir    = os.path.join(cache_dir, f"s1_resume_{enc_name}")
    meta_path     = os.path.join(cache_dir, f"s1_resume_{enc_name}_meta.json")

    # Resume 메타데이터 로드
    has_resume = os.path.isdir(resume_dir) and os.path.exists(meta_path)
    if has_resume:
        with open(meta_path) as f:
            meta = json.load(f)
        start_epoch     = meta["epoch"]
        global_step     = meta["global_step"]
        best_val_loss   = meta["best_val_loss"]
        batches_to_skip = meta.get("batches_to_skip", 0)
        if accelerator.is_main_process:
            print(f"\n[Stage 1 Resume] ep={start_epoch} step={global_step} "
                  f"skip={batches_to_skip} ← {resume_dir}")
    else:
        start_epoch     = 0
        global_step     = 0
        best_val_loss   = float("inf")
        batches_to_skip = 0

    # Stage 1 완료 여부 확인 (resume 없을 때만)
    if not has_resume and os.path.exists(proj_path):
        if accelerator.is_main_process:
            print(f"\n[Stage 1 Skip] checkpoint found: {proj_path}")
        return proj_path, 0

    pid_path = os.path.join(cache_dir, "train.pid")
    if accelerator.is_main_process:
        signal.signal(signal.SIGUSR1, _handle_sigusr1)
        with open(pid_path, "w") as f:
            f.write(str(os.getpid()))
        print(f"\n{'='*45}")
        print(f"Stage 1: Projector Alignment  LR={cfg['stage1_lr']}")
        print(f"  Early stop: kill -USR1 $(cat {pid_path})")
        print(f"{'='*45}\n")

    model    = build_model(cfg, accelerator)
    tokenizer = model.tokenizer
    collate  = collate_fn_factory(tokenizer, cfg["max_text_len"])

    num_replicas = accelerator.num_processes
    rank         = accelerator.process_index
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
                              collate_fn=collate, num_workers=1, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_sampler=val_sampler,
                              collate_fn=collate, num_workers=1, pin_memory=True)

    model.freeze_llm()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["stage1_lr"],
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    scheduler = make_scheduler(optimizer, train_loader, cfg["stage1_epochs"], cfg, accelerator)

    # Resume: 모델·옵티마이저·스케줄러·RNG 복원
    if has_resume:
        accelerator.load_state(resume_dir)
        if accelerator.is_main_process:
            print(f"  Loaded full state from {resume_dir}")

    save_steps = cfg.get("save_steps", 0)

    for epoch in range(start_epoch, cfg["stage1_epochs"]):
        train_loader.batch_sampler.set_epoch(epoch)
        model.train()

        # mid-epoch resume: 이미 처리한 배치 건너뜀
        skip_batches = batches_to_skip if epoch == start_epoch else 0
        if skip_batches:
            active_loader = accelerator.skip_first_batches(train_loader, skip_batches)
            if accelerator.is_main_process:
                print(f"  Skipping {skip_batches} batches (mid-epoch resume)")
        else:
            active_loader = train_loader

        progress = tqdm(active_loader, desc=f"Stage1 Epoch {epoch+1}",
                        disable=not accelerator.is_main_process)
        accum_loss, accum_count, accum_bsz = 0.0, 0, 0
        steps_in_epoch = 0  # 이 epoch에서의 optimizer step 수

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
                    steps_in_epoch += 1
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
                    step_tensor = torch.tensor([global_step], device=accelerator.device)
                    torch.distributed.broadcast(step_tensor, src=0)
                    global_step = step_tensor[0].item()

                    if save_steps and global_step % save_steps == 0:
                        # 전체 resume 상태 저장 (mid-epoch)
                        # skip할 배치 수 = (이 epoch에서 처리한 step) × grad_accum + 원래 skip 수
                        done_batches = (steps_in_epoch * cfg["gradient_accumulation_steps"]
                                        + skip_batches)
                        _save_resume_state(
                            accelerator, model, optimizer, scheduler,
                            cache_dir, enc_name,
                            epoch=epoch, global_step=global_step,
                            best_val_loss=best_val_loss,
                            batches_to_skip=done_batches,
                        )
                        # projector-only 스냅샷 (Stage 2용)
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

        # epoch 완료 후 전체 resume 상태 저장 (batches_to_skip=0 → 다음 epoch 처음부터)
        _save_resume_state(
            accelerator, model, optimizer, scheduler,
            cache_dir, enc_name,
            epoch=epoch + 1, global_step=global_step,
            best_val_loss=min(best_val_loss, val_loss),
            batches_to_skip=0,
        )

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
# Entry point
# ==========================================

def _parse_datasets(s: str) -> list:
    valid = {"ls100", "ls360", "ls500", "mls", "gs"}
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
                        choices=["4b", "2b"],
                        help="LLM 크기: 4b=Qwen3.5-4B (기본), 2b=Qwen3.5-2B")
    parser.add_argument("--data-path",  default=None, help="데이터 루트 경로 (기본: config 값)")
    parser.add_argument("--cache-dir",  default=None, help="모델 캐시 경로 (기본: config 값)")
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--debug",      default="",   help="디버그 옵션 (w: step 50에 full weight 저장)")

    # 데이터셋 선택 및 샘플 수
    parser.add_argument("--datasets",
                        default=None, type=_parse_datasets,
                        metavar="ls100,ls360,ls500,mls,gs",
                        help="사용할 데이터셋 (쉼표 구분). 기본: ls100,ls360,ls500,mls")
    parser.add_argument("--ls-samples",     default=None, type=int,
                        metavar="N", help="LibriSpeech 서브샘플 수 (기본: 전체)")
    parser.add_argument("--mls-samples",    default=None, type=int,
                        metavar="N", help="MLS 샘플 수 (기본: 전체)")
    parser.add_argument("--gs-subset",      default="l",
                        choices=["xs", "s", "m", "l", "xl"],
                        help="GigaSpeech subset (기본: l=2500h)")
    parser.add_argument("--gs-samples",     default=None, type=int,
                        metavar="N", help="GigaSpeech 샘플 수 (기본: 전체)")
    args = parser.parse_args()

    cfg = get_config(args.encoder)
    if args.llm == "2b":    cfg["llm_model"] = "Qwen/Qwen3.5-2B"
    if args.data_path:      cfg["data_path"]       = args.data_path
    if args.cache_dir:      cfg["model_cache_dir"] = args.cache_dir
    if args.wandb_mode:     cfg["wandb_mode"]      = args.wandb_mode
    if args.datasets:       cfg["datasets"]        = args.datasets
    if args.ls_samples  is not None: cfg["librispeech_num_samples"] = args.ls_samples
    if args.mls_samples is not None: cfg["mls_num_samples"]         = args.mls_samples
    cfg["gs_subset"]     = args.gs_subset
    if args.gs_samples  is not None: cfg["gs_num_samples"]          = args.gs_samples

    os.makedirs(cfg["model_cache_dir"], exist_ok=True)
    os.environ.setdefault("HF_HOME",    cfg["model_cache_dir"])
    os.environ.setdefault("TORCH_HOME", os.path.join(os.path.dirname(cfg["model_cache_dir"]), "torch"))

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        mixed_precision="no",
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=7200))],
    )

    if accelerator.is_main_process:
        import datetime
        llm_tag  = "2b" if "2B" in cfg["llm_model"] else "4b"
        run_name = f"{args.encoder}_{llm_tag}_s1_{datetime.datetime.now().strftime('%m%d_%H%M')}"
        wandb.init(project=cfg["project_name"], config=cfg,
                   name=run_name, mode=cfg["wandb_mode"])

    # 데이터셋 다운로드 (rank 0만, 선택된 LibriSpeech split만)
    all_datasets = cfg.get("datasets", ["ls100", "ls360", "ls500", "mls"])
    _ls_url_map  = {"ls100": "train-clean-100", "ls360": "train-clean-360", "ls500": "train-other-500"}
    need_splits  = {"dev-clean"}
    for name in all_datasets:
        if name in _ls_url_map:
            need_splits.add(_ls_url_map[name])
    if accelerator.is_main_process:
        for split in sorted(need_splits):
            torchaudio.datasets.LIBRISPEECH(root=cfg["data_path"], url=split, download=True)
    accelerator.wait_for_everyone()

    train_dataset, val_dataset = build_datasets(cfg)

    run_stage1(cfg, accelerator, train_dataset, val_dataset, debug=args.debug)

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
