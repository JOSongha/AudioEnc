"""
Stage 1 (Projector Alignment) — EOS-first collate 버전.

train_stage1.py와 동일하지만 collate_fn의 패딩 순서를 변경:
  기존: [tok1, tok2, tok3, PAD, PAD, EOS]  ← tokenize(padding) → append EOS
  변경: [tok1, tok2, tok3, EOS, PAD, PAD]  ← tokenize → append EOS → pad

변경 사항 두 곳:
  1. collate_fn_factory_eos_first  (이 파일 내 재정의)
  2. AudioQwenEosFirst.forward     (label 마스킹 단순화)

사용법:
    torchrun --nproc_per_node=8 train_stage1_eos_first.py --encoder fb_dacvae
"""

import argparse
import gc
import json
import os
import signal
import warnings
from datetime import timedelta

import torch
import torch.nn as nn
import torchaudio
import wandb
from accelerate import Accelerator, InitProcessGroupKwargs
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from config import get_config
from dataset import build_datasets, get_dataset_lengths, DynamicBatchSampler
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
# [변경 1] EOS-first collate_fn
#
# 기존: tokenize(padding=True, max_length=L) → append EOS
#       → [tok1, tok2, PAD, PAD, EOS]   attention_mask: [1,1,0,0,1]
#
# 변경: tokenize(padding=False, max_length=L-1) → append EOS → pad
#       → [tok1, tok2, EOS, PAD, PAD]   attention_mask: [1,1,1,0,0]
# ==========================================

def collate_fn_factory_eos_first(tokenizer, max_text_len: int = 256):
    """
    batch: [(waveform, transcript), ...]
    반환: (audios_padded, audio_lengths, input_ids)
      - input_ids: [tok1, ..., EOS, PAD, PAD]  — EOS 먼저, PAD 뒤
    """
    def collate_fn(batch):
        audios = [item[0] for item in batch]
        texts  = [item[1] for item in batch]

        audio_lengths = torch.tensor([a.shape[0] for a in audios], dtype=torch.long)
        audios_padded = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)

        # EOS 자리 확보를 위해 max_text_len-1로 truncate, 패딩은 나중에
        text_inputs = tokenizer(
            texts,
            return_tensors=None,
            padding=False,
            truncation=True,
            max_length=max_text_len - 1,
            add_special_tokens=False,
        )
        # 각 샘플: EOS append 후 텐서 변환
        sequences = []
        for ids in text_inputs["input_ids"]:
            seq = torch.tensor(ids + [tokenizer.eos_token_id], dtype=torch.long)
            sequences.append(seq)

        # 배치 내 최대 길이로 right-padding → [tok1, ..., EOS, PAD, PAD]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        return audios_padded, audio_lengths, input_ids

    return collate_fn


# ==========================================
# [변경 2] AudioQwenEosFirst
#
# 기존 label 마스킹:
#   tgt_labels[:, :-1][tgt_labels[:, :-1] == pad_id] = -100
#   → 마지막 열(항상 EOS)을 보존하기 위한 [:, :-1] trick
#
# 변경:
#   tgt_labels[tgt_labels == pad_id] = -100
#   → EOS(248046) ≠ PAD(248044)이므로 직접 PAD만 마스킹
#   → EOS는 시퀀스 중간에 있어도 자동 보존
# ==========================================

class AudioQwenEosFirst(AudioQwen):
    """EOS-first collate에 맞춘 label 마스킹 버전."""

    def forward(self, audio, audio_lengths=None, transcript_input_ids=None):
        audio_embeds, audio_mask = self._get_audio_embeds(audio, audio_lengths)

        if transcript_input_ids is None:
            return audio_embeds, audio_mask

        device = audio_embeds.device
        B      = audio_embeds.shape[0]
        embed  = self.llm.get_input_embeddings()

        p1_embeds         = embed(self.prompt_p1_ids).expand(B, -1, -1)
        p2_embeds         = embed(self.prompt_p2_ids).expand(B, -1, -1)
        transcript_embeds = embed(transcript_input_ids)

        inputs_embeds = torch.cat([p1_embeds, audio_embeds, p2_embeds, transcript_embeds], dim=1)

        p1_mask         = torch.ones(B, p1_embeds.shape[1], device=device, dtype=torch.long)
        p2_mask         = torch.ones(B, p2_embeds.shape[1], device=device, dtype=torch.long)
        transcript_mask = (transcript_input_ids != self.tokenizer.pad_token_id).long()
        attention_mask  = torch.cat([p1_mask, audio_mask.long(), p2_mask, transcript_mask], dim=1)

        len_ctx    = p1_embeds.shape[1] + audio_embeds.shape[1] + p2_embeds.shape[1]
        ctx_labels = torch.full((B, len_ctx), -100, dtype=torch.long, device=device)
        tgt_labels = transcript_input_ids.clone()

        # [변경] PAD(248044) 위치만 -100으로 마스킹.
        # EOS(248046) ≠ PAD이므로 EOS는 위치에 무관하게 자동 보존됨.
        tgt_labels[tgt_labels == self.tokenizer.pad_token_id] = -100

        labels = torch.cat([ctx_labels, tgt_labels], dim=1)

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )


# ==========================================
# Utilities (train_stage1.py와 동일)
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
        model   = AudioQwenEosFirst(encoder, cfg)
    accelerator.wait_for_everyone()
    if local_rank != 0:
        encoder = build_encoder(enc_name, enc_cfg, cache_dir)
        model   = AudioQwenEosFirst(encoder, cfg)
    accelerator.wait_for_everyone()
    return model


# ==========================================
# Stage 1: Projector Alignment
# ==========================================

def _save_resume_state(accelerator, model, optimizer, scheduler,
                        cache_dir, enc_name,
                        epoch, global_step, best_val_loss, batches_to_skip=0):
    """accelerator.save_state()로 전체 학습 상태 저장 + 메타데이터 JSON 기록."""
    resume_dir = os.path.join(cache_dir, f"s1_resume_{enc_name}_eosfirst")
    meta_path  = os.path.join(cache_dir, f"s1_resume_{enc_name}_eosfirst_meta.json")
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
    enc_name  = cfg["encoder_name"]
    cache_dir = cfg["model_cache_dir"]
    proj_path = os.path.join(cache_dir, f"s1_proj_{enc_name}_eosfirst.pt")
    resume_dir = os.path.join(cache_dir, f"s1_resume_{enc_name}_eosfirst")
    meta_path  = os.path.join(cache_dir, f"s1_resume_{enc_name}_eosfirst_meta.json")

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

    if not has_resume and os.path.exists(proj_path):
        if accelerator.is_main_process:
            print(f"\n[Stage 1 Skip] checkpoint found: {proj_path}")
        return proj_path, 0

    pid_path = os.path.join(cache_dir, "train_eosfirst.pid")
    if accelerator.is_main_process:
        signal.signal(signal.SIGUSR1, _handle_sigusr1)
        with open(pid_path, "w") as f:
            f.write(str(os.getpid()))
        print(f"\n{'='*45}")
        print(f"Stage 1 (EOS-first): Projector Alignment  LR={cfg['stage1_lr']}")
        print(f"  Early stop: kill -USR1 $(cat {pid_path})")
        print(f"{'='*45}\n")

    model     = build_model(cfg, accelerator)
    tokenizer = model.tokenizer
    # [변경 1 적용] EOS-first collate 사용
    collate   = collate_fn_factory_eos_first(tokenizer, cfg["max_text_len"])

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

    if has_resume:
        accelerator.load_state(resume_dir)
        if accelerator.is_main_process:
            print(f"  Loaded full state from {resume_dir}")

    save_steps = cfg.get("save_steps", 0)

    for epoch in range(start_epoch, cfg["stage1_epochs"]):
        train_loader.batch_sampler.set_epoch(epoch)
        model.train()

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
        steps_in_epoch = 0

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
                        done_batches = (steps_in_epoch * cfg["gradient_accumulation_steps"]
                                        + skip_batches)
                        _save_resume_state(
                            accelerator, model, optimizer, scheduler,
                            cache_dir, enc_name,
                            epoch=epoch, global_step=global_step,
                            best_val_loss=best_val_loss,
                            batches_to_skip=done_batches,
                        )
                        if accelerator.is_main_process:
                            unwrapped  = accelerator.unwrap_model(model)
                            proj_state = {k: v.cpu().half() for k, v in unwrapped.state_dict().items()
                                          if "projector" in k or "proj_norm" in k}
                            step_proj_path = os.path.join(
                                cache_dir, f"s1_proj_{enc_name}_eosfirst_ep{epoch+1}_step{global_step}.pt"
                            )
                            torch.save(proj_state, step_proj_path)
                            print(f"  [Step {global_step}] Projector saved → {step_proj_path}")

        val_loss = run_validation(model, val_loader, accelerator)
        if accelerator.is_main_process:
            print(f"  [Stage1 Epoch {epoch+1}] val_loss={val_loss:.4f}")
            wandb.log({"stage": 1, "val/loss": val_loss, "epoch": epoch + 1},
                      step=global_step)

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
                    cache_dir, f"s1_proj_{enc_name}_eosfirst_ep{epoch+1}_step{global_step}_best.pt"
                )
                torch.save(proj_state, named_proj_path)
                torch.save(proj_state, proj_path)
                print(f"  Projector saved (ep={epoch+1}, step={global_step}, val_loss={val_loss:.4f}) → {named_proj_path}")

        stop_flag = torch.zeros(1, device=accelerator.device)
        if accelerator.is_main_process and _stop_stage1:
            stop_flag[0] = 1.0
            print(f"  [Stage 1] SIGUSR1 received — stopping early after epoch {epoch+1}")
        torch.distributed.broadcast(stop_flag, src=0)
        if stop_flag[0].item() == 1.0:
            break

    if accelerator.is_main_process and os.path.exists(pid_path):
        os.remove(pid_path)

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
    parser.add_argument("--llm",        default=None, choices=["4b", "2b"])
    parser.add_argument("--data-path",  default=None)
    parser.add_argument("--cache-dir",  default=None)
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--debug",      default="")
    parser.add_argument("--datasets",   default=None, type=_parse_datasets,
                        metavar="ls100,ls360,ls500,mls,gs")
    parser.add_argument("--ls-samples",  default=None, type=int)
    parser.add_argument("--mls-samples", default=None, type=int)
    parser.add_argument("--gs-subset",   default="xl", choices=["xs", "s", "m", "l", "xl"])
    parser.add_argument("--gs-samples",  default=None, type=int)
    args = parser.parse_args()

    cfg = get_config(args.encoder)
    if args.llm == "2b":    cfg["llm_model"] = "Qwen/Qwen3.5-2B"
    if args.data_path:      cfg["data_path"]       = args.data_path
    if args.cache_dir:      cfg["model_cache_dir"] = args.cache_dir
    if args.wandb_mode:     cfg["wandb_mode"]      = args.wandb_mode
    if args.datasets:       cfg["datasets"]        = args.datasets
    if args.ls_samples  is not None: cfg["librispeech_num_samples"] = args.ls_samples
    if args.mls_samples is not None: cfg["mls_num_samples"]         = args.mls_samples
    cfg["gs_subset"] = args.gs_subset
    if args.gs_samples is not None:  cfg["gs_num_samples"] = args.gs_samples

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
        run_name = f"{args.encoder}_{llm_tag}_s1_eosfirst_{datetime.datetime.now().strftime('%m%d_%H%M')}"
        wandb.init(project=cfg["project_name"], config=cfg,
                   name=run_name, mode=cfg["wandb_mode"])

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
