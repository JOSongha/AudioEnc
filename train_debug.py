"""
Debug 학습 스크립트 — train-clean-100만 사용, 빠른 수렴 확인용.

사용법:
    torchrun --nproc_per_node=1 train_debug.py --encoder fb_dacvae
    torchrun --nproc_per_node=2 train_debug.py --encoder encodec --stage1-epochs 3
"""

import argparse
import csv
import gc
import os
import signal
import warnings
from datetime import timedelta

import torch
import torchaudio
import wandb
from accelerate import Accelerator, InitProcessGroupKwargs
import random
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from config import get_config
from dataset import (
    LibriSpeechDataset,
    collate_fn_factory,
    get_dataset_lengths,
    DynamicBatchSampler,
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


# ==========================================
# WER helpers
# ==========================================

def _edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def _compute_wer(hyps: list, refs: list) -> float:
    total_err, total_words = 0, 0
    for hyp, ref in zip(hyps, refs):
        ref_w = ref.lower().split()
        hyp_w = hyp.lower().split()
        total_words += max(len(ref_w), 1)
        total_err   += _edit_distance(ref_w, hyp_w)
    return total_err / max(total_words, 1)


@torch.no_grad()
def _greedy_batch(model, waveforms: list, sample_rate: int = 16000) -> list:
    """배치 greedy decode. waveforms: list of 1-D tensors."""
    device = next(model.parameters()).device
    B = len(waveforms)

    lengths = [w.shape[0] for w in waveforms]
    max_len = max(lengths)
    audio_padded = torch.zeros(B, max_len, device=device)
    for i, w in enumerate(waveforms):
        audio_padded[i, :lengths[i]] = w.to(device)
    audio_lengths = torch.tensor(lengths, device=device)

    audio_embeds, audio_mask = model._get_audio_embeds(audio_padded, audio_lengths)

    embed = model.llm.get_input_embeddings()
    p1_e  = embed(model.prompt_p1_ids).expand(B, -1, -1)
    p2_e  = embed(model.prompt_p2_ids).expand(B, -1, -1)
    inputs_embeds = torch.cat([p1_e, audio_embeds, p2_e], dim=1)

    p1_m = torch.ones(B, p1_e.shape[1], device=device, dtype=torch.long)
    p2_m = torch.ones(B, p2_e.shape[1], device=device, dtype=torch.long)
    attn = torch.cat([p1_m, audio_mask.long(), p2_m], dim=1)

    # 배치 내 최장 오디오 기준으로 max_new_tokens 동적 설정
    # 7 tokens/sec (≈ 4tok/sec × 1.75배 여유), floor=32, ceil=300
    max_new_tokens = max(32, min(300, int(max(lengths) / sample_rate * 7)))

    out = model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=model.tokenizer.eos_token_id,
        pad_token_id=model.tokenizer.eos_token_id,
        repetition_penalty=1.1,
        no_repeat_ngram_size=4,
    )
    return model.tokenizer.batch_decode(out, skip_special_tokens=True)


@torch.no_grad()
def evaluate_wer(raw_model, val_dataset, train_dataset, num_samples: int = 200,
                 print_samples: int = 10, max_eval_tokens: int = 2600,
                 samples_per_token: float = 853.3) -> tuple:
    """val/train 앞 num_samples개로 WER 계산. 학습과 동일한 동적 배칭 사용."""
    raw_model.eval()

    def run_split(dataset, label):
        n = min(num_samples, len(dataset))
        items = [dataset[i] for i in range(n)]  # (waveform, transcript)

        # 오디오 길이 기준 정렬 → greedy pack
        order = sorted(range(n), key=lambda i: items[i][0].shape[0])
        batches = []
        cur, cur_max = [], 0
        for orig_i in order:
            tok = max(1, int(items[orig_i][0].shape[0] / samples_per_token))
            new_max = max(cur_max, tok)
            if cur and (len(cur) + 1) * new_max > max_eval_tokens:
                batches.append(cur)
                cur, cur_max = [orig_i], tok
            else:
                cur.append(orig_i)
                cur_max = new_max
        if cur:
            batches.append(cur)

        hyps = [None] * n
        for batch_idxs in tqdm(batches, desc=f"WER [{label}]", unit="batch"):
            waveforms  = [items[i][0] for i in batch_idxs]
            batch_hyps = _greedy_batch(raw_model, waveforms)
            for orig_i, hyp in zip(batch_idxs, batch_hyps):
                hyps[orig_i] = hyp
        # 길이 분포 전체에서 균등 간격으로 출력
        if print_samples > 0:
            step = max(1, n // print_samples)
            for orig_i in range(0, n, step)[:print_samples]:
                print(f"  [{orig_i:3d}] REF: {items[orig_i][1]}")
                print(f"        HYP: {hyps[orig_i]}")

        refs = [items[i][1] for i in range(n)]
        return hyps, refs

    hypsT, refsT = run_split(train_dataset, "train")
    hyps,  refs  = run_split(val_dataset,  "valid")

    raw_model.train()
    return _compute_wer(hyps, refs), _compute_wer(hypsT, refsT)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! trainset으로도 wer 계산.

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
    # rank마다 n이 달라질 수 있으므로 all_reduce로 동기화
    # → val_loss < best_val_loss 분기가 모든 rank에서 동일하게 평가됨 (deadlock 방지)
    t = torch.tensor([total_loss, float(n)], device=accelerator.device)
    torch.distributed.all_reduce(t)
    return (t[0] / t[1].clamp(min=1)).item()


def build_model(cfg, accelerator):
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


def append_csv(csv_path: str, row: dict):
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ==========================================
# Stage 1: Projector Alignment
# ==========================================

def run_stage1(cfg, accelerator, train_dataset, val_dataset, train_eval_dataset, csv_path, debug=""):
    enc_name   = cfg["encoder_name"]
    debug_dir  = cfg["debug_dir"]
    ctc_tag    = "ctc" if "c" in debug else "nonctc"
    proj_path  = os.path.join(debug_dir, f"s1_proj_{enc_name}_{ctc_tag}.pt")
    s1_epochs  = cfg["stage1_epochs"]

    if os.path.exists(proj_path):
        if accelerator.is_main_process:
            print(f"\n[Stage 1 Skip] checkpoint found: {proj_path}")
        return proj_path, 0

    pid_path = os.path.join(debug_dir, "train.pid")
    if accelerator.is_main_process:
        signal.signal(signal.SIGUSR1, _handle_sigusr1)
        with open(pid_path, "w") as f:
            f.write(str(os.getpid()))
        print(f"\n{'='*45}")
        print(f"Stage 1 (debug)  LR={cfg['stage1_lr']}  epochs={s1_epochs}")
        print(f"{'='*45}\n")

    model    = build_model(cfg, accelerator)
    tokenizer = model.tokenizer
    collate  = collate_fn_factory(tokenizer, cfg["max_text_len"])

    num_replicas = accelerator.num_processes
    rank         = accelerator.process_index
    spt  = cfg["samples_per_token"]
    mt   = cfg["max_text_len"]
    mbt  = cfg["max_batch_tokens"]
    train_lengths = get_dataset_lengths(train_dataset, spt, mt, cache_dir=cfg["model_cache_dir"])
    val_lengths   = get_dataset_lengths(val_dataset,   spt, mt, cache_dir=cfg["model_cache_dir"])

    train_sampler = DynamicBatchSampler(train_lengths, mbt, num_replicas=num_replicas, rank=rank)
    val_sampler   = DynamicBatchSampler(val_lengths,   mbt, num_replicas=num_replicas, rank=rank)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              collate_fn=collate, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_sampler=val_sampler,
                              collate_fn=collate, num_workers=1, pin_memory=True)

    model.freeze_llm()
    if "c" in debug:
        model.init_ctc_head()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["stage1_lr"],
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    scheduler = make_scheduler(optimizer, train_loader, s1_epochs, cfg, accelerator)

    global_step = 0
    best_werT   = float("inf")

    for epoch in range(s1_epochs):
        train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        if "e" in debug:
            if "d" in debug:
                decay_over = cfg["eos_decay_epochs"] or s1_epochs
                t = min(epoch / max(decay_over - 1, 1), 1.0)
                eos_w = cfg["eos_weight"] * (1.0 - t) + 1.0 * t
            else:
                eos_w = cfg["eos_weight"]
        else:
            eos_w = 1.0
        progress = tqdm(train_loader, desc=f"S1 Epoch {epoch+1}/{s1_epochs}",
                        disable=not accelerator.is_main_process)
        accum_loss, accum_count, accum_bsz = 0.0, 0, 0

        for audio, audio_lengths, transcript_ids in progress:
            audio_lengths  = audio_lengths.to(accelerator.device)
            transcript_ids = transcript_ids.to(accelerator.device)
            with accelerator.accumulate(model):
                if "c" in debug:
                    texts = tokenizer.batch_decode(transcript_ids, skip_special_tokens=True)
                    ctc_tgt, ctc_tgt_len = _text_to_ctc_targets(texts)
                    outputs = model(audio, audio_lengths=audio_lengths,
                                    transcript_input_ids=transcript_ids,
                                    ctc_targets=ctc_tgt, ctc_target_lengths=ctc_tgt_len,
                                    eos_weight=eos_w)
                else:
                    outputs = model(audio, audio_lengths=audio_lengths,
                                    transcript_input_ids=transcript_ids,
                                    eos_weight=eos_w)
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
                    step_tensor = torch.tensor([global_step], device=accelerator.device)
                    torch.distributed.broadcast(step_tensor, src=0)
                    global_step = step_tensor[0].item()

        val_loss = run_validation(model, val_loader, accelerator)
        if accelerator.is_main_process:
            print(f"  [S1 Epoch {epoch+1}] val_loss={val_loss:.4f}")
            wandb.log({"stage": 1, "val/loss": val_loss, "epoch": epoch + 1,
                       "train/eos_weight": eos_w}, step=global_step)

        # WER 평가 (매 에폭) — 앞뒤 barrier로 non-main rank가 여기서 대기
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            raw_model = accelerator.unwrap_model(model)
            wer, werT = evaluate_wer(raw_model, val_dataset, train_eval_dataset,
                               num_samples=cfg.get("wer_samples", 200),
                               print_samples=cfg.get("print_samples", 10),
                               max_eval_tokens=cfg["max_batch_tokens"],
                               samples_per_token=cfg["samples_per_token"])
            print(f"  [S1 Epoch {epoch+1}] WER={wer*100:.1f}%, WER_T={werT*100:.1f}%")
            wandb.log({"stage": 1, "val/wer": wer, "train/wer": werT}, step=global_step)
            append_csv(csv_path, {
                "stage": 1, "epoch": epoch + 1, "step": global_step,
                "val_loss": round(val_loss, 4), "wer": round(wer, 4), "werT": round(werT, 4)
            })
            unwrapped  = accelerator.unwrap_model(model)
            proj_state = {k: v.cpu().half() for k, v in unwrapped.state_dict().items()
                          if "projector" in k or "proj_norm" in k}
            wer_ckpt_path = os.path.join(
                debug_dir, f"s1_proj_{enc_name}_{ctc_tag}_ep{epoch+1}_step{global_step}_wer{wer*100:.1f}.pt"
            )
            torch.save(proj_state, wer_ckpt_path)
            print(f"  [WER ckpt] Projector saved → {wer_ckpt_path}")
            if werT < best_werT:
                best_werT = werT
                torch.save(proj_state, proj_path)
                print(f"  [Best werT={werT*100:.1f}%] Projector saved → {proj_path}")
        accelerator.wait_for_everyone()


        stop_flag = torch.zeros(1, device=accelerator.device)
        if accelerator.is_main_process and _stop_stage1:
            stop_flag[0] = 1.0
            print(f"  [S1] SIGUSR1 — stopping after epoch {epoch+1}")
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
# Stage 2: LoRA Fine-tuning
# ==========================================

def run_stage2(cfg, accelerator, train_dataset, val_dataset, train_eval_dataset, proj_path, step_offset, csv_path, debug=""):
    enc_name  = cfg["encoder_name"]
    debug_dir = cfg["debug_dir"]
    ctc_tag   = "ctc" if "c" in debug else "nonctc"
    s2_epochs = cfg["stage2_epochs"]

    if accelerator.is_main_process:
        print(f"\n{'='*45}")
        print(f"Stage 2 (debug)  LR={cfg['stage2_lr']}  epochs={s2_epochs}")
        print(f"{'='*45}\n")

    model = build_model(cfg, accelerator)
    model.apply_lora()

    proj_state = torch.load(proj_path, map_location="cpu", weights_only=True)
    model.load_state_dict(proj_state, strict=False)

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
        lr=cfg["stage2_lr"], weight_decay=0.01,
    )

    collate      = collate_fn_factory(model.tokenizer, cfg["max_text_len"])
    num_replicas = accelerator.num_processes
    rank         = accelerator.process_index
    spt  = cfg["samples_per_token"]
    mt   = cfg["max_text_len"]
    mbt  = cfg["max_batch_tokens"]
    train_lengths = get_dataset_lengths(train_dataset, spt, mt, cache_dir=cfg["model_cache_dir"])
    val_lengths   = get_dataset_lengths(val_dataset,   spt, mt, cache_dir=cfg["model_cache_dir"])

    train_sampler = DynamicBatchSampler(train_lengths, mbt, num_replicas=num_replicas, rank=rank)
    val_sampler   = DynamicBatchSampler(val_lengths,   mbt, num_replicas=num_replicas, rank=rank)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              collate_fn=collate, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_sampler=val_sampler,
                              collate_fn=collate, num_workers=2, pin_memory=True)

    model, optimizer = accelerator.prepare(model, optimizer)
    scheduler = make_scheduler(optimizer, train_loader, s2_epochs, cfg, accelerator)

    best_werT   = float("inf")
    global_step = step_offset
    epoch_offset  = cfg["stage1_epochs"]

    for epoch in range(s2_epochs):
        train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        if "e" in debug:
            if "d" in debug:
                decay_over = cfg["eos_decay_epochs"] or s2_epochs
                t = min(epoch / max(decay_over - 1, 1), 1.0)
                eos_w = cfg["eos_weight"] * (1.0 - t) + 1.0 * t
            else:
                eos_w = cfg["eos_weight"]
        else:
            eos_w = 1.0
        progress = tqdm(train_loader, desc=f"S2 Epoch {epoch+1}/{s2_epochs}",
                        disable=not accelerator.is_main_process)
        accum_loss, accum_count, accum_bsz = 0.0, 0, 0

        for audio, audio_lengths, transcript_ids in progress:
            audio_lengths  = audio_lengths.to(accelerator.device)
            transcript_ids = transcript_ids.to(accelerator.device)
            with accelerator.accumulate(model):
                outputs = model(audio, audio_lengths=audio_lengths,
                                transcript_input_ids=transcript_ids,
                                eos_weight=eos_w)
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
                        wandb.log({"stage": 2, "train/loss": avg_loss,
                                   "train/lr": lr_now, "train/batch_size": accum_bsz},
                                  step=global_step)
                        progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr_now:.2e}",
                                             bsz=accum_bsz)
                        accum_bsz = 0
                    step_tensor = torch.tensor([global_step], device=accelerator.device)
                    torch.distributed.broadcast(step_tensor, src=0)
                    global_step = step_tensor[0].item()

        val_loss = run_validation(model, val_loader, accelerator)
        if accelerator.is_main_process:
            print(f"  [S2 Epoch {epoch+1}] val_loss={val_loss:.4f}")
            wandb.log({"stage": 2, "val/loss": val_loss,
                       "epoch": epoch + 1 + epoch_offset,
                       "train/eos_weight": eos_w}, step=global_step)

        # WER 평가 (매 에폭)
        accelerator.wait_for_everyone()
        wer_tensor = torch.tensor([0.0, 0.0], device=accelerator.device)  # [wer, werT]
        if accelerator.is_main_process:
            raw_model = accelerator.unwrap_model(model)
            wer, werT = evaluate_wer(raw_model, val_dataset, train_eval_dataset,
                               num_samples=cfg.get("wer_samples", 200),
                               print_samples=cfg.get("print_samples", 10),
                               max_eval_tokens=cfg["max_batch_tokens"],
                               samples_per_token=cfg["samples_per_token"])
            print(f"  [S2 Epoch {epoch+1}] WER={wer*100:.1f}%, WER_T={werT*100:.1f}%")
            wandb.log({"stage": 2, "val/wer": wer, "train/wer": werT}, step=global_step)
            append_csv(csv_path, {
                "stage": 2, "epoch": epoch + 1, "step": global_step,
                "val_loss": round(val_loss, 4), "wer": round(wer, 4), "werT": round(werT, 4),
            })
            wer_tensor[0], wer_tensor[1] = wer, werT
        torch.distributed.broadcast(wer_tensor, src=0)
        wer_ckpt_dir = os.path.join(
            debug_dir, f"s2_{enc_name}_{ctc_tag}_ep{epoch+1}_step{global_step}_wer{wer_tensor[0].item()*100:.1f}"
        )
        accelerator.save_state(wer_ckpt_dir)
        if accelerator.is_main_process:
            print(f"  [WER ckpt] Checkpoint saved → {wer_ckpt_dir}")
        if wer_tensor[1].item() < best_werT:
            best_werT = wer_tensor[1].item()
            best_wer_dir = os.path.join(debug_dir, f"s2_{enc_name}_{ctc_tag}_best_werT")
            accelerator.save_state(best_wer_dir)
            if accelerator.is_main_process:
                print(f"  [Best werT={best_werT*100:.1f}%] Checkpoint saved → {best_wer_dir}")
        accelerator.wait_for_everyone()


    accelerator.wait_for_everyone()
    final_dir = os.path.join(debug_dir, f"final_{enc_name}_ckpt_ep{s2_epochs}_step{global_step}")
    accelerator.save_state(final_dir)
    if accelerator.is_main_process:
        print(f"Final checkpoint → {final_dir}")


# ==========================================
# Entry point
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True,
                        choices=["encodec", "dac", "fb_dacvae", "mimi_acoustic", "mimi_semantic"],
                        help="사용할 audio encoder")
    parser.add_argument("--llm", default=None,
                        help="LLM 모델 이름 (예: Qwen/Qwen3.5-0.8B). 기본: config 값")
    parser.add_argument("--debug", default="",
                        help="디버그 옵션 (c: CTC loss 추가)")
    parser.add_argument("--stage1-epochs", type=int, default=65)
    parser.add_argument("--stage2-epochs", type=int, default=50)
    parser.add_argument("--debug-dir",  default=None,
                        help="체크포인트·CSV 저장 경로 (기본: model_cache_dir/debug_<encoder>)")
    parser.add_argument("--wandb-mode", default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wer-samples",   type=int, default=200,
                        help="에폭당 WER 평가 샘플 수")
    parser.add_argument("--print-samples", type=int, default=5,
                        help="에폭당 REF/HYP 출력 샘플 수")
    parser.add_argument("--div",           type=int, default=5,
                        help="데이터셋 서브셋 분모 (기본 5 → 1/5 사용)")
    parser.add_argument("--eos-weight",        type=float, default=3.0,
                        help="EOS 토큰 loss 가중치 (debug=e일 때 적용, 기본 3.0)")
    parser.add_argument("--eos-decay-epochs",  type=int,   default=None,
                        help="EOS weight를 1.0까지 decay할 에폭 수 (debug=ed일 때 적용, 기본: 전체 에폭)")
    args = parser.parse_args()

    cfg = get_config(args.encoder)
    if args.llm:
        cfg["llm_model"] = args.llm
    cfg["stage1_epochs"]  = args.stage1_epochs
    cfg["stage2_epochs"]  = args.stage2_epochs
    cfg["wandb_mode"]     = args.wandb_mode
    cfg["wer_samples"]    = args.wer_samples
    cfg["print_samples"]  = args.print_samples
    cfg["eos_weight"]       = args.eos_weight
    cfg["eos_decay_epochs"] = args.eos_decay_epochs

    # debug_dir: 체크포인트 저장 경로 (model_cache_dir와 분리)
    import datetime
    ctc_tag   = "ctc" if "c" in args.debug else "nonctc"
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    debug_dir = args.debug_dir or os.path.join(
        cfg["model_cache_dir"], f"debug_{args.encoder}_{ctc_tag}_{timestamp}"
    )
    cfg["debug_dir"] = debug_dir

    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(cfg["model_cache_dir"], exist_ok=True)
    os.environ.setdefault("HF_HOME",    cfg["model_cache_dir"])
    os.environ.setdefault("TORCH_HOME",
                          os.path.join(os.path.dirname(cfg["model_cache_dir"]), "torch"))

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        mixed_precision="no",
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3600))],
    )

    if accelerator.is_main_process:
        llm_tag  = "2b" if "2B" in cfg["llm_model"] else "4b"
        run_name = f"debug_{args.encoder}_{llm_tag}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
        wandb.init(project=cfg["project_name"], config=cfg,
                   name=run_name, mode=cfg["wandb_mode"])
        print(f"debug_dir : {debug_dir}")
        print(f"model_cache_dir: {cfg['model_cache_dir']}")

    # LibriSpeech train-clean-100 다운로드
    if accelerator.is_main_process:
        for split in ["train-clean-100", "dev-clean"]:
            torchaudio.datasets.LIBRISPEECH(root=cfg["data_path"], url=split, download=True)
    accelerator.wait_for_everyone()

    max_len = cfg["max_audio_len"]
    train_dataset_full = LibriSpeechDataset(root=cfg["data_path"], url="train-clean-100",
                                            max_len=max_len)
    val_dataset_full   = LibriSpeechDataset(root=cfg["data_path"], url="dev-clean",
                                            max_len=max_len)

    # 서브셋 (seed 고정, 재현 가능)
    div = args.div
    rng = random.Random(42)
    train_idx = rng.sample(range(len(train_dataset_full)), len(train_dataset_full) // div)
    val_idx   = rng.sample(range(len(val_dataset_full)),   len(val_dataset_full)   // div)
    train_dataset = Subset(train_dataset_full, sorted(train_idx))
    val_dataset   = Subset(val_dataset_full,   sorted(val_idx))
    # eval용 train 서브셋: 학습 서브셋의 1/5 (매 에폭 동일하므로 미리 생성)
    # train_eval_dataset = Subset(train_dataset, range(0, len(train_dataset), 5))
    train_eval_dataset = Subset(train_dataset, range(0, len(train_dataset), 4))

    if accelerator.is_main_process:
        print(f"Train: {len(train_dataset)} / Val: {len(val_dataset)} (1/{div} subset)")

    csv_path = os.path.join(debug_dir, f"results_{args.encoder}.csv")

    proj_path, step_offset = run_stage1(cfg, accelerator,
                                        train_dataset, val_dataset, train_eval_dataset, csv_path,
                                        debug=args.debug)
    run_stage2(cfg, accelerator, train_dataset, val_dataset, train_eval_dataset,
               proj_path, step_offset, csv_path, debug=args.debug)

    if accelerator.is_main_process:
        wandb.finish()
        print(f"\nResults saved → {csv_path}")


if __name__ == "__main__":
    main()
