"""
§43: legacy train.py (arXiv/scripts/train.py) 의 convergent training dynamics 를
유지하면서 precomputed encoder features 를 사용해 가속한 버전.

핵심:
  - 기존 legacy 의 **non-packed per-sample forward** 유지 (block-diag packing 회피)
  - Encoder 호출 skip (precomputed arrow 로드) → 1.5-2x 속도
  - DynamicBatchSampler: 총 LLM token ≤ max_batch_tokens 기반 배치
  - AdamW + cosine + warmup_ratio 0.1 (legacy 동일)
  - Liger kernel (optional)
  - bf16

사용법:
    torchrun --nproc_per_node=8 train_legacy_precomp.py --encoder fb_dacvae \\
        --precomputed-dir /mnt/fr20tb/wbl_residency/jos/AudioEnc/log/tmp/precomputed
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
import torch.nn as nn
import wandb
from accelerate import Accelerator, InitProcessGroupKwargs
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from config import get_config  # noqa: E402
from encoders import build_encoder  # noqa: E402
from model import AudioQwen  # noqa: E402


# ══════════════════════════════════════════════════════════
# Precomputed Arrow Dataset (per-sample, memory-mapped)
# ══════════════════════════════════════════════════════════
class PrecomputedArrowDataset(Dataset):
    """Rank 별 per-sample arrow file 을 memory-map 으로 로드 (RAM 절약).

    각 __getitem__ → (features (T_enc, C) float32, feat_len int, text str)
    전체 데이터 소비 시에도 disk I/O 만 발생, RAM 에 올리지 않음.
    """

    def __init__(self, arrow_paths: list[str], out_dim: int | None = None,
                 max_text_len: int = 256):
        self.arrow_paths = arrow_paths
        self.max_text_len = max_text_len
        self._readers = []
        self._index: list[tuple[int, int, int]] = []
        self._feat_lens: list[int] = []
        detected_dim = None
        for fi, path in enumerate(arrow_paths):
            mm = pa.memory_map(path, "r")
            rdr = ipc.open_file(mm)
            self._readers.append(rdr)
            for bi in range(rdr.num_record_batches):
                batch = rdr.get_batch(bi)
                feat_lens_col = batch.column("feat_len").to_pylist()
                feats_col = batch.column("features")
                for ri, fl in enumerate(feat_lens_col):
                    self._index.append((fi, bi, ri))
                    self._feat_lens.append(int(fl))
                    if detected_dim is None and fl > 0:
                        # Detect out_dim from flat feature length / feat_len
                        flat_len = len(feats_col[ri])
                        detected_dim = flat_len // int(fl)
        if detected_dim is None:
            raise RuntimeError(f"Could not detect out_dim from arrow files")
        self.out_dim = detected_dim
        if out_dim is not None and out_dim != detected_dim:
            print(f"[Dataset] out_dim mismatch: config={out_dim}, detected={detected_dim}. Using detected={detected_dim}")

    def __len__(self):
        return len(self._index)

    @property
    def feat_lens(self) -> list[int]:
        return self._feat_lens

    def __getitem__(self, idx: int):
        fi, bi, ri = self._index[idx]
        batch = self._readers[fi].get_batch(bi)
        text = batch.column("text")[ri].as_py()
        feat_len = int(batch.column("feat_len")[ri].as_py())
        # features: list<float> flat (feat_len × out_dim)
        flat = batch.column("features")[ri].as_py()
        feats = np.asarray(flat, dtype=np.float32).reshape(feat_len, self.out_dim)
        return feats, feat_len, text


# ══════════════════════════════════════════════════════════
# DynamicBatchSampler — total LLM tokens ≤ max_batch_tokens
# ══════════════════════════════════════════════════════════
class DynamicBatchSampler(Sampler):
    """legacy DynamicBatchSampler 과 동일. feat_len 기반으로 bucket packing.

    실제 LLM token = ceil(feat_len / proj_stride) + max_text_len + prompt_len.
    근사치로 ceil(feat_len / stride) 만 가지고 group — legacy 공식과 동일.
    """

    def __init__(self, feat_lens: list[int], max_batch_tokens: int,
                 proj_stride: int, max_text_len: int,
                 num_replicas: int, rank: int, shuffle: bool = True, seed: int = 42):
        self.feat_lens = feat_lens
        self.mbt = max_batch_tokens
        self.proj_stride = proj_stride
        self.mt = max_text_len
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        # Estimated LLM tokens per sample = audio tokens + text + ~6 prompt
        # audio LLM tokens = ceil(feat_len / proj_stride)  — encoder frame → LLM token
        self._est_tokens = [
            int(math.ceil(fl / self.proj_stride)) + max_text_len + 6
            for fl in feat_lens
        ]

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        # per-rank dataset 이미 sharded (rank{N}_s*.arrow) 이므로 여기선 rank 별 slicing X.
        n = len(self.feat_lens)
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch + self.rank)  # rank 별 다른 shuffle
        indices = torch.randperm(n, generator=g).tolist() if self.shuffle else list(range(n))
        batch, batch_tokens = [], 0
        for idx in indices:
            tok = self._est_tokens[idx]
            if batch and batch_tokens + tok > self.mbt:
                yield batch
                batch, batch_tokens = [], 0
            batch.append(idx)
            batch_tokens += tok
        if batch:
            yield batch

    def __len__(self):
        n = len(self.feat_lens)
        avg_tok = sum(self._est_tokens) / max(1, len(self._est_tokens))
        return max(1, int(n / max(1, self.mbt // max(1, int(avg_tok)))))


# ══════════════════════════════════════════════════════════
# Collator
# ══════════════════════════════════════════════════════════
def make_collator(tokenizer, out_dim: int, max_text_len: int = 256):
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    def collate(batch):
        # batch: list of (feats (T, C), feat_len, text)
        Ts = [b[1] for b in batch]
        T_max = max(Ts)
        B = len(batch)
        feats_t = torch.zeros(B, T_max, out_dim, dtype=torch.float32)
        feat_lens = torch.zeros(B, dtype=torch.long)
        texts = []
        for i, (f, tl, text) in enumerate(batch):
            feats_t[i, :tl, :] = torch.from_numpy(f)
            feat_lens[i] = tl
            texts.append(text)
        # tokenize text — legacy format: lower + strip + EOS 추가
        enc = tokenizer(
            [t.lower().strip() for t in texts],
            return_tensors="pt", padding=True,
            truncation=True, max_length=max_text_len,
            add_special_tokens=False,
        )
        ids = enc["input_ids"]
        # append EOS to each sequence (non-pad 끝에)
        B, T = ids.shape
        new_ids = torch.full((B, T + 1), pad_id, dtype=torch.long)
        for i in range(B):
            nz = (ids[i] != pad_id).nonzero(as_tuple=False)
            n_valid = nz.shape[0] if nz.numel() else 0
            new_ids[i, :n_valid] = ids[i, :n_valid]
            new_ids[i, n_valid] = eos_id
        return feats_t, feat_lens, new_ids

    return collate


# ══════════════════════════════════════════════════════════
# Accelerator + model build
# ══════════════════════════════════════════════════════════
def build_accelerator():
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    acc = Accelerator(kwargs_handlers=[kwargs])
    # DynamicBatchSampler has variable batch size → even_batches=False 필요
    acc.even_batches = False
    return acc


def build_model(cfg, accelerator):
    cache_dir = cfg["model_cache_dir"]
    enc_name = cfg["encoder_name"]
    enc_cfg = cfg["encoder"]
    with accelerator.main_process_first():
        encoder = build_encoder(enc_name, enc_cfg, cache_dir)
        model = AudioQwen(encoder, cfg)
    return model


# ══════════════════════════════════════════════════════════
# Dataset loader
# ══════════════════════════════════════════════════════════
def load_precomputed_datasets(precomputed_dir: str, encoder_name: str,
                               datasets: list[str], rank: int, out_dim: int,
                               accelerator=None) -> PrecomputedArrowDataset:
    """Per-rank 의 모든 dataset arrow 파일을 합쳐 단일 Dataset 반환."""
    base = Path(precomputed_dir) / encoder_name
    paths = []
    for ds_key in datasets:
        ds_dir = base / ds_key
        if not ds_dir.exists():
            if accelerator is None or accelerator.is_main_process:
                print(f"[warn] {ds_dir} not found, skip")
            continue
        # Prefer sharded if exists
        shards = sorted(ds_dir.glob(f"rank{rank}_s*.arrow"))
        if shards:
            paths.extend(str(p) for p in shards)
        else:
            single = ds_dir / f"rank{rank}.arrow"
            if single.exists():
                paths.append(str(single))
    if not paths:
        raise RuntimeError(f"No precomputed arrow files found for rank {rank}")
    if accelerator is None or accelerator.is_main_process:
        print(f"[rank {rank}] Loading {len(paths)} arrow files:")
        for p in paths[:3]:
            print(f"  - {Path(p).name}")
        if len(paths) > 3:
            print(f"  ... ({len(paths) - 3} more)")
    return PrecomputedArrowDataset(paths, out_dim=out_dim)


# ══════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════
_stop_stage1 = False


def _handle_sigusr1(signum, frame):
    global _stop_stage1
    _stop_stage1 = True


def run_stage1(cfg, accelerator, model, train_dataset, run_id: str):
    tokenizer = model.tokenizer
    enc_name = cfg["encoder_name"]
    out_dim = cfg["encoder"]["out_dim"]

    pid_path = os.path.join(cfg["model_cache_dir"], "train.pid")
    if accelerator.is_main_process:
        signal.signal(signal.SIGUSR1, _handle_sigusr1)
        with open(pid_path, "w") as f:
            f.write(str(os.getpid()))
        print(f"\n{'='*55}")
        print(f" Stage 1: Projector Alignment  LR={cfg['stage1_lr']}")
        print(f" Early stop: kill -USR1 $(cat {pid_path})")
        print(f" Dataset size: {len(train_dataset)}")
        print(f"{'='*55}\n")

    model.freeze_llm(projector_fp32=False)  # keep bf16 uniform
    model.to(dtype=torch.bfloat16)           # projector + proj_norm + encoder bf16
    # Re-enable gradient on projector / proj_norm after model-wide dtype cast
    for p in model.projector.parameters():
        p.requires_grad = True
    for p in model.proj_norm.parameters():
        p.requires_grad = True

    # Sampler + Loader
    # DDP 안정성: DynamicBatchSampler 대신 고정 batch size + DistributedSampler.
    # per-rank arrow 가 이미 sharded 이므로 각 rank 는 자체 dataset 만 봄.
    # DistributedSampler 의 num_replicas=1, rank=0 으로 설정 → 이중 샤딩 방지.
    fixed_bs = cfg.get("stage1_batch_size", 8)
    sampler = DistributedSampler(
        train_dataset, num_replicas=1, rank=0, shuffle=True,
        seed=cfg.get("seed", 42), drop_last=True,
    )
    collator = make_collator(tokenizer, train_dataset.out_dim, cfg["max_text_len"])
    train_loader = DataLoader(
        train_dataset, batch_size=fixed_bs, sampler=sampler,
        collate_fn=collator,
        num_workers=0,           # §43 debug: worker 없이 inline 데이터 로딩
        pin_memory=False,
        drop_last=True,
    )
    # 모든 rank 가 같은 batch 수를 돌도록 최솟값으로 맞춤
    local_n_batches = len(train_loader)
    if accelerator.num_processes > 1:
        t = torch.tensor([local_n_batches], device=accelerator.device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
        min_n_batches = int(t.item())
    else:
        min_n_batches = local_n_batches
    if accelerator.is_main_process:
        print(f"  local_n_batches={local_n_batches}  min_n_batches={min_n_batches}  bs={fixed_bs}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["stage1_lr"],
    )
    # DynamicBatchSampler 가 이미 per-rank slicing 을 수행하므로 DataLoader 는 prepare 하지 않음
    # (accelerator 가 BatchSamplerShard 로 재샤딩 시도하면 variable batch size 오류 발생)
    model, optimizer = accelerator.prepare(model, optimizer)

    # cosine + warmup — 모든 rank 가 min_n_batches 로 동기화
    total_steps = min_n_batches * cfg["stage1_epochs"]
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if accelerator.is_main_process:
        print(f" total_steps={total_steps}  warmup_steps={warmup_steps}")

    global_step = 0
    log_every = cfg.get("log_every", 1)
    save_steps = cfg.get("save_steps", 0)
    output_dir = os.path.join(cfg["model_cache_dir"], enc_name, f"s1_legacy_{run_id}")
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(cfg["stage1_epochs"]):
        sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(train_loader, desc=f"Stage1 Epoch {epoch+1}",
                    disable=not accelerator.is_main_process, total=min_n_batches)
        batch_counter = 0
        for batch in pbar:
            if batch_counter >= min_n_batches:
                break
            batch_counter += 1
            feats_t, feat_lens, transcript_ids = batch
            feats_t = feats_t.to(accelerator.device)
            feat_lens = feat_lens.to(accelerator.device)
            transcript_ids = transcript_ids.to(accelerator.device)

            # Legacy forward with precomputed features
            outputs = model(
                precomputed_enc_feats=feats_t,
                audio_lengths=feat_lens,
                transcript_input_ids=transcript_ids,
            )
            loss = outputs.loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 1.0))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            if accelerator.is_main_process:
                lr = scheduler.get_last_lr()[0]
                loss_v = loss.detach().item()
                if global_step % log_every == 0:
                    wandb.log({
                        "stage": 1, "train/loss": loss_v, "train/lr": lr,
                        "train/step": global_step,
                        "train/batch": feats_t.shape[0],
                    }, step=global_step)
                    pbar.set_postfix(loss=f"{loss_v:.3f}", lr=f"{lr:.2e}",
                                     bsz=feats_t.shape[0])
                if save_steps and global_step % save_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(model)
                        proj = {k: v.cpu() for k, v in unwrapped.state_dict().items()
                                if "projector" in k or "proj_norm" in k}
                        torch.save(proj, os.path.join(output_dir, f"s1_proj_step{global_step}.pt"))

            if _stop_stage1:
                break
        if _stop_stage1:
            break

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        proj = {k: v.cpu() for k, v in unwrapped.state_dict().items()
                if "projector" in k or "proj_norm" in k}
        final_path = os.path.join(output_dir, "s1_proj.pt")
        torch.save(proj, final_path)
        print(f"[Stage 1] final projector saved: {final_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--precomputed-dir", required=True)
    parser.add_argument("--datasets", default="ls100,ls360,ls500,mls,gs,vp")
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage1-lr", type=float, default=None)
    parser.add_argument("--max-batch-tokens", type=int, default=None)
    parser.add_argument("--liger", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    cfg = get_config(args.encoder)
    cfg["encoder_name"] = args.encoder
    if args.stage1_epochs is not None:
        cfg["stage1_epochs"] = args.stage1_epochs
    if args.stage1_lr is not None:
        cfg["stage1_lr"] = args.stage1_lr
    if args.max_batch_tokens is not None:
        cfg["max_batch_tokens"] = args.max_batch_tokens
    cfg["use_liger_kernel"] = args.liger
    cfg["attn_implementation"] = cfg.get("attn_implementation", "flash_attention_2")

    accelerator = build_accelerator()
    run_id = datetime.now().strftime("%m%d_%H%M")

    if accelerator.is_main_process:
        wandb.init(
            project=f"Qwen3.5-2b-ASR-{args.encoder}",
            name=f"{args.encoder}_legacy_precomp_{run_id}",
            config=dict(
                stage1_lr=cfg["stage1_lr"],
                stage1_epochs=cfg["stage1_epochs"],
                max_batch_tokens=cfg["max_batch_tokens"],
                datasets=args.datasets,
                use_liger_kernel=cfg["use_liger_kernel"],
            ),
        )

    # Model build
    model = build_model(cfg, accelerator)

    # Dataset build (per-rank)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    train_dataset = load_precomputed_datasets(
        args.precomputed_dir, args.encoder, datasets,
        rank=accelerator.process_index, out_dim=cfg["encoder"]["out_dim"],
        accelerator=accelerator,
    )

    run_stage1(cfg, accelerator, model, train_dataset, run_id)

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
