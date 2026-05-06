"""
2-layer (configurable) Transformer probe over frame-level encoder embeddings.
GPU-optimized: pre-load entire dataset to GPU once, no DataLoader, autocast bf16.

Output: _results/probe_transformer_results.csv
  encoder, dataset, fold, num_layers, val_acc, test_acc
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).parent
EMB_FRAMES_ROOT = ROOT / "embeds_frames"
MANIFEST_DIR = ROOT / "manifests"
RESULTS_CSV = ROOT / "_results/probe_transformer_results.csv"

LABEL_COL = {"iemocap_4class": "emotion", "ravdess": "emotion",
             "cremad": "emotion", "esc50": "label",
             "gtzan": "label", "nsynth_train_30k": "label", "nsynth_test": "label"}
FOLD_COL = {"iemocap_4class": "session", "ravdess": "fold",
            "cremad": "fold", "esc50": "fold",
            "gtzan": "fold", "nsynth_train_30k": "fold", "nsynth_test": "fold"}


def preload_to_gpu(emb_dir, df, device):
    """
    Load all .npz frames into GPU tensor (N, T_max, D) bf16 + mask (N, T_max) bool + labels (N,).
    """
    rows = df.to_dict("records")
    paths = [emb_dir / f"{r['utt_id']}.npz" for r in rows]
    # First pass: get T_max + D
    sample = np.load(str(paths[0]))
    D = sample["frames"].shape[-1]
    T_max = 0
    for p in paths:
        d = np.load(str(p))
        T_max = max(T_max, d["frames"].shape[0])

    N = len(rows)
    logger.info(f"  Preloading N={N} T_max={T_max} D={D} to GPU as bf16")
    X = torch.zeros((N, T_max, D), dtype=torch.bfloat16, device=device)
    mask = torch.zeros((N, T_max), dtype=torch.bool, device=device)
    y = torch.tensor([r["label_idx"] for r in rows], dtype=torch.long, device=device)

    for i, p in enumerate(paths):
        d = np.load(str(p))
        f = torch.from_numpy(d["frames"]).to(torch.bfloat16)
        t = f.shape[0]
        X[i, :t] = f.to(device)
        mask[i, :t] = True

    mem_gb = X.numel() * 2 / 1e9
    logger.info(f"  GPU tensor size: {mem_gb:.2f} GB")
    return X, mask, y, T_max, D


class TransformerProbe(nn.Module):
    def __init__(self, input_dim, hidden=256, num_layers=2, num_heads=4,
                 num_classes=4, max_seq=520, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq, hidden))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads,
            dim_feedforward=hidden * 4, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: (B, T, D) (bf16/fp32), mask: (B, T) bool, True=valid
        x = self.input_proj(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        T = x.size(1)
        x = x + self.pos_embed[:, :T]
        cls_mask = torch.ones(x.size(0), 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)
        src_kp = ~full_mask  # PyTorch: True=ignore
        x = self.transformer(x, src_key_padding_mask=src_kp)
        cls_out = self.norm(x[:, 0])
        cls_out = self.dropout(cls_out)
        return self.head(cls_out)


@torch.no_grad()
def evaluate(model, X, mask, y, idx, batch_size, device):
    model.eval()
    correct = total = 0
    for i in range(0, len(idx), batch_size):
        b = idx[i:i+batch_size]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(X[b], mask[b])
        pred = logits.argmax(-1)
        correct += (pred == y[b]).sum().item()
        total += b.numel()
    return correct / total


def train_fold(X, mask, y, train_idx, val_idx, test_idx, input_dim, num_classes,
               max_seq, device, epochs=50, lr=5e-4, batch_size=256, patience=10,
               num_layers=2, hidden=256, num_heads=4):
    model = TransformerProbe(
        input_dim=input_dim, hidden=hidden, num_layers=num_layers, num_heads=num_heads,
        num_classes=num_classes, max_seq=max_seq, dropout=0.1
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()

    best_val = 0.0
    best_state = None
    bad_epochs = 0
    n_train = len(train_idx)

    for epoch in range(epochs):
        model.train()
        t_start = time.time()
        perm = train_idx[torch.randperm(n_train, device=device)]
        loss_sum = 0
        n_batches = 0
        for i in range(0, n_train, batch_size):
            b = perm[i:i+batch_size]
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(X[b], mask[b])
                loss = crit(logits, y[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item()
            n_batches += 1

        val_acc = evaluate(model, X, mask, y, val_idx, batch_size, device)
        if val_acc > best_val + 0.001:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        elapsed = time.time() - t_start
        logger.info(f"  ep{epoch+1:02d}: loss={loss_sum/n_batches:.4f} val={val_acc:.4f} best={best_val:.4f} ({elapsed:.2f}s)")
        if bad_epochs >= patience:
            logger.info(f"  early stop at ep{epoch+1}")
            break

    model.load_state_dict(best_state)
    test_acc = evaluate(model, X, mask, y, test_idx, batch_size, device)
    return {"val_acc": best_val, "test_acc": test_acc}


def run(encoder, dataset, device="cuda:0", seed=42, val_frac=0.15,
        epochs=50, lr=5e-4, batch_size=256, num_layers=2, hidden=256, num_heads=4):
    torch.manual_seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(MANIFEST_DIR / f"{dataset}.csv")
    label_col, fold_col = LABEL_COL[dataset], FOLD_COL[dataset]
    classes = sorted(df[label_col].unique())
    label2idx = {c: i for i, c in enumerate(classes)}
    df["label_idx"] = df[label_col].map(label2idx)
    num_classes = len(classes)

    emb_dir = EMB_FRAMES_ROOT / encoder / dataset

    logger.info(f"\n=== {encoder} × {dataset} × L={num_layers} ===  classes={num_classes}")

    # Pre-load whole dataset to GPU once
    X, mask, y, T_max, input_dim = preload_to_gpu(emb_dir, df, device)
    max_seq = T_max + 1 + 8  # +1 CLS, +8 safety

    folds = sorted(df[fold_col].unique(), key=str)
    rng = np.random.default_rng(seed)
    results = []

    fold_arr = df[fold_col].values

    for f in folds:
        is_test = fold_arr == f
        is_rest = ~is_test
        rest_idx_np = np.where(is_rest)[0]
        rng.shuffle(rest_idx_np)
        n_val = int(len(rest_idx_np) * val_frac)
        val_idx_np = rest_idx_np[:n_val]
        train_idx_np = rest_idx_np[n_val:]
        test_idx_np = np.where(is_test)[0]

        train_idx = torch.tensor(train_idx_np, dtype=torch.long, device=device)
        val_idx = torch.tensor(val_idx_np, dtype=torch.long, device=device)
        test_idx = torch.tensor(test_idx_np, dtype=torch.long, device=device)
        logger.info(f"\n[fold {f}] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

        m = train_fold(X, mask, y, train_idx, val_idx, test_idx,
                       input_dim, num_classes, max_seq, device,
                       epochs=epochs, lr=lr, batch_size=batch_size,
                       num_layers=num_layers, hidden=hidden, num_heads=num_heads)
        logger.info(f"[fold {f}] val={m['val_acc']:.4f} test={m['test_acc']:.4f}")
        results.append({"encoder": encoder, "dataset": dataset, "fold": str(f),
                        "num_layers": num_layers, **m})

    # Free GPU memory before next config
    del X, mask, y
    torch.cuda.empty_cache()

    # Save (append/replace by encoder × dataset × num_layers)
    df_out = pd.DataFrame(results)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV)
        if "num_layers" not in existing.columns:
            existing["num_layers"] = 2
        keys = set(zip(df_out.encoder, df_out.dataset, df_out.num_layers))
        existing = existing[~existing.apply(
            lambda r: (r.encoder, r.dataset, r.num_layers) in keys, axis=1)]
        df_out = pd.concat([existing, df_out], ignore_index=True)
    df_out.to_csv(RESULTS_CSV, index=False)

    mean_test = np.mean([r["test_acc"] for r in results])
    std_test = np.std([r["test_acc"] for r in results])
    logger.info(f"\n  >>> {encoder} × {dataset} × L={num_layers}: test = {mean_test:.4f} ± {std_test:.4f}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", nargs="+", default=["whisper_small", "dacvae"])
    p.add_argument("--datasets", nargs="+", default=["iemocap_4class"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, nargs="+", default=[2])
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    for nl in args.num_layers:
        for enc in args.encoders:
            for ds in args.datasets:
                run(enc, ds, device=args.device,
                    epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                    num_layers=nl, hidden=args.hidden, num_heads=args.num_heads)


if __name__ == "__main__":
    main()
