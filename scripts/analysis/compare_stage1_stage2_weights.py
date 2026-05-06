"""Compare Stage 1 (full safetensors) vs Stage 2 (LoRA + full projector) weights.

Computes per-tensor relative L2 magnitude and Frobenius cosine similarity for:
  • encoder        — verifies it's untouched (Stage 2 must not contain encoder weights)
  • projector      — full-tunable in Stage 2 (modules_to_save), direct W vs W diff
  • LLM attention  — LoRA: ΔW = (B @ A) × (alpha/r), compared against base W

Outputs a 2x2 heatmap (rows: L2 magnitude / cosine drift, cols: projector / LLM).
Pairs with the analysis in stage2_eval_harness.md §weight-drift.

Usage:
    python scripts/analysis/compare_stage1_stage2_weights.py \\
        --stage1 external/ckpts/.../Qwen3.5_wavtok_40_unify_Stage1/checkpoint-100000 \\
        --stage2 /mnt/tmp/results/.../checkpoint-12000 \\
        --out    /mnt/tmp/results/weight_change_heatmap.png

Notes
-----
* Frobenius cosine: treats weight matrix as a flattened vector, computes
  <a,b> / (||a|| ||b||). Computed in float64 to avoid Cauchy-Schwarz violations
  from float32/bf16 accumulation on >1M-element tensors.
* `modules_to_save` (PEFT) means projector is stored as full weight in
  adapter_model.safetensors — direct comparison works.
* `lora_target` modules store only A,B factors; we reconstruct the effective
  delta and compare to the base weight loaded from Stage 1.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


# ──────────────────────────────────────────────────────────────────────────────
# loaders
# ──────────────────────────────────────────────────────────────────────────────
def open_stage1_index(stage1_dir: Path) -> tuple[dict, Path]:
    """Return (weight_map, dir) for safetensors-sharded Stage 1 ckpt."""
    idx_path = stage1_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"missing weight index at {idx_path}")
    return json.load(idx_path.open())["weight_map"], stage1_dir


def get_s1_tensor(key: str, weight_map: dict, stage1_dir: Path) -> torch.Tensor:
    shard = weight_map[key]
    with safe_open(stage1_dir / shard, framework="pt") as f:
        return f.get_tensor(key)


# ──────────────────────────────────────────────────────────────────────────────
# metrics
# ──────────────────────────────────────────────────────────────────────────────
def cos_flat_f64(a: torch.Tensor, b: torch.Tensor) -> float:
    """Frobenius cosine similarity in float64 (numerically stable)."""
    a = a.double().flatten(); b = b.double().flatten()
    val = (a @ b / (a.norm() * b.norm() + 1e-30)).item()
    return min(max(val, -1.0), 1.0)


def rel_l2_f64(w1: torch.Tensor, w2: torch.Tensor) -> float:
    """||w2 - w1||_F / ||w1||_F (× 100 — returned as percentage)."""
    w1, w2 = w1.double(), w2.double()
    return ((w2 - w1).norm() / (w1.norm() + 1e-30) * 100).item()


# ──────────────────────────────────────────────────────────────────────────────
# comparisons
# ──────────────────────────────────────────────────────────────────────────────
def verify_encoder_frozen(s2_adapter_path: Path, s1_keys: set[str]) -> tuple[int, int]:
    """Return (n_encoder_tensors_in_s1, n_encoder_tensors_in_s2_adapter).
    Stage 2 LoRA fine-tune with freeze_vision_tower=true must not touch encoder."""
    n_s1 = sum(1 for k in s1_keys if "audio_encoder" in k and "projector" not in k)
    with safe_open(s2_adapter_path, framework="pt") as f:
        n_s2 = sum(1 for k in f.keys() if "audio_encoder" in k and "projector" not in k)
    return n_s1, n_s2


PROJ_PARAM_TYPES = (
    "input_layernorm", "post_attention_layernorm",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


def projector_change(s2_adapter_path: Path, weight_map: dict, stage1_dir: Path,
                     n_proj_layers: int = 4):
    """Return (l2_pct, cos_sim) numpy arrays of shape (n_proj_layers, len(types)).

    Also returns dict of standalone projector params (input_proj/output_proj/final_norm).
    """
    L2  = np.full((n_proj_layers, len(PROJ_PARAM_TYPES)), np.nan)
    COS = np.full((n_proj_layers, len(PROJ_PARAM_TYPES)), np.nan)
    standalone: dict[str, tuple[float, float]] = {}

    pat_layered = re.compile(r".*audio_encoder\.projector\.layers\.(\d+)\.(.+)\.weight$")
    standalone_keys = ("input_proj.weight", "output_proj.weight", "final_norm.weight")

    with safe_open(s2_adapter_path, framework="pt") as fa:
        for k_s2 in fa.keys():
            if "audio_encoder.projector" not in k_s2:
                continue
            k_s1 = k_s2.replace("base_model.model.", "")
            if k_s1 not in weight_map:
                continue
            w1 = get_s1_tensor(k_s1, weight_map, stage1_dir)
            w2 = fa.get_tensor(k_s2)
            l2 = rel_l2_f64(w1, w2); cs = cos_flat_f64(w1, w2)

            m = pat_layered.match(k_s1)
            if m:
                L = int(m.group(1)); pt = m.group(2)
                if pt in PROJ_PARAM_TYPES:
                    j = PROJ_PARAM_TYPES.index(pt)
                    L2 [L, j] = l2; COS[L, j] = cs
                continue
            for tag in standalone_keys:
                if k_s1.endswith(tag):
                    standalone[tag] = (l2, cs)
                    break
    return L2, COS, standalone


def llm_lora_change(s2_adapter_path: Path, weight_map: dict, stage1_dir: Path,
                    lora_layers: list[int]):
    """Reconstruct ΔW = B@A * alpha/r per LoRA-targeted q/k/v/o_proj and compare to base W."""
    proj_types = ["q_proj", "k_proj", "v_proj", "o_proj"]
    L2  = np.full((len(lora_layers), len(proj_types)), np.nan)
    COS = np.full((len(lora_layers), len(proj_types)), np.nan)

    cfg = json.load((s2_adapter_path.parent / "adapter_config.json").open())
    scale = cfg["lora_alpha"] / cfg["r"]
    pat_pair = re.compile(r"^(.*\.(?:q|k|v|o)_proj)\.lora_(A|B)\.weight$")

    with safe_open(s2_adapter_path, framework="pt") as fa:
        pairs: dict[str, dict[str, str]] = {}
        for k in fa.keys():
            m = pat_pair.match(k)
            if m: pairs.setdefault(m.group(1), {})[m.group(2)] = k

        for base, ab in pairs.items():
            if "A" not in ab or "B" not in ab: continue
            A = fa.get_tensor(ab["A"]).double()
            B = fa.get_tensor(ab["B"]).double()
            delta = (B @ A) * scale
            s1_key = base.replace("base_model.model.", "") + ".weight"
            if s1_key not in weight_map: continue
            W = get_s1_tensor(s1_key, weight_map, stage1_dir).double()
            W_new = W + delta

            m = re.search(r"layers\.(\d+)\.self_attn\.(\w+)_proj", base)
            if not m: continue
            layer, ptype = int(m.group(1)), m.group(2) + "_proj"
            if layer not in lora_layers: continue
            i, j = lora_layers.index(layer), proj_types.index(ptype)
            L2 [i, j] = (delta.norm() / W.norm() * 100).item()
            COS[i, j] = cos_flat_f64(W, W_new)
    return L2, COS, scale, cfg


def get_lora_layers(adapter_cfg_path: Path) -> list[int]:
    cfg = json.load(adapter_cfg_path.open())
    layers = set()
    for m in cfg["target_modules"]:
        mm = re.search(r"layers\.(\d+)\.", m)
        if mm: layers.add(int(mm.group(1)))
    return sorted(layers)


# ──────────────────────────────────────────────────────────────────────────────
# plotting
# ──────────────────────────────────────────────────────────────────────────────
def plot_heatmaps(P_l2, P_cos, L_l2, L_cos, lora_layers, out_path: Path,
                  title_suffix: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xt_p = [t.replace("self_attn.","").replace("mlp.","").replace("_layernorm","_LN")
            for t in PROJ_PARAM_TYPES]
    yt_p = [f"L{i}" for i in range(P_l2.shape[0])]
    xt_l = ["q_proj", "k_proj", "v_proj", "o_proj"]
    yt_l = [f"L{i}" for i in lora_layers]

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 0.6],
                          height_ratios=[1, 1], wspace=0.3, hspace=0.55)

    def _hm(ax, M, xt, yt, title, cmap, vmin, vmax, fmt, cbar_lab):
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(xt))); ax.set_xticklabels(xt, rotation=45, ha="right")
        ax.set_yticks(range(len(yt))); ax.set_yticklabels(yt)
        ax.set_title(title)
        span = vmax - vmin
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    norm = (M[i, j] - vmin) / max(span, 1e-9)
                    color = "white" if norm < 0.55 else "black"
                    ax.text(j, i, f"{M[i,j]:{fmt}}", ha="center", va="center",
                            color=color, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=cbar_lab)

    vmax_l2 = max(np.nanmax(P_l2), np.nanmax(L_l2))
    _hm(fig.add_subplot(gs[0, 0]), P_l2, xt_p, yt_p,
        "Projector — relative L2 change (%) [full fine-tune]",
        "viridis", 0, vmax_l2, ".1f", "rel L2 diff (%)")
    _hm(fig.add_subplot(gs[0, 1]), L_l2, xt_l, yt_l,
        "LLM — LoRA effective Δ (%)",
        "viridis", 0, vmax_l2, ".1f", "rel L2 diff (%)")

    P_drift = 1 - P_cos; L_drift = 1 - L_cos
    vmax_d = max(np.nanmax(P_drift), np.nanmax(L_drift))
    _hm(fig.add_subplot(gs[1, 0]), P_drift, xt_p, yt_p,
        "Projector — (1 − cosine sim)",
        "magma", 0, vmax_d, ".5f", "1 − cos sim")
    _hm(fig.add_subplot(gs[1, 1]), L_drift, xt_l, yt_l,
        "LLM — (1 − cosine sim)",
        "magma", 0, vmax_d, ".5f", "1 − cos sim")

    fig.suptitle(f"Stage1 → Stage2 weight change{title_suffix}\n"
                 "[top: rel L2 magnitude · bottom: directional drift via cosine]",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1", required=True, type=Path,
                   help="Stage 1 checkpoint dir (must contain model.safetensors.index.json)")
    p.add_argument("--stage2", required=True, type=Path,
                   help="Stage 2 checkpoint dir (must contain adapter_model.safetensors)")
    p.add_argument("--out", required=True, type=Path, help="Output PNG path")
    p.add_argument("--n-proj-layers", type=int, default=4)
    p.add_argument("--title-suffix", default="")
    args = p.parse_args()

    weight_map, s1_dir = open_stage1_index(args.stage1)
    s1_keys = set(weight_map.keys())
    s2_adapter = args.stage2 / "adapter_model.safetensors"
    if not s2_adapter.exists():
        raise FileNotFoundError(s2_adapter)

    # 1. encoder frozen check
    n_enc_s1, n_enc_s2 = verify_encoder_frozen(s2_adapter, s1_keys)
    print(f"[encoder] Stage1 has {n_enc_s1} encoder tensors. "
          f"Stage2 adapter contains {n_enc_s2} encoder tensors "
          f"({'OK — frozen' if n_enc_s2 == 0 else 'WARNING: encoder modified!'}).")

    # 2. projector
    P_l2, P_cos, standalone = projector_change(s2_adapter, weight_map, s1_dir,
                                               n_proj_layers=args.n_proj_layers)
    print("\n[projector] standalone:")
    for k, (l2, cs) in standalone.items():
        print(f"  {k:<20} L2={l2:5.2f}%   cos={cs:.5f}")

    # 3. LLM LoRA
    lora_layers = get_lora_layers(args.stage2 / "adapter_config.json")
    print(f"\n[LLM] LoRA layers: {lora_layers}")
    L_l2, L_cos, scale, cfg = llm_lora_change(s2_adapter, weight_map, s1_dir, lora_layers)
    print(f"  alpha={cfg['lora_alpha']}  r={cfg['r']}  scale={scale}")
    print(f"  cos sim range: {np.nanmin(L_cos):.6f} ~ {np.nanmax(L_cos):.6f}")

    # 4. plot
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_heatmaps(P_l2, P_cos, L_l2, L_cos, lora_layers, args.out,
                  title_suffix=args.title_suffix)


if __name__ == "__main__":
    main()
