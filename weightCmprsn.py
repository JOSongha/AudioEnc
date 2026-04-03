"""
Compare two AudioQwen full-weight .pt checkpoints and compute
per-module average weight difference (L2 norm of diff, then mean).

사용법:
    python weightCmprsn.py path/to/step1.pt path/to/step20.pt
    python weightCmprsn.py path/to/step1.pt path/to/step20.pt --depth 1
"""

import argparse
import torch
from collections import defaultdict


def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Load state dict from a .pt file saved via torch.save(model.state_dict(), path)."""
    return torch.load(path, map_location="cpu", weights_only=True)


def get_module_name(param_name: str, depth: int) -> str:
    """Extract module name up to given depth from parameter name."""
    parts = param_name.split(".")
    return ".".join(parts[:depth])


def compare(sd1: dict, sd2: dict, depth: int):
    """Compare two state dicts and return per-module stats."""
    common_keys = sorted(set(sd1.keys()) & set(sd2.keys()))
    only_in_1 = sorted(set(sd1.keys()) - set(sd2.keys()))
    only_in_2 = sorted(set(sd2.keys()) - set(sd1.keys()))

    module_diffs = defaultdict(lambda: {"sum_abs_diff": 0.0, "sum_l2_diff": 0.0, "num_params": 0, "num_elements": 0})

    for key in common_keys:
        t1 = sd1[key].float()
        t2 = sd2[key].float()
        diff = (t1 - t2)

        abs_diff = diff.abs().sum().item()
        l2_diff = diff.norm(2).item()
        numel = t1.numel()

        module = get_module_name(key, depth)
        module_diffs[module]["sum_abs_diff"] += abs_diff
        module_diffs[module]["sum_l2_diff"] += l2_diff
        module_diffs[module]["num_params"] += 1
        module_diffs[module]["num_elements"] += numel

    return module_diffs, common_keys, only_in_1, only_in_2


def main():
    parser = argparse.ArgumentParser(description="Compare two HuggingFace checkpoints")
    parser.add_argument("ckpt1", help="Path or HF hub ID for checkpoint 1")
    parser.add_argument("ckpt2", help="Path or HF hub ID for checkpoint 2")
    parser.add_argument("--depth", type=int, default=2,
                        help="Module name depth for grouping (default: 2)")
    args = parser.parse_args()

    print(f"Loading checkpoint 1: {args.ckpt1}")
    sd1 = load_state_dict(args.ckpt1)
    print(f"Loading checkpoint 2: {args.ckpt2}")
    sd2 = load_state_dict(args.ckpt2)

    module_diffs, common, only1, only2 = compare(sd1, sd2, args.depth)

    # Summary
    print(f"\n{'='*80}")
    print(f"Comparison: {args.ckpt1}  vs  {args.ckpt2}")
    print(f"{'='*80}")
    print(f"Common parameters : {len(common)}")
    print(f"Only in ckpt1     : {len(only1)}")
    print(f"Only in ckpt2     : {len(only2)}")

    if only1:
        print(f"\n  [Only in ckpt1] {only1[:5]}{'...' if len(only1) > 5 else ''}")
    if only2:
        print(f"\n  [Only in ckpt2] {only2[:5]}{'...' if len(only2) > 5 else ''}")

    # Per-module table
    print(f"\n{'Module':<50} {'#Params':>8} {'#Elements':>12} {'MeanAbsDiff':>14} {'MeanL2Diff':>14}")
    print("-" * 100)

    total_abs = 0.0
    total_l2 = 0.0
    total_elements = 0

    for module in sorted(module_diffs.keys()):
        d = module_diffs[module]
        mean_abs = d["sum_abs_diff"] / d["num_elements"]
        mean_l2 = d["sum_l2_diff"] / d["num_params"]
        total_abs += d["sum_abs_diff"]
        total_l2 += d["sum_l2_diff"]
        total_elements += d["num_elements"]
        print(f"{module:<50} {d['num_params']:>8} {d['num_elements']:>12,} {mean_abs:>14.6e} {mean_l2:>14.6e}")

    print("-" * 100)
    if total_elements > 0:
        print(f"{'TOTAL':<50} {len(common):>8} {total_elements:>12,} {total_abs/total_elements:>14.6e} {total_l2/len(common):>14.6e}")

    print()


if __name__ == "__main__":
    main()
