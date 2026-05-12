#!/usr/bin/env python
"""Stitch distance / probe / cka CSVs into one markdown report.

Inputs (under `--out_dir`):
  dist_per_layer.csv         (from distances.py)
  probe_per_layer.csv        (from probe.py, optional)
  cka.csv                    (from cka.py, optional)
  *.png                      (figures, referenced in-place)

Output: `{out_dir}/report.md`
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def f(x, prec=3):
    try:
        return f"{float(x):.{prec}f}"
    except Exception:
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="analyze output dir (where csvs live)")
    args = ap.parse_args()
    out_dir = Path(args.out)

    dist = read_csv(out_dir / "dist_per_layer.csv")
    probe = read_csv(out_dir / "probe_per_layer.csv")
    cka = read_csv(out_dir / "cka.csv")

    lines: list[str] = []
    lines += [
        "# RQ2 / Analysis 3 — Speaker invariance across encoder, projector, LLM",
        "",
        f"Generated from analysis CSVs under `{out_dir}`.",
        "",
        "## Models and corpora",
        "",
    ]
    models = sorted({r["model"] for r in dist})
    srcs = sorted({r["src"] for r in dist})
    lines += [
        f"- Models in this run: {', '.join(f'`{m}`' for m in models)}",
        f"- Corpora: {', '.join(f'`{s}`' for s in srcs)}",
        "",
        "## Distance ratios (cosine, paper-style)",
        "",
        "`ratio_paper = mean(same-transcript / diff-speaker) / mean(same-speaker / diff-transcript)`",
        "",
        "Interpretation:",
        "- `>1` ⇒ speaker variation dominates over content variation (speaker info preserved)",
        "- `<1` ⇒ content variation dominates over speaker variation (speaker info attenuated)",
        "- `=1` ⇒ matched",
        "",
    ]

    # one table per corpus
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in dist:
        if r["metric"] != "cos":
            continue
        by_src[r["src"]].append(r)

    for s, rows in by_src.items():
        lines.append(f"### `{s}`")
        lines.append("")
        layers = sorted({(r["group"], int(r["layer_idx"]), r["layer"]) for r in rows},
                        key=lambda t: (t[0], t[1]))
        header = "| layer | " + " | ".join(f"`{m}`" for m in models) + " |"
        sep = "|:--|" + "--:|" * len(models)
        lines += [header, sep]
        for grp, lidx, lname in layers:
            row = [f"`{lname}`"]
            for m in models:
                cell = next((r for r in rows if r["model"] == m and r["layer"] == lname), None)
                row.append(f(cell["ratio_paper"]) if cell else "-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    if probe:
        lines += ["## Linear-probe accuracy", "",
                  "Speaker-ID vs Content-ID 5-fold linear-probe accuracy. The gap",
                  "`content_acc - speaker_acc` quantifies how cleanly a layer separates",
                  "content from speaker — large positive gap = content-recoverable, speaker-suppressed.",
                  ""]
        for s in srcs:
            lines.append(f"### `{s}`")
            lines.append("")
            layers = sorted({(r["group"], int(r["layer_idx"]), r["layer"])
                              for r in probe if r["src"] == s},
                             key=lambda t: (t[0], t[1]))
            header = "| layer | " + " | ".join(f"`{m}` spk / cnt / gap" for m in models) + " |"
            sep = "|:--|" + "--:|" * len(models)
            lines += [header, sep]
            for grp, lidx, lname in layers:
                row = [f"`{lname}`"]
                for m in models:
                    rs = [r for r in probe if r["model"] == m and r["layer"] == lname and r["src"] == s]
                    spk_r = next((r for r in rs if r["target"] == "speaker"), None)
                    cnt_r = next((r for r in rs if r["target"] == "content"), None)
                    if spk_r and cnt_r:
                        sp, ct = float(spk_r["acc"]), float(cnt_r["acc"])
                        row.append(f"{sp:.2f} / {ct:.2f} / {ct - sp:+.2f}")
                    else:
                        row.append("-")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    if cka:
        lines += ["## CKA between models", ""]
        pairs = sorted({(r["model_a"], r["model_b"]) for r in cka})
        for ma, mb in pairs:
            sub = [r for r in cka if r["model_a"] == ma and r["model_b"] == mb]
            if not sub:
                continue
            top = sorted(sub, key=lambda r: -float(r["cka"]))[:5]
            lines.append(f"### `{ma}` ↔ `{mb}`")
            lines.append("")
            lines.append("Top 5 layer pairs by CKA:")
            lines.append("")
            lines.append("| layer_a | layer_b | CKA | N |")
            lines.append("|:--|:--|--:|--:|")
            for r in top:
                lines.append(f"| `{r['layer_a']}` | `{r['layer_b']}` | {f(r['cka'])} | {r['n']} |")
            lines.append("")
            lines.append(f"![CKA heatmap](cka__{ma}__vs__{mb}.png)")
            lines.append("")

    # auto-pick up figures
    figs = sorted(out_dir.glob("*.png"))
    if figs:
        lines += ["## Figures", ""]
        for fig in figs:
            lines.append(f"![{fig.name}]({fig.name})")
            lines.append("")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"[report] wrote {report_path}")


if __name__ == "__main__":
    main()
