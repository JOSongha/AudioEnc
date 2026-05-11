"""Generate a diagnostic HTML page for the 4-set (W±A±) analysis.

Covers:
  - Representative audio samples per set with prediction overlays
  - Per-set source / pitch / family breakdowns
  - Confusion patterns (DAC on W+A-, Whisper on W-A+)
  - "Why" narrative panels

Usage:
    python -m experiments.audio_encoder_probe.phase_a.make_diagnostic_html
    python -m experiments.audio_encoder_probe.phase_a.make_diagnostic_html \
        --combined_csv ... --meta_csv ... --out_html ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROBE_ROOT = ROOT / "experiments" / "audio_encoder_probe"
PRED_DIR = PROBE_ROOT / "_results" / "predictions"

SET_LABELS = ["W+A+", "W+A-", "W-A+", "W-A-"]
SET_COLORS = {
    "W+A+": "#22c55e",
    "W+A-": "#ef4444",
    "W-A+": "#3b82f6",
    "W-A-": "#9ca3af",
}
SET_BG = {
    "W+A+": "#f0fdf4",
    "W+A-": "#fef2f2",
    "W-A+": "#eff6ff",
    "W-A-": "#f9fafb",
}
SET_DESC = {
    "W+A+": "Whisper ✓  DAC ✓ — 둘 다 맞춤. 두 인코더 모두 악기 특징을 잘 포착한 '쉬운' 샘플.",
    "W+A-": "Whisper ✓  DAC ✗ — Whisper만 맞춤. Whisper의 언어·음향 사전학습이 DAC z_e가 놓치는 정보를 포착.",
    "W-A+": "Whisper ✗  DAC ✓ — DAC만 맞춤. 드문 케이스 — Whisper가 틀리고 DAC이 맞는 예외적 샘플.",
    "W-A-": "Whisper ✗  DAC ✗ — 둘 다 틀림. 두 인코더 모두 어려운 샘플. electronic 비중이 높음.",
}
SOURCE_COLORS = {"acoustic": "#3b82f6", "electronic": "#f97316", "synthetic": "#8b5cf6"}


def _audio_relpath(audio_path: str) -> str:
    """Convert absolute audio path to server-relative path."""
    p = str(audio_path)
    if p.startswith("/mnt/tmp/datasets/music/"):
        return "nsynth_audio_root/" + p[len("/mnt/tmp/datasets/music/"):]
    return "file://" + p


def _bar(value: float, total: float, color: str, width_px: int = 140) -> str:
    pct = value / total * 100 if total > 0 else 0
    w = int(pct / 100 * width_px)
    return (
        f'<span style="display:inline-block;background:{color};height:10px;'
        f'width:{w}px;border-radius:3px;vertical-align:middle;"></span>'
        f' <small>{pct:.1f}%&nbsp;({int(value)})</small>'
    )


def _prob_badge(prob: float, correct: bool) -> str:
    bg = "#bbf7d0" if correct else "#fecaca"
    border = "#16a34a" if correct else "#dc2626"
    return (
        f'<span style="background:{bg};border:1px solid {border};'
        f'border-radius:4px;padding:1px 6px;font-size:12px;font-weight:600;">'
        f'{prob:.3f}</span>'
    )


def _sample_card(row: pd.Series, audio_root_rel: str = "") -> str:
    audio_rel = _audio_relpath(row.audio_path)
    w_ok = str(row.whisper_small_correct).lower() in ("true", "1")
    d_ok = str(row.dacvae_correct).lower() in ("true", "1")
    card_bg = "#fff"
    return f"""
<div style="border:1px solid #e5e7eb;border-radius:8px;padding:12px 14px;
            background:{card_bg};min-width:240px;max-width:280px;flex-shrink:0;">
  <div style="font-size:11px;color:#6b7280;margin-bottom:4px;word-break:break-all;">
    {row.utt_id}</div>
  <audio controls preload="none"
         style="width:100%;height:30px;margin-bottom:8px;"
         src="{audio_rel}"></audio>
  <table style="font-size:12px;border-collapse:collapse;width:100%">
    <tr>
      <td style="color:#6b7280;padding:2px 0;">true</td>
      <td style="font-weight:700;padding:2px 6px;">{row.true_label}</td>
      <td></td>
    </tr>
    <tr>
      <td style="color:#6b7280;padding:2px 0;">W pred</td>
      <td style="padding:2px 6px;">{"✓ " if w_ok else "✗ "}{row.whisper_small_pred}</td>
      <td>{_prob_badge(float(row.whisper_small_proba), w_ok)}</td>
    </tr>
    <tr>
      <td style="color:#6b7280;padding:2px 0;">D pred</td>
      <td style="padding:2px 6px;">{"✓ " if d_ok else "✗ "}{row.dacvae_pred}</td>
      <td>{_prob_badge(float(row.dacvae_proba), d_ok)}</td>
    </tr>
    <tr>
      <td style="color:#6b7280;padding:2px 0;">source</td>
      <td colspan="2" style="padding:2px 6px;">
        <span style="background:{SOURCE_COLORS.get(getattr(row,'source',''),
              '#e5e7eb')};color:#fff;border-radius:3px;
              padding:1px 5px;font-size:11px;">{getattr(row,'source','?')}</span>
        pitch {getattr(row,'pitch','?')}
      </td>
    </tr>
  </table>
</div>"""


def _confusion_table(pairs: pd.Series, n: int = 10) -> str:
    rows_html = ""
    for (true_l, pred_l), cnt in pairs.head(n).items():
        rows_html += (
            f"<tr><td style='padding:3px 10px 3px 0'>{true_l}</td>"
            f"<td style='padding:3px 10px'>→</td>"
            f"<td style='padding:3px 10px 3px 0'><b>{pred_l}</b></td>"
            f"<td style='padding:3px 0;color:#6b7280'>{cnt}</td></tr>"
        )
    return f"<table style='font-size:12px;border-collapse:collapse'>{rows_html}</table>"


def _source_bars(counts: dict, total: int) -> str:
    lines = []
    for src in ("acoustic", "electronic", "synthetic"):
        n = counts.get(src, 0)
        lines.append(
            f"<tr><td style='font-size:12px;padding:2px 8px 2px 0;'>{src}</td>"
            f"<td>{_bar(n, total, SOURCE_COLORS[src], 160)}</td></tr>"
        )
    return f"<table style='border-collapse:collapse'>{''.join(lines)}</table>"


def _family_bars(vc: pd.Series, n: int = 6) -> str:
    top = vc.head(n)
    total = vc.sum()
    lines = []
    for fam, cnt in top.items():
        lines.append(
            f"<tr><td style='font-size:12px;padding:2px 8px 2px 0;width:80px'>{fam}</td>"
            f"<td>{_bar(cnt, total, '#64748b', 140)}</td></tr>"
        )
    return f"<table style='border-collapse:collapse'>{''.join(lines)}</table>"


def _set_section(label: str, sub: pd.DataFrame, samples: pd.DataFrame,
                 confusions: pd.Series | None, total: int) -> str:
    color = SET_COLORS[label]
    bg = SET_BG[label]
    src_counts = sub["source"].value_counts().to_dict() if "source" in sub else {}
    mean_w = sub["whisper_small_proba"].mean()
    mean_d = sub["dacvae_proba"].mean()
    mean_p = sub["pitch"].mean() if "pitch" in sub else float("nan")

    sample_cards = "\n".join(_sample_card(row) for _, row in samples.iterrows())

    confusion_html = ""
    if confusions is not None and len(confusions) > 0:
        conf_title = "DAC confusion (top predicted instead)" if "A-" in label else "Whisper confusion (top predicted instead)"
        confusion_html = f"""
<div style="margin-top:16px">
  <div style="font-size:13px;font-weight:600;margin-bottom:6px;">{conf_title}</div>
  {_confusion_table(confusions)}
</div>"""

    family_html = _family_bars(sub["true_label"].value_counts()) if "true_label" in sub else ""

    return f"""
<section id="{label.replace('+','p').replace('-','m')}"
         style="background:{bg};border:2px solid {color};border-radius:12px;
                padding:20px 24px;margin-bottom:28px;">
  <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:6px">
    <h2 style="margin:0;font-size:22px;color:{color}">{label}</h2>
    <span style="font-size:14px;color:#374151;font-weight:500">
      n={len(sub):,} &nbsp;({len(sub)/total*100:.1f}% of total)
    </span>
  </div>
  <p style="margin:0 0 16px;font-size:14px;color:#374151">{SET_DESC[label]}</p>

  <!-- Stats row -->
  <div style="display:flex;gap:32px;flex-wrap:wrap;margin-bottom:18px;font-size:13px">
    <div>
      <div style="color:#6b7280;margin-bottom:4px">Whisper mean proba</div>
      <div style="font-size:20px;font-weight:700;color:{color}">{mean_w:.3f}</div>
    </div>
    <div>
      <div style="color:#6b7280;margin-bottom:4px">DAC mean proba</div>
      <div style="font-size:20px;font-weight:700;color:{color}">{mean_d:.3f}</div>
    </div>
    <div>
      <div style="color:#6b7280;margin-bottom:4px">mean pitch (MIDI)</div>
      <div style="font-size:20px;font-weight:700;color:{color}">{mean_p:.1f}</div>
    </div>
    <div>
      <div style="color:#6b7280;margin-bottom:4px">source 분포</div>
      {_source_bars(src_counts, len(sub))}
    </div>
    <div>
      <div style="color:#6b7280;margin-bottom:4px">instrument family (top 6)</div>
      {family_html}
    </div>
    {confusion_html}
  </div>

  <!-- Sample cards -->
  <div style="font-size:13px;font-weight:600;margin-bottom:10px;color:#374151">
    대표 샘플 ({len(samples)}개)
    {"— 가장 높은 Whisper confidence" if "W+" in label else "— 가장 높은 DAC confidence" if label == "W-A+" else "— 두 모델 모두 높은 confidence로 틀린 샘플"}
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:12px;">
    {sample_cards}
  </div>
</section>"""


def build_html(df: pd.DataFrame, total: int, chi2_df: pd.DataFrame) -> str:
    sets = {
        "W+A+": df[df.whisper_small_correct.astype(bool) & df.dacvae_correct.astype(bool)],
        "W+A-": df[df.whisper_small_correct.astype(bool) & ~df.dacvae_correct.astype(bool)],
        "W-A+": df[~df.whisper_small_correct.astype(bool) & df.dacvae_correct.astype(bool)],
        "W-A-": df[~df.whisper_small_correct.astype(bool) & ~df.dacvae_correct.astype(bool)],
    }

    sections = []
    for label in SET_LABELS:
        sub = sets[label]
        if label == "W+A+":
            samples = sub.nlargest(8, "whisper_small_proba")
            confusions = None
        elif label == "W+A-":
            samples = sub.nlargest(8, "whisper_small_proba")
            confusions = sub.groupby(["true_label", "dacvae_pred"]).size().sort_values(ascending=False)
        elif label == "W-A+":
            samples = sub.nlargest(8, "dacvae_proba")
            confusions = sub.groupby(["true_label", "whisper_small_pred"]).size().sort_values(ascending=False)
        else:  # W-A-
            # both wrong with high combined confidence
            sub2 = sub.copy()
            sub2["combined_proba"] = sub2["whisper_small_proba"] + sub2["dacvae_proba"]
            samples = sub2.nlargest(8, "combined_proba")
            confusions = None
        sections.append(_set_section(label, sub, samples, confusions, total))

    # Overview table
    overview_rows = ""
    for label in SET_LABELS:
        sub = sets[label]
        c = SET_COLORS[label]
        src = sub["source"].value_counts() if "source" in sub else pd.Series()
        overview_rows += f"""
<tr>
  <td style="padding:8px 12px;font-weight:700;color:{c}">{label}</td>
  <td style="padding:8px 12px;text-align:right">{len(sub):,}</td>
  <td style="padding:8px 12px;text-align:right">{len(sub)/total*100:.1f}%</td>
  <td style="padding:8px 12px;text-align:right">{sub['whisper_small_proba'].mean():.3f}</td>
  <td style="padding:8px 12px;text-align:right">{sub['dacvae_proba'].mean():.3f}</td>
  <td style="padding:8px 12px;text-align:right">{sub['pitch'].mean():.1f}</td>
  <td style="padding:8px 12px;font-size:12px">
    acou {src.get('acoustic',0)/len(sub)*100:.0f}%
    elec {src.get('electronic',0)/len(sub)*100:.0f}%
    syn {src.get('synthetic',0)/len(sub)*100:.0f}%
  </td>
</tr>"""

    # Chi-square table
    chi2_rows = ""
    for _, r in chi2_df.iterrows():
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "ns"
        chi2_rows += f"""
<tr>
  <td style="padding:6px 12px">{r['attribute']}</td>
  <td style="padding:6px 12px;text-align:right">{r['chi2']:.1f}</td>
  <td style="padding:6px 12px;text-align:right">{r['p']:.2e}</td>
  <td style="padding:6px 12px;font-weight:700;color:#ef4444">{sig}</td>
  <td style="padding:6px 12px;text-align:right">{int(r['dof'])}</td>
</tr>"""

    # Why narrative
    wam = sets["W+A-"]
    wap = sets["W-A+"]
    wam_src = wam.source.value_counts()
    base_src = df.source.value_counts()

    narrative = f"""
<section style="background:#fffbeb;border:2px solid #f59e0b;border-radius:12px;
                padding:20px 24px;margin-bottom:28px;">
  <h2 style="margin:0 0 16px;color:#b45309">왜 Whisper는 맞추고 DAC은 틀리는가? (W+A- 분석)</h2>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;font-size:14px">
    <div>
      <h3 style="font-size:15px;margin:0 0 8px;">Whisper 성공 요인</h3>
      <ul style="margin:0;padding-left:20px;line-height:1.8">
        <li>평균 confidence <b>{wam['whisper_small_proba'].mean():.3f}</b> — 매우 높음</li>
        <li>언어 모델 사전학습이 악기 <em>이름</em> 관련 audio 패턴 학습</li>
        <li>50fps 고해상도 프레임으로 onset 특징 포착</li>
        <li>16kHz + log-mel 입력: 배음 구조에 민감</li>
      </ul>
    </div>
    <div>
      <h3 style="font-size:15px;margin:0 0 8px;">DAC 실패 요인</h3>
      <ul style="margin:0;padding-left:20px;line-height:1.8">
        <li>평균 confidence <b>{wam['dacvae_proba'].mean():.3f}</b> — 낮음</li>
        <li>synthetic 비중 <b>{wam_src.get('synthetic',0)/len(wam)*100:.0f}%</b>
          (전체 대비 {wam_src.get('synthetic',0)/len(wam)*100 - base_src.get('synthetic',0)/len(df)*100:+.0f}pp 과다)</li>
        <li>전음성(timbre) 유사 악기 간 혼동: keyboard↔mallet, guitar↔mallet, flute↔reed</li>
        <li>VAE bottleneck 압축(1024→128)으로 섬세한 배음 정보 손실</li>
        <li>48kHz 코덱 손실 최소화 목적으로 학습 → 악기 식별보다 재현에 최적화</li>
      </ul>
    </div>
  </div>
</section>
<section style="background:#f0f9ff;border:2px solid #0ea5e9;border-radius:12px;
                padding:20px 24px;margin-bottom:28px;">
  <h2 style="margin:0 0 16px;color:#0369a1">왜 DAC은 맞추고 Whisper는 틀리는가? (W-A+ 분석, n={len(wap)})</h2>
  <div style="font-size:14px">
    <p style="margin:0 0 10px">W-A+는 전체의 <b>{len(wap)/total*100:.1f}%</b>로 매우 드문 케이스입니다.</p>
    <ul style="margin:0;padding-left:20px;line-height:1.8">
      <li>Whisper 평균 confidence <b>{wap['whisper_small_proba'].mean():.3f}</b> (낮음) —
          불확실한 경계 케이스에서 DAC가 우연히 맞춤</li>
      <li>mallet→keyboard({wap[(wap.true_label=='mallet')&(wap.whisper_small_pred=='keyboard')].shape[0]}건),
          bass→guitar({wap[(wap.true_label=='bass')&(wap.whisper_small_pred=='guitar')].shape[0]}건):
          Whisper의 언어적 연상이 오히려 혼동 유발</li>
      <li>electronic 비중 <b>{wap.source.value_counts().get('electronic',0)/len(wap)*100:.0f}%</b>
          — 전자악기 특유의 파형에서 DAC 코덱 표현이 유리</li>
      <li>평균 pitch <b>{wap.pitch.mean():.1f}</b> MIDI
          (W+A- {wam.pitch.mean():.1f}보다 {"낮음" if wap.pitch.mean() < wam.pitch.mean() else "높음"})</li>
    </ul>
  </div>
</section>"""

    sections_html = "\n".join(sections)

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NSynth train_30k — 4-set Diagnostic (W±A±)</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 24px 32px;
          background: #f8fafc; color: #1e293b; }}
  h1 {{ font-size: 26px; margin: 0 0 4px }}
  p.subtitle {{ color: #64748b; margin: 0 0 28px; font-size: 14px }}
  nav {{ display:flex; gap: 12px; margin-bottom:28px; flex-wrap:wrap }}
  nav a {{ padding:6px 14px; border-radius:20px; text-decoration:none;
           font-size:13px; font-weight:600; border:2px solid }}
  table.overview {{ border-collapse:collapse; background:white;
                   border:1px solid #e5e7eb; border-radius:8px; overflow:hidden;
                   margin-bottom:28px; font-size:13px }}
  table.overview th {{ background:#f1f5f9; padding:8px 12px; text-align:left;
                      font-size:12px; color:#64748b; font-weight:600 }}
  table.overview tr:hover {{ background:#f8fafc }}
  @media (max-width:700px) {{ body {{ padding:12px }} }}
</style>
</head>
<body>
<h1>NSynth train_30k — 4-set Diagnostic (W±A±)</h1>
<p class="subtitle">
  Whisper-small (85.6%) vs DAC-VAE z_e (55.1%) · NSynth train_30k · n=27,218<br>
  W = whisper_small correct / A = dacvae_prevq correct
</p>

<nav>
  {''.join(f'<a href="#{l.replace("+","p").replace("-","m")}" '
           f'style="color:{SET_COLORS[l]};border-color:{SET_COLORS[l]}">{l}</a>'
           for l in SET_LABELS)}
  <a href="#analysis" style="color:#b45309;border-color:#f59e0b">분석</a>
</nav>

<!-- Overview table -->
<table class="overview">
  <thead>
    <tr>
      <th>Set</th><th>n</th><th>%</th>
      <th>W̄ proba</th><th>D̄ proba</th>
      <th>mean pitch</th><th>source 분포</th>
    </tr>
  </thead>
  <tbody>{overview_rows}</tbody>
</table>

{sections_html}

<div id="analysis">
{narrative}
</div>

<!-- Chi-square table -->
<section style="background:white;border:1px solid #e5e7eb;border-radius:12px;
                padding:20px 24px;margin-bottom:28px;">
  <h2 style="margin:0 0 12px;font-size:18px">Meta-attribute χ² test (W+A- vs 전체)</h2>
  <p style="font-size:13px;color:#6b7280;margin:0 0 12px">
    W+A- 집합이 전체 분포와 통계적으로 다른지 검정 (chi-square, dof=categories-1)
  </p>
  <table style="border-collapse:collapse;font-size:13px">
    <thead>
      <tr style="background:#f1f5f9">
        <th style="padding:6px 12px;text-align:left">attribute</th>
        <th style="padding:6px 12px;text-align:right">χ²</th>
        <th style="padding:6px 12px;text-align:right">p-value</th>
        <th style="padding:6px 12px">sig</th>
        <th style="padding:6px 12px;text-align:right">dof</th>
      </tr>
    </thead>
    <tbody>{chi2_rows}</tbody>
  </table>
  <p style="font-size:12px;color:#9ca3af;margin-top:8px">*** p&lt;0.001 &nbsp;** p&lt;0.01 &nbsp;* p&lt;0.05 &nbsp;ns not significant</p>
</section>

<footer style="font-size:12px;color:#9ca3af;margin-top:32px;padding-top:16px;
               border-top:1px solid #e5e7eb">
  Generated by experiments/audio_encoder_probe/phase_a/make_diagnostic_html.py
</footer>
</body>
</html>"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--combined_csv", type=Path,
        default=PRED_DIR / "combined_whisper_small_dacvae_prevq_nsynth_train_30k_phaseA.csv",
    )
    p.add_argument(
        "--meta_csv", type=Path,
        default=PROBE_ROOT / "manifests" / "nsynth_train_30k_meta.csv",
    )
    p.add_argument(
        "--chi2_csv", type=Path,
        default=PROBE_ROOT / "_results" / "phase_a_prevq" / "chi_square_results.csv",
    )
    p.add_argument(
        "--out_html", type=Path,
        default=PRED_DIR / "set_diagnostic_prevq.html",
    )
    args = p.parse_args(argv)

    combined = pd.read_csv(args.combined_csv)
    meta = pd.read_csv(args.meta_csv)
    chi2_df = pd.read_csv(args.chi2_csv)

    df = combined.merge(meta[["utt_id", "source", "pitch", "velocity"]], on="utt_id", how="left")
    df["whisper_small_correct"] = df["whisper_small_correct"].astype(str).str.lower().isin(("true", "1"))
    df["dacvae_correct"] = df["dacvae_correct"].astype(str).str.lower().isin(("true", "1"))

    html = build_html(df, len(df), chi2_df)
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out_html}")
    print(f"Serve: cd {args.out_html.parent.parent.parent.parent} && python -m http.server 8765")
    print(f"Open:  http://localhost:8765/{args.out_html.relative_to(args.out_html.parent.parent.parent.parent)}")


if __name__ == "__main__":
    main()
