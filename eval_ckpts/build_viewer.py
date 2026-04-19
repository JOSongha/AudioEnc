"""
build_viewer.py — eval_all_ckpts.py 결과 디렉토리에서 self-contained HTML 뷰어 생성.

Usage:
    python eval_ckpts/build_viewer.py eval_ckpts/results/<run_tag>/<split>/

생성: <results_dir>/index.html  (브라우저에서 바로 열기 가능)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HTML_TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>ASR ckpt comparison — __TITLE__</title>
<style>
  body {
    font-family: -apple-system, "SF Pro Text", "Helvetica Neue", sans-serif;
    margin: 0; padding: 24px;
    background: #0f1115; color: #d8dee9;
    font-size: 14px;
  }
  h1 { margin: 0 0 4px 0; font-size: 18px; font-weight: 600; }
  .meta { color: #88909e; margin-bottom: 18px; font-size: 12px; }
  .layout { display: grid; grid-template-columns: 280px 1fr; gap: 18px; }
  .sidebar {
    background: #161922; border-radius: 8px; padding: 12px;
    max-height: calc(100vh - 100px); overflow-y: auto;
  }
  .sidebar h3 { margin: 4px 0 10px 0; font-size: 12px; color: #a8b2c1;
                text-transform: uppercase; letter-spacing: .5px; }
  .ckpt-row {
    padding: 8px 10px; border-radius: 6px; cursor: pointer;
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 4px; user-select: none; gap: 8px;
  }
  .ckpt-row:hover { background: #1f2533; }
  .ckpt-row.active { background: #2a3550; color: #fff; }
  .ckpt-name { font-family: ui-monospace, monospace; font-size: 12px; }
  .ckpt-wer { font-weight: 600; font-size: 12px; }
  .wer-best { color: #a3e635; }
  .wer-bad { color: #f87171; }
  .main {
    background: #161922; border-radius: 8px; padding: 16px;
    max-height: calc(100vh - 100px); overflow-y: auto;
  }
  .main-header {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 14px; flex-wrap: wrap; gap: 8px;
  }
  .main-header h2 { margin: 0; font-size: 15px; font-family: ui-monospace, monospace; }
  .main-header .stats { color: #88909e; font-size: 12px; }
  .filter-bar {
    display: flex; gap: 8px; margin-bottom: 12px; align-items: center;
    flex-wrap: wrap;
  }
  .filter-bar input[type=text] {
    flex: 1; min-width: 180px;
    background: #0f1115; border: 1px solid #2a3041; color: #d8dee9;
    padding: 6px 10px; border-radius: 5px; font-family: inherit; font-size: 12px;
  }
  .filter-bar label { font-size: 12px; color: #a8b2c1; cursor: pointer; }
  .filter-bar select {
    background: #0f1115; border: 1px solid #2a3041; color: #d8dee9;
    padding: 6px 8px; border-radius: 5px; font-size: 12px;
  }
  table { width: 100%; border-collapse: collapse; }
  th, td {
    padding: 8px 10px; vertical-align: top; text-align: left;
    border-bottom: 1px solid #1f2533;
  }
  th { font-size: 11px; text-transform: uppercase; letter-spacing: .5px;
       color: #88909e; font-weight: 500; position: sticky; top: 0;
       background: #161922; }
  td.idx { color: #88909e; font-family: ui-monospace, monospace;
           width: 50px; font-size: 11px; }
  td.text { font-family: ui-monospace, monospace; font-size: 12px;
            line-height: 1.6; }
  .word { padding: 1px 2px; border-radius: 3px; }
  .w-match  { color: #d8dee9; }
  .w-sub    { background: #4a3a14; color: #fcd34d; }
  .w-del    { background: #4a1414; color: #fca5a5; text-decoration: line-through; }
  .w-ins    { background: #14401a; color: #86efac; }
  .row-perfect td.text { opacity: .55; }
  .badge {
    display: inline-block; padding: 2px 6px; border-radius: 4px;
    font-size: 10px; margin-left: 6px;
  }
  .badge.err { background: #4a1414; color: #fca5a5; }
  .badge.ok  { background: #14401a; color: #86efac; }
  .empty { color: #66707f; font-style: italic; padding: 30px; text-align: center; }
  .legend { font-size: 11px; color: #88909e; margin-top: 10px; }
  .legend .word { font-size: 11px; }
</style>
</head>
<body>
  <h1>ASR checkpoint comparison</h1>
  <div class="meta">
    <span id="meta-run"></span> · <span id="meta-split"></span> ·
    <span id="meta-samples"></span> samples · <span id="meta-encoder"></span>
  </div>
  <div class="layout">
    <div class="sidebar">
      <h3>Checkpoints (sorted)</h3>
      <div id="ckpt-list"></div>
    </div>
    <div class="main">
      <div class="main-header">
        <h2 id="active-ckpt">—</h2>
        <div class="stats" id="active-stats"></div>
      </div>
      <div class="filter-bar">
        <input id="search" type="text" placeholder="search ref/hyp...">
        <select id="filter-mode">
          <option value="all">all</option>
          <option value="err">errors only</option>
          <option value="perfect">perfect only</option>
        </select>
        <select id="sort-mode">
          <option value="idx">sort: index</option>
          <option value="errs">sort: most errors</option>
        </select>
      </div>
      <div class="legend">
        <span class="word w-match">match</span>
        <span class="word w-sub">substitution</span>
        <span class="word w-del">deletion (in ref)</span>
        <span class="word w-ins">insertion (in hyp)</span>
      </div>
      <div id="content"></div>
    </div>
  </div>

<script>
const DATA = __DATA__;

function normalize(s) {
  return (s || "").toLowerCase()
    .replace(/[.,!?;:"'`()\\[\\]{}]/g, " ")
    .replace(/\\s+/g, " ").trim();
}

function tokens(s) { return normalize(s).split(" ").filter(Boolean); }

// Word-level edit distance with backtrace → tagged ref/hyp arrays
function diff(refS, hypS) {
  const r = tokens(refS), h = tokens(hypS);
  const m = r.length, n = h.length;
  const dp = Array.from({length: m+1}, () => new Int32Array(n+1));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++) {
      if (r[i-1] === h[j-1]) dp[i][j] = dp[i-1][j-1];
      else dp[i][j] = 1 + Math.min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]);
    }
  // backtrace
  const refOut = [], hypOut = [];
  let i = m, j = n, errs = 0;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && r[i-1] === h[j-1]) {
      refOut.push({w: r[i-1], k: "match"});
      hypOut.push({w: h[j-1], k: "match"});
      i--; j--;
    } else if (i > 0 && j > 0 && dp[i][j] === dp[i-1][j-1] + 1) {
      refOut.push({w: r[i-1], k: "sub"});
      hypOut.push({w: h[j-1], k: "sub"});
      i--; j--; errs++;
    } else if (i > 0 && dp[i][j] === dp[i-1][j] + 1) {
      refOut.push({w: r[i-1], k: "del"});
      i--; errs++;
    } else {
      hypOut.push({w: h[j-1], k: "ins"});
      j--; errs++;
    }
  }
  refOut.reverse(); hypOut.reverse();
  return {refOut, hypOut, errs, ref_len: m};
}

function renderTagged(arr) {
  return arr.map(t => `<span class="word w-${t.k}">${escapeHtml(t.w)}</span>`).join(" ");
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

let activeCkpt = null;
let activePairs = [];    // [{idx, ref, hyp, tagged...}]
let bestWer = Infinity;
let worstWer = -Infinity;

function init() {
  const sum = DATA.summary;
  document.getElementById("meta-run").textContent = sum.run_dir.split("/").pop();
  document.getElementById("meta-split").textContent = sum.split;
  document.getElementById("meta-samples").textContent = sum.n_samples;
  document.getElementById("meta-encoder").textContent = sum.encoder;

  for (const c of sum.checkpoints) {
    if (c.wer < bestWer) bestWer = c.wer;
    if (c.wer > worstWer) worstWer = c.wer;
  }

  const list = document.getElementById("ckpt-list");
  for (const c of sum.checkpoints) {
    const div = document.createElement("div");
    div.className = "ckpt-row";
    div.dataset.ckpt = c.ckpt;
    const cls = c.wer === bestWer ? "wer-best" : (c.wer === worstWer ? "wer-bad" : "");
    div.innerHTML = `
      <div>
        <div class="ckpt-name">${escapeHtml(c.ckpt)}</div>
        <div style="font-size:10px;color:#66707f;">${escapeHtml(c.saved_at || "")}</div>
      </div>
      <span class="ckpt-wer ${cls}">${(c.wer*100).toFixed(2)}%</span>`;
    div.addEventListener("click", () => selectCkpt(c.ckpt));
    list.appendChild(div);
  }

  document.getElementById("search").addEventListener("input", render);
  document.getElementById("filter-mode").addEventListener("change", render);
  document.getElementById("sort-mode").addEventListener("change", render);

  if (sum.checkpoints.length) selectCkpt(sum.checkpoints[0].ckpt);
}

function selectCkpt(name) {
  activeCkpt = name;
  document.querySelectorAll(".ckpt-row").forEach(r => {
    r.classList.toggle("active", r.dataset.ckpt === name);
  });
  const ck = DATA.checkpoints[name];
  if (!ck) {
    document.getElementById("content").innerHTML =
      '<div class="empty">No data for this checkpoint.</div>';
    return;
  }
  activePairs = ck.pairs.map(p => {
    const d = diff(p.ref, p.hyp);
    return {...p, ...d};
  });
  document.getElementById("active-ckpt").textContent = name;
  document.getElementById("active-stats").textContent =
    `WER ${(ck.wer*100).toFixed(2)}%  ·  ${ck.n_samples} samples  ·  ${ck.elapsed_sec.toFixed(1)}s`;
  render();
}

function render() {
  if (!activePairs.length) return;
  const q = document.getElementById("search").value.toLowerCase().trim();
  const mode = document.getElementById("filter-mode").value;
  const sort = document.getElementById("sort-mode").value;

  let rows = activePairs.slice();
  if (q) rows = rows.filter(r =>
    r.ref.toLowerCase().includes(q) || r.hyp.toLowerCase().includes(q));
  if (mode === "err") rows = rows.filter(r => r.errs > 0);
  if (mode === "perfect") rows = rows.filter(r => r.errs === 0);
  if (sort === "errs") rows.sort((a, b) => b.errs - a.errs);

  if (!rows.length) {
    document.getElementById("content").innerHTML =
      '<div class="empty">No rows match the current filters.</div>';
    return;
  }

  const html = ['<table>',
    '<thead><tr><th>idx</th><th>reference / hypothesis</th></tr></thead><tbody>'];
  for (const r of rows) {
    const badge = r.errs === 0
      ? '<span class="badge ok">0 err</span>'
      : `<span class="badge err">${r.errs} err / ${r.ref_len}w</span>`;
    const cls = r.errs === 0 ? "row-perfect" : "";
    html.push(`<tr class="${cls}">
      <td class="idx">${r.idx}${badge}</td>
      <td class="text">
        <div>${renderTagged(r.refOut)}</div>
        <div style="margin-top:4px;">${renderTagged(r.hypOut)}</div>
      </td></tr>`);
  }
  html.push('</tbody></table>');
  document.getElementById("content").innerHTML = html.join("");
}

init();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path,
                        help="eval_all_ckpts 결과 디렉토리 (summary.json 포함)")
    parser.add_argument("--out", type=Path, default=None,
                        help="출력 HTML 경로 (기본: <results_dir>/index.html)")
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        print(f"summary.json not found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    checkpoints = {}
    for entry in summary["checkpoints"]:
        json_path = results_dir / entry["json"]
        if not json_path.exists():
            print(f"  skip (missing): {json_path}", file=sys.stderr)
            continue
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        checkpoints[entry["ckpt"]] = {
            "wer": data["wer"],
            "n_samples": data["n_samples"],
            "elapsed_sec": data["elapsed_sec"],
            "pairs": data["pairs"],
        }

    payload = {"summary": summary, "checkpoints": checkpoints}
    title = f"{Path(summary['run_dir']).name} / {summary['split']}"
    html = HTML_TEMPLATE.replace("__TITLE__", title)
    html = html.replace("__DATA__", json.dumps(payload, ensure_ascii=False))

    out_path = args.out or (results_dir / "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    n_ck = len(checkpoints)
    n_pairs = sum(len(c["pairs"]) for c in checkpoints.values())
    print(f"Wrote {out_path}")
    print(f"  {n_ck} checkpoints, {n_pairs} total ref/hyp pairs")
    print(f"  open in browser: file://{out_path.resolve()}")


if __name__ == "__main__":
    main()
