"""Download Audiostock samples (LAION csv) via audiostock.jp redirect.

CSV row has `audiostock.net/audio/<id>/play` — broken. Swap to .jp variant
which 302-redirects to working cloudfront sample mp3 (~25 KB each, 128 kbps).

10 001 rows × ~25 KB ≈ 250 MB total. Concurrency 8, with backoff.
"""
import csv, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests

CSV = "/mnt/tmp/datasets/laion_csvs/Audiostock.csv"
OUT = Path("/mnt/tmp/datasets/laion_audiostock/audio")
META_OUT = Path("/mnt/tmp/datasets/laion_audiostock/meta.csv")
OUT.mkdir(parents=True, exist_ok=True)
META_OUT.parent.mkdir(parents=True, exist_ok=True)

ID_RE = re.compile(r"audiostock\.\w+/audio/(\d+)/play")

def fix_url(u: str) -> tuple[str, str] | None:
    m = ID_RE.search(u or "")
    if not m: return None
    aid = m.group(1)
    return aid, f"https://audiostock.jp/audio/{aid}/play"

def fetch(row: dict) -> dict:
    fixed = fix_url(row.get("url", ""))
    if not fixed:
        return {"id": "?", "status": "BAD_URL"}
    aid, url = fixed
    out_path = OUT / f"{aid}.mp3"
    if out_path.exists() and out_path.stat().st_size > 1024:
        return {"id": aid, "status": "skip", "size": out_path.stat().st_size}
    try:
        r = requests.get(url, allow_redirects=True, timeout=30, stream=True)
        if r.status_code != 200:
            return {"id": aid, "status": f"HTTP_{r.status_code}"}
        out_path.write_bytes(r.content)
        return {"id": aid, "status": "ok", "size": len(r.content),
                "caption": row.get("caption1", ""), "url": url}
    except Exception as e:
        return {"id": aid, "status": f"ERR:{type(e).__name__}"}

def main():
    rows = list(csv.DictReader(open(CSV)))
    print(f"[audiostock] {len(rows)} rows", flush=True)

    meta_writer = csv.DictWriter(open(META_OUT, "w"),
                                  fieldnames=["id", "status", "size", "caption", "url"])
    meta_writer.writeheader()

    n_ok = n_skip = n_err = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(fetch, r) for r in rows]
        for i, f in enumerate(as_completed(futs)):
            res = f.result()
            meta_writer.writerow(res)
            if res["status"] == "ok": n_ok += 1
            elif res["status"] == "skip": n_skip += 1
            else: n_err += 1
            if i % 200 == 0:
                dt = time.time() - t0
                rate = (i+1) / max(dt, 1e-6)
                eta = (len(rows) - i - 1) / max(rate, 1e-6)
                print(f"[audiostock] {i+1}/{len(rows)}  ok={n_ok} skip={n_skip} err={n_err}  "
                      f"{rate:.1f} req/s  eta {eta/60:.1f}m", flush=True)
    print(f"[audiostock] DONE: ok={n_ok} skip={n_skip} err={n_err}", flush=True)

if __name__ == "__main__":
    main()
