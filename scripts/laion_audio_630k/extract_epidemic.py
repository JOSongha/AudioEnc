"""Extract LAION Epidemic CLAPv2 parquets → flat <id>.flac files + caption jsonl.

Source: /mnt/tmp/datasets/laion_epidemic_clapv2/data/{train,test}/*.parquet
Schema per row:
  index: string  (e.g. "mnt_epidemic_sound_effects_split_train_10001")
  text: string   (T5 debiased caption)
  raw_text: list<string>  (Title:..., Id:..., Added:...)
  audio: struct{bytes: binary (FLAC), path: string}
  audio_len: float

Output:
  /mnt/tmp/datasets/laion_extracted/epidemic/<safe_index>.flac
  /mnt/tmp/datasets/manifests/v3/laion_epidemic_<split>_<NNNN>.jsonl
"""
import json, re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path("/mnt/tmp/datasets/laion_epidemic_clapv2/data")
OUT_AUDIO = Path("/mnt/tmp/datasets/laion_extracted/epidemic")
OUT_MANIFEST = Path("/mnt/tmp/datasets/manifests/v3")
OUT_AUDIO.mkdir(parents=True, exist_ok=True)
OUT_MANIFEST.mkdir(parents=True, exist_ok=True)

SHARD_ROWS = 15000
SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def safe_name(idx: str) -> str:
    return SAFE_RE.sub("_", idx)[:200]


def process_parquet(pq_path: Path) -> tuple[int, list[dict]]:
    rows = []
    n = 0
    try:
        t = pq.read_table(pq_path)
        df = t.to_pandas()
        for _, r in df.iterrows():
            idx = r["index"]
            if not isinstance(idx, str) or not idx:
                continue
            txt = r["text"]
            if not isinstance(txt, str) or not txt.strip():
                continue
            audio = r["audio"]
            if isinstance(audio, dict):
                audio_bytes = audio.get("bytes")
            else:
                audio_bytes = None
            if not audio_bytes:
                continue
            stem = safe_name(idx)
            out_flac = OUT_AUDIO / f"{stem}.flac"
            if not out_flac.exists() or out_flac.stat().st_size == 0:
                out_flac.write_bytes(audio_bytes)
            rows.append({
                "modality": "audio_env_sound",
                "source": "laion_epidemic",
                "audio_path": str(out_flac),
                "captions": [txt.strip()],
            })
            n += 1
    except Exception as e:
        print(f"[epidemic] parquet {pq_path.name} ERR: {e}", flush=True)
    return n, rows


def write_shards(rows: list[dict], split: str) -> int:
    n = 0
    for i in range(0, len(rows), SHARD_ROWS):
        chunk = rows[i:i + SHARD_ROWS]
        out = OUT_MANIFEST / f"laion_epidemic_{split}_{n:04d}.jsonl"
        with open(out, "w") as f:
            for row in chunk:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
    return n


def main():
    splits = sorted([d for d in ROOT.iterdir() if d.is_dir()])
    overall = 0
    for sd in splits:
        split = sd.name
        parqs = sorted(sd.glob("*.parquet"))
        if not parqs:
            continue
        print(f"[epidemic] {split}: {len(parqs)} parquets", flush=True)
        all_rows = []
        with ProcessPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(process_parquet, p): p for p in parqs}
            done = 0
            for fut in as_completed(futs):
                n, rows = fut.result()
                all_rows.extend(rows)
                done += 1
                if done % 100 == 0:
                    print(f"[epidemic] {split} {done}/{len(parqs)} parquets  rows={len(all_rows)}", flush=True)
        n_shards = write_shards(all_rows, split)
        overall += len(all_rows)
        print(f"[epidemic] {split} done: {len(all_rows)} clips → {n_shards} shards", flush=True)
    print(f"[epidemic] TOTAL: {overall} clips", flush=True)


if __name__ == "__main__":
    main()
