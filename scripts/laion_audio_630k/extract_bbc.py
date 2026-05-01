"""Extract LAION BBC tars → flat <id>.flac files + caption jsonl.

Source: /mnt/tmp/datasets/laion_bbc_hf/{train,test}/{N}.tar
Output:
    /mnt/tmp/datasets/laion_extracted/bbc/<id>.flac
    /mnt/tmp/datasets/manifests/v3/laion_bbc_<split>_<NNNN>.jsonl

Each tar holds pairs `mnt/audio_clip/processed_datasets/BBCSoundEffects/<split>/<id>.flac` + `<id>.json`.
JSON has single `text` field (caption).
"""
import json, tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/mnt/tmp/datasets/laion_bbc_hf")
OUT_AUDIO = Path("/mnt/tmp/datasets/laion_extracted/bbc")
OUT_MANIFEST = Path("/mnt/tmp/datasets/manifests/v3")
OUT_AUDIO.mkdir(parents=True, exist_ok=True)
OUT_MANIFEST.mkdir(parents=True, exist_ok=True)

SHARD_ROWS = 15000


def extract_tar(tar_path: Path) -> tuple[int, list[dict]]:
    rows = []
    n_clips = 0
    try:
        with tarfile.open(tar_path, mode="r") as tar:
            members = tar.getmembers()
            json_members = {m.name: m for m in members if m.name.endswith(".json")}
            flac_members = {m.name: m for m in members if m.name.endswith(".flac")}
            for jname, jmember in json_members.items():
                fname = jname[:-5] + ".flac"
                fmember = flac_members.get(fname)
                if not fmember:
                    continue
                clip_id = Path(jname).stem
                out_flac = OUT_AUDIO / f"{clip_id}.flac"
                if not out_flac.exists() or out_flac.stat().st_size == 0:
                    fe = tar.extractfile(fmember)
                    if fe is None:
                        continue
                    out_flac.write_bytes(fe.read())
                je = tar.extractfile(jmember)
                if je is None:
                    continue
                meta = json.loads(je.read().decode("utf-8", errors="replace"))
                cap = meta.get("text")
                if isinstance(cap, list):
                    cap = cap[0] if cap else None
                if not cap or not str(cap).strip():
                    continue
                rows.append({
                    "modality": "audio_env_sound",
                    "source": "laion_bbc",
                    "audio_path": str(out_flac),
                    "captions": [str(cap).strip()],
                })
                n_clips += 1
    except Exception as e:
        print(f"[bbc] tar {tar_path.name} ERR: {e}", flush=True)
    return n_clips, rows


def write_shards(rows: list[dict], split: str) -> int:
    n_shards = 0
    for i in range(0, len(rows), SHARD_ROWS):
        chunk = rows[i:i + SHARD_ROWS]
        out = OUT_MANIFEST / f"laion_bbc_{split}_{n_shards:04d}.jsonl"
        with open(out, "w") as f:
            for row in chunk:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_shards += 1
    return n_shards


def main():
    splits = sorted([d for d in ROOT.iterdir() if d.is_dir()])
    overall = 0
    for sd in splits:
        split = sd.name
        tars = sorted(sd.glob("*.tar"))
        if not tars:
            continue
        print(f"[bbc] {split}: {len(tars)} tars", flush=True)
        all_rows = []
        with ProcessPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(extract_tar, t): t for t in tars}
            done = 0
            for f in as_completed(futs):
                n, rows = f.result()
                all_rows.extend(rows)
                done += 1
                if done % 10 == 0:
                    print(f"[bbc] {split} {done}/{len(tars)} tars  rows={len(all_rows)}", flush=True)
        n_shards = write_shards(all_rows, split)
        overall += len(all_rows)
        print(f"[bbc] {split} done: {len(all_rows)} clips → {n_shards} shards", flush=True)
    print(f"[bbc] TOTAL: {overall} clips", flush=True)


if __name__ == "__main__":
    main()
