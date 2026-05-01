"""Extract LAION freesound_no_overlap tars → flat <id>.flac files + caption jsonl.

Source: /mnt/tmp/datasets/laion_freesound/freesound_no_overlap/{train_1,train_2,test}/{N}.tar
Output:
    /mnt/tmp/datasets/laion_extracted/freesound/<id>.flac
    /mnt/tmp/datasets/manifests/v3/laion_freesound_<split>_<NNNN>.jsonl

Each tar holds pairs `mnt/freesound/split/<split>/<id>.flac` + `<id>.json`.
JSON has `text` (list 1-2 caps), `tag`, `original_data`.
"""
import json, os, tarfile, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/mnt/tmp/datasets/laion_freesound/freesound_no_overlap")
OUT_AUDIO = Path("/mnt/tmp/datasets/laion_extracted/freesound")
OUT_MANIFEST = Path("/mnt/tmp/datasets/manifests/v3")
OUT_AUDIO.mkdir(parents=True, exist_ok=True)
OUT_MANIFEST.mkdir(parents=True, exist_ok=True)

SHARD_ROWS = 15000  # rows per output jsonl shard


def extract_tar(tar_path: Path) -> tuple[int, list[dict]]:
    """Extract one tar, return (n_clips, list of manifest rows)."""
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
                    f_extract = tar.extractfile(fmember)
                    if f_extract is None:
                        continue
                    out_flac.write_bytes(f_extract.read())
                j_extract = tar.extractfile(jmember)
                if j_extract is None:
                    continue
                meta = json.loads(j_extract.read().decode("utf-8", errors="replace"))
                caps = meta.get("text") or []
                if isinstance(caps, str):
                    caps = [caps]
                caps = [c.strip() for c in caps if c and c.strip()]
                if not caps:
                    continue
                rows.append({
                    "modality": "audio_env_sound",
                    "source": "laion_freesound",
                    "audio_path": str(out_flac),
                    "captions": caps,
                })
                n_clips += 1
    except Exception as e:
        print(f"[freesound] tar {tar_path.name} ERR: {e}", flush=True)
    return n_clips, rows


def write_shards(rows: list[dict], split: str) -> int:
    """Write rows into <split>_<NNNN>.jsonl shards of size SHARD_ROWS."""
    n_shards = 0
    for i in range(0, len(rows), SHARD_ROWS):
        chunk = rows[i:i + SHARD_ROWS]
        out = OUT_MANIFEST / f"laion_freesound_{split}_{n_shards:04d}.jsonl"
        with open(out, "w") as f:
            for row in chunk:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_shards += 1
    return n_shards


def main():
    splits = sorted([d for d in ROOT.iterdir() if d.is_dir()])
    overall_clips = 0
    for split_dir in splits:
        split = split_dir.name
        tars = sorted(split_dir.glob("*.tar"))
        print(f"[freesound] {split}: {len(tars)} tars", flush=True)
        all_rows = []
        with ProcessPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(extract_tar, t): t for t in tars}
            done = 0
            for f in as_completed(futs):
                n, rows = f.result()
                all_rows.extend(rows)
                done += 1
                if done % 20 == 0:
                    print(f"[freesound] {split} {done}/{len(tars)} tars  rows={len(all_rows)}", flush=True)
        n_shards = write_shards(all_rows, split)
        overall_clips += len(all_rows)
        print(f"[freesound] {split} done: {len(all_rows)} clips → {n_shards} shards", flush=True)
    print(f"[freesound] TOTAL: {overall_clips} clips", flush=True)


if __name__ == "__main__":
    main()
