#!/usr/bin/env python3
"""Per-utt JSON → split JSONL → Arrow IPC"""
import json, sys, time
from pathlib import Path
import pyarrow as pa
import pyarrow.json as pa_json
import pyarrow.ipc as pa_ipc

DATASETS = [
    {
        "name": "librispeech",
        "base": Path("/mnt/fr20tb/wbl_residency/jos/AudioEnc/log/tmp/cache/word_alignments"),
        "splits": ["dev-clean", "train-clean-100", "train-clean-360", "train-other-500"],
    },
    {
        "name": "mls",
        "base": Path("/mnt/tmp/cache/word_alignments_qwen3/mls"),
        "splits": ["train"],
    },
    {
        "name": "gigaspeech",
        "base": Path("/mnt/tmp/cache/word_alignments_qwen3/gigaspeech"),
        "splits": ["train"],
    },
    {
        "name": "voxpopuli",
        "base": Path("/mnt/tmp/cache/word_alignments_qwen3/voxpopuli"),
        "splits": ["train"],
    },
]

OUT_BASE = Path("/mnt/tmp/cache/word_alignments_merged")

def merge_to_jsonl(split_dir: Path, jsonl_path: Path) -> int:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(split_dir.rglob("*.json"))
    count = 0
    t0 = time.time()
    with open(jsonl_path, "w") as fout:
        for jf in files:
            try:
                fout.write(jf.read_text().rstrip() + "\n")
                count += 1
                if count % 200_000 == 0:
                    elapsed = time.time() - t0
                    print(f"  {count:,} ({elapsed:.0f}s, {count/elapsed:.0f}/s)", flush=True)
            except Exception as e:
                print(f"  WARNING {jf}: {e}", flush=True)
    return count

def jsonl_to_arrow(jsonl_path: Path, arrow_path: Path) -> int:
    """청크 단위로 JSONL 읽어서 Arrow IPC로 저장 (max rows 에러 방지)."""
    import io
    print(f"  Arrow 변환 중...", flush=True)
    t0 = time.time()
    CHUNK = 100_000
    tables = []
    buf = []
    total = 0
    with open(jsonl_path) as f:
        for line in f:
            buf.append(line)
            if len(buf) >= CHUNK:
                tables.append(pa_json.read_json(io.BytesIO("".join(buf).encode())))
                total += len(buf)
                buf = []
                if total % 500_000 == 0:
                    print(f"  {total:,} rows 처리 중...", flush=True)
    if buf:
        tables.append(pa_json.read_json(io.BytesIO("".join(buf).encode())))
        total += len(buf)
    table = pa.concat_tables(tables)
    with pa_ipc.new_file(str(arrow_path), table.schema) as writer:
        writer.write_table(table, max_chunksize=100_000)
    print(f"  {len(table):,} rows → {arrow_path} ({time.time()-t0:.1f}s)", flush=True)
    return len(table)

for ds in DATASETS:
    for split in ds["splits"]:
        split_dir = ds["base"] / split
        if not split_dir.exists():
            print(f"[SKIP] {ds['name']}/{split}: 디렉토리 없음")
            continue

        out_dir = OUT_BASE / ds["name"]
        jsonl_path = out_dir / f"{split}.jsonl"
        arrow_path = out_dir / f"{split}.arrow"

        print(f"\n{'='*50}")
        if arrow_path.exists():
            print(f"[SKIP] {ds['name']}/{split}: arrow 이미 존재")
            continue

        if not jsonl_path.exists():
            print(f"[{ds['name']}/{split}] JSONL 병합 중...", flush=True)
            t0 = time.time()
            n = merge_to_jsonl(split_dir, jsonl_path)
            print(f"  {n:,} 항목 완료 ({time.time()-t0:.1f}s) → {jsonl_path}", flush=True)
        else:
            print(f"[{ds['name']}/{split}] JSONL 이미 존재, Arrow 변환만 실행")

        jsonl_to_arrow(jsonl_path, arrow_path)

print("\n\n모두 완료!")
