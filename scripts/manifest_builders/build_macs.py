"""Build MACS manifest. Audio is NOT in MACS Zenodo (only yaml).

MACS files come from TAU Urban Acoustic Scenes 2019 Development (Zenodo 2589280).
Strategy:
    1. Read MACS.yaml → set of target filenames (3930 files: airport/park/public_square).
    2. For each TAU audio zip (1..21):
         - Download to /mnt/tmp/datasets/env_sound/MACS_dl/audio_N.zip
         - For each entry whose basename is in MACS targets and not yet extracted:
              extract bytes → /mnt/tmp/datasets/laion_extracted/macs/<basename>
         - Delete zip after.
    3. Build manifest from yaml: one row per file with `captions` = list of all annotators' sentences.
       Skip files where audio is missing.

Output:
    /mnt/tmp/datasets/laion_extracted/macs/<filename>.wav
    /mnt/tmp/datasets/manifests/v3/macs_<NNNN>.jsonl

Captions: keep all 2-5 annotator sentences per clip.
"""
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

MACS_YAML = Path("/mnt/tmp/datasets/env_sound/MACS/MACS.yaml")
DL_DIR = Path("/mnt/tmp/datasets/env_sound/MACS_dl")
AUDIO_OUT = Path("/mnt/tmp/datasets/laion_extracted/macs")
MANIFEST_OUT = Path("/mnt/tmp/datasets/manifests/v3")
SHARD_ROWS = 15000

ZIP_URL_FMT = (
    "https://zenodo.org/api/records/2589280/files/"
    "TAU-urban-acoustic-scenes-2019-development.audio.{n}.zip/content"
)
N_ZIPS = 21

DL_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_OUT.mkdir(parents=True, exist_ok=True)
MANIFEST_OUT.mkdir(parents=True, exist_ok=True)


def load_targets() -> dict[str, list[str]]:
    """Returns {basename.wav: [caption1, caption2, ...]}."""
    with open(MACS_YAML) as f:
        d = yaml.safe_load(f)
    out: dict[str, list[str]] = {}
    for entry in d["files"]:
        fname = entry["filename"]
        caps = []
        for a in entry.get("annotations", []):
            s = a.get("sentence", "").strip()
            if s:
                caps.append(s)
        if caps:
            out[fname] = caps
    return out


def download(url: str, dst: Path) -> None:
    print(f"[macs] downloading {url} → {dst}", flush=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[macs] already exists {dst.stat().st_size}", flush=True)
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        chunk = 1 << 20
        n = 0
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            n += len(buf)
            if n % (256 * (1 << 20)) < chunk:
                print(f"  ... {n / (1 << 30):.2f} GB", flush=True)
    tmp.rename(dst)
    print(f"[macs] downloaded {dst.stat().st_size / (1 << 30):.2f} GB", flush=True)


def extract_targets(zip_path: Path, targets: set[str]) -> int:
    """Extract entries whose basename is in targets (and not already extracted). Returns count."""
    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            base = Path(info.filename).name
            if base not in targets:
                continue
            out_path = AUDIO_OUT / base
            if out_path.exists() and out_path.stat().st_size > 0:
                continue
            with zf.open(info) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            extracted += 1
    return extracted


def fetch_audio(targets: set[str]) -> None:
    for n in range(1, N_ZIPS + 1):
        zip_path = DL_DIR / f"audio_{n}.zip"
        if not zip_path.exists() or zip_path.stat().st_size == 0:
            download(ZIP_URL_FMT.format(n=n), zip_path)
        try:
            cnt = extract_targets(zip_path, targets)
        except zipfile.BadZipFile as e:
            print(f"[macs] bad zip {zip_path}: {e} — re-download", flush=True)
            zip_path.unlink(missing_ok=True)
            download(ZIP_URL_FMT.format(n=n), zip_path)
            cnt = extract_targets(zip_path, targets)
        # check coverage
        have = sum(1 for t in targets if (AUDIO_OUT / t).exists())
        print(f"[macs] zip {n}: extracted {cnt}, total have {have}/{len(targets)}", flush=True)
        zip_path.unlink(missing_ok=True)
        if have >= len(targets):
            print("[macs] all targets covered — stopping early", flush=True)
            break


def build_manifest(targets_dict: dict[str, list[str]]) -> None:
    rows: list[dict] = []
    n_missing = 0
    for fname, caps in targets_dict.items():
        audio_path = AUDIO_OUT / fname
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            n_missing += 1
            continue
        rows.append({
            "modality": "audio_env_sound",
            "source": "macs",
            "audio_path": str(audio_path),
            "captions": caps,
        })

    print(f"[macs] manifest rows={len(rows)} missing_audio={n_missing}", flush=True)

    n_shards = 0
    for i in range(0, len(rows), SHARD_ROWS):
        chunk = rows[i:i + SHARD_ROWS]
        out = MANIFEST_OUT / f"macs_{n_shards:04d}.jsonl"
        with open(out, "w") as f:
            for r in chunk:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_shards += 1
    print(f"[macs] wrote {n_shards} shards", flush=True)
    if rows:
        print(f"[macs] sample: {json.dumps(rows[0], ensure_ascii=False)}", flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    targets_dict = load_targets()
    print(f"[macs] target files: {len(targets_dict)}", flush=True)
    targets_set = set(targets_dict.keys())

    if mode in ("all", "fetch"):
        fetch_audio(targets_set)
    if mode in ("all", "manifest"):
        build_manifest(targets_dict)


if __name__ == "__main__":
    main()
