"""Filter qwen3_5_dacvae_asr_shuffled_128 shards to libri/mls/voxpopuli only."""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "datasets" / "qwen3_5_dacvae_asr_shuffled_128"
DST = REPO / "datasets" / "libri_mls_vox"
KEEP = ("/mls/", "/en_LibriTTS_R_single/", "/voxpopuli/")


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    grand_kept = grand_total = 0
    for src in sorted(SRC.glob("shard_*.jsonl")):
        dst = DST / src.name
        kept = total = 0
        with src.open() as fin, dst.open("w") as fout:
            for line in fin:
                total += 1
                if any(k in line for k in KEEP):
                    fout.write(line)
                    kept += 1
        grand_kept += kept
        grand_total += total
        print(f"{src.name}: {kept}/{total}")
    print(f"TOTAL: {grand_kept}/{grand_total}")


if __name__ == "__main__":
    main()
