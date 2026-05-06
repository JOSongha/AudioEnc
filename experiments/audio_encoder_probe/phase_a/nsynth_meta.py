"""NSynth utt_id -> meta-attribute parser and manifest augmenter."""

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# NSynth utt_id format: {family}_{source}_{id}-{pitch}-{velocity}
# - source is constrained to 3 enum values, which anchors the split even when
#   family contains underscores (e.g. "synth_lead")
UTT_RE = re.compile(
    r"^(?P<family>[a-z_]+)_(?P<source>acoustic|electronic|synthetic)_"
    r"(?P<inst_id>\d+)-(?P<pitch>\d+)-(?P<velocity>\d+)$"
)


def parse_nsynth_utt_id(utt_id: str) -> dict[str, Any]:
    """Parse one NSynth utt_id into structured meta.

    Args:
        utt_id: E.g. "bass_electronic_034-061-100" or "synth_lead_synthetic_001-060-127".

    Returns:
        Dict with keys: family (str), source (str), inst_id (int), pitch (int), velocity (int).

    Raises:
        ValueError: If utt_id does not match the NSynth format.
    """
    m = UTT_RE.match(utt_id)
    if m is None:
        raise ValueError(f"utt_id does not match NSynth format: {utt_id!r}")
    return {
        "family": m["family"],
        "source": m["source"],
        "inst_id": int(m["inst_id"]),
        "pitch": int(m["pitch"]),
        "velocity": int(m["velocity"]),
    }


def augment_manifest(manifest_csv: Path, out_csv: Path) -> pd.DataFrame:
    """Read manifest, parse utt_id -> 5 meta cols, write augmented CSV.

    Args:
        manifest_csv: Input CSV with at least column ``utt_id``.
        out_csv: Output CSV path. Parent dirs created if missing.

    Returns:
        The augmented DataFrame (also written to disk).
    """
    df = pd.read_csv(manifest_csv)
    if "utt_id" not in df.columns:
        raise ValueError(f"manifest missing 'utt_id' column; got {list(df.columns)}")

    meta_records = [parse_nsynth_utt_id(uid) for uid in df["utt_id"]]
    meta_df = pd.DataFrame.from_records(meta_records)
    out_df = pd.concat([df.reset_index(drop=True), meta_df], axis=1)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    logger.info("Wrote %d rows x %d cols -> %s", len(out_df), len(out_df.columns), out_csv)
    return out_df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    args = p.parse_args()
    augment_manifest(args.manifest, args.out_csv)


if __name__ == "__main__":
    main()
