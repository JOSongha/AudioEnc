#!/usr/bin/env bash
# Download CMU Arctic for the speakers we need. ~7 GB total.
# Each speaker is a separate tarball at the festvox mirror.
set -euo pipefail

ARCTIC_ROOT="${ARCTIC_ROOT:-/mnt/tmp/datasets/cmu_arctic}"
mkdir -p "${ARCTIC_ROOT}"
cd "${ARCTIC_ROOT}"

SPEAKERS=(bdl clb slt rms ksp awb jmk)
BASE_URL="http://festvox.org/cmu_arctic/cmu_arctic/packed"

for spk in "${SPEAKERS[@]}"; do
  pkg="cmu_us_${spk}_arctic-0.95-release.tar.bz2"
  dir="cmu_us_${spk}_arctic"
  if [ -d "${dir}" ]; then
    echo "[arctic] ${spk}: already extracted, skipping"
    continue
  fi
  if [ ! -f "${pkg}" ]; then
    echo "[arctic] ${spk}: downloading"
    wget -q --show-progress "${BASE_URL}/${pkg}"
  fi
  echo "[arctic] ${spk}: extracting"
  tar xjf "${pkg}"
done

echo "[arctic] DONE  root=${ARCTIC_ROOT}"
ls -d cmu_us_*_arctic
