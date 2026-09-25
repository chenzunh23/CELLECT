#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
JWST_PYTHON="${JWST_PYTHON:-/home/czh23/miniconda3/envs/jwst_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/czh23/JWST/Abell2744_preprocessing}"
CUTOUT_WORKERS="${CUTOUT_WORKERS:-2}"
ZARR_WORKERS="${ZARR_WORKERS:-4}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUTPUT_ROOT/logs"
# Prevent two schedulers from writing the same grid/stores concurrently.
exec 9>"$OUTPUT_ROOT/.scheduler.lock"
flock -n 9 || { echo 'Another Abell scheduler is running'; exit 1; }
cd "$REPO_ROOT"
"$JWST_PYTHON" -u -m preprocessing.build_image_level_zarr \
  --datasets abell --abell-stage cutouts --output-root "$OUTPUT_ROOT" \
  --workers "$CUTOUT_WORKERS" "$@" 2>&1 | tee -a "$OUTPUT_ROOT/logs/cutouts.log"
"$JWST_PYTHON" -u -m preprocessing.build_image_level_zarr \
  --datasets abell --abell-stage zarr --output-root "$OUTPUT_ROOT" \
  --confidence-mode psf-ee \
  --workers "$ZARR_WORKERS" "$@" 2>&1 | tee -a "$OUTPUT_ROOT/logs/zarr.log"
