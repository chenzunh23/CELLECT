#!/usr/bin/env bash
# Reusable mixed training Zarr recipe. Reuses completed backgrounds; does not
# regenerate them. Default: HSC half + noisy + COSMOS + Abell, using backgrounds
# from each actual training image. Noisy masks must be generated first.
#
# DRY_RUN=1 bash run_generate_zarr.sh              # print command, no writes
# CHECK_ONLY=1 bash run_generate_zarr.sh           # metadata check, no Zarr
# ZARR_WORKERS=8 bash run_generate_zarr.sh         # generate / resume
# OUT=/path/to/new/output bash run_generate_zarr.sh
# CLI selectors are forwarded, e.g. --training-kinds abell --jwst-bands F444W
# --abell-parent x+00_y+00. Use a separate OUT for such pilot runs.
# Existing matching receipts are reused; --overwrite explicitly rebuilds jobs.
# Add only noisy to an existing mixed root:
# TRAINING_KINDS=hsc_noisy OUT=/existing/root bash run_generate_zarr.sh --append
# Weak shape: confidence region=1, shape reliability controlled by training
# --center-only-shape-factor (default .2). Confidence-region weights come from
# preprocessing/labels.py and are printed by preflight (STAR_WINGS currently .2).
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"
PYTHON=${PYTHON:-/home/czh23/miniconda3/envs/cellect/bin/python}
BASE=${BASE:-/data/czh23/analysis/$(date +%Y-%m/%Y-%m-%d)}
OUT=${OUT:-$BASE/training_zarr_gaia_star_wings}
BACKGROUND_ROOT=${BACKGROUND_ROOT:-/data/czh23/analysis/2026-10/2026-10-19/background_sextractor15_half}
HSC_PRODUCTS_ROOT=${HSC_PRODUCTS_ROOT:-/data/czh23/Subaru_products}
ZARR_WORKERS=${ZARR_WORKERS:-8}
CHUNK_TILES=${CHUNK_TILES:-20}
read -r -a kinds <<< "${TRAINING_KINDS:-hsc_half hsc_noisy abell cosmos}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
args=(--training-batch --training-kinds "${kinds[@]}"
  --background-root "$BACKGROUND_ROOT"
  --hsc-products-root "$HSC_PRODUCTS_ROOT" --hsc-background-method sextractor-precomputed
  --output-root "$OUT" --workers "$ZARR_WORKERS"
  --confidence-mode psf-ee --tile-size 512 --stride 368 --chunk-tiles "$CHUNK_TILES"
  --clip-threshold 5 --bright-threshold 5 --bright-dilation 2
  --image-scaling-mode zscore-log-lupton-rgb --image-scaling-scope patch
  --bright-mask-mode log-lupton --jwst-max-invalid-fraction 0.10)
command=("$PYTHON" -u -m preprocessing.build_image_level_zarr "${args[@]}" "$@")
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${command[@]}"; printf '\n'
  exit 0
fi
# Resolve the actual final CLI output path (including a forwarded --output-root)
# before opening the lock or logs. This also rejects malformed CLI arguments.
OUT=$("$PYTHON" -c 'from preprocessing.build_image_level_zarr import parse_args; import sys; from pathlib import Path; print(Path(parse_args(sys.argv[1:]).output_root).resolve())' "${args[@]}" "$@")
mkdir -p "$OUT/logs"
exec 9>"$OUT/logs/generation.lock"
flock -n 9 || { echo "Zarr generation already running in $OUT" >&2; exit 1; }
RUN_ID=$(date +%Y%m%d_%H%M%S)_$$
printf '%q ' "${command[@]}" > "$OUT/logs/${RUN_ID}_command.txt"
printf '\n' >> "$OUT/logs/${RUN_ID}_command.txt"
"$PYTHON" -u -m preprocessing.zarr_preflight \
  --report "$OUT/logs/${RUN_ID}_preflight.json" "${args[@]}" "$@" \
  2>&1 | tee "$OUT/logs/${RUN_ID}_preflight.log"
if [[ ${CHECK_ONLY:-0} == 1 ]]; then exit 0; fi
"${command[@]}" 2>&1 | tee "$OUT/logs/${RUN_ID}_zarr.log"
