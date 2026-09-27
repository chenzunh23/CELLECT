#!/usr/bin/env bash
set -euo pipefail
cd /home/czh23/CELLECT
PYTHON=${PYTHON:-/home/czh23/miniconda3/envs/cellect/bin/python}
# New output tree: old singleton stores must not coexist with packed parents.
OUT=${OUT:-/data/czh23/analysis/2026-09/2026-09-26/training_zarr_psfee_parent20_v2}
BACKGROUND_ROOT=${BACKGROUND_ROOT:-/data/czh23/analysis/2026-09/2026-09-26/batch_background_grow6}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$PYTHON" -u -m preprocessing.build_image_level_zarr \
  --training-batch --training-kinds hsc_half hsc_noisy abell cosmos \
  --background-root "$BACKGROUND_ROOT" --output-root "$OUT" \
  --workers "${WORKERS:-8}" --confidence-mode psf-ee --chunk-tiles 20 "$@"
