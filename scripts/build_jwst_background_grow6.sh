#!/usr/bin/env bash
set -euo pipefail
cd /home/czh23/CELLECT
PYTHON=${PYTHON:-/home/czh23/miniconda3/envs/cellect/bin/python}
BACKGROUND_ROOT=${BACKGROUND_ROOT:-/data/czh23/analysis/2026-09/2026-09-26/batch_background_grow6}
OLD_BACKGROUND_ROOT=${OLD_BACKGROUND_ROOT:-/data/czh23/analysis/2026-09/2026-09-25/batch_aggressive_background}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# HSC noisy keeps its existing background. HSC half is generated on demand by Zarr.
# Only the JWST source-exclusion growth is changed in this background generation.
if [[ ! -e "$BACKGROUND_ROOT/hsc" && ! -L "$BACKGROUND_ROOT/hsc" ]]; then
  [[ -d "$OLD_BACKGROUND_ROOT/hsc" ]] || { echo "Missing HSC backgrounds: $OLD_BACKGROUND_ROOT/hsc" >&2; exit 1; }
  mkdir -p "$BACKGROUND_ROOT"
  ln -s "$OLD_BACKGROUND_ROOT/hsc" "$BACKGROUND_ROOT/hsc"
fi
exec "$PYTHON" -u /home/czh23/analysis/2026-09/2026-09-25/batch_aggressive_background/batch_background.py \
  --datasets abell cosmos --output-root "$BACKGROUND_ROOT" --workers "${WORKERS:-16}" \
  --jwst-grow 6 16 32 --jwst-thresholds 1.8 2.2 2.7 "$@"
