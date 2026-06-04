#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/batch-heavyfp-kron-refit-mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

cd "${ROOT}"

"${PYTHON}" "bin/make_example_inputs.py"

"${PYTHON}" "batch_heavyfp_kron_refit.py" \
  --meas-catalog "examples/input/example_meas_catalog.fits" \
  --reference-image "examples/input/example_reference_image.fits" \
  --rows-file "examples/input/example_rows.txt" \
  --output-dir "examples/output" \
  --artifact-name example_run

echo "summary: examples/output/example_run/summary.md"
