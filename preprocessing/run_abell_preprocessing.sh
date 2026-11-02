#!/usr/bin/env bash
set -euo pipefail
# Parse the complete function before launching long-running commands, so an
# on-disk edit cannot change the commands read after a pipeline finishes.
main() {
run_started=$SECONDS
stage=initialization
progress() { printf '[%s] [abell-run] %s\n' "$(date --iso-8601=seconds)" "$*"; }
finish() {
  local code=$?
  if (( code == 0 )); then
    progress "END status=success elapsed=$((SECONDS-run_started))s"
  else
    progress "END status=failed exit=$code stage=$stage elapsed=$((SECONDS-run_started))s"
  fi
}
trap finish EXIT
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
JWST_PYTHON="${JWST_PYTHON:-/home/czh23/miniconda3/envs/jwst_env/bin/python}"
CUTOUT_WORKERS="${CUTOUT_WORKERS:-2}"
ZARR_WORKERS="${ZARR_WORKERS:-4}"
BACKGROUND_ROOT="${BACKGROUND_ROOT:-/data/czh23/analysis/2026-10/2026-10-19/background_sextractor15_half}"
ABELL_STAGE="${ABELL_STAGE:-all}"
case "$ABELL_STAGE" in
  all|cutouts|zarr) ;;
  *) printf 'Invalid ABELL_STAGE=%s (expected all/cutouts/zarr)\n' "$ABELL_STAGE" >&2; exit 2 ;;
esac
# Training-only opt-in. USE_UNCOVER=0 retains the foundation catalog/full grid.
# These paths do not change eval/hsctiles coadd discovery.
USE_UNCOVER="${USE_UNCOVER:-1}"
if [[ "$USE_UNCOVER" == 1 ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT:-/data/czh23/JWST/Abell2744_preprocessing_uncover}"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-/data/czh23/JWST/Abell2744_preprocessing}"
fi
UNCOVER_CATALOG="${UNCOVER_CATALOG:-/home/czh23/fits/UNCOVER_DR3_LW_SUPER_catalog.fits}"
UNCOVER_DETECTION="${UNCOVER_DETECTION:-/data/czh23/JWST/UNCOVER/UNCOVER_DR3_DETECTION.fits}"
UNCOVER_SEGMENTATION="${UNCOVER_SEGMENTATION:-/data/czh23/JWST/UNCOVER/UNCOVER_DR3_SEGMAP.fits}"
UNCOVER_MODEL_ROOT="${UNCOVER_MODEL_ROOT:-/data/czh23/JWST/UNCOVER/models}"
UNCOVER_NO_KRON_SNR="${UNCOVER_NO_KRON_SNR:-3}"
UNCOVER_RISK_NO_KRON_SNR="${UNCOVER_RISK_NO_KRON_SNR:-5}"
UNCOVER_RISK_BRIGHT_MAG="${UNCOVER_RISK_BRIGHT_MAG:-22}"
UNCOVER_RISK_SIGMA="${UNCOVER_RISK_SIGMA:-3}"
UNCOVER_REJECT_UNTRUSTED="${UNCOVER_REJECT_UNTRUSTED:-1}"
JWST_MAX_INVALID_FRACTION="${JWST_MAX_INVALID_FRACTION:-0.10}"
uncover_args=(--no-use-uncover)
if [[ "$USE_UNCOVER" == 1 ]]; then
  uncover_args=(--use-uncover --uncover-catalog "$UNCOVER_CATALOG"
    --uncover-detection "$UNCOVER_DETECTION" --uncover-segmentation "$UNCOVER_SEGMENTATION"
    --uncover-model-root "$UNCOVER_MODEL_ROOT" --uncover-no-kron-snr "$UNCOVER_NO_KRON_SNR"
    --uncover-risk-no-kron-snr "$UNCOVER_RISK_NO_KRON_SNR" --uncover-risk-bright-mag "$UNCOVER_RISK_BRIGHT_MAG"
    --uncover-risk-sigma "$UNCOVER_RISK_SIGMA")
  if [[ "$UNCOVER_REJECT_UNTRUSTED" == 0 ]]; then
    uncover_args+=(--no-uncover-reject-untrusted)
  fi
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
progress "START output=$OUTPUT_ROOT stage=$ABELL_STAGE use_uncover=$USE_UNCOVER cutout_workers=$CUTOUT_WORKERS zarr_workers=$ZARR_WORKERS"
mkdir -p "$OUTPUT_ROOT/logs"
# Prevent two schedulers from writing the same grid/stores concurrently.
exec 9>"$OUTPUT_ROOT/.scheduler.lock"
flock -n 9 || { echo 'Another Abell scheduler is running'; exit 1; }
cd "$REPO_ROOT"
# Arrays preserve the complete argv when options are edited across lines.
# A missing shell continuation must never silently drop catalog/worker flags.
common_args=(--datasets abell --output-root "$OUTPUT_ROOT"
  --jwst-max-invalid-fraction "$JWST_MAX_INVALID_FRACTION" "${uncover_args[@]}")
if [[ "$ABELL_STAGE" == all || "$ABELL_STAGE" == cutouts ]]; then
stage=cutouts
progress 'START stage=cutouts (align HALF/FULL FITS; Zarr starts after all cutout tasks complete)'
command=("$JWST_PYTHON" -u -m preprocessing.build_image_level_zarr
  "${common_args[@]}" --abell-stage cutouts --workers "$CUTOUT_WORKERS" "$@")
printf '[abell-run] COMMAND '; printf '%q ' "${command[@]}"; printf '\n'
"${command[@]}" 2>&1 | tee -a "$OUTPUT_ROOT/logs/cutouts.log"
progress "END stage=cutouts elapsed=$((SECONDS-run_started))s"
fi
if [[ "$ABELL_STAGE" == all || "$ABELL_STAGE" == zarr ]]; then
stage=zarr
stage_started=$SECONDS
progress 'START stage=zarr'
command=("$JWST_PYTHON" -u -m preprocessing.build_image_level_zarr
  "${common_args[@]}" --abell-stage zarr --confidence-mode psf-ee
  --background-root "$BACKGROUND_ROOT" --workers "$ZARR_WORKERS" "$@")
printf '[abell-run] COMMAND '; printf '%q ' "${command[@]}"; printf '\n'
"${command[@]}" 2>&1 | tee -a "$OUTPUT_ROOT/logs/zarr.log"
progress "END stage=zarr elapsed=$((SECONDS-stage_started))s"
fi
stage=complete
}
main "$@"
