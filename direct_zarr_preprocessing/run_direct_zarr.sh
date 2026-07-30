#!/usr/bin/env bash
set -euo pipefail

# Direct preprocessing to Zarr. This does not create legacy cutout/target/catalog
# folders. It writes patch-level Zarr stores directly from FITS images, catalogs,
# refit CSVs, and LSST det footprints.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-cellect}"
PYTHON="${PYTHON:-/home/czh23/miniconda3/envs/${CONDA_ENV}/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="conda run -n ${CONDA_ENV} python"
fi

DATA_ROOT="${DATA_ROOT:-/nvme0/zc/scarlet}"
COADD_ROOT="${COADD_ROOT:-${DATA_ROOT}}"
CATALOG_ROOT="${CATALOG_ROOT:-${DATA_ROOT}}"
BAND_CATALOG_ROOT="${BAND_CATALOG_ROOT:-${CATALOG_ROOT}}"
DENOISED_FITS_ROOT="${DENOISED_FITS_ROOT-${DATA_ROOT}/denoised_fits}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/direct_zarr}"
REFIT_ROOT="${REFIT_ROOT:-${DATA_ROOT}/refit}"
VARIANT_LSST_BACKGROUND_ROOT="${VARIANT_LSST_BACKGROUND_ROOT:-${OUTPUT_ROOT}/variant_lsst_background}"
VARIANT_LSST_BACKGROUND_POLICY="${VARIANT_LSST_BACKGROUND_POLICY:-run-if-missing}"
IMAGE_VARIANT_BACKGROUND_SOURCE="${IMAGE_VARIANT_BACKGROUND_SOURCE:-variant-lsst}"
LSST_DETECT_PYTHON="${LSST_DETECT_PYTHON:-}"

TRACT="${TRACT:-9813}"
PATCHES="${PATCHES:-all}"
BANDS="${BANDS:-HSC-G HSC-R HSC-I HSC-Z HSC-Y}"
IMAGE_VARIANTS="${IMAGE_VARIANTS:-denoised noisy}"
IMAGE_VARIANT_GROUPS="${IMAGE_VARIANT_GROUPS:-}"

PATCH_WORKERS="${PATCH_WORKERS:-1}"
TILE_WORKERS="${TILE_WORKERS:-2}"
CHUNK_TILES="${CHUNK_TILES:-8}"
WORKER_THREADS="${WORKER_THREADS:-1}"
OVERWRITE="${OVERWRITE:-0}"
INCLUDE_COADD="${INCLUDE_COADD:-1}"
MISSING_BAND_POLICY="${MISSING_BAND_POLICY:-skip}"
WRITE_IMAGE_LEVEL_ZARR="${WRITE_IMAGE_LEVEL_ZARR:-0}"
IMAGE_LEVEL_ONLY="${IMAGE_LEVEL_ONLY:-0}"
QUALITY_FILTER="${QUALITY_FILTER:-0}"
QUALITY_BAD_SCORE_THRESHOLD="${QUALITY_BAD_SCORE_THRESHOLD:-0.13}"
QUALITY_BAD_SCORE_WEIGHTS="${QUALITY_BAD_SCORE_WEIGHTS:-}"
QUALITY_FILTER_MISSING_POLICY="${QUALITY_FILTER_MISSING_POLICY:-keep}"
NONCOADD_SNR_FILTER="${NONCOADD_SNR_FILTER:-1}"
NONCOADD_SNR_USE_SOURCE_MASK="${NONCOADD_SNR_USE_SOURCE_MASK:-1}"
NONCOADD_SNR_USE_QUALITY_MASK="${NONCOADD_SNR_USE_QUALITY_MASK:-1}"
NONCOADD_SNR_EXCLUDE_SELF_SOURCE="${NONCOADD_SNR_EXCLUDE_SELF_SOURCE:-1}"
NONCOADD_SNR_MASK_PLANES="${NONCOADD_SNR_MASK_PLANES:-BRIGHT_OBJECT SAT BAD NO_DATA EDGE UNMASKEDNAN}"
ALIGN_DENOISED_NOISY_SNR_LABELS="${ALIGN_DENOISED_NOISY_SNR_LABELS:-1}"
ENABLE_BRIGHT_BACKGROUND_MASK="${ENABLE_BRIGHT_BACKGROUND_MASK:-0}"
PU_BRIGHT_MASK_MODE="${PU_BRIGHT_MASK_MODE:-log-lupton}"
PU_BRIGHT_LOG_A="${PU_BRIGHT_LOG_A:-300}"
PU_BRIGHT_LOG_HIGH_PERCENTILE="${PU_BRIGHT_LOG_HIGH_PERCENTILE:-99.5}"
PU_BRIGHT_LUPTON_STRETCH="${PU_BRIGHT_LUPTON_STRETCH:-0.5}"
PU_BRIGHT_LUPTON_Q="${PU_BRIGHT_LUPTON_Q:-20}"
PU_BRIGHT_ANSCOMBE_SCALE="${PU_BRIGHT_ANSCOMBE_SCALE:-1000}"
PU_BRIGHT_Z_THRESHOLD="${PU_BRIGHT_Z_THRESHOLD:-3.0}"
PU_BRIGHT_MASK_DILATE="${PU_BRIGHT_MASK_DILATE:-2}"
EXTERNAL_BRIGHT_LABEL_ROOT="${EXTERNAL_BRIGHT_LABEL_ROOT:-}"
IMAGE_SCALING_MODE="${IMAGE_SCALING_MODE:-astro-zscore}"
IMAGE_LOG_A="${IMAGE_LOG_A:-300}"
IMAGE_LOG_HIGH_PERCENTILE="${IMAGE_LOG_HIGH_PERCENTILE:-99.5}"
IMAGE_LUPTON_STRETCH="${IMAGE_LUPTON_STRETCH:-0.5}"
IMAGE_LUPTON_Q="${IMAGE_LUPTON_Q:-20}"
IMAGE_ANSCOMBE_SCALE="${IMAGE_ANSCOMBE_SCALE:-1000}"

B_MAG_MIN="${B_MAG_MIN:-15}"
B_MAG_MAX="${B_MAG_MAX:-35}"
CENTER_ONLY_FILL_AREA_MIN="${CENTER_ONLY_FILL_AREA_MIN:-500}"
CENTER_ONLY_FILL_RATIO_MAX="${CENTER_ONLY_FILL_RATIO_MAX:-0.3}"
AP2_KRON_BRIGHT_MAG_THRESHOLD="${AP2_KRON_BRIGHT_MAG_THRESHOLD:-22}"
AP2_KRON_BRIGHT_ABS_MAX="${AP2_KRON_BRIGHT_ABS_MAX:-2}"
AP2_KRON_LARGE_BRIGHT_REGION_AREA_MIN="${AP2_KRON_LARGE_BRIGHT_REGION_AREA_MIN:-1000}"
MAG_MIN="${MAG_MIN:-15}"
MAG_MAX="${MAG_MAX:-35}"

args=(
  "${PYTHON}" "${ROOT_DIR}/direct_zarr_preprocessing/direct_preprocess_zarr.py"
  --coadd-root "${COADD_ROOT}"
  --catalog-root "${CATALOG_ROOT}"
  --band-catalog-root "${BAND_CATALOG_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --tract "${TRACT}"
  --patches ${PATCHES}
  --bands ${BANDS}
  --kron-refit-csv "${REFIT_ROOT}/{tract}/{band}/{patch}/batch_heavyfp_kron_refit/batch_heavyfp_kron_refit.csv"
  --b-mag-min "${B_MAG_MIN}"
  --b-mag-max "${B_MAG_MAX}"
  --center-only-fill-area-min "${CENTER_ONLY_FILL_AREA_MIN}"
  --center-only-fill-ratio-max "${CENTER_ONLY_FILL_RATIO_MAX}"
  --ap2-kron-bright-mag-threshold "${AP2_KRON_BRIGHT_MAG_THRESHOLD}"
  --ap2-kron-bright-abs-max "${AP2_KRON_BRIGHT_ABS_MAX}"
  --ap2-kron-large-bright-region-area-min "${AP2_KRON_LARGE_BRIGHT_REGION_AREA_MIN}"
  --tile-workers "${TILE_WORKERS}"
  --patch-workers "${PATCH_WORKERS}"
  --worker-threads "${WORKER_THREADS}"
  --chunk-tiles "${CHUNK_TILES}"
  --missing-band-policy "${MISSING_BAND_POLICY}"
  --image-variant-background-source "${IMAGE_VARIANT_BACKGROUND_SOURCE}"
  --variant-lsst-background-policy "${VARIANT_LSST_BACKGROUND_POLICY}"
  --quality-bad-score-threshold "${QUALITY_BAD_SCORE_THRESHOLD}"
  --quality-filter-missing-policy "${QUALITY_FILTER_MISSING_POLICY}"
  --image-scaling-mode "${IMAGE_SCALING_MODE}"
  --image-log-a "${IMAGE_LOG_A}"
  --image-log-high-percentile "${IMAGE_LOG_HIGH_PERCENTILE}"
  --image-lupton-stretch "${IMAGE_LUPTON_STRETCH}"
  --image-lupton-q "${IMAGE_LUPTON_Q}"
  --image-anscombe-scale "${IMAGE_ANSCOMBE_SCALE}"
)

if [[ -n "${DENOISED_FITS_ROOT}" ]]; then
  args+=(--denoised-fits-root "${DENOISED_FITS_ROOT}")
fi
if [[ -n "${IMAGE_VARIANTS}" ]]; then
  args+=(--image-variants ${IMAGE_VARIANTS})
fi
if [[ -n "${IMAGE_VARIANT_GROUPS}" ]]; then
  args+=(--image-variant-groups ${IMAGE_VARIANT_GROUPS})
fi
if [[ -n "${VARIANT_LSST_BACKGROUND_ROOT}" ]]; then
  args+=(--variant-lsst-background-root "${VARIANT_LSST_BACKGROUND_ROOT}")
fi
if [[ -n "${LSST_DETECT_PYTHON}" ]]; then
  args+=(--lsst-detect-python "${LSST_DETECT_PYTHON}")
fi
if [[ "${ENABLE_BRIGHT_BACKGROUND_MASK}" == "1" ]]; then
  args+=(
    --pu-enable-bright-background-mask
    --pu-bright-mask-mode "${PU_BRIGHT_MASK_MODE}"
    --pu-bright-log-a "${PU_BRIGHT_LOG_A}"
    --pu-bright-log-high-percentile "${PU_BRIGHT_LOG_HIGH_PERCENTILE}"
    --pu-bright-lupton-stretch "${PU_BRIGHT_LUPTON_STRETCH}"
    --pu-bright-lupton-q "${PU_BRIGHT_LUPTON_Q}"
    --pu-bright-anscombe-scale "${PU_BRIGHT_ANSCOMBE_SCALE}"
    --pu-bright-z-threshold "${PU_BRIGHT_Z_THRESHOLD}"
    --pu-bright-mask-dilate "${PU_BRIGHT_MASK_DILATE}"
  )
fi
if [[ -n "${EXTERNAL_BRIGHT_LABEL_ROOT}" ]]; then
  args+=(--external-bright-label-root "${EXTERNAL_BRIGHT_LABEL_ROOT}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi
if [[ "${INCLUDE_COADD}" == "0" ]]; then
  args+=(--no-include-coadd)
fi
if [[ "${WRITE_IMAGE_LEVEL_ZARR}" == "1" ]]; then
  args+=(--write-image-level-zarr)
fi
if [[ "${IMAGE_LEVEL_ONLY}" == "1" ]]; then
  args+=(--image-level-only)
fi
if [[ "${QUALITY_FILTER}" == "1" ]]; then
  args+=(--quality-filter)
fi
if [[ -n "${QUALITY_BAD_SCORE_WEIGHTS}" ]]; then
  # shellcheck disable=SC2206
  quality_weight_args=(${QUALITY_BAD_SCORE_WEIGHTS})
  args+=(--quality-bad-score-weights "${quality_weight_args[@]}")
fi
if [[ "${NONCOADD_SNR_FILTER}" == "0" ]]; then
  args+=(--no-noncoadd-snr-filter)
fi
if [[ "${NONCOADD_SNR_USE_SOURCE_MASK}" == "0" ]]; then
  args+=(--no-noncoadd-snr-use-source-mask)
fi
if [[ "${NONCOADD_SNR_USE_QUALITY_MASK}" == "0" ]]; then
  args+=(--no-noncoadd-snr-use-quality-mask)
fi
if [[ "${NONCOADD_SNR_EXCLUDE_SELF_SOURCE}" == "0" ]]; then
  args+=(--no-noncoadd-snr-exclude-self-source)
fi
if [[ "${ALIGN_DENOISED_NOISY_SNR_LABELS}" == "1" ]]; then
  args+=(--align-denoised-noisy-snr-labels)
fi
if [[ -n "${NONCOADD_SNR_MASK_PLANES}" ]]; then
  # shellcheck disable=SC2206
  mask_plane_args=(${NONCOADD_SNR_MASK_PLANES})
  args+=(--noncoadd-snr-mask-planes "${mask_plane_args[@]}")
fi
if [[ -n "${COMPARE_ORIGIN:-}" ]]; then
  args+=(--compare-origin ${COMPARE_ORIGIN})
fi
if [[ -n "${TILE_FILTER:-}" ]]; then
  args+=(--tile-filter ${TILE_FILTER})
fi
if [[ -n "${MAX_TILES:-}" ]]; then
  args+=(--max-tiles "${MAX_TILES}")
fi

echo "[direct-zarr] output=${OUTPUT_ROOT} patches=${PATCHES} bands=${BANDS} image_level=${WRITE_IMAGE_LEVEL_ZARR} quality_filter=${QUALITY_FILTER} image_scaling=${IMAGE_SCALING_MODE}"
"${args[@]}"
