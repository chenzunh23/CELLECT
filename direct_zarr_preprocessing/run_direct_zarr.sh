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
VARIANT_LSST_BACKGROUND_ROOT="${VARIANT_LSST_BACKGROUND_ROOT:-${COADD_ROOT}/lsst_background_masks}"
IMAGE_VARIANT_BACKGROUND_SOURCE="${IMAGE_VARIANT_BACKGROUND_SOURCE:-auto}"

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
NONCOADD_SNR_FILTER="${NONCOADD_SNR_FILTER:-1}"
NONCOADD_SNR_USE_SOURCE_MASK="${NONCOADD_SNR_USE_SOURCE_MASK:-1}"
NONCOADD_SNR_USE_QUALITY_MASK="${NONCOADD_SNR_USE_QUALITY_MASK:-1}"
NONCOADD_SNR_EXCLUDE_SELF_SOURCE="${NONCOADD_SNR_EXCLUDE_SELF_SOURCE:-1}"
NONCOADD_SNR_MASK_PLANES="${NONCOADD_SNR_MASK_PLANES:-BRIGHT_OBJECT SAT BAD NO_DATA EDGE UNMASKEDNAN}"

B_MAG_MIN="${B_MAG_MIN:-15}"
B_MAG_MAX="${B_MAG_MAX:-35}"
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
  --tile-workers "${TILE_WORKERS}"
  --patch-workers "${PATCH_WORKERS}"
  --worker-threads "${WORKER_THREADS}"
  --chunk-tiles "${CHUNK_TILES}"
  --image-variant-background-source "${IMAGE_VARIANT_BACKGROUND_SOURCE}"
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
if [[ -n "${VARIANT_LSST_BACKGROUND_ROOT}" && -d "${VARIANT_LSST_BACKGROUND_ROOT}" ]]; then
  args+=(--variant-lsst-background-root "${VARIANT_LSST_BACKGROUND_ROOT}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi
if [[ "${INCLUDE_COADD}" == "0" ]]; then
  args+=(--no-include-coadd)
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

echo "[direct-zarr] output=${OUTPUT_ROOT} patches=${PATCHES} bands=${BANDS}"
"${args[@]}"
