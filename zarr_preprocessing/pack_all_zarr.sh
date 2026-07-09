#!/usr/bin/env bash
set -euo pipefail

# Pack existing CELLECT legacy preprocessing outputs into patch-level Zarr v2 stores.
# This script does not rerun catalog filtering; it reuses band_targets,
# band_tile_metadata, and precomputed zscale cache.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-cellect}"
PYTHON="${PYTHON:-/home/czh23/miniconda3/envs/${CONDA_ENV}/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="conda run -n ${CONDA_ENV} python"
fi

TRACT="${TRACT:-9813}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-/nvme0/zc/scarlet/preprocessed}"
ZSCALE_ROOT="${ZSCALE_ROOT:-/nvme0/zc/scarlet/cellect_zscale_cache}"
ZARR_ROOT="${ZARR_ROOT:-/nvme0/zc/scarlet/preprocessed_zarr}"

PATCHES="${PATCHES:-all}"
BANDS="${BANDS:-HSC-G HSC-R HSC-I HSC-Z HSC-Y}"
IMAGE_VARIANTS="${IMAGE_VARIANTS:-denoised noisy}"
PACK_COADD="${PACK_COADD:-1}"

PATCH_WORKERS="${PATCH_WORKERS:-4}"
ZARR_TILE_WORKERS="${ZARR_TILE_WORKERS:-4}"
CHUNK_TILES="${CHUNK_TILES:-8}"
IMAGE_DTYPE="${IMAGE_DTYPE:-float16}"
TARGET_FLOAT_DTYPE="${TARGET_FLOAT_DTYPE:-float16}"
INCLUDE_SHAPE="${INCLUDE_SHAPE:-0}"
FITS_HDU="${FITS_HDU:-1}"
OVERWRITE_ZARR="${OVERWRITE_ZARR:-0}"
TILE_FILTER="${TILE_FILTER:-}"

patch_list() {
  if [[ "${PATCHES}" == "all" ]]; then
    for r in {0..8}; do
      for c in {0..8}; do
        printf '%s,%s\n' "${r}" "${c}"
      done
    done
  else
    for patch in ${PATCHES}; do
      printf '%s\n' "${patch}"
    done
  fi
}

throttle_jobs() {
  local limit="$1"
  while [[ "$(jobs -rp | wc -l)" -ge "${limit}" ]]; do
    wait -n
  done
}

pack_one() {
  local dataset_source="$1"
  local patch="$2"
  local patch_root="$3"
  local output="$4"

  if [[ ! -d "${patch_root}" ]]; then
    echo "[zarr-all] skip missing ${dataset_source} ${TRACT}/${patch}: ${patch_root}"
    return 0
  fi
  if [[ -d "${output}" && "${OVERWRITE_ZARR}" != "1" ]]; then
    echo "[zarr-all] skip existing ${dataset_source} ${TRACT}/${patch}: ${output}"
    return 0
  fi

  mkdir -p "$(dirname "${output}")"
  local cmd=(
    ${PYTHON}
    "${ROOT_DIR}/zarr_preprocessing/pack_patch_zarr.py"
    --patch-root "${patch_root}"
    --zscale-root "${ZSCALE_ROOT}"
    --output "${output}"
    --tract "${TRACT}"
    --patch "${patch}"
    --dataset-source "${dataset_source}"
    --bands ${BANDS}
    --fits-hdu "${FITS_HDU}"
    --image-dtype "${IMAGE_DTYPE}"
    --target-float-dtype "${TARGET_FLOAT_DTYPE}"
    --chunk-tiles "${CHUNK_TILES}"
    --workers "${ZARR_TILE_WORKERS}"
  )
  if [[ "${INCLUDE_SHAPE}" == "1" ]]; then
    cmd+=(--include-shape)
  fi
  if [[ "${OVERWRITE_ZARR}" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  if [[ -n "${TILE_FILTER}" ]]; then
    cmd+=(--tile-filter ${TILE_FILTER})
  fi

  echo "[zarr-all] pack ${dataset_source} ${TRACT}/${patch} -> ${output}"
  "${cmd[@]}"
}

main() {
  mkdir -p "${ZARR_ROOT}"
  local failed=0
  for patch in $(patch_list); do
    if [[ "${PACK_COADD}" == "1" ]]; then
      throttle_jobs "${PATCH_WORKERS}"
      pack_one \
        "coadd" \
        "${patch}" \
        "${PREPROCESSED_ROOT}/${TRACT}/${patch}" \
        "${ZARR_ROOT}/coadd/${TRACT}/${patch}.zarr" &
    fi
    for variant in ${IMAGE_VARIANTS}; do
      throttle_jobs "${PATCH_WORKERS}"
      pack_one \
        "${variant}" \
        "${patch}" \
        "${PREPROCESSED_ROOT}/${variant}/${TRACT}/${patch}" \
        "${ZARR_ROOT}/${variant}/${TRACT}/${patch}.zarr" &
    done
  done
  for job in $(jobs -rp); do
    if ! wait "${job}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "[zarr-all] one or more packing jobs failed" >&2
    return 1
  fi
}

main "$@"
