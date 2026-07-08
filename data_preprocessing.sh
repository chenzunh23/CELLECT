#!/usr/bin/env bash
set -euo pipefail

# End-to-end data preprocessing for detection-only AstroCELLECT training.
#
# The script has two stages:
#   1. Optionally run batch-heavyfp-kron-refit for every patch x band in parallel.
#   2. Run astro_data_preprocessing.py once with patch-level worker processes.
#
# Override any variable from the command line, e.g.
#   PATCHES="0,0 0,1 0,2" REFIT_WORKERS=2 PREPROCESS_WORKERS=4 bash data_preprocessing.sh

TRACT="${TRACT:-9813}"
RAW_ROOT="${RAW_ROOT:-/nvme0/zc/scarlet}"
PREP_ROOT="${PREP_ROOT:-/nvme0/zc/scarlet/preprocessed}"
FAST_ROOT="${FAST_ROOT:-/nvme0/zc/scarlet/preprocessed}"
ZSCALE_ROOT="${ZSCALE_ROOT:-/nvme0/zc/scarlet/cellect_zscale_cache}"
BANDS="${BANDS:-HSC-G HSC-R HSC-I HSC-Z HSC-Y}"

PATCHES="${PATCHES:-all}"
PATCH_FILE="${PATCH_FILE:-}"

CONDA_ENV="${CONDA_ENV:-cellect}"
RUN_REFIT="${RUN_REFIT:-1}"
RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
REUSE_EXISTING_PREPROCESSED="${REUSE_EXISTING_PREPROCESSED:-0}"
REBUILD_IMAGE_VARIANTS="${REBUILD_IMAGE_VARIANTS:-0}"
SKIP_EXISTING_REFIT="${SKIP_EXISTING_REFIT:-1}"
REFIT_CSV_ONLY="${REFIT_CSV_ONLY:-1}"
COPY_REFIT_INPUTS_TO_TMP="${COPY_REFIT_INPUTS_TO_TMP:-0}"
REFIT_INCLUDE_SHAPE_FLAGGED="${REFIT_INCLUDE_SHAPE_FLAGGED:-0}"
REFIT_INCLUDE_CENTROID_FLAGGED="${REFIT_INCLUDE_CENTROID_FLAGGED:-0}"

REFIT_ROOT="${REFIT_ROOT:-/nvme0/zc/scarlet/refit}"
REFIT_WORKERS="${REFIT_WORKERS:-1}"
REFIT_OMP_THREADS="${REFIT_OMP_THREADS:-}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-8}"
VARIANT_PREPROCESS_WORKERS="${VARIANT_PREPROCESS_WORKERS:-${PREPROCESS_WORKERS}}"
PREPROCESS_WORKER_THREADS="${PREPROCESS_WORKER_THREADS:-1}"

MAG_MIN="${MAG_MIN:-15}"
MAG_MAX="${MAG_MAX:-35}"

PU_REQUIRE_KRON_REFIT_MATCH="${PU_REQUIRE_KRON_REFIT_MATCH:-1}"
PU_B_MAG_MIN="${PU_B_MAG_MIN:-18}"
PU_B_MAG_MAX="${PU_B_MAG_MAX:-30}"
USE_BAND_LIMIT_B_FILTER="${USE_BAND_LIMIT_B_FILTER:-1}"
PU_B_CLOSE_CENTER_ARCSEC="${PU_B_CLOSE_CENTER_ARCSEC:-0.5}"
PU_B_AXIS_RATIO_MAX="${PU_B_AXIS_RATIO_MAX:-5}"
PU_CONTAINMENT_THRESHOLD="${PU_CONTAINMENT_THRESHOLD:-0.80}"
PU_B_FLAGS="${PU_B_FLAGS:-base_SdssShape_flag base_SdssCentroid_flag}" #detect_isPrimary
BAD_BAND_CATALOG_POLICY="${BAD_BAND_CATALOG_POLICY:-error}"
OVERWRITE_ZSCALE="${OVERWRITE_ZSCALE:-1}"
OVERWRITE_CUTOUTS="${OVERWRITE_CUTOUTS:-0}"
SKIP_CUTOUTS="${SKIP_CUTOUTS:-0}"
WRITE_TARGET_FITS="${WRITE_TARGET_FITS:-0}"
LSST_BACKGROUND_POLICY="${LSST_BACKGROUND_POLICY:-run-if-missing}"
LSST_BACKGROUND_CACHE_ROOT="${LSST_BACKGROUND_CACHE_ROOT:-}"
VARIANT_LSST_BACKGROUND_ROOT="${VARIANT_LSST_BACKGROUND_ROOT:-}"
LSST_DETECT_PYTHON="${LSST_DETECT_PYTHON:-}"
OVERWRITE_LSST_BACKGROUND="${OVERWRITE_LSST_BACKGROUND:-0}"
WRITE_LSST_BACKGROUND_PRODUCTS="${WRITE_LSST_BACKGROUND_PRODUCTS:-0}"
LSST_BACKGROUND_DETECT_CUTOUTS="${LSST_BACKGROUND_DETECT_CUTOUTS:-0}"
USE_LSST_DETECTION_CALEXP_CUTOUTS="${USE_LSST_DETECTION_CALEXP_CUTOUTS:-0}"
DENOISED_FITS_ROOT="${DENOISED_FITS_ROOT:-}"
IMAGE_VARIANTS="${IMAGE_VARIANTS:-denoised noisy}"
IMAGE_VARIANT_GROUPS="${IMAGE_VARIANT_GROUPS:-}"
NONCOADD_SNR_FILTER="${NONCOADD_SNR_FILTER:-1}"
NONCOADD_SNR_IGNORE_THRESH="${NONCOADD_SNR_IGNORE_THRESH:-2.0}"
NONCOADD_SNR_CENTER_ONLY_THRESH="${NONCOADD_SNR_CENTER_ONLY_THRESH:-3.0}"
NONCOADD_SNR_AP_RADIUS="${NONCOADD_SNR_AP_RADIUS:-6.0}"
NONCOADD_SNR_ANNULUS_R_IN="${NONCOADD_SNR_ANNULUS_R_IN:-10.0}"
NONCOADD_SNR_ANNULUS_R_OUT="${NONCOADD_SNR_ANNULUS_R_OUT:-15.0}"
NONCOADD_SNR_SOURCE_MASK_ELLIPSE_SIGMA="${NONCOADD_SNR_SOURCE_MASK_ELLIPSE_SIGMA:-1.0}"
NONCOADD_SNR_MIN_ANNULUS_PIXELS="${NONCOADD_SNR_MIN_ANNULUS_PIXELS:-20}"
NONCOADD_SNR_MASK_PLANES="${NONCOADD_SNR_MASK_PLANES:-BRIGHT_OBJECT SAT BAD NO_DATA EDGE UNMASKEDNAN}"

BAND_LIMIT_MAGS="${BAND_LIMIT_MAGS:-HSC-G=27.4 HSC-R=27.1 HSC-I=26.9 HSC-Z=26.3 HSC-Y=25.3}"
STRICT_CENTER_ONLY_SATURATION_MAGS="${STRICT_CENTER_ONLY_SATURATION_MAGS:-${STRICT_IGNORE_SATURATION_MAGS:-HSC-G=18.0 HSC-R=18.2 HSC-I=18.6 HSC-Z=17.7 HSC-Y=17.4}}"
ENABLE_STRICT_BRIGHT_CENTER_ONLY="${ENABLE_STRICT_BRIGHT_CENTER_ONLY:-0}"
PU_AP2_KRON_ABS_MAX="${PU_AP2_KRON_ABS_MAX:-1.0}"
PU_AP2_FLUX_COLUMN="${PU_AP2_FLUX_COLUMN:-base_CircularApertureFlux_6_0_instFlux}"
PU_AP2_KRON_FLUX_COLUMN="${PU_AP2_KRON_FLUX_COLUMN:-ext_photometryKron_KronFlux_instFlux}"

WRITE_CLEAN_REGIONS="${WRITE_CLEAN_REGIONS:-0}"
CLEAN_REGION_OUT_DIR="${CLEAN_REGION_OUT_DIR:-output/preprocessed_clean_regions}"
CLEAN_REGION_PATCHES="${CLEAN_REGION_PATCHES:-${PATCHES}}"
CLEAN_REGION_BANDS="${CLEAN_REGION_BANDS:-${BANDS}}"
CLEAN_REGION_CLASSES="${CLEAN_REGION_CLASSES:-clean}"
CLEAN_REGION_TILE_NAME="${CLEAN_REGION_TILE_NAME:-}"
CLEAN_REGION_X0="${CLEAN_REGION_X0:-}"
CLEAN_REGION_Y0="${CLEAN_REGION_Y0:-}"
CLEAN_REGION_SIZE="${CLEAN_REGION_SIZE:-}"
CLEAN_REGION_WIDTH="${CLEAN_REGION_WIDTH:-}"
CLEAN_REGION_HEIGHT="${CLEAN_REGION_HEIGHT:-}"
CLEAN_REGION_MARGIN="${CLEAN_REGION_MARGIN:-0}"
CLEAN_REGION_LOCAL_COORDS="${CLEAN_REGION_LOCAL_COORDS:-0}"
CLEAN_REGION_WRITE_CLASS_FILES="${CLEAN_REGION_WRITE_CLASS_FILES:-0}"
CLEAN_REGION_AP2_KRON_ABS_MAX="${CLEAN_REGION_AP2_KRON_ABS_MAX:-}"
CLEAN_REGION_AP2_FLUX_COLUMN="${CLEAN_REGION_AP2_FLUX_COLUMN:-base_CircularApertureFlux_6_0_instFlux}"
CLEAN_REGION_KRON_FLUX_COLUMN="${CLEAN_REGION_KRON_FLUX_COLUMN:-ext_photometryKron_KronFlux_instFlux}"
CLEAN_REGION_PHOTOMETRY_ZEROPOINT="${CLEAN_REGION_PHOTOMETRY_ZEROPOINT:-27.0}"

LOG_DIR="${LOG_DIR:-output/data_preprocessing_logs}"
mkdir -p "${LOG_DIR}"

run_python() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
    python "$@"
  else
    conda run -n "${CONDA_ENV}" python "$@"
  fi
}

split_words() {
  local value="$1"
  value="${value//;/ }"
  # shellcheck disable=SC2206
  SPLIT_WORDS_OUT=(${value})
}

build_patch_list() {
  PATCH_LIST=()
  if [[ -n "${PATCH_FILE}" ]]; then
    if [[ ! -f "${PATCH_FILE}" ]]; then
      echo "ERROR: PATCH_FILE does not exist: ${PATCH_FILE}" >&2
      exit 1
    fi
    while IFS= read -r line; do
      line="${line%%#*}"
      line="${line#"${line%%[![:space:]]*}"}"
      line="${line%"${line##*[![:space:]]}"}"
      [[ -z "${line}" ]] && continue
      PATCH_LIST+=("${line}")
    done < "${PATCH_FILE}"
    return
  fi

  if [[ "${PATCHES}" == "all" ]]; then
    local x y
    for x in {0..8}; do
      for y in {0..8}; do
        PATCH_LIST+=("${x},${y}")
      done
    done
    return
  fi

  split_words "${PATCHES}"
  PATCH_LIST=("${SPLIT_WORDS_OUT[@]}")
}

throttle_jobs() {
  local max_jobs="$1"
  while [[ "$(jobs -rp | wc -l)" -ge "${max_jobs}" ]]; do
    sleep 1
  done
}

refit_csv_path() {
  local band="$1"
  local patch="$2"
  printf "%s/%s/%s/%s/batch_heavyfp_kron_refit/batch_heavyfp_kron_refit.csv" \
    "${REFIT_ROOT}" "${TRACT}" "${band}" "${patch}"
}

refit_summary_path() {
  local band="$1"
  local patch="$2"
  printf "%s/%s/%s/%s/batch_heavyfp_kron_refit/summary.json" \
    "${REFIT_ROOT}" "${TRACT}" "${band}" "${patch}"
}

run_refit_one() {
  local band="$1"
  local patch="$2"
  local meas="${RAW_ROOT}/${TRACT}/${band}/${patch}/meas-${band}-${TRACT}-${patch}.fits"
  local image="${RAW_ROOT}/${TRACT}/${band}/${patch}/calexp-${band}-${TRACT}-${patch}.fits"
  local out_dir="${REFIT_ROOT}/${TRACT}/${band}/${patch}"
  local csv_path
  csv_path="$(refit_csv_path "${band}" "${patch}")"
  local summary_path
  summary_path="$(refit_summary_path "${band}" "${patch}")"

  if [[ "${SKIP_EXISTING_REFIT}" == "1" && -s "${csv_path}" && -s "${summary_path}" ]]; then
    echo "[refit] skip existing ${band} ${patch}: ${csv_path}"
    return 0
  fi
  if [[ ! -f "${meas}" ]]; then
    echo "[refit] missing meas catalog, skip ${band} ${patch}: ${meas}" >&2
    return 2
  fi
  if [[ ! -f "${image}" ]]; then
    echo "[refit] missing reference image, skip ${band} ${patch}: ${image}" >&2
    return 2
  fi

  local mpl_config_dir="${MPLCONFIGDIR:-${LOG_DIR}/matplotlib}"
  mkdir -p "${out_dir}" "${LOG_DIR}/refit" "${mpl_config_dir}"
  echo "[refit] start ${band} ${patch}"
  local refit_meas="${meas}"
  local refit_image="${image}"
  local tmp_meas=""
  local tmp_image=""
  if [[ "${COPY_REFIT_INPUTS_TO_TMP}" == "1" ]]; then
    tmp_meas="/tmp/cellect_refit_${band}_${patch//,/_}_$(basename "${meas}")"
    tmp_image="/tmp/cellect_refit_${band}_${patch//,/_}_$(basename "${image}")"
    cp -f "${meas}" "${tmp_meas}"
    cp -f "${image}" "${tmp_image}"
    refit_meas="${tmp_meas}"
    refit_image="${tmp_image}"
  fi
  local env_args=()
  if [[ -n "${REFIT_OMP_THREADS}" ]]; then
    env_args+=("OMP_NUM_THREADS=${REFIT_OMP_THREADS}" "OMP_THREAD_LIMIT=${REFIT_OMP_THREADS}")
  fi
  env_args+=("MPLCONFIGDIR=${mpl_config_dir}")
  local python_cmd=()
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
    python_cmd=(python)
  else
    python_cmd=(conda run -n "${CONDA_ENV}" python)
  fi
  local refit_optional_args=()
  if [[ "${REFIT_CSV_ONLY}" == "1" ]]; then
    refit_optional_args+=(--csv-only)
  fi
  if [[ "${REFIT_INCLUDE_SHAPE_FLAGGED}" == "1" ]]; then
    refit_optional_args+=(--include-shape-flagged)
  fi
  if [[ "${REFIT_INCLUDE_CENTROID_FLAGGED}" == "1" ]]; then
    refit_optional_args+=(--include-centroid-flagged)
  fi
  env "${env_args[@]}" "${python_cmd[@]}" batch-heavyfp-kron-refit/batch_heavyfp_kron_refit.py \
      --meas-catalog "${refit_meas}" \
      --reference-image "${refit_image}" \
      --mag-min "${MAG_MIN}" \
      --mag-max "${MAG_MAX}" \
      --output-dir "${out_dir}" \
      --allow-missing-heavy-footprints \
      --leaf-only \
      --include-non-primary \
      "${refit_optional_args[@]}" \
      > "${LOG_DIR}/refit/${band}_${patch//,/_}.log" 2>&1
  if [[ "${COPY_REFIT_INPUTS_TO_TMP}" == "1" ]]; then
    rm -f "${tmp_meas}" "${tmp_image}"
  fi
  echo "[refit] done ${band} ${patch}: ${csv_path}"
}

run_refit_all() {
  local band patch pid
  local pids=()
  local failed=0

  split_words "${BANDS}"
  local bands_array=("${SPLIT_WORDS_OUT[@]}")

  echo "[refit] patches=${#PATCH_LIST[@]} bands=${#bands_array[@]} workers=${REFIT_WORKERS}"
  for patch in "${PATCH_LIST[@]}"; do
    for band in "${bands_array[@]}"; do
      throttle_jobs "${REFIT_WORKERS}"
      run_refit_one "${band}" "${patch}" &
      pids+=("$!")
    done
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "ERROR: at least one refit task failed. See ${LOG_DIR}/refit/*.log" >&2
    exit 1
  fi
}

run_preprocess() {
  export OMP_NUM_THREADS="${PREPROCESS_WORKER_THREADS}"
  export OMP_THREAD_LIMIT="${PREPROCESS_WORKER_THREADS}"
  export MKL_NUM_THREADS="${PREPROCESS_WORKER_THREADS}"
  export OPENBLAS_NUM_THREADS="${PREPROCESS_WORKER_THREADS}"
  export NUMEXPR_NUM_THREADS="${PREPROCESS_WORKER_THREADS}"
  export VECLIB_MAXIMUM_THREADS="${PREPROCESS_WORKER_THREADS}"
  export BLIS_NUM_THREADS="${PREPROCESS_WORKER_THREADS}"
  export NUMBA_NUM_THREADS="${PREPROCESS_WORKER_THREADS}"
  export TORCH_NUM_THREADS="${PREPROCESS_WORKER_THREADS}"
  export TORCH_NUM_INTEROP_THREADS=1

  split_words "${BANDS}"
  local bands_array=("${SPLIT_WORDS_OUT[@]}")
  split_words "${BAND_LIMIT_MAGS}"
  local band_limit_args=("${SPLIT_WORDS_OUT[@]}")
  split_words "${STRICT_CENTER_ONLY_SATURATION_MAGS}"
  local saturation_args=("${SPLIT_WORDS_OUT[@]}")
  split_words "${PU_B_FLAGS}"
  local pu_b_flags_args=("${SPLIT_WORDS_OUT[@]}")

  local patch_args=()
  if [[ -n "${PATCH_FILE}" ]]; then
    patch_args=(--patch-file "${PATCH_FILE}")
  elif [[ "${PATCHES}" == "all" ]]; then
    patch_args=(--patches all)
  else
    patch_args=(--patches "${PATCH_LIST[@]}")
  fi

  local optional_args=()
  if [[ "${PU_REQUIRE_KRON_REFIT_MATCH}" == "1" ]]; then
    optional_args+=(--pu-require-kron-refit-match)
  fi
  if [[ "${OVERWRITE_ZSCALE}" == "1" ]]; then
    optional_args+=(--overwrite-zscale)
  fi
  if [[ "${OVERWRITE_CUTOUTS}" == "1" ]]; then
    optional_args+=(--overwrite)
  fi
  if [[ "${SKIP_CUTOUTS}" == "1" ]]; then
    optional_args+=(--skip-cutouts)
  fi
  if [[ "${REUSE_EXISTING_PREPROCESSED}" == "1" ]]; then
    optional_args+=(--reuse-existing-preprocessed)
  fi
  if [[ "${REBUILD_IMAGE_VARIANTS}" == "1" ]]; then
    optional_args+=(--rebuild-image-variants)
  fi
  if [[ "${WRITE_TARGET_FITS}" == "1" ]]; then
    optional_args+=(--write-target-fits)
  fi
  if [[ -n "${FAST_ROOT}" && "${FAST_ROOT}" != "${PREP_ROOT}" ]]; then
    optional_args+=(--fast-root "${FAST_ROOT}")
  fi
  if [[ -n "${LSST_BACKGROUND_CACHE_ROOT}" ]]; then
    optional_args+=(--lsst-background-cache-root "${LSST_BACKGROUND_CACHE_ROOT}")
  fi
  if [[ -n "${VARIANT_LSST_BACKGROUND_ROOT}" ]]; then
    optional_args+=(--variant-lsst-background-root "${VARIANT_LSST_BACKGROUND_ROOT}")
  fi
  if [[ -n "${LSST_DETECT_PYTHON}" ]]; then
    optional_args+=(--lsst-detect-python "${LSST_DETECT_PYTHON}")
  fi
  if [[ "${OVERWRITE_LSST_BACKGROUND}" == "1" ]]; then
    optional_args+=(--overwrite-lsst-background)
  fi
  if [[ "${WRITE_LSST_BACKGROUND_PRODUCTS}" == "1" ]]; then
    optional_args+=(--write-lsst-background-products)
  fi
  if [[ "${LSST_BACKGROUND_DETECT_CUTOUTS}" == "1" ]]; then
    optional_args+=(--lsst-background-detect-cutouts)
  fi
  if [[ "${USE_LSST_DETECTION_CALEXP_CUTOUTS}" == "1" ]]; then
    optional_args+=(--use-lsst-detection-calexp-cutouts)
  fi
  if [[ "${ENABLE_STRICT_BRIGHT_CENTER_ONLY}" == "1" ]]; then
    optional_args+=(
      --pu-enable-strict-bright-center-only
      --pu-strict-bright-center-only-saturation-mags "${saturation_args[@]}"
      --pu-strict-bright-center-only-radius-column proxy_nan0_flux_aperture_radius
    )
  fi
  if [[ "${USE_BAND_LIMIT_B_FILTER}" == "1" ]]; then
    optional_args+=(
      --pu-use-band-limit-b-filter
      --pu-band-limit-mags "${band_limit_args[@]}"
    )
  fi
  if [[ -n "${DENOISED_FITS_ROOT}" ]]; then
    split_words "${IMAGE_VARIANTS}"
    local image_variant_args=("${SPLIT_WORDS_OUT[@]}")
    optional_args+=(
      --denoised-fits-root "${DENOISED_FITS_ROOT}"
      --image-variants "${image_variant_args[@]}"
      --noncoadd-snr-ignore-thresh "${NONCOADD_SNR_IGNORE_THRESH}"
      --noncoadd-snr-center-only-thresh "${NONCOADD_SNR_CENTER_ONLY_THRESH}"
      --noncoadd-snr-ap-radius "${NONCOADD_SNR_AP_RADIUS}"
      --noncoadd-snr-annulus-r-in "${NONCOADD_SNR_ANNULUS_R_IN}"
      --noncoadd-snr-annulus-r-out "${NONCOADD_SNR_ANNULUS_R_OUT}"
      --noncoadd-snr-source-mask-ellipse-sigma "${NONCOADD_SNR_SOURCE_MASK_ELLIPSE_SIGMA}"
      --noncoadd-snr-min-annulus-pixels "${NONCOADD_SNR_MIN_ANNULUS_PIXELS}"
    )
    split_words "${NONCOADD_SNR_MASK_PLANES}"
    local noncoadd_snr_mask_plane_args=("${SPLIT_WORDS_OUT[@]}")
    optional_args+=(--noncoadd-snr-mask-planes "${noncoadd_snr_mask_plane_args[@]}")
    if [[ "${NONCOADD_SNR_FILTER}" == "1" ]]; then
      optional_args+=(--noncoadd-snr-filter)
    else
      optional_args+=(--no-noncoadd-snr-filter)
    fi
    if [[ -n "${IMAGE_VARIANT_GROUPS}" ]]; then
      split_words "${IMAGE_VARIANT_GROUPS}"
      local image_variant_group_args=("${SPLIT_WORDS_OUT[@]}")
      optional_args+=(--image-variant-groups "${image_variant_group_args[@]}")
    fi
  fi

  mkdir -p "${LOG_DIR}"
  echo "[preprocess] start patches=${PATCHES} workers=${PREPROCESS_WORKERS}"
  run_python astro_data_preprocessing.py \
    --coadd-root "${RAW_ROOT}" \
    --catalog-root "${RAW_ROOT}" \
    --band-catalog-root "${RAW_ROOT}" \
    --tract "${TRACT}" \
    "${patch_args[@]}" \
    --bands "${bands_array[@]}" \
    --output-root "${PREP_ROOT}" \
    --zscale-root "${ZSCALE_ROOT}" \
    --label-mode pu \
    --target-shape-source kron \
    --ellipse-sigma 1.0 \
    --source-filter nchild0 \
    --pu-b-mag-min "${PU_B_MAG_MIN}" \
    --pu-b-mag-max "${PU_B_MAG_MAX}" \
    --pu-ap2-kron-abs-max "${PU_AP2_KRON_ABS_MAX}" \
    --pu-ap2-flux-column "${PU_AP2_FLUX_COLUMN}" \
    --pu-ap2-kron-flux-column "${PU_AP2_KRON_FLUX_COLUMN}" \
    --pu-b-close-center-arcsec "${PU_B_CLOSE_CENTER_ARCSEC}" \
    --pu-b-axis-ratio-max "${PU_B_AXIS_RATIO_MAX}" \
    --pu-containment-threshold "${PU_CONTAINMENT_THRESHOLD}" \
    --pu-keep-all-ab-clean \
    --pu-b-flags "${pu_b_flags_args[@]}" \
    --pu-kron-refit-csv "${REFIT_ROOT}/{tract}/{band}/{patch}/batch_heavyfp_kron_refit/batch_heavyfp_kron_refit.csv" \
    --pu-kron-refit-radius-column proxy_nan0_flux_aperture_radius \
    --lsst-background-policy "${LSST_BACKGROUND_POLICY}" \
    --bad-band-catalog-policy "${BAD_BAND_CATALOG_POLICY}" \
    --num-workers "${PREPROCESS_WORKERS}" \
    --variant-num-workers "${VARIANT_PREPROCESS_WORKERS}" \
    --worker-threads "${PREPROCESS_WORKER_THREADS}" \
    "${optional_args[@]}" \
    2>&1 | tee "${LOG_DIR}/astro_data_preprocessing.log"
  echo "[preprocess] done"
}

write_clean_regions() {
  split_words "${CLEAN_REGION_PATCHES}"
  local clean_region_patches=("${SPLIT_WORDS_OUT[@]}")
  if [[ "${CLEAN_REGION_PATCHES}" == "all" ]]; then
    clean_region_patches=("${PATCH_LIST[@]}")
  fi
  split_words "${CLEAN_REGION_BANDS}"
  local clean_region_bands=("${SPLIT_WORDS_OUT[@]}")
  split_words "${CLEAN_REGION_CLASSES}"
  local clean_region_classes=("${SPLIT_WORDS_OUT[@]}")

  local optional_args=()
  if [[ -n "${CLEAN_REGION_TILE_NAME}" ]]; then
    optional_args+=(--tile-name "${CLEAN_REGION_TILE_NAME}")
  fi
  if [[ -n "${CLEAN_REGION_X0}" ]]; then
    optional_args+=(--crop-x0 "${CLEAN_REGION_X0}")
  fi
  if [[ -n "${CLEAN_REGION_Y0}" ]]; then
    optional_args+=(--crop-y0 "${CLEAN_REGION_Y0}")
  fi
  if [[ -n "${CLEAN_REGION_SIZE}" ]]; then
    optional_args+=(--crop-size "${CLEAN_REGION_SIZE}")
  fi
  if [[ -n "${CLEAN_REGION_WIDTH}" ]]; then
    optional_args+=(--crop-width "${CLEAN_REGION_WIDTH}")
  fi
  if [[ -n "${CLEAN_REGION_HEIGHT}" ]]; then
    optional_args+=(--crop-height "${CLEAN_REGION_HEIGHT}")
  fi
  if [[ "${CLEAN_REGION_LOCAL_COORDS}" == "1" ]]; then
    optional_args+=(--local-coordinates)
  fi
  if [[ "${CLEAN_REGION_WRITE_CLASS_FILES}" == "1" ]]; then
    optional_args+=(--write-class-files)
  fi
  if [[ -n "${CLEAN_REGION_AP2_KRON_ABS_MAX}" ]]; then
    optional_args+=(
      --ap2-kron-abs-max "${CLEAN_REGION_AP2_KRON_ABS_MAX}"
      --ap2-flux-column "${CLEAN_REGION_AP2_FLUX_COLUMN}"
      --kron-flux-column "${CLEAN_REGION_KRON_FLUX_COLUMN}"
      --photometry-zeropoint "${CLEAN_REGION_PHOTOMETRY_ZEROPOINT}"
    )
  fi

  echo "[clean-regions] start patches=${clean_region_patches[*]} bands=${clean_region_bands[*]} classes=${clean_region_classes[*]}"
  run_python export_preprocessed_clean_regions.py \
    --data "${RAW_ROOT}" \
    --root "${PREP_ROOT}" \
    --tract "${TRACT}" \
    --patches "${clean_region_patches[@]}" \
    --bands "${clean_region_bands[@]}" \
    --classes "${clean_region_classes[@]}" \
    --output-dir "${CLEAN_REGION_OUT_DIR}" \
    --crop-margin "${CLEAN_REGION_MARGIN}" \
    --allow-missing-classes \
    "${optional_args[@]}" \
    2>&1 | tee "${LOG_DIR}/export_preprocessed_clean_regions.log"
  echo "[clean-regions] done: ${CLEAN_REGION_OUT_DIR}"
}

main() {
  build_patch_list
  echo "TRACT=${TRACT}"
  echo "RAW_ROOT=${RAW_ROOT}"
  echo "PREP_ROOT=${PREP_ROOT}"
  echo "FAST_ROOT=${FAST_ROOT}"
  echo "ZSCALE_ROOT=${ZSCALE_ROOT}"
  echo "BANDS=${BANDS}"
  echo "PATCHES=${PATCHES} PATCH_FILE=${PATCH_FILE:-<none>} expanded_patch_count=${#PATCH_LIST[@]}"
  echo "PU_B_MAG_MIN=${PU_B_MAG_MIN} PU_B_MAG_MAX=${PU_B_MAG_MAX}"
  echo "USE_BAND_LIMIT_B_FILTER=${USE_BAND_LIMIT_B_FILTER} BAND_LIMIT_MAGS=${BAND_LIMIT_MAGS}"
  echo "PU_B_CLOSE_CENTER_ARCSEC=${PU_B_CLOSE_CENTER_ARCSEC} PU_B_AXIS_RATIO_MAX=${PU_B_AXIS_RATIO_MAX} PU_CONTAINMENT_THRESHOLD=${PU_CONTAINMENT_THRESHOLD}"
  echo "PU_B_FLAGS=${PU_B_FLAGS}"
  echo "PU_REQUIRE_KRON_REFIT_MATCH=${PU_REQUIRE_KRON_REFIT_MATCH}"
  echo "ENABLE_STRICT_BRIGHT_CENTER_ONLY=${ENABLE_STRICT_BRIGHT_CENTER_ONLY}"
  echo "STRICT_CENTER_ONLY_SATURATION_MAGS=${STRICT_CENTER_ONLY_SATURATION_MAGS}"
  echo "LSST_BACKGROUND_POLICY=${LSST_BACKGROUND_POLICY}"
  echo "VARIANT_LSST_BACKGROUND_ROOT=${VARIANT_LSST_BACKGROUND_ROOT:-<none>}"
  echo "LSST_BACKGROUND_DETECT_CUTOUTS=${LSST_BACKGROUND_DETECT_CUTOUTS}"
  echo "USE_LSST_DETECTION_CALEXP_CUTOUTS=${USE_LSST_DETECTION_CALEXP_CUTOUTS}"
  echo "WRITE_CLEAN_REGIONS=${WRITE_CLEAN_REGIONS}"

  if [[ "${RUN_REFIT}" == "1" ]]; then
    run_refit_all
  else
    echo "[refit] skipped because RUN_REFIT=${RUN_REFIT}"
  fi

  if [[ "${RUN_PREPROCESS}" == "1" ]]; then
    run_preprocess
  else
    echo "[preprocess] skipped because RUN_PREPROCESS=${RUN_PREPROCESS}"
  fi

  if [[ "${WRITE_CLEAN_REGIONS}" == "1" ]]; then
    write_clean_regions
  fi
}

main "$@"
