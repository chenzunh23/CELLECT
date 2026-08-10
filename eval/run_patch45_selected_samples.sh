#!/usr/bin/env bash
set -euo pipefail

# Batch CELLECT visual diagnostics for the hand-picked 9813/4,5 samples.
#
# Defaults cover:
#   - v3 anscombe image-level zarr
#   - v3 zscore-no-upper image-level zarr
#   - original 2026-07-12 sam_per_band checkpoint on old multi-band zarr
#   - broad-band noisy HSC-G/HSC-I where noisy zarrs exist
#
# Override paths or sample lists with environment variables, for example:
#   SAMPLES="50 59" NOISY_GROUPS="group_00 group_02" bash eval/run_patch45_selected_samples.sh

PATCH="${PATCH:-4,5}"
TRACT="${TRACT:-9813}"
SAMPLES_STR="${SAMPLES:-50 59 110 114 87}"
COADD_BANDS_STR="${COADD_BANDS:-HSC-G HSC-I NB1010}"
NOISY_BANDS_STR="${NOISY_BANDS:-HSC-I}"
NOISY_GROUPS_STR="${NOISY_GROUPS:-group_01}"

PYTHON_CMD_STR="${PYTHON_CMD:-${PYTHON_BIN:-python}}"
read -r -a PYTHON_CMD <<< "$PYTHON_CMD_STR"
DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:-output/eval_visualizations/cellect_outputs}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
MAKE_MASKS="${MAKE_MASKS:-1}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-2.0}"
NMS_RADIUS="${NMS_RADIUS:-3}"
MASK_CHUNK_SIZE="${MASK_CHUNK_SIZE:-512}"
SHAPE_OVERLAY_CENTERS="${SHAPE_OVERLAY_CENTERS:-0}"
INPUT_SHAPE_OVERLAY="${INPUT_SHAPE_OVERLAY:-1}"

V3_ANSCOMBE_ROOT="${V3_ANSCOMBE_ROOT:-/data/czh23/direct_zarr_v3_anscombe}"
V3_ZNO_ROOT="${V3_ZNO_ROOT:-/data/czh23/direct_zarr_v3_zscore_no_upper}"
V3_LOG_LUPTON_ROOT="${V3_LOG_LUPTON_ROOT:-/data/czh23/direct_zarr_v3_lupton}"
OLD_ZARR_ROOT="${OLD_ZARR_ROOT:-/data/czh23/direct_zarr_v2_0711}"
FITS_ROOT="${FITS_ROOT:-/data/shared/Subaru}"

ANSCOMBE_CKPT="${ANSCOMBE_CKPT:-/data/czh23/ckpts/sam_anscombe_0803/epoch_0030.pt}"
ZNO_CKPT="${ZNO_CKPT:-/data/czh23/ckpts/sam_zscore_no_upper/epoch_0030.pt}"
LOG_LUPTON_CKPT="${LOG_LUPTON_CKPT:-/data/czh23/ckpts/sam_log_lupton_0804/best.pt}"
ORIGINAL_CKPT="${ORIGINAL_CKPT:-/data/czh23/ckpts/sam_control_0803/epoch_0030.pt}"
RUN_ORIGINAL_ZNO="${RUN_ORIGINAL_ZNO:-0}"
RUN_ORIGINAL_NOISY="${RUN_ORIGINAL_NOISY:-1}"
RUN_ORIGINAL_NOISY_ZNO="${RUN_ORIGINAL_NOISY_ZNO:-0}"

read -r -a SAMPLES <<< "$SAMPLES_STR"
read -r -a COADD_BANDS <<< "$COADD_BANDS_STR"
read -r -a NOISY_BANDS <<< "$NOISY_BANDS_STR"
read -r -a NOISY_GROUPS <<< "$NOISY_GROUPS_STR"

declare -A TILE_BY_SAMPLE=(
  [50]="grid_r04_c06_x18108_y21372"
  [59]="grid_r05_c04_x17372_y21740"
  [87]="grid_r07_c10_x19588_y22476"
  [110]="grid_r10_c00_x15900_y23588"
  [114]="grid_r10_c04_x17372_y23588"
)

bool_flag() {
  local enabled="$1"
  local true_flag="$2"
  local false_flag="$3"
  if [[ "$enabled" == "1" || "$enabled" == "true" || "$enabled" == "yes" ]]; then
    printf '%s\n' "$true_flag"
  else
    printf '%s\n' "$false_flag"
  fi
}

normalize_group_name() {
  local group="${1:-}"
  if [[ -z "$group" ]]; then
    printf ''
  elif [[ "$group" == group_* ]]; then
    printf '%s' "$group"
  elif [[ "$group" =~ ^[0-9]+$ ]]; then
    printf 'group_%02d' "$group"
  else
    printf '%s' "$group"
  fi
}

common_eval_args() {
  bool_flag "$MAKE_MASKS" "--make-masks" "--no-make-masks"
  bool_flag "$SKIP_EXISTING" "--skip-existing" "--no-skip-existing"
  bool_flag "$SHAPE_OVERLAY_CENTERS" "--shape-overlay-centers" "--no-shape-overlay-centers"
  bool_flag "$INPUT_SHAPE_OVERLAY" "--input-shape-overlay" "--no-input-shape-overlay"
  printf '%s\n' \
    "--device" "$DEVICE" \
    "--confidence-threshold" "$CONFIDENCE_THRESHOLD" \
    "--nms-radius" "$NMS_RADIUS" \
    "--mask-chunk-size" "$MASK_CHUNK_SIZE" \
    "--out-dir" "$OUT_DIR"
}

image_level_store_exists() {
  local root="$1"
  local source="$2"
  local band="$3"
  local group
  group="$(normalize_group_name "${4:-}")"
  if [[ -n "$group" ]]; then
    [[ -d "$root/image_level/$source/$band/${PATCH}__${group}.zarr" || -d "$root/$TRACT/image_level/$source/$band/${PATCH}__${group}.zarr" ]]
  else
    [[ -d "$root/image_level/$source/$band/$PATCH.zarr" || -d "$root/$TRACT/image_level/$source/$band/$PATCH.zarr" ]]
  fi
}

image_level_store_hint() {
  local root="$1"
  local source="$2"
  local band="$3"
  local group
  group="$(normalize_group_name "${4:-}")"
  if [[ -n "$group" ]]; then
    printf '%s or %s\n' \
      "$root/image_level/$source/$band/${PATCH}__${group}.zarr" \
      "$root/$TRACT/image_level/$source/$band/${PATCH}__${group}.zarr"
  else
    printf '%s or %s\n' \
      "$root/image_level/$source/$band/$PATCH.zarr" \
      "$root/$TRACT/image_level/$source/$band/$PATCH.zarr"
  fi
}

old_patch_store_exists() {
  local source="$1"
  [[ -d "$OLD_ZARR_ROOT/$TRACT/$source/$PATCH.zarr" ]]
}

run_v3_one() {
  local label="$1"
  local root="$2"
  local ckpt="$3"
  local source="$4"
  local band="$5"
  local sample="$6"
  local group
  group="$(normalize_group_name "${7:-}")"
  local tile="${TILE_BY_SAMPLE[$sample]:-}"
  if [[ -z "$tile" ]]; then
    echo "[skip] unknown sample ${sample}; add it to TILE_BY_SAMPLE" >&2
    return 0
  fi
  if ! image_level_store_exists "$root" "$source" "$band" "$group"; then
    echo "[skip] ${label} ${source} ${band} ${PATCH}${group:+ ${group}}: zarr store missing at $(image_level_store_hint "$root" "$source" "$band" "$group")"
    return 0
  fi
  local cmd=(
    "${PYTHON_CMD[@]}" eval/visualize_cellect_outputs.py
    --checkpoint "$ckpt"
    --root "$root"
    --dataset-source "$source"
    --patch "$PATCH"
    --tile-name "$tile"
    --zarr-band "$band"
    --sample-index 0
    --band "$band"
  )
  if [[ -n "$group" ]]; then
    cmd+=(--group "$group")
  fi
  while IFS= read -r arg; do
    cmd+=("$arg")
  done < <(common_eval_args)
  echo "[run] ${label} ${source} ${band} sample=${sample}${group:+ group=${group}}"
  "${cmd[@]}"
}

run_original_one() {
  local scaling="$1"
  local source="$2"
  local band="$3"
  local sample="$4"
  local group
  group="$(normalize_group_name "${5:-}")"
  local tile="${TILE_BY_SAMPLE[$sample]:-}"
  if [[ -z "$tile" ]]; then
    echo "[skip] unknown sample ${sample}; add it to TILE_BY_SAMPLE" >&2
    return 0
  fi
  if ! old_patch_store_exists "$source"; then
    echo "[skip] original ${source} ${PATCH}: old patch zarr missing"
    return 0
  fi
  local cmd=(
    "${PYTHON_CMD[@]}" eval/visualize_multiband_zarr_outputs.py
    --checkpoint "$ORIGINAL_CKPT"
    --zarr-root "$OLD_ZARR_ROOT"
    --tract "$TRACT"
    --patch "$PATCH"
    --dataset-source "$source"
    --tile-name "$tile"
    --band "$band"
    --input-scaling-mode "$scaling"
  )
  if [[ "$scaling" != "zarr" ]]; then
    cmd+=(--fits-root "$FITS_ROOT")
  fi
  if [[ -n "$group" ]]; then
    cmd+=(--group "$group")
  fi
  while IFS= read -r arg; do
    cmd+=("$arg")
  done < <(common_eval_args)
  echo "[run] original ${scaling} ${source} ${band} sample=${sample}${group:+ group=${group}}"
  "${cmd[@]}"
}

for sample in "${SAMPLES[@]}"; do
  for band in "${COADD_BANDS[@]}"; do
    # run_v3_one "anscombe" "$V3_ANSCOMBE_ROOT" "$ANSCOMBE_CKPT" coadd "$band" "$sample"
    # run_v3_one "zscore-no-upper" "$V3_ZNO_ROOT" "$ZNO_CKPT" coadd "$band" "$sample"
    run_v3_one "log-lupton" "$V3_LOG_LUPTON_ROOT" "$LOG_LUPTON_CKPT" coadd "$band" "$sample"
  done
  for band in "${NOISY_BANDS[@]}"; do
    run_original_one zarr coadd "$band" "$sample"
    if [[ "$RUN_ORIGINAL_ZNO" == "1" || "$RUN_ORIGINAL_ZNO" == "true" || "$RUN_ORIGINAL_ZNO" == "yes" ]]; then
      run_original_one zscore_no_upper coadd "$band" "$sample"
    fi
    for group in "${NOISY_GROUPS[@]}"; do
      # run_v3_one "anscombe" "$V3_ANSCOMBE_ROOT" "$ANSCOMBE_CKPT" noisy "$band" "$sample" "$group"
      # run_v3_one "zscore-no-upper" "$V3_ZNO_ROOT" "$ZNO_CKPT" noisy "$band" "$sample" "$group"
      run_v3_one "log-lupton" "$V3_LOG_LUPTON_ROOT" "$LOG_LUPTON_CKPT" noisy "$band" "$sample" "$group"
      if [[ "$RUN_ORIGINAL_NOISY" == "1" || "$RUN_ORIGINAL_NOISY" == "true" || "$RUN_ORIGINAL_NOISY" == "yes" ]]; then
        run_original_one zarr noisy "$band" "$sample" "$group"
      fi
      if [[ "$RUN_ORIGINAL_NOISY_ZNO" == "1" || "$RUN_ORIGINAL_NOISY_ZNO" == "true" || "$RUN_ORIGINAL_NOISY_ZNO" == "yes" ]]; then
        run_original_one zscore_no_upper noisy "$band" "$sample" "$group"
      fi
    done
  done
done

echo "[done] selected sample experiments"
