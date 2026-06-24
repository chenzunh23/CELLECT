#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Run native SAM-astro AMG on one Zangetsu cutout band and export DS9 regions.
This mirrors lsst_pipeline/scarlet_deblend_from_fits.py per-band SAM detection:
the selected band is repeated as an RGB triplet, scaling_mode=astro_rgb,
astro_rgb_mode=none, and astro preprocessing runs inside the SAM model.

Configure with environment variables, for example:
  DATASET=denoised BAND=HSC-I MODEL_TYPE=vit_h CHECKPOINT=/home/czh23/sam_ckpts/sam_vit_h_4b8939.pth \
    /home/czh23/CELLECT/zangetsu_demo/run_native_sam_astro_zangetsu.sh

Main variables:
  DATA_ROOT, DATASET, TRACT, PATCH, TILE, BAND, OUT_DIR
  MODEL_TYPE, CHECKPOINT, DEVICE, HDU
  POINTS_PER_SIDE, POINTS_PER_BATCH, PRED_IOU_THRESH, STABILITY_SCORE_THRESH
  BOX_NMS_THRESH, CROP_N_LAYERS, CROP_NMS_THRESH, CROP_OVERLAP_RATIO
  CROP_N_POINTS_DOWNSCALE_FACTOR, MIN_MASK_REGION_AREA, MAX_MASK_AREA_RATIO
  ASTRO_PREPROCESS_IN_MODEL, ASTRO_PREPROCESS_Z_CLIP, ASTRO_CROP_SIZE
  LOW_PERCENTILE, HIGH_PERCENTILE, OVERLAY_ALPHA, CENTER_RADIUS
EOF
  exit 0
fi

SAM_ASTRO_ROOT="${SAM_ASTRO_ROOT:-/home/czh23/SAM-astro}"
CELLECT_ROOT="${CELLECT_ROOT:-/home/czh23/CELLECT}"
DATA_ROOT="${DATA_ROOT:-${CELLECT_ROOT}/output/sam_cellect_combination_260611/preprocessing_diagnostics_260611/zangetsu_preprocessed_cutouts_260611}"
DATASET="${DATASET:-denoised}"
TRACT="${TRACT:-9813}"
PATCH="${PATCH:-6,1}"
TILE="${TILE:-zangetsu_lower_right_x27366_y6453}"
BAND="${BAND:-HSC-I}"
MODEL_TYPE="${MODEL_TYPE:-vit_b}"
OUT_DIR="${OUT_DIR:-${CELLECT_ROOT}/zangetsu_demo/output/native_sam_astro_${MODEL_TYPE}_${DATASET}_${BAND}}"
CHECKPOINT="${CHECKPOINT:-/home/czh23/sam_ckpts/sam_vit_b_01ec64.pth}"
DEVICE="${DEVICE:-cuda}"
HDU="${HDU:-1}"
POINTS_PER_SIDE="${POINTS_PER_SIDE:-32}"
POINTS_PER_BATCH="${POINTS_PER_BATCH:-128}"
PRED_IOU_THRESH="${PRED_IOU_THRESH:-0.8}"
STABILITY_SCORE_THRESH="${STABILITY_SCORE_THRESH:-0.95}"
BOX_NMS_THRESH="${BOX_NMS_THRESH:-0.7}"
CROP_N_LAYERS="${CROP_N_LAYERS:-1}"
CROP_NMS_THRESH="${CROP_NMS_THRESH:-0.7}"
CROP_OVERLAP_RATIO="${CROP_OVERLAP_RATIO:-0.3413333333333333}"
CROP_N_POINTS_DOWNSCALE_FACTOR="${CROP_N_POINTS_DOWNSCALE_FACTOR:-1}"
MAX_MASK_AREA_RATIO="${MAX_MASK_AREA_RATIO:-0.5}"
MIN_MASK_REGION_AREA="${MIN_MASK_REGION_AREA:-15}"
OVERLAY_ALPHA="${OVERLAY_ALPHA:-0.35}"
ASTRO_PREPROCESS_IN_MODEL="${ASTRO_PREPROCESS_IN_MODEL:-1}"
ASTRO_PREPROCESS_CLIP_SIGMA="${ASTRO_PREPROCESS_CLIP_SIGMA:-3.0}"
ASTRO_PREPROCESS_SIGMA_ITERS="${ASTRO_PREPROCESS_SIGMA_ITERS:--1}"
ASTRO_PREPROCESS_Z_CLIP="${ASTRO_PREPROCESS_Z_CLIP:--3.0 3.0}"
ASTRO_STATS_MODE="${ASTRO_STATS_MODE:-sigmaclip}"
# scarlet_deblend_from_fits.py treats astro_crop_size<=0 as full-frame before
# calling amg_fits_core.build_astro_input(). The CLI has no such wrapper, so
# default to the Zangetsu cutout size while keeping this overridable.
ASTRO_CROP_SIZE="${ASTRO_CROP_SIZE:-512}"
LOW_PERCENTILE="${LOW_PERCENTILE:-0.1}"
HIGH_PERCENTILE="${HIGH_PERCENTILE:-99.5}"
CENTER_RADIUS="${CENTER_RADIUS:-7}"

FITS_PATH="${DATA_ROOT}/${DATASET}/${TRACT}/${PATCH}/cutouts/${TILE}/${BAND}"
FITS_FILE="$(find "${FITS_PATH}" -maxdepth 1 -type f -name '*.fits' | sort | head -n 1)"
if [[ -z "${FITS_FILE}" ]]; then
  echo "No FITS file found under ${FITS_PATH}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
read -r -a ASTRO_PREPROCESS_Z_CLIP_VALUES <<< "${ASTRO_PREPROCESS_Z_CLIP}"
AMG_ARGS=(
  "${SAM_ASTRO_ROOT}/scripts/amg_fits_core.py"
  --input "${FITS_FILE}" "${FITS_FILE}" "${FITS_FILE}"
  --output "${OUT_DIR}"
  --model-type "${MODEL_TYPE}"
  --checkpoint "${CHECKPOINT}"
  --device "${DEVICE}"
  --hdu "${HDU}"
  --points-per-side "${POINTS_PER_SIDE}"
  --points-per-batch "${POINTS_PER_BATCH}"
  --pred-iou-thresh "${PRED_IOU_THRESH}"
  --stability-score-thresh "${STABILITY_SCORE_THRESH}"
  --box-nms-thresh "${BOX_NMS_THRESH}"
  --crop-n-layers "${CROP_N_LAYERS}"
  --crop-nms-thresh "${CROP_NMS_THRESH}"
  --crop-overlap-ratio "${CROP_OVERLAP_RATIO}"
  --crop-n-points-downscale-factor "${CROP_N_POINTS_DOWNSCALE_FACTOR}"
  --max-mask-area-ratio "${MAX_MASK_AREA_RATIO}"
  --min-mask-region-area "${MIN_MASK_REGION_AREA}"
  --overlay-alpha "${OVERLAY_ALPHA}"
  --scaling-mode astro_rgb
  --astro-rgb-mode none
  --astro-stats-mode "${ASTRO_STATS_MODE}"
  --astro-crop-size "${ASTRO_CROP_SIZE}"
  --astro-preprocess-clip-sigma "${ASTRO_PREPROCESS_CLIP_SIGMA}"
  --astro-preprocess-sigma-iters "${ASTRO_PREPROCESS_SIGMA_ITERS}"
  --low-percentile "${LOW_PERCENTILE}"
  --high-percentile "${HIGH_PERCENTILE}"
  --overlay-style fill
)
if [[ "${ASTRO_PREPROCESS_IN_MODEL}" == "1" || "${ASTRO_PREPROCESS_IN_MODEL}" == "true" || "${ASTRO_PREPROCESS_IN_MODEL}" == "yes" ]]; then
  AMG_ARGS+=(--astro-preprocess-in-model)
fi
if [[ "${#ASTRO_PREPROCESS_Z_CLIP_VALUES[@]}" -gt 0 ]]; then
  AMG_ARGS+=(--astro-preprocess-z-clip "${ASTRO_PREPROCESS_Z_CLIP_VALUES[@]}")
fi

python "${AMG_ARGS[@]}"

python - "${OUT_DIR}" "${CENTER_RADIUS}" <<'PY'
import math
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

out_dir = Path(sys.argv[1])
center_radius = float(sys.argv[2])
label_paths = sorted(out_dir.glob("*_sam_labelmap.fits"))
if not label_paths:
    raise SystemExit(f"No *_sam_labelmap.fits found in {out_dir}")
label_path = label_paths[-1]
label_map = np.asarray(fits.getdata(label_path), dtype=np.int32)
prefix = label_path.name[: -len("_sam_labelmap.fits")]

header = [
    "# Region file format: DS9 version 4.1",
    'global color=green dashlist=8 3 width=2 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
    "image",
]

def contours(mask, max_vertices=128):
    try:
        from skimage import measure
        for contour in measure.find_contours(mask.astype(np.float32), 0.5):
            if contour.shape[0] < 3:
                continue
            step = max(1, int(math.ceil(contour.shape[0] / max_vertices)))
            pts = contour[::step]
            if pts.shape[0] >= 3:
                yield [(float(x + 1.0), float(y + 1.0)) for y, x in pts]
        return
    except Exception:
        pass
    ys, xs = np.where(mask)
    if ys.size:
        yield [
            (float(xs.min() + 1), float(ys.min() + 1)),
            (float(xs.max() + 2), float(ys.min() + 1)),
            (float(xs.max() + 2), float(ys.max() + 2)),
            (float(xs.min() + 1), float(ys.max() + 2)),
        ]

mask_lines = header + [f"# Native SAM mask contours from {label_path.name}"]
center_lines = header + [f"# Native SAM geometric mask centers from {label_path.name}"]
labels = [int(x) for x in np.unique(label_map) if int(x) > 0]
for label in labels:
    mask = label_map == label
    ys, xs = np.where(mask)
    if ys.size == 0:
        continue
    cx = float(xs.mean())
    cy = float(ys.mean())
    area = int(ys.size)
    center_lines.append(f"circle({cx + 1:.3f},{cy + 1:.3f},{center_radius:.3f}) # color=cyan width=2 text={{id={label} area={area}}}")
    for contour in contours(mask):
        coords = ",".join(f"{x:.2f},{y:.2f}" for x, y in contour)
        mask_lines.append(f"polygon({coords}) # color=green width=1 text={{id={label} area={area}}}")

(out_dir / f"{prefix}_mask_contours.reg").write_text("\n".join(mask_lines) + "\n")
(out_dir / f"{prefix}_mask_centers.reg").write_text("\n".join(center_lines) + "\n")
print(f"wrote {out_dir / (prefix + '_mask_contours.reg')}")
print(f"wrote {out_dir / (prefix + '_mask_centers.reg')}")
PY
