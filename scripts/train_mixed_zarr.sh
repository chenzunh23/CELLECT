#!/usr/bin/env bash
set -euo pipefail
cd /home/czh23/CELLECT
PYTHON=${PYTHON:-/home/czh23/miniconda3/envs/cellect/bin/python}
ROOT=${ROOT:-/data/czh23/analysis/2026-09/2026-09-25/training_zarr_psfee_v1/zarr}
# No implicit all-field split: use the same explicit train/val lists as legacy runs.
SPLITS=${SPLITS:-}
TRAIN_FILE=${TRAIN_FILE:-${SPLITS:+$SPLITS/train.txt}}
VAL_FILE=${VAL_FILE:-${SPLITS:+$SPLITS/val.txt}}
OUT=${OUT:-/data/czh23/analysis/2026-09/2026-09-25/cellect_mixed_psfee_v1}
SAM_CKPT=${SAM_CKPT:-/home/czh23/sam_ckpts/sam_vit_b_01ec64.pth}
NPROC=${NPROC:-1}
MODE=${MODE:-train}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
# Also accept the existing Python CLI spelling after the script name.
forward_args=("$@")
for (( i=0; i<${#forward_args[@]}; i++ )); do
  case "${forward_args[i]}" in
    --train-patches-file) TRAIN_FILE=${forward_args[i+1]:-} ;;
    --train-patches-file=*) TRAIN_FILE=${forward_args[i]#*=} ;;
    --val-patches-file|--eval-patches-file) VAL_FILE=${forward_args[i+1]:-} ;;
    --val-patches-file=*|--eval-patches-file=*) VAL_FILE=${forward_args[i]#*=} ;;
  esac
done
if [[ "$MODE" == train && ! -s "$TRAIN_FILE" ]]; then
  echo 'Specify a nonempty TRAIN_FILE or --train-patches-file; no full-field default is selected.' >&2
  exit 1
fi
[[ -s "$VAL_FILE" ]] || { echo 'Specify a nonempty VAL_FILE or --val-patches-file/--eval-patches-file.' >&2; exit 1; }
[[ -f "$SAM_CKPT" ]] || { echo "Missing SAM checkpoint: $SAM_CKPT" >&2; exit 1; }
# Baseline: /data/czh23/ckpts/sam_lupton_half_control_0828/run_config.json
# Pin the historical training recipe here instead of inheriting Python defaults.
# Dataset paths/selectors, PSF-EE labels, new mask supervision/geometry, independent
# band evaluation and dynamic-mask DDP handling are intentional current changes.
common=(--mode "$MODE" --data-format zarr --root "$ROOT" --bands all
  --dataset-sources all --fits-hdu 1
  --zarr-random-image-batches --zarr-chunk-local-batches --zarr-worker-owned-chunks
  --zarr-shuffle-within-chunk --zarr-drop-last
  --batch-size "${BATCH_SIZE:-20}" --num-workers "${WORKERS:-4}"
  --pin-memory --persistent-workers --prefetch-factor 2 --seed 7
  --device cuda --amp bf16
  --model-variant sam_per_band --sam-model-type vit_b --sam-checkpoint "$SAM_CKPT"
  --base-channels 32 --embedding-dim 64 --seg-classes 2
  --no-sam-decoder-film --no-sam-encoder-style-prompt
  --style-prompt-dim 32 --style-adapter-dim 32
  --style-router-temperature 1 --style-router-loss-weight 0.1
  --sam-compile --sam-compile-backend inductor --sam-compile-mode default
  --lr "${LR:-0.0000707}" --sam-encoder-lr "${ENCODER_LR:-0.00001414}"
  --weight-decay 0.1 --sam-warmup-ratio 0
  --sam-lr-drop-fractions 0.7 0.9 --sam-lr-drop-gamma 0.1
  --sam-lr-phase2-epoch 35 --sam-encoder-lr-after 0 --sam-head-lr-after 0
  --freeze-proposal-after-epochs -1
  --confidence-loss-mode ce_hard --confidence-loss-weight 1 --confidence-pos-weight 32
  --confidence-ce-weights 1 4 8 16 32
  --confidence-score ordinal_expectation --confidence-threshold 2
  --no-use-ordinal-expectation --no-debug-ordinal-expectation
  --nms-radius 1 --center-refinement integer --center-refinement-radius 1
  --match-radius 3 --center-tolerance-arcsec 0.5 --pixel-scale-arcsec 0.168
  --no-detection-linking --no-train-detect-ex-link --disable-ex-loss --ignore-mask-during-detection
  --center-loss-weight 0 --seg-loss-weight 0 --seg-loss-stride 1
  --segmentation-class-weights 1 32
  --shape-loss-weight 0.5 --shape-loss-mode source_center --shape-center-size 3
  --shape-geometry-loss log_spd --shape-angle-weight 4
  --small-shape-loss-weight 0 --small-shape-area-min 20 --small-shape-area-tau 5
  --small-shape-ordinal-threshold 2 --small-shape-scope ignore
  --ellipse-sigma 1 --core-radius 2 --shape-source kron --source-filter nchild0
  --noncoadd-snr-filter --noncoadd-snr-ignore-thresh 2 --noncoadd-snr-center-only-thresh 3
  --noncoadd-snr-ap-radius 6 --noncoadd-snr-annulus-r-in 10
  --noncoadd-snr-annulus-r-out 15 --noncoadd-snr-annulus-exclude-radius 6
  --mask-loss-weight 5 --mask-loss-warmup-epochs 3
  --mask-loss-interval 1 --mask-loss-interval-scale
  --mask-centroid-weight 0.4 --mask-outside-weight 1
  --mask-min-area-weight 1 --mask-max-area-weight 0.1
  --mask-pred-iou-weight 0.1 --mask-stability-weight 0
  --mask-unmatched-prompt-weight 0.2 --center-only-shape-factor 0.2 --mask-pred-iou-thresh 0.8
  --mask-stability-score-thresh 0.95 --mask-stability-score-offset 1 --mask-stability-temperature 10
  --mask-prompt-gt-epochs 10 --mask-prompt-pred-epoch 30 --mask-selection loss
  --mask-max-gt-per-sample 128 --mask-max-pred-per-sample 128
  --mask-prompt-chunk-size "${MASK_PROMPT_CHUNK_SIZE:-256}"
  # Current supervised-mask additions; do not restore the old zero BCE/Dice.
  --mask-supervision-weight 0.2 --mask-bce-weight 1 --mask-dice-weight 1
  --mask-outside-kron-scale 1.5 --mask-area-ratio-lower 0.05 --mask-area-ratio-upper 2
  --mask-min-area-px 6 --mask-max-area-ratio 0.8 
  --ddp-static-graph off --ddp-progress-step-mode global --ddp-timeout-minutes 60
  --wandb-project Astro_CELLECT2D_SAM --wandb-log-interval 10
  --wandb-run-name "${WANDB_RUN_NAME:-cellect_mixed_psfee_v1}"
  --wandb-mode "${WANDB_MODE:-online}")
if [[ "$MODE" == train ]]; then
  common+=(--train-patches-file "$TRAIN_FILE" --val-patches-file "$VAL_FILE"
    --out-dir "$OUT" --epochs "${EPOCHS:-60}"
    --detect-every "${DETECT_EVERY:-5}" --ckpt-interval "${CKPT_INTERVAL:-2}")
elif [[ "$MODE" == eval ]]; then
  common+=(--eval-patches-file "$VAL_FILE" --out-dir "$OUT/eval"
    --checkpoint "${CHECKPOINT:-$OUT/best.pt}")
else
  echo 'MODE must be train or eval' >&2; exit 1
fi
launcher=("$PYTHON")
if (( NPROC > 1 )); then
  launcher+=(-m torch.distributed.run --standalone --nproc_per_node "$NPROC")
fi
command=("${launcher[@]}" astro_train_eval.py "${common[@]}" "$@")
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${command[@]}"; printf '\n'
  exit 0
fi
exec "${command[@]}"
