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
common=(--mode "$MODE" --data-format zarr --root "$ROOT" --bands all
  --dataset-sources all --zarr-random-image-batches --model-variant sam_per_band
  --sam-model-type vit_b --sam-checkpoint "$SAM_CKPT"
  --batch-size "${BATCH_SIZE:-4}" --num-workers "${WORKERS:-4}" --amp bf16
  --confidence-loss-mode ce_hard --match-radius 3 --no-detection-linking --no-train-detect-ex-link
  --mask-loss-weight 1 --mask-loss-warmup-epochs 5 --mask-supervision-weight 0.2
  --mask-bce-weight 1 --mask-dice-weight 1 --mask-prompt-chunk-size 32
  --mask-max-gt-per-sample 128 --mask-max-pred-per-sample 128
  --ddp-static-graph off --wandb-mode "${WANDB_MODE:-disabled}")
if [[ "$MODE" == train ]]; then
  common+=(--train-patches-file "$TRAIN_FILE" --val-patches-file "$VAL_FILE"
    --out-dir "$OUT" --epochs "${EPOCHS:-100}" --lr 0.0001 --sam-encoder-lr 0.00002
    --detect-every 1 --ckpt-interval 5)
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
