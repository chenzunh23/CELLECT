# CELLECT Astronomy Workflow

This repository contains the astronomy adaptation of CELLECT for HSC/LSST
cutout detection, SAM-assisted mask prediction, and photometry evaluation.  The
main workflow is:

1. Optionally refit Kron/shape proxies from HSC `meas` catalogs.
2. Build coadd training cutouts, catalogs, dense targets, and zscale caches.
3. Build denoised/noisy image-variant targets, including per-group labels.
4. Train or evaluate AstroCELLECT/SAM-CELLECT models.
5. Run patch-level photometry evaluation and magnitude-binned diagnostics.

The actively maintained entry points are:

```text
data_preprocessing.sh                  End-to-end refit + preprocessing wrapper
astro_data_preprocessing.py             Preprocessing implementation
astro_train_eval.py                     Train/evaluate AstroCELLECT and SAM-CELLECT
evaluate_sam_cellect_photometry.py      Patch/tile photometry evaluation
zangetsu_demo/visualize_sam_cellect.py  Small-tile visualization and mask overlays
scripts/swims_cellect_pipeline.py       SWIMS tiling, source detection, REG, and optional masks
```

## SWIMS Inference

`scripts/swims_cellect_pipeline.py` applies existing CELLECT SAM checkpoints to
SWIMS stacked or single-exposure FITS images without requiring a reference
catalog. It rotates neither field by default, rejects 512x512 tiles whose
largest connected invalid region exceeds 30%, writes source CSV and DS9 REG
files, and can write an instance-mask FITS, label CSV, and zscale overlay PNG
per accepted tile with `--output-masks`. Overlay order places smaller masks on
top of larger overlapping detections.

See [docs/swims_cellect.md](docs/swims_cellect.md) for checkpoint presets,
input discovery, tiling rules, and runnable examples.

## Environment

Most commands assume the `cellect` conda environment:

```bash
conda activate cellect
cd /home/czh23/CELLECT
```

The examples below use the current local data layout:

```text
/nvme0/zc/scarlet/                         Raw HSC/coadd data root
/nvme0/zc/scarlet/preprocessed/            Preprocessed metadata, targets, catalogs
/nvme0/zc/scarlet/cellect_zscale_cache/    Precomputed zscale image tensors
/nvme0/zc/scarlet/denoised_fits/           Denoised/noisy full-patch FITS
/nvme0/zc/scarlet/lsst_background_masks/   LSST-detection background masks
```

Default bands are:

```text
HSC-G HSC-R HSC-I HSC-Z HSC-Y
```

## Data Layout

Coadd preprocessing writes one tree per tract/patch:

```text
<preprocessed>/<tract>/<patch>/
  cutouts/
  reference_catalogs/
  reference_catalogs_csv/
  center_only_catalogs/
  ignore_catalogs/
  strict_center_only_catalogs/
  band_reference_catalogs/<band>/
  band_reference_center_only/<band>/
  band_reference_ignore/<band>/
  targets/
  band_targets/<band>/
  tile_metadata/
  band_tile_metadata/<band>/
  manifest.json
  tiles.csv
  cutout_paths.json
```

Denoised/noisy variants are stored under a variant prefix and include group
names in tile IDs:

```text
<preprocessed>/denoised/<tract>/<patch>/
<preprocessed>/noisy/<tract>/<patch>/

tile example:
group_01_grid_r08_c01_x20268_y30844
```

Training discovery supports all of these forms through `--root
/nvme0/zc/scarlet/preprocessed`.

## Refit

The wrapper `data_preprocessing.sh` can run the batch HeavyFootprint Kron refit
before preprocessing.  Refit is controlled by:

```text
RUN_REFIT=1                 Enable refit stage
REFIT_WORKERS=...           Patch/band parallelism
SKIP_EXISTING_REFIT=1       Skip completed refit outputs
REFIT_CSV_ONLY=1            Write compact CSV-oriented outputs
REFIT_INCLUDE_SHAPE_FLAGGED=0
REFIT_INCLUDE_CENTROID_FLAGGED=0
```

Example:

```bash
PATCHES="4,5 6,3" \
RUN_REFIT=1 \
RUN_PREPROCESS=0 \
REFIT_WORKERS=4 \
bash data_preprocessing.sh
```

The refit implementation is in:

```text
batch-heavyfp-kron-refit/batch_heavyfp_kron_refit.py
```

Refit products are used by preprocessing to improve source selection, Kron
proxy filtering, and dense shape labels.  If the refit outputs already exist,
keep `SKIP_EXISTING_REFIT=1` to avoid unnecessary work.

## Coadd Preprocessing

For a full preprocessing pass:

```bash
PATCHES="4,5 6,3" \
RUN_REFIT=1 \
RUN_PREPROCESS=1 \
PREPROCESS_WORKERS=8 \
PREPROCESS_WORKER_THREADS=1 \
bash data_preprocessing.sh
```

Important wrapper variables:

```text
TRACT=9813
RAW_ROOT=/nvme0/zc/scarlet
PREP_ROOT=/nvme0/zc/scarlet/preprocessed
FAST_ROOT=/nvme0/zc/scarlet/preprocessed
ZSCALE_ROOT=/nvme0/zc/scarlet/cellect_zscale_cache
PATCHES="all" or "4,5 6,3"
PATCH_FILE=/path/to/patches.txt
BANDS="HSC-G HSC-R HSC-I HSC-Z HSC-Y"
PREPROCESS_WORKERS=8
PREPROCESS_WORKER_THREADS=1
```

Useful preprocessing modes:

```text
RUN_REFIT=0                         Do not rerun Kron/shape refit
REUSE_EXISTING_PREPROCESSED=1       Reuse existing coadd tree
REBUILD_IMAGE_VARIANTS=1            Rebuild denoised/noisy targets only
OVERWRITE_ZSCALE=1                  Regenerate zscale cache
SKIP_CUTOUTS=1                      Reuse existing FITS cutouts
```

The wrapper writes:

```text
output/data_preprocessing_logs/
preprocess_manifest.json
preprocess_failed_patches.json
preprocess_failed_patches.csv
```

If preprocessing fails for a subset of patches, inspect
`preprocess_failed_patches.csv` and rerun only those patches.

## Denoised/Noisy Variant Preprocessing

Set `DENOISED_FITS_ROOT` to enable variant preprocessing:

```bash
PATCHES="1,4 2,4 3,4 3,5 4,1 4,2 4,5 4,6 4,7 5,1 5,2 5,3 5,5 5,6 5,7 6,2 6,3 6,4 7,4 8,4" \
RUN_REFIT=0 \
REUSE_EXISTING_PREPROCESSED=1 \
REBUILD_IMAGE_VARIANTS=1 \
DENOISED_FITS_ROOT=/nvme0/zc/scarlet/denoised_fits \
IMAGE_VARIANTS="denoised noisy" \
PREPROCESS_WORKERS=16 \
VARIANT_PREPROCESS_WORKERS=8 \
PREPROCESS_WORKER_THREADS=1 \
bash data_preprocessing.sh
```

To refresh only denoised targets with LSST background masks:

```bash
PATCHES="1,4 2,4 3,4 3,5 4,1 4,2 4,5 4,6 4,7 5,1 5,2 5,3 5,5 5,6 5,7 6,2 6,3 6,4 7,4 8,4" \
RUN_REFIT=0 \
REUSE_EXISTING_PREPROCESSED=1 \
REBUILD_IMAGE_VARIANTS=1 \
DENOISED_FITS_ROOT=/nvme0/zc/scarlet/denoised_fits \
IMAGE_VARIANTS="denoised" \
VARIANT_LSST_BACKGROUND_ROOT=/nvme0/zc/scarlet/lsst_background_masks \
NONCOADD_SNR_FILTER=1 \
PREPROCESS_WORKERS=16 \
VARIANT_PREPROCESS_WORKERS=8 \
PREPROCESS_WORKER_THREADS=1 \
bash data_preprocessing.sh
```

Variant targets reuse the coadd catalogs as the GT source list, then apply
non-coadd visibility filtering:

```text
SNR < 2       ignore
2 <= SNR < 3 center-only
SNR >= 3     clean GT
```

The thresholds are configurable:

```text
NONCOADD_SNR_IGNORE_THRESH=2.0
NONCOADD_SNR_CENTER_ONLY_THRESH=3.0
NONCOADD_SNR_AP_RADIUS=6.0
NONCOADD_SNR_ANNULUS_R_IN=10.0
NONCOADD_SNR_ANNULUS_R_OUT=15.0
```

If `VARIANT_LSST_BACKGROUND_ROOT` is set, variant targets use masks from:

```text
<root>/<variant>/<tract>/<patch>/<group>/<band>/background_mask.npz
```

before falling back to the coadd target background mask.

## Patch and Group Selection

Patch files can be used for train/val/preprocessing lists.  In training, source
qualifiers and group selectors are supported:

```text
4,5
9813/4,5
denoised:4,5
noisy:9813/6,3
denoised:4,5@group_01
noisy:6,3@random
```

If only a patch ID is given, all available groups for denoised/noisy records are
eligible unless another dataset filter restricts them.

## SAM-CELLECT Training

Recommended SAM training command shape:

```bash
CUDA_VISIBLE_DEVICES=0 python astro_train_eval.py \
  --mode train \
  --model-variant sam_per_band \
  --sam-model-type vit_b \
  --sam-checkpoint /home/czh23/sam_ckpts/sam_vit_b_01ec64.pth \
  --root /nvme0/zc/scarlet/preprocessed/ \
  --image-cache-dir /nvme0/zc/scarlet/cellect_zscale_cache/ \
  --targets-dir /nvme0/zc/scarlet/preprocessed/ \
  --bands HSC-G HSC-R HSC-I HSC-Z HSC-Y \
  --dataset-sources coadd denoised noisy \
  --train-patches-file train_patches.txt \
  --val-patches-file val_patches.txt \
  --out-dir /nvme0/zc/scarlet/ckpts/example_run \
  --wandb-run-name example_run \
  --epochs 100 \
  --batch-size 4 \
  --num-workers 4 \
  --pin-memory \
  --persistent-workers \
  --prefetch-factor 2 \
  --lr 5e-5 \
  --sam-encoder-lr 1e-5 \
  --sam-warmup-ratio 0.01 \
  --sam-lr-drop-fractions 0.70 0.90 \
  --sam-lr-drop-gamma 0.1 \
  --confidence-loss-mode ce_hard \
  --confidence-ce-weights 1 4 8 16 32 \
  --shape-loss-weight 2 \
  --center-only-shape-factor 0.2 \
  --small-shape-loss-weight 0.05 \
  --small-shape-area-min 20 \
  --small-shape-area-tau 5 \
  --small-shape-ordinal-threshold 2.0 \
  --small-shape-scope ignore \
  --mask-loss-warmup-epochs 8 \
  --mask-prompt-gt-epochs 10 \
  --mask-prompt-pred-epoch 30 \
  --mask-loss-weight 5.0 \
  --mask-centroid-weight 0.4 \
  --mask-outside-weight 1.0 \
  --mask-min-area-weight 1.0 \
  --mask-stability-weight 0 \
  --mask-selection loss \
  --mask-max-gt-per-sample 128 \
  --mask-max-pred-per-sample 128 \
  --mask-prompt-chunk-size 1024 \
  --confidence-score ordinal_expectation \
  --amp bf16 \
  --wandb-log-interval 10 \
  --ckpt-interval 2 \
  --ddp-static-graph auto \
  --linking-metrics-json output/training_logs/example_run_linking_metrics.json
```

Key training concepts:

```text
--dataset-sources              Select coadd/denoised/noisy trees.
--confidence-loss-mode         ordinal_legacy or ce_hard.
--confidence-ce-weights        Class weights for ce_hard level 0..4.
--shape-loss-mode              source_center (per-source core) or dense_pixel.
--shape-center-size            Odd source-core size, normally 3 or 5.
--shape-geometry-loss          legacy_area_ratio or matrix-free log_spd.
--small-shape-loss-weight      Epoch-0 loss suppressing high ordinal confidence
                               for tiny shapes in ignore/non-clean regions.
--use-ordinal-expectation      Use ordinal expectation for center detection.
--confidence-score             Score map used for detection and pred prompts.
--mask-prompt-gt-epochs        Use GT prompts before this epoch.
--mask-prompt-pred-epoch       Fully predicted prompts after this epoch.
--sam-lr-phase2-epoch          Optional later LR override.
--sam-head-lr-after            Use 0 to freeze proposal head after phase2.
```

When `--confidence-loss-mode ce_hard` is used, `--confidence-pos-weight` no
longer affects the confidence loss.  It is only used by `ordinal_legacy`.

For DDP:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 astro_train_eval.py ...
```

`--batch-size` is per rank, so the global batch size is:

```text
global_batch = batch_size * number_of_ranks
```

## Validation During Training

`--mode train` performs validation every epoch using the train/val split or the
explicit patch lists:

```text
--train-patches-file train_patches.txt
--val-patches-file val_patches.txt
--val-patches 4,5 6,1
```

Checkpoints and metadata are written to `--out-dir`:

```text
run_config.json
last.pt
best.pt
epoch_0002.pt
epoch_0004.pt
...
linking_metrics_latest.json
linking_metrics_epoch_XXXX.json
```

Use `--debug-batch-start N --debug-batch-end M` to print per-stage timing:

```text
data_wait
h2d
forward
dense_loss
mask_loss
backward
step
```

This is useful for distinguishing I/O bottlenecks from SAM mask-loss cost.

## Standalone Evaluation

Use `astro_train_eval.py --mode eval` to run detection metrics and optionally
write per-source detections:

```bash
CUDA_VISIBLE_DEVICES=0 python astro_train_eval.py \
  --mode eval \
  --model-variant sam_per_band \
  --sam-model-type vit_b \
  --checkpoint /nvme0/zc/scarlet/ckpts/example_run/epoch_0018.pt \
  --root /nvme0/zc/scarlet/preprocessed/ \
  --image-cache-dir /nvme0/zc/scarlet/cellect_zscale_cache/ \
  --targets-dir /nvme0/zc/scarlet/preprocessed/ \
  --bands HSC-G HSC-R HSC-I HSC-Z HSC-Y \
  --dataset-sources coadd denoised noisy \
  --eval-patches 4,5 \
  --confidence-score ordinal_expectation \
  --out-dir /nvme0/zc/scarlet/eval/example_run_epoch0018 \
  --eval-sources-csv output/training_logs/example_eval_sources.csv \
  --linking-metrics-json output/training_logs/example_linking_metrics.json \
  --amp bf16
```

Useful eval flags:

```text
--debug-detection-metrics       Candidate-stage histograms and retention counts.
--debug-ordinal-expectation     Save ordinal-expectation debug maps.
--center-refinement softargmax  Sub-pixel center refinement.
--nms-radius 1                  Local-max radius.
--match-radius 2.976            Pixel match radius.
```

## Photometry Evaluation

`evaluate_sam_cellect_photometry.py` evaluates one or more checkpoints across
all discovered tiles in a patch/group and generates magnitude-binned plots.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_sam_cellect_photometry.py \
  --ckpt-dir /nvme0/zc/scarlet/ckpts/example_run/ \
  -c epoch_0018.pt -l epoch_0018 \
  --data-root /nvme0/zc/scarlet/preprocessed \
  --patch 4,5 \
  --group 01 \
  --datasets coadd denoised noisy \
  --out-dir zangetsu_demo/output/example_run/patch45_photometry_eval \
  --tile-workers 4
```

Important photometry flags:

```text
--gt-visibility-filter raw|snr_ge2|snr_ge3
--tile-workers N
--tile-worker-devices cuda:0 cuda:1
--min-mask-area 15
--max-mask-area-ratio 0.50
--confidence-score ordinal_expectation
--threshold ...
```

The default visibility filter is `snr_ge2`, so completeness is measured against
sources that are visible enough in the preprocessed target labels.  The output
directory contains per-tile CSV files, aggregate CSVs, completeness/purity plots,
flux-ratio histograms, and reference GT count plots.

## Small-Tile Visualization

For the Zangetsu/SAM demo cutout, use:

```bash
CUDA_VISIBLE_DEVICES=0 python zangetsu_demo/visualize_sam_cellect.py \
  --band HSC-I \
  --ckpt-dir /nvme0/zc/scarlet/ckpts/example_run/ \
  --out-dir zangetsu_demo/output/example_run/sam \
  -c epoch_0018.pt \
  -l epoch_0018 \
  --native-sam-dir /home/czh23/CELLECT/zangetsu_demo/output/native_sam_astro_vit_b_64_coadd_HSC-I \
  --native-sam-dataset coadd \
  --native-match-radius 2.9761904761904763 \
  --variant-group group_01 \
  --gt-visibility-filter snr_ge2
```

For peak-stage region overlays:

```bash
python zangetsu_demo/export_peak_stage_regs.py \
  --checkpoint /nvme0/zc/scarlet/ckpts/example_run/epoch_0018.pt \
  --dataset denoised
```

For confidence/background overlays:

```bash
python zangetsu_demo/export_confidence_map_overlays.py \
  --checkpoint /nvme0/zc/scarlet/ckpts/example_run/epoch_0018.pt \
  --dataset denoised
```

## Restoring or Refreshing a Patch

To restore a patch from a remote backup, use trailing slashes so `rsync` copies
the directory contents rather than nesting another `4,5` directory:

```bash
rsync -a --delete -e 'ssh -i ~/.ssh/id_rsa6 -p 8210' \
  czh23@101.6.89.15:/data/czh23/preprocessed/9813/4,5/ \
  /nvme0/zc/scarlet/preprocessed/9813/4,5/
```

If the remote source path does not exist, rsync fails before reading files from
the sender and does not modify the remote directory.

After restoring coadd catalogs, refresh denoised/noisy targets with:

```bash
RUN_REFIT=0 \
REUSE_EXISTING_PREPROCESSED=1 \
REBUILD_IMAGE_VARIANTS=1 \
DENOISED_FITS_ROOT=/nvme0/zc/scarlet/denoised_fits \
IMAGE_VARIANTS="denoised" \
VARIANT_LSST_BACKGROUND_ROOT=/nvme0/zc/scarlet/lsst_background_masks \
PATCHES="4,5" \
bash data_preprocessing.sh
```

## Common Checks

List patches modified after a date:

```bash
find /nvme0/zc/scarlet/preprocessed -type f -newermt '2026-06-30 00:00:00'
```

Check preprocessing records discovered by training:

```bash
python astro_train_eval.py \
  --mode eval \
  --root /nvme0/zc/scarlet/preprocessed \
  --image-cache-dir /nvme0/zc/scarlet/cellect_zscale_cache \
  --dataset-sources coadd denoised noisy \
  --eval-patches 4,5 \
  --max-records 4 \
  --wandb-mode disabled
```

Profile a slow training batch:

```bash
--debug-batch-start 20 --debug-batch-end 25
```

If `data_wait` is low but `mask_loss` is high, the bottleneck is usually SAM
mask decoding or prompt count, not file I/O.

## Related Documents

Detailed background and older notes are kept in:

```text
ASTRO_DATA_WORKFLOW.md
SAM_TRAINING_GUIDE.md
DETECTION_ONLY_WORKFLOW.md
SCARLET_CATALOG_FLAGS.md
batch-heavyfp-kron-refit/README.md
```
