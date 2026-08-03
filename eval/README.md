# CELLECT Eval Visualizers

Small scripts for inspecting current CELLECT labels, CELLECT checkpoint outputs,
and native SAM baselines. They intentionally reuse `data_filtering` scaling and
label definitions instead of reimplementing preprocessing.

## 1. Label Visualization

Visualize one direct-zarr sample:

```bash
python eval/visualize_labels.py \
  --zarr-store /data/czh23/direct_zarr_zscore_no_upper/9813/image_level/coadd/HSC-I/4,5.zarr \
  --sample-index 0 \
  --band HSC-I \
  --out-dir output/eval_visualizations/labels
```

Or select from a root:

```bash
python eval/visualize_labels.py \
  --root /data/czh23/direct_zarr_zscore_no_upper \
  --dataset-source coadd \
  --patch 4,5 \
  --tile-name grid_r04_c06_x18108_y21372 \
  --band HSC-I
```

For noisy/denoised variants, add `--group group_00` (or `--group 0`) to select
a specific variant group.

Run coadd, noisy, and denoised together from the same root:

```bash
python eval/visualize_labels.py \
  --root /data/czh23/direct_zarr_zscore_no_upper \
  --dataset-source all \
  --group group_00 \
  --patch 4,5 \
  --tile-name grid_r04_c06_x18108_y21372 \
  --band HSC-I
```

The `--group` filter is applied to noisy/denoised and ignored for coadd.

Outputs include PU-region overlays, GT confidence overlays, source shape REGs,
center REGs, and input-channel heatmaps.

## 2. CELLECT Output Visualization

Run a checkpoint on arbitrary FITS crops with on-the-fly scaling:

```bash
CUDA_VISIBLE_DEVICES=0 python eval/visualize_cellect_outputs.py \
  --checkpoint /data/czh23/ckpts/sam_zscore_no_upper/best.pt \
  --input /data/shared/Subaru/9813/HSC-I/4,5/calexp-HSC-I-9813-4,5.fits \
  --band HSC-I \
  --x0 18108 --y0 21372 \
  --scaling-mode zscore_no_upper \
  --confidence-threshold 2.0
```

For zarr input:

```bash
CUDA_VISIBLE_DEVICES=0 python eval/visualize_cellect_outputs.py \
  --checkpoint /data/czh23/ckpts/sam_zscore_no_upper/best.pt \
  --zarr-store /data/czh23/direct_zarr_zscore_no_upper/image_level/coadd/HSC-I/9813_4,5.zarr \
  --sample-index 0 \
  --zarr-band HSC-I
```

Run the same checkpoint on coadd, noisy, and denoised zarr samples:

```bash
CUDA_VISIBLE_DEVICES=0 python eval/visualize_cellect_outputs.py \
  --checkpoint /data/czh23/ckpts/sam_zscore_no_upper/best.pt \
  --root /data/czh23/direct_zarr_zscore_no_upper \
  --dataset-source all \
  --group group_00 \
  --patch 4,5 \
  --tile-name grid_r04_c06_x18108_y21372 \
  --zarr-band HSC-I
```

Outputs include predicted confidence heatmaps, centers, shapes, mask labelmaps,
mask overlays, REG files, CSV source tables, and model-input heatmaps.

## 3. Native SAM Baseline

Run native SAM AMG with the same scaling definitions:

```bash
CUDA_VISIBLE_DEVICES=0 python eval/native_sam_baseline.py \
  --checkpoint /home/czh23/sam_ckpts/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --input /data/shared/Subaru/9813/HSC-I/4,5/calexp-HSC-I-9813-4,5.fits \
  --band HSC-I \
  --x0 18108 --y0 21372 \
  --points-per-side 64
```

By default this runs four scalings: `zscore_clip`, `zscore_no_clip`,
`log_lupton`, and `anscombe`. Repeat `--scaling-mode` to restrict the set.
