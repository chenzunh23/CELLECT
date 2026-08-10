# CELLECT Eval Tools

Small scripts for inspecting current CELLECT labels, CELLECT checkpoint outputs,
native SAM baselines, and detector-vs-zarr-label matching. They intentionally
reuse `data_filtering` scaling and zarr label definitions instead of
reimplementing preprocessing.

Run these scripts from the repository root. Use the `cellect` environment unless
you know a different environment has the same dependencies:

```bash
conda run -n cellect python ...
```

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

Important: label visualization reads the dense target already stored in zarr.
It should not be treated as an independent reconstruction of preprocessing
labels.

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

For zarr input, output stems include the zarr scaling stored in `.zattrs`, for
example:

```text
4_5_grid_r04_c06_x18108_y21372_zscore-log-lupton_sam_lupton_epochbest
4_5_grid_r04_c06_x18108_y21372_zscore-no-upper_sam_zscore_no_upper_epochbest
```

When the input is zarr, the script also runs matching automatically under:

```text
<stem>/<band>/matching/
```

The matching GT is taken from the same zarr sample:

- `clean`
- `weak_shape`
- `strict_center_only`

The default match radius is `3 px`.

For `--dataset-source all`, noisy/denoised outputs are placed under the same
coadd-style stem in variant subfolders:

```text
<stem>/<band>/...                 # coadd
<stem>/noisy/group_XX/<band>/...  # noisy
<stem>/denoised/group_XX/<band>/...  # denoised
```

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

This baseline now passes the scaled float image directly to SAM and disables
SAM's default RGB8 `pixel_mean/pixel_std` normalization (`mean=0`, `std=1`).
It therefore tests the actual scaling tensor, not a re-normalized RGB8 preview.

AMG masks larger than 15% of the crop area are dropped by default:

```bash
--max-mask-area-fraction 0.15
```

Set `--max-mask-area-fraction 0` to disable this filter, or increase/decrease
the value for large-source experiments.

## 4. Raw HSC 256 Pack Evaluation

Run a CELLECT checkpoint directly on the raw 256x256 HSC tile pack described in
`/home/czh23/hsctile/HANDOFF.md`.

Single 256x256 frame, using dynamic image-size inference and ViT positional
embedding interpolation:

```bash
CUDA_VISIBLE_DEVICES=0 python eval/eval_hsctile_pack.py \
  --checkpoint /data/czh23/ckpts/sam_zscore_no_upper/best.pt \
  --sample-index 0 \
  --mode single256 \
  --scaling-mode zscore_no_upper
```

Four neighboring 256x256 frames stitched into a native 512x512 input:

```bash
CUDA_VISIBLE_DEVICES=0 python eval/eval_hsctile_pack.py \
  --checkpoint /data/czh23/ckpts/sam_zscore_no_upper/best.pt \
  --band HSC-I \
  --patch 4,5 \
  --tile-id x070_y083 \
  --frame-rank 0 \
  --mode mosaic2x2 \
  --scaling-mode zscore_no_upper
```

The script also accepts `--sample-id`, `--visit`, repeated `--band`, and
`--make-masks`.  For mosaics it prefers the same visit across the four tiles
when possible; add `--strict-visit` to fail instead of falling back to the same
frame rank.

Outputs include model-input heatmaps, confidence overlays, center/shape REGs,
source CSVs, and optional mask overlays under:

```text
output/eval_visualizations/hsctile_pack/<yyyy-mm>/<date>/<stem>/
```

## 5. Matching Existing Outputs

Use `eval/matching.py` when predictions have already been written by
`visualize_cellect_outputs.py` and you want to match them against zarr GT
without rerunning the model.

Example with an explicit prediction CSV:

```bash
conda run -n cellect python eval/matching.py \
  --pred-csv output/eval_visualizations/cellect_outputs/2026-08/2026-08-04/x18108_y21372_anscombe_sam_anscombe_0803_epoch30/HSC-I/HSC-I_sources.csv \
  --root /data/czh23/lupton_zarr_test_0803 \
  --patch 4,5 \
  --tile-name grid_r04_c06_x18108_y21372 \
  --dataset-source coadd \
  --band HSC-I \
  --out-dir output/eval_visualizations/cellect_outputs/matching_debug
```

Or search for `<band>_sources.csv` below an existing stem directory:

```bash
conda run -n cellect python eval/matching.py \
  --output-stem-dir output/eval_visualizations/cellect_outputs/2026-08/2026-08-04/4_5_grid_r04_c06_x18108_y21372_sam_zscore_no_upper_epochbest \
  --root /data/czh23/direct_zarr_v3_zscore_no_upper \
  --patch 4,5 \
  --tile-name grid_r04_c06_x18108_y21372 \
  --dataset-source coadd \
  --band HSC-I
```

Matching outputs:

- `<band>_matching_summary.json`: TP/FP/FN, precision/purity, recall/completeness.
- `<band>_matching_sources.csv`: per-detection and FN rows.
- `<band>_matching_centers.reg`: `circle(...,7)` center regions.
- `<band>_matching_shapes.reg`: predicted and GT shape regions.
- `<band>_matching_overlay.png`: image overlay.
- `<band>_gt_sources.reg`: zarr GT source regions.

REG color and tag conventions:

- TP detections: cyan, `tag={TP center}` / `tag={TP shape}`
- FN labels: red, `tag={FN center}` / `tag={FN shape}`
- Unmatched detections: magenta, `tag={FP center}` / `tag={FP shape}`

`strict_center_only` labels may not have a usable shape; in that case an
unmatched strict-center GT is written as `FN center`.

## 6. Scaling Names

The eval scripts use these command-line names:

- `zscore_clip`: SAM-astro style clipped zscore RGB.
- `zscore_no_clip`: no first sigma clip, then zscore clipped to `[-3, 3]`.
- `zscore_no_upper`: zscore with no upper cap.
- `log_lupton`: zscore/log/lupton RGB.
- `anscombe`: Anscombe RGB.

## 6. Selected 4,5 Samples

Run the hand-picked sample set `50 59 110 114 87` for coadd
`HSC-G HSC-I NB0816 NB1010`, plus noisy broad-band `HSC-G HSC-I` when noisy
zarr stores are available:

```bash
CUDA_VISIBLE_DEVICES=0 bash eval/run_patch45_selected_samples.sh
```

Defaults:

- anscombe v3 checkpoint: `/data/czh23/ckpts/sam_anscombe_0803/epoch_0030.pt`
- zscore-no-upper v3 checkpoint: `/data/czh23/ckpts/sam_zscore_no_upper/epoch_0030.pt`
- original/control checkpoint: `/data/czh23/ckpts/sam_control_0803/epoch_0030.pt`

The original/control branch uses `eval/visualize_multiband_zarr_outputs.py`.
That old zarr stores five broad bands together, but the model is still
`sam_per_band`; the script extracts one band at a time and runs single-band
inference.

Useful overrides:

```bash
SAMPLES="50 59" \
NOISY_GROUPS="group_00 group_01" \
SKIP_EXISTING=1 \
MAKE_MASKS=0 \
bash eval/run_patch45_selected_samples.sh
```

By default, existing outputs with `<band>/matching/<band>_matching_summary.json`
are skipped. Missing zarr stores or missing bands are also skipped, so the same
script can be reused before noisy/denoised products are complete.

For FITS input, scaling is built on the fly from the crop. For zarr input, the
script reads the stored image tensor and visualizes exactly what training used.
