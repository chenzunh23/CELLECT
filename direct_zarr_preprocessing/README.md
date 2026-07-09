# Direct-to-Zarr preprocessing

This is the new preprocessing path for CELLECT training data. It keeps the old
catalog filtering and target painting logic, but bypasses the legacy output tree:

- no `cutouts/`
- no `targets/*.npz`
- no `band_targets/`
- no per-tile FITS catalogs
- no zscale `.pt` cache

Each dataset source and patch is written directly as one Zarr v2 directory:

```text
<OUTPUT_ROOT>/<tract>/coadd/<patch>.zarr
<OUTPUT_ROOT>/<tract>/denoised/<patch>.zarr
<OUTPUT_ROOT>/<tract>/noisy/<patch>.zarr
```

## Filter contract

This script intentionally removes the old `USE_BAND_LIMIT_B_FILTER` switch from
the user interface. The direct path always uses the broad fixed PU B magnitude
range:

```text
B_MAG_MIN=15
B_MAG_MAX=35
```

This is the convention needed to reproduce the old SAM diagnostic tile behavior
where HSC-I patch `9813/4,5`, tile `sam_x18204_y20924`, had about 298 GT sources.
With current refit + remeasurement enabled, this direct path keeps that broad
convention and can add reliable sources on top of it.

## Arrays

The Zarr store contains:

- `/images`: `[N, B, H, W]`, `float16` by default
- `/band_confidence`: `[N, B, H, W]`, `uint8`
- `/band_conf_weight`: `[N, B, H, W]`, `float16`
- `/band_shape`: `[N, B, 3, H, W]`, `float16`
- `/band_shape_weight`: `[N, B, H, W]`, `float16`
- `/band_pu_class_mask`: `[N, B, H, W]`, `uint8`
  - `1`: clean
  - `2`: center only
  - `3`: ordinary ignore
  - `4`: background
  - `5`: strict bright center only
- `/source_centers`, `/source_ids`, `/source_offsets`
- `/tile_x0`, `/tile_y0`, `/tile_name`, `/group`, `/dataset_source`

`source_centers/source_ids` currently store clean source centers. Center-only,
ignore, and background regions are represented in `/band_pu_class_mask`.

## Background Masks

Coadd targets use the official coadd `det-*.fits` footprint background from
`COADD_ROOT`. For denoised/noisy targets, the runner now passes
`VARIANT_LSST_BACKGROUND_ROOT=${COADD_ROOT}/lsst_background_masks` when that
directory exists. With the default `IMAGE_VARIANT_BACKGROUND_SOURCE=auto`,
variant targets prefer:

```text
<VARIANT_LSST_BACKGROUND_ROOT>/<variant>/<tract>/<patch>/<group>/<band>/background_mask.npz
```

and fall back to the coadd background if a variant/group mask is missing.
Set `IMAGE_VARIANT_BACKGROUND_SOURCE=coadd-target` to force old behavior,
`variant-lsst` to require only variant masks, or `none` to disable variant
background labels.

## Full patch command

```bash
COADD_ROOT=/nvme0/zc/scarlet \
CATALOG_ROOT=/nvme0/zc/scarlet \
BAND_CATALOG_ROOT=/nvme0/zc/scarlet \
DENOISED_FITS_ROOT=/nvme0/zc/scarlet/denoised_fits \
REFIT_ROOT=/nvme0/zc/scarlet/refit \
VARIANT_LSST_BACKGROUND_ROOT=/nvme0/zc/scarlet/lsst_background_masks \
IMAGE_VARIANT_BACKGROUND_SOURCE=auto \
OUTPUT_ROOT=/nvme0/zc/scarlet/direct_zarr \
TRACT=9813 \
PATCHES=all \
BANDS="HSC-G HSC-R HSC-I HSC-Z HSC-Y" \
IMAGE_VARIANTS="denoised noisy" \
B_MAG_MIN=15 \
B_MAG_MAX=35 \
PATCH_WORKERS=4 \
TILE_WORKERS=2 \
CHUNK_TILES=2 \
OVERWRITE=0 \
bash direct_zarr_preprocessing/run_direct_zarr.sh
```

This reads full patch images once per band, builds dense targets in memory, and
writes Zarr chunks directly. It does not create intermediate cutout FITS files.

## SAM diagnostic tile

Quick HSC-I coadd-only sanity check:

```bash
COADD_ROOT=/nvme0/zc/scarlet \
CATALOG_ROOT=/nvme0/zc/scarlet \
BAND_CATALOG_ROOT=/nvme0/zc/scarlet \
DENOISED_FITS_ROOT= \
OUTPUT_ROOT=output/direct_zarr_sam_test \
PATCHES="4,5" \
BANDS="HSC-I" \
COMPARE_ORIGIN="18204 20924" \
TILE_FILTER="sam_x18204_y20924" \
INCLUDE_COADD=1 \
IMAGE_VARIANTS="" \
OVERWRITE=1 \
bash direct_zarr_preprocessing/run_direct_zarr.sh
```

In the current workspace this produced `source_centers=339` for the HSC-I SAM
tile, i.e. it preserves the broad old GT convention and adds sources relative to
the earlier 298-source diagnostic.

Inspect the output:

```bash
python zarr_preprocessing/inspect_patch_zarr.py \
  output/direct_zarr_sam_test/9813/coadd/4,5.zarr
```

## Train or eval from Zarr

Use `--data-format zarr` and point `--root` to the direct Zarr root. The
training/eval code will discover `*.zarr` stores and use the adapter in
`astro_train_zarr_data.py`.

```bash
CUDA_VISIBLE_DEVICES=0 python astro_train_eval.py \
  --data-format zarr \
  --mode train \
  --root /nvme0/zc/scarlet/direct_zarr \
  --bands HSC-G HSC-R HSC-I HSC-Z HSC-Y \
  --model-variant per_band \
  --detection-only \
  --enable-en-loss \
  --batch-size 8 \
  --num-workers 8 \
  --pin-memory \
  --out-dir output/zarr_detector_test
```

Current adapter notes:

- self-training pseudo labels are disabled for Zarr mode for now
- ignore/center/background masks are read from `/band_pu_class_mask`
- clean source centers are read from `/source_centers`
- ordinary-ignore source centers are not yet stored in the direct Zarr schema;
  eval still uses ignore/background masks correctly, but ignore-source center
  CSV output should be extended later if needed

## Notes

The first direct version still imports filtering and painting functions from
`astro_data_preprocessing.py` to keep the scientific selection behavior aligned.
The expensive legacy filesystem layout is not used.
