# CELLECT Zarr preprocessing

This folder adds a separate packing stage for existing CELLECT legacy
preprocessing outputs. It does not rerun source filtering. It reads:

- `band_targets/<band>/<tile>.npz`
- `band_tile_metadata/<band>/<tile>.npz`
- precomputed zscale `.pt` files under `ZSCALE_ROOT`

and writes one patch-level Zarr v2 directory per dataset source.

Important: Zarr packing preserves whatever source filtering was already written
by `astro_data_preprocessing.py`. It does not apply `USE_BAND_LIMIT_B_FILTER`,
`PU_B_MAG_MIN`, or any PU filter by itself. Those options must be set when
running `data_preprocessing.sh`.

For the SAM comparison tile used in diagnostics, the old behavior with about
298 HSC-I GT sources on patch `9813/4,5`, tile `sam_x18204_y20924`, requires
turning off the per-band limiting-magnitude B filter:

```bash
USE_BAND_LIMIT_B_FILTER=0
PU_B_MAG_MIN=15
PU_B_MAG_MAX=35
MAG_MIN=15
MAG_MAX=35
```

If `USE_BAND_LIMIT_B_FILTER=1`, faint valid sources outside `[m-5,m)` are moved
to ordinary ignore, which was the reason the same SAM tile dropped below the old
GT count. To reproduce the old 298-source behavior and then add sources from the
new remeasurement logic, run legacy preprocessing with the wide fixed B magnitude
range above, then pack the result to Zarr.

## Output schema

Each patch store contains:

- `/images`: `[N, B, H, W]`, `float16` by default
- `/band_confidence`: `[N, B, H, W]`, `uint8`
- `/band_conf_weight`: `[N, B, H, W]`, `float16` by default
- `/band_shape`: `[N, B, 3, H, W]`, optional
- `/band_shape_weight`: `[N, B, H, W]`, optional
- `/source_centers`: ragged flat source centers, `[total_sources, 2]`
- `/source_ids`: ragged flat source ids, `[total_sources]`
- `/source_offsets`: `[N, B + 1]`; for tile `i`, band `b` uses
  `source_offsets[i,b]:source_offsets[i,b+1]`
- `/tile_x0`, `/tile_y0`
- `/tile_name`, `/group`, `/dataset_source`: fixed-width UTF-8 byte arrays

Sidecars:

- `<patch>.zarr_samples.csv`
- `<patch>.zarr_manifest.json`

## Pack all patches

```bash
PREPROCESSED_ROOT=/nvme0/zc/scarlet/preprocessed \
ZSCALE_ROOT=/nvme0/zc/scarlet/cellect_zscale_cache \
ZARR_ROOT=/nvme0/zc/scarlet/preprocessed_zarr \
PATCHES=all \
PACK_COADD=1 \
IMAGE_VARIANTS="denoised noisy" \
PATCH_WORKERS=4 \
ZARR_TILE_WORKERS=4 \
CHUNK_TILES=8 \
OVERWRITE_ZARR=0 \
bash zarr_preprocessing/pack_all_zarr.sh
```

## Rebuild patch 4,5 with the SAM-tile GT convention, then pack Zarr

This command writes to an isolated debug root. It keeps the current refit and PU
logic, but disables the band-limit B filter so the SAM tile keeps the old broad
GT convention rather than the `[m-5,m)` limited convention.

```bash
DEBUG_ROOT=/nvme0/zc/scarlet/debug_patch45_zarr_298gt

RUN_REFIT=1 \
RUN_PREPROCESS=1 \
REUSE_EXISTING_PREPROCESSED=0 \
SKIP_EXISTING_REFIT=1 \
REFIT_CSV_ONLY=0 \
PATCHES="4,5" \
RAW_ROOT=/nvme0/zc/scarlet \
PREP_ROOT="${DEBUG_ROOT}/preprocessed" \
FAST_ROOT="${DEBUG_ROOT}/preprocessed" \
ZSCALE_ROOT="${DEBUG_ROOT}/cellect_zscale_cache" \
REFIT_ROOT=/nvme0/zc/scarlet/refit \
DENOISED_FITS_ROOT=/nvme0/zc/scarlet/denoised_fits \
IMAGE_VARIANTS="denoised noisy" \
REBUILD_IMAGE_VARIANTS=1 \
MAG_MIN=15 \
MAG_MAX=35 \
PU_B_MAG_MIN=15 \
PU_B_MAG_MAX=35 \
USE_BAND_LIMIT_B_FILTER=0 \
PU_REQUIRE_KRON_REFIT_MATCH=1 \
PU_AP2_KRON_ABS_MAX=1.0 \
PU_B_CLOSE_CENTER_ARCSEC=0.5 \
PU_B_AXIS_RATIO_MAX=5 \
PU_CONTAINMENT_THRESHOLD=0.80 \
PU_B_FLAGS="base_SdssShape_flag base_SdssCentroid_flag" \
PREPROCESS_WORKERS=8 \
VARIANT_PREPROCESS_WORKERS=4 \
PREPROCESS_WORKER_THREADS=1 \
OVERWRITE_ZSCALE=1 \
bash data_preprocessing.sh
```

Then pack coadd, denoised, and noisy into patch-level Zarr stores:

```bash
DEBUG_ROOT=/nvme0/zc/scarlet/debug_patch45_zarr_298gt

PREPROCESSED_ROOT="${DEBUG_ROOT}/preprocessed" \
ZSCALE_ROOT="${DEBUG_ROOT}/cellect_zscale_cache" \
ZARR_ROOT="${DEBUG_ROOT}/preprocessed_zarr" \
PATCHES="4,5" \
PACK_COADD=1 \
IMAGE_VARIANTS="denoised noisy" \
PATCH_WORKERS=3 \
ZARR_TILE_WORKERS=4 \
CHUNK_TILES=8 \
IMAGE_DTYPE=float16 \
TARGET_FLOAT_DTYPE=float16 \
INCLUDE_SHAPE=0 \
OVERWRITE_ZARR=1 \
bash zarr_preprocessing/pack_all_zarr.sh
```

For a quick SAM-tile-only Zarr sanity check:

```bash
DEBUG_ROOT=/nvme0/zc/scarlet/debug_patch45_zarr_298gt

PREPROCESSED_ROOT="${DEBUG_ROOT}/preprocessed" \
ZSCALE_ROOT="${DEBUG_ROOT}/cellect_zscale_cache" \
ZARR_ROOT="${DEBUG_ROOT}/preprocessed_zarr_sam_tile" \
PATCHES="4,5" \
PACK_COADD=1 \
IMAGE_VARIANTS="denoised noisy" \
TILE_FILTER="sam_x18204_y20924 group_00_sam_x18204_y20924 group_01_sam_x18204_y20924 group_02_sam_x18204_y20924 group_03_sam_x18204_y20924" \
OVERWRITE_ZARR=1 \
bash zarr_preprocessing/pack_all_zarr.sh
```

Because coadd and variants have different tile names, the full patch command is
usually safer than the mixed `TILE_FILTER` debug command. Coadd uses
`sam_x18204_y20924`; denoised/noisy use `group_XX_sam_x18204_y20924`.

For a single patch or debugging:

```bash
PATCHES="4,5" TILE_FILTER="group_00_grid_r00_c00_x15900_y19900" \
PACK_COADD=0 IMAGE_VARIANTS=denoised OVERWRITE_ZARR=1 \
bash zarr_preprocessing/pack_all_zarr.sh
```

## Inspect a store

```bash
python zarr_preprocessing/inspect_patch_zarr.py \
  /nvme0/zc/scarlet/preprocessed_zarr/denoised/9813/4,5.zarr
```

## Notes

The writer emits standard Zarr v2 metadata and raw uncompressed chunks directly.
This avoids a deadlock observed with the current `zarr==3.1.6` synchronous API
in the `cellect` environment during local-store creation.
