# SWIMS CELLECT inference

`scripts/swims_cellect_pipeline.py` runs existing CELLECT `sam_per_band`
checkpoints on the read-only SWIMS images under `/data/shared/SWIMS`.

The pipeline:

1. Discovers the full stacks and/or individual resampled noisy exposures.
2. Cuts the native image grid into 512x512 tiles with configurable overlap.
3. Treats original NaN and exact-zero pixels as invalid, replaces NaN with zero,
   and rejects a tile when its largest connected invalid region exceeds 30%.
4. Applies CELLECT's training-time `astro_zscale_preprocess` on valid pixels and
   restores invalid pixels to zero before inference.
5. Runs ordinal-expectation center detection and samples the predicted ellipse
   shape at each center.
6. Rejects detections within 16 pixels of an internal tile edge, then merges
   overlapping-tile detections within 3 pixels in full-image coordinates.
7. Writes a source CSV, center REG, shape REG, tile manifest, and JSON summary.

During inference, every accepted tile also gets local-coordinate
`tile_xXXXXX_yYYYYY_centers.reg` and `tile_xXXXXX_yYYYYY_shapes.reg` files in
the `tiles/` directory. These are enabled by default, including when tile FITS
are streamed rather than saved. Add `--save-tiles` to place each 512x512 FITS
beside its two matching REG files, or `--no-output-tile-regs` to suppress the
per-tile REG products. Full-image REG files are still written after overlap
deduplication.

Optionally, `--output-masks` runs the SAM mask decoder and writes exactly one
integer instance-label FITS, one label CSV, and one zscale mask-overlay PNG per
accepted tile. Pixel value 0 is background and positive values are local
instance labels. The overlay uses the same palette and default alpha (0.38) as
`visualize_sam_cellect.py`; larger masks are drawn first so smaller detections
remain visible on top. Overlay PNG encoding runs concurrently with subsequent
GPU batches using four threads by default. The full-image source CSV includes
the originating tile mask path and label.

No reference catalog is loaded and no completeness or purity metric is computed.
Field 1 is not interpolated or rotated; its rotated zero borders are handled by
the connected-invalid-area tile filter. This preserves the native PSF and WCS.

## Quick test

Run one stack with the 0712 shape checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 /home/czh23/miniconda3/envs/cellect/bin/python \
  scripts/swims_cellect_pipeline.py \
  --input-kind stack \
  --field-chip field1_chip1 \
  --max-files 1 \
  --checkpoint /data/czh23/ckpts/sam_shape_0712/epoch_0030.pt \
  --out-dir output/swims_shape0712_e30_20260720
```

Run all stacks and noisy exposures with another checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 /home/czh23/miniconda3/envs/cellect/bin/python \
  scripts/swims_cellect_pipeline.py \
  --input-kind all \
  --checkpoint /data/czh23/ckpts/sam_shape_0719/epoch_0030.pt \
  --out-dir output/swims_shape0719_e30_20260720
```

The same command accepts, for example:

```text
/data/czh23/ckpts/sam_zarr_0709/epoch_0030.pt
```

To inspect the accepted cutouts before inference:

```bash
/home/czh23/miniconda3/envs/cellect/bin/python scripts/swims_cellect_pipeline.py \
  --prepare-only \
  --input-kind stack \
  --save-tiles \
  --out-dir output/swims_tiles_20260720
```

By default tiles are streamed and not saved. Use `--save-tiles` only when the
intermediate FITS files are needed, because processing all noisy exposures can
consume several GB.

For direct DS9 inspection of one tile and both of its region layers:

```bash
ds9 output/.../tiles/tile_x00896_y00448.fits \
  -regions load output/.../tiles/tile_x00896_y00448_shapes.reg \
  -regions load output/.../tiles/tile_x00896_y00448_centers.reg
```

Mask output is also disabled by default because decoding hundreds of prompts per
tile is substantially slower. Enable it with:

```bash
CUDA_VISIBLE_DEVICES=0 /home/czh23/miniconda3/envs/cellect/bin/python \
  scripts/swims_cellect_pipeline.py \
  --input-kind stack \
  --checkpoint /data/czh23/ckpts/sam_shape_0712/epoch_0030.pt \
  --output-masks \
  --out-dir output/swims_shape0712_masks_20260720
```

The defaults use point+predicted-shape box prompts, mask-logit threshold 0,
minimum area 15 pixels, maximum area 50% of a tile, and single-mask SAM output.
Use `--overlay-alpha`, `--mask-center-only`, `--mask-multimask`, or the other
`--mask-*` options for ablations.

Set `--overlay-workers N` to control PNG encoding concurrency. The default is
4; use 1 for low-memory systems or 0 for synchronous writing. The pending queue
is bounded to twice the worker count.

## Output coordinates

`x_image` and `y_image` in the CSV are zero-based pixels in the original full
SWIMS FITS image. DS9 REG coordinates are one-based. When a noisy exposure has
no WCS keywords, the pipeline uses the sibling stack WCS because the inputs are
already resampled onto that stack grid. `ra_deg` and `dec_deg` remain NaN if no
usable WCS is available.
