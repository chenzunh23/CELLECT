# HSC Tile Pack Evaluation

This folder contains CELLECT-side tools for interactive evaluation of image tile
datasets. The browser normalizes each dataset to a common
`tract/patch/band/tile/group` interface.

## Offline Evaluation

Run one tile directly:

```bash
CUDA_VISIBLE_DEVICES=0 python eval/hsctiles/eval_hsctile_pack.py \
  --checkpoint /data/czh23/ckpts/sam_anscombe_0803/epoch_0030.pt \
  --root /data/zc/Subaru/data/hsctile/pack_full_9813_256/9813 \
  --tile-id x004_y009 \
  --patch 4,5 \
  --band HSC-I --band HSC-G --band HSC-Y --band NB1010 --band NB0816 \
  --mode single256 \
  --scaling-mode anscombe \
  --no-shape-overlay-centers
```

Run a 2x2 mosaic:

```bash
CUDA_VISIBLE_DEVICES=0 python eval/hsctiles/eval_hsctile_pack.py \
  --checkpoint /data/czh23/ckpts/sam_anscombe_0803/epoch_0030.pt \
  --root /data/zc/Subaru/data/hsctile/pack_full_9813_256/9813 \
  --tile-id x004_y009 \
  --patch 4,5 \
  --band HSC-I \
  --mode mosaic2x2 \
  --scaling-mode anscombe
```

`--visit` is optional. If omitted, the scripts use `--frame-rank` or random frame selection depending on mode/options.

## Interactive Browser

FITS loading uses `--image-workers 4` concurrent CPU readers by default.
`--scaling-workers 2` separately limits expensive input-scaling work (including
4096-parent statistics). Add these options to the existing server command and
restart the server to change them; refreshing the browser alone does not reload
Python code. Keep the existing checkpoint, port and dataset arguments.

Raw reads no longer share a global lock with input scaling. Concurrent requests
for the same cached image or parent statistics share one computation. Detect
also prepares its current batch in parallel; these options do not add GPU model
workers. FITS remains the source, and display scaling is still adjustable. Array
cache memory is controlled by `--browser-cache-mb` (128 MiB by default); temporary
parent-fitting arrays require additional memory, hence the separate scaling cap.

Slow operations (at least one second) print `[image-read]`, `[image-scale]`, and
`[parent-scaling]` logs with queue `wait` and actual `work` times. `[image-png]`
reports the total image-handler time, including rendering. A first uncached
parent fit still takes time: extra workers reduce independent-request queueing,
not necessarily the latency of one isolated request. Start with 4 readers and
2 scaling workers rather than increasing both aggressively.

Concurrency regressions can be checked with:

```bash
python -m unittest eval.tests.test_parallel_image_load -v
```

Start the browser server:

```bash
CUDA_VISIBLE_DEVICES=0 python eval/hsctiles/serve_hsctile_pack_browser.py \
  --host 0.0.0.0 \
  --port 8050 \
  --data-root /data/zc/Subaru/data/hsctile/pack_full_9813_256/9813 \
  --checkpoint /data/czh23/ckpts/sam_anscombe_0803/epoch_0030.pt
```

The browser UI is split into static page files:

```text
eval/hsctiles/pages/index.html
eval/hsctiles/pages/style.css
eval/hsctiles/pages/app.js
```

The menu groups datasets behind four main buttons:

- **HSC**: raw 256×256 packs, or coadd/noisy/denoised images.
- **JWST NIRCam**: existing fields, COSMOS 1727/5893, or Abell 2744 half coadds.
- **Sitian**: Messier images; each object is a patch, `tract=default`.
- **ZTF**: the existing ZTF image loader.

HSC and JWST open a subtype chooser before selecting patches and bands.

For Sitian/Messier, the loader defaults to each object's `all/*.tif(f)` stack so
different depth stacks are not mixed as groups. If no `all/` TIFF exists for an
object, it falls back to the other image files under that object. Default tile
selection smooths that full `all/` image and picks up to four 512x512 crops
centered on the brightest de-duplicated peaks with a 256-pixel exclusion radius.
`--messier-tile-mode random_grid` switches to random 512x512 grid tile
selection, and `max` loads all grid tiles.

Example with Messier/Sitian enabled:

```bash
CUDA_VISIBLE_DEVICES=0 python eval/hsctiles/serve_hsctile_pack_browser.py \
  --host 0.0.0.0 \
  --port 8050 \
  --data-root /data/zc/Subaru/data/hsctile/pack_full_9813_256/9813 \
  --messier-root /data/czh23/Messier \
  --checkpoint /data/czh23/ckpts/sam_anscombe_0803/epoch_0030.pt
```

The default browser configuration is model/scaling/visualization related, not visit-specific:

- checkpoint: `/data/czh23/ckpts/sam_anscombe_0803/epoch_0030.pt`
- scaling: `anscombe`
- bands: `HSC-I HSC-G HSC-Y NB1010 NB0816`
- center crosses on shape overlays: disabled by default
- detect batch size: `20` tile-slots per model forward pass

The menu page can override:

- `n-tiles`: number of spatial 256x256 tiles to browse
- `groups per tile`: number of frame groups to display per tile
- `tiles per page`: number of spatial tiles per browser page
- `run name`: optional label used for storage directories. Sessions and exports
  are written as `<run_name>_<YYYYmmdd_HHMMSS>`; if omitted, the timestamp alone
  is used.

If a tile has fewer frame groups than requested, only the available groups are
shown.

The page number input jumps directly when Enter is pressed. The `Search` button
opens a full-screen dialog:

- `tile x,y`: jumps to a loaded tile by tile id. HSC raw ids accept unpadded
  `4,9` and padded `004,009` forms.
- `pixel x,y`: jumps to the loaded tile containing a full-image pixel
  coordinate.

Search only covers tiles loaded into the current session. Use `max` on the menu
page when coordinate search needs to cover every available tile.

The browser has independent detection and view controls:

- `Detect`: runs CELLECT on the current page and toggles cached detection
  visualization for that page. Changing page/patch resets this button to
  `Detect`.
- `View` opens a DS9-style menu. `Shape` is checked by default; `Input Scaling`
  and `Center` are off by default.
- `Input Scaling`: switches the displayed background to the selected model-input
  channel without running detection.
- `Shape`: draws predicted ellipses when detections are visible.
- `Center`: draws yellow `+` center marks when detections are visible.
- `Smooth`: opens a display-only smoothing dialog. It does not change model
  inference or exported products. Gaussian uses `r=ceil(2*sigma)` and
  `D=2*r+1`; Boxcar and Tophat use the chosen radius directly with
  `D=2*r+1`.

Selected exports keep their existing format and do not depend on the current
browser `View` menu state.

Selected exports are stored under:

```text
<export_dir>/<tract>/<patch>/<band>/<tile_id>/
```

With the default `--export-dir`, `<export_dir>` is already a timestamped
directory under `eval/hsctiles/interactive_selected/`.

Each exported tile includes raw PNG/NPZ, detection CSV, raw-background detection
overlay, and input-shape overlay.

## COSMOS / Abell and grouped dataset menu

The landing page now has **HSC**, **JWST NIRCam**, **Sitian**, and **ZTF** buttons.
HSC opens a chooser for raw tiles / coadd-noisy-denoised; JWST opens a chooser for
existing NIRCam fields / COSMOS 1727 & 5893 / Abell 2744 half coadd. Selecting a
subtype updates available patches, bands and defaults. Escape or Close cancels.

```bash
cd /home/czh23/CELLECT
/home/czh23/miniconda3/envs/cellect/bin/python eval/hsctiles/serve_hsctile_pack_browser.py \
  --checkpoint /path/to/epoch_011.pt --dataset jwst_cosmos \
  --host 127.0.0.1 --port 8788
```

Restart the Python server and refresh the page after updating. Existing launch
scripts and dataset IDs continue to work. New IDs are `jwst_cosmos` and
`jwst_abell`; no separate server process is required per dataset.

- COSMOS defaults to `/data/czh23/JWST/COSMOS_1727_5893`, overridable with
  `--cosmos-root`. Patch names such as `p1727_P0019` / `p5893_P0019` keep proposals
  separate. The formal manifest selects the current final products.
- Abell defaults to `/data/czh23/JWST/Abell2744_preprocessing`, overridable with
  `--abell-plan`. Patches such as `x+00_y+00` are the aligned 4096 parents. Reads
  half-coadd parents or reconstructs a bounded window from the original half
  mosaic when needed; the full coadd is only a WCS reference.
- All 20 available Abell bands and all formal COSMOS bands are discovered.
  Missing bands in a patch are skipped. All selected, present bands must share
  at least 90% valid pixels in a tile; reduce the band selection if their
  footprints do not overlap.
- New COSMOS/Abell datasets use preprocessing's parent-based three-channel RGB
  normalization, for both Detect and Input Scaling. Generic NIRCam and HSC retain
  existing scaling options. First access to each parent fits its statistics;
  subsequent windows reuse them. Browser pixels come from FITS; use the CLI's
  Zarr mode for exact stored training tensors (see `eval/README.md`).

## Fast FITS browsing and SAM masks

FITS remains the default for all JWST datasets; no Zarr conversion is required.
Unscaled science images use memory-mapped rectangular reads. Scaled integer or
compressed HDUs use safe FITS sections. Candidate validity and display reuse the
same projected pixels. Compatible TAN WCS pairs use an exact projective matrix;
other projections/distortions still use Astropy. The cache preserves linear
pixel intensities, so display scaling and inversion remain adjustable.

The session has a byte-bounded raw/scaled cache (`--browser-cache-mb 128`), plus
bounded projected-tile and mask caches. No complete mosaic is copied into these
caches. COSMOS/Abell parent RGB statistics are also cached as small JSON files in
`~/.cache/cellect/jwst_parent_scaling/`; changing FITS/quality files or scaling
code invalidates the key. First access to an uncached parent still fits its
statistics. To deliberately use the browser's per-crop detector normalization
instead of training-parent RGB, set `--jwst-input-scaling browser --scaling-mode
anscombe` (or another supported scaling mode). This affects model input, while
View controls only change the displayed image.

After **Detect**, open **View → Show Masks**:

- SAM masks and their tight bounding boxes are green. The mask is **70%
  transparent (alpha=0.30)**; overlaps blend once rather than darkening repeatedly.
- Mask decoding reuses the exact detection image embeddings, with default
  `--mask-chunk-size 32`, `--mask-threshold 0`, `--mask-box-scale 2`. No image
  encoder rerun is needed unless a result was evicted from the bounded mask cache.
- Boxes are computed from predicted mask support; they are not the Kron prompt
  boxes. The existing shape/center overlays can be toggled independently.
- **Export Selected** additionally writes `<candidate>_mask_overlay.png` and
  includes its path in the export manifest. **Save CSV** also saves mask PNGs
  beside `selection.csv` for selected images that have already been detected.
- Saved mask PNGs always use **non-inverted ZScale**, with green masks and boxes;
  they do not inherit display inversion, smoothing, custom limits, shape overlays
  or SNR filtering. Full floating-point mask arrays are not exported.

Mask decoding adds work to Detect, especially in crowded fields. Use
`--no-make-masks` for an explicitly detection-only session. Reload the server and
refresh the browser after updating these files.

Regression checks:

```bash
/home/czh23/miniconda3/envs/cellect/bin/python -m unittest \
  eval.tests.test_jwst_fields eval.tests.test_browser_fast_masks -v
node eval/tests/test_dataset_menu.js
```
