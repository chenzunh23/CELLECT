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

The menu starts with dataset cards:

- `HSC raw tiles`: existing 256x256 pack.zarr workflow.
- `Sitian`: Messier images under `/data/czh23/Messier`; each Messier object is a
  patch, `tract=default`, `band=default`, and tiles are 512x512.
- `HSC coadd/noisy/denoised`: placeholder.
- `ZTF`: placeholder.

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
