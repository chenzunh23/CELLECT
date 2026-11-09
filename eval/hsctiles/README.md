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

All browser instances now share a lightweight entry at `http://127.0.0.1:8800/`.
Keep using distinct `--port` values for independent experiments; each new browser
automatically registers and starts/reuses the portal. For example, browsers on
8787 and 8813 are accessible at `/browser/8787/` and `/browser/8813/` on port 8800.
Different tabs retain their own backend, including Detect, masks, SNR, exports,
and data-quality pages. The portal does not load a model or use a GPU.

Forward **only remote port 8800** using the VS Code Ports panel, then use its
Forwarded Address in a local browser. The workspace settings enable automatic
forwarding and restoring previous forwards. VS Code may choose a different
local port if 8800 is occupied; use the address shown in the Ports panel.
An existing SSH/VS Code connection still carries the traffic: listening on
`0.0.0.0` alone does not create a connection from a personal computer.

If using ordinary SSH instead of VS Code, put the following in the **personal
computer's** `~/.ssh/config`, under the existing host entry. It reuses each normal
SSH login instead of requiring one extra terminal/tunnel per experiment:

```sshconfig
Host opus
    HostName 101.6.89.15
    Port 8210
    User czh23
    LocalForward 127.0.0.1:8800 127.0.0.1:8800
```

Previously running browser processes can be registered without restarting them:

```bash
python eval/hsctiles/browser_portal.py --backend '8787=sky_1104 epoch21'
```

This foreground command starts the portal; it is needed only if no portal is
already running. If it is already running, register an old process from the
repository root without starting a second portal:

```bash
python -c "from eval.hsctiles.browser_portal import register_backend; register_backend(8813, 'another experiment')"
```

Registration and portal logs live in `~/.cache/cellect/browser_portal/`.
The default portal listens only on loopback and only proxies registered local
ports. It stays alive when one experiment closes. A stopped backend gives an
explicit 502 error; it cannot silently switch to another experiment.
Use `--portal-port 0` to disable auto-registration/startup, or choose another
shared port with `--portal-port PORT` for all instances.
Changing backend code requires restarting that backend; the portal can proxy
already running older browsers without restart. HTTP/API/image URLs are
translated at the portal; raw FITS paths and image bytes remain unchanged.

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

COSMOS is an exception: every tile in the selected pointing(s) is indexed from
FITS headers, regardless of the tile-count setting. Search and page jumps can
reach the entire mosaic without scanning science pixels for coverage. Empty and
partially covered edge tiles remain searchable. On opening a page, only its
4096×4096 parent is read for each requested band; neighboring pages reuse those
pixels. Pages stay within a parent, so a page may contain fewer tiles than the
configured maximum. The browser receives the visible 512×512 PNGs, not the full
mosaic. The byte-bounded parent cache defaults to 320 MiB (four float32 science
and boolean validity parents), adjustable with `--cosmos-parent-cache-mb`.
FITS remain the image source and display scaling remains adjustable. This lazy
index is browser-only; offline coverage filtering is unchanged.

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

Selected export now uses a bounded CPU thread pool, default **4 workers**.
Set `--export-workers 4` in the server command, or change **Export workers**
beside **Export Selected** for the next export. This is independent of
`--image-workers` and `--scaling-workers`. Encoder and mask decoding remain
serialized on the existing GPU model; CPU workers render/write PNG/NPZ/CSV
and calculate SNR. Detection runs in bounded chunks, with masks retained until
their export finishes, avoiding re-inference caused by mask-cache eviction.
AP2 and Kron share the same 65×65 median background. The fixed-aperture
angular-size defaults were revised on 2026-10-22 as described below; export
file layouts and the two noise estimators remain unchanged. Tiles with no detections skip SNR
calculation. Manifest/selection order remains stable across worker counts.

The button immediately displays `Exporting ...` while images load and detections
are prepared; routine page-loading messages do not replace export progress.
The browser shows completed/total images and elapsed time. `GET /api/export_status`
returns progress; `POST /api/export` accepts `{"write_png": true, "workers": 4}`.
Only one export per browser session can run at a time. Each completed export
adds `export_timings.json` with wall time, detection/mask time and per-candidate
CPU timings. CPU stage totals sum concurrent work and can exceed wall time.
Server logs print `[export]` progress and timings. Existing Python servers must
be restarted and the browser refreshed to load these changes.

SNR work also prints flushed `[snr]` lines for both export and the interactive
SNR view and appends the same lines to `<session_dir>/snr.log` immediately.
`/api/state` and `browser_manifest.json` expose `snr_log_path`.
Each line identifies the candidate, worker thread, source count,
method, stage and cache hit/miss. Calculation emits start/done/error for the
shared `background` (65×65 median filtering), `ap2`, `kron`, and `total` stages;
rendering is explicitly labeled `stage=plot`. Cached results and empty
detections print `status=cached` / `status=skipped`, without pretending to
recalculate. `export_timings.json` includes `snr_background`, `snr_ap2`, and
`snr_kron` as components of `snr_measurement`. These sums represent concurrent
worker elapsed times, not overall export wall time.

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
  or SNR/Mask filtering. Full floating-point mask arrays are not exported.

**View → Mask Filter** uses the SNR Filter-style settings dialog: Pred IoU
defaults to **0.80**, Stability to **0.90**; **Apply** enables filtering and
**Off** restores all detections. Both thresholds are inclusive and must pass.
Shape, Center and Show Masks use the same retained detection subset; SNR Filter
can be enabled at the same time. Show Confidence remains the complete pixel map.
Pred IoU is the selected candidate's raw decoder score, without clipping or a
sigmoid. Stability is the selected mask's area above `mask_threshold + 1`
divided by its area above `mask_threshold - 1`, measured on the output image
grid and valid pixels. Empty masks are excluded. Scores are recorded at Detect
time, so changing thresholds does not rerun inference. Mask Filter starts off
and requires mask decoding; it does not change saved/exported mask selection.

Mask decoding adds work to Detect, especially in crowded fields. Use
`--no-make-masks` for an explicitly detection-only session. Reload the server and
refresh the browser after updating these files.

After **Detect**, **View → Show Confidence** overlays the per-pixel predicted
class before the detection threshold and NMS: 1 blue, 2 green, 3 yellow, 4 red;
class 0 is transparent. This is the class argmax, not a rounded ordinal score.
For a heatmap checkpoint, the same button displays continuous sigmoid
probabilities in [0,1], with a probability legend instead of ordinal classes.
Colors vary continuously from blue to green/yellow to red. Zero and invalid
pixels are transparent; weak responses below 0.1 are not discarded. Existing
single-channel confidence NPZ files are recognized from their saved logit shape.
Scaling, custom limits, inversion and smoothing remain adjustable without model
inference. Confidence maps are saved even with `--no-make-masks` or zero detections.
Each detection writes `session_dir/confidence/<candidate>_confidence.npz` with
float32 `logits` (C,H,W), `labels`, `score`, `ordinal_expectation`, and `valid`.
Heatmap NPZ files additionally identify their proposal/display mode; `score`
is the actual sigmoid probability, while `labels` retain old diagnostic bins.
Arrays use lower-origin image coordinates. A bounded RAM cache reloads labels
or continuous heatmap scores
from these files after eviction; it does not rerun the encoder.
**Export Selected** always copies this NPZ and additionally writes
`<candidate>_confidence_overlay.png` using the display scaling selected when
export starts (default ZScale for API exports without view settings), regardless
of whether Show Confidence is enabled. Paths and display settings are included
in the manifest, including `confidence_display_kind` and the value range.
The same confidence overlay filename contains a continuous probability overlay
for heatmap checkpoints. Existing saved mask PNGs continue to use normal ZScale.

Regression checks:

```bash
/home/czh23/miniconda3/envs/cellect/bin/python -m unittest \
  eval.tests.test_jwst_fields eval.tests.test_browser_fast_masks -v
node eval/tests/test_dataset_menu.js
```


### Mask inference alignment (2026-10-01)

The browser now reads `mask_multimask` from the checkpoint run configuration:
trained multi-mask tokens 1–3 are used for the current experiments. Center and
rotated-Kron bounding-box prompts use the training helper, with `--mask-box-scale 1`
(default); remove any old explicit `--mask-box-scale 2` argument. Restart the
server to clear model/mask caches after this change. Bounding boxes shown on the
mask overlay are tight *output* bounds, not the input prompt boxes.

GT masks are unavailable during inference. `--mask-selection pred_iou` (default)
selects the highest raw predicted IoU score among the trained candidates. Scores
are not clamped to [0,1] and no absolute IoU threshold is imposed. The alternative
`--mask-selection geometry` reuses centroid/outside/area terms; this can prefer
small centered masks and is retained for comparisons. Neither mode reproduces
GT-dependent training selection. Empty masks are not replaced by the largest
candidate. Training still selects candidates by its configured loss. JWST invalid coverage
is excluded. New session/export manifests record numeric scaling and detection
settings as well as the mask policy, for repeatable comparisons.

### Angular SNR apertures and ordinary-view workers (2026-10-22)

Previously `AP2` meant a **5-pixel radius for every dataset**, not a 2-arcsec
aperture. Browser and export now use an angular **diameter** of 2 arcsec for
HSC, 6 arcsec for ZTF, and 0.3 arcsec for JWST (legacy NIRCam, COSMOS and Abell). Radius is
`diameter / (2 * pixel_scale_arcsec)`. HSC uses its 0.168-arcsec tract grid; ZTF
uses its adapter's 1.013-arcsec grid; JWST uses the actual WCS of the common
detection/reference-band grid, including aligned bands with other native scales.
The sky annulus remains at 2–3 aperture radii; its minimum pixel count scales
with area (minimum 4), so ZTF's small annulus does not incorrectly fail the old
100-pixel requirement. Messier/standalone utility pixel defaults are unchanged.

`--snr-workers 4` (default) enables normal-page SNR concurrency and limits active
SNR work shared with export/image requests. Equal concurrent requests share one
measurement. Workers operate across images, not individual sources within an
image; GPU model inference remains serialized. `--snr-aperture-diameter-arcsec`
optionally overrides the angular diameter for the active dataset. Restart the
Python server with its existing checkpoint/dataset/session arguments and refresh
the page to load these changes; changing the export-worker field alone does not
change ordinary-view SNR concurrency. The UI and export titles show the diameter;
the internal method key and `_ap2_snr_overlay.png` suffix remain compatible.

Per-source neighbor masks are now rasterized only in the aperture/annulus
window. Regression checks preserve the previous full-image-mask flux and
background values. The 65×65 median-filter background remains expensive.

Noise definitions: fixed aperture uses the sigma-clipped standard deviation of
source-free aperture sums, scaled by `sqrt(actual_pixels / kernel_area)`; Kron
uses one robust sky **pixel** sigma per image times `sqrt(Kron_pixels)`. Neither
uses a spatially varying per-pixel variance/sigma map, weight/ERR plane or added
source-Poisson model. Bad/untrusted measurements remain shown (class 7); only
trusted finite SNR below the chosen threshold is hidden. Cache hits require no
remeasurement. Logs record diameter, pixel radius, scale, annulus and stage costs.

Focused checks:

```bash
python -m unittest eval.tests.test_browser_angular_snr eval.tests.test_browser_parallel_export -q
```

ZTF follow-up (2026-10-22): the initial 2-arcsec trial is superseded by a
**6-arcsec diameter** default (radius 2.9615 pixels at 1.013 arcsec/pixel),
with sky annulus radii 5.923–8.885 pixels (6–9 arcsec). The 2-arcsec trial
could measure just 1–4 center pixels and estimate sky inside a bright source;
a negative source flux is not proof that a visually bright detection is false.
A verified P00005/zi elongated source changed from SNR 1.316 to 17.851;
the bright P00000/zg source at local (389,325) changed from -1306.733 to
8569.946. These are background-limited SNR values, without source Poisson noise.
The HSC and JWST defaults are unchanged. Remove any explicit
`--snr-aperture-diameter-arcsec 2` when restarting to use the new ZTF default.
# HSC 独立 coadd 与原始网格（2026-11-03）

在原有 server 命令中增加 `--dataset hsc_image --hsc-sources coadd --hsc-grid-mode fits`，
即可脱离训练 Zarr 的保留范围，从原始 FITS 选块；
2026-11-04 起默认使用 512 像素大小、368 步长及边缘对齐，具体清单与坐标见下节。
`--hsc-coadd-profile half` 读取 `/data/czh23/Subaru_products/half_coadd`，
`--hsc-coadd-profile official` 读取 `/data/shared/Subaru` 的 full coadd。
这里只做 coadd 推理，不加入 noisy/denoised，也不重建训练标签。
继续看 Zarr 保存的小块时用 `--hsc-grid-mode zarr`；现在识别最新 `half_coadd` 目录，
`--hsc-image-root` 可以指向 mixed Zarr 根目录或其 `hsc` 子目录。
更改这些参数后需重启 server。

## HSC 默认现场切片与独立清单（2026-11-04）

HSC image 模式现在默认 `--hsc-grid-mode fits`，不读取任何训练 Zarr。
`direct_zarr_v2_0711/9813/coadd/<patch>.zarr` 的多波段 patch 格式与当前
image-level 单波段扫描格式不同；不要再将它作为默认浏览索引。
FITS 模式下 `--hsc-image-root` 不参与文件发现或切片。

每个 patch 使用 512×512、stride 368 的预处理网格，最后一行/列对齐边缘。
标准 4200×4200 patch 有 11×11 块；例如 patch 4,5 的
`x010_y007` 即 r07/c10，tract 坐标 `(19588,22476)`，FITS 数组坐标 `(3688,2576)`。
小于 512 的图像用 NaN 补齐。FITS 模式不会应用训练时的删块规则，
因此可浏览训练中被剔除或未写入 Zarr 的网格。

清单只包含 FITS 路径、SCI HDU、尺寸、LTV 原点、文件大小/mtime 和网格坐标；
不包含科学像素或标签。默认在 `~/.cache/cellect/hsc_fits_plans/` 按配置保存，
可用 `--hsc-cutout-plan` 指定路径。重新打开时检查文件大小和 mtime，
新文件或修改过的文件会刷新表头；不同 coadd profile/stride 的默认缓存互相独立。
查看/推理时按清单现场读取 bounded FITS section，应用当前 `--scaling-mode`
及 clip/log/Lupton 参数；统计量使用该 512×512 切片，与原 HSC browser 一致。
这不会按旧 Zarr 已缩放像素重复缩放，也不会写出新的 Zarr。

仅生成清单，不启动服务或加载 checkpoint：

```bash
cd /home/czh23/CELLECT/eval
python hsctiles/serve_hsctile_pack_browser.py --hsc-coadd-profile official \
  --hsc-cutout-plan /home/czh23/analysis/2026-11/2026-11-04/hsc_browser_fits_plan/official.json \
  --prepare-hsc-plan-only
```

在原启动命令中增加 `--dataset hsc_image --hsc-coadd-profile official` 使用 full coadd；
使用 half coadd 则指定 `--hsc-coadd-profile half`（默认）。
`--hsc-sources coadd` 只看 coadd，省略则同时发现 noisy/denoised。
需要旧的 image-level Zarr 保留块索引时显式使用 `--hsc-grid-mode zarr`；
这不支持 v2 的多波段 patch 存储格式。

