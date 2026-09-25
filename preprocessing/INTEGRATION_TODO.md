# HSC / COSMOS / A2744 integration

Implement in this order; keep dataset-specific catalog decisions separate from
image preparation, astrometry, targets and storage.

- [x] Shared A filter: convert pixel area to HSC-equivalent area (0.168 arcsec/pixel);
  drop both oversized and faint-oversized sources; test boundaries and scales.
- [x] Shared Gaia projection: propagate finite proper motions from each row's
  ref_epoch to FITS observation time; retain original positions when unavailable;
  preserve integer IDs and explicit pixel origin; test masked values and WCS.
- [x] Integrate angular Gaia matching into HSC bright labeling, retaining current
  HSC distances and component membership; test outside-component rejection.
- [ ] Catalog adapters for COSMOS and A2744: standard centers, Kron axes/radians,
  magnitudes, errors, flags, pixel scale and observation epoch.
- [x] Dataset-specific ordinary rules: preserve the approved B/containment,
  photometry and SNR candidate transitions; apply SNR before final center labels.
- [x] Shared image preparation: FITS-unit/surface-brightness conversion, source-aware
  NaN fill, scaling and local bright masks; keep raw validity separately.
- [ ] Shared bright policy: dataset-configured cluster rules, Gaia first for empty
  components, geometry fallback; direct 1000-pixel bright threshold; boundary policy.
- [ ] Shared targets: clean > weak > bright > background > ignore; subtract all
  source/bright regions from background; truncate only clean/weak shapes; Euclidean
  PSF confidence with center level 4 and FWHM clipped to [2, 8].
- [ ] Consolidate orchestration behind the existing Zarr entry; preserve schema,
  record dataset/rules/units/epoch metadata and validate on HSC and both JWST fields.

Verification command for the first shared modules:
`python -m unittest preprocessing.tests.test_shared_a_gaia -v`

## Image preparation

Pure adapters live in `preprocessing/utils/` (`image.py`, `area_filter.py`,
`gaia.py`). Stage orchestration stays in the existing preprocessing modules.

`prepare_image` converts intensities to ZP=27 flux per 0.168-arcsec HSC pixel,
without resampling the source grid. MJy/sr requires no native pixel-area factor;
Jy/pixel requires PIXAR_A2, PIXAR_SR, celestial WCS or an explicit pixel scale.
Unknown units fail explicitly; calibrated HSC can use `input_unit="hsc-zp27"`.

```python
from preprocessing.image_processing import (
    ImagePreparationConfig, ImageProcessingConfig, BrightRegionConfig,
    prepare_image, scale_image_for_training, build_bright_components,
)

prepared = prepare_image(raw, header=header, config=ImagePreparationConfig())
scaled = scale_image_for_training(prepared, config=ImageProcessingConfig(
    clip_threshold=5, statistics_clip_sigma=5, log_a=1000,
    anscombe_scale=1000,
))
bright, components = build_bright_components(prepared, config=BrightRegionConfig(
    threshold=5, clip_threshold=5, statistics_clip_sigma=5,
))
```

Use `scaling_mode="anscombe-rgb"` and bright `mode="anscombe"` for Anscombe.
Default log/Lupton bright regions are the intersection followed by dilation.
The hybrid fill uses an 8-pixel source test ring, nearest finite values for
source holes, and normalized Gaussian smoothing (sigma=32) for background holes.
Filled cores participate in bright detection; `prepared.finite_mask` retains
original validity and must not be interpreted as observed flux after filling.
Statistics are local to the supplied image, not implicitly global or per tile.

Legacy ndarray calls do not convert units or fill NaNs. The explicit diagnostic
configuration above uses statistics sigma=5; the legacy default remains 3.
Rawclip-no-upper modes retain their existing clip_threshold-based statistics.
`PipelineConfig.image_preparation` opts into preparation once per image. Dataset
CLI adapters, validity storage and full JWST Zarr integration remain pending.

Verification: `python -m unittest preprocessing.tests.test_image_preparation
preprocessing.tests.test_shared_a_gaia -v` (run on one line).
Real 512-pixel F444W cutouts from A2744 and COSMOS Pointing 19 were compared with
the September 6 reference: identical repaired images and scaling errors <1e-6.

## Ordinary source rules

- `meas_processing.classify_catalog_basics`: normalized geometry/magnitude input,
  shared HSC-equivalent A thresholds (10000 pixels; mag>28 and area>900).
  COSMOS/A2744 do not inherit HSC's axis-ratio/magnitude/close-pair defaults.
  A2744 close pairs must follow its PSF size test, so run in its ordinary module.
- `ordinary_common`: `OrdinaryInput`, common result, monotonic label degradation,
  stage snapshots and dataset dispatch. Coordinates and axes are image pixels;
  angles are image-frame radians, already WCS-adapted. Pixel origins must agree
  with supplied image/masks. No implicit sky-angle or arcsecond conversion.
- `ordinary_hsc`: unchanged HSC rules and result fields. `ordinary.py` preserves
  old imports used by the HSC pipeline and image-level Zarr builder.
- `ordinary_cosmos`: warn categories 1/4/5/6 (NOT bits), star-mask center candidate,
  SE++/Kron photometry (missing or diff>2 ignore; 1<=diff<=2 strict center),
  boundary containment >=0.995, then SNR<=3 ignore, 3<SNR<5 strict center.
  Star mask is the COSMOS `flag_star`, not an implicit union with HSC star masks.
  Containment excludes already-known SNR failures, matching the reference's
  earlier SNR step and avoiding invalid neighbors demoting surviving sources.
- `ordinary_a2744`: both semi-axes >= band PSF FWHM; close pairs keep the fainter
  source; high IoU>=0.8 keeps the brighter. Iterative containment uses the
  reference 72-boundary-points plus center fraction >=0.8, with flag priorities
  0 > 2 > 1/3 > other. Flags 16-19 produce strict-center candidates, not final
  SNR exemptions. Every update only reduces supervision; ignored sources no
  longer participate. Optional star footprint: center hit or sampled overlap
  >=0.35 ignores mag>=22, while brighter sources remain for the bright pipeline.
- `snr.py`: shared candidate cuts and optional A2744 remeasurement entry point;
  `utils/source_snr.py`: reusable radius=16 pixel aperture, 128-pixel blocks,
  clipped background and blank-aperture noise. An external LSST `sky_mask=True`
  identifies sky pixels; `excluded_mask=True` excludes bright/source regions.
  Without external sky, the reference high-threshold heuristic is used and
  explicitly named in diagnostics, not mislabeled as an LSST product.

Remeasurement uses linear images, not scaled RGB. The latest A2744 diagnostic
remeasures mag>22 (configurable), including strict-center candidates. For those
strict candidates, nonfinite or SNR<3 is ignore even when untrusted; clean/weak
sources retain the conservative trusted-only rule. Pass an image or precomputed
measurements; otherwise `aperture_snr_pending` records the still-pending rows.
Keep original invalid pixels invalid in SNR images, even when bright detection
uses repaired pixels. Bright/Gaia insertion is not part of ordinary filtering.

```python
from preprocessing.meas_processing import classify_catalog_basics
from preprocessing.ordinary_common import OrdinaryInput, classify_ordinary_sources
from preprocessing.ordinary_a2744 import A2744OrdinaryConfig

basics = classify_catalog_basics(geom, mag, dataset="a2744", pixel_scale_arcsec=scale)
data = OrdinaryInput(geom, mag, snr=catalog_snr, flags=sextractor_flags)
result = classify_ordinary_sources(
    data, basics.after_b_basic, basics.labels, dataset="a2744",
    config=A2744OrdinaryConfig(psf_fwhm_pixels=fwhm_arcsec / scale,
                               image_shape=raw.shape),
    image=raw, sky_mask=lsst_sky, excluded_mask=bright_mask,
    star_footprint=star_footprint,
)
```

For COSMOS supply `model_mag` (SE++), `star_mask` and enumerated `flags` in
`OrdinaryInput`, then dispatch `dataset="cosmos"`. Full FITS catalog adapters and
JWST Zarr CLI wiring are still separate integration tasks.

### Ordinary verification (2026-09-08)

`python -m unittest preprocessing.tests.test_ordinary_datasets
preprocessing.tests.test_image_preparation preprocessing.tests.test_shared_a_gaia -v`
(one line): 25 tests passed, including legacy HSC fill behavior, enum flags,
photometry boundaries, non-resurrection, close/IoU opposite decisions,
containment priorities, star mask brightness protection and strict-candidate SNR.

`python -m preprocessing.tests.compare_ordinary_references` is a read-only
real-catalog control against the requested September 5/8 directories:

| Reference selection | Sources | Ordinary differences | Live-pair control differences |
| --- | ---: | ---: | ---: |
| COSMOS left top F444W | 1579 | 0 | 0 |
| COSMOS center F444W | 1812 | 0 | 0 |
| COSMOS top column 3 F444W | 1421 | 0 | 0 |
| A2744 F115W-center F070W | 13955 | 9 | 0 |
| A2744 F115W-center F444W | 13960 | 1 | 0 |
| A2744 F250M-center F070W | 13496 | 16 | 0 |
| A2744 F250M-center F444W | 13500 | 3 | 0 |

A2744's old inner loop checked only whether the second source remained active;
an ignored first source continued affecting later neighbors. Changing only this
guard in the reference eliminates all differences. PSF/close/flag stages match
exactly. Shared A changes no selected COSMOS sources; each F250M-center band has
one additional A drop, consistent with the preceding shared-A comparison.
Noise background, aperture sigma, sky counts and sky mask match the reference
exactly on a finite-image fixture. This validates ordinary/SNR components, not
a newly generated full bright/Gaia/REG/Zarr run.

## JWST bright postprocessing (2026-09-09)

`bright_label_jwst.py` implements COSMOS/A2744 separately from HSC's parallel
bright branch. It consumes `OrdinaryInput` and completed `SourceLabels`, and
returns the existing `BrightLabelResult`. It rejects unassigned labels; callers
must finish optional aperture SNR before invoking this stage. No HSC dispatch or
dataset CLI is changed here.

```python
from preprocessing.bright_label_jwst import JWSTBrightConfig, label_bright_sources

bright = label_bright_sources(
    data, ordinary.labels, image_shape=raw.shape,
    config=JWSTBrightConfig(mode="a2744", pixel_scale_arcsec=scale,
                           gaia_reference_epoch=2016.0),  # only for known DR3
    bright_region=bright_mask, component_labels=components,
    gaia_table=gaia, image_header=cutout_header, source_ids=catalog_ids,
)
```

All coordinates are zero-based cutout pixels. The header must describe that
same cutout. Shared proper-motion propagation uses the observation epoch and
catalog ref_epoch; the optional explicit reference epoch fills a missing column
without guessing DR2 versus DR3. Preprojected `gaia_rows` are also supported;
they bypass propagation and may carry `astrometric_evidence` for COSMOS.

COSMOS uses Gaia G<22, a 1-arcsec catalog neighborhood and band PSF FWHM. Stellar
evidence is parallax SNR>=3, combined proper-motion SNR>=3, or motion>=3 mas/year.
With evidence, exactly one PSF-close catalog center is accepted unchanged;
otherwise Gaia is inserted. Without evidence, insert only when no catalog center
matches. As in the reference, matching includes ignored/dropped catalog centers:
accepting such a match does NOT resurrect its ordinary label. Configure
`mode="cosmos", psf_fwhm_arcsec=...`; bright masks are not used, and no geometric
centers or component fallback masks are generated in this mode.

A2744 uses Gaia G<=22 and remaining catalog sources mag<=22. A source without a
bright component, or the sole surviving bright source matched to a component,
uses flag 0 -> clean, 2 -> weak, others -> ignore, capped by its existing label.
Source-component association retains the reference 5-pixel search; Gaia centers
must be inside the actual component. Multi-source components use HSC-scaled
cluster area/distance and angular Gaia matching. Original cluster sources become
ignore; matched and otherwise unused component Gaia supply strict centers.
Empty components of area>=1000 get Gaia first, then geometric-center fallback.
The area cutoff is native pixels, not HSC-scaled. Boundary rejection is optional
and defaults off to preserve the original A2744 diagnostic behavior.

`restricted_fallback_component_ids` contains retained >=1000-pixel bright
components, including those already occupied by sources. Smaller components are
in `ordinary_ignore_component_ids`, even if they have an independently inserted
Gaia center. Synthetic rows use the existing constructors and int64 source IDs;
Gaia is deduplicated by ID. Existing strict centers stay in the catalog labels;
only added centers appear in the strict_center_* arrays, preventing duplicates.

Verification:
- `test_bright_label_jwst`: eight tests for mode separation, label preservation,
  isolated flags, threshold equality, small-component masks, Gaia membership,
  proper motion, IDs and stage ordering. Together with previous suites: 33 pass.
- `compare_ordinary_references.compare_bright()`: same post-ordinary inputs for
  four A2744 combinations (13955, 13960, 13496, 13500 sources), zero bright label
  differences; inserted centers exactly match (2, 2, 2, 1 respectively).
- `compare_ordinary_references.compare_cosmos_bright()`: Pointing 19 same-policy
  control, 42337 catalog centers, 52 Gaia insertions and 42 accepted matches;
  insertion IDs and ordinary labels unchanged.

These are component-level controls; complete new REG/Zarr products were not
regenerated. A2744 ordinary containment fixes remain independent of this control.

## A2744 Frozen Exposure Interface (2026-09-10)

A2744 containment now measures the fraction of the inner ellipse's sampled
area covered by the outer ellipse, using HSC's sampling convention. The 0.8
containment and IoU thresholds and flag-priority actions remain unchanged.
The two F444W previews in `a2744_lsst_r10_all_sources` have been overwritten;
SNR still uses LSST sky, radius 10, and all remaining candidates.

`A2744OrdinaryConfig` exposes `enable_containment`, `enable_catalog_snr`,
`enable_aperture_snr`, and `enable_star_mask` (all default true). These switches
are for selective reruns; they do not by themselves freeze a previous result.

For half-coadd training, preserve the final source snapshot instead:

```python
from preprocessing.utils.frozen_sources import prepare_frozen_variant

variant = prepare_frozen_variant(
    snapshot_dir,       # sources.csv, inserted.csv, summary.json
    half_coadd_fits,
    new_background_npz, # sibling LSST summary.json must identify this FITS
    origin=(8064, 19102),
    shape=(4096, 4096),
)
sources = variant['sources']
```

This maps catalog ellipses and inserted centers through WCS, preserves labels
and reasons, prepares the new image, and recomputes bright components. It does
not run A/B/SNR or bright-label filtering, and does not insert further centers.
Previously ignored sources stay ignored. New LSST background generation is
required separately; stale background provenance is rejected. The default
bright configuration uses threshold/clip/statistics sigma 5.

The returned source geometry, labels, background mask and bright components
are inputs for subsequent dense-target/Zarr construction, not a completed Zarr
writer. Bright-component IDs from the old exposure must not be reused on the
new components. Both real F444W half-coadd WCS mappings were validated (13,500
and 13,960 catalog sources, plus one inserted center per region).

## Segmentation Zarr Arrays (2026-09-10)

`utils/segmentation.py` now owns both WCS/coverage utilities and isolated
positive-only targets; `segmentation_targets.py` has been removed.
`utils/image_level.py` owns StoreTask, PatchLabels, input resolution and tile
target helpers. The build module re-exports the previous helper names for its
existing diagnostic callers.

The training writer accepts optional paired `band_segmentation_ids` (int32)
and `band_segmentation_weight` (float32), both `(N,B,H,W)`. IDs are global COSMOS
catalog IDs, zero means unlabelled, and weights are zero outside masks (NOT
negative supervision). Default positive weight is 0.25. These arrays do not
replace dense PU labels. Missing supervision omits both arrays and records
`segmentation_supervision="none"`; supplied arrays record `"positive_only"`.

COSMOS adapters can now call the public `write_classified_patch` entry point:

```python
from preprocessing.build_image_level_zarr import write_classified_patch

write_classified_patch(
    task, linear_hsc_unit_image, patch_labels, origin=(x0, y0),
    cosmos_catalog=master_catalog_path, image_wcs=full_image_wcs,
    provenance={"image_fits": str(image_path)},
)
```

`patch_labels` is a PatchLabels with final classes, geometry, dense targets and
inserted centers already populated. This entry point bypasses HSC catalog
classification. Only when `cosmos_catalog` is supplied does it sample native
segment/star maps, select isolated final clean/weak sources, then crop targets
with exactly the same tile origins as images. Isolation is computed before
tiling and IDs are not renumbered across tiles. HSC/A2744 leave this argument
unset. Already prepared PatchLabels may instead carry segmentation arrays.

The command-line dataset discovery remains HSC-specific; automatic COSMOS
task discovery and training-loss/dataloader consumption are separate work.
This change implements generation, tiling and serialization, not the loss.

## Unified Input Paths (2026-09-23)

`dataset_paths.json` now centralizes HSC paths, the curated 1727/5893 COSMOS
manifest, and separate Abell half-coadd training/full-coadd reference roots.
`build_image_level_zarr --datasets hsc cosmos abell --list-inputs` discovers
all three datasets without running labels or writing Zarr. Existing HSC CLI
execution is preserved. `dataset_inputs.load_image` reads science/validity
cutouts; `role="reference"` explicitly reads the full Abell coadd for catalog
selection. `write_classified_patch(input_source=...)` namespaces JWST stores
and records both paths. The automatic JWST catalog-classification dispatcher
is still a subsequent integration step; JWST cannot fall through HSC filters.
See `DATASET_INPUTS_zh.md` for configuration, loading and writer examples.

2026-09-23 confidence 部分已完成：公共 image-level Zarr 支持 auto/manhattan/psf-matched；第一版 JWST 按用户新要求改为 FWHM clipped to [1.6,8]，替代上方原 [2,8] 约定。HSC auto 保留 Manhattan。详情与现存编排限制见 CONFIDENCE_zh.md。

2026-09-24 confidence 更新：JWST auto 默认切换为 assets 中 OVERSAMP 原始归一化 EE10/35/60/70 定义；旧 FWHM 定义保存在 assets 并以 psf-matched 模式保留。支持配置路径和Zarr来源记录，训练中心监督/解码器尚未修改。详见 CONFIDENCE_zh.md。
