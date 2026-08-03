# CELLECT Data Filtering Standard

This note separates source-quality definitions from product generation.  It
describes how coadd sources are classified before any non-coadd/noisy filtering,
how the recent SNR filters modify those classes, and how the variance-plane
diagnostic estimates noisy visibility.

## Scope

This folder is for data standards and diagnostics only:

- included: catalog source-class definitions, SNR threshold definitions,
  variance-SNR diagnostic logic, REG/CSV debugging outputs;
- excluded: target NPZ generation, PT zscale cache generation, zarr packaging,
  and training/evaluation loops.

The current production code still lives in:

- `data_filtering/pu_source_filter.py`: PU class assignment from a meas catalog;
- `astro_data_preprocessing.py`: patch/tile preprocessing and non-coadd image
  variants;
- `data_filtering/noncoadd_snr.py`: non-coadd image SNR measurement and
  visibility downgrade;
- `data_preprocessing.sh`: shell-level defaults and orchestration;
- `scripts/export_legacy_noisy_variance_snr_regs.py`: recent variance-SNR
  diagnostic.

## Coadd Source Classification Before SNR

The coadd classification starts from the HSC/LSST meas catalog plus the
batch-heavyfp-kron-refit output.  The current default source population is leaf
sources:

```text
source_filter = nchild0
```

The refit radius is attached from:

```text
proxy_nan0_flux_aperture_radius
```

and a source normally must have a good refit match before it can become a
usable clean/center-only/ignore ellipse.

### Base Eligibility

A row first needs:

- finite source center;
- `deblend_nChild == 0` or equivalent `nChild == 0`;
- finite positive Kron/refit ellipse parameters;
- source passes the selected source-filter mode.

Rows outside this base population are not normal supervised labels.

### A Filter

The A filter removes clearly unusable large/faint aperture labels before the B
filter.  Current defaults are:

```text
area > 10000                        -> dropped large ellipse
area > 900 and mag > 28             -> A failed
```

Important distinction:

- `dropped large ellipse` is removed from clean/center/ignore source labels;
- `A failed` but not large-dropped can enter ordinary ignore.

This is why huge pathological ellipses do not necessarily appear as ordinary
ignore sources.

### Aperture Fill Center-Only Rule

Before the B filter, a source can become center-only if the refit aperture has
too little footprint support:

```text
area > 500 and aperture_pixel_count / aperture_area < 0.3
```

This rule came from the 2026-07-23 request recorded in
`/home/czh23/.codex/history.jsonl` around the `data_filter_0723/refit_diagnostics`
work.  The intent was explicit: these sources have a usable center, but their
Kron aperture/shape extends far beyond the supported footprint, so shape should
not be trained.

Historical code stored them as `center_only` plus
`pu_no_shape_supervision=True`:

```text
center_only_by_fill = area > 500 and fill_ratio < 0.3
no_shape_supervision = center_only_by_fill
```

In the v3 label vocabulary, this fill-ratio class maps to
`strict_center_only`, not `weak_shape`.  It trains the center/confidence only and
must have zero shape/mask supervision.  This is separate from the AP2-SNR
post-filter below, whose historical `center_only` maps to `weak_shape`.

### Refined Bright AP2-Kron Hard Reject

Before AP2-Kron remeasurement rescue, bright sources (`mag < 22`) use the same
bright-region-aware AP2 rule as the 2026-07-29 diagnostics:

```text
center outside bright region:
  invalid AP2/Kron or abs(AP2_mag - Kron_mag) >= 1.0 -> ordinary ignore

center inside small bright region (component area < 1000):
  abs(AP2_mag - Kron_mag) >= 2.0 -> ordinary ignore

center inside large bright region (component area >= 1000):
  skip AP2 filtering; the external bright-source flow handles the region
```

These rejected sources become ordinary ignore and are not remeasured.  Non-bright
sources still use the ordinary AP2/Kron B filter.

### B Filter

Among remaining A candidates, B removes sources from clean supervision.  Current
criteria include:

```text
mag outside [18, 30]
or, with band-limit mode, mag outside [m_limit - 5, m_limit)
missing/good refit match failure
abs(AP2_mag - Kron_mag) >= 1
axis_ratio > 5
base_SdssShape_flag
base_SdssCentroid_flag
close-center pair within 0.5 arcsec: remove the dimmer source
large source contains >=80% of a smaller source: remove the larger source
```

B-removed sources become ordinary ignore unless they were separately dropped as
large ellipses.

### AP2-Kron Remeasurement

For old AP2-Kron outliers, preprocessing can remeasure Kron aperture flux using
the refit aperture and footprint/archive information.  This can promote some
old B-ignore sources back to clean or center-only if the remeasured AP2-Kron
difference is acceptable.

Current remeasurement checks include:

```text
new abs(AP2_mag - Kron_mag) < 1.0       -> clean
1.0 <= new absdiff <= 1.5               -> center_only
invalid/new absdiff > 1.5               -> ignore
area > 10000                            -> drop
mag > 28 and area > 900                 -> ignore
axis_ratio > 5                          -> ignore
small footprint fill / containment fail -> center_only or ignore
```

The measurement surface is either heavy footprint or direct image fallback,
depending on whether usable heavy-footprint data are available.

### Strict Bright Center-Only

Bright clean sources can be moved to strict center-only rather than strict
ignore.  The current per-band saturation thresholds are:

```text
HSC-G 18.0, HSC-R 18.2, HSC-I 18.6, HSC-Z 17.7, HSC-Y 17.4
NB0387 14.8, NB0816 16.8, NB0921/NB0924 16.9, NB1010 14.8
```

These sources keep center supervision but should not be treated as ordinary
clean shape labels.

### Coadd Output Classes

The patch-level preprocessed reference catalogs are:

```text
band_reference_catalogs/             clean
band_reference_center_only/          center_only
band_reference_strict_center_only/   bright strict center_only
band_reference_ignore/               ordinary ignore
band_reference_rejected/             rejected/dropped diagnostic set
band_reference_pu_all/               all PU-classified rows with reasons
```

Clean means reliable center + confidence + shape.  Center-only means useful
center/confidence with weak or no shape.  Ordinary ignore means the region
should not be used as background or clean supervision.  Dropped large ellipses
are excluded from source masks by design.

## Coadd AP2-SNR Post Filter

A separate recent diagnostic applies AP2 SNR to already classified coadd rows.
This is not the same as the variance-plane diagnostic.  Its intent is to remove
or weaken low-SNR labels even on coadd.

Broad-band defaults:

```text
AP2 SNR <= 3             -> ignore
3 < AP2 SNR < 5          -> center_only
area > 500 and SNR <= 8  -> center_only
```

Narrow-band defaults:

```text
AP2 SNR <= 5             -> ignore
5 < AP2 SNR < 8          -> center_only
area > 500 and SNR <= 8  -> center_only
```

In the v3 label vocabulary, this historical `center_only` class maps to
`weak_shape`: it keeps center/confidence supervision and may keep weak shape
supervision according to the downstream shape-weight policy.  This AP2-SNR
post-filter does **not** create `strict_center_only` sources.  `strict_center_only`
is reserved for hard center-only labels such as Gaia-inserted bright-star
centers, geometric bright-component centers, or explicitly shape-forbidden
sources.

The 2026-07-23 diagnostic run in
`output/data_filter_0723/snr_post_filter_45` was generated from the request:

```text
area > 500 and SNR <= 8 -> center_only
broad bands:  AP2 SNR <= 3 ignore, 3-5 center_only
narrow bands: AP2 SNR <= 5 ignore, 5-8 center_only
```

Its implementation was `scripts/apply_snr_post_filter_regions.py`.  That script
normalizes any input `strict_center_only` row to `center_only` before applying
SNR:

```text
old_norm = "center_only" if old_class == "strict_center_only" else old_class
```

and writes only three output classes: `clean`, `center_only`, and `ignore`.
The saved summaries confirm the intended broad/narrow split:

```text
HSC-I 4,5 broad:  before clean=24850 center_only=351 ignore=745
                  after  clean=23069 center_only=1594 ignore=1283
HSC-Z 4,5 broad:  before clean=23703 center_only=441 ignore=719
                  after  clean=20314 center_only=2844 ignore=1705
NB0816 4,5 narrow: before clean=16285 center_only=728 ignore=917
                   after  clean=9496 center_only=3171 ignore=5263
NB1010 4,5 narrow: before clean=6035 center_only=387 ignore=630
                   after  clean=3206 center_only=1148 ignore=2698
```

After this pass, a final close-center dedup can demote duplicates to ignore.

## Bright Source And Saturated Region Handling

Bright stars and large galaxies are the main failure mode for the normal
catalog-only PU filter.  They can create many deblend residual rows with large
or overlapping Kron apertures; if all of them are ignored, the detector still
sees a bright plateau and may emit many small false centers.  The current
diagnostic standard therefore separates the source label from the bright
image region.

### Bright Image Region

The shared implementation is `data_filtering/sam_input_scaling.py`.
Preprocessing can request a bright/background mask with:

```text
--pu-enable-bright-background-mask
--pu-bright-mask-mode log-lupton | anscombe | raw | none
```

Definitions:

```text
log-lupton:
  log-scaled image and Lupton/asinh image are each self-standardized;
  bright = log_z > threshold AND lupton_z > threshold.

anscombe:
  Anscombe-transformed image is self-standardized;
  bright = anscombe_z > threshold.

raw / none:
  no bright region is generated.
```

Current default threshold is:

```text
pu_bright_z_threshold = 3.0
```

The bright mask is not a source shape label.  In the PU dense target priority
map it behaves like a supervised non-source region after clean/center-only
regions have been protected.  This is intended to suppress multiple spurious
small detections inside saturated/flat bright plateaus while preserving explicit
catalog centers where they are trusted.

The production dense target priority in `astro_data_preprocessing.py` is:

```text
clean > center_only / strict_center_only > bright > explicit ignore > LSST background
```

Pixels outside clean/center/bright/background are also assigned to ignore when
an LSST background mask is available.  This differs from the external diagnostic
partition below, which is only a visualization/debugging product.

### External Bright-Source Diagnostic

The diagnostic entry point is:

```text
data_filtering/build_external_bright_labels_v2.py
```

It is not yet the default production preprocessing path, but it documents the
intended bright-source policy:

1. Start from bright HSC rows, usually `mag < 22`, after Kron refit attachment.
2. Reject obviously pathological huge apertures first:

```text
area >= 10000 -> ignore
```

3. Existing labels keep priority:

```text
existing clean       -> clean
existing center_only -> center_only / strict_center_only
```

4. Sources outside the bright image component become ordinary ignore.  If their
   centers are also in SAT/BAD/EDGE, the reason records that mask plane.
5. Remaining bright-component sources are clustered by nearby centers and Kron
   aperture overlap:

```text
center distance <= 50 px
Kron IoU >= 1/3
source area < 10000 participates in clustering
```

6. Gaia DR3 is used only as an external bright-star anchor, not as a complete
   galaxy catalog.  Matching considers both individual source centers and the
   cluster centroid:

```text
source/cluster-to-Gaia match radius ~= 1 arcsec
cluster centroid tolerance ~= 10 px
Gaia bright star threshold ~= G <= 18
```

7. If a component is in SAT/BAD/EDGE and contains a bright Gaia star, all HSC
   deblend residuals in that cluster become ignore.  These are treated as
   unreliable saturated-star artifacts.
8. If a bad-mask component has no bright Gaia star and the chosen brightest HSC
   source is a galaxy with usable shape, the brightest source becomes
   `center_only_external`; the rest become `restricted_bright_region`.
9. Outside SAT/BAD/EDGE, Gaia-matched bright stars become
   `strict_center_only_external`.  Gaia-matched or Gaia-unmatched HSC galaxies
   with usable shape become `center_only_external`; other cluster members become
   `restricted_bright_region`.
10. Gaia-unmatched stars or unknown objects become ignore.

### Bright Priority Partition

The diagnostic writes a five-class priority partition:

```text
clean > center_only > bright > background > ignore
```

This diagnostic priority matters for visual review only.  Clean/center-only
labels must not be overwritten by bright or ignore masks.  The bright region can
fill saturated plateaus, but it cannot erase trusted source centers.

Generated diagnostic products include:

```text
*_bright_reclassification_v2.csv
*_component_summary_v2.csv
*_cluster_summary_v2.csv
*_clean.reg
*_center_only.reg
*_strict_center_only_external.reg
*_restricted_bright_region.reg
*_priority_partition.fits
*_priority_partition_overlay.png
*_log_lupton_bright_mask.fits
```

Important limitation: Gaia is mostly a stellar catalog for this use case.  Many
obvious bright galaxies have no Gaia counterpart, so HSC star/galaxy
classification and shape sanity checks remain necessary.

## Non-Coadd Image SNR Filter In Preprocessing

For denoised/noisy image variants, `astro_data_preprocessing.py` reuses the
coadd catalog classes and calls `data_filtering/noncoadd_snr.py` to classify the
visibility of clean sources on the specific image variant.

The current image-level implementation measures SNR directly on the non-coadd
image:

```text
source aperture radius: 6 px
background annulus: 10-15 px
source/center masks excluded from annulus
optional LSST quality-mask exclusion:
  BRIGHT_OBJECT SAT BAD NO_DATA EDGE UNMASKEDNAN
```

Classification:

```text
SNR >= center_only_thresh          -> remains normal/clean for that image
ignore_thresh <= SNR < center_only -> center_only for that image
SNR < ignore_thresh or invalid     -> ignore for that image
insufficient annulus pixels        -> center_only
```

The shell defaults currently are:

```text
NONCOADD_SNR_IGNORE_THRESH=2
NONCOADD_SNR_CENTER_ONLY_THRESH=3
```

These can be overridden when a stricter 3/5 split is desired.  This direct
image-SNR filter only acts on coadd clean sources for variant target generation;
existing coadd ordinary-ignore regions remain ignore and are not rescued.

## Variance-Plane SNR Diagnostic

The newest diagnostic estimates noisy visibility from variance planes instead
of remeasuring source flux on noisy images.  It is useful for comparing old
legacy zarr classes with a variance-scaled expected SNR.

The diagnostic input currently starts from:

```text
preprocessed/<tract>/<patch>/band_reference_catalogs/<band>/
```

That means it starts from coadd clean sources only.  Large bright overlaps,
abnormal shapes, AP2-Kron failures, and other original PU ignore/rejected
sources have already been removed before this diagnostic begins.

### Scale Estimation

Some old noisy FITS images are not on the same raw pixel scale as official
coadd FITS.  The diagnostic first estimates a per-band image scale using AP2
aperture sums on coadd clean sources:

```text
scale = median(noisy_AP2_sum / coadd_AP2_sum)
```

The default uses up to 5000 clean sources.

### Effective Exposure Ratio

For each source, the local variance is averaged inside the same AP2 aperture in
both coadd and noisy images.  After scale correction:

```text
T = var_coadd * scale^2 / var_noisy
```

By default:

```text
T is capped at 1
```

because a noisy realization should not normally be treated as deeper than the
coadd reference.

### Predicted Noisy SNR

The diagnostic uses the catalog coadd AP2 SNR and scales it by effective
exposure:

```text
SNR_noisy_pred = SNR_coadd_AP2 * sqrt(T)
```

Classification:

```text
SNR_noisy_pred <= 3 or invalid -> variance ignore
3 < SNR_noisy_pred <= 5        -> variance center_only
SNR_noisy_pred > 5             -> variance clean
```

This is a diagnostic class, not the original PU class.  A `variance ignore`
source means “coadd clean but expected to be too faint in this noisy image,”
not “bad shape/overlap/original ordinary ignore.”

### Required Ordering

For a full non-coadd preprocessing standard, the order should be:

```text
raw catalog
  -> refit + PU/AP2/shape/overlap/flag filtering
  -> coadd clean / center_only / strict_center_only / ordinary ignore / dropped
  -> image-specific SNR or variance-SNR visibility downgrade
```

Original ordinary-ignore and rejected labels must not be rescued by variance
SNR.  Variance SNR alone cannot detect bad labels, because bad bright overlaps
can have high flux and high formal SNR.

## Warp-Weight Ratio SNR Diagnostic

Some non-coadd FITS products have no usable variance plane.  For those cases,
the diagnostic can estimate visibility from warp weights:

```text
global_T = sum(noisy_group_selected_weights) / sum(coadd_used_warp_weights)
coverage = local_effective_count / number_of_selected_noisy_warps
T_eff = min(global_T * min(coverage, 1), cap_t_max)
SNR_noncoadd_pred = SNR_coadd_AP2 * sqrt(T_eff)
```

Inputs:

```text
coadd used weights:
/data/shared/handoff/2026-06-21_171607_hsc_metadata_warp_n2n_epoch006_full-all-warp-weights/<band>/<patch>/weights.csv

noisy/denoised selected weights and local coverage:
/data/czh23/denoised_fits/patch_<x>_<y>/<group>/<band>/meta.json
/data/czh23/denoised_fits/patch_<x>_<y>/<group>/<band>/effective_count.fits
```

The same class thresholds are used as the variance diagnostic:

```text
SNR <= 3 or invalid -> ignore
3 < SNR <= 5        -> center_only
SNR > 5             -> clean
```

Like variance-SNR, this is a visibility downgrade for already clean coadd
labels.  It should not turn original ordinary-ignore/rejected sources into
clean labels.

Diagnostic entry point:

```bash
conda run -n cellect python scripts/export_weight_ratio_snr_regs.py \
  --patch 4,5 \
  --groups group_00 group_01 \
  --bands HSC-G HSC-R HSC-I HSC-Z HSC-Y \
  --out-dir output/data_filter_0725/weight_ratio_snr_patch45
```

## Current Debug Outputs

The variance diagnostic writes REG and CSV files such as:

```text
*_legacy_noisy_clean.reg
*_legacy_noisy_center_only.reg
*_variance_snr_clean.reg
*_variance_snr_center_only.reg
*_variance_snr_ignore.reg
*_variance_snr_sources.csv
*_legacy_vs_variance_transition.csv
*_variance_snr_summary.csv
```

The important summary fields are:

```text
noisy_to_coadd_scale
t_eff_median / p10 / p90
snr_noisy_median
legacy_* counts
variance_* counts
```

Interpretation:

- many `variance ignore` sources can be normal if `T` is small;
- if `scale` is not estimated, `T` can be wrong by `scale^2`;
- if the diagnostic starts from clean-only input, it does not represent the
  full original ordinary-ignore population.

## Example Diagnostic Command

```bash
python data_filtering/variance_snr_diagnostics.py \
  --zarr-root /data/czh23/legacy_zarr/legacy_zarr \
  --preprocessed-root /data/czh23/preprocessed \
  --noisy-fits-root /home/czh23/fits/noisy \
  --coadd-root /data/shared/Subaru \
  --catalog-root /data/shared/Subaru \
  --tract 9813 \
  --patch 4,5 \
  --variant noisy \
  --group group_00 \
  --bands HSC-G HSC-I \
  --out-dir output/data_filter_0725/legacy_noisy_variance_snr_regs
```

## Narrow-Band Calexp Quality Filter

Narrow-band data can contain missing regions, stripes, large interpolation
artifacts, and low-quality patches that are not well described by source-count
alone.  The current quality score is based only on severe MASK planes:

```text
NO_DATA      weight 1.0
UNMASKEDNAN  weight 1.0
EDGE         weight 0.7
BAD          weight 0.5
INTRP        weight 0.3
```

The score for a pixel is the maximum active weight among these planes; patch or
tile score is the mean pixel score.  Diagnostic-only planes such as `CR`,
`CROSSTALK`, `REJECTED`, and `BRIGHT_OBJECT` can still be plotted separately,
but are not part of the default union score.

Current filtering standard for narrow-band image-level training:

```text
drop whole patch if patch_bad_score >= 13%
drop 512x512 tile if tile_bad_score >= 13%
tile size = 512
stride = 368
edge partial tiles are not used
```

The implementation lives in:

```text
data_filtering/calexp_quality.py
data_filtering/analyze_calexp_mask_quality.py
data_filtering/overlay_calexp_tile_bad_score.py
```

Direct zarr uses the same functions through:

```text
--quality-filter
--quality-bad-score-threshold 0.13
--quality-bad-score-weights PLANE=WEIGHT ...
```
