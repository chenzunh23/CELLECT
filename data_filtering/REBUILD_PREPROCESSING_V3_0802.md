# Preprocessing V3 Rebuild Plan

This document defines the clean replacement path for the current mixed PU /
bright-label preprocessing code.  The goal is to remove historical branches and
make source labels a single pass with explicit names.

## Why Rebuild

The current pipeline classifies sources in `_classify_pu_catalog()`, then applies
remeasurement, fill-ratio downgrades, external bright labels, integrated bright
labels, SNR downgrades, and dense-target generation in different modules.  That
ordering makes bright-source behavior hard to reason about.  In particular,
bright deblend fragments can be promoted by late remeasurement/fill logic before
the bright-source branch sees them.

V3 should not patch that flow.  It should build one source-label table first,
then write targets/zarr from that immutable table.

## Output Label Vocabulary

Use these labels only:

```text
clean
weak_shape
strict_center_only
restricted_bright_region
ordinary_ignore
strict_ignore
background
dropped
```

Meaning:

```text
clean
  trains confidence center + shape/mask.

weak_shape
  trains confidence center and weak shape/mask.  This is used when the center is
  trustworthy but the aperture or mask is lower quality than `clean`.

strict_center_only
  trains confidence center only; no shape/mask.  This includes Gaia centers,
  synthetic component centers, and catalog sources whose shape is unreliable
  but whose center is still useful.

restricted_bright_region
  bright HSC fragments that should not train a positive center.  Region can be
  used as a non-source/bright penalty region with configurable weight.

ordinary_ignore
  low-quality catalog source or uncertain non-bright source.  No foreground
  supervision and not background.

strict_ignore
  SAT/BAD/EDGE/NO_DATA or similar unusable pixels.  No loss by default.

background
  LSST-detection background, only after removing clean/center/bright/ignore
  priority regions.

dropped
  removed from masks entirely, for pathological ellipses such as area > 10000.
```

Dense class ids should be explicit and stable:

```text
0 unlabeled
1 clean
2 weak_shape
3 ordinary_ignore
4 background
5 strict_center_only
6 restricted_bright_region
7 strict_ignore
```

## Required Order

### 1. Load Patch/Band Inputs

Inputs:

```text
calexp IMAGE/MASK/VARIANCE
meas catalog
kron refit CSV
det/background product
optional Gaia catalog
optional noisy/denoised image + variance/weight metadata
```

Attach refit aperture once.  The canonical shape for labels is the smaller of
the refit Kron aperture and official Kron aperture when that policy is enabled.

### 2. Compute Image Regions

Compute these once per image:

```text
bright_region
strict_ignore_region
lsst_background_region
```

Bright modes:

```text
log-lupton
  log uses full-image minimum, band log_a broad=1000, NB1010=100, NB0387=3000.
  lupton uses current zscore median as minimum.
  bright = standardized_log >= 3 AND standardized_lupton >= 3.

anscombe
  bright = standardized_anscombe >= 3.

zscore-no-upper
  no threshold bright image region.  Use source clusters plus Gaia only.
```

Strict ignore region starts from FITS mask planes, at minimum:

```text
SAT BAD EDGE NO_DATA UNMASKEDNAN
```

`strict_ignore_region` has higher dense priority than background and lower
priority than trusted source centers.

### 3. Split Catalog Into Bright And Ordinary

After refit and A filter:

```text
A drop:
  area > 10000 -> dropped
  area > 900 and mag > 28 -> ordinary_ignore
```

B basics apply to all non-dropped sources before the branch:

```text
mag <= 30
axis_ratio <= 5
close-center pair within 0.5 arcsec: remove dimmer
```

Then split:

```text
bright source: mag < 22
ordinary source: mag >= 22
```

Bright sources must not run the ordinary AP2 remeasurement, ordinary fill-ratio
downgrade, ordinary containment, ordinary B flags, or ordinary SNR flow.

### 4. Ordinary Source Branch

Ordinary sources follow DATA_FILTERING_STANDARD:

```text
mag in normal range or band-limit range
AP2-Kron absdiff < 1
valid refit match
B flags pass
containment pass
fill-ratio strict_center_only downgrade
coadd AP2-SNR downgrade
variant variance/weight SNR downgrade for noisy/denoised
SAT/BAD/EDGE center -> strict_ignore or ordinary_ignore according to mask policy
```

Outputs:

```text
clean
weak_shape
ordinary_ignore
dropped
```

Original ordinary-ignore and dropped rows cannot be rescued by noisy/denoised
SNR.

#### Aperture Fill Strict-Center Rule

Before ordinary AP2/SNR processing, non-bright sources with very low refit
aperture support become strict center-only labels:

```text
area > 500 and aperture_pixel_count / aperture_area < 0.3
  -> strict_center_only
```

This is the rule from the 2026-07-23 refit-diagnostics discussion.  The
historical implementation stored the rows as `center_only` with
`pu_no_shape_supervision=True`; v3 makes that intent explicit by mapping them to
`strict_center_only`.  They train center/confidence only and must not provide
shape or mask supervision.  Bright sources skip this ordinary fill-ratio branch.

#### Ordinary SNR Standards

The historical ordinary SNR diagnostic in
`output/data_filter_0723/snr_post_filter_45` is the most complete reference for
this branch.  It was requested with these rules:

```text
area > 500 and AP2 SNR <= 8 -> center_only
broad bands:  AP2 SNR <= 3 ignore, 3 < AP2 SNR < 5 center_only
narrow bands: AP2 SNR <= 5 ignore, 5 < AP2 SNR < 8 center_only
```

For v3, historical `center_only` in this ordinary SNR post-filter maps to
`weak_shape`, not `strict_center_only`.  The old script
`scripts/apply_snr_post_filter_regions.py` wrote only `clean`, `center_only`,
and `ignore`; it also normalized any input `strict_center_only` to
`center_only` before applying SNR.  Therefore this ordinary SNR branch must not
create special strict-center sources by itself.

The 2026-07-23 saved summary counts were:

```text
HSC-I 4,5 broad:   before clean=24850 center_only=351 ignore=745
                   after  clean=23069 center_only=1594 ignore=1283
HSC-Z 4,5 broad:   before clean=23703 center_only=441 ignore=719
                   after  clean=20314 center_only=2844 ignore=1705
NB0816 4,5 narrow: before clean=16285 center_only=728 ignore=917
                   after  clean=9496 center_only=3171 ignore=5263
NB1010 4,5 narrow: before clean=6035 center_only=387 ignore=630
                   after  clean=3206 center_only=1148 ignore=2698
```

Non-coadd visibility SNR is a separate downgrade applied only to already usable
coadd labels:

```text
variance/weight SNR <= 3 or invalid -> ordinary_ignore
3 < variance/weight SNR <= 5        -> weak_shape
variance/weight SNR > 5             -> clean
```

The older direct image-annulus SNR default was looser (`2/3`), but the current
variance/weight path should use the `3/5` visibility split unless an experiment
explicitly requests otherwise.

### 5. Bright Source Branch

Bright sources use the external bright flow directly.
This AP2 gate is implemented in `preprocessing/bright_ap2.py` and must run
after A/B basics but before the bright Gaia/component labeler.

For image-threshold modes (`log-lupton`, `anscombe`):

```text
if source center outside bright_region:
  use only AP2 threshold for outside bright region:
    abs(AP2-Kron) >= 1 or invalid -> ordinary_ignore
    otherwise allow clean/weak-shape decision below

if source center inside small bright component (area < 1000):
  abs(AP2-Kron) >= 2 -> ordinary_ignore

if source center inside large bright component:
  skip AP2.
```

Cluster bright sources inside each bright component:

```text
two sources are connected if centers are within 50 px and Kron IoU >= 1/3
for log-lupton/anscombe bright components, isolated means:
  the bright component contains exactly one source cluster
  and that cluster contains exactly one source
```

Rules:

```text
single-cluster component with component area < 1000 and usable shape -> clean
single-cluster component with component area >= 1000 and usable shape -> weak_shape

for each Gaia source matched to a cluster:
  add one strict_center_only at the Gaia center
  all HSC sources in that matched cluster -> restricted_bright_region

for each bright Gaia source inside a bright component but not consumed by any
matched cluster:
  add one strict_center_only at the Gaia center

unmatched clusters in a multi-cluster bright component -> restricted_bright_region

large bright component with no usable center:
  add one strict_center_only at geometric component center
```

For `zscore-no-upper`, there is no image bright component.  Build clusters from
bright source apertures only:

```text
isolated small source -> clean
isolated larger source -> weak_shape
Gaia-matched cluster -> Gaia strict_center_only + HSC fragments restricted
unmatched non-isolated cluster -> ordinary_ignore
```

### 6. Dense Target Priority

Dense masks are derived from the final source-label table and image-region
tables only.  No source relabeling is allowed during target writing.

Priority:

```text
clean
weak_shape / strict_center_only
restricted_bright_region
strict_ignore
ordinary_ignore
background
unlabeled
```

Default loss behavior:

```text
clean: confidence + shape/mask
weak_shape: confidence + weak shape/mask
strict_center_only: confidence only
restricted_bright_region: no positive center; configurable non-source weight
background: configurable non-source weight
ordinary_ignore: weight 0
strict_ignore: weight 0
```

## New Module Boundary

Implement V3 in a separate package first:

```text
preprocessing/
  labels.py          canonical label enum and schema
  io.py              FITS/catalog/refit/background/Gaia readers
  image_regions.py   scaling, bright, strict-ignore, background masks
  source_shapes.py   refit/official Kron aperture attachment
  ordinary.py        ordinary branch
  bright.py          bright branch
  snr.py             coadd and noncoadd SNR downgrades
  targets.py         dense target writer from final labels only
  zarr_writer.py     zarr packaging
  cli.py             single explicit command-line entry point
```

Old scripts should eventually call `preprocessing.cli` rather than importing
individual old helpers.  Until parity is verified, keep old outputs readable but
do not add new behavior there.

## Validation Plan

Start with small deterministic checks:

```text
HSC-I 9813/4,5 sample 59 and 114
HSC-Y 9813/4,5 sample 59 and 114
HSC-I/HSC-Y 9813/6,1 bright examples
NB1010/NB0816 9813/4,5 bright examples
NB0387 9813/7,6 sparse examples
```

For each check write:

```text
source_labels.csv with source_id, final_label, reason, component_id, cluster_id
label overlay PNG
source class REG
dense target panel
```

Zarr writing must use the training vocabulary only:

```text
source_centers/source_ids:
  clean and weak_shape positive centers only
strict_center_only_centers/ids:
  strict_center_only centers only, including Gaia and geometric synthetic centers
shape_source_*:
  clean and weak_shape shape supervision only
restricted_bright_region / ordinary_ignore / strict_ignore / dropped:
  never written as positive source centers
diagnostic_source_rows.json:
  optional, keeps detailed reasons such as Gaia-source strict center vs added
  geometric center for debugging
```

The first acceptance criterion is that no HSC deblend fragment inside a
Gaia-matched bright cluster appears as `clean` or `weak_shape`; only the Gaia
center is positive.
