# Batch HeavyFootprint Kron Refit Handoff

This package is a standalone handoff of the HSC/LSST batch HeavyFootprint Kron
proxy analysis originally developed under:

```text
lsst-kron-radius-sanity-check-for-hsc-9813-meas-sources-20260508-01
```

The maintained entry point is:

```text
batch_heavyfp_kron_refit.py
```

It does not import the LSST/HSC stack and no longer depends on the local
`run.artifacts` package, `RUN_ID`, `RUN_DIR`, or hard-coded `/nvme1` paths.

## What It Does

The script reads an official-style `meas` FITS catalog with serialized source
footprints and computes a sparse-pixel Kron-radius proxy for selected sources.

For each source it:

- reads source metadata, `base_SdssCentroid`, `base_SdssShape`, official Kron
  fields, and flags from the main source table;
- resolves `Footprint` -> `SpanSet` -> `HeavyFootprintF` archive rows;
- uses `HeavyFootprintF` pixels when present;
- falls back to direct reference-image sampling inside ordinary `Footprint`
  spans when `HeavyFootprintF` is absent;
- computes a one-iteration, LSST-like `determineRadius` proxy using the
  sparse pixels;
- writes CSV, JSON, DS9 region layers, a fractional-difference histogram, and
  summary files.

This is a proxy/triage analysis. It does not rebuild `NoiseReplacer` and is not
an exact production LSST `KronFluxAlgorithm` replay.

## Files

```text
batch_heavyfp_kron_refit.py        Standalone analysis script
requirements.txt                   Minimal external Python dependencies
bin/make_example_inputs.py         Synthetic FITS input generator
bin/run_example.sh                 End-to-end example launcher
examples/input/                    Included tiny example inputs
examples/output/example_run/       Included example outputs
```

## Python Dependencies

Install into any ordinary Python environment:

```bash
python -m pip install -r requirements.txt
```

The expected external packages are:

```text
numpy
astropy
matplotlib
```

## Run The Included Example

From this package directory:

```bash
PYTHON=python ./bin/run_example.sh
```

The launcher regenerates the tiny synthetic FITS inputs and writes outputs to:

```text
examples/output/example_run/
```

The main summary is:

```text
examples/output/example_run/summary.md
```

The included example is deliberately small: three sources, two measured from
`HeavyFootprintF` payloads and one measured through the direct-reference-image
fallback.

## Run On Real Data

Use either `--rows-file` or a magnitude selection with `--mag-min --mag-max`.

Rows-file mode:

```bash
python batch_heavyfp_kron_refit.py \
  --meas-catalog /path/to/meas-HSC-I-9813-4,5.fits \
  --reference-image /path/to/step-1_image_njy.fits \
  --rows-file /path/to/rows.txt \
  --output-dir /path/to/output \
  --artifact-name kron_refit_rows
```

Magnitude-bin mode:

```bash
python batch_heavyfp_kron_refit.py \
  --meas-catalog /path/to/meas-HSC-I-9813-4,5.fits \
  --reference-image /path/to/step-1_image_njy.fits \
  --mag-min 25 \
  --mag-max 25.25 \
  --output-dir /path/to/output \
  --artifact-name kron_refit_psfmag25_25p25
```

Set `MPLCONFIGDIR` to a writable directory on restricted hosts:

```bash
MPLCONFIGDIR=/tmp/matplotlib python batch_heavyfp_kron_refit.py ...
```

## Input Requirements

The `--meas-catalog` FITS file must expose the same surface used by the original
HSC `meas-HSC-I-9813-4,5.fits` archive:

- HDU 1: main source table
- HDU 2: archive index table with `id`, `cat.archive`, `name`, `row0`, `nrows`
- HDU 3: footprint reference table with `id`
- HDU 4: span table with `y`, `x0`, `x1`
- HDU 6: HeavyFootprint table with variable-length `image` payloads

The main source table must include:

```text
id
parent
deblend_nChild
footprint
base_FootprintArea_value
coord_ra
coord_dec
base_SdssCentroid_x
base_SdssCentroid_y
base_SdssShape_xx
base_SdssShape_xy
base_SdssShape_yy
base_PsfFlux_instFlux
ext_photometryKron_KronFlux_radius
ext_photometryKron_KronFlux_radius_for_radius
ext_photometryKron_KronFlux_instFlux
ext_photometryKron_KronFlux_instFluxErr
flags
```

The source-table header should provide `TFLAG*` entries for any flags you want
the selection filters to use. Missing flag names are treated as all false.
For magnitude-bin mode, either provide `detect_isPrimary` or pass
`--include-non-primary`; otherwise the default primary-source filter selects no
rows.

The `--reference-image` FITS file must have:

- WCS, used to turn source RA/Dec into image coordinates for DS9 regions;
- `LTV1` and `LTV2`, used to map full-image footprint pixels into the reference
  image array for direct-footprint fallback rows.

## Outputs

Each run writes:

```text
batch_heavyfp_kron_refit.csv
batch_heavyfp_kron_refit.json
official_minus_proxy_over_proxy_histogram.png
official_minus_proxy_over_proxy_histogram.csv
official_minus_proxy_over_proxy_stats.json
regions/
regions_manifest.json
summary.json
summary.md
```

Important output fields:

```text
measurement_surface
proxy_nan0_radius_for_radius
proxy_nan0_raw_first_moment
proxy_nan0_candidate_radius
proxy_nan0_determine_radius_returned_radius
proxy_nan0_flux_aperture_radius
official_minus_proxy_over_proxy
```

`measurement_surface` is either:

```text
heavyfootprintf
direct_footprint_reference_image
```

## Scientific Caveat

Do not describe `proxy_nan0_determine_radius_returned_radius` as the official
LSST/HSC Kron radius. It is a sparse-pixel proxy designed for ranking,
inspection, DS9 overlays, and anomaly triage. Exact production parity would
require reproducing the measurement-time image state and the full LSST/HSC
Kron plugin path.
