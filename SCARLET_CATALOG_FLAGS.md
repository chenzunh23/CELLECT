# Scarlet/HSC Catalog Flags

This document summarizes the packed `flags` bit column in the HSC/Scarlet source catalogs used by this project. It was generated from the FITS schema metadata in `/data1/czh23/Subaru/9813/HSC-I/0,0/meas-HSC-I-9813-0,0.fits` HDU 1, with the peak-catalog flags from HDU 5 listed separately.

The source table stores flags in a compressed boolean-vector column named `flags`. The mapping from bit index to semantic name is in FITS metadata keys `TFLAG<n>` and `TFDOC<n>`. Astropy expands this column as a 2D boolean array.

```python
from astropy.table import Table
import numpy as np

t = Table.read(path, hdu=1)
flag_name_to_bit = {str(v): int(k[5:]) - 1 for k, v in t.meta.items() if k.startswith("TFLAG")}
flags = np.asarray(t["flags"], dtype=bool)
bright_center = flags[:, flag_name_to_bit["base_PixelFlags_flag_bright_objectCenter"]]
```

## Practical Filtering Policy

For CELLECT-style training, do not use one monolithic reject rule. Split rows into `clean_positive`, `uncertain_positive`, and `rejected_bad`. Clean positives should be high-confidence labels. Uncertain positives should be ignored for background/negative loss rather than treated as background.

Recommended conservative clean-GT hard rejects:

- non-finite centroid, flux, or shape; non-positive shape axes
- `deblend_nChild != 0` when training on leaf sources
- `base_PixelFlags_flag_saturatedCenter`
- `base_PixelFlags_flag_clippedCenter`
- `base_PixelFlags_flag_sensor_edgeCenter`
- `base_PixelFlags_flag_bright_objectCenter` for strict clean evaluation
- `modelfit_CModel_flag_badCentroid`
- `modelfit_CModel_flag_region_maxBadPixelFraction`
- `base_SdssShape_flag_shift`, or at least `base_SdssShape_flag_shift and ellipse_area_3sigma > 400`
- explicit shape-size guard such as `ellipse_area_3sigma < 900` for clean shape supervision

Use these mostly as strata, not hard rejects everywhere:

- `base_PixelFlags_flag_bright_object`: footprint-level bright-object overlap; often too broad.
- `base_SdssShape_flag`: broad shape failure; can remove many otherwise detectable sources. Combine with area/axis sanity checks.
- `base_PixelFlags_flag_inexact_psfCenter`, `interpolatedCenter`, `crCenter`: useful for quality diagnostics and clean-eval variants.
- `deblend_*`: crowding diagnostics; combine with `parent`, `deblend_nChild`, footprint area, and bright flags.

## Flag Groups

| Group | Meaning | Suggested use |
|---|---|---|
| Blendedness | Measurement-specific quality/provenance flags. | Use only if the affected measurement is used by the loss or analysis. |
| CModel de Vaucouleur fit | Measurement-specific quality/provenance flags. | Use only if the affected measurement is used by the loss or analysis. |
| CModel exponential fit | Measurement-specific quality/provenance flags. | Use only if the affected measurement is used by the loss or analysis. |
| CModel final fit | Final CModel fit status and fit-region issues. | badCentroid/maxBadPixelFraction are strong rejects; maxArea is a large-object diagnostic. |
| CModel initial fit | Measurement-specific quality/provenance flags. | Use only if the affected measurement is used by the loss or analysis. |
| Calibration use flags | Whether source was used for calibration. | Not a GT quality flag. |
| Circular aperture flux | Measurement-specific quality/provenance flags. | Use only if the affected measurement is used by the loss or analysis. |
| Convolved flux | Measurement-specific quality/provenance flags. | Use only if the affected measurement is used by the loss or analysis. |
| Deblender | Flags from HSC deblender. | Useful for crowded/uncertain strata, especially with parent/nChild and area. |
| Detection region flags | Patch/tract primary-region flags. | Useful for survey-catalog cuts, not local image quality by itself. |
| Double shapelet PSF approximation | Measurement-specific quality/provenance flags. | Use only if the affected measurement is used by the loss or analysis. |
| Gaussian flux | Measurement-specific quality/provenance flags. | Use only if the affected measurement is used by the loss or analysis. |
| HSM PSF moments | Measurement-specific quality/provenance flags. | Use only if the affected measurement is used by the loss or analysis. |
| HSM Regauss shape | Measurement-specific quality/provenance flags. | Use only if the affected measurement is used by the loss or analysis. |
| HSM source moments | Measurement-specific quality/provenance flags. | Use only if the affected measurement is used by the loss or analysis. |
| Input count | Measurement-specific quality/provenance flags. | Use only if the affected measurement is used by the loss or analysis. |
| Kron flux | Measurement-specific quality/provenance flags. | Use only if the affected measurement is used by the loss or analysis. |
| Local background | Measurement-specific quality/provenance flags. | Use only if the affected measurement is used by the loss or analysis. |
| Merge / multi-band detection | Multi-band merge provenance: whether a footprint/peak was detected in each filter. | Do not use as quality reject; useful for band-presence/linking diagnostics. |
| Naive centroid | Measurement-specific quality/provenance flags. | Use only if the affected measurement is used by the loss or analysis. |
| PSF flux | Measurement-specific quality/provenance flags. | Use only if the affected measurement is used by the loss or analysis. |
| Pixel mask flags | Whether source footprint/center touches image mask planes. | Center flags are stronger than footprint flags; bright_object footprint is broad. |
| SDSS centroid | Centroid measurement status. | Use severe center failures for reject/ignore; notAtMaximum is a crowded-source diagnostic. |
| SDSS shape | Adaptive second-moment shape status. | Critical for shape loss. shift=True or huge area means unreliable ellipse. |
| Star/galaxy classification | Measurement-specific quality/provenance flags. | Use only if the affected measurement is used by the loss or analysis. |
| Variance | Measurement-specific quality/provenance flags. | Use only if the affected measurement is used by the loss or analysis. |

## Complete Source-Level Flag Table (HDU 1)

### Blendedness

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 45 | `base_Blendedness_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 46 | `base_Blendedness_flag_noCentroid` | Object has no centroid | Diagnostic; inspect before using as a hard filter. |
| 47 | `base_Blendedness_flag_noShape` | Object has no shape | Diagnostic; inspect before using as a hard filter. |

### CModel de Vaucouleur fit

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 196 | `modelfit_CModel_dev_flag` | flag set when the flux for the de Vaucouleur flux failed | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 197 | `modelfit_CModel_dev_flag_trSmall` | the optimizer converged because the trust radius became too small; this is a less-secure result than when the gradient is below the threshold, but usually not a problem | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 198 | `modelfit_CModel_dev_flag_maxIter` | the optimizer hit the maximum number of iterations and did not converge | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 199 | `modelfit_CModel_dev_flag_numericError` | numerical underflow or overflow in model evaluation; usually this means the prior was insufficient to regularize the fit, or all pixel values were zero. | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 200 | `modelfit_CModel_dev_flag_noFlux` | no flux was measured on the image; this means the error will be non-finite. | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 242 | `modelfit_CModel_dev_flag_apCorr` | set if unable to aperture correct modelfit_CModel_dev | Photometry correction quality; relevant for calibrated flux, less important for center labels. |

### CModel exponential fit

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 191 | `modelfit_CModel_exp_flag` | flag set when the flux for the exponential flux failed | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 192 | `modelfit_CModel_exp_flag_trSmall` | the optimizer converged because the trust radius became too small; this is a less-secure result than when the gradient is below the threshold, but usually not a problem | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 193 | `modelfit_CModel_exp_flag_maxIter` | the optimizer hit the maximum number of iterations and did not converge | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 194 | `modelfit_CModel_exp_flag_numericError` | numerical underflow or overflow in model evaluation; usually this means the prior was insufficient to regularize the fit, or all pixel values were zero. | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 195 | `modelfit_CModel_exp_flag_noFlux` | no flux was measured on the image; this means the error will be non-finite. | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 243 | `modelfit_CModel_exp_flag_apCorr` | set if unable to aperture correct modelfit_CModel_exp | Photometry correction quality; relevant for calibrated flux, less important for center labels. |

### CModel final fit

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 201 | `modelfit_CModel_flag` | flag set if the final cmodel fit (or any previous fit) failed | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 202 | `modelfit_CModel_flag_region_maxArea` | number of pixels in fit region exceeded the region.maxArea value | Conditional reject or quality stratum; do not use as sole global reject without checking counts. |
| 203 | `modelfit_CModel_flag_region_maxBadPixelFraction` | the fraction of bad/clipped pixels in the fit region exceeded region.maxBadPixelFraction | Hard reject for clean GT / shape loss; put in uncertain or rejected. |
| 204 | `modelfit_CModel_flags_region_usedFootprintArea` | the pixel region for the initial fit was defined by the area of the Footprint | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 205 | `modelfit_CModel_flags_region_usedPsfArea` | the pixel region for the initial fit was set to a fixed factor of the PSF area | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 206 | `modelfit_CModel_flags_region_usedInitialEllipseMin` | the pixel region for the final fit was set to the lower bound defined by the initial fit | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 207 | `modelfit_CModel_flags_region_usedInitialEllipseMax` | the pixel region for the final fit was set to the upper bound defined by the initial fit | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 208 | `modelfit_CModel_flag_noShape` | the shape slot needed to initialize the parameters failed or was not defined | Conditional reject or quality stratum; do not use as sole global reject without checking counts. |
| 209 | `modelfit_CModel_flags_smallShape` | initial parameter guess resulted in negative radius; used minimum of 0.100000 pixels instead. | Conditional reject or quality stratum; do not use as sole global reject without checking counts. |
| 210 | `modelfit_CModel_flag_noShapeletPsf` | the multishapelet fit to the PSF model did not succeed | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 211 | `modelfit_CModel_flag_badCentroid` | input centroid was not within the fit region (probably because it's not within the Footprint) | Hard reject for clean GT / shape loss; put in uncertain or rejected. |
| 212 | `modelfit_CModel_flag_noFlux` | no flux was measured on the image; this means the error will be non-finite. | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 241 | `modelfit_CModel_flag_apCorr` | set if unable to aperture correct modelfit_CModel | Photometry correction quality; relevant for calibrated flux, less important for center labels. |

### CModel initial fit

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 186 | `modelfit_CModel_initial_flag` | flag set when the flux for the initial flux failed | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 187 | `modelfit_CModel_initial_flag_trSmall` | the optimizer converged because the trust radius became too small; this is a less-secure result than when the gradient is below the threshold, but usually not a problem | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 188 | `modelfit_CModel_initial_flag_maxIter` | the optimizer hit the maximum number of iterations and did not converge | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 189 | `modelfit_CModel_initial_flag_numericError` | numerical underflow or overflow in model evaluation; usually this means the prior was insufficient to regularize the fit, or all pixel values were zero. | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 190 | `modelfit_CModel_initial_flag_noFlux` | no flux was measured on the image; this means the error will be non-finite. | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 244 | `modelfit_CModel_initial_flag_apCorr` | set if unable to aperture correct modelfit_CModel_initial | Photometry correction quality; relevant for calibrated flux, less important for center labels. |

### Calibration use flags

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 216 | `calib_psf_candidate` | Propagated from visits | Calibration sample membership; not a science-object quality reject. |
| 217 | `calib_psf_used` | Propagated from visits | Calibration sample membership; not a science-object quality reject. |
| 218 | `calib_psf_reserved` | Propagated from visits | Calibration sample membership; not a science-object quality reject. |
| 219 | `calib_astrometry_used` | Propagated from visits | Calibration sample membership; not a science-object quality reject. |
| 220 | `calib_photometry_used` | Propagated from visits | Calibration sample membership; not a science-object quality reject. |
| 221 | `calib_photometry_reserved` | Propagated from visits | Calibration sample membership; not a science-object quality reject. |

### Circular aperture flux

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 77 | `base_CircularApertureFlux_3_0_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 78 | `base_CircularApertureFlux_3_0_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 79 | `base_CircularApertureFlux_3_0_flag_sincCoeffsTruncated` | full sinc coefficient image did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 80 | `base_CircularApertureFlux_4_5_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 81 | `base_CircularApertureFlux_4_5_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 82 | `base_CircularApertureFlux_4_5_flag_sincCoeffsTruncated` | full sinc coefficient image did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 83 | `base_CircularApertureFlux_6_0_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 84 | `base_CircularApertureFlux_6_0_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 85 | `base_CircularApertureFlux_6_0_flag_sincCoeffsTruncated` | full sinc coefficient image did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 86 | `base_CircularApertureFlux_9_0_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 87 | `base_CircularApertureFlux_9_0_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 88 | `base_CircularApertureFlux_9_0_flag_sincCoeffsTruncated` | full sinc coefficient image did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 89 | `base_CircularApertureFlux_12_0_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 90 | `base_CircularApertureFlux_12_0_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 91 | `base_CircularApertureFlux_12_0_flag_sincCoeffsTruncated` | full sinc coefficient image did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 92 | `base_CircularApertureFlux_17_0_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 93 | `base_CircularApertureFlux_17_0_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 94 | `base_CircularApertureFlux_25_0_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 95 | `base_CircularApertureFlux_25_0_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 96 | `base_CircularApertureFlux_35_0_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 97 | `base_CircularApertureFlux_35_0_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 98 | `base_CircularApertureFlux_50_0_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 99 | `base_CircularApertureFlux_50_0_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 100 | `base_CircularApertureFlux_70_0_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 101 | `base_CircularApertureFlux_70_0_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |

### Convolved flux

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 141 | `ext_convolved_ConvolvedFlux_0_deconv` | deconvolution required for seeing 3.500000; no measurement made | Diagnostic; inspect before using as a hard filter. |
| 142 | `ext_convolved_ConvolvedFlux_0_3_3_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 143 | `ext_convolved_ConvolvedFlux_0_3_3_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 144 | `ext_convolved_ConvolvedFlux_0_3_3_flag_sincCoeffsTruncated` | full sinc coefficient image did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 145 | `ext_convolved_ConvolvedFlux_0_4_5_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 146 | `ext_convolved_ConvolvedFlux_0_4_5_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 147 | `ext_convolved_ConvolvedFlux_0_4_5_flag_sincCoeffsTruncated` | full sinc coefficient image did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 148 | `ext_convolved_ConvolvedFlux_0_6_0_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 149 | `ext_convolved_ConvolvedFlux_0_6_0_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 150 | `ext_convolved_ConvolvedFlux_0_6_0_flag_sincCoeffsTruncated` | full sinc coefficient image did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 151 | `ext_convolved_ConvolvedFlux_0_kron_flag` | convolved Kron flux failed: seeing 3.500000 | Diagnostic; inspect before using as a hard filter. |
| 152 | `ext_convolved_ConvolvedFlux_1_deconv` | deconvolution required for seeing 5.000000; no measurement made | Diagnostic; inspect before using as a hard filter. |
| 153 | `ext_convolved_ConvolvedFlux_1_3_3_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 154 | `ext_convolved_ConvolvedFlux_1_3_3_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 155 | `ext_convolved_ConvolvedFlux_1_3_3_flag_sincCoeffsTruncated` | full sinc coefficient image did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 156 | `ext_convolved_ConvolvedFlux_1_4_5_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 157 | `ext_convolved_ConvolvedFlux_1_4_5_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 158 | `ext_convolved_ConvolvedFlux_1_4_5_flag_sincCoeffsTruncated` | full sinc coefficient image did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 159 | `ext_convolved_ConvolvedFlux_1_6_0_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 160 | `ext_convolved_ConvolvedFlux_1_6_0_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 161 | `ext_convolved_ConvolvedFlux_1_6_0_flag_sincCoeffsTruncated` | full sinc coefficient image did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 162 | `ext_convolved_ConvolvedFlux_1_kron_flag` | convolved Kron flux failed: seeing 5.000000 | Diagnostic; inspect before using as a hard filter. |
| 163 | `ext_convolved_ConvolvedFlux_2_deconv` | deconvolution required for seeing 6.500000; no measurement made | Diagnostic; inspect before using as a hard filter. |
| 164 | `ext_convolved_ConvolvedFlux_2_3_3_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 165 | `ext_convolved_ConvolvedFlux_2_3_3_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 166 | `ext_convolved_ConvolvedFlux_2_3_3_flag_sincCoeffsTruncated` | full sinc coefficient image did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 167 | `ext_convolved_ConvolvedFlux_2_4_5_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 168 | `ext_convolved_ConvolvedFlux_2_4_5_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 169 | `ext_convolved_ConvolvedFlux_2_4_5_flag_sincCoeffsTruncated` | full sinc coefficient image did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 170 | `ext_convolved_ConvolvedFlux_2_6_0_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 171 | `ext_convolved_ConvolvedFlux_2_6_0_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 172 | `ext_convolved_ConvolvedFlux_2_6_0_flag_sincCoeffsTruncated` | full sinc coefficient image did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 173 | `ext_convolved_ConvolvedFlux_2_kron_flag` | convolved Kron flux failed: seeing 6.500000 | Diagnostic; inspect before using as a hard filter. |
| 174 | `ext_convolved_ConvolvedFlux_3_deconv` | deconvolution required for seeing 8.000000; no measurement made | Diagnostic; inspect before using as a hard filter. |
| 175 | `ext_convolved_ConvolvedFlux_3_3_3_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 176 | `ext_convolved_ConvolvedFlux_3_3_3_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 177 | `ext_convolved_ConvolvedFlux_3_3_3_flag_sincCoeffsTruncated` | full sinc coefficient image did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 178 | `ext_convolved_ConvolvedFlux_3_4_5_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 179 | `ext_convolved_ConvolvedFlux_3_4_5_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 180 | `ext_convolved_ConvolvedFlux_3_4_5_flag_sincCoeffsTruncated` | full sinc coefficient image did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 181 | `ext_convolved_ConvolvedFlux_3_6_0_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 182 | `ext_convolved_ConvolvedFlux_3_6_0_flag_apertureTruncated` | aperture did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 183 | `ext_convolved_ConvolvedFlux_3_6_0_flag_sincCoeffsTruncated` | full sinc coefficient image did not fit within measurement image | Diagnostic; inspect before using as a hard filter. |
| 184 | `ext_convolved_ConvolvedFlux_3_kron_flag` | convolved Kron flux failed: seeing 8.000000 | Diagnostic; inspect before using as a hard filter. |
| 185 | `ext_convolved_ConvolvedFlux_flag` | error in running ConvolvedFluxPlugin | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 224 | `ext_convolved_ConvolvedFlux_0_3_3_flag_apCorr` | set if unable to aperture correct ext_convolved_ConvolvedFlux_0_3_3 | Photometry correction quality; relevant for calibrated flux, less important for center labels. |
| 225 | `ext_convolved_ConvolvedFlux_0_4_5_flag_apCorr` | set if unable to aperture correct ext_convolved_ConvolvedFlux_0_4_5 | Photometry correction quality; relevant for calibrated flux, less important for center labels. |
| 226 | `ext_convolved_ConvolvedFlux_0_6_0_flag_apCorr` | set if unable to aperture correct ext_convolved_ConvolvedFlux_0_6_0 | Photometry correction quality; relevant for calibrated flux, less important for center labels. |
| 227 | `ext_convolved_ConvolvedFlux_0_kron_flag_apCorr` | set if unable to aperture correct ext_convolved_ConvolvedFlux_0_kron | Photometry correction quality; relevant for calibrated flux, less important for center labels. |
| 228 | `ext_convolved_ConvolvedFlux_1_3_3_flag_apCorr` | set if unable to aperture correct ext_convolved_ConvolvedFlux_1_3_3 | Photometry correction quality; relevant for calibrated flux, less important for center labels. |
| 229 | `ext_convolved_ConvolvedFlux_1_4_5_flag_apCorr` | set if unable to aperture correct ext_convolved_ConvolvedFlux_1_4_5 | Photometry correction quality; relevant for calibrated flux, less important for center labels. |
| 230 | `ext_convolved_ConvolvedFlux_1_6_0_flag_apCorr` | set if unable to aperture correct ext_convolved_ConvolvedFlux_1_6_0 | Photometry correction quality; relevant for calibrated flux, less important for center labels. |
| 231 | `ext_convolved_ConvolvedFlux_1_kron_flag_apCorr` | set if unable to aperture correct ext_convolved_ConvolvedFlux_1_kron | Photometry correction quality; relevant for calibrated flux, less important for center labels. |
| 232 | `ext_convolved_ConvolvedFlux_2_3_3_flag_apCorr` | set if unable to aperture correct ext_convolved_ConvolvedFlux_2_3_3 | Photometry correction quality; relevant for calibrated flux, less important for center labels. |
| 233 | `ext_convolved_ConvolvedFlux_2_4_5_flag_apCorr` | set if unable to aperture correct ext_convolved_ConvolvedFlux_2_4_5 | Photometry correction quality; relevant for calibrated flux, less important for center labels. |
| 234 | `ext_convolved_ConvolvedFlux_2_6_0_flag_apCorr` | set if unable to aperture correct ext_convolved_ConvolvedFlux_2_6_0 | Photometry correction quality; relevant for calibrated flux, less important for center labels. |
| 235 | `ext_convolved_ConvolvedFlux_2_kron_flag_apCorr` | set if unable to aperture correct ext_convolved_ConvolvedFlux_2_kron | Photometry correction quality; relevant for calibrated flux, less important for center labels. |
| 236 | `ext_convolved_ConvolvedFlux_3_3_3_flag_apCorr` | set if unable to aperture correct ext_convolved_ConvolvedFlux_3_3_3 | Photometry correction quality; relevant for calibrated flux, less important for center labels. |
| 237 | `ext_convolved_ConvolvedFlux_3_4_5_flag_apCorr` | set if unable to aperture correct ext_convolved_ConvolvedFlux_3_4_5 | Photometry correction quality; relevant for calibrated flux, less important for center labels. |
| 238 | `ext_convolved_ConvolvedFlux_3_6_0_flag_apCorr` | set if unable to aperture correct ext_convolved_ConvolvedFlux_3_6_0 | Photometry correction quality; relevant for calibrated flux, less important for center labels. |
| 239 | `ext_convolved_ConvolvedFlux_3_kron_flag_apCorr` | set if unable to aperture correct ext_convolved_ConvolvedFlux_3_kron | Photometry correction quality; relevant for calibrated flux, less important for center labels. |

### Deblender

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 26 | `deblend_deblendedAsPsf` | Deblender thought this source looked like a PSF | Crowding/deblend diagnostic; combine with parent/nChild/area for uncertain labels. |
| 27 | `deblend_tooManyPeaks` | Source had too many peaks; only the brightest were included | Crowding/deblend diagnostic; combine with parent/nChild/area for uncertain labels. |
| 28 | `deblend_parentTooBig` | Parent footprint covered too many pixels | Crowding/deblend diagnostic; combine with parent/nChild/area for uncertain labels. |
| 29 | `deblend_masked` | Parent footprint was predominantly masked | Crowding/deblend diagnostic; combine with parent/nChild/area for uncertain labels. |
| 30 | `deblend_skipped` | Deblender skipped this source | Crowding/deblend diagnostic; combine with parent/nChild/area for uncertain labels. |
| 31 | `deblend_rampedTemplate` | This source was near an image edge and the deblender used "ramp" edge-handling. | Crowding/deblend diagnostic; combine with parent/nChild/area for uncertain labels. |
| 32 | `deblend_patchedTemplate` | This source was near an image edge and the deblender used "patched" edge-handling. | Crowding/deblend diagnostic; combine with parent/nChild/area for uncertain labels. |
| 33 | `deblend_hasStrayFlux` | This source was assigned some stray flux | Crowding/deblend diagnostic; combine with parent/nChild/area for uncertain labels. |

### Detection region flags

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 213 | `detect_isPatchInner` | true if source is in the inner region of a coadd patch | Patch/tract primary-region selection; useful for survey-level catalog cuts. |
| 214 | `detect_isTractInner` | true if source is in the inner region of a coadd tract | Patch/tract primary-region selection; useful for survey-level catalog cuts. |
| 215 | `detect_isPrimary` | true if source has no children and is in the inner region of a coadd patch and is in the inner region of a coadd tract and is not "detected" in a pseudo-filter (see config.pseudoFilterList) | Patch/tract primary-region selection; useful for survey-level catalog cuts. |

### Double shapelet PSF approximation

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 73 | `modelfit_DoubleShapeletPsfApprox_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 74 | `modelfit_DoubleShapeletPsfApprox_flag_invalidPointForPsf` | PSF model could not be evaluated at the source position | Diagnostic; inspect before using as a hard filter. |
| 75 | `modelfit_DoubleShapeletPsfApprox_flag_invalidMoments` | Moments of the PSF model were not a valid ellipse | Diagnostic; inspect before using as a hard filter. |
| 76 | `modelfit_DoubleShapeletPsfApprox_flag_maxIterations` | optimizer exceeded the maximum number (inner or outer) iterations | Diagnostic; inspect before using as a hard filter. |

### Gaussian flux

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 102 | `base_GaussianFlux_flag` | General Failure Flag | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 222 | `base_GaussianFlux_flag_apCorr` | set if unable to aperture correct base_GaussianFlux | Photometry correction quality; relevant for calibrated flux, less important for center labels. |

### HSM PSF moments

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 56 | `ext_shapeHSM_HsmPsfMoments_flag` | general failure flag, set if anything went wrong | Diagnostic; inspect before using as a hard filter. |
| 57 | `ext_shapeHSM_HsmPsfMoments_flag_no_pixels` | no pixels to measure | Diagnostic; inspect before using as a hard filter. |
| 58 | `ext_shapeHSM_HsmPsfMoments_flag_not_contained` | center not contained in footprint bounding box | Diagnostic; inspect before using as a hard filter. |
| 59 | `ext_shapeHSM_HsmPsfMoments_flag_parent_source` | parent source, ignored | Diagnostic; inspect before using as a hard filter. |

### HSM Regauss shape

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 60 | `ext_shapeHSM_HsmShapeRegauss_flag` | general failure flag, set if anything went wrong | Diagnostic; inspect before using as a hard filter. |
| 61 | `ext_shapeHSM_HsmShapeRegauss_flag_no_pixels` | no pixels to measure | Diagnostic; inspect before using as a hard filter. |
| 62 | `ext_shapeHSM_HsmShapeRegauss_flag_not_contained` | center not contained in footprint bounding box | Diagnostic; inspect before using as a hard filter. |
| 63 | `ext_shapeHSM_HsmShapeRegauss_flag_parent_source` | parent source, ignored | Diagnostic; inspect before using as a hard filter. |
| 64 | `ext_shapeHSM_HsmShapeRegauss_flag_galsim` | GalSim failure | Diagnostic; inspect before using as a hard filter. |

### HSM source moments

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 65 | `ext_shapeHSM_HsmSourceMoments_flag` | general failure flag, set if anything went wrong | Diagnostic; inspect before using as a hard filter. |
| 66 | `ext_shapeHSM_HsmSourceMoments_flag_no_pixels` | no pixels to measure | Diagnostic; inspect before using as a hard filter. |
| 67 | `ext_shapeHSM_HsmSourceMoments_flag_not_contained` | center not contained in footprint bounding box | Diagnostic; inspect before using as a hard filter. |
| 68 | `ext_shapeHSM_HsmSourceMoments_flag_parent_source` | parent source, ignored | Diagnostic; inspect before using as a hard filter. |
| 69 | `ext_shapeHSM_HsmSourceMomentsRound_flag` | general failure flag, set if anything went wrong | Diagnostic; inspect before using as a hard filter. |
| 70 | `ext_shapeHSM_HsmSourceMomentsRound_flag_no_pixels` | no pixels to measure | Diagnostic; inspect before using as a hard filter. |
| 71 | `ext_shapeHSM_HsmSourceMomentsRound_flag_not_contained` | center not contained in footprint bounding box | Diagnostic; inspect before using as a hard filter. |
| 72 | `ext_shapeHSM_HsmSourceMomentsRound_flag_parent_source` | parent source, ignored | Diagnostic; inspect before using as a hard filter. |

### Input count

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 48 | `base_InputCount_flag` | Set for any fatal failure | Diagnostic; inspect before using as a hard filter. |
| 49 | `base_InputCount_flag_noInputs` | No coadd inputs available | Diagnostic; inspect before using as a hard filter. |

### Kron flux

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 131 | `ext_photometryKron_KronFlux_flag` | general failure flag, set if anything went wrong | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 132 | `ext_photometryKron_KronFlux_flag_edge` | bad measurement due to image edge | Diagnostic; inspect before using as a hard filter. |
| 133 | `ext_photometryKron_KronFlux_flag_bad_shape_no_psf` | bad shape and no PSF | Diagnostic; inspect before using as a hard filter. |
| 134 | `ext_photometryKron_KronFlux_flag_no_minimum_radius` | minimum radius could not enforced: no minimum value or PSF | Diagnostic; inspect before using as a hard filter. |
| 135 | `ext_photometryKron_KronFlux_flag_no_fallback_radius` | no minimum radius and no PSF provided | Diagnostic; inspect before using as a hard filter. |
| 136 | `ext_photometryKron_KronFlux_flag_bad_radius` | bad Kron radius | Diagnostic; inspect before using as a hard filter. |
| 137 | `ext_photometryKron_KronFlux_flag_used_minimum_radius` | used the minimum radius for the Kron aperture | Diagnostic; inspect before using as a hard filter. |
| 138 | `ext_photometryKron_KronFlux_flag_used_psf_radius` | used the PSF Kron radius for the Kron aperture | Diagnostic; inspect before using as a hard filter. |
| 139 | `ext_photometryKron_KronFlux_flag_small_radius` | measured Kron radius was smaller than that of the PSF | Diagnostic; inspect before using as a hard filter. |
| 140 | `ext_photometryKron_KronFlux_flag_bad_shape` | shape for measuring Kron radius is bad; used PSF shape | Diagnostic; inspect before using as a hard filter. |
| 240 | `ext_photometryKron_KronFlux_flag_apCorr` | set if unable to aperture correct ext_photometryKron_KronFlux | Photometry correction quality; relevant for calibrated flux, less important for center labels. |

### Local background

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 103 | `base_LocalBackground_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 104 | `base_LocalBackground_flag_noGoodPixels` | no good pixels in the annulus | Diagnostic; inspect before using as a hard filter. |
| 105 | `base_LocalBackground_flag_noPsf` | no PSF provided | Diagnostic; inspect before using as a hard filter. |

### Merge / multi-band detection

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 0 | `merge_footprint_i2` | Detection footprint overlapped with a detection from filter i2 | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 1 | `merge_footprint_i` | Detection footprint overlapped with a detection from filter i | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 2 | `merge_footprint_r2` | Detection footprint overlapped with a detection from filter r2 | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 3 | `merge_footprint_r` | Detection footprint overlapped with a detection from filter r | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 4 | `merge_footprint_z` | Detection footprint overlapped with a detection from filter z | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 5 | `merge_footprint_y` | Detection footprint overlapped with a detection from filter y | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 6 | `merge_footprint_g` | Detection footprint overlapped with a detection from filter g | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 7 | `merge_footprint_N921` | Detection footprint overlapped with a detection from filter N921 | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 8 | `merge_footprint_N816` | Detection footprint overlapped with a detection from filter N816 | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 9 | `merge_footprint_N1010` | Detection footprint overlapped with a detection from filter N1010 | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 10 | `merge_footprint_N387` | Detection footprint overlapped with a detection from filter N387 | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 11 | `merge_footprint_N515` | Detection footprint overlapped with a detection from filter N515 | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 12 | `merge_footprint_sky` | Detection footprint overlapped with a detection from filter sky | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 13 | `merge_peak_i2` | Peak detected in filter i2 | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 14 | `merge_peak_i` | Peak detected in filter i | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 15 | `merge_peak_r2` | Peak detected in filter r2 | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 16 | `merge_peak_r` | Peak detected in filter r | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 17 | `merge_peak_z` | Peak detected in filter z | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 18 | `merge_peak_y` | Peak detected in filter y | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 19 | `merge_peak_g` | Peak detected in filter g | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 20 | `merge_peak_N921` | Peak detected in filter N921 | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 21 | `merge_peak_N816` | Peak detected in filter N816 | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 22 | `merge_peak_N1010` | Peak detected in filter N1010 | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 23 | `merge_peak_N387` | Peak detected in filter N387 | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 24 | `merge_peak_N515` | Peak detected in filter N515 | Detection provenance; useful for diagnostics, not a quality reject by itself. |
| 25 | `merge_peak_sky` | Peak detected in filter sky | Detection provenance; useful for diagnostics, not a quality reject by itself. |

### Naive centroid

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 34 | `base_NaiveCentroid_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 35 | `base_NaiveCentroid_flag_noCounts` | Object to be centroided has no counts | Diagnostic; inspect before using as a hard filter. |
| 36 | `base_NaiveCentroid_flag_edge` | Object too close to edge | Diagnostic; inspect before using as a hard filter. |
| 37 | `base_NaiveCentroid_flag_resetToPeak` | set if CentroidChecker reset the centroid | Diagnostic; inspect before using as a hard filter. |

### PSF flux

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 126 | `base_PsfFlux_flag` | General Failure Flag | Photometry/model-fit quality; use only for the flux/shape quantity it affects. |
| 127 | `base_PsfFlux_flag_noGoodPixels` | not enough non-rejected pixels in data to attempt the fit | Diagnostic; inspect before using as a hard filter. |
| 128 | `base_PsfFlux_flag_edge` | object was too close to the edge of the image to use the full PSF model | Diagnostic; inspect before using as a hard filter. |
| 223 | `base_PsfFlux_flag_apCorr` | set if unable to aperture correct base_PsfFlux | Photometry correction quality; relevant for calibrated flux, less important for center labels. |

### Pixel mask flags

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 106 | `base_PixelFlags_flag` | General failure flag, set if anything went wrong | Diagnostic; inspect before using as a hard filter. |
| 107 | `base_PixelFlags_flag_offimage` | Source center is off image | Diagnostic; inspect before using as a hard filter. |
| 108 | `base_PixelFlags_flag_edge` | Source is outside usable exposure region (masked EDGE or NO_DATA) | Diagnostic; inspect before using as a hard filter. |
| 109 | `base_PixelFlags_flag_interpolated` | Interpolated pixel in the Source footprint | Diagnostic; inspect before using as a hard filter. |
| 110 | `base_PixelFlags_flag_saturated` | Saturated pixel in the Source footprint | Conditional reject or quality stratum; do not use as sole global reject without checking counts. |
| 111 | `base_PixelFlags_flag_cr` | Cosmic ray in the Source footprint | Diagnostic; inspect before using as a hard filter. |
| 112 | `base_PixelFlags_flag_bad` | Bad pixel in the Source footprint | Diagnostic; inspect before using as a hard filter. |
| 113 | `base_PixelFlags_flag_suspect` | Source's footprint includes suspect pixels | Diagnostic; inspect before using as a hard filter. |
| 114 | `base_PixelFlags_flag_interpolatedCenter` | Interpolated pixel in the Source center | Conditional reject or quality stratum; do not use as sole global reject without checking counts. |
| 115 | `base_PixelFlags_flag_saturatedCenter` | Saturated pixel in the Source center | Hard reject for clean GT / shape loss; put in uncertain or rejected. |
| 116 | `base_PixelFlags_flag_crCenter` | Cosmic ray in the Source center | Conditional reject or quality stratum; do not use as sole global reject without checking counts. |
| 117 | `base_PixelFlags_flag_suspectCenter` | Source's center is close to suspect pixels | Diagnostic; inspect before using as a hard filter. |
| 118 | `base_PixelFlags_flag_clippedCenter` | Source center is close to CLIPPED pixels | Hard reject for clean GT / shape loss; put in uncertain or rejected. |
| 119 | `base_PixelFlags_flag_sensor_edgeCenter` | Source center is close to SENSOR_EDGE pixels | Hard reject for clean GT / shape loss; put in uncertain or rejected. |
| 120 | `base_PixelFlags_flag_inexact_psfCenter` | Source center is close to INEXACT_PSF pixels | Conditional reject or quality stratum; do not use as sole global reject without checking counts. |
| 121 | `base_PixelFlags_flag_bright_objectCenter` | Source center is close to BRIGHT_OBJECT pixels | Hard reject for clean GT / shape loss; put in uncertain or rejected. |
| 122 | `base_PixelFlags_flag_clipped` | Source footprint includes CLIPPED pixels | Conditional reject or quality stratum; do not use as sole global reject without checking counts. |
| 123 | `base_PixelFlags_flag_sensor_edge` | Source footprint includes SENSOR_EDGE pixels | Conditional reject or quality stratum; do not use as sole global reject without checking counts. |
| 124 | `base_PixelFlags_flag_inexact_psf` | Source footprint includes INEXACT_PSF pixels | Conditional reject or quality stratum; do not use as sole global reject without checking counts. |
| 125 | `base_PixelFlags_flag_bright_object` | Source footprint includes BRIGHT_OBJECT pixels | Conditional reject or quality stratum; do not use as sole global reject without checking counts. |

### SDSS centroid

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 38 | `base_SdssCentroid_flag` | General Failure Flag | Diagnostic; inspect before using as a hard filter. |
| 39 | `base_SdssCentroid_flag_edge` | Object too close to edge | Diagnostic; inspect before using as a hard filter. |
| 40 | `base_SdssCentroid_flag_noSecondDerivative` | Vanishing second derivative | Diagnostic; inspect before using as a hard filter. |
| 41 | `base_SdssCentroid_flag_almostNoSecondDerivative` | Almost vanishing second derivative | Diagnostic; inspect before using as a hard filter. |
| 42 | `base_SdssCentroid_flag_notAtMaximum` | Object is not at a maximum | Conditional reject or quality stratum; do not use as sole global reject without checking counts. |
| 43 | `base_SdssCentroid_flag_resetToPeak` | set if CentroidChecker reset the centroid | Diagnostic; inspect before using as a hard filter. |
| 44 | `base_SdssCentroid_flag_badError` | Error on x and/or y position is NaN | Diagnostic; inspect before using as a hard filter. |

### SDSS shape

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 50 | `base_SdssShape_flag` | General Failure Flag | Conditional reject or quality stratum; do not use as sole global reject without checking counts. |
| 51 | `base_SdssShape_flag_unweightedBad` | Both weighted and unweighted moments were invalid | Diagnostic; inspect before using as a hard filter. |
| 52 | `base_SdssShape_flag_unweighted` | Weighted moments converged to an invalid value; using unweighted moments | Conditional reject or quality stratum; do not use as sole global reject without checking counts. |
| 53 | `base_SdssShape_flag_shift` | centroid shifted by more than the maximum allowed amount | Hard reject for clean GT / shape loss; put in uncertain or rejected. |
| 54 | `base_SdssShape_flag_maxIter` | Too many iterations in adaptive moments | Conditional reject or quality stratum; do not use as sole global reject without checking counts. |
| 55 | `base_SdssShape_flag_psf` | Failure in measuring PSF model shape | Diagnostic; inspect before using as a hard filter. |

### Star/galaxy classification

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 245 | `base_ClassificationExtendedness_flag` | Set to 1 for any fatal failure. | Diagnostic; inspect before using as a hard filter. |

### Variance

| Bit | Flag | FITS description | Recommended use |
|---:|---|---|---|
| 129 | `base_Variance_flag` | Set for any fatal failure | Diagnostic; inspect before using as a hard filter. |
| 130 | `base_Variance_flag_emptyFootprint` | Set to True when the footprint has no usable pixels | Diagnostic; inspect before using as a hard filter. |

## Peak-Catalog Flags (HDU 5)

The FITS file also contains a peak catalog in HDU 5. Its `flags` column has only merge-peak provenance bits. These are not source quality flags.

| Bit | Flag | FITS description |
|---:|---|---|
| 0 | `merge_peak_i2` | Peak detected in filter i2 |
| 1 | `merge_peak_i` | Peak detected in filter i |
| 2 | `merge_peak_r2` | Peak detected in filter r2 |
| 3 | `merge_peak_r` | Peak detected in filter r |
| 4 | `merge_peak_z` | Peak detected in filter z |
| 5 | `merge_peak_y` | Peak detected in filter y |
| 6 | `merge_peak_g` | Peak detected in filter g |
| 7 | `merge_peak_N921` | Peak detected in filter N921 |
| 8 | `merge_peak_N816` | Peak detected in filter N816 |
| 9 | `merge_peak_N1010` | Peak detected in filter N1010 |
| 10 | `merge_peak_N387` | Peak detected in filter N387 |
| 11 | `merge_peak_N515` | Peak detected in filter N515 |
| 12 | `merge_peak_sky` | Peak detected in filter sky |

## Notes for Current CELLECT Data Cleaning

- Very large SDSS ellipses can pass bright/CModel filters if `base_SdssShape_flag_shift` is not rejected. In patch `9813/0,0`, clean candidates included ellipses with `ellipse_area_3sigma ~ 1e7` until an explicit area/shift filter was added.
- `base_FootprintArea_value` and SDSS ellipse area are different concepts. A source can have small footprint but invalid huge SDSS moments, or large footprint with small fitted ellipse. Use both for diagnostics.
- For PU training, uncertain/rejected-source neighborhoods should be ignored or sampled carefully; they should not become dense background negatives.
