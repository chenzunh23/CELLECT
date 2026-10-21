"""Rebuild fixed Gaia PSF geometry once, never per parent image."""
import hashlib
import json
from pathlib import Path
from astropy.io import fits
from preprocessing.utils.psf import PSF_FWHM_ARCSEC
from preprocessing.utils.jwst_star_psf import measure_psf,ASSET

def main():
    config=dict(schema_version=1,definition_id='jwst_gaia_rp18_fixed_psf_star_wings_v1',
        psf_extension='OVERDIST',normalization='sum of all finite pixels in the full simulation stamp',
        policy=dict(bright_rp_max=18.,untrusted_margin_pixels=32,
            gaia_model_selection='parallax SNR>=3 OR proper-motion SNR>=3; otherwise require image PSF compatibility; no magnitude limit',
            psf_fallback=dict(reference_band='F444W',max_offset_arcsec=.3,min_amplitude_snr=5.,
                min_correlation=.5,min_core_finite_fraction=.5,extension_score_threshold=10.,
                extension_score='2*(robust_point_cost-robust_extended_cost)-log(number_of_fit_pixels)',
                unresolved_sigma_floor_pixels=.5,unresolved_sigma_fwhm_fraction=.25,
                center_policy='PSF-refined center only for fallback candidates; fixed Kron size'),
            star_wings_confidence_weight=.1,reference_band='F444W',star_match_arcsec=.15,
            geometry='fixed per-band full-stamp moments; Kron factor 2.5, first moment within 6 moment radii',
            orientation='rotation only; fit F444W reference when available; current-band fallback recorded',
            flag_veto='continuous raw-Kron intersection with F444W EE95; nonzero flags -> ignore',
            untrusted_region_test='catalog center within RP<=18 footprint dilated by Euclidean 32 pixels'),filters={})
    for band in PSF_FWHM_ARCSEC:
        path=Path('/data/czh23/JWST/NIRCam_PSF')/f'PSF_NIRCam_in_flight_opd_filter_{band}.fits'
        with fits.open(path,memmap=False) as hd:
            hdu=hd[config['psf_extension']]; row=measure_psf(hdu.data,float(hdu.header['PIXELSCL']))
        config['filters'][band]={**row,'path':str(path),'extension':config['psf_extension'],
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        print(band, row['kron_a_arcsec'],row['ee95_a_arcsec'],flush=True)
    ASSET.write_text(json.dumps(config,indent=2)+'\n')

if __name__=='__main__':main()
