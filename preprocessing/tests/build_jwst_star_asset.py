"""Rebuild fixed Gaia PSF geometry once, never per parent image."""
import hashlib
import json
from pathlib import Path
from astropy.io import fits
from preprocessing.utils.psf import PSF_FWHM_ARCSEC
from preprocessing.utils.jwst_star_psf import measure_psf,ASSET

def main():
    # Geometry rebuilding must not reset the current selection policy or
    # measured adaptive RMS used by SExtractor stellar masks.
    previous=json.loads(ASSET.read_text())
    config={k:v for k,v in previous.items() if k!='filters'}
    config['filters']={}
    for band in PSF_FWHM_ARCSEC:
        path=Path('/data/czh23/JWST/NIRCam_PSF')/f'PSF_NIRCam_in_flight_opd_filter_{band}.fits'
        with fits.open(path,memmap=False) as hd:
            hdu=hd[config['psf_extension']]; row=measure_psf(hdu.data,float(hdu.header['PIXELSCL']))
        config['filters'][band]={**row,'adaptive_sigma_arcsec':previous['filters'][band]['adaptive_sigma_arcsec'],
            'path':str(path),'extension':config['psf_extension'],
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        print(band, row['kron_a_arcsec'],row['ee95_a_arcsec'],flush=True)
    ASSET.write_text(json.dumps(config,indent=2)+'\n')

if __name__=='__main__':main()
