"""Measure HSC persisted CoaddPsf EE rings in an LSST subprocess, then cache."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np

ASSET = Path(__file__).resolve().parents[1]/'assets/hsc_confidence_ee10_35_60_70_v1.json'


def measure_psf(reference, output):
    recipe = json.loads(ASSET.read_text()); reference=Path(reference); output=Path(output)
    signature=dict(path=str(reference.resolve()),mtime_ns=reference.stat().st_mtime_ns,
                   recipe_sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest(),
                   extractor_sha256=hashlib.sha256(Path(__file__).with_name('hsc_psf_extract.py').read_bytes()).hexdigest())
    if output.exists():
        saved=json.loads(output.read_text())
        if saved.get('signature')==signature:return saved
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hsc_psf_') as tmp:
        stamp=Path(tmp)/'psf.npz'
        script=Path(__file__).with_name('hsc_psf_extract.py')
        command='source "$1" >/dev/null 2>&1; setup lsst_distrib >/dev/null 2>&1; exec python "$2" "$3" "$4"'
        subprocess.run(['bash','-c',command,'hsc-psf',recipe['lsst_setup'],str(script),str(reference),str(stamp)],check=True)
        from photutils.aperture import CircularAperture
        from scipy.optimize import brentq
        with np.load(stamp) as z:
            results=[]
            positions=json.loads(str(z['positions_json']));sample_errors=json.loads(str(z['errors_json']))
            for key in sorted(k for k in z.files if k.startswith('psf')):
                raw=z[key].astype(float)
                if not np.isfinite(raw).all():raise ValueError('Nonfinite HSC PSF')
                a=np.maximum(raw,0);total=float(a.sum())
                if not total>0:raise ValueError('Empty HSC PSF')
                center=((a.shape[1]-1)/2,(a.shape[0]-1)/2)
                def flux(r):
                    mask=CircularAperture(center,r).to_mask(method='exact')
                    return float(mask.multiply(a).sum())/total
                radii=[brentq(lambda r:flux(r)-f,1e-6,min(a.shape)/2) for f in recipe['fractions']]
                results.append(dict(key=key,position_xy=positions[key],radii_pixels=radii,stamp_sum=float(raw.sum()),
                    positive_sum=total,negative_fraction=float(-raw[raw<0].sum()/total)))
            radii=np.median([r['radii_pixels'] for r in results],axis=0)
            scale=float(z['pixel_scale'])
    saved=dict(mode='psf-ee',definition_id=recipe['definition_id'],signature=signature,
        level_radii_pixels={str(k):float(r) for k,r in zip([4,3,2,1],radii)},
        level_radii_arcsec={str(k):float(r*scale) for k,r in zip([4,3,2,1],radii)},
        pixel_scale_arcsec=scale,ee_fractions={str(k):f for k,f in zip([4,3,2,1],recipe['fractions'])},
        normalization=recipe['normalization'],samples=results,sample_errors=sample_errors,psf_path=str(reference),asset_path=str(ASSET),
        asset_sha256=signature['recipe_sha256'],measurement='exact circular aperture; median of 3x3 spatial samples',
        fallback={'nearest_xy':'floor(source_xy + 0.5)'},fwhm_clip_pixels=None)
    # Distinct jobs may measure the same reference; atomic publication prevents partial JSON.
    with tempfile.NamedTemporaryFile(mode='w',dir=output.parent,delete=False) as f:
        json.dump(saved,f,indent=2);temp=Path(f.name)
    temp.replace(output)
    return saved
