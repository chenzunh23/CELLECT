"""Source-ID aligned eligibility for catalog/instance mask moment targets."""
import numpy as np

from preprocessing.labels import SourceClass as C

POLICY = 'clean_or_isolated_bright_weak_v1'
JWST_POLICY = 'jwst_clean_or_high_snr_weak_v1'
ISOLATED_BRIGHT_WEAK_REASONS = (
    'single_cluster_large_bright_component_weak_shape',
    'isolated_bright_galaxy_weak_shape',
    'no_upper_isolated_large_aperture_weak_shape',
)


def jwst_moment_snr(table, *, band=None, count=None):
    """Prefer the classification's measured SNR, never fall back on a bad row.

    UNCOVER/legacy Abell use measured reference-image aperture SNR. COSMOS
    uses its own band's catalog SNR. Masked appended PSF rows stay unknown.
    """
    count = len(table) if count is None else count
    names = ['remeasured_aperture_snr', 'classification_snr']
    if band:
        names.append(f'snr_{str(band).lower()}')
    for name in names:
        if name in table.colnames:
            values = np.ma.asarray(table[name], dtype=float).filled(np.nan)
            if values.shape != (count,):
                raise ValueError('Moment SNR/source classes must align')
            return values, name
    return np.full(count, np.nan), None


def moment_eligibility(table, classes, geometry, *, reasons=None, dataset='hsc',
                       weak_snr_threshold=7., band=None):
    """Never infer bright eligibility from a dense region or from SNR alone.

    Bright galaxies reuse the authoritative isolated/low-blendedness decisions.
    Gaia PSF weak-shape rows additionally need no ellipse intersection with
    another active catalog source, using the bright pipeline's overlap test.
    """
    classes=np.asarray(classes)
    out=classes == int(C.CLEAN)
    if dataset in ('cosmos', 'abell', 'a2744', 'jwst'):
        if not np.isfinite(weak_snr_threshold) or weak_snr_threshold <= 0:
            raise ValueError('JWST moment weak-SNR threshold must be finite and positive')
        snr, _ = jwst_moment_snr(table, band=band, count=len(classes))
        out |= ((classes == int(C.WEAK_SHAPE)) & np.isfinite(snr)
                & (snr >= float(weak_snr_threshold)))
        return out & geometry.valid()
    if dataset != 'hsc':
        raise ValueError(f'Unknown moment-eligibility dataset: {dataset}')
    if reasons is None:
        reasons=(np.ma.asarray(table['classification_reason']).filled('')
                 if 'classification_reason' in table.colnames else np.full(len(classes),'',object))
    reasons=np.asarray(reasons,str)
    if reasons.shape != classes.shape:raise ValueError('Moment reasons/source classes must align')
    weak=classes == int(C.WEAK_SHAPE)
    out |= weak & np.isin(reasons,ISOLATED_BRIGHT_WEAK_REASONS)
    models=(np.ma.asarray(table['gaia_psf_model'],dtype=bool).filled(False)
            if 'gaia_psf_model' in table.colnames else np.zeros(len(classes),bool))
    candidates=np.flatnonzero(weak & models & (reasons == 'hsc_gaia_psf_sat_within_ee95'))
    if len(candidates):
        from preprocessing.bright_label import source_intersects_any
        sources=[dict(x=float(x),y=float(y),major=float(a),minor=float(b),theta_deg=float(np.degrees(t)),
                      area=float(np.pi*a*b))
                 for x,y,a,b,t in zip(geometry.x,geometry.y,geometry.major,geometry.minor,geometry.theta)]
        active=np.flatnonzero(np.isin(classes,[C.CLEAN,C.WEAK_SHAPE,C.STRICT_CENTER_ONLY]) & geometry.valid())
        for i in candidates:
            out[i]=not source_intersects_any(int(i),active,sources)
    return out & geometry.valid()
