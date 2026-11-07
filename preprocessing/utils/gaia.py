"""Shared Gaia astrometry, independent of dataset label policy."""

import numpy as np
from functools import lru_cache
from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time


def astrometric_evidence(table, *, allow_absolute_motion=True):
    """COSMOS reference stellar evidence; missing astrometry is not evidence."""
    n = len(table)
    def col(name):
        if name not in table.colnames:
            return np.full(n, np.nan)
        return np.ma.asarray(table[name], dtype=float).filled(np.nan)
    def significance(value, error):
        out = np.full(n, np.nan)
        good = np.isfinite(value) & np.isfinite(error) & (error > 0)
        out[good] = value[good] / error[good]
        return out
    parallax = significance(col('parallax'), col('parallax_error'))
    pmra, pmdec = col('pmra'), col('pmdec')
    pm_snr = np.hypot(significance(pmra, col('pmra_error')), significance(pmdec, col('pmdec_error')))
    evidence = (parallax >= 3) | (pm_snr >= 3)
    return evidence | (np.hypot(pmra, pmdec) >= 3) if allow_absolute_motion else evidence


def matching_radius_pixels(pixel_scale_arcsec, *, arcsec=None, hsc_pixels=6.0):
    """Convert an angular radius, with legacy HSC-pixel option as fallback."""
    radius = float(arcsec) if arcsec is not None else float(hsc_pixels) * 0.168
    if not np.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec <= 0:
        raise ValueError("pixel_scale_arcsec must be finite and positive")
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("Gaia matching radius must be finite and nonnegative")
    return radius / pixel_scale_arcsec


def observation_time(header):
    """Prefer exposure midpoint; do not invent an epoch for undated images."""
    if header is None:
        return None
    for key, fmt in (("MJD-AVG", "mjd"), ("DATE-AVG", None),
                     ("MJD-OBS", "mjd"), ("DATE-OBS", None), ("DATE-BEG", None)):
        if key not in header:
            continue
        try:
            time = Time(float(header[key]), format=fmt) if fmt else Time(header[key])
            if np.isfinite(time.mjd):
                return time
        except (ValueError, TypeError):
            continue
    return None


def propagated_gaia_radec(table, obstime):
    """Use pmra=mu_alpha*cos(dec); missing motion/epoch keeps catalog position."""
    def column(name):
        if name not in table.colnames:
            return np.full(len(table), np.nan)
        return np.asarray(np.ma.asarray(table[name], dtype=float).filled(np.nan))

    ra, dec = column("ra").copy(), column("dec").copy()
    pmra, pmdec, epoch = column("pmra"), column("pmdec"), column("ref_epoch")
    movable = np.isfinite(ra) & np.isfinite(dec) & (np.abs(dec) <= 90)
    movable &= np.isfinite(pmra) & np.isfinite(pmdec) & np.isfinite(epoch)
    if obstime is None:
        movable[:] = False
    if np.any(movable):
        coord = SkyCoord(ra=ra[movable]*u.deg, dec=dec[movable]*u.deg,
                         pm_ra_cosdec=pmra[movable]*u.mas/u.yr,
                         pm_dec=pmdec[movable]*u.mas/u.yr,
                         obstime=Time(epoch[movable], format="jyear"))
        moved = coord.apply_space_motion(new_obstime=obstime)
        ra[movable], dec[movable] = moved.ra.deg, moved.dec.deg
    return ra, dec, movable


@lru_cache(maxsize=32)
def _hsc_visit_epochs(path, mtime_ns):
    """Read persisted VisitInfo/CoaddInputs tables, without loading SCI or LSST."""
    from astropy.io import fits
    with fits.open(path, memmap=True) as hd:
        tables = {int(h.header['AR_CATN']):h for h in hd
                  if isinstance(h,fits.BinTableHDU) and 'AR_CATN' in h.header}
        index = next((h.data for h in tables.values() if {'id','cat.archive','row0','name'} <= set(h.columns.names)), None)
        if index is None:raise ValueError(f'No persisted HSC archive: {path}')
        vi = {}
        for row in index[np.char.strip(np.asarray(index['name']).astype(str))=='VisitInfo']:
            h = tables[int(row['cat.archive'])]
            value = int(h.data['tai'][int(row['row0'])])
            if value > 0:
                vi[int(row['id'])] = float(Time(value/1e9,format='unix_tai',scale='tai').utc.mjd)
        epochs = {}
        for h in tables.values():
            if not {'visit','visitInfo'} <= set(h.columns.names):continue
            for row in h.data:
                date = vi.get(int(row['visitInfo']))
                if date is None:continue
                visit=int(row['visit']);weight=float(row['weight']) if 'weight' in h.columns.names else 1.
                if not np.isfinite(weight) or weight<=0:continue
                if visit in epochs and abs(epochs[visit][0]-date)>1e-4:
                    raise ValueError(f'Inconsistent exposure dates for HSC visit {visit}')
                epochs[visit]=(date,epochs.get(visit,(date,0.))[1]+weight)
    if not epochs:raise ValueError(f'No valid HSC input-exposure dates: {path}')
    return epochs


def hsc_observation_epoch(image_path, reference_path=None):
    """Effective UTC MJD, using selected-product visit weights when available.

    If a stripped half/noisy FITS has no matching weight rows, the full coadd
    reference epoch is explicitly reported as a fallback rather than invented.
    """
    import csv,json
    from astropy.io import fits
    image_path=Path(image_path).resolve();reference=Path(reference_path or image_path).resolve()
    with fits.open(image_path,memmap=True) as hd:
        for h in hd:
            epoch=observation_time(h.header)
            if epoch is not None:
                return dict(mjd_utc=float(epoch.utc.mjd),source='input FITS header',reference=str(image_path))
    epochs=_hsc_visit_epochs(str(reference),reference.stat().st_mtime_ns)
    selected={};weight_path=None
    # Product layout: <products>/<half_coadd|noisy>/<tract>/<band>/<patch>/<file>.
    if len(image_path.parents)>=5 and image_path.parents[3].name in ('half_coadd','noisy'):
        weight_path=image_path.parents[4]/'warp_weights'/image_path.parents[2].name/image_path.parents[1].name/image_path.parent.name/'weights.csv'
        if weight_path.is_file():
            with weight_path.open() as handle:
                for row in csv.DictReader(handle):
                    raw=row.get('product_image_mask_fits','')
                    if not raw or Path(raw).resolve()!=image_path:continue
                    visit=int(row['visit']);weight=float(row['weight'])
                    if weight>0 and np.isfinite(weight):selected[visit]=selected.get(visit,0.)+weight
        if not selected and image_path.parents[3].name=='half_coadd':
            # Some native weight CSVs contain only noisy rows. Recover the
            # exact half selection from its original build manifest, not a
            # guessed half of the full exposure list.
            products=image_path.parents[4];patch=image_path.parent.name;band=image_path.parents[1].name
            choices=[]
            for summary in (products/'_runs').glob(f'*/analysis_summaries/{patch}/half/{band}/*summary.json'):
                doc=json.loads(summary.read_text());target=doc.get('outputs',{}).get('product_image_mask_fits')
                if not target or Path(target).resolve()!=image_path:continue
                manifest=Path(doc['input_manifest']);raw=json.loads(manifest.read_text())
                rows=raw['selected_inputs']
                if len(rows)!=doc['n_inputs'] or len(rows)!=raw['n_selected_inputs']:
                    raise ValueError(f'Invalid HSC input manifest: {manifest}')
                choice={int(r['visit']):float(r['weight']) for r in rows}
                if any(not np.isfinite(w) or w<=0 for w in choice.values()):raise ValueError('Invalid HSC visit weights')
                choices.append((choice,manifest))
            if choices:
                if any(c!=choices[0][0] for c,_ in choices):raise ValueError(f'Ambiguous HSC input manifests: {image_path}')
                selected,weight_path=choices[0]
    if selected:
        missing=set(selected)-set(epochs)
        if missing:raise ValueError(f'HSC product visits absent from reference dates: {sorted(missing)}')
        values=[epochs[v][0] for v in selected];weights=list(selected.values())
        method='selected-product visit-weighted epoch'
    else:
        values=[v[0] for v in epochs.values()];weights=[v[1] for v in epochs.values()]
        method='full-coadd input-weighted epoch' if image_path==reference else 'full-reference epoch fallback: selected-product dates unavailable'
    return dict(mjd_utc=float(np.average(values,weights=weights)),source=method,reference=str(reference),
                visits=len(values),mjd_range=[min(values),max(values)],weight_csv=str(weight_path) if weight_path else None)
