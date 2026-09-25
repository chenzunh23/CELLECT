"""Shared Gaia astrometry, independent of dataset label policy."""

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time


def astrometric_evidence(table):
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
    return (parallax >= 3) | (pm_snr >= 3) | (np.hypot(pmra, pmdec) >= 3)


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

    ra, dec = column("ra"), column("dec")
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
