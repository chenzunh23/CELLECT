"""COSMOS catalog classification on an independent WCS window (no HSC adapter)."""
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from ..meas_processing import classify_catalog_basics
from ..ordinary_common import OrdinaryInput, classify_ordinary_sources
from ..bright_label_jwst import JWSTBrightConfig, label_bright_sources
from ..region_filling import fill_dense_regions
from .geometry import EllipseGeometry
from .image_level import PatchLabels
from .segmentation import cosmos_fill_ratios
from .psf import PSF_FWHM_ARCSEC


def load_catalog(path, band):
    band = band.lower()
    names = ['id','ra','dec','kron2_a','kron2_b','theta_world','warn_flag','flag_star',
             f'mag_auto_{band}',f'mag_model_{band}',f'snr_{band}']
    with fits.open(path, memmap=True) as hs:
        phot, gal = hs[1].data, hs[7].data
        return Table({k: np.array((gal if k in gal.names else phot)[k]) for k in names})


def classify_window(raw, header, catalog, catalog_path, gaia, background, band):
    w = WCS(header).celestial
    scale = float(np.mean(proj_plane_pixel_scales(w))*3600)
    x, y = w.all_world2pix(catalog['ra'], catalog['dec'], 0)
    h, width = raw.shape
    ii = np.flatnonzero(np.isfinite(x+y)&(x>=0)&(y>=0)&(x<width)&(y<h))
    tab = catalog[ii]
    # Preserve the catalog/reference pipeline angle convention on COSMOS grids.
    # All registered COSMOS products use the same north-up square-pixel WCS.
    matrix = w.pixel_scale_matrix
    if abs(matrix[0,1])+abs(matrix[1,0]) > 1e-5*np.max(np.abs(matrix)):
        raise ValueError('COSMOS catalog angles require the registered north-up grid')
    a = np.asarray(tab['kron2_a'], float)/scale
    b = np.asarray(tab['kron2_b'], float)/scale
    theta = np.deg2rad(np.asarray(tab['theta_world'], float))
    geom = EllipseGeometry(x[ii], y[ii], a, b, theta, np.pi*a*b)
    low = band.lower()
    data = OrdinaryInput(geom, np.asarray(tab[f'mag_auto_{low}']),
        snr=np.asarray(tab[f'snr_{low}']), flags=np.asarray(tab['warn_flag']),
        star_mask=np.asarray(tab['flag_star']), model_mag=np.asarray(tab[f'mag_model_{low}']))
    basics = classify_catalog_basics(geom, data.mag, dataset='cosmos',pixel_scale_arcsec=scale,image=raw)
    data.center_invalid = basics.nan_center_ignore
    keep = basics.after_b_basic & ~basics.nan_center_ignore
    data.segmentation_fill_ratio = cosmos_fill_ratios(geom, np.asarray(tab['id']), w, catalog_path,
                                                     selected=keep & (data.mag<25.5))
    ordinary = classify_ordinary_sources(data, keep, basics.labels, dataset='cosmos')
    bright = label_bright_sources(data, ordinary.labels, image_shape=raw.shape, image_header=header,
        gaia_table=gaia,source_ids=np.asarray(tab['id']),
        config=JWSTBrightConfig('cosmos',scale,PSF_FWHM_ARCSEC[band],gaia_reference_epoch=2016.))
    dense = fill_dense_regions(tab, bright.labels, raw.shape, geometry=geom,
                              background_mask=background, quality_ignore_mask=~np.isfinite(raw))
    return PatchLabels(tab,dense,bright.labels.source_class.copy(),geom.x,geom.y,geom.major,geom.minor,
        geom.theta,np.asarray(tab['id'],np.int64),bright.strict_center_x,bright.strict_center_y,
        bright.strict_center_source_id,
        strict_is_gaia=np.array(['gaia' in str(r) for r in bright.strict_center_reason],bool))
