"""Abell source labeling on full-coadd references using existing v3 policies."""
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from scipy import ndimage
from ..image_processing import prepare_image, build_bright_components, BrightRegionConfig
from ..meas_processing import classify_catalog_basics
from ..ordinary_common import OrdinaryInput, classify_ordinary_sources
from ..ordinary_a2744 import A2744OrdinaryConfig
from ..bright_label import project_gaia_rows
from ..bright_label_jwst import JWSTBrightConfig, label_bright_sources
from ..region_filling import fill_dense_regions
from ..labels import SourceClass, DenseLabel
from .geometry import EllipseGeometry, component_at
from .image_level import PatchLabels
from .jwst_background import PSF_FWHM_ARCSEC


def star_footprint(prepared, bright, gaia_rows):
    # 09-08/09-09 Gaia-seeded high/low threshold footprint policy.
    stars=[r for r in gaia_rows if np.isfinite(r['phot_g_mean_mag']) and r['phot_g_mean_mag']<18]
    if not stars:return np.zeros(bright.shape,bool)
    a=prepared.image
    _,med,std=sigma_clipped_stats(a[np.isfinite(a)],sigma=3.,maxiters=5)
    if not np.isfinite(std) or std<=0:std=float(np.nanstd(a)) or 1.
    z=(a-med)/std
    high,_=ndimage.label(z>=2.5);low,_=ndimage.label(z>=1.25)
    touched={component_at(high,float(r['x']),float(r['y']),search_radius=12) for r in stars}
    touched.discard(0)
    if not touched:return np.zeros(a.shape,bool)
    mask=np.isin(high,list(touched))
    comps,n=ndimage.label(bright);areas=np.bincount(comps.ravel())
    big=areas>=1000;big[0]=False
    ids=np.unique(high[mask&big[comps]&(high>0)])
    if len(ids):mask &= np.isin(high,ids)
    seed=ndimage.binary_dilation(mask,iterations=3)
    ids=np.unique(low[seed&(low>0)])
    if len(ids):mask |= np.isin(low,ids)
    for radius,op in [(4,ndimage.binary_closing),(8,ndimage.binary_dilation)]:
        y,x=np.ogrid[-radius:radius+1,-radius:radius+1]
        mask=op(mask,structure=x*x+y*y<=radius*radius)
    return mask


def classify_reference(reference, header, catalog, gaia, background, band, *, training_background=None):
    w=WCS(header).celestial
    x,y=w.all_world2pix(catalog['ALPHA_J2000'],catalog['DELTA_J2000'],0)
    ny,nx=reference.shape
    idx=np.flatnonzero(np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<nx)&(y>=0)&(y<ny))
    table=catalog[idx]
    a=np.asarray(table['KRON_RADIUS']*table['A_IMAGE'],float)
    b=np.asarray(table['KRON_RADIUS']*table['B_IMAGE'],float)
    # Existing detection-catalog axes are in the 0.03" F444W reference frame.
    geom=EllipseGeometry(x[idx],y[idx],a,b,np.deg2rad(np.asarray(table['THETA_IMAGE'],float)),np.pi*a*b)
    flux=np.asarray(table['FLUX_AUTO'],float);err=np.asarray(table['FLUXERR_AUTO'],float)
    mag=np.full(len(idx),np.nan);good=np.isfinite(flux)&(flux>0);mag[good]=27-2.5*np.log10(flux[good])
    snr=np.asarray(table['SNR_WIN'],float).copy()
    fallback=(~np.isfinite(snr)|(snr<=0))&np.isfinite(err)&(err>0)
    snr[fallback]=flux[fallback]/err[fallback]
    data=OrdinaryInput(geom,mag,snr=snr,flags=np.asarray(table['FLAGS']))
    prepared=prepare_image(reference,header=header)
    bright,components=build_bright_components(prepared,config=BrightRegionConfig(threshold=5,clip_threshold=5,statistics_clip_sigma=5))
    if 'ref_epoch' not in gaia.colnames:
        gaia=gaia.copy();gaia['ref_epoch']=2016.
    gaia_rows=project_gaia_rows(gaia,image_shape=reference.shape,image_header=header,pixel_origin=0)
    stars=star_footprint(prepared,bright,gaia_rows)
    basics=classify_catalog_basics(geom,mag,dataset='a2744',pixel_scale_arcsec=.03,valid_mask=prepared.finite_mask)
    data.center_invalid=basics.nan_center_ignore
    linear=prepared.image.copy();linear[~prepared.finite_mask]=np.nan
    ordinary=classify_ordinary_sources(data,basics.after_b_basic,basics.labels,dataset='a2744',
        config=A2744OrdinaryConfig(PSF_FWHM_ARCSEC[band]/.03,image_shape=reference.shape),
        image=linear,excluded_mask=bright.astype(bool)|stars,star_footprint=stars,sky_mask=background)
    result=label_bright_sources(data,ordinary.labels,image_shape=reference.shape,
        bright_region=bright,component_labels=components,gaia_table=gaia,image_header=header,
        source_ids=np.asarray(table['NUMBER']),
        config=JWSTBrightConfig('a2744',.03,gaia_reference_epoch=2016.))
    restricted=np.isin(components,result.restricted_fallback_component_ids) if len(result.restricted_fallback_component_ids) else None
    ignored=np.isin(components,result.ordinary_ignore_component_ids) if len(result.ordinary_ignore_component_ids) else None
    dense=fill_dense_regions(table,result.labels,reference.shape,geometry=geom,
        background_mask=background if training_background is None else training_background,quality_ignore_mask=~prepared.finite_mask,
        bright_region_mask=bright,restricted_fallback_mask=restricted,ordinary_ignore_mask=ignored)
    dense[~prepared.finite_mask]=int(DenseLabel.STRICT_IGNORE)
    labels=PatchLabels(table,dense,result.labels.source_class.copy(),geom.x,geom.y,geom.major,geom.minor,
        geom.theta,np.asarray(table['NUMBER'],np.int64),result.strict_center_x,result.strict_center_y,result.strict_center_source_id,
        strict_is_gaia=np.array(['gaia' in str(reason) for reason in result.strict_center_reason],bool))
    from .truncated_kron import large_kron_geometry
    labels.truncation_geometry=large_kron_geometry(catalog['NUMBER'],x,y,
        np.asarray(catalog['KRON_RADIUS']*catalog['A_IMAGE'],float),
        np.asarray(catalog['KRON_RADIUS']*catalog['B_IMAGE'],float),
        np.deg2rad(np.asarray(catalog['THETA_IMAGE'],float)))
    return labels,dict(sources=len(table),classes={str(k):int(v) for k,v in zip(*np.unique(labels.label_classes,return_counts=True))})
