"""Abell source labeling on full-coadd references using existing v3 policies."""
import numpy as np
from astropy.wcs import WCS
from ..image_processing import prepare_image, build_bright_components, BrightRegionConfig
from ..meas_processing import classify_catalog_basics
from ..ordinary_common import OrdinaryInput, classify_ordinary_sources
from ..ordinary_a2744 import A2744OrdinaryConfig, protect_ignored_children
from ..bright_label import project_gaia_rows
from ..bright_label_jwst import JWSTBrightConfig, label_bright_sources
from ..region_filling import fill_dense_regions
from ..labels import DenseLabel, SourceLabels
from .geometry import EllipseGeometry
from .image_level import PatchLabels
from .jwst_background import PSF_FWHM_ARCSEC


def star_footprint(prepared, bright, gaia_rows):
    from .jwst_star_protection import bright_star_footprint
    return bright_star_footprint(prepared, bright, gaia_rows)


def classify_reference(reference, header, catalog, gaia, background, band, *, training_background=None,
                       bright_image=None, bright_header=None):
    """Select ordinary sources on full coadd; optionally build bright on half.

    Formal Zarr callers supply the aligned training half image for bright,
    including independent centered stamps. Reference-only diagnostics can
    still explicitly omit it to reproduce historical source classifications.
    """
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
    if bright_image is None:
        bright_prepared=prepared
    else:
        from ..image_processing import PreparedImage
        if bright_header is None:
            raise ValueError('Training bright image requires its WCS header')
        bright_prepared=bright_image if isinstance(bright_image,PreparedImage) else prepare_image(bright_image,header=bright_header)
        if bright_prepared.image.shape != reference.shape:
            raise ValueError('Training bright and reference image shapes differ')
        points=np.array([[0,0],[nx-1,0],[0,ny-1],[nx-1,ny-1],[(nx-1)/2,(ny-1)/2]])
        projected=np.column_stack(WCS(bright_header).celestial.all_world2pix(*w.all_pix2world(points[:,0],points[:,1],0),0))
        if not np.allclose(points,projected,atol=1e-3,rtol=0):
            raise ValueError('Training bright image must align with full-reference WCS')
    bright,components=build_bright_components(bright_prepared,config=BrightRegionConfig(threshold=5,clip_threshold=5,statistics_clip_sigma=5))
    if 'ref_epoch' not in gaia.colnames:
        gaia=gaia.copy();gaia['ref_epoch']=2016.
    gaia_rows=project_gaia_rows(gaia,image_shape=reference.shape,image_header=header,pixel_origin=0)
    stars=star_footprint(bright_prepared,bright,gaia_rows)
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
    # Bright-cluster rejection may introduce additional ignored catalog children.
    # Preserve center-only supervision on their surviving parents as well.
    if 'ignore_child_parents' in ordinary.diagnostics:
        from ..snr import apply_remeasured_aperture_snr
        while protect_ignored_children(data, ordinary, A2744OrdinaryConfig(PSF_FWHM_ARCSEC[band]/.03)).any():
            measurement = ordinary.diagnostics.get('aperture_measurement')
            if measurement is not None:
                apply_remeasured_aperture_snr(ordinary, data.mag, measurement, mag_min=None)
    restricted=np.isin(components,result.restricted_fallback_component_ids) if len(result.restricted_fallback_component_ids) else None
    ignored=np.isin(components,result.ordinary_ignore_component_ids) if len(result.ordinary_ignore_component_ids) else None
    labels=PatchLabels(table,np.empty(reference.shape,np.uint8),result.labels.source_class.copy(),geom.x,geom.y,geom.major,geom.minor,
        geom.theta,np.asarray(table['NUMBER'],np.int64),result.strict_center_x,result.strict_center_y,result.strict_center_source_id,
        strict_is_gaia=np.array(['gaia' in str(reason) for reason in result.strict_center_reason],bool))
    from .jwst_star_protection import apply_star_protection
    training_linear=bright_prepared.image.copy()
    training_linear[~bright_prepared.finite_mask]=np.nan
    wings=apply_star_protection(labels,image=training_linear,header=bright_header if bright_header is not None else header,
        gaia=gaia,band=band,pixel_scale=.03,footprint=stars,
        # Recheck only near-star sources, including bright rows which the
        # ordinary faint-source SNR pass did not measure. Full coadd remains
        # the measurement reference; rotations/bright footprints use HALF.
        sky_mask=background,snr_image=linear,valid_mask=bright_prepared.finite_mask)
    final_geom=EllipseGeometry(labels.geom_x,labels.geom_y,labels.geom_major,labels.geom_minor,labels.geom_theta,
                              np.pi*labels.geom_major*labels.geom_minor)
    labels.dense=fill_dense_regions(labels.table,SourceLabels(labels.label_classes,np.full(len(labels.source_ids),'',object)),
        reference.shape,geometry=final_geom,
        background_mask=background if training_background is None else training_background,
        quality_ignore_mask=~bright_prepared.finite_mask,bright_region_mask=bright,
        restricted_fallback_mask=restricted,ordinary_ignore_mask=ignored,star_wings_mask=wings)
    labels.dense[~bright_prepared.finite_mask]=int(DenseLabel.STRICT_IGNORE)
    from .truncated_kron import large_kron_geometry
    labels.truncation_geometry=large_kron_geometry(catalog['NUMBER'],x,y,
        np.asarray(catalog['KRON_RADIUS']*catalog['A_IMAGE'],float),
        np.asarray(catalog['KRON_RADIUS']*catalog['B_IMAGE'],float),
        np.deg2rad(np.asarray(catalog['THETA_IMAGE'],float)))
    return labels,dict(sources=len(table),bright_source='training_half_coadd' if bright_image is not None else 'full_reference',
                       classes={str(k):int(v) for k,v in zip(*np.unique(labels.label_classes,return_counts=True))},
                       star_protection=labels.star_protection_policy)
