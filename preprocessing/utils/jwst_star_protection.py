"""Final common Gaia PSF-prior / star-wing policy for COSMOS and Abell.

No spatial PSF fit, no scale fit, no per-star binary mask sidecars. The only
fitted shape parameter is rotation. Original catalog rows remain auditable;
their matching stellar row is dropped and replaced by one Gaia-ID weak shape.
"""
from pathlib import Path
import re
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from .geometry import paint_ellipse
from .jwst_star_psf import star_asset, star_policy, fit_angle, assess_point_source
from ..labels import SourceClass as C


def bright_star_footprint(prepared, bright, gaia_rows):
    """Existing Gaia-seeded high/low footprint, now RP<=18 for both families."""
    from .geometry import component_at
    stars=[r for r in gaia_rows if np.isfinite(r.get('phot_rp_mean_mag',np.nan)) and r['phot_rp_mean_mag']<=18.]
    if not stars:return np.zeros(bright.shape,bool)
    image=np.asarray(prepared.image if hasattr(prepared,'image') else prepared)
    finite=np.isfinite(image)
    if not finite.any():return np.zeros(bright.shape,bool)
    _,med,std=sigma_clipped_stats(image[finite],sigma=3.,maxiters=5)
    if not np.isfinite(std) or std<=0:std=float(np.nanstd(image)) or 1.
    z=(image-med)/std
    high,_=ndimage.label(finite&(z>=2.5));low,_=ndimage.label(finite&(z>=1.25))
    touched={component_at(high,float(r['x']),float(r['y']),search_radius=12) for r in stars};touched.discard(0)
    if not touched:return np.zeros(image.shape,bool)
    mask=np.isin(high,list(touched))
    # Keep every RP-selected seed. A different star on a large bright
    # component must not erase a smaller star's protection in the same image.
    seed=ndimage.binary_dilation(mask,iterations=3)
    ids=np.unique(low[seed&(low>0)])
    if len(ids):mask |= np.isin(low,ids)
    for radius,operation in [(4,ndimage.binary_closing),(8,ndimage.binary_dilation)]:
        y,x=np.ogrid[-radius:radius+1,-radius:radius+1]
        mask=operation(mask,structure=x*x+y*y<=radius*radius)
    return mask&finite


def resolve_f444_path(header, band):
    """Find the actual training F444W sibling, never silently a different survey."""
    for key in ('ORIGFILE','FILENAME'):
        value=header.get(key)
        if not value:continue
        original=Path(str(value))
        candidate=Path(str(original).replace(band,'F444W').replace(band.lower(),'f444w'))
        if candidate.is_file() and ('f444w' in str(candidate).lower()):return candidate
        # NIRCam pupil-selected bands are named, e.g., f150w2_f162m.
        # F444W uses f444w_clear on the same half-coadd field.
        name=re.sub(r'_f\d{3,4}(?:w2|w|m|n)_(?:clear|f\d{3,4}(?:w2|w|m|n))(?=_i2d)',
                    '_f444w_clear',original.name,flags=re.IGNORECASE)
        candidate=original.with_name(name)
        if candidate!=original and candidate.is_file():return candidate
    return None


def reference_stamp(path, header, center, size=128):
    """Bounded original F444W pixels sampled on the current output WCS grid."""
    yy,xx=np.indices((size,size),dtype=float);xx+=center[0]-size//2;yy+=center[1]-size//2
    world=WCS(header).celestial.pixel_to_world(xx,yy)
    with fits.open(path,memmap=True) as hd:
        index='SCI' if 'SCI' in hd else 0;hdu=hd[index]
        sx,sy=WCS(hdu.header).celestial.world_to_pixel(world)
        h,w=hdu.shape
        finite=np.isfinite(sx)&np.isfinite(sy)
        if not finite.any():return None
        x0,x1=max(0,int(np.floor(sx[finite].min()))-1),min(w,int(np.ceil(sx[finite].max()))+2)
        y0,y1=max(0,int(np.floor(sy[finite].min()))-1),min(h,int(np.ceil(sy[finite].max()))+2)
        if x0>=x1 or y0>=y1:return None
        data=np.asarray(hdu.section[y0:y1,x0:x1],float)
        # Mask interpolation support touching a raw NaN instead of repairing it.
        valid=np.isfinite(data)
        result=ndimage.map_coordinates(np.where(valid,data,0),[sy-y0,sx-x0],order=1,
                                      mode='constant',cval=0,prefilter=False)
        support=ndimage.map_coordinates(valid.astype(float),[sy-y0,sx-x0],order=1,
                                       mode='constant',cval=0,prefilter=False)
        result[support<1-1e-6]=np.nan
    return result


def ellipse_intersects(first, second):
    """Continuous ellipse intersection, including containment and tangency."""
    x,y,a,b,t=first;cx,cy,aa,bb,tt=second
    if not np.isfinite(first).all() or min(a,b)<=0:return False
    def q(xx,yy,ellipse):
        ex,ey,ea,eb,et=ellipse;dx,dy=xx-ex,yy-ey;c,s=np.cos(et),np.sin(et)
        return ((dx*c+dy*s)/ea)**2+((-dx*s+dy*c)/eb)**2
    if q(x,y,second)<=1 or q(cx,cy,first)<=1:return True
    c,s=np.cos(t),np.sin(t)
    def value(phase):
        return q(x+a*np.cos(phase)*c-b*np.sin(phase)*s,
                 y+a*np.cos(phase)*s+b*np.sin(phase)*c,second)
    phase=np.linspace(0,2*np.pi,128,endpoint=False);values=value(phase);dt=2*np.pi/128
    if np.min(values)<=1:return True
    # Refine every local minimum: highly elongated catalog ellipses can
    # otherwise hide a narrow intersection between angular samples.
    minima=np.flatnonzero((values<=np.roll(values,1))&(values<=np.roll(values,-1)))
    return any(minimize_scalar(value,bounds=(phase[i]-dt,phase[i]+dt),method='bounded').fun<=1+1e-9
               for i in minima)


def untrusted_near_stars(image, labels, footprint, sky, trusted=None):
    if not np.any(footprint):return np.zeros(len(labels.source_ids),bool)
    region=ndimage.distance_transform_edt(~footprint)<=32
    h,w=region.shape; x,y=labels.geom_x,labels.geom_y
    valid=np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<w)&(y>=0)&(y<h)
    hit=np.zeros(len(x),bool);ii=np.flatnonzero(valid)
    hit[ii]=region[np.minimum(np.floor(y[ii]+.5).astype(int),h-1),np.minimum(np.floor(x[ii]+.5).astype(int),w-1)]
    active=hit & ~np.isin(labels.label_classes,[C.DROPPED,C.ORDINARY_IGNORE,C.STRICT_IGNORE])
    if trusted is None and active.any():
        from utils.source_snr import measure_local_aperture_snr
        ii=np.flatnonzero(active)
        measured=measure_local_aperture_snr(image,labels.geom_x[ii],labels.geom_y[ii],
            major=labels.geom_major[ii],minor=labels.geom_minor[ii],sky_mask=sky,
            excluded_mask=footprint,radius=10.,box=128,min_sky_apertures=8)
        trusted=np.ones(len(x),bool);trusted[ii]=measured['trusted']
    trusted=np.ones(len(x),bool) if trusted is None else np.asarray(trusted,bool)
    if trusted.shape!=active.shape:raise ValueError('SNR trust/source length mismatch')
    return active & ~trusted


def _append_model_rows(labels, records, header):
    if not records:return
    table=Table(labels.table, masked=True, copy=True)
    id_column='NUMBER' if 'NUMBER' in table.colnames else 'id'
    table[id_column]=np.asarray(table[id_column],np.int64)
    # Source IDs are Gaia DR3 IDs, not negative or truncated int32 placeholders.
    new=table[:0].copy()
    worlds=WCS(header).celestial.pixel_to_world([r['x'] for r in records],[r['y'] for r in records])
    for j,r in enumerate(records):
        values={id_column:int(r['gaia_id'])}
        for name,val in [('ra',worlds.ra.deg[j]),('dec',worlds.dec.deg[j]),
                         ('ALPHA_J2000',worlds.ra.deg[j]),('DELTA_J2000',worlds.dec.deg[j]),
                         ('X_IMAGE',r['x']+1),('Y_IMAGE',r['y']+1)]:
            if name in new.colnames:values[name]=val
        masks={name:name not in values for name in new.colnames}
        for name in new.colnames:
            if name not in values:
                values[name]=np.zeros(new[name].shape[1:],dtype=new[name].dtype)
        new.add_row(values,mask=masks)
    labels.table=vstack([table,new],metadata_conflicts='silent')
    n=len(table); m=len(records)
    for field,key in [('geom_x','x'),('geom_y','y'),('geom_major','a'),('geom_minor','b'),('geom_theta','theta')]:
        setattr(labels,field,np.concatenate([getattr(labels,field),np.array([r[key] for r in records],float)]))
    labels.source_ids=np.concatenate([labels.source_ids,np.array([r['gaia_id'] for r in records],np.int64)])
    labels.label_classes=np.concatenate([labels.label_classes,np.full(m,int(C.WEAK_SHAPE),np.int16)])
    labels.table['gaia_psf_model']=np.r_[np.zeros(n,bool),np.ones(m,bool)]


def apply_star_protection(labels, *, image, header, gaia, band, pixel_scale,
                          footprint=None, sky_mask=None, trusted=None, valid_mask=None,
                          f444_path=None, snr_image=None):
    """Apply after all ordinary/bright/SNR source filters, before dense painting."""
    from ..bright_label_jwst import _gaia_rows,JWSTBrightConfig
    from .psf import PSF_FWHM_ARCSEC
    mode='a2744' if 'NUMBER' in labels.table.colnames else 'cosmos'
    rows=_gaia_rows(gaia,None,header,image.shape,JWSTBrightConfig(mode,pixel_scale,
                    PSF_FWHM_ARCSEC[band],gaia_mag_max=None,gaia_reference_epoch=2016.))
    flags_name='FLAGS' if mode=='a2744' else 'warn_flag'
    flags=np.ma.asarray(labels.table[flags_name],dtype=float).filled(np.nan)
    from .gaia import astrometric_evidence
    reliable={} if gaia is None else {
        int(sid):bool(ok) for sid,ok in zip(gaia['source_id'] if 'source_id' in gaia.colnames else np.arange(len(gaia)),
                                           astrometric_evidence(gaia,allow_absolute_motion=False))}
    original_class=labels.label_classes.copy()
    bad_snr=untrusted_near_stars(image if snr_image is None else snr_image,labels,
                               np.zeros(image.shape,bool) if footprint is None else footprint,
                               sky_mask,trusted=trusted)
    labels.label_classes[bad_snr]=int(C.ORDINARY_IGNORE)
    config=star_asset()[0]; policy=star_policy(); item=config['filters'][band];red=config['filters']['F444W']
    f444_path=Path(f444_path) if f444_path is not None else resolve_f444_path(header,band)
    stars=[];psf_audits=[]
    for row in rows:
        row=dict(row);row['gaia_position']=[row['x'],row['y']]
        if reliable.get(int(row['source_id']),False):
            row['selection']='astrometry';row['psf_match']=None;stars.append(row)
            continue
        matching_image=image;matching_center=(row['x'],row['y']);matching_band=band
        if band!='F444W' and f444_path is not None:
            stamp=reference_stamp(f444_path,header,matching_center)
            if stamp is not None and np.isfinite(stamp).any():
                matching_image=stamp;matching_center=(stamp.shape[1]//2,stamp.shape[0]//2);matching_band='F444W'
        assessment=assess_point_source(matching_image,matching_center,matching_band,pixel_scale)
        if matching_band!=band and assessment['status'].startswith('insufficient_'):
            assessment=assess_point_source(image,(row['x'],row['y']),band,pixel_scale)
        psf_audits.append(dict(gaia_id=int(row['source_id']),**assessment))
        if assessment['accepted']:
            row['x']+=assessment['delta_x'];row['y']+=assessment['delta_y']
            row['selection']='psf';row['psf_match']=assessment;stars.append(row)
    if not stars and not bad_snr.any():
        labels.star_protection_policy={**policy,'model_stars':0,'untrusted_ignored':0,
            'astrometry_selected':0,'psf_selected':0,
            'flagged_ignored':0,'replaced_catalog_rows':0,'reference_image':None,'psf_fallback_audit':psf_audits}
        labels.star_protection_sources=[]
        return None
    wings=np.zeros(image.shape,np.uint8);records=[];drop=np.zeros(len(flags),bool);flagged=np.zeros(len(flags),bool)
    xy=np.c_[labels.geom_x,labels.geom_y];finite=np.isfinite(xy).all(axis=1)
    idx=np.flatnonzero(finite);tree=cKDTree(xy[idx]) if len(idx) else None
    radii=np.maximum(labels.geom_major,labels.geom_minor)
    for row in stars:
        center=(row['x'],row['y'])
        current=fit_angle(image,center,band,pixel_scale)
        redfit=current if band=='F444W' else None
        reference_source='current_F444W'
        if redfit is None and f444_path is not None:
            stamp=reference_stamp(f444_path,header,center)
            if stamp is not None:
                redfit=fit_angle(stamp,(stamp.shape[1]//2,stamp.shape[0]//2),'F444W',pixel_scale)
                reference_source=str(f444_path)
        if redfit is None or (redfit.get('status')=='insufficient_finite_wings' and current.get('status')=='rotation_only'):
            redfit=current;reference_source='current_band_rotation_fallback'
        ellipse=(row['x'],row['y'],red['ee95_a_arcsec']/pixel_scale,
                 red['ee95_b_arcsec']/pixel_scale,red['theta_rad']+redfit['angle_rad'])
        paint_ellipse(wings,*ellipse,1)
        distance=np.hypot(labels.geom_x-row['x'],labels.geom_y-row['y'])
        near=(distance<=ellipse[2]+radii)&finite
        hits=[i for i in np.flatnonzero(near) if ellipse_intersects(
            (labels.geom_x[i],labels.geom_y[i],labels.geom_major[i],labels.geom_minor[i],labels.geom_theta[i]),ellipse)]
        flagged[hits] |= ~np.isfinite(flags[hits]) | (flags[hits]!=0)
        match=None
        if tree is not None:
            d,j=tree.query(center)
            if d<=config['policy']['star_match_arcsec']/pixel_scale:
                match=int(idx[j]);drop[match]=True
        records.append(dict(gaia_id=int(row['source_id']),x=row['x'],y=row['y'],
            selection=row['selection'],gaia_position=row['gaia_position'],psf_match=row['psf_match'],
            a=item['kron_a_arcsec']/pixel_scale,b=item['kron_b_arcsec']/pixel_scale,
            theta=item['theta_rad']+current['angle_rad'],rotation=current,
            f444_rotation=redfit,f444_reference=reference_source,ee95=list(ellipse),
            replaced_catalog_id=None if match is None else int(labels.source_ids[match]),
            intersecting_catalog_ids=[int(labels.source_ids[i]) for i in hits],
            rp=float(row['phot_rp_mean_mag']) if np.isfinite(row.get('phot_rp_mean_mag',np.nan)) else None,
            g=float(row['phot_g_mean_mag']) if np.isfinite(row['phot_g_mean_mag']) else None))
    # No upgrades to ordinary neighbors, and no resurrection of dropped rows.
    flagged &= labels.label_classes!=int(C.DROPPED)
    labels.label_classes[flagged]=int(C.ORDINARY_IGNORE)
    labels.label_classes[drop]=int(C.DROPPED)
    # Preserve table classes for audits and suppress duplicate external positives.
    if records and len(labels.strict_x):
        positions=np.c_[labels.strict_x,labels.strict_y]
        centers=np.array([[r['x'],r['y']] for r in records])
        close=cKDTree(centers).query(positions)[0]<=config['policy']['star_match_arcsec']/pixel_scale
        gaia_ids=np.array([r['gaia_id'] for r in records],np.int64)
        close |= np.isin(labels.strict_ids,np.r_[gaia_ids,-gaia_ids,-gaia_ids-1])
        for name in ('strict_x','strict_y','strict_ids','strict_is_gaia'):
            arr=getattr(labels,name)
            if arr is not None:setattr(labels,name,arr[~close])
    _append_model_rows(labels,records,header)
    labels.table['star_protection_class_before']=np.r_[original_class,np.full(len(records),-1)]
    labels.table['star_protection_untrusted']=np.r_[bad_snr,np.zeros(len(records),bool)]
    labels.table['star_protection_flag_veto']=np.r_[flagged,np.zeros(len(records),bool)]
    labels.table['star_protection_replaced']=np.r_[drop,np.zeros(len(records),bool)]
    labels.star_protection_policy={**policy,'model_stars':len(records),
        'astrometry_selected':sum(r['selection']=='astrometry' for r in records),
        'psf_selected':sum(r['selection']=='psf' for r in records),'psf_fallback_audit':psf_audits,
        'untrusted_ignored':int(bad_snr.sum()),'flagged_ignored':int(flagged.sum()),
        'replaced_catalog_rows':int(drop.sum()),'reference_image':None if f444_path is None else str(f444_path)}
    labels.star_protection_sources=records
    if valid_mask is not None:wings &= np.asarray(valid_mask,bool).astype(np.uint8)
    return wings.astype(bool)
