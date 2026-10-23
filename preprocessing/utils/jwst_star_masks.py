"""Gaia-seeded SExtractor components for vetoing stellar-wing positives.

These masks are source-selection evidence ONLY. They are never passed to
dense painting as background, instance GT, or STAR_WINGS negative targets.
"""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import tempfile
import numpy as np
from scipy import ndimage
from .geometry import component_at
from .jwst_background_local import fit_native_background
from .jwst_background_sextractor import AggressiveSkyConfig, _detect_pass
from .jwst_star_psf import star_asset


@dataclass
class StarMask:
    origin_xy: tuple
    pixels: np.ndarray
    metadata: dict

    def intersects(self, ellipse):
        """Exact continuous ellipse versus the union of selected pixel squares.

        Testing just a catalog center, or just mask pixel centers, misses long
        Kron ellipses and subpixel contacts. Pixel squares have half-width .5.
        """
        x,y,a,b,theta=ellipse
        if not np.isfinite(ellipse).all() or min(a,b)<=0 or not self.pixels.size:return False
        c,s=np.cos(theta),np.sin(theta)
        rx,ry=np.hypot(a*c,b*s),np.hypot(a*s,b*c)
        ox,oy=self.origin_xy;h,w=self.pixels.shape
        xa,ya=max(0,int(np.ceil(x-rx-.5))-ox),max(0,int(np.ceil(y-ry-.5))-oy)
        xb,yb=min(w,int(np.floor(x+rx+.5))+1-ox),min(h,int(np.floor(y+ry+.5))+1-oy)
        if xa>=xb or ya>=yb:return False
        py,px=np.nonzero(self.pixels[ya:yb,xa:xb])
        if not len(px):return False
        dx=px+xa+ox-x;dy=py+ya+oy-y
        qxx=c*c/a**2+s*s/b**2;qyy=s*s/a**2+c*c/b**2
        qxy=c*s*(1/a**2-1/b**2)
        if np.any(qxx*dx*dx+2*qxy*dx*dy+qyy*dy*dy<=1+1e-12):return True
        xl,xh=dx-.5,dx+.5;yl,yh=dy-.5,dy+.5
        if np.any((xl<=0)&(xh>=0)&(yl<=0)&(yh>=0)):return True
        # A convex quadratic's constrained minimum is on an edge unless its
        # unconstrained minimum (the ellipse center) lies inside the square.
        for u in (xl,xh):
            v=np.clip(-qxy*u/qyy,yl,yh)
            if np.any(qxx*u*u+2*qxy*u*v+qyy*v*v<=1+1e-12):return True
        for v in (yl,yh):
            u=np.clip(-qxy*v/qxx,xl,xh)
            if np.any(qxx*u*u+2*qxy*u*v+qyy*v*v<=1+1e-12):return True
        return False


def source_intersections(mask, labels):
    """Return ALL catalog Kron intersections, including off-center contacts."""
    ox,oy=mask.origin_xy;h,w=mask.pixels.shape
    c,s=np.cos(labels.geom_theta),np.sin(labels.geom_theta)
    rx=np.hypot(labels.geom_major*c,labels.geom_minor*s)
    ry=np.hypot(labels.geom_major*s,labels.geom_minor*c)
    near=(labels.geom_x+rx>=ox-.5)&(labels.geom_x-rx<=ox+w-.5)
    near&=(labels.geom_y+ry>=oy-.5)&(labels.geom_y-ry<=oy+h-.5)
    fields=(labels.geom_x,labels.geom_y,labels.geom_major,labels.geom_minor,labels.geom_theta)
    return [int(i) for i in np.flatnonzero(near) if mask.intersects(tuple(f[i] for f in fields))]


def detection_magnitudes(labels, band):
    """Catalog magnitudes used by ordinary filtering, never Gaia G/RP.

    Abell: detection FLUX_AUTO with the same zeropoint 27 as abell_labels.
    COSMOS: current-band MAG_AUTO, matching cosmos_labels' OrdinaryInput.
    Missing/invalid flux is not evidence for either side of a magnitude cut.
    """
    table=labels.table
    def values(key):return np.ma.asarray(table[key],dtype=float).filled(np.nan)
    if 'NUMBER' in table.colnames:
        if 'FLUX_AUTO' not in table.colnames:
            raise ValueError('Abell stellar protection requires detection FLUX_AUTO')
        flux=values('FLUX_AUTO');mag=np.full(len(flux),np.nan)
        good=np.isfinite(flux)&(flux>0);mag[good]=27-2.5*np.log10(flux[good])
        return mag
    key='mag_auto_'+band.lower()
    if key not in table.colnames:raise ValueError('COSMOS stellar protection requires '+key)
    return values(key)


def source_centers_in_mask(mask, labels):
    ox,oy=mask.origin_xy;h,w=mask.pixels.shape
    x,y=np.asarray(labels.geom_x),np.asarray(labels.geom_y)
    valid=np.isfinite(x)&np.isfinite(y)
    use=np.flatnonzero(valid);hit=np.zeros(len(x),bool)
    ix=np.floor(x[use]+.5).astype(int)-ox;iy=np.floor(y[use]+.5).astype(int)-oy
    inside=(ix>=0)&(iy>=0)&(ix<w)&(iy<h)
    hit[use[inside]]=mask.pixels[iy[inside],ix[inside]]
    return hit


def neighbor_veto(labels, stars, masks, flags, magnitudes, *, faint_mag_min=None):
    """All stars: faint centers in mask OR flagged Kron contact -> ignore."""
    from ..labels import SourceClass as C
    if faint_mag_min is None:faint_mag_min=star_asset()[0]['policy']['star_mask_faint_mag_min']
    flags=np.asarray(flags,float)
    mag=np.asarray(magnitudes,float)
    if flags.shape!=labels.source_ids.shape or mag.shape!=flags.shape:
        raise ValueError('Stellar-mask flag/magnitude/source length mismatch')
    faint=np.isfinite(mag)&(mag>=faint_mag_min)
    flagged=np.zeros(len(labels.source_ids),bool);magnitude_veto=np.zeros_like(flagged);hits={}
    for star in stars:
        sid=int(star.get('source_id',star.get('gaia_id')))
        selected=source_intersections(masks[sid],labels);hits[sid]=selected
        flagged[selected] |= ~np.isfinite(flags[selected]) | (flags[selected]!=0)
        magnitude_veto |= faint & source_centers_in_mask(masks[sid],labels)
    can_ignore=~np.isin(labels.label_classes,[int(C.DROPPED),int(C.STRICT_IGNORE)])
    return flagged&can_ignore,magnitude_veto&can_ignore,hits


def masks_from_components(components, valid, stars, band, pixel_scale, stats=None):
    """Shared morphology for production detections and cached diagnostic maps."""
    config=star_asset()[0];cfg=config['policy']['star_mask']
    sigma=max(cfg['smoothing_sigma_floor_pixels'],config['filters'][band]['adaptive_sigma_arcsec']/pixel_scale)
    boxes=ndimage.find_objects(components)
    masks={};closed=cfg['closing_radius_pixels'];grow=cfg['dilation_radius_pixels']
    for star in stars:
        sid=int(star.get('source_id',star.get('gaia_id')))
        seed=component_at(components,float(star['x']),float(star['y']),search_radius=cfg['seed_search_radius_pixels'])
        meta={**(stats or {}), 'method':cfg['method'],'config':dict(cfg),
            'smoothing_sigma_pixels':float(sigma),'band':band,
            'seed_component':int(seed),'negative_supervision':False}
        if not seed:
            meta.update(status='no_seed_component',mask_pixels=0,bounds_xyxy=[0,0,0,0])
            masks[sid]=StarMask((0,0),np.zeros((0,0),bool),meta);continue
        sy,sx=boxes[seed-1];pad=2*closed+grow+1
        x0,y0=max(0,sx.start-pad),max(0,sy.start-pad)
        x1,y1=min(valid.shape[1],sx.stop+pad),min(valid.shape[0],sy.stop+pad)
        mask=components[y0:y1,x0:x1]==seed
        for radius,op in ((closed,ndimage.binary_closing),(grow,ndimage.binary_dilation)):
            if radius:
                yy,xx=np.ogrid[-radius:radius+1,-radius:radius+1]
                mask=op(mask,structure=xx*xx+yy*yy<=radius*radius)
        mask&=valid[y0:y1,x0:x1]
        meta.update(status='seeded_component',mask_pixels=int(mask.sum()),bounds_xyxy=[x0,y0,x1,y1])
        masks[sid]=StarMask((x0,y0),mask,meta)
    return masks


def build_star_masks(image, header, stars, band, pixel_scale, *, debug_dir=None):
    """One native SExtractor pass, then select every accepted Gaia star's island.

    There is NO RP/G magnitude cut here. The caller supplies only the stars
    accepted by the existing astrometry/PSF rule. Debug products are optional;
    production does not persist per-star binary masks into Zarr.
    """
    if not stars:return {}
    config=star_asset()[0];cfg=config['policy']['star_mask']
    from ..image_processing import prepare_image
    # Same original-FITS-aware repair as model/bright preparation: stellar
    # holes may seed a component; edge-connected/background NaNs stay invalid.
    prepared=prepare_image(image,header=header)
    data=prepared.image;valid=np.isfinite(data)
    sigma=max(cfg['smoothing_sigma_floor_pixels'],
              config['filters'][band]['adaptive_sigma_arcsec']/pixel_scale)
    detection=AggressiveSkyConfig(factors=(1,),smoothing_sigma_pixels=(sigma,),
        thresholds_sigma=(cfg['threshold_sigma'],),minarea_native_pixels=(cfg['minarea_pixels'],),
        grow_native_pixels=(0,))
    executable=shutil.which('source-extractor') or shutil.which('sex')
    if executable is None:raise FileNotFoundError('SExtractor is required for Gaia stellar masks')
    debug=Path(debug_dir) if debug_dir is not None else None
    if debug:debug.mkdir(parents=True,exist_ok=True)
    stats={}
    with tempfile.TemporaryDirectory(prefix='jwst-star-mask-') as temp:
        work=Path(temp);fit=work/'fit';fit.mkdir();detect=work/'detect';detect.mkdir()
        if valid.any():
            bg=fit_native_background(data,valid,SimpleNamespace(
                background_mesh=cfg['background_mesh'],background_filter_size=cfg['background_filter_size']),
                fit,executable,work/'fit.log')
            found,stats=_detect_pass(data-bg,valid,1,sigma,cfg['threshold_sigma'],
                cfg['minarea_pixels'],0,detect,executable,work/'detection.log',detection)
        else:found=np.zeros(data.shape,bool)
        if debug:
            for name in ('fit.log','detection.log'):
                if (work/name).exists():shutil.copyfile(work/name,debug/name)
    components,_=ndimage.label(found&valid,structure=np.ones((3,3),bool))
    masks=masks_from_components(components,valid,stars,band,pixel_scale,stats)
    if debug:
        np.savez_compressed(debug/'sextractor_components.npz',components=components)
        (debug/'star_masks.json').write_text(json.dumps({str(k):v.metadata for k,v in masks.items()},indent=2)+'\n')
    return masks
