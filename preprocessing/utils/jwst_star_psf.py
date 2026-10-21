"""Fixed finite-stamp PSFs for Gaia shape priors; rotation is the only fit."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates

ASSET = Path(__file__).resolve().parents[1]/'assets/jwst_star_protection_v1.json'


@lru_cache(maxsize=1)
def star_asset():
    raw = ASSET.read_bytes()
    config = json.loads(raw)
    return config, {'asset_path':str(ASSET), 'asset_sha256':hashlib.sha256(raw).hexdigest(),
                    'definition_id':config['definition_id']}


def measure_psf(image, scale):
    """Same full-stamp moments/Kron/elliptical EE convention as October audit."""
    from photutils.aperture import EllipticalAperture
    from scipy.optimize import brentq
    p = np.asarray(image, float)
    if p.ndim != 2 or not np.isfinite(p).all() or (p<0).any() or p.sum()<=0:
        raise ValueError('PSF must be a finite nonnegative stamp')
    total = float(p.sum()); y,x = np.indices(p.shape,dtype=float)
    cx,cy = float((p*x).sum()/total),float((p*y).sum()/total)
    dx,dy = (x-cx)*scale,(y-cy)*scale
    cov = np.array([[(p*dx*dx).sum(),(p*dx*dy).sum()],[(p*dx*dy).sum(),(p*dy*dy).sum()]])/total
    values,vectors = np.linalg.eigh(cov); a,b = np.sqrt(values[::-1])
    theta = float(np.arctan2(vectors[1,-1],vectors[0,-1]))
    u,v = dx*np.cos(theta)+dy*np.sin(theta),-dx*np.sin(theta)+dy*np.cos(theta)
    rho = np.hypot(u/a,v/b); selected = rho<=6
    r1 = float((p[selected]*rho[selected]).sum()/p[selected].sum())
    def energy(r):
        mask=EllipticalAperture((cx,cy),a=r*a/scale,b=r*b/scale,theta=theta).to_mask(method='exact')
        return float(mask.multiply(p).sum())/total
    r95 = float(brentq(lambda r:energy(r)-.95,.001,float(rho.max())+1))
    return dict(center_x=cx,center_y=cy,pixel_scale_arcsec=float(scale),stamp_sum=total,
        theta_rad=theta,kron_a_arcsec=float(2.5*r1*a),kron_b_arcsec=float(2.5*r1*b),
        kron_energy_fraction=energy(2.5*r1),ee95_a_arcsec=float(r95*a),ee95_b_arcsec=float(r95*b))


@lru_cache(maxsize=4)
def template(band):
    item = star_asset()[0]['filters'][band]
    with fits.open(item['path'],memmap=False) as hd:
        image = np.asarray(hd[item['extension']].data,float)
    return image/image.sum(),item


def sample_template(band, dx, dy, pixel_scale, angle):
    image,item = template(band)
    ratio = pixel_scale/item['pixel_scale_arcsec']; c,s=np.cos(angle),np.sin(angle)
    x,y = dx*ratio,dy*ratio
    return map_coordinates(image,[(-x*s+y*c)+item['center_y'],
                                   (x*c+y*s)+item['center_x']],order=1,mode='constant',cval=0,prefilter=False)


def fit_angle(image, center, band, pixel_scale):
    """Robust finite-wing fit; no scale, ellipticity, or spatial interpolation."""
    radius = int(np.ceil(1.5/pixel_scale))+2; h,w=image.shape
    x0,x1=max(0,int(center[0])-radius),min(w,int(center[0])+radius+1)
    y0,y1=max(0,int(center[1])-radius),min(h,int(center[1])+radius+1)
    yy,xx=np.mgrid[y0:y1,x0:x1]; dx,dy=xx-center[0],yy-center[1]
    cut=np.asarray(image[y0:y1,x0:x1]); r=np.hypot(dx,dy)*pixel_scale
    valid=np.isfinite(cut)&(r>=.10)&(r<=1.5)
    if valid.sum()<64:
        return dict(angle_rad=0.,correlation=None,status='insufficient_finite_wings')
    dx,dy,data=dx[valid][::2],dy[valid][::2],cut[valid][::2].astype(float)
    sigma=max(float(1.4826*np.median(np.abs(data-np.median(data)))),1e-15)
    def evaluate(angle):
        model=sample_template(band,dx,dy,pixel_scale,np.deg2rad(angle))
        design=np.c_[model,np.ones(len(model))]; weights=np.ones(len(model))
        for _ in range(3):
            root=np.sqrt(weights);coef=np.linalg.lstsq(design*root[:,None],data*root,rcond=None)[0]
            residual=data-design@coef; width=max(sigma,float(1.4826*np.median(np.abs(residual-np.median(residual)))))
            weights=np.minimum(1.,3*width/np.maximum(np.abs(residual),1e-30))
        loss=float(np.mean(weights*residual**2)) if coef[0]>0 else float('inf')
        corr=float(np.corrcoef(model,data)[0,1]) if np.std(model)>0 and np.std(data)>0 else 0.
        return loss,corr
    coarse=np.arange(0,360,10.); scores=[evaluate(a) for a in coarse]
    best=coarse[int(np.argmin([s[0] for s in scores]))]
    angles=best+np.arange(-10.,11.);scores=[evaluate(a) for a in angles];j=int(np.argmin([s[0] for s in scores]))
    if not np.isfinite(scores[j][0]):
        return dict(angle_rad=0.,correlation=0.,status='no_positive_amplitude')
    return dict(angle_rad=float(np.deg2rad(angles[j]%360)),correlation=scores[j][1],
                status='rotation_only',fit_pixels=int(valid.sum()))


def star_policy():
    config,meta=star_asset()
    return {**meta, **config['policy'], 'psf_extension':config['psf_extension'],
            'geometry_normalization':config['normalization']}


def assess_point_source(image, center, band, pixel_scale):
    """Classify an astrometrically uncertain Gaia object, without resizing labels.

    Compare a rotation-aligned PSF (with a small resampling tolerance) to the
    same PSF broadened by an intrinsic Gaussian. All fits share position,
    nonnegative amplitude and a local background plane. Robust residuals
    reduce contamination by neighbors. Broadening is a diagnostic only.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.optimize import least_squares
    from .psf import PSF_FWHM_ARCSEC
    cfg=star_asset()[0]['policy']['psf_fallback']
    image=np.asarray(image);h,w=image.shape
    offset_max=cfg['max_offset_arcsec']/pixel_scale
    search=int(np.ceil(offset_max))+2
    x0,x1=max(0,int(center[0])-search),min(w,int(center[0])+search+1)
    y0,y1=max(0,int(center[1])-search),min(h,int(center[1])+search+1)
    cut=image[y0:y1,x0:x1];yy,xx=np.mgrid[y0:y1,x0:x1]
    valid=np.isfinite(cut)&(np.hypot(xx-center[0],yy-center[1])<=offset_max)
    def fail(status,**extra):return dict(accepted=False,status=status,band=band,**extra)
    if not valid.any():return fail('insufficient_finite_core')
    base=np.median(cut[valid]);smooth=gaussian_filter(np.where(np.isfinite(cut),cut,base).astype(float),.5)
    peak=np.where(valid,smooth,-np.inf);j,i=np.unravel_index(np.argmax(peak),peak.shape)
    seed=(float(x0+i),float(y0+j))
    fwhm=PSF_FWHM_ARCSEC[band]/pixel_scale
    radius=max(.25/pixel_scale,3*fwhm);outer=max(.75/pixel_scale,radius+4)
    pad=int(np.ceil(outer))+2
    xa,xb=max(0,int(seed[0])-pad),min(w,int(seed[0])+pad+1)
    ya,yb=max(0,int(seed[1])-pad),min(h,int(seed[1])+pad+1)
    yy,xx=np.mgrid[ya:yb,xa:xb];data=np.asarray(image[ya:yb,xa:xb],float)
    rr=np.hypot(xx-seed[0],yy-seed[1]);finite=np.isfinite(data)
    core=rr<=max(fwhm,1.5)
    fraction=float((core&finite).sum()/max(1,core.sum()))
    if fraction<cfg['min_core_finite_fraction'] or (finite&(rr<=radius)).sum()<32:
        return fail('insufficient_finite_core',core_finite_fraction=fraction)
    sky=data[finite&(rr>=radius)&(rr<=outer)]
    if len(sky)<32:return fail('insufficient_local_background')
    background=float(np.median(sky));noise=float(1.4826*np.median(np.abs(sky-background)))
    if not np.isfinite(noise) or noise<=max(abs(background)*1e-12,1e-15):
        return fail('insufficient_signal')
    select=finite&(rr<=radius)
    dx,dy=(xx-seed[0])[select],(yy-seed[1])[select]
    observed=(data[select]-background)/noise
    if np.max(observed)<cfg['min_amplitude_snr']:return fail('insufficient_signal')
    rotation=fit_angle(image,seed,band,pixel_scale)
    angle=rotation['angle_rad']
    tolerance=max(cfg['unresolved_sigma_floor_pixels'],cfg['unresolved_sigma_fwhm_fraction']*fwhm)
    widths=np.unique(np.r_[0.,.25,.5,.75,1.,tolerance,tolerance+.5,2.,3.,4.,6.,8.])
    grid_radius=int(np.ceil(radius))+int(np.ceil(4*widths.max()))+3
    gy,gx=np.mgrid[-grid_radius:grid_radius+1,-grid_radius:grid_radius+1]
    psf=sample_template(band,gx,gy,pixel_scale,angle)
    plane=np.c_[np.ones(len(dx)),dx/radius,dy/radius]
    fits=[]
    for width in widths:
        model=gaussian_filter(psf,width,mode='constant') if width else psf
        model=model/max(float(model.max()),1e-30)
        def vector(sx,sy):
            return map_coordinates(model,[dy-sy+grid_radius,dx-sx+grid_radius],
                                   order=1,mode='constant',cval=0,prefilter=False)
        design=np.c_[vector(0,0),plane]
        coef=np.linalg.lstsq(design,observed,rcond=None)[0]
        initial=np.r_[max(float(coef[0]),1e-3),coef[1:],0.,0.]
        def residual(p):return p[0]*vector(p[4],p[5])+plane@p[1:4]-observed
        fitted=least_squares(residual,initial,
            bounds=([0,-np.inf,-np.inf,-np.inf,-1.5,-1.5],[np.inf,np.inf,np.inf,np.inf,1.5,1.5]),
            loss='soft_l1',f_scale=2.,max_nfev=60,x_scale='jac')
        fits.append(dict(width=float(width),cost=float(fitted.cost),params=fitted.x,model=vector(fitted.x[4],fitted.x[5])))
    point=min((r for r in fits if r['width']<=tolerance+1e-8),key=lambda r:r['cost'])
    broad=min((r for r in fits if r['width']>tolerance+1e-8),key=lambda r:r['cost'])
    extension_score=2*(point['cost']-broad['cost'])-np.log(len(observed))
    p=point['params'];design=np.c_[point['model'],plane]
    amp_error=float(np.sqrt(max(np.linalg.pinv(design.T@design)[0,0],1e-30)))
    snr=float(p[0]/amp_error)
    signal=observed-plane@p[1:4]
    correlation=float(np.corrcoef(signal,point['model'])[0,1]) if np.std(signal)>0 else 0.
    fitted_center=(seed[0]+p[4],seed[1]+p[5])
    offset=float(np.hypot(fitted_center[0]-center[0],fitted_center[1]-center[1]))
    extended=extension_score>cfg['extension_score_threshold']
    good=not extended and snr>=cfg['min_amplitude_snr'] and correlation>=cfg['min_correlation'] and offset<=offset_max
    return dict(accepted=bool(good),status='extended' if extended else 'point_source' if good else 'ambiguous_psf',
        band=band,center_x=float(fitted_center[0]),center_y=float(fitted_center[1]),
        delta_x=float(fitted_center[0]-center[0]),delta_y=float(fitted_center[1]-center[1]),
        center_offset_arcsec=offset*pixel_scale,amplitude_snr=snr,correlation=correlation,
        unresolved_sigma_limit_pixels=float(tolerance),point_sigma_pixels=point['width'],
        extended_sigma_pixels=broad['width'],extension_score=float(extension_score),
        core_finite_fraction=fraction,rotation=rotation,fit_pixels=int(len(observed)))
