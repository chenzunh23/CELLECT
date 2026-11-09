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


@lru_cache(maxsize=12)
def _compact_template(band, pixel_scale, blur_sigma_pixels):
    """Fixed library PSF; optional predeclared broadening, never fit per source."""
    from scipy.ndimage import gaussian_filter
    image,item=template(band)
    if blur_sigma_pixels:
        image=gaussian_filter(image,blur_sigma_pixels*pixel_scale/item['pixel_scale_arcsec'])
    return image,item


def pixel_integrated_psf(band, dx, dy, pixel_scale=.03, *, angle=0.,
                         subsamples=6, blur_sigma_pixels=0.):
    """Integrate OVERDIST fine-grid flux onto the requested output pixels.

    Normalize by the complete library stamp, never the fitting cutout. This
    approximate coadd PRF does not include a field/visit-dependent drizzle
    kernel; template mismatch and correlated noise require separate checks.
    Existing Gaia geometry/wing sampling is deliberately unchanged.
    """
    if pixel_scale<=0 or subsamples<1 or blur_sigma_pixels<0:
        raise ValueError('Invalid PSF integration parameters')
    image,item=_compact_template(band,float(pixel_scale),float(blur_sigma_pixels))
    dx,dy=np.broadcast_arrays(np.asarray(dx,float),np.asarray(dy,float))
    ratio=pixel_scale/item['pixel_scale_arcsec'];c,s=np.cos(angle),np.sin(angle)
    result=np.zeros(dx.shape,float)
    for oy in (np.arange(subsamples)+.5)/subsamples-.5:
        for ox in (np.arange(subsamples)+.5)/subsamples-.5:
            x,y=(dx+ox)*ratio,(dy+oy)*ratio
            result+=map_coordinates(image,[(-x*s+y*c)+item['center_y'],
                (x*c+y*s)+item['center_x']],order=1,mode='constant',cval=0.,prefilter=False)
    return result*(ratio/subsamples)**2


def measure_fixed_psf_core(image, rms, x, y, band, *, neighbor_xy=None,
                           pixel_scale=.03, radius=12, angle=0.,
                           blur_sigma_pixels=0., background_order=2,
                           max_neighbors=8, noise_factor=1., return_models=False,
                           neighbor_margin_pixels=0.):
    """Signed forced PRF amplitude + simultaneous local sky/nearby PRFs.

    Center, angle, width and neighbor positions are FIXED. No peak search or
    positivity constraint. Covariance includes nuisance amplitudes and sky.
    The reported SNR inflates formal errors for residual chi2 and an optional
    blank-position noise factor (never reduces noise for chi2<1).
    """
    if background_order not in (1,2) or radius<4 or noise_factor<1:
        raise ValueError('Invalid compact-core fit configuration')
    image=np.asarray(image);rms=np.asarray(rms)
    if image.shape!=rms.shape or image.ndim!=2:raise ValueError('Image/RMS mismatch')
    fail=dict(success=False,snr=np.nan,snr_formal=np.nan,flux=np.nan,flux_error=np.nan,
        chi2=np.nan,neighbor_count=0,fit_pixels=0,template_energy=np.nan,status='invalid_patch')
    if not np.isfinite([x,y]).all():return fail
    cx,cy=int(np.floor(x+.5)),int(np.floor(y+.5));h,w=image.shape
    xa,xb=cx-radius,cx+radius+1;ya,yb=cy-radius,cy+radius+1
    if xa<0 or ya<0 or xb>w or yb>h:return {**fail,'status':'edge'}
    yy,xx=np.mgrid[ya:yb,xa:xb];dx,dy=xx-x,yy-y
    data=np.asarray(image[ya:yb,xa:xb],float);err=np.asarray(rms[ya:yb,xa:xb],float)
    good=np.isfinite(data)&np.isfinite(err)&(err>0)
    if good.sum()<.9*good.size:return {**fail,'status':'incomplete_patch'}
    sample=lambda nx,ny:pixel_integrated_psf(band,xx-nx,yy-ny,pixel_scale,
        angle=angle,blur_sigma_pixels=blur_sigma_pixels)
    core=sample(x,y);columns=[core];neighbors=[]
    if neighbor_xy is not None:
        xy=np.asarray(neighbor_xy,float).reshape(-1,2)
        distance=np.hypot(xy[:,0]-x,xy[:,1]-y)
        idx=np.flatnonzero(np.isfinite(distance)&(distance>=1.)&(distance<=radius+neighbor_margin_pixels))
        for i in idx[np.argsort(distance[idx])][:max_neighbors]:
            columns.append(sample(*xy[i]));neighbors.append(xy[i])
    ncore=len(columns);qx,qy=dx/radius,dy/radius
    columns.extend([np.ones_like(dx),qx,qy])
    if background_order==2:columns.extend([qx*qx,qx*qy,qy*qy])
    design=np.stack(columns,axis=-1);a=design[good]/err[good,None];b=data[good]/err[good]
    coef,_,rank,_=np.linalg.lstsq(a,b,rcond=None)
    if rank!=len(columns):return {**fail,'status':'rank_deficient'}
    covariance=np.linalg.pinv(a.T@a)
    error=float(np.sqrt(max(0.,covariance[0,0])))
    if not np.isfinite(error) or error<=0:return {**fail,'status':'invalid_covariance'}
    model=design@coef;residual=data-model
    chi2=float(np.sum((residual[good]/err[good])**2)/max(1,good.sum()-len(columns)))
    factor=max(1.,np.sqrt(chi2))*noise_factor
    result=dict(success=True,snr=float(coef[0]/error/factor),snr_formal=float(coef[0]/error),
        flux=float(coef[0]),flux_error=error*factor,chi2=chi2,neighbor_count=ncore-1,
        fit_pixels=int(good.sum()),template_energy=float(core.sum()),status='ok',
        noise_factor=float(noise_factor),residual_error_factor=float(max(1.,np.sqrt(chi2))))
    if return_models:
        result.update(data=data,core_model=coef[0]*core,
            background_model=design[...,ncore:]@coef[ncore:],
            neighbor_model=design[...,1:ncore]@coef[1:ncore],residual=residual,
            bounds=[xa,xb,ya,yb],neighbors=np.asarray(neighbors).reshape(-1,2))
    return result


def empirical_core_rms(image, sky_mask, *, box=128):
    """Raw FULL noise budget, never inferred from model-subtracted flux.

    Robust source-masked mesh MAD; insufficient meshes fall back to finite
    samples conservatively. Interpolate only the noise map, not science data.
    """
    from scipy import ndimage
    from utils.source_snr import clipped_median_std
    arr=np.asarray(image,np.float32);sky=np.asarray(sky_mask,bool)&np.isfinite(arr)
    if sky.shape!=arr.shape:raise ValueError('PSF sky/image mismatch')
    h,w=arr.shape;ny,nx=(h+box-1)//box,(w+box-1)//box
    def mesh(values):
        noise=np.full((ny,nx),np.nan,np.float32)
        for iy in range(ny):
            for ix in range(nx):
                sl=np.s_[iy*box:min((iy+1)*box,h),ix*box:min((ix+1)*box,w)]
                local=values[sl];sample=local[sky[sl]]
                if len(sample)<128:sample=local[np.isfinite(local)]
                if len(sample)<32:continue
                _,sigma,_=clipped_median_std(sample,rounds=4,sigma=3.)
                if np.isfinite(sigma) and sigma>0:noise[iy,ix]=sigma
        good=np.isfinite(noise)&(noise>0)
        if not good.any():raise ValueError('No usable raw FULL PSF noise samples')
        nearest=ndimage.distance_transform_edt(~good,return_distances=False,return_indices=True)
        noise[~good]=noise[tuple(nearest[:,~good])]
        noise=ndimage.median_filter(noise,size=3,mode='nearest')
        yy=(np.arange(h,dtype=np.float32)+.5)/box-.5
        xx=(np.arange(w,dtype=np.float32)+.5)/box-.5
        out=ndimage.map_coordinates(noise,np.broadcast_arrays(yy[:,None],xx[None,:]),order=1,mode='nearest')
        return np.maximum(out,float(np.median(noise))*.5)
    if not np.isfinite(arr).any():raise ValueError('No finite PSF reference')
    filled=np.where(np.isfinite(arr),arr,np.nanmedian(arr))
    return np.maximum(mesh(arr),mesh(filled-ndimage.gaussian_filter(filled,3.))).astype(np.float32)


def calibrate_core_noise(image, rms, band, sky_mask, *, samples=100, seed=196):
    """Empirical blank-position amplitude scatter; never decrease errors."""
    from scipy import ndimage
    valid=np.asarray(sky_mask,bool)&np.isfinite(image)
    distance=ndimage.distance_transform_edt(valid)
    h,w=valid.shape;rng=np.random.default_rng(seed);values=[]
    if h<27 or w<27:return 1.,dict(blank_samples=0,status='window_too_small')
    for _ in range(10000):
        x=float(rng.uniform(13,w-13));y=float(rng.uniform(13,h-13))
        if distance[int(y),int(x)]<9:continue
        fit=measure_fixed_psf_core(image,rms,x,y,band)
        if fit['success']:values.append(fit['snr_formal'])
        if len(values)>=samples:break
    if len(values)<20:return 1.,dict(blank_samples=len(values),status='insufficient_blank_controls')
    v=np.asarray(values);median=float(np.median(v));sigma=float(1.4826*np.median(np.abs(v-median)))
    return max(1.,sigma),dict(blank_samples=len(v),blank_snr_median=median,blank_snr_mad=sigma,status='ok')


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


def measure_fixed_psf_catalog(image, rms, band, candidates, *, noise_factor=1., progress=None,selection=None):
    """Force the current-band PSF at BLUE positions; never search red peaks.

    Neighbors and fit configuration match the validated compact-center pilot.
    The caller reuses the same RMS and blank-control inflation for native
    source review, so the two branches have a single current-band noise scale.
    """
    from scipy.spatial import cKDTree
    from astropy.table import Table
    xy=np.c_[np.asarray(candidates['x'],float),np.asarray(candidates['y'],float)]
    selected=np.asarray(candidates['blue_candidate'],bool)&np.isfinite(xy).all(axis=1)
    neighbors=xy[selected];tree=cKDTree(neighbors) if len(neighbors) else None
    if selection is not None:selected&=np.asarray(selection,bool)
    rows=[];snr=np.full(len(candidates),np.nan)
    h,w=image.shape
    for i in np.flatnonzero(selected):
        x,y=xy[i]
        if not (0<=x<w and 0<=y<h):continue
        near=tree.query_ball_point([x,y],12.)
        fit=measure_fixed_psf_core(image,rms,x,y,band,neighbor_xy=neighbors[near],noise_factor=noise_factor)
        snr[i]=fit['snr']
        rows.append(dict(debug_id=int(candidates['debug_id'][i]),x=x,y=y,**fit))
        if progress and len(rows)%100==0:progress('blue_fixed_psf',len(rows))
    if progress:progress('blue_fixed_psf_complete',len(rows))
    return snr,Table(rows=rows) if rows else Table(names=['debug_id','x','y','snr'],dtype=[np.int64,float,float,float])


def star_policy():
    config,meta=star_asset()
    return {**meta, **config['policy'], 'psf_extension':config['psf_extension'],
            'geometry_normalization':config['normalization']}


def known_nonstellar_ids():
    """Explicit user-reviewed exceptions; IDs stay lossless int64 integers."""
    return {int(sid) for sid in star_asset()[0]['policy'].get('nonstellar_gaia_ids',[])}


def assess_point_source(image, center, band, pixel_scale, *, config_override=None, return_fit_grid=False):
    """Classify an astrometrically uncertain Gaia object, without resizing labels.

    Compare a rotation-aligned PSF (with a small resampling tolerance) to the
    same PSF broadened by an intrinsic Gaussian. All fits share position,
    nonnegative amplitude and a local background plane. Robust residuals
    reduce contamination by neighbors. Broadening is a diagnostic only.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.optimize import least_squares
    from .psf import PSF_FWHM_ARCSEC
    cfg={**star_asset()[0]['policy']['psf_fallback'], **(config_override or {})}
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
    # A compact galaxy's nucleus can correlate extremely well with a PSF.
    # Do not let the FWHM-scaled resampling allowance grow without a cap.
    tolerance=min(tolerance,cfg.get('unresolved_sigma_cap_pixels',np.inf))
    # Resolve both sides of the cutoff: a half-pixel gap in the extended
    # grid can make an intermediate-width galaxy prefer the "point" bin.
    widths=np.unique(np.r_[np.arange(0.,2.01,.25),tolerance,tolerance+.5,3.,4.,6.,8.])
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
    result=dict(accepted=bool(good),status='extended' if extended else 'point_source' if good else 'ambiguous_psf',
        band=band,center_x=float(fitted_center[0]),center_y=float(fitted_center[1]),
        delta_x=float(fitted_center[0]-center[0]),delta_y=float(fitted_center[1]-center[1]),
        center_offset_arcsec=offset*pixel_scale,amplitude_snr=snr,correlation=correlation,
        unresolved_sigma_limit_pixels=float(tolerance),point_sigma_pixels=point['width'],
        extended_sigma_pixels=broad['width'],extension_score=float(extension_score),
        core_finite_fraction=fraction,rotation=rotation,fit_pixels=int(len(observed)))
    if return_fit_grid:
        result['width_fits']=[dict(sigma_pixels=r['width'],cost=r['cost']) for r in fits]
    return result
