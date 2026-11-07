"""Measure HSC persisted CoaddPsf EE rings in an LSST subprocess, then cache."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np

ASSET = Path(__file__).resolve().parents[1]/'assets/hsc_confidence_ee10_35_60_70_v1.json'


def measure_psf(reference, output):
    recipe = json.loads(ASSET.read_text()); reference=Path(reference); output=Path(output)
    signature=dict(path=str(reference.resolve()),mtime_ns=reference.stat().st_mtime_ns,
                   recipe_sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest(),
                   extractor_sha256=hashlib.sha256(Path(__file__).with_name('hsc_psf_extract.py').read_bytes()).hexdigest())
    if output.exists():
        saved=json.loads(output.read_text())
        if saved.get('signature')==signature:return saved
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hsc_psf_') as tmp:
        stamp=Path(tmp)/'psf.npz'
        script=Path(__file__).with_name('hsc_psf_extract.py')
        command='source "$1" >/dev/null 2>&1; setup lsst_distrib >/dev/null 2>&1; exec python "$2" "$3" "$4"'
        subprocess.run(['bash','-c',command,'hsc-psf',recipe['lsst_setup'],str(script),str(reference),str(stamp)],check=True)
        from photutils.aperture import CircularAperture
        from scipy.optimize import brentq
        with np.load(stamp) as z:
            results=[]
            positions=json.loads(str(z['positions_json']));sample_errors=json.loads(str(z['errors_json']))
            for key in sorted(k for k in z.files if k.startswith('psf')):
                raw=z[key].astype(float)
                if not np.isfinite(raw).all():raise ValueError('Nonfinite HSC PSF')
                a=np.maximum(raw,0);total=float(a.sum())
                if not total>0:raise ValueError('Empty HSC PSF')
                center=((a.shape[1]-1)/2,(a.shape[0]-1)/2)
                def flux(r):
                    mask=CircularAperture(center,r).to_mask(method='exact')
                    return float(mask.multiply(a).sum())/total
                radii=[brentq(lambda r:flux(r)-f,1e-6,min(a.shape)/2) for f in recipe['fractions']]
                results.append(dict(key=key,position_xy=positions[key],radii_pixels=radii,stamp_sum=float(raw.sum()),
                    positive_sum=total,negative_fraction=float(-raw[raw<0].sum()/total)))
            radii=np.median([r['radii_pixels'] for r in results],axis=0)
            scale=float(z['pixel_scale'])
    saved=dict(mode='psf-ee',definition_id=recipe['definition_id'],signature=signature,
        level_radii_pixels={str(k):float(r) for k,r in zip([4,3,2,1],radii)},
        level_radii_arcsec={str(k):float(r*scale) for k,r in zip([4,3,2,1],radii)},
        pixel_scale_arcsec=scale,ee_fractions={str(k):f for k,f in zip([4,3,2,1],recipe['fractions'])},
        normalization=recipe['normalization'],samples=results,sample_errors=sample_errors,psf_path=str(reference),asset_path=str(ASSET),
        asset_sha256=signature['recipe_sha256'],measurement='exact circular aperture; median of 3x3 spatial samples',
        fallback={'nearest_xy':'floor(source_xy + 0.5)'},fwhm_clip_pixels=None)
    # Distinct jobs may measure the same reference; atomic publication prevents partial JSON.
    with tempfile.NamedTemporaryFile(mode='w',dir=output.parent,delete=False) as f:
        json.dump(saved,f,indent=2);temp=Path(f.name)
    temp.replace(output)
    return saved


def star_geometry(reference, centers, output_root):
    """Local theoretical Kron/EE95, cached by reference and exact star positions."""
    from astropy.io import fits
    from .jwst_star_psf import measure_psf as measure_geometry
    reference=Path(reference);root=Path(output_root);root.mkdir(parents=True,exist_ok=True)
    with fits.open(reference,memmap=True) as hd:
        h=hd[0].header.copy();h.update(next(x.header for x in hd if x.header.get('NAXIS')==2))
    ox,oy=-float(h.get('LTV1',0)),-float(h.get('LTV2',0))
    positions={f'psf_{int(c["gaia_id"])}':[float(c['x'])+ox,float(c['y'])+oy] for c in centers}
    script=Path(__file__).with_name('hsc_psf_extract.py')
    signature=dict(reference=str(reference.resolve()),mtime_ns=reference.stat().st_mtime_ns,
        extractor=hashlib.sha256(script.read_bytes()).hexdigest(),positions=positions)
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:24]
    cached=root/f'{key}.npz';request=root/f'{key}_positions.json'
    if not cached.exists():
        request.write_text(json.dumps(positions,indent=2)+'\n')
        recipe=json.loads(ASSET.read_text())
        command='source "$1" >/dev/null 2>&1; setup lsst_distrib >/dev/null 2>&1; exec python "$2" "$3" "$4" "$5"'
        with tempfile.TemporaryDirectory(dir=root,prefix='star_psf_') as tmp:
            dest=Path(tmp)/'psf.npz'
            with (root/f'{key}.log').open('w') as log:
                subprocess.run(['bash','-c',command,'hsc-star-psf',recipe['lsst_setup'],str(script),
                    str(reference),str(dest),str(request)],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=300)
            dest.replace(cached)
    result={}
    with np.load(cached) as z:
        scale=float(z['pixel_scale'])
        for c in centers:
            name=f'psf_{int(c["gaia_id"])}'
            if name not in z:continue  # No spatial PSF support: never substitute an average.
            raw=np.asarray(z[name],float);positive=np.maximum(raw,0.)
            measured=measure_geometry(positive,scale)
            border=np.ones(raw.shape,bool);border[2:-2,2:-2]=False
            result[int(c['gaia_id'])]={**measured,'a':measured['kron_a_arcsec']/scale,
                'b':measured['kron_b_arcsec']/scale,'theta':measured['theta_rad'],
                'ee95_a':measured['ee95_a_arcsec']/scale,'ee95_b':measured['ee95_b_arcsec']/scale,
                'stamp_shape':list(raw.shape),'raw_sum':float(raw.sum()),'positive_sum':float(positive.sum()),
                'outer2_fraction':float(positive[border].sum()/positive.sum()),
                'negative_fraction':float(-raw[raw<0].sum()/positive.sum()),'psf_cache':str(cached)}
    return result


def read_sat_mask(path, shape):
    """Separate SAT from BAD/NO_DATA; unknown SAT coverage is not a clear mask."""
    from astropy.io import fits
    from .image_level import _mask_plane_bit
    with fits.open(path,memmap=False) as hd:
        index=next(i for i,h in enumerate(hd) if h.header.get('NAXIS')==2)
        mask=hd['MASK'] if 'MASK' in hd else hd[index+1] if index+1<len(hd) else None
        if mask is None or mask.header.get('NAXIS')!=2:return None
        bit=_mask_plane_bit(mask.header,'SAT')
        if bit is None:return None
        raw=np.asarray(mask.data,np.int64)
        if raw.shape!=tuple(shape):raise ValueError('HSC SAT/image shape mismatch')
        return (raw&(1<<bit))!=0


def sat_containment(sat_labels, ellipse, *, associated_ids=(), boxes=None):
    """Require every corner of every associated SAT pixel inside model EE95.

    Associated SAT islands touch EE95 or share the star's bright component.
    Entire connected islands are checked, including long bleed trails and
    disconnected SAT segments linked by the same detected bright component.
    """
    from scipy import ndimage
    cx,cy,a,b,theta=map(float,ellipse)
    if not np.isfinite([cx,cy,a,b,theta]).all() or min(a,b)<=0:raise ValueError('Invalid EE95 ellipse')
    if boxes is None:boxes=ndimage.find_objects(sat_labels)
    c,s=np.cos(theta),np.sin(theta);height,width=sat_labels.shape
    rx,ry=np.hypot(a*c,b*s),np.hypot(a*s,b*c)
    xa,xb=max(0,int(np.floor(cx-rx-1))),min(width,int(np.ceil(cx+rx+2)))
    ya,yb=max(0,int(np.floor(cy-ry-1))),min(height,int(np.ceil(cy+ry+2)))
    yy,xx=np.mgrid[ya:yb,xa:xb];dx,dy=xx-cx,yy-cy
    def q(dx,dy):return ((dx*c+dy*s)/a)**2+((-dx*s+dy*c)/b)**2
    contact=q(dx,dy)<=1
    for sx in (-.5,.5):
        for sy in (-.5,.5):contact |= q(dx+sx,dy+sy)<=1
    ids=set(map(int,associated_ids))|set(map(int,np.unique(sat_labels[ya:yb,xa:xb][contact])))
    # An extremely narrow ellipse can lie wholly within one SAT pixel.
    nx,ny=int(np.floor(cx+.5)),int(np.floor(cy+.5))
    if 0<=nx<width and 0<=ny<height:ids.add(int(sat_labels[ny,nx]))
    ids.discard(0);total=outside=0;edge=False
    for sid in ids:
        sl=boxes[sid-1]
        if sl is None:continue
        yy,xx=np.nonzero(sat_labels[sl]==sid);xx=xx+sl[1].start;yy=yy+sl[0].start
        total+=len(xx);inside=np.ones(len(xx),bool)
        for sx in (-.5,.5):
            for sy in (-.5,.5):inside &= q(xx+sx-cx,yy+sy-cy)<=1+1e-12
        outside+=int((~inside).sum())
        edge |= bool(np.any((xx==0)|(xx==width-1)|(yy==0)|(yy==height-1)))
    return dict(sat_pixels=total,sat_outside_ee95=outside,sat_component_ids=sorted(ids),
        sat_touches_image_boundary=edge,contained=outside==0 and not edge)


def apply_star_priors(labels, *, reference, image_path, components, center_components, output_root,
                      parameters=None, sat_mask=None):
    """Promote eligible Gaia centers to weak PSF shapes only after the SAT gate."""
    from scipy import ndimage
    from .jwst_star_protection import _append_model_rows
    is_gaia=np.zeros(len(labels.strict_x),bool) if labels.strict_is_gaia is None else np.asarray(labels.strict_is_gaia,bool)
    indices=np.flatnonzero(is_gaia)
    if not len(indices):return labels
    centers=[dict(gaia_id=-int(labels.strict_ids[i])-1,x=float(labels.strict_x[i]),y=float(labels.strict_y[i]),
                  component_id=int(center_components[i]),strict_index=int(i)) for i in indices]
    psf_error=None
    if parameters is None:
        try:parameters=star_geometry(reference,centers,output_root)
        except (OSError,ValueError,subprocess.SubprocessError) as exc:
            parameters={};psf_error=str(exc)
            print(f'[hsc-gaia-psf] extraction failed; keeping center-only: {exc}',flush=True)
    if sat_mask is None:sat_mask=read_sat_mask(image_path,components.shape)
    sat_labels=boxes=None;by_bright={}
    if sat_mask is not None:
        if np.shape(sat_mask)!=components.shape:raise ValueError('SAT/bright component shape mismatch')
        sat_labels,_=ndimage.label(sat_mask,structure=np.ones((3,3),bool));boxes=ndimage.find_objects(sat_labels)
        active=sat_labels>0
        if active.any():
            pairs=np.unique(np.c_[components[active],sat_labels[active]],axis=0)
            for comp,sat_id in pairs:
                if comp>0:by_bright.setdefault(int(comp),set()).add(int(sat_id))
    records=[];models=[];remove=np.zeros(len(labels.strict_x),bool)
    for c in centers:
        model=parameters.get(c['gaia_id']);record={**c,'status':'center_only_no_psf'}
        if model is not None:
            record.update(model)
            if sat_labels is None:record['status']='center_only_no_sat_mask'
            else:
                check=sat_containment(sat_labels,(c['x'],c['y'],model['ee95_a'],model['ee95_b'],model['theta']),
                    associated_ids=by_bright.get(c['component_id'],()),boxes=boxes)
                record.update(check)
                record['status']='psf_weak_shape' if check['contained'] else 'center_only_sat_outside_ee95'
                if check['contained']:
                    models.append(record);remove[c['strict_index']]=True
        records.append(record)
    if models:
        _append_model_rows(labels,models,fits_header(reference))
        if 'classification_reason' in labels.table.colnames:
            reasons=np.asarray(labels.table['classification_reason'].filled('')
                if hasattr(labels.table['classification_reason'],'filled') else labels.table['classification_reason'],object)
            reasons[-len(models):]='hsc_gaia_psf_sat_within_ee95'
            labels.table['classification_reason']=np.asarray(reasons,str)
    if remove.any():
        for name in ('strict_x','strict_y','strict_ids','strict_is_gaia'):
            value=getattr(labels,name)
            if value is not None:setattr(labels,name,value[~remove])
    labels.star_protection_sources=records
    labels.star_protection_policy=dict(definition_id='hsc_gaia_local_psf_sat_ee95_v1',
        model_stars=len(models),center_only_stars=len(records)-len(models),psf_error=psf_error,
        sat_source=str(image_path),sat_available=sat_mask is not None,
        sat_association='8-connected islands touching EE95 or sharing the detected bright component; test all pixel corners',
        fallback='center only; preserve restricted bright component',
        psf_normalization='nonnegative finite persisted CoaddPsf stamp; not total observed saturated-star halo')
    print(f'[hsc-gaia-psf] {len(models)} weak shapes; {len(records)-len(models)} center-only; '
          f'SAT from {image_path}',flush=True)
    return labels


def fits_header(path):
    from astropy.io import fits
    with fits.open(path,memmap=True) as hd:
        h=hd[0].header.copy();h.update(next(x.header for x in hd if x.header.get('NAXIS')==2))
    return h
