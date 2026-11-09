"""All-parent BLUE compact detections: >=3 of F090/F115/F150/F200 at SNR>=4.

The validated pilot method is retained: progressive SExtractor seeds on raw
and model/pedestal-subtracted references; confirmation fits ORIGINAL FULL
with a pixel-integrated Gaussian and quadratic sky. Models affect seeds and
nucleus vetoes only. No training image or Zarr is written by this module.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,gc,hashlib,json,os,subprocess,tempfile,time,multiprocessing as mp
import numpy as np
from astropy.io import fits
from astropy.table import Table,vstack
from astropy.wcs import WCS
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.special import erf
from .uncover_labels import project_product,model_for_band
from .sky_tiles import reproject_tile,tile_header,SkyTile
from ..abell_cutouts import configure_progress_warnings,progress
from ..dataset_inputs import discover_abell,header_info,image_hdu,load_paths

BANDS=('F090W','F115W','F150W','F200W')
RISK_BANDS=BANDS+('F277W','F356W')
CONFIG=dict(version='all_parent_blue4_three4_raw_quad_v2',detection_bands=list(BANDS),risk_bands=list(RISK_BANDS),
    minimum_bands=3,snr_min=4.,thresholds=[10.,5.,3.,2.,1.2,1.],minarea=3,
    deblend_nthresh=64,deblend_mincont=1e-5,merge_pixels=2.,background_mesh=128,
    fit_radius=8,fit_sigma_max=4.3,fit_chi2_max=5.,fit_shift_max=2.5,
    center_scatter_max=1.,seed_isoarea_max=300,seed_moment_max=4.5,
    nucleus_distance_min=6.,neighbor_distance_min=3.,risk_model_sigma=3.,pixel_scale=.03)

# This exact predecessor rejected EVERY overflow before publishing a parent.
# Recovery only changes a path that never produced a completed old parent;
# its successful, hash-verified catalogs are scientifically unchanged.
COMPATIBLE_CODE_HASHES={'8c99f17c2f88481158f408670dacbf2dd15dc085bc791aafc83e2ea3286727d8'}


def compatible_signature(previous,current):
    if previous==current:return True
    return previous.get('code') in COMPATIBLE_CODE_HASHES and (
        {k:v for k,v in previous.items() if k!='code'}=={k:v for k,v in current.items() if k!='code'})


class SExtractorDeblendOverflow(RuntimeError):
    """A connected parent could not be deblended; retry spatially."""


def _mad(v):
    return float(1.4826*np.median(np.abs(v-np.median(v))))


def mesh(image,exclude,step=128):
    """Same source-masked median/MAD mesh as the approved detection pilot."""
    h,w=image.shape;ny,nx=(h+step-1)//step,(w+step-1)//step
    bg=np.full((ny,nx),np.nan,np.float32);rms=bg.copy()
    for y in range(ny):
        for x in range(nx):
            sl=np.s_[y*step:min((y+1)*step,h),x*step:min((x+1)*step,w)]
            a=image[sl];v=a[np.isfinite(a)&~exclude[sl]]
            if len(v)<128:v=a[np.isfinite(a)]
            if len(v)<128:continue
            for _ in range(4):
                m=float(np.median(v));s=_mad(v)
                if not s>0:break
                cut=v[np.abs(v-m)<3*s]
                if len(cut)<64 or len(cut)==len(v):break
                v=cut
            bg[y,x]=np.median(v);rms[y,x]=_mad(v)
    for a in (bg,rms):
        good=np.isfinite(a)&((a>0) if a is rms else True)
        if not good.any():raise ValueError('No usable background/noise meshes')
        nearest=ndi.distance_transform_edt(~good,return_distances=False,return_indices=True)
        a[~good]=a[tuple(nearest[:,~good])];a[:]=ndi.median_filter(a,size=3,mode='nearest')
    coords=np.broadcast_arrays(((np.arange(h)+.5)/step-.5)[:,None],((np.arange(w)+.5)/step-.5)[None,:])
    b=ndi.map_coordinates(bg,coords,order=1,mode='nearest').astype(np.float32)
    r=ndi.map_coordinates(rms,coords,order=1,mode='nearest').astype(np.float32)
    return b,np.maximum(r,.5*float(np.median(rms)))


def gaussian_model_jacobian(p,xx,yy,bg):
    """Analytic derivatives avoid ten finite-difference evaluations per fit."""
    amp,dx,dy,s=p[:4];root=np.sqrt(2)*s
    ux,lx=xx+.5-dx,xx-.5-dx;uy,ly=yy+.5-dy,yy-.5-dy
    exu,exl=np.exp(-.5*(ux/s)**2),np.exp(-.5*(lx/s)**2)
    eyu,eyl=np.exp(-.5*(uy/s)**2),np.exp(-.5*(ly/s)**2)
    ix=.5*(erf(ux/root)-erf(lx/root));iy=.5*(erf(uy/root)-erf(ly/root))
    norm=np.sqrt(2*np.pi)
    gx=(exl-exu)/(norm*s);gy=(eyl-eyu)/(norm*s)
    sx=(lx*exl-ux*exu)/(norm*s*s);sy=(ly*eyl-uy*eyu)/(norm*s*s)
    core=ix*iy;model=amp*core+bg@p[4:]
    jac=np.column_stack((core,amp*gx*iy,amp*ix*gy,amp*(sx*iy+ix*sy),bg))
    return model,jac


def fit_blue_core(image,rms,x0,y0,neighbors=()):
    radius=CONFIG['fit_radius'];xc,yc=int(np.floor(x0+.5)),int(np.floor(y0+.5))
    if xc-radius<0 or yc-radius<0 or xc+radius>=image.shape[1] or yc+radius>=image.shape[0]:return None
    sl=np.s_[yc-radius:yc+radius+1,xc-radius:xc+radius+1]
    im=np.asarray(image[sl],float);err=np.asarray(rms[sl],float)
    yy,xx=np.mgrid[yc-radius:yc+radius+1,xc-radius:xc+radius+1];xx=xx-x0;yy=yy-y0
    good=np.isfinite(im)&np.isfinite(err)&(err>0)
    for nx,ny in neighbors:
        if np.hypot(nx-x0,ny-y0)>3:good&=((xx+x0-nx)**2+(yy+y0-ny)**2>4)
    if good.sum()<100:return None
    scale=float(np.median(err[good]));im=im/scale;err=err/scale
    qx,qy=xx/radius,yy/radius
    bg=np.stack((np.ones_like(xx),qx,qy,qx*qx,qx*qy,qy*qy),axis=-1)
    outside=good&((xx*xx+yy*yy)>36)
    if outside.sum()<20:return None
    coeff=np.linalg.lstsq(bg[outside]/err[outside,None],im[outside]/err[outside],rcond=None)[0]
    peak=max(float(im[radius,radius]-(bg@coeff)[radius,radius]),1.)
    xxg,yyg,bgg=xx[good],yy[good],bg[good];im=im[good];err=err[good]
    def fun(p):return (gaussian_model_jacobian(p,xxg,yyg,bgg)[0]-im)/err
    def jac(p):return gaussian_model_jacobian(p,xxg,yyg,bgg)[1]/err[:,None]
    try:
        result=least_squares(fun,np.r_[peak*2*np.pi*1.2**2,0,0,1.2,coeff],jac=jac,
            bounds=(np.r_[0,-2,-2,.45,[-np.inf]*6],np.r_[np.inf,2,2,4.5,[np.inf]*6]),
            loss='soft_l1',max_nfev=70)
        p=result.x;chi2=float(np.sum(fun(p)**2)/max(1,len(im)-len(p)))
        covariance=np.linalg.pinv(result.jac.T@result.jac)*max(1.,chi2)
        return dict(x=x0+p[1],y=y0+p[2],snr=float(p[0]/max(np.sqrt(max(covariance[0,0],0)),1e-6)),
            sigma=float(p[3]),chi2=chi2,pos_err=max(.15,float(np.sqrt(max(covariance[1,1]+covariance[2,2],0)/2))),
            shift=float(np.hypot(p[1],p[2])),success=bool(result.success))
    except (ValueError,np.linalg.LinAlgError):return None


def good_detection(fit):
    return bool(fit is not None and fit['success'] and fit['snr']>=CONFIG['snr_min']
        and fit['sigma']<CONFIG['fit_sigma_max'] and fit['chi2']<CONFIG['fit_chi2_max']
        and fit['shift']<CONFIG['fit_shift_max'])


def _sextractor(image_path,folder,threshold,*,memory=None):
    folder.mkdir(parents=True,exist_ok=True)
    conf=folder/'default.sex';params=folder/'params.txt';conv=folder/'gauss.conv';cat=folder/'catalog.txt'
    conf.write_text(subprocess.check_output(['/usr/bin/source-extractor','-d'],text=True))
    params.write_text('\n'.join(['X_IMAGE','Y_IMAGE','XWIN_IMAGE','YWIN_IMAGE','FLUX_MAX','ISOAREA_IMAGE','A_IMAGE','B_IMAGE','FLAGS'])+'\n')
    yy,xx=np.mgrid[-3:4,-3:4];kernel=np.exp(-(xx*xx+yy*yy)/(2*(2/2.35482)**2))
    conv.write_text('CONV NORM\n'+'\n'.join(' '.join(f'{v:.7g}' for v in row) for row in kernel)+'\n')
    opts=dict(CATALOG_NAME=cat,CATALOG_TYPE='ASCII_HEAD',PARAMETERS_NAME=params,DETECT_TYPE='CCD',
        THRESH_TYPE='ABSOLUTE',DETECT_THRESH=threshold,ANALYSIS_THRESH=threshold,DETECT_MINAREA=3,
        FILTER='Y',FILTER_NAME=conv,DEBLEND_NTHRESH=64,DEBLEND_MINCONT=1e-5,CLEAN='N',
        BACK_TYPE='MANUAL',BACK_VALUE=0,BACKPHOTO_TYPE='GLOBAL',WEIGHT_TYPE='NONE',GAIN=0,
        PIXEL_SCALE=.03,SEEING_FWHM=.15,SATUR_LEVEL=1e20,PHOT_APERTURES=5,CHECKIMAGE_TYPE='NONE',
        MEMORY_PIXSTACK=32000000,MEMORY_OBJSTACK=60000,MEMORY_BUFSIZE=1024,VERBOSE_TYPE='NORMAL')
    if memory:opts.update(memory)
    cmd=['/usr/bin/source-extractor',str(image_path),'-c',str(conf)]
    for key,value in opts.items():cmd.extend(['-'+key,str(value)])
    with (folder/'sextractor.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=log,check=True)
    # SExtractor can return zero after a stack overflow and silently merge or
    # truncate detections. Such a catalog must not become a completed checkpoint.
    messages=(folder/'sextractor.log').read_text(errors='replace').lower()
    other_messages=messages.replace('deblending overflow','deblending failure')
    if any(s in other_messages for s in ('overflow','not enough memory','memory allocation failed')):
        raise RuntimeError(f'SExtractor memory/stack failure; raise --sex-memory-* and resume: {folder / "sextractor.log"}')
    if 'deblending overflow' in messages:
        raise SExtractorDeblendOverflow(f'SExtractor deblending overflow; spatial recovery required: {folder / "sextractor.log"}')
    if not any(s.strip() and not s.startswith('#') for s in cat.read_text().splitlines()):return Table()
    table=Table.read(cat,format='ascii.sextractor')
    flags=np.asarray(table['FLAGS'],int)
    if np.any(flags&128):raise RuntimeError(f'SExtractor extraction stack failure: {folder / "sextractor.log"}')
    if np.any(flags&64):raise SExtractorDeblendOverflow(f'SExtractor FLAGS contains deblending overflow: {folder / "sextractor.log"}')
    return table


def tiled_sextractor(image_path,folder,threshold,*,memory=None,core_size=1024,halo=64,min_core=128):
    """Recover only an overflowing threshold; fit confirmation stays full-frame.

    Every core owns its centers; the halo supplies detection context. Recursive
    splitting prevents an enormous parent from exhausting the deblending tree.
    """
    folder.mkdir(parents=True,exist_ok=True);tables=[];audit=[]
    with fits.open(image_path,memmap=True) as hd, tempfile.TemporaryDirectory(prefix='blue_sex_tiles_') as td:
        image=hd[0].data;header=hd[0].header;h,w=image.shape
        def visit(x0,y0,x1,y1):
            xa,ya=max(0,x0-halo),max(0,y0-halo);xb,yb=min(w,x1+halo),min(h,y1+halo)
            tile=folder/f'x{x0}_y{y0}_w{x1-x0}_h{y1-y0}'
            path=Path(td)/'tile.fits';th=header.copy();th['CRPIX1']=th.get('CRPIX1',0)-xa;th['CRPIX2']=th.get('CRPIX2',0)-ya
            fits.writeto(path,np.asarray(image[ya:yb,xa:xb],np.float32),th,overwrite=True)
            try:table=_sextractor(path,tile,threshold,memory=memory)
            except SExtractorDeblendOverflow:
                audit.append(dict(core=[x0,y0,x1,y1],status='split'))
                if max(x1-x0,y1-y0)<=min_core:raise
                xm,ym=(x0+x1)//2,(y0+y1)//2
                xs=[x0,xm,x1] if x1-x0>min_core else [x0,x1]
                ys=[y0,ym,y1] if y1-y0>min_core else [y0,y1]
                progress(f'[blue-detect] tiled_SExtractor SPLIT core={x0},{y0},{x1},{y1}')
                for ylo,yhi in zip(ys[:-1],ys[1:]):
                    for xlo,xhi in zip(xs[:-1],xs[1:]):visit(xlo,ylo,xhi,yhi)
                return
            if len(table):
                for name in ('X_IMAGE','XWIN_IMAGE'):table[name]=np.asarray(table[name],float)+xa
                for name in ('Y_IMAGE','YWIN_IMAGE'):table[name]=np.asarray(table[name],float)+ya
                x=np.asarray(table['XWIN_IMAGE'],float)-1;y=np.asarray(table['YWIN_IMAGE'],float)-1
                bad=~np.isfinite(x+y);x[bad]=np.asarray(table['X_IMAGE'])[bad]-1;y[bad]=np.asarray(table['Y_IMAGE'])[bad]-1
                table=table[(x>=x0)&(x<x1)&(y>=y0)&(y<y1)]
                if len(table):tables.append(table)
            audit.append(dict(core=[x0,y0,x1,y1],status='complete',owned_detections=len(table)))
        for y0 in range(0,h,core_size):
            for x0 in range(0,w,core_size):visit(x0,y0,min(w,x0+core_size),min(h,y0+core_size))
    result=vstack(tables,metadata_conflicts='silent') if tables else Table()
    (folder/'recovery.json').write_text(json.dumps(dict(threshold=threshold,core_size=core_size,halo=halo,
        min_core=min_core,tiles=audit,detections=len(result)),indent=2)+'\n')
    return result


def seed_catalog(source,folder,threshold,*,memory=None,input_sha='',legacy_resume=False):
    """Reuse proven successful thresholds and retry only deblending failures."""
    receipt=folder/'catalog_complete.json';cached=folder/'seed_catalog.fits'
    signature=dict(input_sha256=input_sha,threshold=float(threshold),memory=memory,
        method='full_or_deblend_tiled1024_halo64_min128_v1')
    if receipt.exists() and cached.exists():
        doc=json.loads(receipt.read_text())
        if doc['signature']==signature and doc['sha256']==hashlib.sha256(cached.read_bytes()).hexdigest():
            progress(f'[blue-detect] CACHE SExtractor step={folder} detections={doc["detections"]}')
            return Table.read(cached)
    log=folder/'sextractor.log';cat=folder/'catalog.txt';mode='full';known_deblend=False
    if legacy_resume and log.exists() and cat.exists():
        message=log.read_text(errors='replace').lower()
        known_deblend='deblending overflow' in message
        if 'all done' in message and not any(s in message for s in ('overflow','not enough memory','memory allocation failed')):
            table=Table.read(cat,format='ascii.sextractor') if any(l.strip() and not l.startswith('#') for l in cat.read_text().splitlines()) else Table()
            if len(table) and np.any(np.asarray(table['FLAGS'],int)&(64|128)):table=None
            mode='validated_predecessor_step'
            if table is not None:progress(f'[blue-detect] CACHE validated predecessor SExtractor step={folder} detections={len(table)}')
        else:table=None
    else:table=None
    if table is None:
        try:
            if known_deblend:raise SExtractorDeblendOverflow('validated predecessor already failed this exact threshold')
            table=_sextractor(source,folder,threshold,memory=memory)
        except SExtractorDeblendOverflow:
            progress(f'[blue-detect] RECOVER deblending step={folder} threshold={threshold:g} core=1024 halo=64')
            table=tiled_sextractor(source,folder/'tiled',threshold,memory=memory);mode='tiled_recovery'
    # A FITS binary table requires columns, including for an empty detection.
    if not table.colnames:
        table=Table({k:np.empty(0,int if k in ('FLAGS','ISOAREA_IMAGE') else float) for k in
            ('X_IMAGE','Y_IMAGE','XWIN_IMAGE','YWIN_IMAGE','FLUX_MAX','ISOAREA_IMAGE','A_IMAGE','B_IMAGE','FLAGS')})
    table.write(cached,overwrite=True)
    receipt.write_text(json.dumps(dict(signature=signature,sha256=hashlib.sha256(cached.read_bytes()).hexdigest(),
        detections=len(table),mode=mode),indent=2)+'\n')
    return table


def progressive_seeds(raw,clean,valid,risk,header,scratch,logs,*,parent='',memory=None,legacy_resume=False):
    records=[];cells={};radius=2.
    for branch,image in [('clean',clean),('raw',raw)]:
        source=scratch/(branch+'.fits');fits.writeto(source,np.where(valid,image,0).astype(np.float32),header,overwrite=True)
        input_sha=hashlib.sha256(source.read_bytes()).hexdigest()
        for threshold in CONFIG['thresholds']:
            started=time.monotonic()
            progress(f'[blue-detect] parent={parent} stage=SExtractor START branch={branch} threshold={threshold:g}')
            table=seed_catalog(source,logs/branch/f't{threshold:g}',threshold,memory=memory,
                input_sha=input_sha,legacy_resume=legacy_resume);added=0
            for row in sorted(table,key=lambda r:-float(r['FLUX_MAX'])):
                x,y=float(row['XWIN_IMAGE'])-1,float(row['YWIN_IMAGE'])-1
                if not np.isfinite(x+y):x,y=float(row['X_IMAGE'])-1,float(row['Y_IMAGE'])-1
                if not np.isfinite(x+y):continue
                ix,iy=int(np.floor(x+.5)),int(np.floor(y+.5))
                if not (0<=ix<valid.shape[1] and 0<=iy<valid.shape[0] and valid[iy,ix] and risk[iy,ix]):continue
                cell=(int(x//radius),int(y//radius))
                near=[v for dx in (-1,0,1) for dy in (-1,0,1) for v in cells.get((cell[0]+dx,cell[1]+dy),[])]
                if any(np.hypot(x-p[0],y-p[1])<radius for p in near):continue
                cells.setdefault(cell,[]).append((x,y));added+=1
                records.append(dict(x_init=x,y_init=y,seed_origin=branch,threshold=threshold,
                    isoarea=int(row['ISOAREA_IMAGE']),a_image=float(row['A_IMAGE']),b_image=float(row['B_IMAGE'])))
            progress(f'[blue-detect] parent={parent} stage=SExtractor END branch={branch} threshold={threshold:g} '
                f'detections={len(table)} new={added} seconds={time.monotonic()-started:.1f}')
    return records


def empty_candidates():
    cols=dict(debug_id=np.empty(0,np.int64),x=np.empty(0),y=np.empty(0),ra=np.empty(0),dec=np.empty(0),
        blue_candidate=np.empty(0,bool),n_good_bands=np.empty(0,np.int16),selection_score=np.empty(0))
    for band in BANDS:
        for key in ('x','y','snr','sigma','chi2','pos_err','shift'):cols[band+'_'+key]=np.empty(0)
        cols[band+'_detected']=np.empty(0,bool)
    return Table(cols)


def _load_full(source,reference_root,parent,header):
    cached=Path(reference_root)/source.band/(parent+'_full.fits')
    if cached.exists():
        image,h=fits.getdata(cached,header=True)
        xy=np.array([[0.,0.],[2048.,2048.],[4095.,4095.]])
        mapped=np.array(WCS(h).celestial.world_to_pixel_values(*WCS(header).celestial.pixel_to_world_values(xy[:,0],xy[:,1]))).T
        if image.shape==(4096,4096) and np.max(np.abs(mapped-xy))<1e-3:return np.asarray(image,np.float32)
        raise ValueError(f'Existing FULL cutout WCS mismatch: {cached}')
    h,_,_=header_info(source.reference_fits)
    with fits.open(source.reference_fits,memmap=True) as hd:
        return reproject_tile(hd[image_hdu(hd)].data,WCS(h).celestial,header)


def run_parent(args,job):
    configure_progress_warnings();started=time.monotonic();parent=job['name'];dest=Path(args.output_root)/parent;dest.mkdir(parents=True,exist_ok=True)
    progress(f'[blue-detect] START parent={parent}')
    sources={s.band:s for s in discover_abell(load_paths(args.paths_config)['abell'],bands=RISK_BANDS)}
    grid,_,_=header_info(job['grid_reference']);native,_,_=header_info(sources['F150W'].reference_fits)
    header=tile_header(grid,SkyTile(**job['tile']),native);shape=(4096,4096)
    models={b:model_for_band(args.model_root,b) for b in RISK_BANDS}
    memory=dict(MEMORY_PIXSTACK=args.sex_memory_pixstack,MEMORY_OBJSTACK=args.sex_memory_objstack,
        MEMORY_BUFSIZE=args.sex_memory_bufsize)
    signature=dict(config=CONFIG,job=job,sextractor_memory=memory,
        code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        assets=[dict(path=str(p),size=p.stat().st_size,mtime=p.stat().st_mtime_ns)
                for p in [Path(args.detection),Path(args.segmentation),*[Path(s.reference_fits) for s in sources.values()],*models.values()]])
    receipt=dest/'complete.json';result=dest/'verified_centers.fits'
    if receipt.exists() and result.exists() and not args.overwrite:
        doc=json.loads(receipt.read_text())
        if compatible_signature(doc.get('signature',{}),signature) and doc.get('catalog_sha256')==hashlib.sha256(result.read_bytes()).hexdigest():
            progress(f'[blue-detect] CACHE parent={parent} candidates={doc["candidates"]} code={doc["signature"]["code"]}');return doc
    # Legacy thresholds have no per-step digest yet. Adopt them only when a
    # completed peer from the exact overflow-rejecting predecessor proves that
    # the common models, images, configuration and memory settings are unchanged.
    legacy_resume=False
    if not args.overwrite and getattr(args,'prior_reference_root',None)==str(args.reference_root):
        common=lambda s:{k:v for k,v in s.items() if k not in ('code','job')}
        for path in sorted(Path(args.output_root).glob('*/complete.json')):
            prior=json.loads(path.read_text())
            if prior.get('signature',{}).get('code') in COMPATIBLE_CODE_HASHES and common(prior['signature'])==common(signature):
                legacy_resume=True;break
    progress(f'[blue-detect] parent={parent} successful_step_reuse={legacy_resume}')
    coverage,_=project_product(args.detection,header,shape,mode='coverage')
    if coverage.any():
        seg,_=project_product(args.segmentation,header,shape,mode='labels');exclude=ndi.binary_dilation(seg>0,iterations=2)
        rawdata={};rmsdata={};sumw=np.zeros(shape,np.float32);numraw=sumw.copy();numclean=sumw.copy()
        modelscore=sumw.copy();validbands=np.zeros(shape,np.uint8);model150=None
        for band in RISK_BANDS:
            progress(f'[blue-detect] parent={parent} band={band} stage=raw_model_pedestal_noise')
            raw=_load_full(sources[band],args.reference_root,parent,header)
            model,_=project_product(models[band],header,shape,mode='surface')
            if band=='F150W':model150=model
            usable=np.isfinite(raw)&np.isfinite(model)&coverage
            if usable.sum()<128:
                if band in BANDS:rawdata[band]=raw;rmsdata[band]=np.full(shape,np.nan,np.float32)
                continue
            braw,_=mesh(raw,exclude|~usable);residual=raw-model;bres,rres=mesh(residual,exclude|~usable)
            filled=np.where(np.isfinite(raw),raw,braw);_,rhp=mesh(filled-ndi.gaussian_filter(filled,3.),exclude|~usable)
            rms=np.maximum(rres,rhp);weight=np.where(usable,1/np.maximum(rms,1e-9)**2,0).astype(np.float32)
            if band in BANDS:
                numraw+=np.where(usable,raw-braw,0)*weight;numclean+=np.where(usable,residual-bres,0)*weight
                sumw+=weight;validbands+=usable.astype(np.uint8);rawdata[band]=raw;rmsdata[band]=rms
            # The approved pilot used these red models to define risk too;
            # they supply no red-only seeds or confirmation measurements.
            modelscore=np.maximum(modelscore,np.where(usable,model/rms,0))
            del residual,filled,braw,bres,rres,rhp,weight;gc.collect()
        valid=ndi.binary_erosion((validbands>=3)&coverage,iterations=8,border_value=0)
        risk=ndi.binary_dilation(modelscore>3,iterations=8)&valid
        progress(f'[blue-detect] parent={parent} stage=reference_ready valid_pixels={int(valid.sum())} risk_pixels={int(risk.sum())}')
        if risk.any():
            score_raw=np.where(valid,numraw/np.sqrt(np.maximum(sumw,1e-20)),0)
            score_clean=np.where(valid,numclean/np.sqrt(np.maximum(sumw,1e-20)),0)
            with tempfile.TemporaryDirectory(prefix='blue_scores_') as td:
                seeds=progressive_seeds(score_raw,score_clean,valid,risk,header,Path(td),dest/'sextractor',parent=parent,
                    memory=memory,legacy_resume=legacy_resume)
            progress(f'[blue-detect] parent={parent} stage=raw_FULL_confirmation START seeds={len(seeds)} fit_workers={args.fit_workers}')
            table=refine_seeds(seeds,rawdata,rmsdata,model150,header,parent,fit_workers=args.fit_workers)
        else:table=empty_candidates()
    else:table=empty_candidates()
    table.meta.update(DETBANDS=','.join(BANDS),MINBANDS=3,BLUECUT=4.,SNRREF='RAW_FULL_QUADRATIC',VERSION=CONFIG['version'])
    table.write(dest/'all_candidates.fits',overwrite=True)
    verified=table[np.asarray(table['blue_candidate'],bool)];verified.write(result,overwrite=True)
    lines=['# Region file format: DS9 version 4.1','image']
    for row in verified:lines.append(f'point({row["x"]+1:.5f},{row["y"]+1:.5f}) # point=cross color=magenta text={{B{int(row["debug_id"])}}}')
    (dest/'verified_centers.reg').write_text('\n'.join(lines)+'\n')
    doc=dict(parent=parent,signature=signature,candidates=len(verified),seeds=len(table),
        status='complete',
        detected_by_band={b:int(np.asarray(verified[b+'_detected']).sum()) for b in BANDS},
        seconds=time.monotonic()-started,catalog_sha256=hashlib.sha256(result.read_bytes()).hexdigest())
    receipt.write_text(json.dumps(doc,indent=2)+'\n')
    progress(f'[blue-detect] END parent={parent} candidates={len(verified)} seconds={doc["seconds"]:.1f}')
    return doc


_FIT_CONTEXT={}


def _fit_seed(index):
    c=_FIT_CONTEXT;seed=c['seeds'][index];x,y=c['positions'][index]
    near=c['tree'].query_ball_point([x,y],10);others=[c['positions'][j] for j in near if j!=index]
    compact=seed['isoarea']<=300 and np.sqrt(seed['a_image']*seed['b_image'])<=4.5
    return {b:fit_blue_core(c['raw'][b],c['rms'][b],x,y,others) if compact else None for b in BANDS}


def refine_seeds(seeds,raw,rms,model,header,parent,*,fit_workers=1):
    global _FIT_CONTEXT
    if not seeds:return empty_candidates()
    positions=np.array([[r['x_init'],r['y_init']] for r in seeds]);tree=cKDTree(positions)
    sm=ndi.gaussian_filter(np.nan_to_num(model),6) if model is not None else np.zeros(raw['F150W'].shape,np.float32)
    finite=rms['F150W'][np.isfinite(rms['F150W'])];noise=float(np.median(finite)) if len(finite) else np.inf
    cy,cx=np.where((sm==ndi.maximum_filter(sm,size=65))&(sm>20*noise));cores=cKDTree(np.c_[cx,cy]) if len(cx) else None
    rows=[];w=WCS(header).celestial
    _FIT_CONTEXT=dict(seeds=seeds,positions=positions,tree=tree,raw=raw,rms=rms)
    pool=mp.get_context('fork').Pool(fit_workers) if fit_workers>1 else None
    fitted_rows=pool.imap(_fit_seed,range(len(seeds)),chunksize=32) if pool else map(_fit_seed,range(len(seeds)))
    for i,(seed,fitted) in enumerate(zip(seeds,fitted_rows)):
        x0,y0=positions[i];near=tree.query_ball_point([x0,y0],10);others=[positions[j] for j in near if j!=i]
        compact=seed['isoarea']<=300 and np.sqrt(seed['a_image']*seed['b_image'])<=4.5
        good={b:r for b,r in fitted.items() if good_detection(r)}
        if good:
            xy=np.array([[r['x'],r['y']] for r in good.values()]);weights=np.array([1/max(r['pos_err'],.2)**2 for r in good.values()])
            x,y=np.average(xy,axis=0,weights=weights);scatter=float(np.hypot(xy[:,0]-x,xy[:,1]-y).max())
        else:x,y=x0,y0;scatter=np.nan
        core_distance=float(cores.query([x,y])[0]) if cores is not None else np.inf
        neighbor_distance=min([np.hypot(x-p[0],y-p[1]) for p in others],default=np.inf)
        accepted=len(good)>=3 and compact and scatter<=1 and core_distance>6 and neighbor_distance>3
        ra,dec=w.pixel_to_world_values(x,y)
        strength=sorted((r['snr'] for r in good.values()),reverse=True)
        row=dict(**seed,debug_id=i+1,x=x,y=y,ra=float(ra),dec=float(dec),blue_candidate=accepted,n_good_bands=len(good),
            center_scatter_pix=scatter,core_distance_pix=core_distance,neighbor_distance_pix=neighbor_distance,
            selection_score=strength[2] if len(strength)>=3 else -np.inf)
        for b,result in fitted.items():
            row[b+'_detected']=b in good
            row[b+'_fit_success']=bool(result and result['success'])
            for key in ('x','y','snr','sigma','chi2','pos_err','shift'):row[b+'_'+key]=result[key] if result else np.nan
        rows.append(row)
        if (i+1)%250==0:progress(f'[blue-detect] parent={parent} confirmed={i+1}/{len(seeds)}')
    if pool:pool.close();pool.join()
    _FIT_CONTEXT={}
    table=Table(rows=rows);keep=[]
    for i in np.argsort(-np.asarray(table['selection_score'])):
        if not table['blue_candidate'][i]:continue
        if any(np.hypot(table['x'][i]-table['x'][j],table['y'][i]-table['y'][j])<=2 for j in keep):table['blue_candidate'][i]=False
        else:keep.append(i)
    return table


def combine(output,docs):
    tables=[];fingerprints=[]
    for doc in sorted(docs,key=lambda d:d['parent']):
        path=output/doc['parent']/'verified_centers.fits';t=Table.read(path)
        fingerprints.append(doc['catalog_sha256'])
        if not len(t):continue
        t.rename_column('debug_id','origin_debug_id');t['origin_parent']=np.full(len(t),doc['parent'],dtype='U16');tables.append(t)
    receipt=output/'combined.json';dest=output/'verified_blue_centers.fits'
    signature=hashlib.sha256(json.dumps([CONFIG,fingerprints],sort_keys=True).encode()).hexdigest()
    if receipt.exists() and dest.exists():
        old=json.loads(receipt.read_text())
        if old.get('signature')==signature and old['sha256']==hashlib.sha256(dest.read_bytes()).hexdigest():return old
    if tables:
        t=vstack(tables,metadata_conflicts='silent');w=WCS(naxis=2)
        w.wcs.crpix=[1,1];w.wcs.crval=[float(np.mean(t['ra'])),float(np.mean(t['dec']))]
        w.wcs.cdelt=[-.03/3600,.03/3600];w.wcs.ctype=['RA---TAN','DEC--TAN']
        x,y=w.world_to_pixel_values(t['ra'],t['dec']);xy=np.c_[x,y];tree=cKDTree(xy);used=np.zeros(len(t),bool);keep=[]
        order=np.lexsort((np.asarray(t['origin_debug_id']),np.asarray(t['origin_parent']),-np.asarray(t['selection_score'])))
        for i in order:
            if used[i]:continue
            keep.append(i);used[tree.query_ball_point(xy[i],2.)]=True
        t=t[keep];t=t[np.lexsort((np.asarray(t['ra']),np.asarray(t['dec'])))];t['debug_id']=np.arange(1,len(t)+1,dtype=np.int64)
    else:t=empty_candidates()
    t.meta.update(DETBANDS=','.join(BANDS),MINBANDS=3,BLUECUT=4.,SNRREF='RAW_FULL_QUADRATIC',VERSION=CONFIG['version'])
    t.write(dest,overwrite=True)
    doc=dict(signature=signature,parents=len(docs),unique_candidates=len(t),catalog=str(dest),
        config=CONFIG,sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    receipt.write_text(json.dumps(doc,indent=2)+'\n');return doc


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--background-root',default='/data/czh23/analysis/2026-10/2026-10-19/background_sextractor15_half')
    parser.add_argument('--reference-root',default='/data/czh23/analysis/2026-11/2026-11-09/training_zarr_space_uncover_blue_manhattan/abell/references')
    parser.add_argument('--output-root',required=True);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--fit-workers',type=int,default=1,help='Forked fitting workers per parent; images are shared read-only')
    parser.add_argument('--sex-memory-pixstack',type=int,default=32000000,help='SExtractor pixel stack entries (not bytes)')
    parser.add_argument('--sex-memory-objstack',type=int,default=60000,help='SExtractor object stack entries (<=65535 for the installed binary)')
    parser.add_argument('--sex-memory-bufsize',type=int,default=1024,help='SExtractor image buffer rows')
    parser.add_argument('--paths-config',default=None);parser.add_argument('--parents',nargs='*')
    parser.add_argument('--detection',default='/data/czh23/JWST/UNCOVER/UNCOVER_DR3_DETECTION.fits')
    parser.add_argument('--segmentation',default='/data/czh23/JWST/UNCOVER/UNCOVER_DR3_SEGMAP.fits')
    parser.add_argument('--model-root',default='/data/czh23/JWST/UNCOVER/models');parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--plan-only',action='store_true');args=parser.parse_args()
    if args.workers<1 or args.fit_workers<1:parser.error('workers and fit-workers must be positive')
    if min(args.sex_memory_pixstack,args.sex_memory_objstack,args.sex_memory_bufsize)<1:
        parser.error('SExtractor stack/buffer sizes must be positive; zero does not mean unlimited')
    if args.sex_memory_objstack>65535:parser.error('Installed SExtractor requires object stack <=65535')
    configure_progress_warnings();jobs={}
    for path in sorted((Path(args.background_root)/'abell').glob('*/*/complete.json')):
        doc=json.loads(path.read_text());job=doc['job']
        if doc.get('status')=='done' and (not args.parents or job['name'] in args.parents):jobs.setdefault(job['name'],job)
    if not jobs:raise ValueError('No Abell grid parents')
    output=Path(args.output_root);output.mkdir(parents=True,exist_ok=True)
    prior_plan=output/'plan.json'
    args.prior_reference_root=json.loads(prior_plan.read_text()).get('args',{}).get('reference_root') if prior_plan.exists() else None
    (output/'plan.json').write_text(json.dumps(dict(parents=sorted(jobs),config=CONFIG,args=vars(args)),indent=2)+'\n')
    progress(f'[blue-detect] PLAN unique_parents={len(jobs)} workers={args.workers} fit_workers={args.fit_workers} '
        f'sex_pixstack={args.sex_memory_pixstack} sex_objstack={args.sex_memory_objstack} sex_bufsize={args.sex_memory_bufsize}')
    if args.plan_only:return
    # Validate the real installed binary before preparing expensive references.
    # Its accepted parameter ranges need not match a newer SExtractor manual.
    progress('[blue-detect] START real_SExtractor_preflight')
    with tempfile.TemporaryDirectory(prefix='blue_sex_preflight_') as td:
        scratch=Path(td);source=scratch/'blank.fits'
        fits.writeto(source,np.zeros((64,64),np.float32))
        _sextractor(source,scratch/'sex',10,memory=dict(MEMORY_PIXSTACK=args.sex_memory_pixstack,
            MEMORY_OBJSTACK=args.sex_memory_objstack,MEMORY_BUFSIZE=args.sex_memory_bufsize))
    progress('[blue-detect] END real_SExtractor_preflight status=passed')
    docs=[];failures=[];started=time.monotonic()
    (output/'summary.json').write_text(json.dumps(dict(results=[],failures=[],total_parents=len(jobs),
        status='running',seconds=0),indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=args.workers,max_tasks_per_child=1) as pool:
        futures={pool.submit(run_parent,args,job):name for name,job in jobs.items()}
        for future in as_completed(futures):
            try:docs.append(future.result())
            except Exception as exc:
                import traceback;traceback.print_exc();failures.append(dict(parent=futures[future],error=repr(exc)))
            progress(f'[blue-detect] PROGRESS finished={len(docs)+len(failures)}/{len(jobs)} failed={len(failures)} '
                f'candidates_before_sky_dedup={sum(d["candidates"] for d in docs)} seconds={time.monotonic()-started:.1f}')
            summary=dict(results=docs,failures=failures,total_parents=len(jobs),
                status='running',seconds=time.monotonic()-started)
            temp=output/'summary.json.tmp';temp.write_text(json.dumps(summary,indent=2)+'\n');temp.replace(output/'summary.json')
    summary.update(status='failed' if failures else 'complete',seconds=time.monotonic()-started)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if failures:raise RuntimeError(f'BLUE detection failed for {len(failures)} parents; resume after resolving failures')
    progress('[blue-detect] COMPLETE '+json.dumps(combine(output,docs)))


if __name__=='__main__':main()
