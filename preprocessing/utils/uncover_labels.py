"""Opt-in UNCOVER DR3 native-catalog policy for Abell full-reference labels.

Missing Krons remain NaN: they never acquire invented PSF/moment ellipses.
The existing foundation-catalog pipeline is unchanged. Angles are native-image
radians, transformed through the official detection-grid WCS.
"""
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.table import Table
from astropy.wcs import WCS
from scipy.spatial import cKDTree
from ..labels import SourceClass as C, SourceLabels, DenseLabel
from ..ordinary_common import OrdinaryInput, DatasetOrdinaryResult, retained, downgrade
from ..ordinary_a2744 import A2744OrdinaryConfig, source_containment
from ..snr import SnrThresholds, apply_candidate_snr
from .geometry import EllipseGeometry, ellipse_axes_from_moments
from .area_filter import area_filter_masks
from .no_data import center_no_data

POLICY_VERSION='uncover_native_abell_v5_raw_full_snr_model_risk_only'


def project_product(path, header, shape, *, mode='coverage'):
    """Read a bounded official image and project by WCS, never resize IDs.

    coverage: finite detection, nearest neighbor; labels: integer IDs, nearest;
    surface: physical MJy/sr model, bilinear. Gzip is inflated only once up to
    the bounded rows; no full-mosaic data array or uncompressed file is made.
    """
    from scipy.ndimage import map_coordinates
    path=Path(path);ow=WCS(header).celestial
    if mode not in ('coverage','labels','surface'):raise ValueError('Unknown UNCOVER projection mode')
    with fits.open(path,memmap=False) as hd:
        ih=next(i for i,h in enumerate(hd) if h.header.get('NAXIS')==2)
        nh=hd[ih].header.copy();nw=WCS(nh).celestial;offset=hd[ih]._data_offset
        height,width=shape
        yy,xx=np.meshgrid(np.linspace(0,height-1,5),np.linspace(0,width-1,5),indexing='ij')
        sx,sy=nw.world_to_pixel_values(*ow.pixel_to_world_values(xx,yy))
        xa=max(0,int(np.floor(np.nanmin(sx)))-2);xb=min(nh['NAXIS1'],int(np.ceil(np.nanmax(sx)))+3)
        ya=max(0,int(np.floor(np.nanmin(sy)))-2);yb=min(nh['NAXIS2'],int(np.ceil(np.nanmax(sy)))+3)
        out=np.zeros(shape,dtype=bool if mode=='coverage' else np.int32 if mode=='labels' else np.float32)
        if xa>=xb or ya>=yb:return out,nh
        if path.suffix!='.gz':cut=np.array(hd[ih].section[ya:yb,xa:xb],copy=True)
    if path.suffix=='.gz':
        bitpix=nh['BITPIX'];dtype=np.dtype(('>' if abs(bitpix)>8 else '|')+('f' if bitpix<0 else 'u' if bitpix==8 else 'i')+str(abs(bitpix)//8))
        cut=np.empty((yb-ya,xb-xa),np.float32 if mode!='labels' else np.int32)
        with gzip.open(path,'rb') as stream:
            stream.seek(offset+ya*nh['NAXIS1']*dtype.itemsize)
            for y in range(ya,yb,64):
                n=min(64,yb-y);raw=stream.read(n*nh['NAXIS1']*dtype.itemsize)
                if len(raw)!=n*nh['NAXIS1']*dtype.itemsize:raise ValueError(f'Truncated model: {path}')
                cut[y-ya:y-ya+n]=np.frombuffer(raw,dtype=dtype).reshape(n,nh['NAXIS1'])[:,xa:xb]*nh.get('BSCALE',1)+nh.get('BZERO',0)
    if mode=='coverage':cut=np.isfinite(cut)
    elif mode=='surface':
        unit=u.Unit(nh['BUNIT'])
        if unit.is_equivalent(u.Jy/u.sr):factor=unit.to(u.MJy/u.sr)
        else:
            jy=unit.to(u.Jy) if unit.is_equivalent(u.Jy) else unit.to(u.Jy/u.pix)
            area=abs(np.linalg.det(nw.pixel_scale_matrix))*(np.pi/180)**2
            factor=jy/area/1e6
        cut=np.asarray(cut,np.float32)*factor
        if not np.isfinite(cut).all():raise ValueError(f'Model contains invalid pixels inside its grid: {path}')
    for start in range(0,height,256):
        yy,xx=np.mgrid[start:min(start+256,height),:width]
        sx,sy=nw.world_to_pixel_values(*ow.pixel_to_world_values(xx,yy))
        if mode in ('coverage','labels'):
            ix=np.floor(sx+.5).astype(int)-xa;iy=np.floor(sy+.5).astype(int)-ya
            good=(ix>=0)&(iy>=0)&(ix<cut.shape[1])&(iy<cut.shape[0])
            part=out[start:start+len(yy)];part[good]=cut[iy[good],ix[good]]
        else:out[start:start+len(yy)]=map_coordinates(cut,[sy-ya,sx-xa],order=1,mode='constant',cval=0.)
    return out,nh


def coverage_on_grid(path, header, shape, *, cache_root=None):
    """Detection finite pixels, shared across bands with identical parent WCS."""
    path=Path(path)
    if cache_root is None:return project_product(path,header,shape,mode='coverage')[0]
    payload=dict(path=str(path.resolve()),mtime=path.stat().st_mtime_ns,size=path.stat().st_size,
        shape=list(shape),wcs=WCS(header).celestial.to_header().tostring())
    key=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:24]
    root=Path(cache_root);root.mkdir(parents=True,exist_ok=True);cache=root/(key+'.npz')
    if cache.exists():
        with np.load(cache) as z:return z['coverage']
    mask=project_product(path,header,shape,mode='coverage')[0]
    with tempfile.NamedTemporaryFile(dir=root,suffix='.npz',delete=False) as f:
        temp=Path(f.name);np.savez_compressed(f,coverage=mask)
    temp.replace(cache)
    return mask


def model_for_band(root, band):
    """No cross-band fallback: every SNR reference uses its own model."""
    root=Path(root).expanduser()
    candidates=list(root.glob('*bcgs_models.fits*'))+list(root.glob('models/*bcgs_models.fits*'))
    matches=sorted(p for p in candidates if p.is_file() and
        f'_{band.lower()}_bcgs_models.fits' in p.name.lower() and p.name.lower().endswith(('.fits','.fits.gz')))
    complete=[]
    for path in matches:
        if path.suffix=='.gz':complete.append(path);continue
        with path.open('rb') as stream:
            h=fits.Header.fromfile(stream)
            nbytes=int(np.prod([h[f'NAXIS{i}'] for i in range(1,h['NAXIS']+1)]))*abs(h['BITPIX'])//8
            expected=stream.tell()+((nbytes+2879)//2880)*2880
        if path.stat().st_size>=expected:complete.append(path)
    if matches and not complete:
        raise ValueError(f'Official {band} model is still incomplete/unpacking under {root}; wait for the full FITS before generation')
    matches=complete
    # Prefer an uncompressed copy if both downloads of the same model exist.
    if len(matches)==2 and {p.name.removesuffix('.gz') for p in matches}=={matches[0].name.removesuffix('.gz')}:
        matches=[p for p in matches if p.suffix!='.gz']
    if len(matches)!=1:raise FileNotFoundError(f'Expected one official {band} bCG model under {root}; found {len(matches)}. No other-band model will be used.')
    return matches[0]


def training_coverage(observed, uncover):
    """One union-of-invalid gate, never sum overlapping invalid fractions."""
    if np.shape(observed)!=np.shape(uncover):raise ValueError('UNCOVER coverage/image shape mismatch')
    return np.asarray(observed,bool)&np.asarray(uncover,bool)


def band_photometry(table, band):
    """Use the requested band's own native flux/error, not detection or F444W."""
    flux_key,error_key=f'f_{band.lower()}',f'e_{band.lower()}'
    mag=np.full(len(table),np.nan);snr=np.full(len(table),np.nan)
    if flux_key in table.colnames and error_key in table.colnames:
        f=np.asarray(np.ma.filled(table[flux_key],np.nan),float)
        e=np.asarray(np.ma.filled(table[error_key],np.nan),float)
        valid=np.isfinite(f)&(f>0)&np.isfinite(e)&(e>0)
        mag[valid]=28.9-2.5*np.log10(f[valid]);snr[valid]=f[valid]/e[valid]
    return mag,snr


@dataclass(frozen=True)
class UncoverSnrConfig:
    """Measured SNR, with an optional externally verified contamination region."""
    no_kron_min_snr: float = 3.
    contaminated_no_kron_min_snr: float = 5.
    reject_untrusted_in_contaminated: bool = True
    risk_bright_mag_max: float | None = 22.
    thresholds: SnrThresholds = SnrThresholds()

    def __post_init__(self):
        if not (np.isfinite(self.no_kron_min_snr) and np.isfinite(self.contaminated_no_kron_min_snr)
                and self.no_kron_min_snr >= 0 and self.contaminated_no_kron_min_snr >= 0):
            raise ValueError('No-Kron SNR thresholds must be finite and >= 0')
        if self.risk_bright_mag_max is not None and not np.isfinite(self.risk_bright_mag_max):
            raise ValueError('Bright risk exemption must be a finite AB magnitude or None')


def apply_uncover_measured_snr(result, measurement, has_kron, *, config=UncoverSnrConfig(),
                              contaminated=None,band_magnitude=None):
    """All retained classes obey configured ignore/weak cuts; never invent shape.

    Existing strict geometry/containment decisions remain strict. Native
    flag_lowsnr is provenance, not a second hard cut or a class cap. Untrusted
    measurements are additionally rejected inside a supplied model/region mask,
    except valid-Kron sources bright in this band. This exception does not
    bypass numerical SNR cuts, native bad flags or missing-geometry decisions.
    """
    snr=np.asarray(measurement['snr'],float);trusted=np.asarray(measurement['trusted'],bool)
    has=np.asarray(has_kron,bool)
    risk=np.zeros(snr.shape,bool) if contaminated is None else np.asarray(contaminated,bool)
    bandmag=np.full(snr.shape,np.nan) if band_magnitude is None else np.asarray(band_magnitude,float)
    if any(a.shape != result.candidate.shape for a in (snr,trusted,has,risk,bandmag)):
        raise ValueError('UNCOVER SNR/source length mismatch')
    tested=retained(result).copy()
    threshold=np.maximum(config.thresholds.ignore_below,
        np.where(risk,config.contaminated_no_kron_min_snr,config.no_kron_min_snr))
    numerical_bad=~np.isfinite(snr)|(snr<config.thresholds.ignore_below)
    no_kron_bad=(~has)&(snr<threshold)
    bright=np.zeros(snr.shape,bool) if config.risk_bright_mag_max is None else (
        has&np.isfinite(bandmag)&(bandmag<config.risk_bright_mag_max))
    untrusted_bad=risk&~trusted&config.reject_untrusted_in_contaminated&~bright
    downgrade(result,numerical_bad,C.ORDINARY_IGNORE,'uncover_measured_snr_ignore')
    downgrade(result,no_kron_bad,C.ORDINARY_IGNORE,'uncover_no_kron_snr_ignore')
    downgrade(result,untrusted_bad,C.ORDINARY_IGNORE,'uncover_model_region_snr_bad')
    weak=retained(result)&(snr>=config.thresholds.ignore_below)&(snr<config.thresholds.weak_below)
    downgrade(result,weak,C.WEAK_SHAPE,'uncover_measured_snr_weak')
    result.diagnostics.update(aperture_snr=snr,aperture_snr_trusted=trusted,
        aperture_snr_tested=tested,aperture_snr_ignore=tested&(numerical_bad|no_kron_bad|untrusted_bad),
        aperture_snr_weak=weak,snr_bad=~np.isfinite(snr)|~trusted,
        snr_contaminated_region=risk,no_kron_snr_threshold=threshold,
        snr_risk_bright_exempt=tested&risk&~trusted&bright&~numerical_bad&~no_kron_bad)
    result.snapshot('aperture_snr')


def adapt_catalog(catalog, header, shape, native_header, *, include_outside=False):
    """Return parent-local catalog + exact local native-to-output ellipse map."""
    output=WCS(header).celestial;native=WCS(native_header).celestial
    x,y=output.world_to_pixel_values(np.asarray(catalog['ra']),np.asarray(catalog['dec']))
    inside=np.isfinite(x)&np.isfinite(y)
    if not include_outside:inside&=(x>=0)&(x<shape[1])&(y>=0)&(y<shape[0])
    tab=Table(catalog[inside],copy=True);x,y=x[inside],y[inside]
    xn,yn=np.asarray(tab['x'],float),np.asarray(tab['y'],float)
    # Central differences transform both native coordinate basis vectors.
    xp,yp=output.world_to_pixel_values(*native.pixel_to_world_values(xn+.5,yn))
    xm,ym=output.world_to_pixel_values(*native.pixel_to_world_values(xn-.5,yn))
    j00,j10=xp-xm,yp-ym
    xp,yp=output.world_to_pixel_values(*native.pixel_to_world_values(xn,yn+.5))
    xm,ym=output.world_to_pixel_values(*native.pixel_to_world_values(xn,yn-.5))
    j01,j11=xp-xm,yp-ym
    t=np.asarray(tab['theta_J2000'],float);k=np.asarray(tab['kron_radius'],float)
    aa=k*np.asarray(tab['a_image'],float);bb=k*np.asarray(tab['b_image'],float)
    cs,sn=np.cos(t),np.sin(t)
    ux,uy=j00*cs+j01*sn,j10*cs+j11*sn
    vx,vy=-j00*sn+j01*cs,-j10*sn+j11*cs
    ell=ellipse_axes_from_moments(aa**2*ux**2+bb**2*vx**2,
        aa**2*uy**2+bb**2*vy**2,aa**2*ux*uy+bb**2*vx*vy)
    geom=EllipseGeometry(x,y,ell['a'],ell['b'],ell['theta_rad'],np.pi*ell['a']*ell['b'])
    has=geom.valid()&np.isfinite(k)&(k>0)
    for values in (geom.major,geom.minor,geom.theta,geom.area):values[~has]=np.nan
    # Detection total magnitude follows the old detection-FLUX_AUTO policy.
    # Native flux ZP=28.9; compatibility FLUX_AUTO uses the old ZP=27.
    flux=np.asarray(tab['fauto_f277w+f356w+f444w'],float)
    mag=np.full(len(tab),np.nan);good=np.isfinite(flux)&(flux>0)
    mag[good]=28.9-2.5*np.log10(flux[good])
    error=np.asarray(tab['eaper_f277w+f356w+f444w'],float)
    snr=np.divide(np.asarray(tab['faper_f277w+f356w+f444w'],float),error,
        out=np.full(len(tab),np.nan),where=np.isfinite(error)&(error>0))
    tab['NUMBER']=np.asarray(tab['id'],np.int64)
    tab['ALPHA_J2000']=np.asarray(tab['ra']);tab['DELTA_J2000']=np.asarray(tab['dec'])
    tab['FLUX_AUTO']=flux*10**(.4*(27.-28.9))
    # Only the requested native bad-source flags map to the existing stellar
    # neighbor veto. flag_kron/use_circle are not SExtractor flag bits.
    tab['FLAGS']=np.where((tab['flag_artifact']!=0)|(tab['flag_nearbcg']!=0),16,0).astype(np.int16)
    tab['has_native_kron']=has;tab['detection_mag']=mag;tab['catalog_aperture_snr']=snr
    return tab,OrdinaryInput(geom,mag,snr=snr,flags=np.asarray(tab['FLAGS'])),has


def classify_ordinary(data, table, has_kron, *, config, image=None, sky_mask=None,
                      excluded_mask=None, star_footprint=None, aperture_measurement=None, progress=None,
                      snr_config=UncoverSnrConfig(),snr_contaminated_mask=None,band_magnitude=None):
    """A unchanged where Kron exists; native B flags; non-ignore nesting; SNR."""
    n=len(data.mag);g=data.geometry;has=np.asarray(has_kron,bool)
    labels=SourceLabels.empty(n);labels.assign(np.ones(n,bool),C.CLEAN,'uncover_candidate')
    invalid=center_no_data(g,image=image) if image is not None else np.zeros(n,bool)
    badmag=~np.isfinite(data.mag)
    labels.assign(badmag,C.ORDINARY_IGNORE,'invalid_detection_magnitude')
    labels.assign(invalid,C.ORDINARY_IGNORE,'center_no_data')
    large,faint=area_filter_masks(g.area,data.mag,pixel_scale_arcsec=.03)
    eligible=~invalid&~badmag
    labels.assign(eligible&(large|faint),C.DROPPED,'A_area_filter')
    result=DatasetOrdinaryResult(labels,eligible&~large&~faint)
    result.diagnostics['nan_center_ignore']=invalid
    result.snapshot('A')
    missing=~has
    dropped=missing&(np.asarray(table['flag_nophot'])!=0)
    labels.assign(dropped,C.DROPPED,'uncover_nophot_without_kron')
    result.candidate[dropped]=False
    artifact=(np.asarray(table['flag_artifact'])!=0)|(np.asarray(table['flag_nearbcg'])!=0)
    downgrade(result,artifact,C.ORDINARY_IGNORE,'uncover_artifact_or_nearbcg')
    downgrade(result,missing,C.STRICT_CENTER_ONLY,'uncover_no_kron')
    result.snapshot('B_flags')
    # Same PSF-size and closer-than-one-PSF rule as ordinary_a2744.py.
    small=retained(result)&has&((g.major<config.psf_fwhm_pixels)|(g.minor<config.psf_fwhm_pixels))
    downgrade(result,small,C.ORDINARY_IGNORE,'a2744_psf_size')
    idx=np.flatnonzero(retained(result));close=np.zeros(n,bool)
    if len(idx)>1:
        tree=cKDTree(np.c_[g.x[idx],g.y[idx]])
        for i,j in sorted(tree.query_pairs(config.psf_fwhm_pixels)):
            i,j=int(idx[i]),int(idx[j])
            if close[i] or close[j]:continue
            loser=i if data.mag[i]<data.mag[j] else j
            close[loser]=True
    downgrade(result,close,C.ORDINARY_IGNORE,'a2744_close_brighter')
    result.snapshot('B_psf_close')
    # Retain exact old IoU/containment thresholds and demotions. All native
    # valid-Kron unflagged sources have equal priority. Ignored rows never enter.
    result.diagnostics['flag_ignore']=np.zeros(n,bool)
    if config.enable_containment:source_containment(data,result,config)
    result.snapshot('B_containment')
    low=np.asarray(table['flag_lowsnr'])!=0
    remeasuring=aperture_measurement is not None or (image is not None and config.enable_aperture_snr)
    if config.enable_catalog_snr and not remeasuring:
        # A catalog fallback is only used when no image/measurement exists.
        # Never pre-reject sources before an independent band remeasurement.
        apply_candidate_snr(result,data.snr,thresholds=snr_config.thresholds)
    result.diagnostics['catalog_snr_applied']=bool(config.enable_catalog_snr and not remeasuring)
    result.snapshot('SNR_catalog')
    if star_footprint is not None and config.enable_star_mask:
        from .containment import mask_sample_fraction
        overlap,hit=mask_sample_fraction(g,star_footprint)
        # Center-only sources still have real centers even without an ellipse.
        ii=np.flatnonzero(~has&np.isfinite(g.x)&np.isfinite(g.y))
        ix=np.floor(g.x[ii]+.5).astype(int);iy=np.floor(g.y[ii]+.5).astype(int)
        inside=(ix>=0)&(iy>=0)&(ix<star_footprint.shape[1])&(iy<star_footprint.shape[0])
        hit[ii[inside]]=star_footprint[iy[inside],ix[inside]]
        downgrade(result,(hit|(overlap>=config.star_overlap_min))&(data.mag>=config.star_mask_mag_min),
            C.ORDINARY_IGNORE,'a2744_star_mask')
        result.snapshot('star_mask')
    if aperture_measurement is None and image is not None and config.enable_aperture_snr:
        from utils.source_snr import measure_local_aperture_snr
        # Diagnostics must cover all catalog centers, including ignored and
        # A-dropped rows, with no magnitude/flag/SNR preselection.
        measure=np.isfinite(g.x)&np.isfinite(g.y)
        if progress:progress('remeasured_aperture_snr_all_catalog_centers',int(measure.sum()))
        # Unknown Kron axes are replaced ONLY for the aperture-completeness
        # check; the source geometry and all exported masks remain NaN.
        aperture_measurement=measure_local_aperture_snr(image,
            np.where(measure,g.x,np.nan),np.where(measure,g.y,np.nan),
            major=np.where(has,g.major,config.aperture_radius),
            minor=np.where(has,g.minor,config.aperture_radius),
            sky_mask=sky_mask,excluded_mask=excluded_mask,radius=config.aperture_radius,
            box=128,min_sky_apertures=8)
    if aperture_measurement is not None:
        risk=np.zeros(n,bool)
        if snr_contaminated_mask is not None:
            risk_image=np.asarray(snr_contaminated_mask,bool)
            if risk_image.ndim!=2 or (image is not None and risk_image.shape!=image.shape):
                raise ValueError('UNCOVER contaminated-region/image shape mismatch')
            ii=np.flatnonzero(np.isfinite(g.x)&np.isfinite(g.y))
            ix=np.floor(g.x[ii]+.5).astype(int);iy=np.floor(g.y[ii]+.5).astype(int)
            inside=(ix>=0)&(iy>=0)&(ix<risk_image.shape[1])&(iy<risk_image.shape[0])
            risk[ii[inside]]=risk_image[iy[inside],ix[inside]]
        apply_uncover_measured_snr(result,aperture_measurement,has,config=snr_config,
            contaminated=risk,band_magnitude=band_magnitude)
        result.diagnostics['aperture_measurement']=aperture_measurement
    result.snapshot('SNR_aperture')
    result.diagnostics.update(has_native_kron=has,psf_small=small,close_brighter=close,
        no_kron_nophot_dropped=dropped,flag_lowsnr=low)
    return result


def classify_reference(reference, header, catalog, gaia, background, band, *,
        native_header, training_background=None, bright_image=None, bright_header=None,
        native_segmentation=None, progress=None,snr_config=UncoverSnrConfig(),
        snr_contaminated_mask=None,aperture_measurement=None,
        snr_risk_model=None,snr_risk_sigma=3.,snr_risk_coverage=None):
    """Same full-reference/half-bright/Gaia framework, with native UNCOVER rules."""
    from ..image_processing import prepare_image,PreparedImage,build_bright_components,BrightRegionConfig
    from ..bright_label import project_gaia_rows
    from ..bright_label_jwst import JWSTBrightConfig,label_bright_sources
    from .abell_labels import star_footprint
    from .jwst_background import PSF_FWHM_ARCSEC
    from .jwst_star_psf import star_policy
    from .image_level import PatchLabels
    from .jwst_star_protection import apply_star_protection
    from ..region_filling import fill_dense_regions
    table,data,has=adapt_catalog(catalog,header,reference.shape,native_header)
    # Native UNCOVER photometry is in 10 nJy (AB ZP=28.9), after bCG/ICL
    # treatment. Use this band's total photometry, never the LW-stack mag or
    # a raw halo-contaminated aperture, for the bright-source risk exception.
    flux_key,error_key=f'f_{band.lower()}',f'e_{band.lower()}'
    bandmag,bandsnr=band_photometry(table,band)
    table['catalog_band_mag']=bandmag;table['catalog_band_snr']=bandsnr
    if progress:progress('adapted_catalog',len(table))
    prepared=prepare_image(reference,header=header)
    if bright_image is not None:
        if bright_header is None:raise ValueError('Training HALF requires its WCS header')
        if np.shape(bright_image.image if hasattr(bright_image,'image') else bright_image)!=reference.shape:
            raise ValueError('HALF and FULL image shapes differ')
        height,width=reference.shape
        points=np.array([[0,0],[width-1,0],[0,height-1],[width-1,height-1],[(width-1)/2,(height-1)/2]])
        projected=np.column_stack(WCS(bright_header).celestial.world_to_pixel_values(
            *WCS(header).celestial.pixel_to_world_values(points[:,0],points[:,1])))
        if not np.allclose(points,projected,atol=1e-3,rtol=0):raise ValueError('HALF and FULL WCS must align')
    bp=prepared if bright_image is None else (bright_image if isinstance(bright_image,PreparedImage)
        else prepare_image(bright_image,header=bright_header))
    bright,components=build_bright_components(bp,config=BrightRegionConfig(threshold=5,clip_threshold=5,statistics_clip_sigma=5))
    gaia=gaia.copy()
    if 'ref_epoch' not in gaia.colnames:gaia['ref_epoch']=2016.
    grows=project_gaia_rows(gaia,image_shape=reference.shape,image_header=header,pixel_origin=0)
    stars=star_footprint(bp,bright,grows)
    linear=prepared.image.copy();linear[~prepared.finite_mask]=np.nan
    excluded=bright.astype(bool)|stars
    model_risk_audit={}
    if snr_risk_model is not None:
        if snr_contaminated_mask is not None or aperture_measurement is not None:
            raise ValueError('Model-risk mode measures raw FULL internally; do not supply a different risk/SNR measurement')
        model=np.asarray(snr_risk_model)
        if model.shape!=linear.shape:raise ValueError('Risk model must align with FULL')
        from scipy.ndimage import binary_dilation
        from utils.source_snr import robust_sigma_value,disk_kernel
        sky=np.asarray(background,bool)&~excluded&np.isfinite(linear)
        sigma=robust_sigma_value(linear[sky])
        if not np.isfinite(sigma) or sigma<=0:raise ValueError('No usable raw FULL sky sigma for model-risk region')
        risk=binary_dilation(np.isfinite(model)&(model>snr_risk_sigma*sigma),
                             structure=disk_kernel(10.).astype(bool))
        if snr_risk_coverage is not None:
            if np.shape(snr_risk_coverage)!=linear.shape:raise ValueError('Risk coverage must align with FULL')
            risk &= np.asarray(snr_risk_coverage,bool)
        snr_contaminated_mask=risk
        model_risk_audit=dict(risk_fraction=float(risk.mean()),risk_pixels=int(risk.sum()),
            blank_sky_pixel_sigma=float(sigma),risk_sigma=float(snr_risk_sigma),
            risk_definition='current-band official model > risk_sigma * raw FULL blank-sky MAD sigma, dilated by 10 pixels, intersect coverage',
            risk_sky='raw FULL reference SExtractor sky minus HALF bright components and Gaia star footprint')
        if progress:progress('raw_FULL_SNR_model_risk_only',len(table))
    config=A2744OrdinaryConfig(PSF_FWHM_ARCSEC[band]/.03,image_shape=reference.shape,
        star_mask_mag_min=star_policy()['star_mask_faint_mag_min'])
    ordinary=classify_ordinary(data,table,has,config=config,image=linear,sky_mask=background,
        excluded_mask=excluded,star_footprint=stars,progress=progress,
        snr_config=snr_config,snr_contaminated_mask=snr_contaminated_mask,
        aperture_measurement=aperture_measurement,band_magnitude=bandmag)
    for name,values in ordinary.stages.items():table['class_after_'+name]=values
    measurement=ordinary.diagnostics['aperture_measurement']
    table['remeasured_aperture_snr']=measurement['snr'];table['aperture_snr_trusted']=measurement['trusted']
    for key in ('snr_bad','snr_contaminated_region','no_kron_snr_threshold','snr_risk_bright_exempt'):
        table[key]=ordinary.diagnostics[key]
    for key in ('flux','aperture_pixels'):
        if key in measurement:table['remeasured_aperture_'+key]=measurement[key]
    for key in ('bcg_raw_snr','bcg_residual_snr','bcg_model_at_center'):
        if key in measurement:table[key]=measurement[key]
    if snr_risk_model is not None:
        table['bcg_raw_snr']=measurement['snr'].copy()
        ix=np.clip(np.floor(data.geometry.x+.5).astype(int),0,reference.shape[1]-1)
        iy=np.clip(np.floor(data.geometry.y+.5).astype(int),0,reference.shape[0]-1)
        table['bcg_model_at_center']=model[iy,ix]
    if progress:progress('bright_gaia',len(table))
    result=label_bright_sources(data,ordinary.labels,image_shape=reference.shape,
        bright_region=bright,component_labels=components,gaia_table=gaia,image_header=header,
        source_ids=np.asarray(table['NUMBER']),config=JWSTBrightConfig('a2744',.03,gaia_reference_epoch=2016.))
    table['class_after_bright_gaia']=result.labels.source_class.copy()
    table['classification_reason']=np.asarray(result.labels.reason,str)
    g=data.geometry
    labels=PatchLabels(table,np.empty(reference.shape,np.uint8),result.labels.source_class.copy(),
        g.x,g.y,g.major,g.minor,g.theta,np.asarray(table['NUMBER'],np.int64),
        result.strict_center_x,result.strict_center_y,result.strict_center_source_id,
        strict_is_gaia=np.array(['gaia' in str(reason) for reason in result.strict_center_reason],bool))
    training=bp.image.copy();training[~bp.finite_mask]=np.nan
    wings=apply_star_protection(labels,image=training,header=bright_header if bright_header is not None else header,
        gaia=gaia,band=band,pixel_scale=.03,footprint=stars,sky_mask=background,
        snr_image=linear,valid_mask=bp.finite_mask)
    if progress:progress('dense_paint',len(labels.source_ids))
    ignored=np.isin(components,result.ordinary_ignore_component_ids) if len(result.ordinary_ignore_component_ids) else np.zeros(reference.shape,bool)
    if native_segmentation is not None:
        if native_segmentation.shape!=reference.shape:raise ValueError('Projected UNCOVER segmentation shape mismatch')
        # Official segments protect known no-Kron sources from negative sky
        # supervision. They are not substitute Kron/instance GT shapes.
        active_ids=labels.source_ids[:len(has)][(~has)&(labels.label_classes[:len(has)]!=int(C.DROPPED))]
        ignored |= np.isin(native_segmentation,active_ids)
    restricted=np.isin(components,result.restricted_fallback_component_ids) if len(result.restricted_fallback_component_ids) else None
    final=EllipseGeometry(labels.geom_x,labels.geom_y,labels.geom_major,labels.geom_minor,labels.geom_theta,np.pi*labels.geom_major*labels.geom_minor)
    labels.dense=fill_dense_regions(labels.table,SourceLabels(labels.label_classes,np.full(len(labels.source_ids),'',object)),
        reference.shape,geometry=final,background_mask=background if training_background is None else training_background,
        quality_ignore_mask=~bp.finite_mask,bright_region_mask=bright,
        restricted_fallback_mask=restricted,ordinary_ignore_mask=ignored,star_wings_mask=wings)
    labels.dense[~bp.finite_mask]=int(DenseLabel.STRICT_IGNORE)
    from .truncated_kron import large_kron_geometry
    rawtab,rawdata,_=adapt_catalog(catalog,header,reference.shape,native_header,include_outside=True)
    rawgeom=rawdata.geometry
    labels.truncation_geometry=large_kron_geometry(rawtab['NUMBER'],rawgeom.x,rawgeom.y,
        rawgeom.major,rawgeom.minor,rawgeom.theta)
    summary=dict(policy_version=POLICY_VERSION,catalog='UNCOVER DR3 native',sources=len(table),
        **model_risk_audit,
        final_classes={C(int(k)).name:int(v) for k,v in zip(*np.unique(labels.label_classes,return_counts=True))},
        extra_strict_centers=len(labels.strict_x),star_protection=labels.star_protection_policy,
        stages={k:{C(int(c)).name:int(n) for c,n in zip(*np.unique(v,return_counts=True))} for k,v in ordinary.stages.items()},
        missing_kron_policy='Only valid centers and magnitudes + available A conditions; no invented ellipse/area. Official segments only exclude background.',
        catalog_snr_policy=f'All catalog centers remeasured on FULL, including flag_lowsnr and already ignored/dropped rows. Native catalog SNR/flag does not pre-cut or cap classes. Measured <{snr_config.thresholds.ignore_below:g}/nonfinite ignore; [{snr_config.thresholds.ignore_below:g},{snr_config.thresholds.weak_below:g}) weak shape where geometry permits; no-Kron remains strict above max(general lower bound, configured no-Kron threshold).',
        snr_config=dict(ignore_below=snr_config.thresholds.ignore_below,weak_below=snr_config.thresholds.weak_below,
            no_kron_min_snr=snr_config.no_kron_min_snr,
            contaminated_no_kron_min_snr=snr_config.contaminated_no_kron_min_snr,
            reject_untrusted_in_contaminated=snr_config.reject_untrusted_in_contaminated,
            risk_bright_mag_max=snr_config.risk_bright_mag_max,
            risk_bright_photometry=f'UNCOVER {flux_key}, AB ZP28.9, finite positive flux and error; valid Kron required',
            contaminated_region_enabled=snr_contaminated_mask is not None),
        snr_risk_bright_exempt_sources=int(np.sum(ordinary.diagnostics['snr_risk_bright_exempt'])),
        containment_policy='Existing IoU>=0.8 / inner fraction>=0.8 rules; valid Kron and retained non-ignore only; no contains-ignore demotion.',
        training_reference='half-coadd image and sky for bright/dense; FULL coadd image/sky for SNR')
    return labels,summary
