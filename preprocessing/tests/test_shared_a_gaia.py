"""Regression checks for shared area policy and Gaia astrometry."""

import numpy as np
import unittest
from unittest.mock import patch
import tempfile,csv,json
from pathlib import Path
from astropy.io import fits
from astropy.table import Table, MaskedColumn
from astropy.time import Time
from astropy.wcs import WCS

from preprocessing.utils.area_filter import area_filter_masks
from preprocessing.utils.gaia import matching_radius_pixels, observation_time, propagated_gaia_radec
from preprocessing.bright_label import (
    BrightLabelConfig, classify_component_bright, project_gaia_rows,
)
from preprocessing.meas_processing import classify_meas_basics
from preprocessing.labels import SourceClass


def test_area_scales_and_boundaries():
    area = np.array([10000., 10001., 901., 900., 901.])
    mag = np.array([22., 22., 29., 29., 28.])
    expected = area_filter_masks(area, mag)
    assert expected[0].tolist() == [False, True, False, False, False]
    assert expected[1].tolist() == [False, False, True, False, False]
    for scale in (0.03, 0.063):
        actual = area_filter_masks(area * (0.168 / scale)**2, mag, pixel_scale_arcsec=scale)
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got, want)
    with unittest.TestCase().assertRaises(ValueError):
        area_filter_masks(area, mag, pixel_scale_arcsec=0)


def test_a_faint_large_is_dropped():
    from types import SimpleNamespace
    import preprocessing.meas_processing as mod
    geom = SimpleNamespace(area=np.array([901.]), x=np.array([10.]), y=np.array([10.]),
                           valid=lambda: np.array([True]), axis_ratio=lambda: np.array([1.]))
    table = Table({'ext_photometryKron_KronFlux_instFlux': [10**(-0.8)], 'deblend_nChild': [0]})
    with patch.object(mod, 'compute_kron_ellipse', lambda *args: geom):
        result = classify_meas_basics(table)
    assert result.labels.mask(SourceClass.DROPPED)[0]
    assert not result.after_a[0]


def test_motion_and_missing_values():
    table = Table({'ra': [10., 10., 10.], 'dec': [60., 60., 60.],
                   'pmra': [100., 100., 100.], 'pmdec': [0., 0., 0.],
                   'ref_epoch': [2015.5, 2016., np.nan]})
    table['pmra'] = MaskedColumn(table['pmra'], mask=[False, True, False])
    ra, dec, moved = propagated_gaia_radec(table, Time(2025.5, format='jyear'))
    assert moved.tolist() == [True, False, False]
    np.testing.assert_allclose((ra[0]-10)*3600, 2., atol=1e-5)
    np.testing.assert_array_equal(ra[1:], [10., 10.])
    assert not propagated_gaia_radec(table, None)[2].any()
    assert observation_time({}) is None
    assert observation_time({'MJD-AVG': 60000., 'MJD-OBS': 59000.}).mjd == 60000.


def test_projection_origin_and_ids():
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [50., 50.]
    wcs.wcs.crval = [10., 20.]
    wcs.wcs.cdelt = [-0.168/3600, 0.168/3600]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    sid = 1234567890123456789
    table = Table({'ra': [10.], 'dec': [20.], 'source_id': [sid]})
    for origin in (0, 1):
        row = project_gaia_rows(table, image_shape=(100, 100), image_header=wcs.to_header(), pixel_origin=origin)[0]
        assert row['source_id'] == sid
        np.testing.assert_allclose(row['x'], 49+origin)
        np.testing.assert_allclose(row['y'], 49+origin)


def test_outside_component_gaia_not_inserted():
    components = np.zeros((50, 50), dtype=np.int32)
    components[5:40, 5:40] = 1
    gaia = [{'x': 41., 'y': 20., 'source_id': 123, 'phot_g_mean_mag': 17.,'astrometric_evidence':True}]
    sources, _, _, _ = classify_component_bright(
        sources=[], gaia_rows=gaia, component_labels=components,
        quality_mask=None, config=BrightLabelConfig())
    assert not any(s.get('class') == 'gaia_star' for s in sources)
    gaia[0]['x'] = 20.
    sources, _, _, _ = classify_component_bright(
        sources=[], gaia_rows=gaia, component_labels=components,
        quality_mask=None, config=BrightLabelConfig())
    assert any(s.get('class') == 'gaia_star' for s in sources)
    for blocked in (components[:30, :30], np.pad(np.ones((10, 10), dtype=np.int32), 15)):
        sources, _, _, ignored = classify_component_bright(
            sources=[], gaia_rows=gaia, component_labels=blocked,
            quality_mask=None, config=BrightLabelConfig())
        assert any(s.get('class') == 'gaia_star' for s in sources)
        assert 1 not in ignored


def test_hsc_default_projection_uses_zero_origin_and_dr3_epoch():
    wcs=WCS(naxis=2);wcs.wcs.crpix=[50.,50.];wcs.wcs.crval=[10.,20.]
    wcs.wcs.cdelt=[-.168/3600,.168/3600];wcs.wcs.ctype=['RA---TAN','DEC--TAN']
    hdr=wcs.to_header();hdr['MJD-AVG']=Time(2026.,format='jyear').mjd
    tab=Table(dict(source_id=[123],ra=[10.],dec=[20.],phot_g_mean_mag=[17.],
        pmra=[100.],pmdec=[0.],pmra_error=[1.],pmdec_error=[1.]))
    row=project_gaia_rows(tab,image_shape=(100,100),image_header=hdr,reference_epoch=2016.)[0]
    self_x,self_y=wcs.all_world2pix(10.,20.,0)
    np.testing.assert_allclose(row['x']-self_x,-1/.168,atol=1e-5)
    np.testing.assert_allclose(row['y'],self_y,atol=1e-5)
    assert row['position_propagated'] and row['astrometric_evidence']
    assert 'ref_epoch' not in tab.colnames


def test_motion_projection_does_not_modify_input_table():
    tab=Table(dict(ra=[10.],dec=[20.],pmra=[100.],pmdec=[1.],ref_epoch=[2016.]))
    before=tab.copy();first=propagated_gaia_radec(tab,Time(2026.,format='jyear'))
    second=propagated_gaia_radec(tab,Time(2026.,format='jyear'))
    np.testing.assert_array_equal(tab['ra'],before['ra'])
    np.testing.assert_array_equal(first[0],second[0])


def test_hsc_epoch_uses_selected_visit_weights_and_original_build_manifest():
    from preprocessing.utils.gaia import hsc_observation_epoch
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);full=root/'full.fits'
        index=fits.BinTableHDU.from_columns([
            fits.Column(name='id',format='K',array=[100,200]),fits.Column(name='cat.archive',format='J',array=[1,1]),
            fits.Column(name='row0',format='J',array=[0,1]),fits.Column(name='name',format='16A',array=['VisitInfo','VisitInfo'])])
        index.header['AR_CATN']=0
        dates=Time([58000.,59000.],format='mjd',scale='utc')
        info=fits.BinTableHDU.from_columns([fits.Column(name='tai',format='K',array=np.rint(dates.tai.unix_tai*1e9).astype(np.int64))])
        info.header['AR_CATN']=1
        inputs=fits.BinTableHDU.from_columns([fits.Column(name='visit',format='K',array=[10,20]),
            fits.Column(name='visitInfo',format='K',array=[100,200]),fits.Column(name='weight',format='D',array=[2.,5.])])
        inputs.header['AR_CATN']=2
        fits.HDUList([fits.PrimaryHDU(),index,info,inputs]).writeto(full)
        np.testing.assert_allclose(hsc_observation_epoch(full)['mjd_utc'],(58000*2+59000*5)/7)
        image=root/'products/half_coadd/9813/HSC-I/4,5/half.fits';image.parent.mkdir(parents=True)
        fits.PrimaryHDU(np.zeros((10,10))).writeto(image)
        weight=root/'products/warp_weights/9813/HSC-I/4,5/weights.csv';weight.parent.mkdir(parents=True)
        with weight.open('w') as f:
            w=csv.DictWriter(f,fieldnames=['product_image_mask_fits','visit','weight']);w.writeheader()
            w.writerows([dict(product_image_mask_fits=str(image),visit=10,weight=3),dict(product_image_mask_fits=str(image),visit=20,weight=1)])
        value=hsc_observation_epoch(image,full)
        np.testing.assert_allclose(value['mjd_utc'],58250.)
        assert value['source']=='selected-product visit-weighted epoch'
        weight.unlink()
        summary=root/'products/_runs/run/analysis_summaries/4,5/half/HSC-I/test_summary.json';summary.parent.mkdir(parents=True)
        manifest=root/'inputs.json';manifest.write_text(json.dumps(dict(n_selected_inputs=1,selected_inputs=[dict(visit=20,weight=2.)])))
        summary.write_text(json.dumps(dict(n_inputs=1,input_manifest=str(manifest),outputs=dict(product_image_mask_fits=str(image)))))
        value=hsc_observation_epoch(image,full)
        np.testing.assert_allclose(value['mjd_utc'],59000.)
        assert value['weight_csv']==str(manifest)


def test_matching_radius_is_angular():
    for hsc_pixels in (6., 10.):
        hsc = matching_radius_pixels(0.168, hsc_pixels=hsc_pixels)
        jwst = matching_radius_pixels(0.03, hsc_pixels=hsc_pixels)
        np.testing.assert_allclose(hsc, hsc_pixels)
        np.testing.assert_allclose(hsc * 0.168, jwst * 0.03)
    assert matching_radius_pixels(0.03, arcsec=0.6) == 20.


def test_hsc_sat_mask_reads_both_official_and_half_header_encodings():
    from preprocessing.utils.hsc_psf_ee import read_sat_mask
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'input.fits'
        for key,value in [('MP_SAT',1),('MP_01','SAT')]:
            mask=fits.ImageHDU(np.array([[2,1],[0,2]],np.int32),name='MASK')
            mask.header[key]=value
            fits.HDUList([fits.PrimaryHDU(),fits.ImageHDU(np.zeros((2,2))),mask]).writeto(path,overwrite=True)
            np.testing.assert_array_equal(read_sat_mask(path,(2,2)),[[True,False],[False,True]])
        mask.header.remove(key)
        fits.HDUList([fits.PrimaryHDU(),fits.ImageHDU(np.zeros((2,2))),mask]).writeto(path,overwrite=True)
        assert read_sat_mask(path,(2,2)) is None


def test_hsc_sat_containment_includes_bleed_disconnected_and_pixel_corners():
    from scipy import ndimage
    from preprocessing.utils.hsc_psf_ee import sat_containment
    sat=np.zeros((51,51),bool);sat[24:27,24:27]=True
    ids,_=ndimage.label(sat,structure=np.ones((3,3)))
    assert sat_containment(ids,(25,25,5,5,0))['contained']
    sat[25,25:40]=True
    ids,_=ndimage.label(sat,structure=np.ones((3,3)))
    result=sat_containment(ids,(25,25,5,5,0))
    assert not result['contained'] and result['sat_outside_ee95']>0
    sat=np.zeros_like(sat);sat[25,25]=True;sat[40,40]=True
    ids,_=ndimage.label(sat)
    assert sat_containment(ids,(25,25,5,5,0))['contained']
    assert not sat_containment(ids,(25,25,5,5,0),associated_ids=[2])['contained']
    # A pixel center inside is insufficient: every pixel corner must fit.
    assert not sat_containment(ids,(25,25,.2,.2,0))['contained']


def test_hsc_psf_sat_gate_changes_only_safe_gaia_and_keeps_restricted_region():
    from scipy import ndimage
    from preprocessing.utils.hsc_psf_ee import apply_star_priors
    from preprocessing.utils.image_level import PatchLabels
    from preprocessing.region_filling import fill_dense_regions
    from preprocessing.utils.geometry import EllipseGeometry
    from preprocessing.labels import SourceLabels
    wcs=WCS(naxis=2);wcs.wcs.crpix=[1,1];wcs.wcs.crval=[10,20]
    wcs.wcs.cdelt=[-.168/3600,.168/3600];wcs.wcs.ctype=['RA---TAN','DEC--TAN']
    comps=np.zeros((51,51),np.int32);comps[10:41,10:41]=1
    model={123:dict(a=3.,b=3.,theta=0.,ee95_a=5.,ee95_b=5.)}
    def run(sat,parameters=model):
        labels=PatchLabels(table=Table(dict(id=np.array([88],np.int64))),dense=np.empty((0,0)),
            label_classes=np.array([SourceClass.ORDINARY_IGNORE]),geom_x=np.array([25.]),geom_y=np.array([25.]),
            geom_major=np.array([3.]),geom_minor=np.array([3.]),geom_theta=np.array([0.]),source_ids=np.array([88]),
            strict_x=np.array([25.]),strict_y=np.array([25.]),strict_ids=np.array([-124],np.int64),strict_is_gaia=np.array([True]))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'ref.fits';fits.PrimaryHDU(np.zeros((51,51)),wcs.to_header()).writeto(path)
            apply_star_priors(labels,reference=path,image_path=path,components=comps,center_components=[1],
                              output_root=folder,parameters=parameters,sat_mask=sat)
        geom=EllipseGeometry(labels.geom_x,labels.geom_y,labels.geom_major,labels.geom_minor,
                             labels.geom_theta,np.pi*labels.geom_major*labels.geom_minor)
        dense=fill_dense_regions(labels.table,SourceLabels(labels.label_classes,np.full(len(labels.table),'')),
                                comps.shape,geometry=geom,restricted_fallback_mask=comps==1)
        assert dense[12,12]==6
        return labels,dense
    sat=np.zeros_like(comps,bool);sat[25,25]=True
    good,dense=run(sat)
    assert good.source_ids.tolist()==[88,123] and not len(good.strict_x)
    assert dense[25,25]==2 and good.star_protection_sources[0]['status']=='psf_weak_shape'
    from preprocessing.utils.image_level import _tile_targets
    from types import SimpleNamespace
    spec=SimpleNamespace(size=51,x0=0,y0=0,x1=51,y1=51)
    good.dense=dense
    targets=_tile_targets(good,spec,(0,0))
    assert targets['shape_ids'].tolist()==[123] and not len(targets['strict_ids'])
    sat[38,38]=True  # disconnected SAT, same restricted bright region
    bad,dense=run(sat)
    assert bad.source_ids.tolist()==[88] and bad.strict_ids.tolist()==[-124]
    assert dense[25,25]==6 and bad.star_protection_sources[0]['sat_outside_ee95']==1
    bad.dense=dense
    targets=_tile_targets(bad,spec,(0,0))
    assert not len(targets['shape_ids']) and targets['strict_ids'].tolist()==[-124]
    assert targets['confidence'][25,25]>0 and targets['shape_weight'][25,25]==0
    sat[:]=False
    clear,_=run(sat)
    assert clear.star_protection_sources[0]['status']=='psf_weak_shape'
    unknown,_=run(None)
    assert unknown.strict_ids.tolist()==[-124] and unknown.star_protection_sources[0]['status']=='center_only_no_sat_mask'
    no_psf,_=run(sat,parameters={})
    assert no_psf.strict_ids.tolist()==[-124] and no_psf.star_protection_sources[0]['status']=='center_only_no_psf'


def _nested_example(parent_class=SourceClass.WEAK_SHAPE,extendedness=1.,psf=False):
    from preprocessing.utils.image_level import PatchLabels
    from preprocessing.utils.geometry import paint_ellipse
    dense=np.full((101,101),4,np.uint8);paint_ellipse(dense,50.,50.,40.,25.,0.,int(parent_class))
    labels=PatchLabels(table=Table(dict(id=[1,2],base_ClassificationExtendedness_value=[extendedness,1.],
            gaia_psf_model=[psf,False])),dense=dense,label_classes=np.array([parent_class,SourceClass.ORDINARY_IGNORE]),
        geom_x=np.array([50.,30.]),geom_y=np.array([50.,50.]),geom_major=np.array([40.,10.]),
        geom_minor=np.array([25.,8.]),geom_theta=np.zeros(2),source_ids=np.array([1,2]),
        strict_x=np.array([],np.float32),strict_y=np.array([],np.float32),strict_ids=np.array([],np.int64),strict_is_gaia=np.array([],bool))
    comp=np.zeros((101,101),np.int32);comp[48:53,28:33]=1;comp[48:53,48:53]=2
    return labels,comp


def test_nested_clean_and_weak_peaks_gain_confidence_without_changing_parent_shape():
    from preprocessing.bright_label import restore_nested_bright_centers
    from preprocessing.utils.image_level import _tile_targets
    from types import SimpleNamespace
    for cls in [SourceClass.CLEAN,SourceClass.WEAK_SHAPE]:
        labels,components=_nested_example(cls);before=labels.dense.copy()
        spec=SimpleNamespace(size=101,x0=0,y0=0)
        old=_tile_targets(labels,spec,(0,0))
        assert old['confidence'][50,30]==0 and old['conf_weight'][50,30]>0
        restore_nested_bright_centers(labels,components)
        assert len(labels.strict_ids)==1 and labels.strict_x.tolist()==[30.] and labels.strict_y.tolist()==[50.]
        assert not labels.strict_is_gaia.any()
        np.testing.assert_array_equal(labels.dense,before)
        new=_tile_targets(labels,spec,(0,0))
        assert new['confidence'][50,30]>0 and new['shape_ids'].tolist()==[1]
        np.testing.assert_array_equal(new['shape'],old['shape'])
        restore_nested_bright_centers(labels,components)
        assert len(labels.strict_ids)==1


def test_nested_unknown_eligible_but_stellar_and_gaia_psf_parents_excluded():
    from preprocessing.bright_label import restore_nested_bright_centers
    for ext,psf,expected in [(np.nan,False,1),(0.,False,0),(1.,True,0)]:
        labels,components=_nested_example(extendedness=ext,psf=psf)
        restore_nested_bright_centers(labels,components)
        assert len(labels.strict_ids)==expected


def test_nested_component_requires_full_containment_and_no_nearby_positive():
    from preprocessing.bright_label import restore_nested_bright_centers
    for mode in ['outside','ignore_pixel','near_center']:
        labels,components=_nested_example()
        if mode=='outside':components[50,8:29]=1
        if mode=='ignore_pixel':labels.dense[50,30]=3
        if mode=='near_center':
            components[:]=0;components[48:53,53:57]=1
        restore_nested_bright_centers(labels,components)
        assert len(labels.strict_ids)==0


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in globals().items()
                              if name.startswith('test_') and callable(fn))


if __name__ == '__main__':
    unittest.main()
