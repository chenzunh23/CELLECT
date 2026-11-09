"""Regression checks for independent HSC/JWST images sharing local channel zero."""
import numpy as np
import torch
from unittest.mock import patch
from astro_train_ops import _update_detection_totals
from utils.eval_metrics_utils import _init_detection_totals, _merge_detection_totals, _finalize_detection_totals
from utils.detection_metadata import sample_band_name, valid_predictions

BANDS = ['HSC-G', 'HSC-I', 'F115W', 'F444W', 'F090W']


def _write_discovery_store(path,tract,band,field):
    import json
    path.mkdir(parents=True)
    (path/'.zattrs').write_text(json.dumps(dict(format='cellect_direct_patch_zarr',image_level_training=True,
        tract=tract,patch=field,bands=[band])))
    (path.parent/(path.name+'_manifest.json')).write_text('{}')
    def array(name,value):
        p=path/name;p.mkdir();value=np.asarray(value)
        (p/'.zarray').write_text(json.dumps(dict(shape=value.shape,chunks=value.shape,dtype=value.dtype.str)))
        (p/'.'.join('0' for _ in value.shape)).write_bytes(value.tobytes())
    def text(value):return np.frombuffer(value.encode()+b'\0',dtype=np.uint8)[None,:].copy()
    p=path/'images';p.mkdir();(p/'.zarray').write_text(json.dumps(dict(shape=[1,1,3,2,2],chunks=[1,1,3,2,2],dtype='<f4')))
    array('tile_x0',np.array([0],np.int32));array('tile_y0',np.array([0],np.int32))
    array('tile_name',text('grid_r00_c00_x0_y0'));array('dataset_source',text('half_coadd'))


def test_independent_ground_and_space_roots_filter_families_and_keep_selectors():
    import tempfile
    from pathlib import Path
    from astro_train_zarr_data import discover_zarr_image_records,discover_zarr_image_bands
    from utils.train_eval_utils import filter_records_by_patches
    with tempfile.TemporaryDirectory() as td:
        base=Path(td);ground=base/'ground';space=base/'space'
        _write_discovery_store(ground/'zarr/hsc/i.zarr','9813','HSC-I','4,5')
        _write_discovery_store(ground/'zarr/wrong/a.zarr','Abell2744','F090W','wrong_field')
        _write_discovery_store(space/'zarr/abell/a.zarr','Abell2744','F444W','x+00_y+00')
        _write_discovery_store(space/'zarr/cosmos/c.zarr','COSMOS_1727','F115W','P0019_x00000_y00000')
        _write_discovery_store(space/'zarr/wrong/i.zarr','9813','HSC-G','8,4')
        kwargs=dict(hsc_root=ground,space_root=space)
        assert discover_zarr_image_bands(space,**kwargs)==['F115W','F444W','HSC-I']
        records=discover_zarr_image_records(space,bands=[],**kwargs)
        assert len(records)==3 and {r.tract for r in records}=={'9813','Abell2744','COSMOS_1727'}
        selected=filter_records_by_patches(records,{'half_coadd:4,5'},space)
        assert len(selected)==1 and selected[0].tract=='9813'
        selected=filter_records_by_patches(records,{records[1].name},space)
        assert len(selected)==1 and selected[0].name==records[1].name
        assert len(discover_zarr_image_records(space,bands=[],max_records=2,**kwargs))==2
        assert len(discover_zarr_image_records(ground/'zarr',bands=[]))==2


def test_shared_family_root_is_scanned_once_and_duplicate_versions_rejected():
    import tempfile
    from pathlib import Path
    from astro_train_zarr_data import discover_zarr_image_records
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        _write_discovery_store(root/'zarr/hsc/i.zarr','9813','HSC-I','4,5')
        _write_discovery_store(root/'zarr/abell/a.zarr','Abell2744','F444W','x+00_y+00')
        assert len(discover_zarr_image_records(root,bands=[],hsc_root=root,space_root=root/'zarr'))==2
        _write_discovery_store(root/'zarr/another/i.zarr','9813','HSC-I','4,5')
        try:discover_zarr_image_records(root,bands=[],hsc_root=root)
        except ValueError as exc:assert 'Duplicate image record' in str(exc)
        else:raise AssertionError('Two versions of the same HSC record were accepted')

def make_totals(names, linking=False):
    n=len(names)
    outputs={'confidence':torch.zeros(n,1,5,8,8)}
    batch={'band_names':[[b] for b in names],
           'band_centers':[[torch.tensor([[2.,3.]])] for _ in names],
           'band_valid_mask':torch.ones(n,1,8,8,dtype=torch.bool)}
    batch['band_valid_mask'][:,:,7,7]=False
    totals=_init_detection_totals(BANDS)
    with patch('astro_train_ops.detect_centers',return_value=[np.array([[2.,3.],[7.,7.]]) for _ in names]), \
         patch('astro_train_ops.build_adjacent_mutual_link_components', side_effect=AssertionError('linking ran')):
        _update_detection_totals(totals,torch.nn.Identity(),outputs,batch,threshold=.5,
            nms_radius=1,confidence_score='max',use_ordinal_expectation=False,
            debug_ordinal_expectation=False,center_refinement='integer',center_refinement_radius=1,
            match_radius=3.,use_en_postprocess=False,en_candidate_count=5,en_threshold=.6,
            use_ex_link_postprocess=True,ex_link_threshold=.5,ex_band_pairs=None,band_names=BANDS,
            detection_linking=linking)
    return totals

def test_mixed_band_metrics_and_no_implicit_linking():
    totals=make_totals(BANDS[:4])
    result=_finalize_detection_totals(totals,band_names=BANDS,use_ex_link_postprocess=True)
    assert result['overall']['tp']==4 and result['overall']['fp']==0
    assert result['overall']['samples']==4
    for b in BANDS[:4]:
        assert result['per_band'][b]['tp']==1 and result['per_band'][b]['evaluated']
    assert not result['per_band']['F090W']['evaluated']
    assert 'linked' not in result and 'link_metrics' not in result

def test_distributed_merge_and_single_band_link_guard():
    totals=_merge_detection_totals([make_totals(BANDS[:2],True),make_totals(BANDS[2:4])],BANDS)
    result=_finalize_detection_totals(totals,band_names=BANDS,use_ex_link_postprocess=False)
    assert result['overall']['tp']==4 and result['samples']==4
    assert not result['linking_enabled']
    assert all(result['per_band'][b]['samples']==1 for b in BANDS[:4])

def test_legacy_names_and_invalid_centers():
    assert sample_band_name({},0,1,BANDS)=='HSC-I'
    mask=torch.ones(1,1,8,8,dtype=torch.bool);mask[0,0,3,2]=False
    xy=valid_predictions([[2,3],[4,4],[-1,0],[float('nan'),0]],{'band_valid_mask':mask},0,0)
    assert np.array_equal(xy,[[4,4]])

def test_half_noisy_and_abell_selector_groups():
    from pathlib import Path
    from astro_train_data import CutoutRecord
    from utils.train_eval_utils import filter_records_by_patches
    def rec(name,uri,source,tract,field):
        return CutoutRecord(name=name,image_paths=(uri,),meas_path='',x0=0,y0=0,
            dataset_source=source,tract=tract,patch=field)
    records=[rec('half','zarr:///tmp/0,0__0.zarr#0','half_coadd','9813','0,0'),
             rec('noisy','zarr:///tmp/0,0__group_02.zarr#0','noisy','9813','0,0'),
             rec('abell','zarr:///tmp/x+00_y+00__half.zarr#0','half_coadd','Abell2744','x+00_y+00')]
    for selector,name in [('half_coadd:0,0@0','half'),('noisy:0,0@group_02','noisy'),
                          ('half_coadd:Abell2744/x+00_y+00@half','abell')]:
        found=filter_records_by_patches(records,{selector},Path('/tmp'))
        assert [r.name for r in found]==[name]

def test_zarr_wcs_tile_offset():
    from types import SimpleNamespace
    from astropy.wcs import WCS
    from utils.train_eval_utils import wcs_for_path
    parent=WCS(naxis=2);parent.wcs.crpix=[2048.,2048.];parent.wcs.crval=[150.,2.]
    parent.wcs.cdelt=[-.03/3600,.03/3600];parent.wcs.ctype=['RA---TAN','DEC--TAN']
    reader=SimpleNamespace(attrs={'sky_wcs_header':parent.to_header().tostring(sep='\n')},
        read_full_small=lambda name:np.array([512 if name=='tile_x0' else 1024]))
    with patch('astro_train_zarr_data._reader',return_value=reader):
        tile=wcs_for_path('zarr:///tmp/test.zarr#0',1,{})
    assert tile is not None
    assert np.allclose(tile.all_pix2world([[100,200]],0),parent.all_pix2world([[612,1224]],0),atol=1e-10)


def test_full_record_selector_keeps_one_band_tile_and_legacy_parent_selection():
    from pathlib import Path
    from astro_train_data import CutoutRecord
    from utils.train_eval_utils import filter_records_by_patches
    def rec(parent,band,tile):
        return CutoutRecord(name=f'Abell2744/half_coadd/{parent}/half/{band}/{tile}',
            image_paths=('zarr:///tmp/parent.zarr#0',),meas_path='',x0=368,y0=2208,
            dataset_source='half_coadd',tract='Abell2744',patch=parent,tile_name=tile)
    records=[rec('x+00_y+00','F115W','r06_c01_x368_y2208'),
             rec('x+00_y+00','F115W','r06_c02_x736_y2208'),
             rec('x+00_y+00','F150W','r06_c01_x368_y2208'),
             rec('x+01_y+00','F115W','r06_c01_x368_y2208')]
    selected=filter_records_by_patches(records,{records[0].name},Path('/tmp'))
    assert [r.name for r in selected]==[records[0].name]
    parent=filter_records_by_patches(records,{'half_coadd:Abell2744/x+00_y+00'},Path('/tmp'))
    assert [r.name for r in parent]==[r.name for r in records[:3]]
