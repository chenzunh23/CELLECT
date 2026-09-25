"""Regression checks for independent HSC/JWST images sharing local channel zero."""
import numpy as np
import torch
from unittest.mock import patch
from astro_train_ops import _update_detection_totals
from utils.eval_metrics_utils import _init_detection_totals, _merge_detection_totals, _finalize_detection_totals
from utils.detection_metadata import sample_band_name, valid_predictions

BANDS = ['HSC-G', 'HSC-I', 'F115W', 'F444W', 'F090W']

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
