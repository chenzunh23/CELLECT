"""Native-resolution heatmap target, ownership, gradient and decoding contracts."""
import json
from dataclasses import replace
import numpy as np
import pytest
import torch

from sam_backbone.heatmap import (HeatmapConfig,choose_anchors,make_targets,center_loss,
    decode,proposal_kwargs_from_state_dict)
from sam_backbone.decoder import SamCellectDecoder
from astro_train_ops import LossWeights,dense_losses_any,detect_centers_with_scores


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def targets(xy, *, image=None,base=None,valid=None,config=HeatmapConfig()):
    return make_targets(torch.tensor(xy,dtype=torch.float32).reshape(-1,2),
        torch.zeros(3,32,32) if image is None else image,
        torch.ones(32,32) if base is None else base,
        torch.ones(32,32,dtype=torch.bool) if valid is None else valid,config=config)


def batch():
    centers=torch.tensor([[8.2,8.3],[18.1,18.2]])
    data=dict(band_centers=[[centers]],band_ids=[[torch.tensor([10,20])]],
        band_strict_center_only_centers=[[torch.tensor([[20.,8.]])]],
        band_strict_center_only_ids=[[torch.tensor([-1])]],
        band_shape_source_centers=[[centers]],band_shape_source_ids=[[torch.tensor([10,20])]],
        band_shape_source_values=[[torch.tensor([[3.,2.,0.],[4.,3.,0.]])]],
        band_shape_source_classes=[[torch.tensor([1,2])]],band_shape_source_weights=[[torch.ones(2)]],
        band_names=[['HSC-I']],proposal_metadata=[{}],image=torch.zeros(1,1,3,32,32))
    for key in ['band_confidence','band_shape_weight','band_clean_mask','band_confidence_weight',
                'band_ignore_mask','band_center_only_mask','band_pseudo_mask','band_pu_class_mask','band_valid_mask']:
        data[key]=torch.ones(1,1,32,32)
    data['band_shape']=torch.zeros(1,1,3,32,32)
    return data


def test_boundary_brightness_only_breaks_near_boundary():
    image=torch.zeros(3,32,32);image[:,8,9]=10
    xy=torch.tensor([[8.49,8.],[8.3,8.],[8.51,8.]])
    q,valid=choose_anchors(xy,image,torch.ones(32,32,dtype=torch.bool))
    assert q.tolist()==[[9,8],[8,8],[9,8]]
    assert valid.all()


def test_gaussian_max_peak_one_four_sigma_and_ignore_override():
    t=targets([[8.2,8.],[12.,8.]],base=torch.zeros(32,32))
    assert t['heatmap'][8,8]==1 and t['heatmap'][8,12]==1
    assert t['heatmap'].max()==1
    assert t['heatmap'][8,1]==0  # >4sigma from every source
    assert t['weight'][8,8]>0 and t['weight'][31,31]==0
    assert t['heatmap'][8,6].item()==pytest.approx(np.exp(-4/4.5))


def test_invalid_input_never_gets_new_gaussian_supervision():
    valid=torch.ones(32,32,dtype=torch.bool);valid[8,7]=False
    t=targets([[8.,8.]],valid=valid)
    assert t['weight'][8,7]==0 and t['heatmap'][8,7]==0


def test_offset_5x5_shape_3x3_and_absolute_vote():
    t=targets([[8.2,8.3]])
    assert t['offset_valid'].sum()==25
    assert t['shape_valid'].sum()==9
    torch.testing.assert_close(t['offset_target']+t['reg_xy'],torch.tensor([[8.2,8.3]]).expand(25,-1))
    assert t['offset_target'].abs().max()>2


def test_nearest_owner_ties_skip_and_no_duplicate_regression():
    t=targets([[8.,10.],[10.,10.]])
    assert not (t['reg_xy'][:,0]==9).any()
    unique=t['reg_xy'].unique(dim=0)
    assert len(unique)==len(t['reg_xy'])
    for xy,source in zip(t['reg_xy'],t['reg_source']):
        assert (xy[0]<9)==(source==0)


@pytest.mark.parametrize('mode',['soft_bce','focal'])
def test_losses_have_finite_gradients_empty_images_and_ignore(mode):
    for xy in [[],[[8.,8.]]]:
        t=targets(xy,base=torch.zeros(32,32));logits=torch.zeros(32,32,requires_grad=True)
        loss=center_loss(logits,t['heatmap'],t['weight'],mode);loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
        assert logits.grad[31,31]==0
        if xy:assert logits.grad[8,8]<0


def test_three_heads_decoupled_and_legacy_unchanged():
    decoder=SamCellectDecoder(in_channels=16,decoder_channels=(16,12,8,4),proposal_mode='heatmap')
    out=decoder(torch.randn(1,1,16,2,2))
    assert out['confidence'].shape==(1,1,1,32,32)
    assert out['offset'].shape==(1,1,2,32,32)
    out['shape'].square().mean().backward()
    assert all(p.grad is None for p in decoder.center_head.parameters())
    assert all(p.grad is None for p in decoder.offset_head.parameters())
    assert decoder.stem[0].weight.grad is not None
    legacy=SamCellectDecoder(in_channels=16,decoder_channels=(16,12,8,4))
    assert legacy(torch.randn(1,16,2,2))['confidence'].shape[1]==5
    assert proposal_kwargs_from_state_dict(decoder.state_dict())['proposal_mode']=='ordinal' # decoder-only has no model prefix
    assert proposal_kwargs_from_state_dict({'module.decoder.'+k:v for k,v in decoder.state_dict().items()})['proposal_mode']=='heatmap'


@pytest.mark.parametrize('mode',['soft_bce','focal'])
def test_collated_training_step_backward_all_three_heads(mode):
    decoder=SamCellectDecoder(in_channels=16,decoder_channels=(16,12,8,4),proposal_mode='heatmap')
    out=decoder(torch.randn(1,1,16,2,2));data=batch()
    losses=dense_losses_any(out,data,weights=LossWeights(heatmap_loss=mode),device=torch.device('cpu'),center_radius_px=3.)
    losses['total'].backward()
    assert torch.isfinite(losses['total'])
    for tower in [decoder.center_head,decoder.offset_head,decoder.shape_head]:
        assert tower[-1].weight.grad is not None if isinstance(tower,torch.nn.Sequential) else tower.weight.grad is not None


def test_strict_center_blocks_neighbors_shape_but_has_offset():
    data=batch();data['band_strict_center_only_centers']=[[torch.tensor([[10.,8.]])]]
    data['band_centers']=[[torch.tensor([[8.,8.]])]];data['band_ids']=[[torch.tensor([10])]]
    shape=torch.ones(1,1,3,32,32,requires_grad=True)
    out=dict(confidence=torch.zeros(1,1,1,32,32,requires_grad=True),
             shape=shape,offset=torch.zeros(1,1,2,32,32,requires_grad=True))
    losses=dense_losses_any(out,data,weights=LossWeights(),device=torch.device('cpu'),center_radius_px=3.)
    losses['total'].backward()
    assert shape.grad[0,0,:,8,10].abs().sum()==0
    assert out['offset'].grad[0,0,:,8,11].abs().sum()>0


def test_decode_threshold_nms_no_connected_component_merge_and_offset():
    logits=torch.full((1,1,20,20),-20.)
    logits[0,0,8,8]=4;logits[0,0,8,10]=3;logits[0,0,8,9]=1
    logits[0,0,15,15]=0 # exactly .5: excluded
    offsets=torch.zeros(1,2,20,20);offsets[0,:,8,8]=torch.tensor([1.7,.2])
    out=dict(confidence=logits,offset=offsets,shape=torch.ones(1,3,20,20))
    found=detect_centers_with_scores(out,threshold=.5,confidence_score='ordinal_expectation')[0]
    assert len(found['xy'])==2
    np.testing.assert_allclose(found['xy'][0],[9.7,8.2],atol=1e-5)
    np.testing.assert_array_equal(found['anchor_xy'][0],[8,8])


def test_equal_adjacent_peak_plateau_uses_centroid_of_corrected_votes():
    logits=torch.full((1,1,12,12),-20.)
    logits[0,0,5,5:7]=2.
    offsets=torch.zeros(1,2,12,12)
    offsets[0,:,5,5]=torch.tensor([1.,0.])  # vote (6,5)
    offsets[0,:,5,6]=torch.tensor([0.,2.])  # vote (6,7)
    found=decode(dict(confidence=logits,offset=offsets))[0]
    torch.testing.assert_close(found['xy'],torch.tensor([[6.,6.]]))
    torch.testing.assert_close(found['anchor_xy'],torch.tensor([[5.,5.]]))
    torch.testing.assert_close(found['score'],torch.tensor([2.]).sigmoid())


def test_equal_plateaus_merge_diagonally_and_transitively_but_not_across_valleys():
    logits=torch.full((1,1,12,12),-20.)
    for x,y in ((2,2),(3,3),(4,3),(6,3)):logits[0,0,y,x]=2.
    found=decode(dict(confidence=logits,offset=torch.zeros(1,2,12,12)))[0]
    torch.testing.assert_close(found['xy'],torch.tensor([[3.,8/3],[6.,3.]]))
    assert len(found['score'])==2


def test_adjacent_unequal_peaks_and_invalid_offset_votes_are_not_averaged():
    logits=torch.full((1,1,12,12),-20.)
    logits[0,0,5,5]=2.;logits[0,0,5,6]=2.001
    offsets=torch.zeros(1,2,12,12)
    assert len(decode(dict(confidence=logits,offset=offsets),nms_radius=0)[0]['xy'])==2
    logits[0,0,5,6]=2.;offsets[0,0,5,5]=float('nan')
    found=decode(dict(confidence=logits,offset=offsets))[0]
    torch.testing.assert_close(found['xy'],torch.tensor([[6.,5.]]))
    torch.testing.assert_close(found['anchor_xy'],torch.tensor([[6.,5.]]))
    offsets[0,0,5,6]=100.
    assert len(decode(dict(confidence=logits,offset=offsets))[0]['xy'])==0


def test_plateau_export_samples_shape_at_original_representative_anchor():
    from eval.eval_utils import detection_rows
    logits=torch.full((1,1,12,12),-20.);logits[0,0,5,5:7]=2.
    offsets=torch.zeros(1,2,12,12);offsets[0,:,5,5]=torch.tensor([1.,0.]);offsets[0,:,5,6]=torch.tensor([0.,2.])
    shape=torch.ones(1,3,12,12);shape[0,:,5,5]=torch.tensor([4.,2.,.3])
    rows=detection_rows(dict(confidence=logits,offset=offsets,shape=shape),threshold=.5,
        nms_radius=1,confidence_score='heatmap',center_refinement='integer',center_refinement_radius=1,width=12,height=12)
    assert len(rows)==1 and rows[0]['x']==6 and rows[0]['y']==6
    assert rows[0]['major']==4 and rows[0]['minor']==2


def test_psf_ee_sigma_conversion_and_clipping():
    from preprocessing.utils.heatmap_sigma import resolve_sigma
    for sigma,expected in [(.1,2/3),(1.5,1.5),(10,8/3)]:
        fractions={'4':.1,'3':.35,'2':.6,'1':.7}
        config={'level_radii_pixels':{k:float(sigma*np.sqrt(-2*np.log1p(-f))) for k,f in fractions.items()},'ee_fractions':fractions}
        assert resolve_sigma(json.dumps({'confidence_config':config}),'HSC-I')==pytest.approx(expected)
    with pytest.raises(ValueError,match='calibration'):
        resolve_sigma(json.dumps({'patch':'4,5'}),'HSC-I')


def test_eval_shapes_sample_original_anchor_after_offset():
    from eval.eval_utils import detection_rows
    logits=torch.full((1,1,16,16),-20.);logits[0,0,5,5]=5
    shape=torch.ones(1,3,16,16);shape[0,:,5,5]=torch.tensor([4.,2.,.3])
    offset=torch.zeros(1,2,16,16);offset[0,:,5,5]=torch.tensor([2.,1.])
    rows=detection_rows(dict(confidence=logits,shape=shape,offset=offset),threshold=.5,nms_radius=1,
        confidence_score='heatmap',center_refinement='integer',center_refinement_radius=1,width=16,height=16)
    assert rows[0]['x']==7 and rows[0]['major']==4 and rows[0]['minor']==2


@pytest.mark.parametrize('proposal_mode',['ordinal','heatmap'])
def test_optional_image_fusion_first_step_gradient_and_checkpoint_detection(proposal_mode):
    decoder=SamCellectDecoder(in_channels=16,decoder_channels=(16,12,8,4),proposal_mode=proposal_mode,heatmap_image_fusion=True)
    out=decoder(torch.randn(1,2,16,2,2),images=torch.randn(1,2,3,32,32))
    assert out['confidence'].shape==(1,2,1 if proposal_mode=='heatmap' else 5,32,32)
    (out['confidence'].square().mean()+out['shape'].square().mean()).backward()
    for parameter in decoder.heatmap_image_pyramid.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum()>0
    state={'decoder.'+k:v for k,v in decoder.state_dict().items()}
    assert proposal_kwargs_from_state_dict(state)==dict(proposal_mode=proposal_mode,heatmap_image_fusion=True)


def test_ordinal_fusion_trains_from_existing_dense_confidence_targets():
    decoder=SamCellectDecoder(in_channels=16,decoder_channels=(16,12,8,4),heatmap_image_fusion=True)
    data=batch();data['image']=torch.randn(1,1,3,32,32)
    for key in ('band_strict_center_only_centers','band_strict_center_only_ids','proposal_metadata'):
        del data[key]  # Old CE training does not need heatmap's sparse strict targets.
    data['band_confidence'][0,0,8,8]=4
    out=decoder(torch.randn(1,1,16,2,2),images=data['image'])
    assert 'offset' not in out and out['confidence'].shape[2]==5
    losses=dense_losses_any(out,data,weights=LossWeights(confidence_loss_mode='ce_hard',
        center_position=0.,shape_geometry_loss='log_spd'),device=torch.device('cpu'),center_radius_px=3.)
    losses['total'].backward()
    assert torch.isfinite(losses['total'])
    assert decoder.confidence_head.weight.grad.abs().sum()>0
    assert decoder.heatmap_image_pyramid.full[0].weight.grad.abs().sum()>0


def test_epoch_and_iteration_metrics_preserve_explicit_heatmap_offset():
    from astro_train_ops import run_epoch,validate_epoch
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder=SamCellectDecoder(in_channels=16,decoder_channels=(16,12,8,4),proposal_mode='heatmap')
            self.register_buffer('features',torch.randn(1,1,16,2,2))
        def forward(self,image):
            return self.decoder(self.features,images=image)
    model=Tiny();records=[]
    kwargs=dict(device=torch.device('cpu'),weights=LossWeights(center_position=0.,mask_outer_weight=0.),
        triplet_loss_fn=None,triplet_enabled=False,ex_enabled=False,en_enabled=False,
        matcher_candidate_count=1,matcher_max_anchors_per_band=1,triplet_max_sources_per_group=1,
        triplet_negative_scope='ignore',ex_band_pairs=None,center_radius_px=3.,show_progress=False)
    train=run_epoch(model,[batch()],optimizer=torch.optim.SGD(model.parameters(),lr=1e-4),
        iteration_log_interval=1,iteration_log_fn=lambda values,step:records.append(values),**kwargs)
    assert train['offset']==train['center'] and train['offset']>0
    assert records[0]['loss/offset']==pytest.approx(train['offset'])
    validation,_=validate_epoch(model,[batch()],compute_detection=False,threshold=.5,nms_radius=1,
        confidence_score='heatmap',use_ordinal_expectation=False,debug_ordinal_expectation=False,
        center_refinement='integer',center_refinement_radius=1,use_en_postprocess=False,en_threshold=.5,
        use_ex_link_postprocess=False,ex_link_threshold=.5,band_names=['HSC-I'],**kwargs)
    assert validation['offset']==validation['center'] and validation['offset']>0


@pytest.mark.parametrize('proposal_mode',['ordinal','heatmap'])
def test_fused_model_browser_checkpoint_roundtrip(tmp_path,monkeypatch,proposal_mode):
    import eval.eval_utils as evaluation
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder=SamCellectDecoder(in_channels=16,decoder_channels=(16,12,8,4),proposal_mode=proposal_mode,heatmap_image_fusion=True)
    model=Tiny();checkpoint=tmp_path/'fusion.pt'
    torch.save({'model':model.state_dict(),'args':{'proposal_mode':proposal_mode}},checkpoint)
    def build(*args,**kwargs):
        assert kwargs['proposal_mode']==proposal_mode and kwargs['heatmap_image_fusion']
        return Tiny()
    monkeypatch.setattr(evaluation,'build_sam_cellect2d',build)
    loaded,_=evaluation.load_cellect_model(checkpoint,None,torch.device('cpu'),['HSC-I'])
    assert loaded.decoder.proposal_mode==proposal_mode
    for name,value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name],value,rtol=0,atol=0)


@pytest.mark.parametrize('proposal_mode',[None,'ordinal','heatmap'])
@pytest.mark.parametrize('switches,expected',[
    ({},None),({'IMAGE_FUSION':'1'},True),({'HEATMAP_IMAGE_FUSION':'1'},True),
    ({'IMAGE_FUSION':'0','HEATMAP_IMAGE_FUSION':'1'},False)])
def test_image_fusion_shell_switch_is_independent_of_proposal_mode(tmp_path,proposal_mode,switches,expected):
    import os
    from pathlib import Path
    import shlex
    import subprocess
    import sys
    from astro_train_eval import build_arg_parser
    selector=tmp_path/'patches.txt';selector.write_text('4,5\n')
    checkpoint=tmp_path/'sam.pth';checkpoint.touch()
    env=dict(os.environ)
    for key in ('PROPOSAL_MODE','IMAGE_FUSION','HEATMAP_IMAGE_FUSION'):
        env.pop(key,None)
    env.update(DRY_RUN='1',MODE='train',NPROC='1',PYTHON=sys.executable,
        TRAIN_FILE=str(selector),VAL_FILE=str(selector),SAM_CKPT=str(checkpoint),**switches)
    if proposal_mode is not None:env['PROPOSAL_MODE']=proposal_mode
    result=subprocess.run(['bash','scripts/train_mixed_zarr.sh'],
        cwd=Path(__file__).resolve().parents[1],env=env,text=True,capture_output=True,check=True)
    args=build_arg_parser().parse_args(shlex.split(result.stdout)[2:])
    assert args.heatmap_image_fusion is expected
    assert args.proposal_mode==proposal_mode
    assert args.confidence_loss_mode=='ce_hard'


@pytest.mark.parametrize('flag,expected',[
    ('--image-fusion',True),('--no-image-fusion',False),
    ('--heatmap-image-fusion',True),('--no-heatmap-image-fusion',False)])
def test_image_fusion_cli_aliases(flag,expected):
    from astro_train_eval import build_arg_parser
    assert build_arg_parser().parse_args(['--root','/tmp',flag]).heatmap_image_fusion is expected


def test_browser_heatmap_score_is_sigmoid_not_one_class_softmax():
    from eval.hsctiles.browser_confidence import confidence_maps
    logits=torch.tensor([[[[-3.,0.],[1.,3.]]]])
    maps=confidence_maps({'confidence':logits},width=2,height=2,valid_mask=np.ones((2,2),bool),score_mode='ordinal_expectation')
    np.testing.assert_allclose(maps['score'],logits.sigmoid().numpy()[0,0])
    assert len(np.unique(maps['labels']))>1


def test_empty_sparse_batch_still_has_all_head_gradients():
    decoder=SamCellectDecoder(in_channels=16,decoder_channels=(16,12,8,4),proposal_mode='heatmap')
    data=batch()
    for key in ['band_centers','band_strict_center_only_centers','band_shape_source_centers']:
        data[key]=[[torch.empty(0,2)]]
    data['band_shape_source_values']=[[torch.empty(0,3)]]
    for key in ['band_ids','band_strict_center_only_ids','band_shape_source_ids','band_shape_source_classes','band_shape_source_weights']:
        data[key]=[[torch.empty(0,dtype=torch.long)]]
    outputs=decoder(torch.randn(1,1,16,2,2))
    losses=dense_losses_any(outputs,data,weights=LossWeights(),device=torch.device('cpu'),center_radius_px=3.)
    losses['total'].backward()
    assert all(p.grad is not None for p in decoder.parameters())
    assert torch.isfinite(losses['total'])


def test_mask_prompt_uses_original_anchor_shape_without_peak_matching():
    from sam_backbone.losses import _build_pred_prompt_tensors
    logits=torch.full((1,1,1,16,16),-20.);logits[0,0,0,5,5]=5
    shape=torch.ones(1,1,3,16,16);shape[0,0,:,5,5]=torch.tensor([4.,2.,.3])
    offset=torch.zeros(1,1,2,16,16);offset[0,0,:,5,5]=torch.tensor([2.,1.])
    def unused(*args,**kwargs):raise AssertionError('Legacy peak decoder must not run')
    prompts=_build_pred_prompt_tensors(dict(confidence=logits,shape=shape,offset=offset),
        dict(band_clean_mask=torch.zeros(1,1,16,16),band_shape=torch.zeros_like(shape)),
        device=torch.device('cpu'),threshold=.5,nms_radius=1,confidence_score='heatmap',
        use_ordinal_expectation=False,debug_ordinal_expectation=False,center_refinement='integer',
        center_refinement_radius=1,max_pred_per_sample=128,sample_fraction=1.,unmatched_weight=.2,
        detach_prompt_shapes=False,detect_centers_fn=unused)
    torch.testing.assert_close(prompts['centers'],torch.tensor([[7.,6.]]))
    torch.testing.assert_close(prompts['prompt_shapes'],torch.tensor([[4.,2.,.3]]))


def test_zarr_augmentation_flips_strict_centers_and_keeps_ids(tmp_path,monkeypatch):
    from astro_train_zarr_data import ZarrCutoutDataset
    from astro_train_data import CutoutRecord,collate_cutouts
    from preprocessing.tests.test_segmentation_zarr import task,labels
    from preprocessing.build_image_level_zarr import write_classified_patch
    lab=labels();lab.strict_x=np.array([4.]);lab.strict_y=np.array([4.]);lab.strict_ids=np.array([-17])
    lab.strict_is_gaia=np.array([True])
    monkeypatch.setattr('preprocessing.build_image_level_zarr._scale_image_chw',lambda im,t:np.repeat(im[None],3,axis=0))
    result=write_classified_patch(task(tmp_path),np.ones((32,32),np.float32),lab)
    record=CutoutRecord('flip',(f'zarr://{result["output"]}#0',),'',0,0)
    monkeypatch.setattr('astro_train_zarr_data.random.random',lambda:0.)
    item=ZarrCutoutDataset([record],augment=True)[0]
    torch.testing.assert_close(item['band_strict_center_only_centers'][0],torch.tensor([[11.,4.]]))
    assert item['band_confidence'][0,4,11]==4
    assert collate_cutouts([item])['band_strict_center_only_ids'][0][0].tolist()==[-17]


def test_compiled_heatmap_decoder_bfloat16_loss_backward():
    decoder=SamCellectDecoder(in_channels=16,decoder_channels=(16,12,8,4),proposal_mode='heatmap')
    compiled=torch.compile(decoder,backend='eager')
    with torch.autocast('cpu',dtype=torch.bfloat16):
        outputs=compiled(torch.randn(1,1,16,2,2))
        losses=dense_losses_any(outputs,batch(),weights=LossWeights(shape_geometry_loss='log_spd'),
            device=torch.device('cpu'),center_radius_px=3.)
    losses['total'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in decoder.parameters())
