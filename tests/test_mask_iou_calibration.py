"""GT-IoU calibration, detached targets and inference-selection regressions."""
from types import SimpleNamespace

import torch

from sam_backbone.losses import _sam_mask_loss_for_prompts
from sam_backbone.mask_metrics import GROUP_STAT_KEYS, finish_group_metrics
from sam_backbone.mask_supervision import mask_iou_targets


def weights(**kwargs):
    base = dict(mask_dice=0., mask_bce=0., mask_supervision_weight=1.,
        mask_centroid=0., mask_outside=0., mask_min_area=0., mask_max_area=0.,
        mask_pred_iou=1., mask_stability=0., mask_max_area_ratio=.8,
        mask_selection='pred_iou', mask_multimask=True, mask_prompt_chunk_size=8,
        mask_prompt_center_only=True, mask_outside_kron_scale=1.5)
    return SimpleNamespace(**(base | kwargs))


def prompts(n, rows):
    return dict(batch_indices=torch.zeros(n, dtype=torch.long),
        centers=torch.tensor([[1., 1.]]*n), prompt_shapes=torch.tensor([[2., 2., 0.]]*n),
        target_shapes=torch.tensor([[2., 2., 0.]]*n), weights=torch.ones(n),
        source_kind=torch.tensor([2]+[1]*(n-1)), instance_masks=rows)


class CandidateModel(torch.nn.Module):
    def __init__(self, logits, scores):
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())
        self.scores = torch.nn.Parameter(scores.clone())

    def forward_sam_masks(self, *args, **kwargs):
        return self.logits, self.scores


def run(model, p, valid, **kwargs):
    return _sam_mask_loss_for_prompts(model,
        dict(confidence=torch.zeros(1), image_embeddings=torch.zeros(1)), p,
        weights=weights(**kwargs), image_hw=valid.shape[-2:], ellipse_sigma=1.,
        batch=dict(band_valid_mask=valid), return_selection=True)


def test_original_pixel_iou_clips_masks_and_excludes_unobserved_pixels():
    logits = torch.full((2, 3, 4, 4), -8., requires_grad=True)
    with torch.no_grad():
        logits[0, 0] = 8.
        logits[0, 1, :2, :2] = 8.
    row = dict(x0=-1, y0=0, mask=torch.ones(2, 3, dtype=torch.bool), weight=.25)
    p = prompts(2, [row, None])
    valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    valid[..., 0, 0] = False
    truth = mask_iou_targets(logits, p, torch.tensor([True, False]),
                             image_hw=(4, 4), batch=dict(band_valid_mask=valid), chunk_size=1)
    torch.testing.assert_close(truth, torch.tensor([[.2, 1., 0.], [0., 0., 0.]]))
    assert not truth.requires_grad


def test_full_resolution_interpolation_keeps_one_pixel_gt():
    logits = torch.tensor([[[[8., -8.], [-8., -8.]]]], requires_grad=True)
    p = prompts(1, [dict(x0=0, y0=0, mask=torch.ones(1, 1, dtype=torch.bool), weight=1.)])
    truth = mask_iou_targets(logits, p, torch.tensor([True]), image_hw=(4, 4))
    # Bilinear logits at image resolution produce four positive pixels.
    torch.testing.assert_close(truth, torch.tensor([[.25]]))
    valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    valid[..., 1, 1] = False
    truth = mask_iou_targets(logits, p, torch.tensor([True]), image_hw=(4, 4),
                             batch=dict(band_valid_mask=valid))
    torch.testing.assert_close(truth, torch.tensor([[1/3]]))


def test_all_candidates_calibrate_above_old_threshold_without_selecting_by_gt():
    logits = torch.full((2, 3, 4, 4), -8.)
    logits[0, 0] = 8.
    logits[0, 1, :2, :2] = 8.
    scores = torch.tensor([[.95, .9, .1], [1.5, .1, .5]])
    model = CandidateModel(logits, scores)
    row = dict(x0=-1, y0=0, mask=torch.ones(2, 3, dtype=torch.bool), weight=.25)
    p = prompts(2, [row, None])
    valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    valid[..., 0, 0] = False
    out = run(model, p, valid)
    # Selection remains predicted-IoU even when a different candidate has GT IoU 1.
    assert out['selected_indices'].tolist() == [0, 0]
    expected = ((scores[0]-torch.tensor([.2, 1., 0.]))**2).mean()*.25
    torch.testing.assert_close(out['pred_iou'], expected)
    torch.testing.assert_close(out['total'], out['gt_total']+out['pred_total'])
    assert out['pred_total'] == 0
    metrics = finish_group_metrics(dict(zip(GROUP_STAT_KEYS, out['group_stats'].tolist())), weights())
    assert abs(metrics['mask_gt_mean_pred_iou']-expected.item()) < 1e-7
    assert metrics['mask_pred_mean_pred_iou'] == 0
    out['total'].backward()
    assert model.scores.grad[0, 0] > 0  # Overconfident large/wrong mask must decrease.
    assert model.scores.grad[0, 1] < 0  # Unselected correct mask must increase.
    assert model.scores.grad[0, 2] > 0  # Unselected empty mask must decrease.
    assert not model.scores.grad[1].any()  # No GT: do not calibrate toward zero.
    assert model.logits.grad is None or not model.logits.grad.any()


def test_unlabelled_quality_loss_is_zero_but_stays_in_ddp_graph():
    model = CandidateModel(torch.full((1, 3, 4, 4), -1.), torch.tensor([[.2, .5, 1.4]]))
    p = prompts(1, [None])
    out = run(model, p, torch.ones(1, 1, 4, 4, dtype=torch.bool))
    assert out['total'] == 0 and not out['iou_labelled'].any()
    out['total'].backward()
    assert model.scores.grad is not None and not model.scores.grad.any()


def test_unlabelled_prompts_do_not_dilute_calibration_and_legacy_threshold_is_unused():
    row = dict(x0=0, y0=0, mask=torch.ones(4, 4, dtype=torch.bool), weight=1.)
    valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    model = CandidateModel(torch.ones(1, 3, 4, 4), torch.tensor([[.9, .2, 1.4]]))
    one = run(model, prompts(1, [row]), valid, mask_pred_iou_thresh=.1)
    model2 = CandidateModel(torch.ones(3, 3, 4, 4), torch.tensor([[.9, .2, 1.4]]*3))
    many = run(model2, prompts(3, [row, None, None]), valid, mask_pred_iou_thresh=.99)
    torch.testing.assert_close(one['pred_iou'], many['pred_iou'])
    assert one['pred_iou'] > 0
