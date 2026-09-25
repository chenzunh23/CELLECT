"""Connected level-4 centroids without changing legacy NMS/merge defaults."""
import numpy as np
import torch
from astro_train_ops import (
    _centroid_level4_peaks, _merge_close_centers_by_score,
    detect_centers, detect_centers_tensors, detect_centers_with_scores,
)


def make_outputs():
    logits = torch.zeros(1, 5, 24, 24)
    logits[:, 0] = 1.
    return {"confidence": logits}


def test_plateau_centroid_and_public_apis():
    out = make_outputs()
    out['confidence'][0, 4, 6:8, 8:10] = 5.
    kwargs = dict(confidence_score='raw', threshold=2.)
    assert np.allclose(detect_centers(out, **kwargs)[0], [[8.5, 6.5]])
    assert np.allclose(detect_centers_tensors(out, **kwargs)[0], [[8.5, 6.5]])
    result = detect_centers_with_scores(out, **kwargs)[0]
    assert np.allclose(result['xy'], [[8.5, 6.5]])
    assert np.allclose(result['score'], [5.])


def test_centroid_uses_whole_component_not_only_maxima():
    out = make_outputs()
    out['confidence'][0, 4, 6:8, 8:10] = 3.
    out['confidence'][0, 4, 6, 8] = 9.
    result = detect_centers_with_scores(out, confidence_score='raw', threshold=2.)[0]
    assert np.allclose(result['xy'], [[8.5, 6.5]])
    assert np.allclose(result['score'], [9.])


def test_eight_connectivity_and_threshold_gate():
    out = make_outputs()
    out['confidence'][0, 4, 6, 8] = 5.
    out['confidence'][0, 4, 7, 9] = 4.
    assert np.allclose(detect_centers(out, confidence_score='raw', threshold=2.)[0], [[8.5, 6.5]])
    assert detect_centers(out, confidence_score='raw', threshold=6.)[0].shape == (0, 2)


def test_disjoint_raw_candidates_keep_old_window_default():
    out = make_outputs()
    out['confidence'][0, 4, 6, 8] = 5.
    out['confidence'][0, 4, 6, 10] = 4.
    kwargs = dict(confidence_score='raw', threshold=2.)
    assert np.allclose(detect_centers(out, **kwargs)[0], [[8., 6.], [10., 6.]])
    # The pre-existing explicit distance-merge opt-in still works.
    assert np.allclose(detect_centers(out, merge_close_centers=True, **kwargs)[0], [[8., 6.]])


def test_cellect_does_not_gain_three_pixel_distance_merge():
    out = make_outputs()
    out['confidence'][0, 4, 6, 8] = 20.
    out['confidence'][0, 4, 6, 10] = 20.
    # Equal nearby peaks can survive the legacy 3x3 CELLECT max filter.
    result = detect_centers(out, confidence_score='cellect', threshold=2.)[0]
    assert len(result) >= 2
    assert len(detect_centers(out, confidence_score='cellect', threshold=2.,
                              merge_close_centers=True)[0]) < len(result)


def test_ordinal_default_merge_and_explicit_mask_prompt_override():
    out = make_outputs()
    out['confidence'][0, 4, 6, 8] = 9.
    out['confidence'][0, 4, 6, 10] = 6.
    kwargs = dict(confidence_score='ordinal_expectation', threshold=3.)
    assert np.allclose(detect_centers(out, **kwargs)[0], [[8., 6.]])
    assert len(detect_centers(out, merge_close_centers=False, **kwargs)[0]) == 2


def test_legacy_distance_boundary_and_transitive_groups():
    xy = torch.tensor([[0., 0.], [2., 0.], [4., 0.], [7., 0.]])
    kept, scores = _merge_close_centers_by_score(xy, torch.tensor([1., 3., 2., 4.]), min_distance=3.)
    assert torch.equal(kept, torch.tensor([[2., 0.], [7., 0.]]))
    assert torch.equal(scores, torch.tensor([3., 4.]))


def test_empty_and_bfloat16_score_export():
    out = make_outputs()
    assert detect_centers(out, confidence_score='raw', threshold=2.)[0].shape == (0, 2)
    out['confidence'][0, 4, 6:8, 8:10] = 5.
    out['confidence'] = out['confidence'].to(torch.bfloat16)
    result = detect_centers_with_scores(out, confidence_score='raw', threshold=2.)[0]
    assert result['score'].dtype == np.float32
    assert np.allclose(result['xy'], [[8.5, 6.5]])
