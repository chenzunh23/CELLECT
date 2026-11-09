"""Dense confidence survives NMS rejection, display changes and cache eviction."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from eval.datasets.array_cache import ArrayCache
from eval.hsctiles import browser_core
from eval.hsctiles.browser_confidence import confidence_maps, confidence_overlay_uint8
from eval.hsctiles.browser_core import _display_input_uint8
from eval.tests.test_browser_parallel_export import make_state


class ConfidenceTests(unittest.TestCase):
    def test_all_positive_levels_saved_before_detection_threshold_and_nms(self):
        logits = torch.zeros(1, 5, 3, 5)
        for level in range(5):
            logits[0, level, 0, level] = 8
        valid = np.ones((3,5), bool)
        valid[0,4] = False
        maps = confidence_maps({'confidence': logits}, width=5, height=3,
                               valid_mask=valid, score_mode='ordinal_expectation')
        np.testing.assert_array_equal(maps['labels'][0], [0,1,2,3,0])
        np.testing.assert_array_equal(maps['logits'], logits[0].numpy())
        self.assertTrue(np.isnan(maps['score'][0,4]))
        self.assertLess(maps['score'][0,1], 2)  # Visible despite the proposal threshold.
        raw = confidence_maps({'confidence': logits}, width=5, height=3,
                              valid_mask=valid, score_mode='raw')
        np.testing.assert_array_equal(raw['logits'], logits[0].numpy())

    def test_overlay_keeps_class_zero_and_uses_lower_origin_map(self):
        background = np.full((3,5,3), 60, np.uint8)
        labels = np.zeros((3,5), np.uint8);labels[0,1] = 1;labels[2,3] = 4
        result = confidence_overlay_uint8(background, labels)
        np.testing.assert_array_equal(result[0,1], background[0,1])
        self.assertGreater(result[2,1,2], result[2,1,0])  # Blue bottom row.
        self.assertGreater(result[0,3,0], result[0,3,2])  # Red top row.

    def test_heatmap_overlay_is_continuous_and_keeps_weak_values_and_invalid_support(self):
        background=np.full((2,4,3),60,np.uint8)
        probability=np.array([[.05,.08,0.,np.nan],[.51,.59,.9,1.]],np.float32)
        result=confidence_overlay_uint8(background,probability)
        np.testing.assert_array_equal(result[1,2:],background[1,2:])
        self.assertFalse(np.array_equal(result[1,0],background[1,0]))  # <.1 remains visible.
        self.assertFalse(np.array_equal(result[1,0],result[1,1]))
        self.assertFalse(np.array_equal(result[0,0],result[0,1]))  # Same old display bin.
        self.assertGreater(result[0,3,0],result[0,3,2])

    def test_heatmap_cache_disk_fallback_and_export_preserve_continuous_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=make_state(tmp,1);ref=state.refs[0]
            state.make_masks=False;state.confidence_score='heatmap';state.confidence_threshold=.5
            state.detect_rows_by_token[ref.token]=[]
            probability=torch.full((1,1,32,32),.001)
            probability[0,0,5,5]=.05;probability[0,0,5,6]=.08;probability[0,0,6,5]=.8
            valid=np.ones((32,32),bool);valid[0,0]=False
            state._store_confidence({'confidence':torch.logit(probability)},ref,valid)
            expected=probability[0,0].numpy().copy();expected[~valid]=np.nan
            self.assertEqual(state.proposal_mode,'heatmap')
            np.testing.assert_allclose(state._confidence_for_token(ref.token),expected,rtol=1e-6)
            state._confidence_cache=ArrayCache(0)
            np.testing.assert_allclose(state._confidence_for_token(ref.token),expected,rtol=1e-6)
            with np.load(state.confidence_paths_by_token[ref.token]) as data:
                self.assertEqual(str(data['proposal_mode']),'heatmap')
                self.assertEqual(str(data['score_definition']),'sigmoid(logits[0])')
                self.assertEqual(data['labels'][5,5],0)  # Browser does not use these old bins.
            state._detect_tile_slots=lambda *args,**kwargs:dict(n_images=0,n_detections=0)
            with patch.object(browser_core,'infer_cellect',side_effect=AssertionError('Unexpected model call')):
                normal=state.image_png(ref.token,detect=True,show_shape=False,show_confidence=True,display_scaling='zscale')
                changed=state.image_png(ref.token,detect=True,show_shape=False,show_confidence=True,display_scaling='log')
                self.assertNotEqual(normal,changed)
                state.export_selected(workers=1,confidence_view={'display_scaling':'log'})
            row=json.loads((state.export_dir/'export_manifest.jsonl').read_text())
            self.assertEqual(Path(row['confidence_png_path']).read_bytes(),changed)
            self.assertEqual(row['confidence_display_kind'],'heatmap')
            self.assertEqual(row['confidence_classes'],{})
            self.assertEqual(row['confidence_value_range'],[0.,1.])

    def test_earlier_heatmap_npz_without_mode_metadata_is_recognized(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=make_state(tmp,1);ref=state.refs[0]
            path=Path(tmp)/'old_confidence.npz'
            score=np.full((32,32),.07,np.float32)
            np.savez_compressed(path,logits=np.full((1,32,32),-2.5866893,np.float32),
                labels=np.zeros((32,32),np.uint8),score=score)
            state.confidence_paths_by_token[ref.token]=path
            np.testing.assert_array_equal(state._confidence_for_token(ref.token),score)

    def test_detection_writes_confidence_even_without_masks_or_centers(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = make_state(tmp, 1);ref = state.refs[0]
            state.device_name='cpu';state.amp='none';state.center_refinement='none';state.center_refinement_radius=1
            state.detect_png_by_token={};state.input_shape_png_by_token={};state.warnings=[]
            state._load_model=lambda bands:(None,{})
            state._prepare_detection_inputs=lambda refs:[(state._raw_image_for_ref(r),state._scaled_input_for_ref(r)) for r in refs]
            state._store_masks=lambda *args:None
            logits=torch.zeros(1,5,32,32);logits[0,1,5:7,5:7]=8
            state._detect_tile_slots=browser_core.BrowserState._detect_tile_slots.__get__(state)
            with patch.object(browser_core,'infer_cellect',return_value={}), \
                 patch.object(browser_core,'select_band_outputs',return_value={'confidence':logits}), \
                 patch.object(browser_core,'detection_rows',return_value=[]):
                result=state._detect_tile_slots(list(state._refs_by_tile_slot()))
            self.assertEqual(result['n_images'],1)
            self.assertEqual(result['n_detections'],0)
            with np.load(state.confidence_paths_by_token[ref.token]) as stored:
                self.assertEqual(stored['logits'].dtype,np.float32)
                self.assertEqual(stored['labels'][5,5],1)

    def test_scaling_export_and_disk_fallback_do_not_rerun_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=make_state(tmp,1);ref=state.refs[0]
            refs_by_key=state._refs_by_tile_slot()
            state._detect_tile_slots(list(refs_by_key),refs_by_key,mask_sink={})
            # Retain the file but force confidence RAM-cache eviction.
            state._confidence_cache=ArrayCache(0)
            with patch.object(browser_core,'infer_cellect',side_effect=AssertionError('Unexpected model call')):
                a=state.image_png(ref.token,detect=True,show_shape=False,show_confidence=True,display_scaling='zscale')
                b=state.image_png(ref.token,detect=True,show_shape=False,show_confidence=True,display_scaling='log',invert_background=True)
                self.assertNotEqual(a,b)
                state.export_selected(workers=1,confidence_view={'display_scaling':'log','invert_background':True})
            row=json.loads((state.export_dir/'export_manifest.jsonl').read_text())
            self.assertTrue(Path(row['confidence_npz_path']).is_file())
            self.assertEqual(Path(row['confidence_png_path']).read_bytes(),b)
            self.assertEqual(row['confidence_view']['display_scaling'],'log')
            labels=state._confidence_for_token(ref.token)
            expected=confidence_overlay_uint8(_display_input_uint8(state._raw_image_for_ref(ref),display_scaling='zscale'),labels)
            np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(a))),expected)

    def test_no_png_still_exports_raw_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=make_state(tmp,1)
            state.export_selected(write_png=False)
            row=json.loads((state.export_dir/'export_manifest.jsonl').read_text())
            self.assertTrue(Path(row['confidence_npz_path']).is_file())
            self.assertEqual(row['confidence_png_path'],'')


if __name__=='__main__':unittest.main()
