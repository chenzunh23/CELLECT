"""Raw nesting, per-instance regularization, and overlap-only isolation."""
import unittest
import numpy as np
from scipy import ndimage
from preprocessing.labels import SourceLabels
from preprocessing.utils.segmentation import isolated_segmentation_targets


def fixture():
    seg=np.zeros((120,120),np.int32);seg[20:100,20:100]=100
    seg[40:46,40:46]=200;seg[70:76,70:76]=300
    return seg


def run(seg, ids=(100,200,300), classes=None, **kwargs):
    lab=SourceLabels(np.array(classes if classes is not None else [1]*len(ids)),np.full(len(ids),'test',object))
    return isolated_segmentation_targets(seg,np.array(ids),lab,**kwargs)


def kept(out):return {r['source_id'] for r in out.sources}


class NestedSegmentationTests(unittest.TestCase):
    def test_all_children_train_parent_filled_only_parent_sidecar(self):
        seg=fixture();original=seg.copy();out=run(seg)
        self.assertEqual(kept(out),{100,200,300})
        np.testing.assert_array_equal(seg,original)
        self.assertEqual(len(out.overlap_masks),1)
        parent=out.overlap_masks[0];self.assertEqual(parent['source_id'],100)
        self.assertTrue(parent['mask'][20:26,20:26].all())
        np.testing.assert_array_equal(ndimage.binary_fill_holes(parent['mask']),parent['mask'])
        self.assertFalse(run(seg,allow_nested=False).sources)

    def test_touching_siblings_and_parent_can_train(self):
        seg=fixture();seg[40:46,46:52]=400
        out=run(seg,ids=(100,200,300,400))
        self.assertEqual(kept(out),{100,200,300,400})
        self.assertEqual({r['source_id'] for r in out.overlap_masks},{100})

    def test_ineligible_unknown_child_rejects_parent_dropped_excluded(self):
        for classes in ([1,3,1],[1,4,1]):
            self.assertEqual(kept(run(fixture(),classes=classes)),{300})
        self.assertEqual(kept(run(fixture(),ids=(100,300))),{300})
        out=run(fixture(),classes=[1,0,1]);self.assertEqual(kept(out),{100,300})
        self.assertEqual(out.sources[0]['enclosed_source_ids'],[300])

    def test_child_independent_of_parent_class(self):
        for cls in (0,3,4):
            out=run(fixture(),classes=[cls,1,2]);self.assertEqual(kept(out),{200,300})
            self.assertFalse(out.overlap_masks)

    def test_nearby_or_touching_unknown_not_overlapping_is_allowed(self):
        seg=fixture();seg[20:25,100:105]=999
        self.assertEqual(kept(run(seg)),{100,200,300})
        self.assertEqual(kept(run(seg,clearance=8)),{200,300})

    def test_open_raw_boundary_not_repaired_into_containment(self):
        seg=fixture();seg[42,20:40]=0
        out=run(seg);byid={r['source_id']:r for r in out.sources}
        # Gaussian can close the raw channel, but must not manufacture a parent.
        self.assertNotIn(100,byid)
        if 200 in byid:self.assertIsNone(byid[200]['parent_source_id'])

    def test_dominant_component_cleanup(self):
        seg=fixture();seg[5,5]=100;out=run(seg)
        self.assertEqual(kept(out),{100,200,300});self.assertEqual(out.instance_ids[5,5],0)
        self.assertEqual(out.sources[0]['raw_components'],2)
        seg[2:12,2:12]=100
        self.assertEqual(kept(run(seg,main_fraction_min=.99)),{200,300})
        self.assertEqual(kept(run(seg)),{100,200,300})
        seg[2:19,2:30]=100
        self.assertEqual(kept(run(seg)),{200,300})

    def test_containment_uses_raw_even_when_outside_fragment_removed(self):
        seg=fixture();seg[50:65,50:65]=200;seg[40:46,40:46]=100
        seg[5,5]=200  # <5% fragment: removed, but raw containment must still fail.
        out=run(seg)
        self.assertNotIn(100,kept(out));self.assertNotIn(200,kept(out))
        self.assertIn(300,kept(out))

    def test_star_invalid_edge_parent_failure_does_not_reject_child(self):
        seg=fixture()
        for key in ['star_mask','valid_mask']:
            bad=np.zeros(seg.shape,bool);bad[20,30]=True
            self.assertEqual(kept(run(seg,**{key:bad if key=='star_mask' else ~bad})),{200,300})
        edge=np.zeros_like(seg);edge[:80,20:100]=100;edge[30:36,40:46]=200
        self.assertEqual(kept(run(edge,ids=(100,200))),{200})
        valid=np.ones(seg.shape,bool);valid[42,42]=False
        self.assertEqual(kept(run(seg,valid_mask=valid)),{300})

    def test_three_levels_bottom_up(self):
        seg=fixture();seg[40:65,40:65]=200;seg[50:55,50:55]=400
        out=run(seg,ids=(100,200,300,400));self.assertEqual(kept(out),{100,200,300,400})
        self.assertEqual({r['source_id'] for r in out.overlap_masks},{100,200})
        self.assertEqual({r['source_id']:r for r in out.sources}[400]['parent_source_id'],200)
        self.assertEqual(kept(run(seg,ids=(100,200,300,400),classes=[1,1,1,4])),{300})
        self.assertEqual(kept(run(seg,ids=(100,200,300,400),classes=[4,1,1,1])),{200,300,400})

    def test_overlap_with_processed_ignore_and_unknown_blocks(self):
        seg=fixture();seg[50:65,50:65]=200;seg[40:46,40:46]=100;seg[5,5]=200
        for cls in (1,2,3,4):
            self.assertNotIn(100,kept(run(seg,classes=[1,cls,1])))
        self.assertNotIn(100,kept(run(seg,ids=(100,300))))
        self.assertEqual(kept(run(seg,classes=[1,0,1])),{100,300})

    def test_small_holes_filled_without_gaussian(self):
        seg=np.zeros((40,40),np.int32);seg[12:17,12:17]=100;seg[14,14]=0;seg[30,30]=200
        out=run(seg,ids=(100,200));self.assertEqual(kept(out),{100,200})
        r={r['source_id']:r for r in out.sources}[100]
        self.assertEqual((r['area'],r['mask_area']),(24,25));self.assertFalse(r['gaussian_applied'])
        self.assertEqual(out.instance_ids[14,14],100);self.assertEqual(out.instance_ids[30,30],200)

    def test_raw_100_vs_101_threshold_and_no_integer_gaussian(self):
        seg=np.zeros((60,60),np.int32);seg[20:30,20:30]=100
        out=run(seg,ids=(100,));r=out.sources[0]
        self.assertFalse(r['gaussian_applied']);self.assertEqual(r['area'],100)
        np.testing.assert_array_equal(out.instance_ids,seg)
        seg[5,5]=100;out=run(seg,ids=(100,));r=out.sources[0]
        self.assertTrue(r['gaussian_applied']);self.assertEqual(r['area'],101)
        self.assertEqual(r['cleaned_area'],100);self.assertEqual(r['mask_area'],96)
        self.assertEqual(out.instance_ids[20,20],0)

    def test_thin_parent_raw_area_smaller_than_child_orders_by_nesting(self):
        seg=np.zeros((60,60),np.int32);seg[15:35,15:35]=100;seg[16:34,16:34]=200
        out=run(seg,ids=(100,200));self.assertEqual(kept(out),{100,200})
        byid={r['source_id']:r for r in out.sources}
        self.assertLess(byid[100]['area'],byid[200]['area'])
        self.assertEqual(byid[200]['parent_source_id'],100)
        self.assertEqual({r['source_id'] for r in out.overlap_masks},{100})
        self.assertEqual(out.instance_ids[25,25],200)

    def test_near_invalid_pixel_no_longer_requires_eight_pixel_margin(self):
        seg=np.zeros((40,40),np.int32);seg[15:20,15:20]=100
        valid=np.ones_like(seg,bool);valid[16,20]=False
        self.assertEqual(kept(run(seg,ids=(100,),valid_mask=valid)),{100})
        valid[16,19]=False;self.assertFalse(run(seg,ids=(100,),valid_mask=valid).sources)

if __name__=='__main__':unittest.main()
