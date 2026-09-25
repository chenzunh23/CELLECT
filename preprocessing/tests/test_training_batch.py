"""Cross-parent sky assembly and production tiling regression checks."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from preprocessing.utils.background_tiles import assemble
from preprocessing.utils.npz_window import mask_window
from preprocessing.utils.inputs import make_tile_specs

class TrainingBatchTests(unittest.TestCase):
    def test_four_parent_background_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=[]
            for i,(x,y) in enumerate([(0,0),(4,0),(0,4),(4,4)]):
                p=Path(tmp)/f'{i}.npz';a=np.full((4,4),i%2==0,bool)
                np.savez_compressed(p,background_mask=a)
                rows.append(dict(origin_xy=[x,y],shape_yx=[4,4],mask=str(p)))
            out,meta=assemble(rows,2,2,4,4)
            self.assertEqual(len(meta['paths']),4)
            np.testing.assert_array_equal(out,np.tile([True,True,False,False],(4,1)))
            rows.append(dict(origin_xy=[2,2],shape_yx=[4,4],mask=rows[1]['mask']))
            out,meta=assemble(rows,2,2,4,4)
            self.assertFalse(out.any());self.assertEqual(meta['overlap_pixels'],16)
            out,_=assemble(rows,-3,-3,2,2);self.assertFalse(out.any())

    def test_quality_window_matches_full_without_large_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'quality.npz';a=np.random.default_rng(0).integers(0,2,(301,503),dtype=np.uint8)
            np.savez_compressed(p,bad=a)
            np.testing.assert_array_equal(mask_window(p,'bad',a.shape,30,70,209,180),a[70:250,30:239].astype(bool))
            with self.assertRaises(ValueError):mask_window(p,'bad',(99,99),0,0,8,8)

    def test_grid_counts_and_extent(self):
        for n,expected in [(4200,121),(4096,100)]:
            specs=make_tile_specs(parent_origin=(0,0),image_shape=(n,n),tile_size=512,stride=368,compare_origin=None)
            self.assertEqual(len(specs),expected)
            self.assertEqual(max(s.x1 for s in specs),n)
        specs=make_tile_specs(parent_origin=(0,0),image_shape=(4096,4096),tile_size=512,stride=512,compare_origin=None)
        self.assertEqual(len(specs),64)

if __name__=='__main__':unittest.main()
