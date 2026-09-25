import unittest
import numpy as np
from preprocessing.utils.segmentation_storage import (pack_overlap_masks, validate_overlap_masks,
    crop_overlap_masks, decode_overlap_mask)


class SegmentationStorageTests(unittest.TestCase):
    def row(self):
        mask=np.zeros((9,11),bool);mask[1:8,2:10]=True
        return dict(sample=1,band=0,source_id=10,parent_source_id=None,
                    x0=12,y0=13,mask=mask,weight=.25)

    def test_pack_non_byte_aligned_and_empty(self):
        row=self.row();packed=pack_overlap_masks([row])
        validate_overlap_masks(packed,2,1,32,32)
        meta,mask=decode_overlap_mask(packed,0)
        np.testing.assert_array_equal(mask,row['mask'])
        self.assertEqual(len(packed['data']),13)
        validate_overlap_masks(pack_overlap_masks([]),1,1,32,32)

    def test_crop_parent_offset_and_invalid(self):
        row=self.row();valid=np.ones((16,16),bool);valid[0,0]=False
        cut=crop_overlap_masks([row],16,16,16,valid)
        self.assertEqual(len(cut),1);self.assertEqual((cut[0]['x0'],cut[0]['y0']),(0,0))
        expected=row['mask'][3:,4:].copy();expected[0,0]=False
        np.testing.assert_array_equal(cut[0]['mask'],expected)
        self.assertTrue(row['mask'][3,4])

    def test_reject_invalid_sidecar_layout(self):
        for field,value in [('offsets',np.array([0,1])),('weights',np.array([np.nan])),
                            ('data',np.zeros(13,np.float32))]:
            packed=pack_overlap_masks([self.row()]);packed[field]=value
            with self.assertRaises(ValueError):validate_overlap_masks(packed,2,1,32,32)
        packed=pack_overlap_masks([self.row(),self.row()])
        with self.assertRaises(ValueError):validate_overlap_masks(packed,2,1,32,32)

if __name__=='__main__':unittest.main()
