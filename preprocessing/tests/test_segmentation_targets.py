import unittest
import numpy as np
from preprocessing.labels import SourceLabels
from preprocessing.utils.segmentation import isolated_segmentation_targets


class SegmentationTargetTests(unittest.TestCase):
    def test_native_wcs_unsigned_id_mapping(self):
        import tempfile
        from pathlib import Path
        from astropy.io import fits
        from astropy.table import Table
        from astropy.wcs import WCS
        from preprocessing.utils.segmentation import read_cosmos_segment_cutout
        w=WCS(naxis=2);w.wcs.ctype=['RA---TAN','DEC--TAN'];w.wcs.crval=[150,2]
        w.wcs.crpix=[1,1];w.wcs.cdelt=[-.03/3600,.03/3600]
        seg=np.zeros((20,20),np.uint32);seg[5:9,6:10]=7;seg[12,12]=99
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'segmentation_maps').mkdir();(root/'star_masks').mkdir()
            path=root/'catalog.fits'
            Table({'id':[1001], 'segment-id':[7], 'tile':['B3']}).write(path)
            fits.writeto(root/'segmentation_maps/detection_chi2pos_SWLW_B3_segmap_v1.3.fits',seg,w.to_header())
            fits.writeto(root/'star_masks/cosmos_web_starmask_jwst_B3.fits',np.zeros_like(seg),w.to_header())
            result,star,valid=read_cosmos_segment_cutout(w,seg.shape,path,chunk_rows=5)
            self.assertTrue(np.all(result[5:9,6:10]==1001))
            self.assertFalse(valid[12,12])
            self.assertFalse(star.any())

    def test_isolation_and_labels(self):
        seg=np.zeros((80,80),np.int32)
        seg[15:18,15:18]=101
        seg[30:33,30:33]=102
        seg[35:38,30:33]=103
        seg[50:53,50:53]=104
        seg[65:68,65:68]=105
        labels=SourceLabels(np.array([1,1,3,4,2]),np.full(5,'test',object))
        out=isolated_segmentation_targets(seg,np.arange(101,106),labels)
        self.assertEqual(set(np.unique(out.instance_ids)),{0,101,102,105})
        self.assertTrue(np.all(out.positive_weight[out.instance_ids==0]==0))
        np.testing.assert_array_equal(labels.source_class,[1,1,3,4,2])

    def test_star_holes_boundary_invalid_disconnected(self):
        seg=np.zeros((60,60),np.int32);seg[25:28,25:28]=1
        labels=SourceLabels(np.array([1]),np.array(['test'],object))
        star=np.zeros_like(seg,bool);star[20:35,20:35]=True;star[21:34,21:34]=False
        self.assertFalse(isolated_segmentation_targets(seg,[1],labels,star_mask=star).sources)
        valid=np.ones_like(seg,bool);valid[25,26]=False
        self.assertFalse(isolated_segmentation_targets(seg,[1],labels,valid_mask=valid).sources)
        seg[45,45]=1
        self.assertFalse(isolated_segmentation_targets(seg,[1],labels).sources)
        seg[:]=0;seg[0:3,0:3]=1
        self.assertFalse(isolated_segmentation_targets(seg,[1],labels).sources)


if __name__=='__main__':unittest.main()
