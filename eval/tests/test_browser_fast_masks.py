import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image
from eval.datasets.parallel_load import ParallelLoad
from eval.datasets.array_cache import ArrayCache
from eval.datasets.jwst import JwstNircamAccess
from eval.datasets.base import TileRow
from eval.hsctiles.browser_masks import decode_masks, overlay_masks, MASK_ALPHA
from eval.hsctiles.browser_core import BrowserState, _display_input_uint8
from eval.tests.test_jwst_fields import header


class FakeDecoder:
    def __init__(self): self.batch_sizes=[]
    def forward_sam_masks(self, embeds, indices, points, boxes, *, chunk_size, output_size, **kw):
        self.batch_sizes.append(len(points))
        h,w=output_size
        out=torch.full((len(points),1,h,w),-1.)
        for i,point in enumerate(points):
            x,y=point.to(torch.int64)
            out[i,0,y-2:y+3,x-2:x+3]=1.
        return out,torch.ones((len(points),1))


class FastFitsAndMasksTests(unittest.TestCase):
    def test_cache_is_bounded(self):
        cache=ArrayCache(10)
        cache.put('a',np.zeros(6,np.uint8));cache.put('b',np.zeros(6,np.uint8))
        self.assertIsNone(cache.get('a'));self.assertEqual(cache.bytes,6)
        cache.put('big',np.zeros(20,np.uint8));self.assertEqual(cache.bytes,6)

    def test_tan_fast_projection_and_sample_match_astropy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);field=root/'field';field.mkdir()
            rng=np.random.default_rng(7)
            h1=header(128);h1['CRVAL1']=359.999;h1['CRVAL2']=-73.
            h2=h1.copy();h2['CRVAL1']=.0001;h2['CRVAL2']=-72.99995
            h2['PC1_1']=.999;h2['PC1_2']=.03;h2['PC2_1']=-.03;h2['PC2_2']=.999
            h2['CDELT1']*=1.01
            data=rng.normal(size=(128,128)).astype(np.float32)
            fits.PrimaryHDU(data,h1).writeto(field/'a_f090w_test.fits')
            fits.PrimaryHDU(data,h2).writeto(field/'a_f444w_test.fits')
            a=JwstNircamAccess(root,tile_size=32);tile=TileRow(0,'x001_y001',32,32,64,64)
            x,y=a._world_to_band_pixels('f090w','f444w','field',tile)
            yy,xx=np.mgrid[32:64,32:64]
            sx,sy=WCS(h2).celestial.world_to_pixel_values(*WCS(h1).celestial.pixel_to_world_values(xx,yy))
            np.testing.assert_allclose(x,sx,rtol=0,atol=1e-6)
            np.testing.assert_allclose(y,sy,rtol=0,atol=1e-6)
            expected=a._bilinear_sample(data,sx,sy)
            actual=a._sample_tile('f444w','field','f090w',tile)
            np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-5,equal_nan=True)
            with patch.object(a,'_read_section',side_effect=AssertionError('cache miss')):
                self.assertIs(a._sample_tile('f444w','field','f090w',tile),actual)
            # Distortion/non-TAN falls back rather than using approximate geometry.
            a._wcs_cache[a.image_file('f444w','field')].wcs.ctype=['RA---SIN','DEC--SIN']
            a._grid_relation.clear()
            self.assertIsNone(a._tan_grid_transform('f090w','f444w','field'))

    def test_parent_statistics_disk_cache_reused_and_invalidated(self):
        from eval.datasets.jwst_fields import CosmosAccess
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            image=root/'image.fits';quality=root/'quality.npz'
            data=np.random.default_rng(3).normal(size=(32,32)).astype(np.float32)
            fits.PrimaryHDU(data,header()).writeto(image)
            np.savez(quality,bad=np.zeros_like(data,dtype=bool))
            (root/'manifest.json').write_text(json.dumps(dict(items=[dict(image_fits=str(image),
                quality_mask_npz=str(quality),sample_name='test',proposal=1727,pointing=19,band='F444W')])))
            with patch('pathlib.Path.home',return_value=root):
                a=CosmosAccess(root)
                with patch.object(a,'read_window',return_value=(data,np.ones_like(data,bool),header())) as read:
                    params=a.scaling_parameters('f444w','p1727_P0019',0,0)
                    self.assertEqual(read.call_count,1)
                b=CosmosAccess(root)
                with patch.object(b,'read_window',side_effect=AssertionError('refitted unchanged parent')):
                    self.assertEqual(b.scaling_parameters('f444w','p1727_P0019',0,0),params)
                old=b._scaling_cache_file('f444w','p1727_P0019',0,0)
                import os
                stamp=image.stat();os.utime(image,ns=(stamp.st_atime_ns,stamp.st_mtime_ns+1000000000))
                self.assertNotEqual(old,b._scaling_cache_file('f444w','p1727_P0019',0,0))

    def test_uint16_memmap_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);field=root/'field';field.mkdir()
            data=np.arange(32*32,dtype=np.uint16).reshape(32,32)
            path=field/'a_f444w_test.fits';fits.PrimaryHDU(data,header()).writeto(path)
            a=JwstNircamAccess(root)
            np.testing.assert_array_equal(a._read_section(path,slice(5,9),slice(7,12)),data[5:9,7:12])

    def test_mask_orientation_overlap_boxes_chunking_and_filter(self):
        decoder=FakeDecoder();rows=[dict(x=6.,y=5.,major=2.,minor=1.,theta=0.) for _ in range(3)]
        masks=decode_masks(decoder,{'image_embeddings':torch.zeros(1,1,2,2)},rows,
            device=torch.device('cpu'),width=16,height=16,chunk_size=2)
        self.assertEqual(decoder.batch_sizes,[2,1])
        self.assertEqual(masks['packed'].size,12) # 3 cropped 5x5 masks, 4 bytes each
        out=overlay_masks(np.zeros((16,16,3),np.uint8),masks)
        np.testing.assert_array_equal(out[15-5,6],[0,round(255*MASK_ALPHA),0])
        np.testing.assert_array_equal(out[15-3,4],[0,255,0]) # tight box, flipped Y
        np.testing.assert_array_equal(out[5,6],[0,0,0])
        self.assertFalse(overlay_masks(np.zeros_like(out),masks,[]).any())

    def test_export_is_fixed_normal_zscale_and_cached_reads(self):
        state=object.__new__(BrowserState)
        state._frame_cache=ArrayCache(1024**2);state._image_load=ParallelLoad(4)
        state.access=Mock(spec=['read_frame']);state.access.read_frame.return_value=np.arange(256,dtype=np.float32).reshape(16,16)
        ref=Mock(token='a');state.ref_by_token={'a':ref}
        state.detect_rows_by_token={'a':[]}
        masks=decode_masks(FakeDecoder(),{},[],device=torch.device('cpu'),width=16,height=16)
        state._masks_for_token=lambda token:masks
        normal=state.mask_png('a')
        # The displayed view may be inverted/logarithmic without changing export.
        view=state.image_png('a',detect=True,show_shape=False,show_masks=True,invert_background=True,display_scaling='log')
        self.assertNotEqual(view,normal)
        self.assertEqual(state.mask_png('a'),normal)
        pixels=np.array(Image.open(io.BytesIO(normal)))
        np.testing.assert_array_equal(pixels,_display_input_uint8(state.access.read_frame.return_value,display_scaling='zscale'))
        self.assertEqual(state.access.read_frame.call_count,1)

    def test_empty_selection_does_not_scan_whole_field(self):
        a=JwstNircamAccess(Path('/missing'))
        a._active_bands_by_patch['p']=('f444w',)
        a._selected_tiles_by_key[('p',('f444w',))]=[]
        with patch.object(a,'_tiles_for_bands',side_effect=AssertionError('full scan')):
            self.assertEqual(a.tiles('f444w','p'),[])


if __name__=='__main__': unittest.main()
