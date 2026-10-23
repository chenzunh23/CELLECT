"""The COSMOS threshold=2 recipe must not relax the other sky checks."""
import json
from pathlib import Path
import tempfile
import unittest
from preprocessing.zarr_preflight import background_receipt,document
from preprocessing.utils.jwst_background_local import METHOD


class BackgroundPreflightTests(unittest.TestCase):
    def test_cosmos_two_and_existing_one_point_five(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'science.fits';source.write_bytes(b'fixture')
            info=source.stat();receipt=Path(folder)/'complete.json'
            config=dict(background_mesh=128,background_filter_size=3,
                minarea_native_pixels=[1],factors=[1],smoothing_mode='psf',
                smoothing_sigma_floor_pixels=1.,grow_factor=2.4)
            def save(threshold,**override):
                doc=dict(status='done',signature=dict(method=METHOD,
                    source=dict(path=str(source),bytes=info.st_size,mtime_ns=info.st_mtime_ns),
                    config={**config,'thresholds_sigma':[threshold],**override}))
                receipt.write_text(json.dumps(doc))
                document.cache_clear()
            save(1.5)
            background_receipt(receipt,source)
            background_receipt(receipt,source,allowed_thresholds=(1.5,2.))
            save(2.)
            background_receipt(receipt,source,allowed_thresholds=(1.5,2.))
            with self.assertRaises(ValueError):background_receipt(receipt,source)
            save(2.,minarea_native_pixels=[16])
            with self.assertRaises(ValueError):background_receipt(receipt,source,allowed_thresholds=(1.5,2.))
            save(3.)
            with self.assertRaises(ValueError):background_receipt(receipt,source,allowed_thresholds=(1.5,2.))


if __name__=='__main__':unittest.main()
