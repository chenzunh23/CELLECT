import csv,json,tempfile,unittest
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from preprocessing.utils.frozen_sources import load_frozen_sources,prepare_frozen_variant


class FrozenSourceTests(unittest.TestCase):
    def test_reuse_and_reject_old_background(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);w=WCS(naxis=2);w.wcs.ctype=['RA---TAN','DEC--TAN']
            w.wcs.crval=[150,2];w.wcs.crpix=[1,1];w.wcs.cdelt=[-.03/3600,.03/3600]
            header=w.to_header();header['BUNIT']='MJy/sr'
            raw=np.random.default_rng(4).normal(size=(64,64)).astype(np.float32)
            fits.writeto(root/'old.fits',raw,header);fits.writeto(root/'new.fits',raw,header)
            (root/'summary.json').write_text(json.dumps(dict(raw_fits=str(root/'old.fits'),origin=[10,10])))
            with (root/'sources.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=['id','x','y','a','b','theta_deg','mag','final','reason'])
                writer.writeheader();writer.writerow(dict(id=1,x=20,y=20,a=4,b=2,theta_deg=30,mag=23,final='ignore',reason='keep'))
            (root/'inserted.csv').write_text('id,x,y\n-1,22,22\n')
            target=w.slice((slice(5,50),slice(5,50)))
            frozen=load_frozen_sources(root,target)
            np.testing.assert_allclose(frozen.geometry.x,[25],atol=1e-5)
            np.testing.assert_allclose(frozen.geometry.major,[4],rtol=1e-5)
            np.testing.assert_array_equal(frozen.labels.source_class,[3])
            np.testing.assert_allclose(frozen.inserted_xy,[[27,27]],atol=1e-5)
            bg=root/'bg';bg.mkdir();np.savez_compressed(bg/'background_mask.npz',background_mask=np.ones((64,64),bool))
            info=dict(status='ok',input=str(root/'old.fits'),origin_xy=[0,0],shape_yx=[64,64])
            (bg/'summary.json').write_text(json.dumps(info))
            with self.assertRaises(ValueError):
                prepare_frozen_variant(root,root/'new.fits',bg/'background_mask.npz',shape=(32,32))
            info['input']=str(root/'new.fits');(bg/'summary.json').write_text(json.dumps(info))
            out=prepare_frozen_variant(root,root/'new.fits',bg/'background_mask.npz',shape=(32,32))
            self.assertEqual(out['bright_region_mask'].shape,(32,32))
            self.assertEqual(out['sources'].labels.source_class[0],3)


if __name__=='__main__':unittest.main()
