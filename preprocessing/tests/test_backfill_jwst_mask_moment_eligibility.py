"""JWST metadata-only writes, exact IDs and independent-window SNR gating."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from astro_train_zarr_data import PatchZarrReader
from direct_zarr_preprocessing.zarr_writer import ZarrArrayWriter
from preprocessing.tests.test_backfill_mask_moment_eligibility import fixture
from preprocessing.backfill_mask_moment_eligibility import ARRAY, POLICY
from preprocessing.backfill_jwst_mask_moment_eligibility import (
    build_flags, process_one, lookup, read_sparse, blocks_for, main,
)
from preprocessing.utils.mask_moments import JWST_POLICY


def attrs(store, **updates):
    p=store/'.zattrs';d=json.loads(p.read_text());d.update(updates);p.write_text(json.dumps(d))


def options(root, **updates):
    return dict(data_root=str(root),cache_dir=str(root/'cache'),replay_missing=False,
                replace_legacy=False,apply=False)|updates


def write_array(store,name,data):
    data=np.asarray(data)
    ZarrArrayWriter(store,name,shape=data.shape,chunks=data.shape,dtype=data.dtype).write_full(data)


def cosmos_fixture(root):
    store,labels=fixture(root)
    w=WCS(naxis=2);w.wcs.crpix=[1,1];w.wcs.cdelt=[-.03/3600,.03/3600]
    w.wcs.crval=[150,2];w.wcs.ctype=['RA---TAN','DEC--TAN']
    ra,dec=w.pixel_to_world_values(labels.geom_x+100,labels.geom_y+200)
    cat=root/'catalog.fits'
    phot=Table(dict(id=labels.source_ids,ra=ra,dec=dec,snr_f444w=[50.,6.,7.]))
    gal=Table(dict(kron2_a=np.full(3,.12),kron2_b=np.full(3,.06),theta_world=np.full(3,np.degrees(.3))))
    fits.HDUList([fits.PrimaryHDU(),fits.BinTableHDU(phot),*[fits.ImageHDU() for _ in range(5)],fits.BinTableHDU(gal)]).writeto(cat)
    attrs(store,dataset='cosmos',bands=['F444W'],snr_thresholds=dict(weak_below=7.),
          sky_wcs_header=w.to_header().tostring(sep='\n'),source_catalog=str(cat),sample_kind='grid')
    return store,labels


def abell_fixture(root, independent=False):
    store,labels=fixture(root)
    write_array(store,'tile_x0',np.array([0,24 if independent else 4]))
    write_array(store,'tile_y0',np.array([0,-5 if independent else 0]))
    audit=root/'abell/labels/F444W/p';audit.mkdir(parents=True)
    Table(dict(NUMBER=labels.source_ids,final_class=labels.label_classes,
        parent_x=labels.geom_x,parent_y=labels.geom_y,original_kron_a=labels.geom_major,
        original_kron_b=labels.geom_minor,remeasured_aperture_snr=[100.,6.,7.])).write(audit/'sources.fits')
    (audit/'complete.json').write_text(json.dumps(dict(signature={})))
    blocks=([dict(start=0,count=1,sample_kind='grid',coordinate_shift=[0,0]),
             dict(start=1,count=1,sample_kind='large_source',coordinate_shift=[20,-5],label_window=[20,-5,52,27])]
            if independent else [dict(start=0,count=2,sample_kind='grid',coordinate_shift=[0,0])])
    attrs(store,dataset='abell',bands=['F444W'],parent_id='p',patch='p',
          snr_thresholds=dict(weak_below=7.),sample_blocks=blocks)
    return store,labels,audit


class JWSTBackfillTests(unittest.TestCase):
    def test_cosmos_dry_run_uses_own_band_snr_and_changes_no_store_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);store,_=cosmos_fixture(root)
            before={str(p.relative_to(store)):p.read_bytes() for p in store.rglob('*') if p.is_file()}
            result=process_one((str(store),options(root)))
            self.assertEqual(result['status'],'dry_run_ok',result)
            self.assertEqual(result['eligible_clean'],2);self.assertEqual(result['eligible_weak'],2)
            after={str(p.relative_to(store)):p.read_bytes() for p in store.rglob('*') if p.is_file()}
            self.assertEqual(before,after)

    def test_cosmos_apply_preserves_images_sparse_arrays_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);store,_=cosmos_fixture(root)
            before={str(p.relative_to(store)):p.read_bytes() for p in store.rglob('*') if p.is_file()}
            result=process_one((str(store),options(root,apply=True)))
            self.assertEqual(result['status'],'written',result)
            reader=PatchZarrReader(store)
            np.testing.assert_array_equal(reader.read_full_small(ARRAY),[1,0,1,1,0,1])
            self.assertEqual(reader.attrs['mask_moment_policy'],JWST_POLICY)
            for name,value in before.items():
                if name!='.zattrs':self.assertEqual((store/name).read_bytes(),value)
            self.assertEqual(process_one((str(store),options(root,apply=True)))['status'],'already_done')

    def test_abell_grid_uses_saved_remeasurement_without_replay(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);store,_,_=abell_fixture(root)
            with patch('preprocessing.backfill_jwst_mask_moment_eligibility.replay_abell') as replay:
                flag,_,stats=build_flags(store,root,root/'cache',False)
                replay.assert_not_called()
            np.testing.assert_array_equal(flag,[1,0,1,1,0,1])
            self.assertEqual(stats['methods'],dict(parent_audit=1))

    def test_fits_masked_geometry_columns_are_converted_before_stacking(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);store,_,audit=abell_fixture(root)
            table=Table(Table.read(audit/'sources.fits'),masked=True)
            # Invalid geometry in a non-selected row must not corrupt numpy's
            # astropy MaskedColumn stack or shift the measured weak rows.
            for key in ['parent_x','original_kron_a','original_kron_b']:
                table[key].mask[0]=True
            table.write(audit/'sources.fits',overwrite=True)
            flag,_,_=build_flags(store,root,root/'cache',False)
            np.testing.assert_array_equal(flag,[1,0,1,1,0,1])

    def test_cli_refuses_generation_lock_without_writes(self):
        import fcntl
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);store,_=cosmos_fixture(root)
            (root/'logs').mkdir()
            with (root/'logs/generation.lock').open('w') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as exc:
                    main(['--root',str(root),'--apply'])
                self.assertEqual(exc.exception.code,2)
            self.assertFalse((store/ARRAY).exists())

    def test_independent_block_never_borrows_parent_snr(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);store,labels,_=abell_fixture(root,independent=True)
            with self.assertRaisesRegex(ValueError,'replay-missing'):
                build_flags(store,root,root/'cache',False)
            self.assertFalse((store/ARRAY).exists())
            measurement=dict(ids=labels.source_ids,classes=labels.label_classes,
                xy=np.c_[labels.geom_x,labels.geom_y],axes=np.c_[labels.geom_major,labels.geom_minor],
                theta=labels.geom_theta,snr=np.array([100.,9.,2.]))
            with patch('preprocessing.backfill_jwst_mask_moment_eligibility.replay_abell',return_value=(measurement,'replay')) as replay:
                flag,_,stats=build_flags(store,root,root/'cache',True)
                self.assertEqual(replay.call_args.args[1]['sample_kind'],'large_source')
            np.testing.assert_array_equal(flag,[1,0,1,1,1,0])
            self.assertEqual(stats['methods'],dict(parent_audit=1,replay=1))

    def test_mismatched_audit_fails_without_writes(self):
        for field in ['parent_x','original_kron_a','final_class','NUMBER']:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as td:
                root=Path(td);store,_,audit=abell_fixture(root)
                table=Table.read(audit/'sources.fits');table[field]+=1;table.write(audit/'sources.fits',overwrite=True)
                result=process_one((str(store),options(root,apply=True)))
                self.assertEqual(result['status'],'error',result);self.assertFalse((store/ARRAY).exists())

    def test_missing_numeric_snr_in_known_psf_rows_remains_ineligible(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);store,labels,_=abell_fixture(root)
            attrs(store,jwst_star_sources=[dict(gaia_id=int(i)) for i in labels.source_ids[1:]])
            # No audit is needed for clean and known, SNR-less PSF model rows.
            flag,_,stats=build_flags(store,root,root/'cache',False)
            np.testing.assert_array_equal(flag,[1,0,0,1,0,0]);self.assertEqual(stats['methods'],{})

    def test_legacy_replacement_is_explicit_and_recoverable(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);store,_=cosmos_fixture(root)
            attrs(store,mask_moment_policy=POLICY)
            write_array(store,ARRAY,np.zeros(6,np.uint8))
            self.assertEqual(process_one((str(store),options(root,apply=True)))['status'],'error')
            result=process_one((str(store),options(root,apply=True,replace_legacy=True)))
            self.assertEqual(result['status'],'written',result)
            self.assertTrue((store/('.'+ARRAY+'.before_jwst_moment_backfill')/'.zarray').is_file())

    def test_sparse_validation_and_exact_int64_lookup(self):
        ids=np.array([2**60+1,2**60+2],np.int64)
        np.testing.assert_array_equal(lookup(ids[::-1],ids),[1,0])
        with self.assertRaises(ValueError):lookup(ids.astype(float),ids)
        with self.assertRaises(ValueError):lookup(ids,np.array([2**60+1]*2,np.int64))
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);store,_=cosmos_fixture(root)
            write_array(store,'shape_source_offsets',np.array([[0,2],[3,6]]))
            with self.assertRaisesRegex(ValueError,'contiguous'):read_sparse(PatchZarrReader(store))
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);store,_=cosmos_fixture(root)
            attrs(store,sample_kind='mixed')
            with self.assertRaisesRegex(ValueError,'sample_blocks'):blocks_for(PatchZarrReader(store))


if __name__=='__main__':unittest.main()
