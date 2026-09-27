"""Packing preserves sparse labels, masks, and both distributed samplers."""
import tempfile,unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from astropy.io import fits
from preprocessing.tests.test_segmentation_zarr import labels,task
from preprocessing.build_image_level_zarr import write_classified_patch
from preprocessing.utils.parent_zarr import merge_parent_stores
from preprocessing.utils.inputs import TileSpec
from preprocessing.utils.segmentation_storage import decode_overlap_mask
from astro_train_data import CutoutRecord
from astro_train_zarr_data import PatchZarrReader,ZarrChunkLocalBatchSampler,ZarrChunkBatchIterableDataset,ZarrCutoutDataset

def only_names(batch):
 return batch["name"]


class ParentPackingTests(unittest.TestCase):
 def test_merge_and_sampling(self):
  with tempfile.TemporaryDirectory() as td,patch('preprocessing.build_image_level_zarr._scale_image_chw',side_effect=lambda image,task:image[None]):
   root=Path(td);lab=labels();lab.source_ids=np.array([100]);lab.label_classes=np.array([1]);lab.geom_x=lab.geom_y=np.array([8.]);lab.geom_major=lab.geom_minor=np.array([3.]);lab.geom_theta=np.zeros(1)
   lab.segmentation_ids=np.zeros((32,32),np.int32);lab.segmentation_ids[6:10,6:10]=100;lab.segmentation_weight=(lab.segmentation_ids>0).astype(np.float32)
   lab.segmentation_overlap_masks=[dict(source_id=100,parent_source_id=None,x0=5,y0=5,mask=np.ones((6,6),bool),weight=.5)]
   a=write_classified_patch(replace(task(root/'parts'),patch='parent',chunk_tiles=20),np.ones((32,32)),lab,tile_specs=[TileSpec(f'grid_{j}',0,0,16) for j in range(23)],provenance={'sample_kind':'grid'})
   b=write_classified_patch(replace(task(root/'parts'),patch='extra',chunk_tiles=20),np.ones((32,32))*7,lab,tile_specs=[TileSpec('large_id100',0,0,16)],provenance={'sample_kind':'large_source'})
   out=root/'parent.zarr';merge_parent_stores([(a['output'],(0,0)),(b['output'],(-7,33))],out,attrs={'patch':'parent'},chunk_tiles=20)
   rd=PatchZarrReader(out);self.assertEqual(rd.meta('images').chunks[0],20);self.assertEqual(rd.meta('images').shape[0],24)
   self.assertEqual(rd.read_full_small('tile_x0')[-1],-7);self.assertEqual(rd.read_full_small('tile_y0')[-1],33)
   for j in range(24):
    self.assertEqual(rd.read_first_axis('images',j).mean(),1 if j<23 else 7)
    for key in ['source_offsets','shape_source_offsets']:np.testing.assert_array_equal(rd.read_full_small(key)[j],[j,j+1])
   packed={k:rd.read_full_small('segmentation_overlap_'+k) for k in ('meta','data','offsets','weights')}
   row,mask=decode_overlap_mask(packed,23);self.assertEqual(row[0],23);self.assertTrue(mask.all())
   records=[CutoutRecord(name=str(j),image_paths=(f'zarr://{out}#{j}',),meas_path='',x0=0,y0=0) for j in range(24)]
   self.assertEqual(len(ZarrCutoutDataset(records)[23]['band_mask_instances'][0]),1)
   for bs in [10,20]:
    for klass in [ZarrChunkLocalBatchSampler,ZarrChunkBatchIterableDataset]:
     allrows=[];lens=[]
     for rank in range(2):
      ds=klass(records,batch_size=bs,shuffle=True,seed=7,num_replicas=2,rank=rank,drop_last=True,equalize_replicas=True)
      chunks=ds._rank_chunks(epoch=0) if klass is ZarrChunkBatchIterableDataset else ds._rank_chunks()
      allrows.extend(i for c in chunks for i in c);lens.append(len(ds));self.assertTrue(all(len(c)%bs==0 for c in chunks))
     self.assertEqual(lens[0],lens[1]);self.assertEqual(set(allrows),set(range(24)))
   from torch.utils.data import DataLoader
   from astro_train_zarr_data import zarr_passthrough_batch
   worker_ds=ZarrChunkBatchIterableDataset(records,batch_size=10,shuffle=True,seed=7,drop_last=True)
   batches=list(DataLoader(worker_ds,batch_size=None,num_workers=2,collate_fn=only_names))
   self.assertEqual(len(batches),len(worker_ds))
   observed={name for batch in batches for name in batch}
   self.assertEqual(observed,set(map(str,range(24))))
   ds=ZarrChunkLocalBatchSampler(records,batch_size=20,shuffle=False,seed=7,drop_last=False)
   self.assertEqual([i for c in ds for i in c],list(range(24)))

 def test_truncated_kron_source_class_and_centered_exemption(self):
  from preprocessing.utils.truncated_kron import filter_truncated_kron_tiles
  lab=labels();lab.source_ids=np.array([100]);lab.geom_x=np.array([0.]);lab.geom_y=np.array([256.]);lab.geom_major=np.array([300.]);lab.geom_minor=np.array([100.]);lab.geom_theta=np.zeros(1)
  specs=[TileSpec('grid',0,0,512),TileSpec('centered',0,0,512,kind='large_source')]
  for cls in [0,1,2,3,4,5,6]:
   lab.label_classes=np.array([cls]);kept,audit=filter_truncated_kron_tiles(lab,specs)
   self.assertEqual([z.name for z in kept],['centered'])
   self.assertAlmostEqual(audit['rejected_tiles'][0]['sources'][0]['kron_fraction'],.5)
  lab.label_classes=np.array([1]);lab.geom_major=np.array([200.]);kept,_=filter_truncated_kron_tiles(lab,specs);self.assertEqual(len(kept),2)
  # Raw catalog geometry can retain an off-window center missing from labels.
  from preprocessing.utils.truncated_kron import large_kron_geometry
  lab.truncation_geometry=large_kron_geometry([200],[-1.],[256.],[300.],[100.],[0.])
  kept,audit=filter_truncated_kron_tiles(lab,specs);self.assertEqual([s.name for s in kept],['centered'])
  self.assertEqual(audit['rejected_tiles'][0]['sources'][0]['source_id'],200)
  self.assertIsNone(audit['rejected_tiles'][0]['sources'][0]['source_class'])
  lab.truncation_geometry=None
  # Fully contained large ellipse is retained.
  lab.geom_major=np.array([220.]);lab.geom_x=np.array([256.]);kept,_=filter_truncated_kron_tiles(lab,specs);self.assertEqual(len(kept),2)

 def test_hsc_edge(self):
  from preprocessing.utils.hsc_quality import hsc_training_tile_quality
  from preprocessing.utils.image_level import _read_fits_quality_mask
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/'hsc.fits';image=np.ones((16,16),np.float32);mask=np.full((16,16),1<<4,np.int32)
   def save():
    mh=fits.ImageHDU(mask,name='MASK')
    for name,bit in [('EDGE',4),('BAD',0),('NO_DATA',8),('UNMASKEDNAN',11),('INTRP',2)]:mh.header['MP_'+name]=bit
    fits.HDUList([fits.PrimaryHDU(),fits.ImageHDU(image,name='IMAGE'),mh]).writeto(path,overwrite=True)
   save();spec=[TileSpec('one',0,0,16)]
   valid,keep,audit=hsc_training_tile_quality(path,image,(0,0),spec);self.assertEqual(len(keep),1);self.assertAlmostEqual(audit['tiles'][0]['bad_score'],.1,places=6);self.assertFalse(_read_fits_quality_mask(path,image.shape).any())
   mask[:8]|=1<<2;save();valid,keep,audit=hsc_training_tile_quality(path,image,(0,0),spec);self.assertEqual(len(keep),0);self.assertTrue(valid.all());self.assertAlmostEqual(audit['tiles'][0]['bad_score'],.2,places=6)
   mask[:]=1<<4;mask[:2]|=1<<8;save();valid,keep,audit=hsc_training_tile_quality(path,image,(0,0),spec);self.assertEqual(len(keep),0);self.assertAlmostEqual(audit['tiles'][0]['invalid_fraction'],.125)

if __name__=='__main__':unittest.main()
