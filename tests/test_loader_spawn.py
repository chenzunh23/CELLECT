"""Spawned Zarr workers isolate parent services and retain shared epoch state."""
import gc
import os
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from torch.utils.data import DataLoader, get_worker_info

from astro_train_data import CutoutRecord
from astro_train_zarr_data import ZarrChunkBatchIterableDataset
from preprocessing.build_image_level_zarr import write_classified_patch
from preprocessing.tests.test_segmentation_zarr import labels, task


_parent_service_marker = None


def inspect_spawned_batch(batch):
    gc.collect()  # The real hang triggered a parent W&B finalizer during GC.
    return dict(names=batch['name'], image=batch['image'], pid=os.getpid(),
                epoch=get_worker_info().dataset._epoch_value(),
                inherited_parent_service=_parent_service_marker is not None)


class SpawnLoaderTests(unittest.TestCase):
    def test_zarr_spawn_workers_and_persistent_epoch_updates(self):
        global _parent_service_marker
        _parent_service_marker = object()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with patch('preprocessing.build_image_level_zarr._scale_image_chw',
                           side_effect=lambda image, task: image[None]):
                    result=write_classified_patch(replace(task(tmp),chunk_tiles=2),
                                                 np.ones((32,32),np.float32),labels())
                store=Path(result['output'])
                records=[CutoutRecord(name=str(i),image_paths=(f'zarr://{store}#{i}',),
                                      meas_path='',x0=0,y0=0) for i in range(4)]
                dataset=ZarrChunkBatchIterableDataset(records,batch_size=2,shuffle=True,
                    seed=7,multiprocessing_context='spawn')
                loader=DataLoader(dataset,batch_size=None,num_workers=2,
                    multiprocessing_context='spawn',persistent_workers=True,
                    collate_fn=inspect_spawned_batch,timeout=30)
                try:
                    first=list(loader)
                    dataset.set_epoch(3)
                    second=list(loader)
                    for epoch,batches in ((0,first),(3,second)):
                        self.assertEqual({n for b in batches for n in b['names']},
                                         set(map(str,range(4))))
                        for batch in batches:
                            self.assertFalse(batch['inherited_parent_service'])
                            self.assertNotEqual(batch['pid'],os.getpid())
                            self.assertEqual(batch['epoch'],epoch)
                            self.assertTrue(bool((batch['image']==1).all()))
                    self.assertEqual({b['pid'] for b in first},{b['pid'] for b in second})
                finally:
                    if loader._iterator is not None:
                        loader._iterator._shutdown_workers()
        finally:
            _parent_service_marker = None


if __name__=='__main__':unittest.main()
