"""Exercise the real epoch step through DDP's zero anchors and mask warmup."""
from collections import defaultdict
import copy
from datetime import timedelta
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from astro_train_ops import run_epoch, LossWeights
from sam_backbone.mask_metrics import GROUP_STAT_KEYS
from tests.test_adam_state import TinyModel
from utils.adam_state import AdamStateLogger
from utils.train_eval_utils import _DDPModelWrapper


def dense_loss(outputs, *args, **kwargs):
    total = outputs['confidence'].square().mean()
    return defaultdict(lambda: total * 0., total=total)


def mask_loss(model, outputs, *args, **kwargs):
    total = (model.mask_decoder.weight.sum() + model.prompt_encoder.weight.sum()).square()
    return defaultdict(lambda: total * 0., total=total,
                       group_stats=total.new_zeros(len(GROUP_STAT_KEYS)))


def exercise(model, opt, logger=None):
    base = model.module._ddp_wrapped_model if hasattr(model, 'module') else model._ddp_wrapped_model
    mask_params = list(base.mask_decoder.parameters()) + list(base.prompt_encoder.parameters())
    initial = [p.detach().clone() for p in mask_params]
    kw = dict(optimizer=opt, device=torch.device('cpu'),
              weights=LossWeights(mask_outer_weight=1., mask_loss_warmup_epochs=1),
              triplet_loss_fn=None, triplet_enabled=False, ex_enabled=False, en_enabled=False,
              matcher_candidate_count=1, matcher_max_anchors_per_band=1,
              triplet_max_sources_per_group=1, triplet_negative_scope='ignore',
              ex_band_pairs=None, center_radius_px=1., show_progress=False,
              adam_state_logger=logger)
    loader = [{'image': torch.ones(2, 1)}]
    with patch('astro_train_ops.dense_losses_any', dense_loss), patch('astro_train_ops.sam_prompt_mask_losses', mask_loss):
        run_epoch(model, loader, epoch_index=0, **kw)
        for p, old in zip(mask_params, initial):
            torch.testing.assert_close(p, old, rtol=0, atol=0)
            assert p.grad is None
            assert p not in opt.state
        run_epoch(model, loader, epoch_index=1, **kw)
        active_weights = [p.detach().clone() for p in mask_params]
        states = [copy.deepcopy(opt.state[p]) for p in mask_params]
        assert all(not torch.equal(p, old) for p, old in zip(mask_params, initial))
        # optimizer step 2 is skipped when the mask interval is 2. Existing
        # momentum as well as weight decay must not change weights or states.
        run_epoch(model, loader, epoch_index=1, global_step_start=1,
                  mask_loss_interval=2, **kw)
        for p, old, state in zip(mask_params, active_weights, states):
            torch.testing.assert_close(p, old, rtol=0, atol=0)
            for name in state:
                torch.testing.assert_close(opt.state[p][name], state[name], rtol=0, atol=0)
        run_epoch(model, loader, epoch_index=1, global_step_start=2,
                  mask_loss_interval=2, **kw)
        for p in mask_params:
            assert opt.state[p]['step'].item() == 2


def worker(rank, tmp, unused):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', rank=rank, world_size=2,
        init_method='file://' + str(Path(tmp) / 'init'), timeout=timedelta(seconds=45))
    try:
        model = DDP(_DDPModelWrapper(TinyModel()), find_unused_parameters=unused)
        opt = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=.1)
        logger = AdamStateLogger(Path(tmp) / 'adam_state', interval=1) if rank == 0 else None
        exercise(model, opt, logger)
        vector = torch.cat([p.detach().flatten() for p in model.parameters()])
        gathered = [torch.empty_like(vector) for _ in range(2)]
        dist.all_gather(gathered, vector)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class MaskOptimizerTests(unittest.TestCase):
    def test_epoch_warmup_and_interval_preserve_weights_and_adam_state(self):
        model = _DDPModelWrapper(TinyModel())
        opt = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=.1)
        exercise(model, opt)

    def test_ddp_can_resume_mask_training_after_skipped_updates(self):
        for unused in (False, True):
            with self.subTest(find_unused_parameters=unused), tempfile.TemporaryDirectory() as tmp:
                mp.spawn(worker, args=(tmp, unused), nprocs=2, join=True)
                self.assertEqual(len((Path(tmp) / 'adam_state/metrics.jsonl').read_text().splitlines()), 4)
                self.assertFalse((Path(tmp) / 'adam_state/errors.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
