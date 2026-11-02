"""Adam summaries are numerically meaningful and cannot change training state."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from utils.adam_state import AdamStateLogger, adam_state_metrics


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(1, 1, bias=False)
        self.decoder = nn.Linear(1, 1, bias=False)
        self.mask_decoder = nn.Linear(1, 1, bias=False)
        self.prompt_encoder = nn.Linear(1, 1, bias=False)
        for p in self.parameters():
            nn.init.constant_(p, 1.)

    def forward(self, image):
        return {'confidence': self.decoder(self.encoder(image))}


class AdamStateTests(unittest.TestCase):
    def test_moments_and_bias_corrected_adaptive_step(self):
        model = TinyModel()
        opt = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=.1)
        for p in model.parameters():
            p.grad = torch.full_like(p, 2.)
        old = [p.detach().clone() for p in model.parameters()]
        opt.step()
        weights = [p.detach().clone() for p in model.parameters()]
        gradients = [p.grad.clone() for p in model.parameters()]
        states = copy.deepcopy(opt.state_dict())
        rng = torch.random.get_rng_state().clone()
        metrics = adam_state_metrics(model, opt)
        self.assertAlmostEqual(metrics['encoder/exp_avg_rms'], .2, places=6)
        self.assertAlmostEqual(metrics['encoder/exp_avg_sq_mean'], .004, places=7)
        self.assertAlmostEqual(metrics['encoder/bias_corrected_sqrt_exp_avg_sq_rms'], 2., places=5)
        expected_update = float((old[0] * (1 - .02 * .1) - weights[0]).norm())
        self.assertAlmostEqual(metrics['encoder/adaptive_step_norm'], expected_update, places=6)
        self.assertAlmostEqual(metrics['mask_decoder/adaptive_step_norm'], expected_update * 2 ** .5, places=6)
        self.assertEqual(metrics['mask_decoder/parameter_count'], 2)
        self.assertEqual(metrics['mask_decoder/step_min'], 1)
        self.assertEqual(metrics['mask_decoder/active_fraction'], 1)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        for p, w, g in zip(model.parameters(), weights, gradients):
            torch.testing.assert_close(p, w, rtol=0, atol=0)
            torch.testing.assert_close(p.grad, g, rtol=0, atol=0)
        for key, val in opt.state_dict()['state'].items():
            for name, tensor in val.items():
                torch.testing.assert_close(tensor, states['state'][key][name], rtol=0, atol=0)

    def test_none_grad_has_no_adaptive_update_and_preserves_state(self):
        model = TinyModel()
        opt = torch.optim.AdamW(model.parameters(), lr=.02)
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        opt.step()
        opt.zero_grad(set_to_none=True)
        metrics = adam_state_metrics(model, opt)
        self.assertEqual(metrics['mask_decoder/adaptive_step_norm'], 0)
        self.assertEqual(metrics['mask_decoder/active_fraction'], 0)
        self.assertGreater(metrics['mask_decoder/exp_avg_norm'], 0)
        self.assertEqual(metrics['mask_decoder/step_max'], 1)

    def test_uninitialized_logging_does_not_create_optimizer_state(self):
        model = TinyModel()
        opt = torch.optim.AdamW(model.parameters(), lr=.02)
        metrics = adam_state_metrics(model, opt)
        self.assertEqual(len(opt.state), 0)
        self.assertEqual(metrics['mask_decoder/initialized_fraction'], 0)
        self.assertEqual(metrics['mask_decoder/step_min'], 0)
        self.assertEqual(metrics['mask_decoder/adaptive_step_norm'], 0)

    def test_interval_jsonl_and_callback_are_independent_of_display_world_size(self):
        model = TinyModel()
        opt = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            logger = AdamStateLogger(tmp, log_fn=lambda *args: calls.append(args))
            kw = dict(iteration=200, epoch=0, mask_active=False)
            self.assertIsNone(logger.maybe_log(model, opt, optimizer_step=99, **kw))
            logger.maybe_log(model, opt, optimizer_step=100, **kw)
            record = json.loads((Path(tmp) / 'metrics.jsonl').read_text())
            self.assertEqual(record['optimizer_step'], 100)
            self.assertEqual(record['iteration'], 200)
            self.assertFalse(record['mask_active'])
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1], 200)
            self.assertFalse((Path(tmp) / 'errors.jsonl').exists())

    def test_different_lrs_steps_and_amsgrad(self):
        model = TinyModel()
        opt = torch.optim.AdamW([
            {'params': model.mask_decoder.parameters(), 'lr': .02},
            {'params': model.prompt_encoder.parameters(), 'lr': .01}], amsgrad=True)
        for i in range(2):
            opt.zero_grad(set_to_none=True)
            model.mask_decoder.weight.grad = torch.ones_like(model.mask_decoder.weight)
            if i == 0:
                model.prompt_encoder.weight.grad = torch.ones_like(model.prompt_encoder.weight)
            opt.step()
        m = adam_state_metrics(model, opt)
        self.assertEqual(m['mask_decoder/step_min'], 1)
        self.assertEqual(m['mask_decoder/step_max'], 2)
        self.assertEqual(m['mask_decoder/active_fraction'], .5)
        self.assertEqual(m['mask_decoder/lr_min'], .01)
        self.assertEqual(m['mask_decoder/lr_max'], .02)
        self.assertAlmostEqual(m['mask_decoder/adaptive_step_norm'], .02, places=6)


if __name__ == '__main__':
    unittest.main()
