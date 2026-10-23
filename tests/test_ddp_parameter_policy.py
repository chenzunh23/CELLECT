"""Explicit unused-parameter policy and SAM loss-only parameter coverage."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from astro_train_ops import clear_sam_proposal_grads
from astro_train_eval import build_arg_parser
from utils import train_eval_utils as training


@pytest.mark.parametrize('static,unused,expected', [
    ('off', None, True), ('on', None, False),
    ('off', False, False), ('off', True, True),
    ('on', False, False), ('on', True, True),
    ('auto', None, True), ('auto', False, False),
])
def test_unused_parameter_policy_can_be_overridden_without_static_graph(monkeypatch, static, unused, expected):
    monkeypatch.setattr(training, 'DistributedDataParallel', lambda module, **kwargs: kwargs)
    args = SimpleNamespace(ddp_static_graph=static, ddp_find_unused_parameters=unused,
                           mask_loss_warmup_epochs=3)
    result = training.wrap_ddp(nn.Linear(2, 2), args, torch.device('cpu'), 0)
    assert result['find_unused_parameters'] is expected
    assert result['static_graph'] is (static == 'on')
    assert args._ddp_find_unused_parameters_resolved is expected


def test_cli_unused_default_and_explicit_false():
    parser = build_arg_parser()
    assert parser.parse_args([]).ddp_find_unused_parameters is None
    assert parser.parse_args(['--ddp-find-unused-parameters']).ddp_find_unused_parameters is True
    args = parser.parse_args(['--ddp-static-graph', 'off', '--no-ddp-find-unused-parameters'])
    assert args.ddp_find_unused_parameters is False
    assert args.ddp_static_graph == 'off'


class _ConditionalMaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.decoder = nn.Linear(4, 8)
        self.prompt_encoder = nn.Linear(4, 4)
        self.mask_decoder = nn.Linear(4, 1)

    def forward(self, x):
        features = self.encoder(x)
        confidence, shape = self.decoder(features).chunk(2, dim=-1)
        return dict(confidence=confidence, shape=shape, image_embeddings=features)


def test_loss_only_parameters_remain_in_graph_across_warmup_and_proposal_freeze():
    model = _ConditionalMaskModel()
    wrapper = training._DDPModelWrapper(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    # Mimic existing confidence/shape zero anchors and mask decoding outside
    # the DDP forward, including the warmup transition and a frozen proposal.
    for mask_active, proposal_frozen in [(False, False), (True, False), (True, True), (False, True)]:
        optimizer.zero_grad(set_to_none=True)
        outputs = wrapper(torch.randn(2, 4))
        proposal_loss = outputs['confidence'].square().mean() + outputs['shape'].square().mean()
        total = proposal_loss * (0. if proposal_frozen else 1.)
        if mask_active:
            total = total + model.mask_decoder(model.prompt_encoder(outputs['image_embeddings'])).square().mean()
        total.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if not mask_active:
            assert all(not p.grad.any() for module in (model.prompt_encoder, model.mask_decoder) for p in module.parameters())
        if proposal_frozen:
            clear_sam_proposal_grads(model)
            assert all(p.grad is None for p in model.decoder.parameters())
        optimizer.step()
