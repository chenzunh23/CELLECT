"""Source-only prompting: gradients, multiscale execution and reload contracts."""
from dataclasses import asdict, replace
import os
from pathlib import Path
import shlex
import subprocess
from unittest.mock import patch

import pytest
import torch
from torch import nn

from sam_backbone.decoder import SamCellectDecoder
from sam_backbone.model import SamCellect2D
from sam_backbone.prompt_ir import PromptIRBlock, PromptIRConfig, PromptIRImagePyramid, prompt_ir_kwargs_from_state_dict
from utils.train_eval_utils import sam_optimizer_param_groups, WarmupStepIterationLR


CONFIG = PromptIRConfig(num_prompts=3, prompt_dim=4, prompt_size=3,
                        condition_dim=8, hidden_dim=8, num_heads=2, image_channels=4)
STAGES = ('neck', 'stem', 'up1', 'up2', 'up3', 'up4', 'refine')


def decoder(enabled=True, stages=STAGES, in_channels=16, image_conditioning=True):
    return SamCellectDecoder(in_channels=in_channels, decoder_channels=(16, 12, 8, 4),
        use_cen=False, use_prompt_ir=enabled, prompt_ir_stages=stages,
        prompt_ir_config=asdict(replace(CONFIG, image_conditioning=image_conditioning)))


def test_all_scales_have_first_step_prompt_gradients_and_updates():
    torch.manual_seed(72)
    module = decoder()
    x = torch.randn(1, 2, 16, 2, 3)
    images = torch.randn(1, 2, 32, 48)
    original = x.clone()
    output = module(x, images=images)
    assert output['confidence'].shape == (1, 2, 5, 32, 48)
    assert output['shape'].shape == (1, 2, 3, 32, 48)
    assert torch.equal(x, original)
    loss = output['confidence'].square().mean() + output['shape'].square().mean()
    loss.backward()
    for stage, block in module.prompt_ir.blocks.items():
        for name, parameter in [('bank', block.prompt_bank), ('router', block.router.weight),
                                ('image_router', block.image_router.weight),
                                ('decoder_project', block.image_condition.decoder_project.weight),
                                ('image_project', block.image_condition.image_project.weight)]:
            assert parameter.grad is not None, (stage, name)
            assert torch.isfinite(parameter.grad).all(), (stage, name)
            assert parameter.grad.abs().sum() > 0, (stage, name)
    assert module.prompt_ir.condition_encoder.project[0].weight.grad.abs().sum() > 0
    for block in module.prompt_ir.image_pyramid.blocks:
        assert block[0].weight.grad is not None and block[0].weight.grad.abs().sum() > 0
    bank = module.prompt_ir.blocks['up4'].prompt_bank
    saved = bank.detach().clone()
    torch.optim.AdamW(module.parameters(), lr=1e-3).step()
    assert not torch.equal(saved, bank)


def test_disabled_decoder_keeps_old_weights_and_output_exactly():
    torch.manual_seed(73)
    a = decoder(False)
    torch.manual_seed(73)
    b = SamCellectDecoder(in_channels=16, decoder_channels=(16, 12, 8, 4), use_cen=False)
    assert not any('prompt_ir' in k for k in a.state_dict())
    x = torch.randn(2, 16, 2, 3)
    for key, value in a(x).items():
        assert torch.equal(value, b(x)[key])


def test_router_is_image_conditioned_even_after_instance_normalization():
    module = decoder(stages=('up3', 'up4'))
    condition = module.prompt_ir.condition_encoder(torch.randn(1, 16, 3, 3)).expand(2, -1)
    block = module.prompt_ir.blocks['up4']
    # Hold both SAM and decoder features fixed: the image branch must affect
    # routing independently, even after source InstanceNorm removes mean shifts.
    x = torch.zeros(2, 4, 8, 8)
    context = module.prompt_ir.prepare_image_condition(torch.randn(2, 1, 8, 8), canvas_size=(8, 8))
    weights = block.prompt_weights(x, condition, context['up4'])
    assert torch.allclose(weights.sum(-1), torch.ones(2))
    assert not torch.equal(weights[0], weights[1])


def test_image_features_only_control_prompt_mixture_without_spatial_skip():
    block = PromptIRBlock(4, CONFIG, image_channels=4)
    x, condition = torch.randn(1, 4, 8, 8), torch.randn(1, CONFIG.condition_dim)
    valid = torch.ones(1, 1, 8, 8)
    first = (torch.randn_like(x), valid)
    second = (torch.randn_like(x), valid)
    assert not torch.equal(block(x, condition, first), block(x, condition, second))
    # Fix the prompt weights. Changing image features must have NO remaining
    # path to the source feature output (unlike a spatial skip connection).
    with patch.object(block, 'prompt_weights', return_value=torch.full((1, CONFIG.num_prompts), 1 / CONFIG.num_prompts)):
        assert torch.equal(block(x, condition, first), block(x, condition, second))


@pytest.mark.parametrize('prefix', ['', 'module._orig_mod.'])
def test_bare_checkpoint_reconstructs_prompt_geometry_and_all_stages(prefix):
    a = nn.Module()
    a.decoder = decoder()
    state = {prefix + key: value for key, value in a.state_dict().items()}
    kwargs = prompt_ir_kwargs_from_state_dict(state)
    b = nn.Module()
    b.decoder = SamCellectDecoder(in_channels=16, decoder_channels=(16, 12, 8, 4),
        use_cen=False, use_prompt_ir=kwargs['decoder_prompt_ir'],
        prompt_ir_stages=kwargs['prompt_ir_stages'], prompt_ir_config=kwargs['prompt_ir_config'])
    b.load_state_dict(a.state_dict(), strict=True)
    x = torch.randn(1, 16, 2, 3)
    images = torch.randn(1, 1, 32, 48)
    assert torch.equal(a.decoder(x, images=images)['confidence'], b.decoder(x, images=images)['confidence'])
    assert kwargs['prompt_ir_stages'] == STAGES
    assert kwargs['prompt_ir_config']['image_conditioning'] is True
    assert kwargs['prompt_ir_config']['image_channels'] == CONFIG.image_channels
    assert prompt_ir_kwargs_from_state_dict({'decoder.shape_head.weight': torch.ones(1)}) == {'decoder_prompt_ir': False}


class _TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 256, kernel_size=16, stride=16)

    def forward(self, x, **kwargs):
        b, bands, h, w = x.shape
        x = torch.nan_to_num(x, nan=0., posinf=0., neginf=0.)
        features = self.conv(x.reshape(b * bands, 1, h, w))
        return {'features': features.reshape(b, bands, 256, h // 16, w // 16),
                'preprocessed_images': x}


def tiny_model(enabled=True, image_conditioning=True):
    return SamCellect2D(_TinyEncoder(), decoder(enabled, in_channels=256, image_conditioning=image_conditioning),
                        image_size=32, patch_size=16, dynamic_image_size=True)


def test_optional_neck_prompt_leaves_shared_embeddings_and_mask_branch_unchanged():
    baseline, prompted = tiny_model(False).eval(), tiny_model().eval()
    missing = prompted.load_state_dict(baseline.state_dict(), strict=False)
    assert missing.missing_keys and all(k.startswith('decoder.prompt_ir.') for k in missing.missing_keys)
    assert not missing.unexpected_keys
    x = torch.randn(1, 1, 32, 48)
    with torch.no_grad():
        before, after = baseline(x), prompted(x)
        assert torch.equal(before['image_embeddings'], after['image_embeddings'])
        assert not torch.equal(before['confidence'], after['confidence'])
        arguments = (torch.tensor([0]), torch.tensor([[20., 10.]]))
        m0, i0 = baseline.forward_sam_masks(before['image_embeddings'], *arguments,
                    boxes=None, multimask_output=True, output_size=(32, 48))
        m1, i1 = prompted.forward_sam_masks(after['image_embeddings'], *arguments,
                    boxes=None, multimask_output=True, output_size=(32, 48))
    assert torch.equal(m0, m1)
    assert torch.equal(i0, i1)


def test_optimizer_and_scheduler_cover_prompt_once_at_independent_lr():
    model = tiny_model()
    groups = sam_optimizer_param_groups(model, head_lr=5e-5, encoder_lr=1e-5, weight_decay=.1)
    params = [p for g in groups for p in g['params']]
    assert len(params) == len(set(map(id, params))) == len(list(model.parameters()))
    prompt = next(g for g in groups if g['name'] == 'prompt_ir')
    assert prompt['lr'] == 2.5e-4
    assert set(map(id, prompt['params'])) == set(map(id, model.decoder.prompt_ir.parameters()))
    groups = sam_optimizer_param_groups(model, head_lr=5e-5, encoder_lr=1e-5, weight_decay=.1, prompt_ir_lr=5e-4)
    optimizer = torch.optim.AdamW(groups)
    schedule = WarmupStepIterationLR(optimizer, total_steps=8, warmup_steps=2, drop_steps=(5,), drop_gamma=.1)
    for _ in range(6):
        rates = {g['name']: g['lr'] for g in optimizer.param_groups}
        assert rates['prompt_ir'] == pytest.approx(10 * rates['head'])
        schedule.step()
    schedule.set_base_lrs({'head': 0., 'prompt_ir': 0.})
    assert all(g['lr'] == 0 for g in optimizer.param_groups if g['name'] in ('head', 'prompt_ir'))


def test_browser_eval_loader_reconstructs_prompt_from_checkpoint(tmp_path, monkeypatch):
    import eval.eval_utils as evaluation
    model = tiny_model().eval()
    checkpoint = tmp_path / 'prompt.pt'
    torch.save({'model': model.state_dict()}, checkpoint)

    def build(_model_type, **kwargs):
        assert kwargs['decoder_prompt_ir'] is True
        assert kwargs['prompt_ir_stages'] == STAGES
        assert kwargs['prompt_ir_config']['prompt_size'] == CONFIG.prompt_size
        return tiny_model()

    monkeypatch.setattr(evaluation, 'build_sam_cellect2d', build)
    loaded, _ = evaluation.load_cellect_model(checkpoint, None, torch.device('cpu'), ['F444W'])
    x = torch.randn(1, 1, 32, 48)
    with torch.no_grad():
        assert torch.equal(model(x)['confidence'], loaded(x)['confidence'])


def test_compile_and_bfloat16_forward_backward():
    block = PromptIRBlock(4, CONFIG)
    compiled = torch.compile(block, backend='eager', fullgraph=True)
    x, condition = torch.randn(2, 4, 4, 4), torch.randn(2, CONFIG.condition_dim)
    assert torch.allclose(block(x, condition), compiled(x, condition))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = compiled(x, condition).square().mean()
    loss.backward()
    assert block.prompt_bank.grad.abs().sum() > 0
    assert torch.isfinite(block.prompt_bank.grad).all()


@pytest.mark.parametrize('mode', ['train', 'eval'])
@pytest.mark.parametrize('image_conditioning', ['0', '1'])
def test_training_wrapper_prompt_arguments_reach_python_parser(tmp_path, mode, image_conditioning):
    from astro_train_eval import build_arg_parser
    split = tmp_path / 'split.txt'
    split.write_text('4,5\n')
    checkpoint = tmp_path / 'sam.pt'
    checkpoint.touch()
    env = dict(os.environ, DRY_RUN='1', MODE=mode, NPROC='1',
        TRAIN_FILE=str(split), VAL_FILE=str(split), SAM_CKPT=str(checkpoint),
        PROMPT_IR='1', PROMPT_IR_STAGES='stem up1 up2 up3 up4', PROMPT_IR_LR='0.0005',
        PROMPT_IR_IMAGE_CONDITIONING=image_conditioning, PROMPT_IR_IMAGE_CHANNELS='4')
    result = subprocess.run(['bash', 'scripts/train_mixed_zarr.sh'],
        cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, check=True)
    args = build_arg_parser().parse_args(shlex.split(result.stdout)[2:])
    assert args.sam_prompt_ir is True
    assert args.sam_prompt_ir_stages == ['stem', 'up1', 'up2', 'up3', 'up4']
    assert args.sam_prompt_ir_lr == .0005
    assert args.sam_prompt_ir_image_conditioning is bool(int(image_conditioning))
    assert args.sam_prompt_ir_image_channels == 4
    assert args.mode == mode
    if mode == 'eval':
        assert args.checkpoint.endswith('best.pt')


def test_legacy_neck_only_checkpoint_remains_exact_and_can_warm_start_image_branch():
    legacy = nn.Module()
    legacy.decoder = decoder(image_conditioning=False)
    state = legacy.state_dict()
    inferred = prompt_ir_kwargs_from_state_dict(state)
    assert inferred['prompt_ir_config']['image_conditioning'] is False
    reconstructed = decoder(image_conditioning=False)
    reconstructed.load_state_dict(legacy.decoder.state_dict(), strict=True)
    x = torch.randn(1, 16, 2, 3)
    assert torch.equal(legacy.decoder(x)['confidence'], reconstructed(x)['confidence'])
    upgraded = decoder()
    result = upgraded.load_state_dict(legacy.decoder.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert result.missing_keys
    assert all(k == 'prompt_ir._image_architecture' or k.startswith('prompt_ir.image_pyramid.') or
               '.image_condition.' in k or '.image_router.' in k for k in result.missing_keys)
    assert torch.equal(upgraded.prompt_ir.blocks['up4'].router.weight, legacy.decoder.prompt_ir.blocks['up4'].router.weight)


def test_invalid_pixels_and_padding_do_not_enter_image_condition_statistics():
    pyramid = PromptIRImagePyramid(4, max_level=4)
    image = torch.randn(1, 1, 30, 45)
    valid = torch.ones_like(image, dtype=torch.bool)
    valid[..., 2:6, 4:9] = False
    first, second = image.clone(), image.clone()
    first[~valid] = float('nan')
    second[~valid] = 1e8
    a = pyramid(first, canvas_size=(32, 48), valid_mask=valid)
    b = pyramid(second, canvas_size=(32, 48), valid_mask=valid)
    padded = torch.nn.functional.pad(second, (0, 3, 0, 2), value=1e8)
    padded_valid = torch.nn.functional.pad(valid, (0, 3, 0, 2), value=False)
    c = pyramid(padded, canvas_size=(32, 48), valid_mask=padded_valid)
    for (fa, va), (fb, vb), (fc, vc) in zip(a, b, c):
        assert torch.equal(fa, fb) and torch.equal(fb, fc)
        assert torch.equal(va, vb) and torch.equal(vb, vc)
    assert a[0][1].sum() == valid.sum()


def test_empty_image_support_is_finite_and_uses_only_existing_neck_route():
    block = PromptIRBlock(4, CONFIG, image_channels=4)
    x, condition = torch.randn(1, 4, 8, 8), torch.randn(1, CONFIG.condition_dim)
    context = (torch.zeros_like(x), torch.zeros(1, 1, 8, 8))
    weights = block.prompt_weights(x, condition, context)
    expected = block.router(torch.cat((x.mean((-2, -1)), condition), 1)).float().softmax(-1)
    assert torch.equal(weights, expected)
    assert torch.isfinite(block(x, condition, context)).all()


def test_dynamic_non_square_rgb_inputs_align_all_stage_grids():
    module = decoder()
    images = torch.randn(1, 2, 3, 30, 45)
    features = torch.randn(1, 2, 16, 2, 3)
    output = module(features, images=images, output_size=(30, 45))
    assert output['confidence'].shape == (1, 2, 5, 30, 45)
    with pytest.raises(ValueError, match='requires scaled input images'):
        module(features)


def test_model_retains_known_input_nan_mask_after_encoder_sanitizes():
    model = tiny_model()
    image = torch.randn(1, 1, 32, 48)
    image[..., 4:7, 5:8] = float('nan')
    masks = []

    def capture_mask(_module, args, kwargs):
        masks.append(kwargs['valid_mask'].clone())

    handle = model.decoder.prompt_ir.image_pyramid.register_forward_pre_hook(capture_mask, with_kwargs=True)
    try:
        output = model(image)
    finally:
        handle.remove()
    assert torch.equal(masks[0], torch.isfinite(image).reshape(1, 1, 32, 48))
    assert torch.isfinite(output['confidence']).all()


def test_image_checkpoint_missing_descriptor_is_rejected():
    module = nn.Module()
    module.decoder = decoder()
    state = dict(module.state_dict())
    del state['decoder.prompt_ir._image_architecture']
    with pytest.raises(ValueError, match='missing its architecture descriptor'):
        prompt_ir_kwargs_from_state_dict(state)


def test_compile_image_conditioned_decoder_and_bfloat16_backward():
    module = decoder(stages=('stem', 'up1', 'up2', 'up3', 'up4'))
    compiled = torch.compile(module, backend='eager', fullgraph=True)
    features, images = torch.randn(1, 16, 2, 3), torch.randn(1, 1, 30, 45)
    output = compiled(features, images=images, output_size=(30, 45))
    assert torch.allclose(output['confidence'], module(features, images=images, output_size=(30, 45))['confidence'])
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result = compiled(features, images=images, output_size=(30, 45))
        loss = result['confidence'].square().mean() + result['shape'].square().mean()
    loss.backward()
    for name, p in module.prompt_ir.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
