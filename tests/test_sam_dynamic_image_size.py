from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from sam_backbone.build import SamPerBandImageEncoder
from sam_backbone.image_encoder import ImageEncoderViT
from sam_backbone.losses import _sam_mask_loss_for_prompts
from sam_backbone.model import SamCellect2D
from sam_backbone.preprocess import pad_to_patch_multiple
from sam_backbone.prompt_encoder import PromptEncoder


class _ShapeOnlyImageEncoder(nn.Module):
    img_size = 512
    patch_size = 16

    def forward(self, x: torch.Tensor, style_prompt=None) -> torch.Tensor:
        del style_prompt
        return x.new_zeros((x.shape[0], 256, x.shape[-2] // 16, x.shape[-1] // 16))


class _MaskLossModel(nn.Module):
    dynamic_image_size = True

    def __init__(self) -> None:
        super().__init__()
        self.requested_output_size = None

    def forward_sam_masks(
        self,
        image_embeddings,
        prompt_batch_indices,
        point_coords,
        boxes,
        *,
        multimask_output,
        chunk_size,
        output_size=None,
    ):
        del image_embeddings, point_coords, boxes, multimask_output, chunk_size
        self.requested_output_size = output_size
        count = int(prompt_batch_indices.numel())
        assert output_size is not None
        masks = torch.zeros((count, 1, *output_size), dtype=torch.float32, requires_grad=True)
        iou = torch.zeros((count, 1), dtype=torch.float32, requires_grad=True)
        return masks, iou


def test_patch_multiple_padding_is_minimal_and_enforces_maximum() -> None:
    x = torch.ones((2, 3, 250, 370))
    padded = pad_to_patch_multiple(x, 16, max_size=512)
    assert padded.shape == (2, 3, 256, 384)
    assert torch.equal(padded[..., :250, :370], x)
    assert not torch.any(padded[..., 250:, :])
    assert not torch.any(padded[..., :, 370:])


def test_default_encoder_padding_remains_fixed_512() -> None:
    encoder = SamPerBandImageEncoder(
        _ShapeOnlyImageEncoder(),
        num_bands=1,
        astro_preprocess_in_model=False,
    )
    result = encoder(torch.ones((1, 1, 250, 370)), input_is_preprocessed=True, return_flat=True)
    assert result["flat_rgb"].shape == (1, 3, 512, 512)
    assert result["features"].shape[-2:] == (32, 32)
    assert (result["padded_height"], result["padded_width"]) == (512, 512)


def test_dynamic_encoder_padding_uses_patch_multiple() -> None:
    encoder = SamPerBandImageEncoder(
        _ShapeOnlyImageEncoder(),
        num_bands=1,
        astro_preprocess_in_model=False,
        dynamic_image_size=True,
    )
    result = encoder(torch.ones((1, 1, 250, 370)), input_is_preprocessed=True, return_flat=True)
    assert result["flat_rgb"].shape == (1, 3, 256, 384)
    assert result["features"].shape[-2:] == (16, 24)
    assert (result["padded_height"], result["padded_width"]) == (256, 384)


def test_image_encoder_interpolates_trainable_absolute_position_embedding() -> None:
    encoder = ImageEncoderViT(
        img_size=512,
        patch_size=16,
        embed_dim=8,
        depth=0,
        num_heads=1,
        out_chans=4,
        use_abs_pos=True,
        use_rel_pos=False,
    )
    output = encoder(torch.randn((1, 3, 32, 48)))
    assert output.shape == (1, 4, 2, 3)
    output.square().mean().backward()
    assert encoder.pos_embed is not None
    assert encoder.pos_embed.grad is not None
    assert encoder.pos_embed.grad.shape == encoder.pos_embed.shape


def test_prompt_encoder_generates_dynamic_dense_embeddings() -> None:
    encoder = PromptEncoder(
        embed_dim=32,
        image_embedding_size=(32, 32),
        input_image_size=(512, 512),
        mask_in_chans=16,
    )
    points = torch.tensor([[[20.0, 10.0]]])
    labels = torch.ones((1, 1), dtype=torch.long)
    sparse, dense = encoder(
        points=(points, labels),
        boxes=None,
        masks=None,
        image_embedding_size=(16, 24),
        input_image_size=(256, 384),
    )
    assert sparse.shape == (1, 2, 32)
    assert dense.shape == (1, 32, 16, 24)
    assert encoder.get_dense_pe((16, 24)).shape == (1, 32, 16, 24)
    assert encoder.get_dense_pe().shape == (1, 32, 32, 32)


def test_mask_decoder_resizes_to_padded_shape_before_cropping() -> None:
    model = SamCellect2D(
        nn.Identity(),
        nn.Identity(),
        image_size=512,
        patch_size=16,
        dynamic_image_size=True,
    )
    image_embeddings = torch.randn((1, 1, 256, 2, 3), requires_grad=True)
    masks, iou = model.forward_sam_masks(
        image_embeddings,
        torch.tensor([0]),
        torch.tensor([[20.0, 10.0]]),
        boxes=None,
        multimask_output=False,
        output_size=(30, 45),
    )
    assert masks.shape == (1, 1, 30, 45)
    assert iou.shape == (1, 1)
    (masks.square().mean() + iou.square().mean()).backward()
    assert image_embeddings.grad is not None


def test_dynamic_mask_loss_requests_full_original_resolution() -> None:
    model = _MaskLossModel()
    outputs = {
        "confidence": torch.zeros((1, 1, 5, 30, 45)),
        "image_embeddings": torch.zeros((1, 1, 256, 2, 3)),
    }
    prompts = {
        "batch_indices": torch.tensor([0]),
        "centers": torch.tensor([[20.0, 10.0]]),
        "prompt_shapes": torch.tensor([[3.0, 2.0, 0.0]]),
        "target_shapes": torch.tensor([[3.0, 2.0, 0.0]]),
        "weights": torch.ones(1),
        "mask_target_weights": torch.ones(1),
    }
    weights = SimpleNamespace(
        mask_prompt_center_only=False,
        mask_multimask=False,
        mask_prompt_chunk_size=8,
        mask_dice=1.0,
        mask_bce=1.0,
        mask_centroid=0.0,
        mask_outside=0.0,
        mask_min_area=0.0,
        mask_max_area=0.0,
        mask_pred_iou=0.0,
        mask_stability=0.0,
        mask_max_area_ratio=0.5,
        mask_selection="loss",
    )
    losses = _sam_mask_loss_for_prompts(
        model,
        outputs,
        prompts,
        weights=weights,
        image_hw=(30, 45),
        ellipse_sigma=2.0,
    )
    assert model.requested_output_size == (30, 45)
    assert torch.isfinite(losses["total"])
