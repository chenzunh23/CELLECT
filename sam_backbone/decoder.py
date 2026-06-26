from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _conv_norm_lrelu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch),
        nn.LeakyReLU(),
    )


def _norm_lrelu_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.InstanceNorm2d(in_ch),
        nn.LeakyReLU(),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
    )


def _lrelu_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LeakyReLU(),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
    )


def _norm_lrelu_upscale_conv_norm_lrelu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.InstanceNorm2d(in_ch),
        nn.LeakyReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch),
        nn.LeakyReLU(),
    )


class CenterEnhancementNet2D(nn.Module):
    """CELLECT-style 2D center enhancement network.

    Input is segmentation logits, segmentation probabilities, and optionally
    the raw/preprocessed single-band image.  Output is an ordinal confidence
    map with ``confidence_levels`` channels.
    """

    def __init__(self, in_channels: int, confidence_levels: int = 5, width: int = 16) -> None:
        super().__init__()
        self.lrelu = nn.LeakyReLU()
        self.dropout2d = nn.Dropout2d(p=0.6)

        self.conv_c1_1 = nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_c1_2 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu_conv_c1 = _lrelu_conv(width, width)
        self.inorm_c1 = nn.InstanceNorm2d(width)

        self.conv_c2 = nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv_c2s1 = nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1, bias=False)
        self.conv_c2s2 = nn.Conv2d(width, width * 2, kernel_size=1, stride=2, bias=False)
        self.norm_lrelu_conv_c2 = _norm_lrelu_conv(width * 2, width * 2)
        self.inorm_c2 = nn.InstanceNorm2d(width * 2)

        self.conv_c3 = nn.Conv2d(width * 2, width * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv_c3s1 = nn.Conv2d(width * 4, width * 4, kernel_size=3, padding=1, bias=False)
        self.conv_c3s2 = nn.Conv2d(width * 2, width * 4, kernel_size=1, stride=2, bias=False)
        self.norm_lrelu_conv_c3 = _norm_lrelu_conv(width * 4, width * 4)

        self.up_l1 = _norm_lrelu_upscale_conv_norm_lrelu(width * 4, width * 2)
        self.conv_l1 = nn.Conv2d(width * 4, width * 2, kernel_size=1, bias=False)
        self.up_l2 = _norm_lrelu_upscale_conv_norm_lrelu(width * 2, width)
        self.conv_l2 = nn.Conv2d(width * 2, width, kernel_size=1, bias=False)
        self.out_refine = _conv_norm_lrelu(width * 2, width * 2)
        self.out = nn.Conv2d(width * 2, confidence_levels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv_c1_2(out)
        out = self.dropout2d(out)
        out = self.lrelu_conv_c1(out)
        out = self.lrelu(out + residual_1)
        context_1 = out
        out = self.lrelu(self.inorm_c1(out))

        outt = self.conv_c2(out)
        outt = self.conv_c2s1(outt)
        out = outt + self.conv_c2s2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout2d(out)
        out = self.norm_lrelu_conv_c2(out)
        out = self.lrelu(self.inorm_c2(out + residual_2))
        context_2 = out

        outt = self.conv_c3(out)
        outt = self.conv_c3s1(outt)
        out = outt + self.conv_c3s2(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout2d(out)
        out = self.norm_lrelu_conv_c3(out)
        out = out + residual_3

        out = self.up_l1(out)
        if out.shape[-2:] != context_2.shape[-2:]:
            out = F.interpolate(out, size=context_2.shape[-2:], mode="nearest")
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_l1(out)
        out = self.up_l2(out)
        if out.shape[-2:] != context_1.shape[-2:]:
            out = F.interpolate(out, size=context_1.shape[-2:], mode="nearest")
        out = torch.cat([out, context_1], dim=1)
        out = self.conv_l2(out)
        out = torch.cat([out, context_1], dim=1)
        return self.out(self.out_refine(out))


class SamCellectDecoder(nn.Module):
    """Region-proposal decoder for SAM image-encoder features.

    Input features can be either ``[B, band, C, h, w]`` or flat
    ``[B * band, C, h, w]``.  Outputs match ``astro_cellect2d``:

    - ``confidence``: ``[B, band, confidence_levels, H, W]``
    - ``shape``: ``[B, band, shape_channels, H, W]``

    For flat input the band axis is omitted from the returned tensors.
    """

    def __init__(
        self,
        *,
        in_channels: int = 256,
        seg_classes: int = 2,
        confidence_levels: int = 5,
        embedding_dim: int = 64,
        shape_channels: int = 3,
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        use_cen: bool = True,
        cen_input_image: bool = True,
        cen_width: int = 8,
    ) -> None:
        super().__init__()
        if len(tuple(decoder_channels)) != 4:
            raise ValueError("decoder_channels must contain four stages for 32x32 -> 512x512 decoding")
        self.seg_classes = int(seg_classes)
        self.confidence_levels = int(confidence_levels)
        self.embedding_dim = int(embedding_dim)
        self.shape_channels = int(shape_channels)
        self.use_cen = bool(use_cen)
        self.cen_input_image = bool(cen_input_image)
        self.pred_channels = self.shape_channels

        c1, c2, c3, c4 = [int(ch) for ch in decoder_channels]
        self.stem = _conv_norm_lrelu(int(in_channels), c1)
        self.up1 = _norm_lrelu_upscale_conv_norm_lrelu(c1, c1)
        self.up2 = _norm_lrelu_upscale_conv_norm_lrelu(c1, c2)
        self.up3 = _norm_lrelu_upscale_conv_norm_lrelu(c2, c3)
        self.up4 = _norm_lrelu_upscale_conv_norm_lrelu(c3, c4)
        self.refine = _conv_norm_lrelu(c4, c4)

        self.confidence_head = nn.Conv2d(c4, self.confidence_levels, kernel_size=1, bias=False)
        self.shape_refine = _conv_norm_lrelu(c4 + self.confidence_levels, c4)
        self.shape_head = nn.Conv2d(c4, self.shape_channels, kernel_size=1, bias=False)

    def forward(
        self,
        image_embeddings: Tensor,
        *,
        images: Optional[Tensor] = None,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Tensor]:
        per_band = image_embeddings.ndim == 5
        if per_band:
            batch, bands, channels, height, width = image_embeddings.shape
            flat = image_embeddings.reshape(batch * bands, channels, height, width)
            flat_images = self._flatten_images(images, batch=batch, bands=bands) if images is not None else None
        elif image_embeddings.ndim == 4:
            flat = image_embeddings
            batch = bands = None
            flat_images = images
            if flat_images is not None and flat_images.ndim == 3:
                flat_images = flat_images.unsqueeze(1)
        else:
            raise ValueError(f"SamCellectDecoder expects 4D or 5D features, got {tuple(image_embeddings.shape)}")

        out = self.stem(flat)
        out = self.up1(out)
        ds2 = self.up2(out)
        ds3 = self.up3(ds2)
        out = self.up4(ds3)
        out = self.refine(out)

        confidence = self.confidence_head(out)
        target_size = tuple(output_size) if output_size is not None else tuple(confidence.shape[-2:])
        if tuple(confidence.shape[-2:]) != target_size:
            confidence = self._crop_to_output_size(confidence, target_size)
            out = self._crop_to_output_size(out, target_size)

        shape_input = torch.cat([out, confidence], dim=1)
        raw_shape = self.shape_head(self.shape_refine(shape_input))
        shape = self._shape_from_raw(raw_shape)

        outputs = {
            "confidence": confidence,
            "shape": shape,
        }
        if per_band:
            outputs = {
                key: value.reshape(batch, bands, *value.shape[1:])
                for key, value in outputs.items()
            }
        return outputs

    def _shape_from_raw(self, raw_shape: Tensor) -> Tensor:
        if self.shape_channels >= 2:
            axes = F.softplus(raw_shape[:, :2]) + 1e-3
            return torch.cat([axes, raw_shape[:, 2:]], dim=1)
        return F.softplus(raw_shape) + 1e-3

    @staticmethod
    def _crop_to_output_size(x: Tensor, output_size: Tuple[int, int]) -> Tensor:
        """Crop the top-left valid image region after SAM encoder padding."""

        target_h, target_w = int(output_size[0]), int(output_size[1])
        height, width = int(x.shape[-2]), int(x.shape[-1])
        if target_h > height or target_w > width:
            raise ValueError(
                f"decoder output size {(height, width)} is smaller than requested output size {(target_h, target_w)}"
            )
        return x[..., :target_h, :target_w]

    @staticmethod
    def _flatten_images(images: Tensor, *, batch: int, bands: int) -> Tensor:
        if images.ndim != 4:
            raise ValueError(f"per-band decoder images must be [B, band, H, W], got {tuple(images.shape)}")
        if images.shape[0] != batch or images.shape[1] != bands:
            raise ValueError(
                f"image shape {tuple(images.shape[:2])} does not match feature batch/bands {(batch, bands)}"
            )
        return images.reshape(batch * bands, 1, images.shape[-2], images.shape[-1])
