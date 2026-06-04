"""
2D CELLECT-style modules for multi-band astronomical source segmentation.

This file is intentionally independent from the original 3D CELLECT modules.
Main differences from the microscopy model:
  - input channels are photometric bands, not adjacent time frames;
  - the backbone is 2D;
  - no division branch is produced or trained;
  - confidence maps are still ordinal center maps with 0..4 levels;
  - EX/EN matching modules are retained without division features/logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


ArrayLike = Union[np.ndarray, Tensor]


def read_fits_bands(
    paths: Union[str, Sequence[str]],
    hdu: int = 0,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Read one multi-band FITS cube or a list of single-band FITS files.

    Returns
    -------
    np.ndarray
        Array in CHW layout, where C is the number of bands.
    """
    try:
        from astropy.io import fits
    except Exception as exc:
        raise RuntimeError("read_fits_bands requires astropy.") from exc

    if isinstance(paths, str):
        data = fits.getdata(paths, hdu).astype(dtype, copy=False)
        if data.ndim == 2:
            return data[None]
        if data.ndim == 3:
            return data
        raise ValueError(f"Expected 2D image or 3D cube FITS data, got shape {data.shape}")

    bands = []
    for path in paths:
        data = fits.getdata(path, hdu).astype(dtype, copy=False)
        if data.ndim != 2:
            raise ValueError(f"Expected single-band 2D FITS data for {path}, got {data.shape}")
        bands.append(data)
    return np.stack(bands, axis=0)


def astro_zscale_preprocess(
    x: ArrayLike,
    *,
    contrast: float = 0.25,
    clip_sigma: float = 3.0,
    sigma_iters: int = -1,
    z_clip: Optional[Tuple[float, float]] = (-5.0, 10.0),
) -> Tensor:
    """Astropy zscale + sigma-clipped normalization for CHW or BCHW data.

    This follows the spirit of the local SAM astro_preprocess implementation:
    normalize each image/band independently, replace non-finite pixels safely,
    use astropy statistics, and produce float32 tensors. The difference is that
    clipping limits come from astropy.visualization.ZScaleInterval instead of
    a median-plus-sigma upper bound.
    """
    try:
        from astropy.stats import sigma_clipped_stats
        from astropy.visualization import ZScaleInterval
    except Exception as exc:
        raise RuntimeError("astro_zscale_preprocess requires astropy.") from exc

    if not torch.is_tensor(x):
        x = torch.from_numpy(np.asarray(x))
    x = x.to(dtype=torch.float32)

    squeeze_batch = x.ndim == 3
    if squeeze_batch:
        work = x.unsqueeze(0)
    elif x.ndim == 4:
        work = x
    else:
        raise ValueError(f"Expected CHW or BCHW input, got shape {tuple(x.shape)}")

    out = torch.empty_like(work)
    maxiters = None if sigma_iters < 0 else int(sigma_iters)
    interval = ZScaleInterval(contrast=float(contrast))

    for b in range(work.shape[0]):
        for c in range(work.shape[1]):
            vals = work[b, c]
            finite = torch.isfinite(vals)
            if not bool(finite.any()):
                out[b, c] = torch.zeros_like(vals)
                continue

            arr = vals[finite].detach().cpu().numpy().astype(np.float64, copy=False)
            raw_mean = np.mean(arr)
            raw_std = np.std(arr)
            # lo, hi = interval.get_limits(arr)
            lo = raw_mean - 3 * raw_std
            hi = raw_mean + 3 * raw_std
            if not np.isfinite(lo):
                lo = float(np.nanpercentile(arr, 0.5))
            if not np.isfinite(hi) or hi <= lo:
                hi = float(np.nanpercentile(arr, 99.5))
            if hi <= lo:
                hi = lo + 1.0

            lo_t = torch.tensor(float(lo), dtype=vals.dtype, device=vals.device)
            hi_t = torch.tensor(float(hi), dtype=vals.dtype, device=vals.device)
            safe = torch.where(finite, vals, lo_t)
            clipped = torch.clamp(safe, min=float(lo), max=float(hi))

            clipped_arr = clipped[finite].detach().cpu().numpy().astype(np.float64, copy=False)
            mean, _median, std = sigma_clipped_stats(
                clipped_arr,
                sigma=float(clip_sigma),
                maxiters=maxiters,
            )
            if not np.isfinite(mean):
                mean = float(np.nanmean(clipped_arr))
            if not np.isfinite(std) or std <= 0:
                std = float(np.nanstd(clipped_arr))
            if not np.isfinite(std) or std <= 0:
                std = 1.0

            mean_t = torch.tensor(float(mean), dtype=vals.dtype, device=vals.device)
            std_t = torch.tensor(float(std), dtype=vals.dtype, device=vals.device)
            z = (clipped - mean_t) / std_t
            if z_clip is not None:
                z = torch.clamp(z, float(z_clip[0]), float(z_clip[1]))
            out[b, c] = z

    return out[0] if squeeze_batch else out


def _conv_norm_lrelu(in_ch: int, out_ch: int) -> nn.Sequential:
    """CELLECT-style Conv -> InstanceNorm -> LeakyReLU block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch),
        nn.LeakyReLU(),
    )


def _norm_lrelu_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """CELLECT-style InstanceNorm -> LeakyReLU -> Conv block."""
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
    """CELLECT localization upsampling block, translated to 2D."""
    return nn.Sequential(
        nn.InstanceNorm2d(in_ch),
        nn.LeakyReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch),
        nn.LeakyReLU(),
    )


class CenterEnhancementNet2D(nn.Module):
    """2D equivalent of CELLECT's LNet/CEN.

    Input is segmentation logits, segmentation probabilities, and raw bands.
    Output is a 5-channel ordinal confidence map: background level 0 plus
    positive center levels 1..4.
    """

    def __init__(self, in_channels: int, confidence_levels: int = 5, width: int = 16) -> None:
        super().__init__()
        self.lrelu = nn.LeakyReLU()
        self.dropout2d = nn.Dropout2d(p=0.6)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

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


class AstroUNet2D(nn.Module):
    """2D CELLECT-style UNet for multi-band astronomical images.

    Parameters
    ----------
    in_channels:
        Number of photometric bands.
    seg_classes:
        Number of dense segmentation classes. Use 2 for background/source or
        3 if you keep the original CELLECT-style foreground tiers.
    confidence_levels:
        Number of confidence channels. Default 5 = background + four center levels.
    embedding_dim:
        Feature dimension sampled at source centers and passed to EX/EN.
    shape_channels:
        Number of size/shape channels. Default 3 can represent semi-major,
        semi-minor, and angle-like source shape predictions.
    """

    def __init__(
        self,
        in_channels: int,
        seg_classes: int = 2,
        confidence_levels: int = 5,
        embedding_dim: int = 64,
        base_channels: int = 32,
        shape_channels: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.seg_classes = int(seg_classes)
        self.confidence_levels = int(confidence_levels)
        self.embedding_dim = int(embedding_dim)
        self.shape_channels = int(shape_channels)

        self.lrelu = nn.LeakyReLU()
        self.dropout2d = nn.Dropout2d(p=0.6)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Context pathway mirrors CELLECT's residual 3D UNet blocks in 2D.
        self.conv_c1_1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False)
        self.conv_c1_2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False)
        self.lrelu_conv_c1 = _lrelu_conv(base_channels, base_channels)
        self.inorm_c1 = nn.InstanceNorm2d(base_channels)

        self.conv_c2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv_c2s1 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1, bias=False)
        self.conv_c2s2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=1, stride=2, bias=False)
        self.norm_lrelu_conv_c2 = _norm_lrelu_conv(base_channels * 2, base_channels * 2)
        self.inorm_c2 = nn.InstanceNorm2d(base_channels * 2)

        self.conv_c3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv_c3s1 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1, bias=False)
        self.conv_c3s2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=1, stride=2, bias=False)
        self.norm_lrelu_conv_c3 = _norm_lrelu_conv(base_channels * 4, base_channels * 4)
        self.inorm_c3 = nn.InstanceNorm2d(base_channels * 4)

        self.conv_c4 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv_c4s1 = nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1, bias=False)
        self.conv_c4s2 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=1, stride=2, bias=False)
        self.norm_lrelu_conv_c4 = _norm_lrelu_conv(base_channels * 8, base_channels * 8)

        # Localization pathway: nearest upsample blocks, 1x1 reductions,
        # skip concatenation, and deep-supervision logits like CELLECT.
        self.up_l0 = _norm_lrelu_upscale_conv_norm_lrelu(base_channels * 8, base_channels * 4)
        self.conv_l0 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=1, bias=False)
        self.inorm_l0 = nn.InstanceNorm2d(base_channels * 4)

        self.conv_norm_lrelu_l1 = _conv_norm_lrelu(base_channels * 8, base_channels * 8)
        self.conv_l1 = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=1, bias=False)
        self.up_l1 = _norm_lrelu_upscale_conv_norm_lrelu(base_channels * 4, base_channels * 2)

        self.conv_norm_lrelu_l2 = _conv_norm_lrelu(base_channels * 4, base_channels * 4)
        self.conv_l2 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1, bias=False)
        self.up_l2 = _norm_lrelu_upscale_conv_norm_lrelu(base_channels * 2, base_channels)

        self.conv_norm_lrelu_l3 = _conv_norm_lrelu(base_channels * 2, base_channels * 2)
        self.pred_channels = self.seg_classes + self.shape_channels
        self.pred_head = nn.Conv2d(base_channels * 2, self.pred_channels, kernel_size=1, bias=False)
        self.ds2_head = nn.Conv2d(base_channels * 8, self.pred_channels, kernel_size=1, bias=False)
        self.ds3_head = nn.Conv2d(base_channels * 4, self.pred_channels, kernel_size=1, bias=False)

        self.conv_norm_lrelu_f = _conv_norm_lrelu(base_channels * 2, base_channels * 4)
        self.conv_f = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=1, bias=False)
        self.conv_norm_lrelu_f1 = _conv_norm_lrelu(base_channels * 4 + self.pred_channels, base_channels * 4)
        self.conv_f1 = nn.Conv2d(base_channels * 4, embedding_dim, kernel_size=1, bias=False)

        cen_in = seg_classes * 2 + in_channels
        self.cen = CenterEnhancementNet2D(
            cen_in,
            confidence_levels=confidence_levels,
            width=max(2, base_channels // 4),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        if x.ndim != 4:
            raise ValueError(f"AstroUNet2D expects BCHW input, got {tuple(x.shape)}")

        # CELLECT context pathway, translated from 3D to 2D.
        # context_1/2/3 are high-resolution skip tensors used by the
        # localization pathway. The original CELLECT UNet3D has one additional
        # deepest context level; this 2D astronomy variant keeps a shallower
        # 4-level encoder to reduce memory on large FITS mosaics.
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
        out = self.lrelu(self.inorm_c3(out + residual_3))
        context_3 = out

        outt = self.conv_c4(out)
        outt = self.conv_c4s1(outt)
        out = outt + self.conv_c4s2(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout2d(out)
        out = self.norm_lrelu_conv_c4(out)
        out = out + residual_4

        # CELLECT localization pathway: nearest-neighbor upsampling, skip
        # concatenation, 1x1 channel reduction, and intermediate tensors for
        # deep-supervision logits.
        out = self.up_l0(out)
        if out.shape[-2:] != context_3.shape[-2:]:
            out = F.interpolate(out, size=context_3.shape[-2:], mode="nearest")
        out = self.conv_l0(out)
        out = self.lrelu(self.inorm_l0(out))

        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        ds2 = out
        out = self.conv_l1(out)
        out = self.up_l1(out)
        if out.shape[-2:] != context_2.shape[-2:]:
            out = F.interpolate(out, size=context_2.shape[-2:], mode="nearest")

        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds3 = out
        out = self.conv_l2(out)
        out = self.up_l2(out)
        if out.shape[-2:] != context_1.shape[-2:]:
            out = F.interpolate(out, size=context_1.shape[-2:], mode="nearest")

        # Final high-resolution localization level. This mirrors CELLECT's
        # Level 4 localization block: one branch seeds the 64-channel embedding,
        # while the other branch produces the shared dense prediction tensor.
        out = torch.cat([out, context_1], dim=1)
        embedding_seed = self.conv_f(self.conv_norm_lrelu_f(out))
        out = self.conv_norm_lrelu_l3(out)
        out_pred = self.pred_head(out)

        # Deep-supervision merge, aligned with CELLECT's:
        #   shared_prediction = high_res_prediction + upsample(ds2 + ds3).
        # In original CELLECT the shared tensor has channels:
        #   segmentation + division + size.
        # Here division is intentionally removed, so channels are:
        #   segmentation + shape/size.
        ds2_logits = self.ds2_head(ds2)
        ds2_logits = F.interpolate(ds2_logits, size=ds3.shape[-2:], mode="nearest")
        ds3_logits = self.ds3_head(ds3)
        ds_logits = F.interpolate(ds2_logits + ds3_logits, size=out_pred.shape[-2:], mode="nearest")
        dense_pred = out_pred + ds_logits

        # Output split, CELLECT-style:
        #   original u  = seg_layer[:, :3]       -> coarse segmentation logits
        #   original ku = relu(seg_layer[:, 5:]) -> size/radius estimate
        # This astronomy variant has no division logits, so dense_pred is split
        # into segmentation logits and shape/size channels only.
        seg_logits = dense_pred[:, : self.seg_classes]
        raw_shape = dense_pred[:, self.seg_classes :]

        # CEN/LNet alignment:
        # CELLECT feeds segmentation logits, segmentation softmax, and original
        # input into LNet to obtain the enhanced ordinal confidence map.
        seg_prob = F.softmax(seg_logits, dim=1)
        confidence = self.cen(torch.cat([seg_logits, seg_prob, x], dim=1))

        # Embedding alignment:
        # CELLECT concatenates the feature branch with the deep-supervision
        # logits before producing the 64-channel latent map used by EX/EN.
        embedding = self.conv_f1(self.conv_norm_lrelu_f1(torch.cat([embedding_seed, ds_logits], dim=1)))

        if self.shape_channels >= 2:
            axes = F.softplus(raw_shape[:, :2]) + 1e-3
            shape_tail = raw_shape[:, 2:]
            shape_map = torch.cat([axes, shape_tail], dim=1)
        else:
            shape_map = F.softplus(raw_shape) + 1e-3

        return {
            "seg_logits": seg_logits,
            "confidence": confidence,
            "embedding": embedding,
            "shape": shape_map,
        }


class AstroMatchNet2D(nn.Module):
    """EX/EN matcher without division features.

    The same module can be used as:
      EN: same-image duplicate suppression;
      EX: cross-epoch or cross-band/catalog association.

    Forward inputs
    --------------
    anchor_features: [N, C] or [N, 1, C]
    candidate_features: [N, K, C]
    candidate_offsets: [N, K, 2]
    candidate_shape_features: optional [N, K, S]

    Returns
    -------
    Tensor [N, K + 1]
        First K logits are candidate match scores. Last logit is none-of-above.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        candidate_count: int = 5,
        shape_feature_dim: int = 6,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.candidate_count = int(candidate_count)
        self.shape_feature_dim = int(shape_feature_dim)
        pair_dim = feature_dim * 4 + 2 + shape_feature_dim
        self.lrelu = nn.LeakyReLU()
        # Keep CELLECT's MLP style: Linear -> BatchNorm1d -> LeakyReLU.
        self.candidate_mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
        )
        self.match_head = nn.Linear(hidden_dim, 1)
        self.none_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        anchor_features: Tensor,
        candidate_features: Tensor,
        candidate_offsets: Tensor,
        candidate_shape_features: Optional[Tensor] = None,
    ) -> Tensor:
        if anchor_features.ndim == 2:
            anchor_features = anchor_features.unsqueeze(1)
        if anchor_features.ndim != 3 or candidate_features.ndim != 3:
            raise ValueError("Expected anchor [N,C] or [N,1,C], candidates [N,K,C]")
        if candidate_offsets.ndim != 3 or candidate_offsets.shape[-1] != 2:
            raise ValueError("candidate_offsets must have shape [N,K,2]")

        n, k, c = candidate_features.shape
        if c != self.feature_dim:
            raise ValueError(f"candidate feature dim {c} != configured {self.feature_dim}")
        anchor = anchor_features.expand(n, k, c)
        diff = candidate_features - anchor
        cosine = F.normalize(anchor, dim=-1) * F.normalize(candidate_features, dim=-1)

        if candidate_shape_features is None:
            shape = candidate_features.new_zeros(n, k, self.shape_feature_dim)
        else:
            shape = candidate_shape_features
            if shape.shape[-1] != self.shape_feature_dim:
                raise ValueError(
                    f"shape feature dim {shape.shape[-1]} != configured {self.shape_feature_dim}"
                )

        pair = torch.cat([anchor, candidate_features, diff, cosine, candidate_offsets, shape], dim=-1)
        hidden = self.candidate_mlp(pair.reshape(n * k, -1)).reshape(n, k, -1)
        match_logits = self.match_head(hidden.reshape(n * k, -1)).reshape(n, k)
        pooled = hidden.max(dim=1).values
        none_logit = self.none_head(pooled)
        return torch.cat([match_logits, none_logit], dim=1)


EXNet2D = AstroMatchNet2D
ENNet2D = AstroMatchNet2D


@dataclass
class AstroCELLECT2D:
    backbone: AstroUNet2D
    EX: AstroMatchNet2D
    EN: AstroMatchNet2D


def build_astro_cellect2d(
    in_channels: int,
    *,
    seg_classes: int = 2,
    confidence_levels: int = 5,
    embedding_dim: int = 64,
    base_channels: int = 32,
    shape_channels: int = 3,
    candidate_count: int = 5,
    shape_feature_dim: int = 6,
) -> AstroCELLECT2D:
    """Create the 2D backbone plus EX/EN matchers."""
    backbone = AstroUNet2D(
        in_channels=in_channels,
        seg_classes=seg_classes,
        confidence_levels=confidence_levels,
        embedding_dim=embedding_dim,
        base_channels=base_channels,
        shape_channels=shape_channels,
    )
    EX = AstroMatchNet2D(
        feature_dim=embedding_dim,
        candidate_count=candidate_count,
        shape_feature_dim=shape_feature_dim,
    )
    EN = AstroMatchNet2D(
        feature_dim=embedding_dim,
        candidate_count=candidate_count,
        shape_feature_dim=shape_feature_dim,
    )
    return AstroCELLECT2D(backbone=backbone, EX=EX, EN=EN)


class MultiBandAstroCELLECT2D(nn.Module):
    """Shared per-band AstroCELLECT model with EX/EN matchers.

    Unlike ``AstroUNet2D(in_channels=N)`` which fuses bands as input channels
    before detection, this module runs one shared 2D CELLECT backbone on each
    band independently.  The returned dense maps keep an explicit band axis:

    - ``seg_logits``: [B, C, seg_classes, H, W]
    - ``confidence``: [B, C, confidence_levels, H, W]
    - ``embedding``: [B, C, embedding_dim, H, W]
    - ``shape``: [B, C, shape_channels, H, W]

    ``EX`` is used to classify cross-band candidates with the same source id.
    ``EN`` is used to classify same-band candidates and suppress duplicates.
    For single-band data, EX should be disabled by the training loop while EN
    can still be trained if duplicate labels or hard negatives are desired.
    """

    def __init__(
        self,
        *,
        num_bands: int,
        seg_classes: int = 2,
        confidence_levels: int = 5,
        embedding_dim: int = 64,
        base_channels: int = 32,
        shape_channels: int = 3,
        candidate_count: int = 5,
        shape_feature_dim: int = 6,
    ) -> None:
        super().__init__()
        self.num_bands = int(num_bands)
        self.backbone = AstroUNet2D(
            in_channels=1,
            seg_classes=seg_classes,
            confidence_levels=confidence_levels,
            embedding_dim=embedding_dim,
            base_channels=base_channels,
            shape_channels=shape_channels,
        )
        self.EX = AstroMatchNet2D(
            feature_dim=embedding_dim,
            candidate_count=candidate_count,
            shape_feature_dim=shape_feature_dim,
        )
        self.EN = AstroMatchNet2D(
            feature_dim=embedding_dim,
            candidate_count=candidate_count,
            shape_feature_dim=shape_feature_dim,
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        if x.ndim != 4:
            raise ValueError(f"MultiBandAstroCELLECT2D expects BCHW input, got {tuple(x.shape)}")
        batch, bands, height, width = x.shape
        if bands != self.num_bands:
            raise ValueError(f"input has {bands} bands but model was built for {self.num_bands}")

        flat = x.reshape(batch * bands, 1, height, width)
        flat_out = self.backbone(flat)
        out: Dict[str, Tensor] = {}
        for key, value in flat_out.items():
            out[key] = value.reshape(batch, bands, *value.shape[1:])
        return out


class FusedEncoderMultiBandAstroCELLECT2D(nn.Module):
    """Multi-band AstroCELLECT with one fused encoder and per-band heads.

    This is the astronomy counterpart of CELLECT's multi-frame input style:
    all bands enter the backbone together as channels, so the expensive UNet/CEN
    path runs once per cutout.  Lightweight band-conditioned heads then expand
    the fused latent map back to per-band dense outputs, preserving the
    ``[B, band, ...]`` layout needed by EX/EN and triplet losses.
    """

    def __init__(
        self,
        *,
        num_bands: int,
        seg_classes: int = 2,
        confidence_levels: int = 5,
        embedding_dim: int = 64,
        base_channels: int = 32,
        shape_channels: int = 3,
        candidate_count: int = 5,
        shape_feature_dim: int = 6,
    ) -> None:
        super().__init__()
        self.num_bands = int(num_bands)
        self.shape_channels = int(shape_channels)
        self.backbone = AstroUNet2D(
            in_channels=num_bands,
            seg_classes=seg_classes,
            confidence_levels=confidence_levels,
            embedding_dim=embedding_dim,
            base_channels=base_channels,
            shape_channels=shape_channels,
        )
        band_head_channels = max(16, embedding_dim // 2)
        self.band_refine = nn.Sequential(
            nn.Conv2d(embedding_dim + 1, band_head_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(band_head_channels),
            nn.LeakyReLU(),
        )
        self.band_seg_head = nn.Conv2d(band_head_channels, seg_classes, kernel_size=1, bias=False)
        self.band_conf_head = nn.Conv2d(band_head_channels, confidence_levels, kernel_size=1, bias=False)
        self.band_embedding_head = nn.Conv2d(band_head_channels, embedding_dim, kernel_size=1, bias=False)
        self.band_shape_head = nn.Conv2d(band_head_channels, shape_channels, kernel_size=1, bias=False)
        self.band_embedding_bias = nn.Parameter(torch.zeros(num_bands, embedding_dim, 1, 1))
        self.EX = AstroMatchNet2D(
            feature_dim=embedding_dim,
            candidate_count=candidate_count,
            shape_feature_dim=shape_feature_dim,
        )
        self.EN = AstroMatchNet2D(
            feature_dim=embedding_dim,
            candidate_count=candidate_count,
            shape_feature_dim=shape_feature_dim,
        )

    def _shape_from_raw(self, base_shape: Tensor, raw: Tensor) -> Tensor:
        if self.shape_channels >= 2:
            base_axes = torch.clamp(base_shape[:, :2], min=1e-3)
            axes = F.softplus(torch.log(base_axes) + raw[:, :2]) + 1e-3
            tail = base_shape[:, 2:] + raw[:, 2:]
            return torch.cat([axes, tail], dim=1)
        return F.softplus(torch.log(torch.clamp(base_shape, min=1e-3)) + raw) + 1e-3

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        if x.ndim != 4:
            raise ValueError(f"FusedEncoderMultiBandAstroCELLECT2D expects BCHW input, got {tuple(x.shape)}")
        batch, bands, _height, _width = x.shape
        if bands != self.num_bands:
            raise ValueError(f"input has {bands} bands but model was built for {self.num_bands}")

        fused = self.backbone(x)
        seg_logits = []
        confidence = []
        embedding = []
        shape = []
        fused_embedding = fused["embedding"]
        for band in range(bands):
            band_feat = self.band_refine(torch.cat([fused_embedding, x[:, band : band + 1]], dim=1))
            seg_logits.append(fused["seg_logits"] + self.band_seg_head(band_feat))
            confidence.append(fused["confidence"] + self.band_conf_head(band_feat))
            embedding.append(
                fused_embedding
                + self.band_embedding_head(band_feat)
                + self.band_embedding_bias[band].unsqueeze(0)
            )
            shape.append(self._shape_from_raw(fused["shape"], self.band_shape_head(band_feat)))

        return {
            "seg_logits": torch.stack(seg_logits, dim=1),
            "confidence": torch.stack(confidence, dim=1),
            "embedding": torch.stack(embedding, dim=1),
            "shape": torch.stack(shape, dim=1),
        }


def ordinal_confidence_loss(
    logits: Tensor,
    target: Tensor,
    *,
    ignore_index: int = -100,
    pos_weight: float = 32.0,
    weight: Optional[Tensor] = None,
) -> Tensor:
    """CELLECT ``crloss`` for 2D ordinal center-confidence maps.

    logits: [B, L, H, W], where L is usually 5.
    target: [B, H, W] with integer levels 0..L-1 and optional ignore_index.

    This intentionally mirrors ``recoloss.crloss`` rather than a standard
    cumulative ordinal loss. CELLECT first compares background channel 0 with
    every positive confidence channel for ``target >= 1``. It then compares each
    higher channel against the max of lower channels, while masking out lower
    positive rings so that, for example, level-1 pixels are not trained as
    negatives for the level-2/3/4 classifiers.
    """
    if logits.ndim != 4:
        raise ValueError("logits must be [B,L,H,W]")
    if target.ndim != 3:
        raise ValueError("target must be [B,H,W]")
    valid = target != ignore_index
    if weight is not None:
        if weight.shape != target.shape:
            raise ValueError("weight must have the same shape as target")
        weight = weight.to(device=logits.device, dtype=logits.dtype).clamp_min(0.0)
        valid = valid & (weight > 0)
    if not bool(valid.any()):
        return logits.sum() * 0.0

    loss_map = logits.new_zeros(target.shape, dtype=logits.dtype)
    class_weight = logits.new_tensor([1.0, float(pos_weight)])
    levels = logits.shape[1]
    safe_target = torch.where(valid, target, torch.zeros_like(target)).long()

    for channel in range(1, levels):
        binary_logits = torch.cat([logits[:, :1], logits[:, channel : channel + 1]], dim=1)
        binary_target = (safe_target >= 1).long()
        loss_map = loss_map + F.cross_entropy(
            binary_logits,
            binary_target,
            weight=class_weight,
            reduction="none",
        )

    for level in range(2, levels):
        supervise = valid & ((safe_target == 0) | (safe_target >= level))
        if not bool(supervise.any()):
            continue
        for channel in range(level, levels):
            prev = logits[:, :channel].max(dim=1).values.unsqueeze(1)
            curr = logits[:, channel : channel + 1]
            binary_logits = torch.cat([prev, curr], dim=1)
            binary_target = (safe_target >= level).long()
            this_loss = F.cross_entropy(
                binary_logits,
                binary_target,
                weight=class_weight,
                reduction="none",
            )
            loss_map = loss_map + this_loss * supervise.to(dtype=this_loss.dtype)

    if weight is not None:
        return (loss_map * weight)[valid].sum() / weight[valid].sum().clamp_min(1.0)
    return loss_map[valid].mean()


def make_center_confidence_target(
    shape: Tuple[int, int],
    centers_xy: ArrayLike,
    *,
    levels: int = 4,
    device: Optional[torch.device] = None,
    ignore_mask: Optional[Tensor] = None,
) -> Tensor:
    """Build a 2D 0..4 confidence target from accurate catalog centers.

    centers_xy is expected in x,y order. Output is [H,W].
    The target is a Manhattan-distance diamond, matching the original CELLECT
    center_kernel logic in 2D.
    """
    h, w = int(shape[0]), int(shape[1])
    target = torch.zeros((h, w), dtype=torch.long, device=device)
    centers = torch.as_tensor(centers_xy, dtype=torch.float32, device=device)
    if centers.numel() == 0:
        return target
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    for center in centers:
        cx = int(torch.round(center[0]).item())
        cy = int(torch.round(center[1]).item())
        y0, y1 = max(0, cy - levels), min(h, cy + levels + 1)
        x0, x1 = max(0, cx - levels), min(w, cx + levels + 1)
        dist = torch.abs(xx[y0:y1, x0:x1] - cx) + torch.abs(yy[y0:y1, x0:x1] - cy)
        vals = torch.clamp(levels - dist, min=0)
        target[y0:y1, x0:x1] = torch.maximum(target[y0:y1, x0:x1], vals.long())
    if ignore_mask is not None:
        target = target.masked_fill(ignore_mask.to(device=device, dtype=torch.bool), -100)
    return target


def segmentation_loss(
    seg_logits: Tensor,
    target: Tensor,
    *,
    ignore_index: int = -100,
    class_weight: Optional[Tensor] = None,
) -> Tensor:
    """Cross-entropy segmentation loss with sparse/uncertain regions ignored."""
    return F.cross_entropy(seg_logits, target.long(), weight=class_weight, ignore_index=ignore_index)


def sample_center_features(outputs: Dict[str, Tensor], centers_xy: ArrayLike) -> Dict[str, Tensor]:
    """Sample embedding and shape predictions at catalog centers.

    centers_xy is x,y in image pixel coordinates. This helper assumes batch size 1.
    """
    centers = torch.as_tensor(centers_xy, dtype=torch.long, device=outputs["embedding"].device)
    if centers.ndim != 2 or centers.shape[-1] != 2:
        raise ValueError("centers_xy must be [N,2] in x,y order")
    h, w = outputs["embedding"].shape[-2:]
    x = centers[:, 0].clamp(0, w - 1)
    y = centers[:, 1].clamp(0, h - 1)
    return {
        "features": outputs["embedding"][0, :, y, x].transpose(0, 1),
        "shape": outputs["shape"][0, :, y, x].transpose(0, 1),
        "confidence": outputs["confidence"][0, :, y, x].transpose(0, 1),
    }


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
