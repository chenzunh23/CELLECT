from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _to_odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _nearest_odd(value: float) -> int:
    rounded = max(1, int(round(float(value))))
    if rounded % 2 == 1:
        return rounded
    lower = max(1, rounded - 1)
    upper = rounded + 1
    return lower if abs(float(value) - lower) <= abs(upper - float(value)) else upper


def _center_crop_or_pad(x: Tensor, size: int) -> Tensor:
    size = int(size)
    if size <= 0 or size % 2 == 0:
        raise ValueError(f"size must be a positive odd integer, got {size}")
    height, width = int(x.shape[-2]), int(x.shape[-1])
    crop_h = min(height, size)
    crop_w = min(width, size)
    y0 = max(0, height // 2 - crop_h // 2)
    x0 = max(0, width // 2 - crop_w // 2)
    x = x[..., y0 : y0 + crop_h, x0 : x0 + crop_w]
    pad_h = size - int(x.shape[-2])
    pad_w = size - int(x.shape[-1])
    if pad_h > 0 or pad_w > 0:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        x = F.pad(x, (left, right, top, bottom))
    return x


def _normalize_psf(psf: Tensor, eps: float) -> Tensor:
    psf = torch.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0)
    normalizer = psf.sum(dim=(-2, -1), keepdim=True)
    normalizer = torch.where(
        normalizer.abs() < float(eps),
        torch.ones_like(normalizer),
        normalizer,
    )
    return psf / normalizer


def flatten_psf_stamps(
    psf_stamps: Optional[Tensor],
    *,
    batch: Optional[int],
    bands: Optional[int],
    flat_batch: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str = "psf_stamps",
) -> Optional[Tensor]:
    """Flatten PSF stamps to ``[flat_batch, 1, P, P]``.

    Supported inputs are ``[B, band, P, P]``, ``[B, band, 1, P, P]``,
    ``[flat_batch, P, P]`` and ``[flat_batch, 1, P, P]``.
    """

    if psf_stamps is None:
        return None
    psf = psf_stamps.to(device=device, dtype=dtype, non_blocking=True)
    if batch is not None and bands is not None:
        if psf.ndim == 4 and tuple(psf.shape[:2]) == (batch, bands):
            psf = psf.reshape(flat_batch, 1, psf.shape[-2], psf.shape[-1])
        elif psf.ndim == 5 and tuple(psf.shape[:3]) == (batch, bands, 1):
            psf = psf.reshape(flat_batch, 1, psf.shape[-2], psf.shape[-1])
        else:
            raise ValueError(
                f"{name} must have shape [B, band, P, P] or [B, band, 1, P, P], "
                f"got {tuple(psf.shape)} for B={batch}, band={bands}"
            )
    elif psf.ndim == 3 and int(psf.shape[0]) == flat_batch:
        psf = psf[:, None]
    elif psf.ndim == 4 and int(psf.shape[0]) == flat_batch and int(psf.shape[1]) == 1:
        pass
    else:
        raise ValueError(
            f"{name} must have shape [flat_batch, P, P] or [flat_batch, 1, P, P], "
            f"got {tuple(psf.shape)} for flat_batch={flat_batch}"
        )
    return psf.contiguous()


def flatten_psf_native_sizes(
    psf_native_sizes: Optional[Tensor],
    *,
    batch: Optional[int],
    bands: Optional[int],
    flat_batch: int,
    device: torch.device,
) -> Optional[Tensor]:
    """Flatten optional native PSF stamp sizes to ``[flat_batch]``."""

    if psf_native_sizes is None:
        return None
    sizes = psf_native_sizes.to(device=device, dtype=torch.long, non_blocking=True)
    if batch is not None and bands is not None:
        if sizes.ndim == 2 and tuple(sizes.shape) == (batch, bands):
            sizes = sizes.reshape(flat_batch)
        elif sizes.ndim == 1 and int(sizes.shape[0]) == batch:
            sizes = sizes[:, None].expand(batch, bands).reshape(flat_batch)
        else:
            raise ValueError(
                f"psf_native_sizes must have shape [B] or [B, band], got {tuple(sizes.shape)} "
                f"for B={batch}, band={bands}"
            )
    elif sizes.ndim != 1 or int(sizes.shape[0]) != flat_batch:
        raise ValueError(f"psf_native_sizes must have shape [{flat_batch}], got {tuple(sizes.shape)}")
    return sizes.contiguous()


class PsfConditionMapEncoder(nn.Module):
    """Encode a PSF stamp as a dense condition map at a feature resolution."""

    def __init__(
        self,
        *,
        out_channels: int,
        hidden_dim: int = 32,
        in_channels: int = 1,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.body = nn.Sequential(
            nn.Conv2d(int(in_channels), int(hidden_dim), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), int(out_channels), kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, psf_stamp: Tensor, spatial_size: tuple[int, int]) -> Tensor:
        if psf_stamp.ndim == 3:
            psf_stamp = psf_stamp[:, None]
        if psf_stamp.ndim != 4:
            raise ValueError(f"Expected psf_stamp shape [B,P,P] or [B,1,P,P], got {tuple(psf_stamp.shape)}")
        psf_stamp = _normalize_psf(psf_stamp, self.eps)
        psf_map = F.interpolate(
            psf_stamp,
            size=tuple(int(v) for v in spatial_size),
            mode="bilinear",
            align_corners=False,
        )
        return self.body(psf_map)


class PsfChannelCrossAttentionBlock(nn.Module):
    """Residual channel cross-attention from image features to PSF condition maps."""

    def __init__(
        self,
        *,
        channels: int,
        num_heads: int,
        hidden_dim: int = 32,
        bias: bool = False,
    ) -> None:
        super().__init__()
        channels = int(channels)
        num_heads = int(num_heads)
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.channels = channels
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.condition_map = PsfConditionMapEncoder(
            out_channels=channels,
            hidden_dim=int(hidden_dim),
        )
        self.q = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=bias,
        )
        self.kv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(
            channels * 2,
            channels * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        nn.init.zeros_(self.project_out.weight)
        if self.project_out.bias is not None:
            nn.init.zeros_(self.project_out.bias)

    def forward(self, x: Tensor, psf_stamp: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x shape [B,C,H,W], got {tuple(x.shape)}")
        batch, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")
        if psf_stamp.shape[0] != batch:
            raise ValueError(f"Expected PSF batch {batch}, got {psf_stamp.shape[0]}")

        condition = self.condition_map(psf_stamp, spatial_size=(height, width))
        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(condition))
        k, v = kv.chunk(2, dim=1)

        channels_per_head = channels // self.num_heads
        q = q.reshape(batch, self.num_heads, channels_per_head, height * width)
        k = k.reshape(batch, self.num_heads, channels_per_head, height * width)
        v = v.reshape(batch, self.num_heads, channels_per_head, height * width)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = (q @ k.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)
        delta = attention @ v
        delta = delta.reshape(batch, channels, height, width)
        return x + self.project_out(delta)


class PsfDepthwiseConvBlock(nn.Module):
    """Residual PSF depthwise convolution with stride-aware PSF downsampling.

    The PSF stamp is interpreted in output-image pixels.  For a feature map with
    stride ``feature_stride`` relative to the output image, this block downsamples
    the PSF by that stride, normalizes its flux, then center-pads/crops it to a
    common odd kernel size for grouped convolution.
    """

    def __init__(
        self,
        *,
        channels: int,
        feature_stride: int,
        min_kernel_size: int = 3,
        max_kernel_size: Optional[int] = None,
        bias: bool = False,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.feature_stride = int(feature_stride)
        self.min_kernel_size = _to_odd(int(min_kernel_size))
        self.max_kernel_size = None if max_kernel_size is None else _to_odd(int(max_kernel_size))
        self.eps = float(eps)
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if self.feature_stride <= 0:
            raise ValueError(f"feature_stride must be positive, got {feature_stride}")
        if self.max_kernel_size is not None and self.max_kernel_size < self.min_kernel_size:
            raise ValueError(
                f"max_kernel_size={self.max_kernel_size} must be >= min_kernel_size={self.min_kernel_size}"
            )
        self.project_out = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=bias)
        nn.init.zeros_(self.project_out.weight)
        if self.project_out.bias is not None:
            nn.init.zeros_(self.project_out.bias)

    def _kernel_size(self, native_size: int) -> int:
        native = int(native_size)
        size = _nearest_odd(native / float(self.feature_stride))
        size = max(self.min_kernel_size, size)
        if self.max_kernel_size is not None:
            size = min(size, self.max_kernel_size)
            size = _to_odd(size)
        return size

    def _make_kernel(self, psf_stamp: Tensor, psf_native_sizes: Optional[Tensor]) -> Tensor:
        if psf_stamp.ndim == 3:
            psf_stamp = psf_stamp[:, None]
        if psf_stamp.ndim != 4:
            raise ValueError(f"Expected psf_stamp shape [B,P,P] or [B,1,P,P], got {tuple(psf_stamp.shape)}")
        batch = int(psf_stamp.shape[0])
        max_native = max(int(psf_stamp.shape[-2]), int(psf_stamp.shape[-1]))
        if psf_native_sizes is None:
            native_sizes = [max_native] * batch
        else:
            if psf_native_sizes.ndim != 1 or int(psf_native_sizes.shape[0]) != batch:
                raise ValueError(
                    f"psf_native_sizes must have shape [{batch}], got {tuple(psf_native_sizes.shape)}"
                )
            native_sizes = [int(v) for v in psf_native_sizes.detach().cpu().tolist()]
        native_sizes = [max(1, min(_to_odd(size), max_native)) for size in native_sizes]
        target_sizes = [self._kernel_size(size) for size in native_sizes]
        common_target = max(target_sizes)

        kernels: list[Tensor] = []
        for index, (native_size, target_size) in enumerate(zip(native_sizes, target_sizes)):
            stamp = _center_crop_or_pad(psf_stamp[index : index + 1], native_size)
            stamp = _normalize_psf(stamp, self.eps)
            if target_size != native_size:
                stamp = F.interpolate(stamp, size=(target_size, target_size), mode="area")
            stamp = _center_crop_or_pad(stamp, common_target)
            kernels.append(_normalize_psf(stamp, self.eps))
        return torch.cat(kernels, dim=0)

    def forward(self, x: Tensor, psf_stamp: Tensor, psf_native_sizes: Optional[Tensor] = None) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x shape [B,C,H,W], got {tuple(x.shape)}")
        batch, channels, _height, _width = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")
        if psf_stamp.shape[0] != batch:
            raise ValueError(f"Expected PSF batch {batch}, got {psf_stamp.shape[0]}")

        kernel = self._make_kernel(psf_stamp, psf_native_sizes)
        kernel_size = int(kernel.shape[-1])
        kernel = kernel.expand(batch, channels, kernel_size, kernel_size)
        kernel = kernel.reshape(batch * channels, 1, kernel_size, kernel_size)
        grouped_x = x.reshape(1, batch * channels, x.shape[-2], x.shape[-1])
        convolved = F.conv2d(
            grouped_x,
            kernel,
            padding=kernel_size // 2,
            groups=batch * channels,
        )
        convolved = convolved.reshape(batch, channels, x.shape[-2], x.shape[-1])
        return x + self.project_out(convolved - x)


def normalize_decoder_psf_stages(stages: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(stages, str):
        raw = [item.strip() for item in stages.split(",")]
    else:
        raw = [str(item).strip() for item in stages]
    allowed = {"stem", "up1", "up2", "up3", "up4", "refine"}
    out: list[str] = []
    for stage in raw:
        if not stage or stage.lower() == "none":
            continue
        if stage not in allowed:
            raise ValueError(f"unknown decoder PSF stage {stage!r}; expected one of {sorted(allowed)}")
        if stage not in out:
            out.append(stage)
    return tuple(out)
