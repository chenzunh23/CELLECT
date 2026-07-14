from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConditionalStyleAdapter(nn.Module):
    """Zero-initialized ViT residual adapter conditioned by a style prompt."""

    def __init__(self, embed_dim: int, *, prompt_dim: int, adapter_dim: int) -> None:
        super().__init__()
        if prompt_dim <= 0 or adapter_dim <= 0:
            raise ValueError("prompt_dim and adapter_dim must be positive")
        self.norm = nn.LayerNorm(embed_dim)
        self.feature_down = nn.Linear(embed_dim, adapter_dim)
        self.prompt_down = nn.Linear(prompt_dim, adapter_dim, bias=False)
        self.activation = nn.GELU()
        self.up = nn.Linear(adapter_dim, embed_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: Tensor, prompt: Optional[Tensor]) -> Tensor:
        if prompt is None:
            raise ValueError("ConditionalStyleAdapter requires a style prompt")
        if prompt.ndim != 2 or prompt.shape[0] != x.shape[0]:
            raise ValueError(
                f"style prompt must be [B, D] with B={x.shape[0]}, got {tuple(prompt.shape)}"
            )
        hidden = self.feature_down(self.norm(x))
        hidden = hidden + self.prompt_down(prompt).to(dtype=hidden.dtype)[:, None, None, :]
        return x + self.up(self.activation(hidden))


class ImageStyleRouter(nn.Module):
    """Infer a sample-level processed-image mixture from multi-band texture."""

    def __init__(self, *, num_bands: int) -> None:
        super().__init__()
        self.num_bands = int(num_bands)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            _RouterBlock(16, 24),
            _RouterBlock(24, 32),
            _RouterBlock(32, 48),
            nn.Conv2d(48, 32, kernel_size=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != self.num_bands:
            raise ValueError(f"style router expected [B, {self.num_bands}, H, W], got {tuple(image.shape)}")
        image = torch.nan_to_num(image.float()).clamp(-5.0, 5.0)
        router_input = torch.stack(
            (
                image,
                image - F.avg_pool2d(image, kernel_size=3, stride=1, padding=1),
                image - F.avg_pool2d(image, kernel_size=7, stride=1, padding=3),
            ),
            dim=2,
        ).reshape(image.shape[0] * self.num_bands, 3, *image.shape[-2:])
        features = self.features(router_input)
        mean = features.mean(dim=(-2, -1))
        std = features.float().var(dim=(-2, -1), unbiased=False).add(1e-6).sqrt().to(mean.dtype)
        per_band = torch.cat((mean, std), dim=1).reshape(image.shape[0], self.num_bands, -1)
        sample = torch.cat(
            (
                per_band.mean(dim=1),
                per_band.float().var(dim=1, unbiased=False).add(1e-6).sqrt().to(per_band.dtype),
            ),
            dim=1,
        )
        return self.classifier(sample).squeeze(1)


class _RouterBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(8 if out_channels % 8 == 0 else 4, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
