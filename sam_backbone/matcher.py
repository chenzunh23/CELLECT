from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class AstroMatchNet2D(nn.Module):
    """CELLECT-style EX/EN matcher without division features."""

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
        pair_dim = self.feature_dim * 4 + 2 + self.shape_feature_dim
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
        none_logit = self.none_head(hidden.max(dim=1).values)
        return torch.cat([match_logits, none_logit], dim=1)


EXNet2D = AstroMatchNet2D
ENNet2D = AstroMatchNet2D
