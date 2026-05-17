"""MLP scoring head over [h_user ∥ h_book]."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import ScoringHead


class MLPHead(ScoringHead):
    def __init__(self, d_model: int, hidden: int, num_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 2 * d_model
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, h_user: Tensor, h_book: Tensor) -> Tensor:
        return self.mlp(torch.cat([h_user, h_book], dim=-1)).squeeze(-1)
