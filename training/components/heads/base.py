"""Scoring head ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class ScoringHead(nn.Module, ABC):
    """(h_user, h_book) → logit. Returns shape (N,)."""

    @abstractmethod
    def forward(self, h_user: Tensor, h_book: Tensor) -> Tensor:
        ...
