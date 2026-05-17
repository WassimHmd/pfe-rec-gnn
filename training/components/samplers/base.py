"""Negative-sampler ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class NegativeSampler(ABC):
    @abstractmethod
    def sample(self, user_idx: Tensor, n_per_pos: int) -> tuple[Tensor, Tensor]:
        """Given a batch of positive user indices, return (user_neg, book_neg) of shape
        (len(user_idx) * n_per_pos,)."""
        ...
