"""Uniform random negative sampling: pair each positive user with `n_per_pos` random books.

Phase 1 accepts the small chance of sampling a real positive (false negative). A strict
variant that checks against the positive set can be added later as a separate class.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import NegativeSampler


class RandomNegativeSampler(NegativeSampler):
    def __init__(self, num_books: int, seed: int):
        self.num_books = num_books
        self.gen = torch.Generator().manual_seed(seed)

    def sample(self, user_idx: Tensor, n_per_pos: int) -> tuple[Tensor, Tensor]:
        n = user_idx.shape[0] * n_per_pos
        u_neg = user_idx.repeat_interleave(n_per_pos)
        b_neg = torch.randint(
            high=self.num_books, size=(n,), generator=self.gen, device=user_idx.device
        )
        return u_neg, b_neg
