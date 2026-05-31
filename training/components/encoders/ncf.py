"""NCF encoder — Neural Collaborative Filtering baseline.

He et al., "Neural Collaborative Filtering", WWW 2017.

MLP variant: user_emb ∥ book_emb → Linear(2d,128) → ReLU → Linear(128,1).
No graph structure — pure ID-embedding + MLP interaction modelling.

Exposes the same .encode() / .head() interface as HGT4Rec so eval.py
works without modification.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class NCF(nn.Module):
    """NCF-MLP: concatenation-based MLP scoring over ID embeddings."""

    def __init__(
        self,
        n_users:    int,
        n_books:    int,
        d_model:    int,
        hidden:     int   = 128,
        num_layers: int   = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, d_model)
        self.book_emb = nn.Embedding(n_books, d_model)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.book_emb.weight, std=0.01)

        # MLP: [2*d → hidden → … → 1]
        layers: list[nn.Module] = []
        in_dim = 2 * d_model
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    # ── public API (matches HGT4Rec) ─────────────────────────────────────────

    def encode(self, _data=None) -> dict[str, Tensor]:
        """Return raw ID embeddings — no graph propagation."""
        return {
            "user": self.user_emb.weight,
            "book": self.book_emb.weight,
        }

    def head(self, h_user: Tensor, h_book: Tensor) -> Tensor:
        """MLP scoring over [h_user ∥ h_book]."""
        return self.mlp(torch.cat([h_user, h_book], dim=-1)).squeeze(-1)

    # ── BCE forward (training) ────────────────────────────────────────────────

    def forward(self, user_idx: Tensor, book_idx: Tensor) -> Tensor:
        h_u = self.user_emb(user_idx)
        h_b = self.book_emb(book_idx)
        return self.mlp(torch.cat([h_u, h_b], dim=-1)).squeeze(-1)
