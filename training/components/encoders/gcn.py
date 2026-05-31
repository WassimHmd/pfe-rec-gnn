"""GCN encoder — graph collaborative filtering baseline.

Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional
Networks", ICLR 2017, adapted for user-item recommendation.

2-layer GCN with learnable weight matrices and ReLU activations.
Sum aggregation (via symmetric-normalised adjacency, no mean-pooling
across layers unlike LightGCN).  Only positive user→book interactions
from the training split are used as graph edges.

Exposes the same .encode() / .head() interface as HGT4Rec so eval.py
works without modification.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .lightgcn import build_norm_adj   # reuse the same norm-adj builder


class GCNRec(nn.Module):
    """2-layer GCN with weight matrices for user-book recommendation."""

    def __init__(
        self,
        n_users:    int,
        n_books:    int,
        d_model:    int,
        num_layers: int,
        norm_adj:   Tensor,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.n_users   = n_users
        self.n_books   = n_books
        self.num_layers = num_layers

        self.user_emb = nn.Embedding(n_users, d_model)
        self.book_emb = nn.Embedding(n_books, d_model)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.book_emb.weight, std=0.01)

        # One weight matrix per layer (shared across user/book sides).
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("norm_adj", norm_adj)

    # ── propagation ───────────────────────────────────────────────────────────

    def _propagate(self) -> tuple[Tensor, Tensor]:
        h = torch.cat([self.user_emb.weight, self.book_emb.weight], dim=0)
        for i, layer in enumerate(self.layers):
            h = torch.sparse.mm(self.norm_adj, h)   # sum aggregation (normalised)
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h[: self.n_users], h[self.n_users :]

    # ── public API (matches HGT4Rec) ─────────────────────────────────────────

    def encode(self, _data=None) -> dict[str, Tensor]:
        h_user, h_book = self._propagate()
        return {"user": h_user, "book": h_book}

    def head(self, h_user: Tensor, h_book: Tensor) -> Tensor:
        return (h_user * h_book).sum(dim=-1)

    # ── BPR forward (training) ────────────────────────────────────────────────

    def forward(
        self,
        user_idx:     Tensor,
        pos_book_idx: Tensor,
        neg_book_idx: Tensor,
    ) -> tuple[Tensor, Tensor]:
        h_user, h_book = self._propagate()
        h_u   = h_user[user_idx]
        h_pos = h_book[pos_book_idx]
        h_neg = h_book[neg_book_idx]
        return (h_u * h_pos).sum(-1), (h_u * h_neg).sum(-1)
