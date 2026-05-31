"""GAT encoder — Graph Attention Network baseline.

Veličković et al., "Graph Attention Networks", ICLR 2018,
adapted for user-item recommendation.

Multi-head attention on the bipartite user-book graph (positive train
interactions only).  Unlike GCN, each neighbour's contribution is weighted
by a learned attention score instead of a fixed normalisation constant.

Architecture:
  Layer 1 : GATConv(d, d//num_heads, heads=num_heads, concat=True)  → d
  Layer 2+: GATConv(d, d,            heads=1,         concat=False) → d
  Activation: ELU between layers (standard for GAT).
  Scoring: dot product (same as LightGCN / GCN baselines).

Exposes the same .encode() / .head() interface as HGT4Rec so eval.py
works without modification.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv


def build_bipartite_edge_index(
    user_idx: np.ndarray,
    book_idx: np.ndarray,
    n_users:  int,
) -> Tensor:
    """Undirected bipartite edge_index (user→book + book→user).

    Node ids: users 0..n_users-1, books n_users..n_users+n_books-1.
    Returns shape (2, 2*E).
    """
    book_idx_shifted = book_idx + n_users
    src = np.concatenate([user_idx,         book_idx_shifted])
    dst = np.concatenate([book_idx_shifted, user_idx])
    return torch.from_numpy(np.stack([src, dst])).long()


class GATRec(nn.Module):
    """GAT recommendation model."""

    def __init__(
        self,
        n_users:         int,
        n_books:         int,
        d_model:         int,
        num_layers:      int,
        num_heads:       int,
        edge_index:      Tensor,
        dropout:         float = 0.1,
        use_checkpoint:  bool  = False,
    ):
        super().__init__()
        self.n_users   = n_users
        self.n_books   = n_books
        self.num_layers = num_layers

        self.user_emb = nn.Embedding(n_users, d_model)
        self.book_emb = nn.Embedding(n_books, d_model)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.book_emb.weight, std=0.01)

        # Intermediate layers: multi-head concat → output stays at d_model
        # Final layer: single head, no concat → output d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        head_dim = d_model // num_heads

        convs = []
        for i in range(num_layers):
            if i < num_layers - 1:
                convs.append(GATConv(d_model, head_dim,
                                     heads=num_heads, concat=True,
                                     dropout=dropout, add_self_loops=False))
            else:
                convs.append(GATConv(d_model, d_model,
                                     heads=1, concat=False,
                                     dropout=dropout, add_self_loops=False))
        self.convs          = nn.ModuleList(convs)
        self.dropout        = nn.Dropout(dropout)
        self.use_checkpoint = use_checkpoint

        self.register_buffer("edge_index", edge_index)

    # ── propagation ───────────────────────────────────────────────────────────

    def _propagate(self) -> tuple[Tensor, Tensor]:
        h = torch.cat([self.user_emb.weight, self.book_emb.weight], dim=0)
        for i, conv in enumerate(self.convs):
            if self.use_checkpoint and self.training:
                # Recompute activations during backward instead of storing them.
                # Halves peak memory at ~33% extra compute cost.
                ei = self.edge_index
                h = checkpoint(conv, h, ei, use_reentrant=False)
            else:
                h = conv(h, self.edge_index)
            if i < len(self.convs) - 1:
                h = F.elu(h)
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
