"""LightGCN encoder — pure collaborative filtering baseline.

He et al., "LightGCN: Simplifying and Powering Graph Convolution Network
for Recommendation", SIGIR 2020.

Only positive user→book edges (RATED_HIGH + READ_UNRATED) are used.
No features, no non-linearity, no weight matrices.  Symmetric-normalised
aggregation repeated L times, then mean-pooled across layers.

Exposes the same .encode() / .head() interface as HGT4Rec so eval.py
works without modification.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def build_norm_adj(user_idx: np.ndarray, book_idx: np.ndarray,
                   n_users: int, n_books: int) -> Tensor:
    """Symmetric-normalised bipartite adjacency as a sparse COO tensor.

    Node ordering: users 0..n_users-1, books n_users..n_users+n_books-1.
    A is the bipartite interaction matrix; we build the full symmetric
    block:  [[0, R], [R^T, 0]] then apply D^{-1/2} A D^{-1/2}.
    """
    N = n_users + n_books
    rows = np.concatenate([user_idx, book_idx + n_users])
    cols = np.concatenate([book_idx + n_users, user_idx])
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N))

    # D^{-1/2}
    deg = np.asarray(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0).astype(np.float32)
    D_inv = sp.diags(d_inv_sqrt)
    norm = (D_inv @ A @ D_inv).tocoo()

    indices = torch.from_numpy(np.stack([norm.row, norm.col])).long()
    values  = torch.from_numpy(norm.data)
    return torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()


class LightGCN(nn.Module):
    """LightGCN model.  Exposes encode() + head() to match HGT4Rec's API."""

    def __init__(
        self,
        n_users: int,
        n_books: int,
        d_model: int,
        num_layers: int,
        norm_adj: Tensor,
    ):
        super().__init__()
        self.n_users   = n_users
        self.n_books   = n_books
        self.num_layers = num_layers

        self.user_emb = nn.Embedding(n_users, d_model)
        self.book_emb = nn.Embedding(n_books, d_model)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.book_emb.weight, std=0.01)

        # Stored as buffer so .to(device) moves it automatically.
        self.register_buffer("norm_adj", norm_adj)

    # ── internal propagation ──────────────────────────────────────────────────

    def _propagate(self) -> tuple[Tensor, Tensor]:
        e0 = torch.cat([self.user_emb.weight, self.book_emb.weight], dim=0)
        all_embs = [e0]
        e = e0
        for _ in range(self.num_layers):
            e = torch.sparse.mm(self.norm_adj, e)
            all_embs.append(e)
        final = torch.stack(all_embs, dim=1).mean(dim=1)
        return final[: self.n_users], final[self.n_users :]

    # ── public API (matches HGT4Rec) ─────────────────────────────────────────

    def encode(self, _data=None) -> dict[str, Tensor]:
        """Return full user + book embeddings.  Ignores HeteroData argument."""
        h_user, h_book = self._propagate()
        return {"user": h_user, "book": h_book}

    def head(self, h_user: Tensor, h_book: Tensor) -> Tensor:
        """Dot-product scoring."""
        return (h_user * h_book).sum(dim=-1)

    # ── BPR forward (used during training only) ───────────────────────────────

    def forward(
        self,
        user_idx: Tensor,
        pos_book_idx: Tensor,
        neg_book_idx: Tensor,
    ) -> tuple[Tensor, Tensor]:
        h_user, h_book = self._propagate()
        h_u   = h_user[user_idx]
        h_pos = h_book[pos_book_idx]
        h_neg = h_book[neg_book_idx]
        return (h_u * h_pos).sum(-1), (h_u * h_neg).sum(-1)
