"""Per-node-type input featurizer.

Concatenates the enabled feature blocks per node type:
  text  (768 dims, from data[nt].x[:, :768])
  numeric (rest of data[nt].x)
  id     (nn.Embedding lookup by global node index)

If everything is disabled for a type, emits a 1-d learned scalar broadcast to
all nodes so the type still has a presence in the encoder's input.

Convention: text occupies the first 768 columns of any stored `x`; numeric the rest.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData


TEXT_DIM = 768


def sinusoidal_time(t: Tensor, dim: int) -> Tensor:
    """Standard transformer-style time encoding.

    Args:
        t:   (N,) float tensor of timestamps (any continuous units; we use years-since-1900)
        dim: output dim (must be even)
    Returns: (N, dim) tensor.
    """
    assert dim % 2 == 0, f"rte_dim must be even, got {dim}"
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    angles = t.float().unsqueeze(-1) * freqs.unsqueeze(0)   # (N, half)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


# (node_type → which config attribute controls each feature block; None means feature N/A)
NODE_FEATURE_CONFIG: Dict[str, Dict[str, str | None]] = {
    "book":      {"text": "book_use_text",     "numeric": "book_use_numeric",     "id": None},
    "review":    {"text": "review_use_text",   "numeric": "review_use_numeric",   "id": None},
    "author":    {"text": "author_use_text",   "numeric": "author_use_numeric",   "id": "author_use_id"},
    "work":      {"text": "work_use_text",     "numeric": "work_use_numeric",     "id": "work_use_id"},
    "user":      {"text": None,                "numeric": None,                   "id": "user_use_id"},
    "genre":     {"text": None,                "numeric": None,                   "id": "u_category_use_id"},
    "shelf":     {"text": None,                "numeric": None,                   "id": "u_category_use_id"},
    "language":  {"text": None,                "numeric": None,                   "id": "u_category_use_id"},
    "format":    {"text": None,                "numeric": None,                   "id": "u_category_use_id"},
    "publisher": {"text": None,                "numeric": None,                   "id": "u_category_use_id"},
}


def _flag(cfg, attr: str | None) -> bool:
    return bool(getattr(cfg, attr)) if attr is not None else False


class Featurizer(nn.Module):
    def __init__(self, data: HeteroData, cfg):
        super().__init__()
        self.cfg = cfg
        self.id_emb = nn.ModuleDict()
        self.fallback = nn.ParameterDict()
        self.in_dims: Dict[str, int] = {}

        for nt in data.node_types:
            spec = NODE_FEATURE_CONFIG[nt]
            stored_dim = data[nt].x.shape[1] if "x" in data[nt] else 0
            text_dim = TEXT_DIM if stored_dim >= TEXT_DIM else 0
            num_dim  = stored_dim - text_dim

            use_text    = _flag(cfg, spec["text"])
            use_numeric = _flag(cfg, spec["numeric"])
            use_id      = _flag(cfg, spec["id"]) or (spec["id"] is None and stored_dim == 0)
            # U-category nodes with no x: ID is the only signal — only honor explicit id flag.
            # Re-evaluate:
            use_id      = _flag(cfg, spec["id"])

            dim = 0
            if use_text:
                dim += text_dim
            if use_numeric:
                dim += num_dim
            if use_id:
                self.id_emb[nt] = nn.Embedding(data[nt].num_nodes, cfg.id_dim)
                dim += cfg.id_dim

            if dim == 0:
                # Every block disabled → 1-d learned scalar broadcast.
                self.fallback[nt] = nn.Parameter(torch.zeros(1))
                dim = 1

            self.in_dims[nt] = dim

    def forward(self, batch: HeteroData) -> Dict[str, Tensor]:
        x_dict: Dict[str, Tensor] = {}

        for nt in batch.node_types:
            spec = NODE_FEATURE_CONFIG[nt]
            stored_x = batch[nt].x if "x" in batch[nt] else None
            n_id     = batch[nt].n_id if "n_id" in batch[nt] else \
                       torch.arange(batch[nt].num_nodes, device=self._device())
            N = n_id.shape[0]

            use_text    = _flag(self.cfg, spec["text"])
            use_numeric = _flag(self.cfg, spec["numeric"])
            use_id      = _flag(self.cfg, spec["id"])

            parts: list[Tensor] = []
            if use_text and stored_x is not None:
                parts.append(stored_x[:, :TEXT_DIM])
            if use_numeric and stored_x is not None:
                parts.append(stored_x[:, TEXT_DIM:])
            if use_id and nt in self.id_emb:
                parts.append(self.id_emb[nt](n_id))

            if not parts:
                parts.append(self.fallback[nt].expand(N, 1))

            x_dict[nt] = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

        return x_dict

    def _device(self):
        for p in self.parameters():
            return p.device
        return torch.device("cpu")
