"""HGT encoder — stacked HGTConv with per-type input projection + residual."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
from torch_geometric.nn import HGTConv

from .base import Encoder


class HGTEncoder(Encoder):
    def __init__(
        self,
        metadata: tuple,
        in_dims: Dict[str, int],
        d_model: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(in_dim, d_model) for nt, in_dim in in_dims.items()
        })
        self.convs = nn.ModuleList([
            HGTConv(d_model, d_model, metadata, num_heads) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[tuple, Tensor],
    ) -> Dict[str, Tensor]:
        h_dict = {nt: self.input_proj[nt](x) for nt, x in x_dict.items()}
        for conv in self.convs:
            h_new = conv(h_dict, edge_index_dict)
            # Residual + dropout. HGTConv may return only types present in the
            # subgraph; fall back to the previous representation otherwise.
            h_dict = {
                nt: self.dropout(h_new[nt]) + h_dict[nt] if nt in h_new else h_dict[nt]
                for nt in h_dict
            }
        return h_dict
