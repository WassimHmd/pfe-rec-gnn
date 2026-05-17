"""Composition: featurizer + encoder + scoring head.

The model also carries the protocol for the synthetic "supervision" edge type
used by LinkNeighborLoader: it is registered in `data` (with empty edge_index)
so PyG can attach edge_label_index/edge_label to it, but is stripped before
being passed to the encoder so HGT doesn't see it as a real relation.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData

from .components.encoders import build_encoder
from .components.heads import build_head
from .featurizer import Featurizer


SUPERVISION_REL = "supervision"
SUPERVISION_KEY: tuple[str, str, str] = ("user", SUPERVISION_REL, "book")


class HGT4Rec(nn.Module):
    def __init__(self, data: HeteroData, cfg):
        super().__init__()
        self.cfg = cfg

        # Build featurizer first so we know each type's input dim.
        self.featurizer = Featurizer(data, cfg)

        # Encoder metadata: drop the supervision edge type if it's already registered.
        node_types, edge_types = data.metadata()
        edge_types = [e for e in edge_types if e[1] != SUPERVISION_REL]
        metadata = (node_types, edge_types)
        self.encoder = build_encoder(cfg, metadata=metadata, in_dims=self.featurizer.in_dims)
        self.head    = build_head(cfg)

    def encode(self, batch: HeteroData) -> Dict[str, Tensor]:
        x_dict = self.featurizer(batch)
        # Strip the synthetic supervision edge type before HGTConv.
        eidx = {k: v for k, v in batch.edge_index_dict.items() if k[1] != SUPERVISION_REL}
        return self.encoder(x_dict, eidx)

    def forward(self, batch: HeteroData) -> Tensor:
        h_dict = self.encode(batch)
        ei = batch[SUPERVISION_KEY].edge_label_index
        return self.head(h_dict["user"][ei[0]], h_dict["book"][ei[1]])
