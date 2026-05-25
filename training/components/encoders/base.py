"""Encoder ABC. Any encoder swappable in must satisfy this interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import Tensor, nn


class Encoder(nn.Module, ABC):
    """Take per-type input features + heterogeneous edge index dict → per-type hidden dict.

    The optional `t_dict` argument carries per-node-type timestamps for encoders that
    use them (currently only HGTRTEEncoder). Encoders that ignore time may omit it
    from their signature via `**_` or document a no-op default.
    """

    @abstractmethod
    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[tuple, Tensor],
        t_dict: Dict[str, Tensor] | None = None,
    ) -> Dict[str, Tensor]:
        ...
