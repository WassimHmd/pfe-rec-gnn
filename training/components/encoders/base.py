"""Encoder ABC. Any encoder swappable in must satisfy this interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import Tensor, nn


class Encoder(nn.Module, ABC):
    """Take per-type input features + heterogeneous edge index dict → per-type hidden dict."""

    @abstractmethod
    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[tuple, Tensor],
    ) -> Dict[str, Tensor]:
        ...
