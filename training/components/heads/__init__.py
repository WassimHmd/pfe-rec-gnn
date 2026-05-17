from .base import ScoringHead
from .mlp import MLPHead


def build_head(cfg):
    if cfg.head == "mlp":
        return MLPHead(
            d_model=cfg.d_model,
            hidden=cfg.head_hidden,
            num_layers=cfg.head_layers,
            dropout=cfg.dropout,
        )
    raise ValueError(f"Unknown head: {cfg.head}")
