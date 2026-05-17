from .base import Encoder
from .hgt import HGTEncoder


def build_encoder(cfg, metadata, in_dims):
    if cfg.encoder == "hgt":
        return HGTEncoder(
            metadata=metadata,
            in_dims=in_dims,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )
    raise ValueError(f"Unknown encoder: {cfg.encoder}")
