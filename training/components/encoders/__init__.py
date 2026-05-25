from .base import Encoder
from .hgt import HGTEncoder
from .hgt_rte import HGTRTEEncoder


def build_encoder(cfg, metadata, in_dims):
    if cfg.encoder == "hgt":
        # cfg.use_rte=True swaps in the RTE variant (Hu et al. 2020 §3.4).
        if getattr(cfg, "use_rte", False):
            return HGTRTEEncoder(
                metadata=metadata,
                in_dims=in_dims,
                d_model=cfg.d_model,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
                rte_dim=cfg.rte_dim,
            )
        return HGTEncoder(
            metadata=metadata,
            in_dims=in_dims,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )
    raise ValueError(f"Unknown encoder: {cfg.encoder}")
