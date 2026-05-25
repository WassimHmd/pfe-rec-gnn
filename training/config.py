"""Single Config dataclass holding every tunable knob.

Edit values here, or instantiate with overrides:
    cfg = Config(book_use_text=False, d_model=128)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # ── Graph (which structural edge types are kept in the message-passing graph) ─
    use_shelved_edges:                bool = True
    use_review_edges:                 bool = True    # WROTE + REVIEWS
    use_genre_edges:                  bool = True
    use_shelf_edges:                  bool = True
    use_lang_format_publisher_edges:  bool = True
    use_authored_by_edges:            bool = True
    use_edition_of_edges:             bool = True
    use_rated_low_as_neg:             bool = True    # if False, drop RATED_LOW from supervision

    # ── Per-node feature toggles ─────────────────────────────────────────────────
    # Book (TN)
    book_use_text:    bool = True
    book_use_numeric: bool = True
    # Review (TN)
    review_use_text:    bool = True
    review_use_numeric: bool = True
    # Author (MN)
    author_use_text:    bool = True
    author_use_numeric: bool = True
    author_use_id:      bool = True
    # Work (MN)
    work_use_text:    bool = True
    work_use_numeric: bool = True
    work_use_id:      bool = True
    # U-category
    user_use_id:        bool = True
    u_category_use_id:  bool = True   # Genre / Shelf / Language / Format / Publisher

    # ── Relative Temporal Encoding (lightweight: absolute time as feature) ──────
    # Appends sinusoidal encoding of each node's timestamp to its input features.
    # Requires `data[nt].t` to exist in the HeteroData (set by preprocess.ipynb step 10).
    # NOT the layer-level Hu et al. 2020 RTE — just a cheap "is temporal info useful?" ablation.
    use_rte: bool = False
    rte_dim: int  = 32     # sinusoidal encoding width (must be even)

    # ── Encoder ──────────────────────────────────────────────────────────────────
    encoder:    str   = "hgt"
    d_model:    int   = 256
    num_layers: int   = 3
    num_heads:  int   = 8
    dropout:    float = 0.2
    id_dim:     int   = 128

    # ── Scoring head ─────────────────────────────────────────────────────────────
    head:        str = "mlp"
    head_hidden: int = 256
    head_layers: int = 2

    # ── Optimization ─────────────────────────────────────────────────────────────
    lr:                  float = 1e-3
    weight_decay:        float = 1e-4
    batch_size:          int   = 1024
    fanout:              list  = field(default_factory=lambda: [10, 5, 5])
    # NOTE: bf16/fp16 autocast crashes inside pyg-lib's grouped_matmul (fp32-only kernel).
    # Default to "none" until HGT's HeteroLinear can be swapped for an AMP-friendly variant.
    amp_dtype:           str   = "none"      # "bfloat16" | "float16" | "none"
    epochs:              int   = 50
    neg_ratio:           int   = 5
    loss:                str   = "bce"  # "bce" | "bpr" | "infonce"
    infonce_tau:         float = 1.0    # softmax temperature for InfoNCE
    neg_pop_alpha:       float = 0.75   # popularity exponent (word2vec convention); 0.0 = uniform
    early_stop_patience: int   = 5
    grad_clip:           float = 1.0
    num_workers:         int   = 4    # NeighborLoader workers; 0 = main process
    persistent_workers:  bool  = False # keep workers alive between epochs; on Windows
                                       # this can trigger working-set trimming over time

    # ── Evaluation ───────────────────────────────────────────────────────────────
    eval_k:           int   = 10
    work_level_dedup: bool  = True
    eval_max_users:   int   = 0    # 0 = all users; >0 caps users for quick smoke tests
    eval_encode_on_cpu: bool = True   # full-graph HGT encode → CPU (avoids 6GB GPU OOM)

    # ── Reproducibility ──────────────────────────────────────────────────────────
    seed: int = 42

    # ── Paths ────────────────────────────────────────────────────────────────────
    data_path:    str = "data/processed/hetero_data.pt"
    splits_dir:   str = "data/processed/splits"
    runs_root:    str = "training/runs"

    def head_input_dim(self) -> int:
        return 2 * self.d_model
