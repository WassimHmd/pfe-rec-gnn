"""Drop edge types from the HeteroData per the Config booleans.

Removes the forward edge type AND its T.ToUndirected-added reverse counterpart in lockstep.
Returns a NEW HeteroData object; the input is not mutated.
"""

from __future__ import annotations

from typing import Iterable

from torch_geometric.data import HeteroData


# Map of config flag → forward edge keys it controls.
EDGE_FLAG_MAP: dict[str, list[tuple[str, str, str]]] = {
    "use_shelved_edges":               [("user",   "SHELVED",      "book")],
    "use_review_edges":                [("user",   "WROTE",        "review"),
                                        ("review", "REVIEWS",      "book")],
    "use_genre_edges":                 [("book",   "HAS_GENRE",    "genre")],
    "use_shelf_edges":                 [("book",   "HAS_SHELF",    "shelf")],
    "use_authored_by_edges":           [("book",   "AUTHORED_BY",  "author")],
    "use_edition_of_edges":            [("book",   "EDITION_OF",   "work")],
    "use_lang_format_publisher_edges": [("book",   "IN_LANGUAGE",  "language"),
                                        ("book",   "IN_FORMAT",    "format"),
                                        ("book",   "PUBLISHED_BY", "publisher")],
}


def _reverse_key(key: tuple[str, str, str]) -> tuple[str, str, str]:
    s, r, d = key
    return (d, f"rev_{r}", s)


def filter_graph(data: HeteroData, cfg) -> HeteroData:
    """Return a copy of `data` with disabled edge types removed."""
    out = data.clone()

    to_drop: list[tuple[str, str, str]] = []
    for flag, fwd_keys in EDGE_FLAG_MAP.items():
        if not getattr(cfg, flag):
            for k in fwd_keys:
                to_drop.append(k)
                to_drop.append(_reverse_key(k))

    for k in to_drop:
        if k in out.edge_types:
            del out[k]

    return out
