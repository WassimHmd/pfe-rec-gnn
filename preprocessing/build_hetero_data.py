"""
Assemble all Phase 1 preprocessing artifacts into a single PyG HeteroData object.

Node features:
  U-category (User, Genre, Shelf, Language, Format, Publisher):
      → only num_nodes; ID embedding lives in the model as nn.Embedding
  TN-category (Book, Review):
      → x = [text_embed ∥ numeric]   (frozen)
  MN-category (Author, Work):
      → x = [text_embed ∥ numeric]   (ID part added inside the model)

Edges:
  Structural (all of these go into the training graph for message passing):
      Book → Author      AUTHORED_BY
      Book → Work        EDITION_OF
      Book → Genre       HAS_GENRE
      Book → Shelf       HAS_SHELF
      Book → Publisher   PUBLISHED_BY
      Book → Language    IN_LANGUAGE
      Book → Format      IN_FORMAT
      User → Review      WROTE
      Review → Book      REVIEWS
      User → Book        SHELVED        (intent only, not supervision)
  Supervision (loaded from splits/train.npz — train portion only goes into graph):
      User → Book        RATED_HIGH     (positive)
      User → Book        READ_UNRATED   (positive)
      User → Book        RATED_LOW      (explicit negative)

Reverse edges are added via T.ToUndirected() — HGT treats each direction as a
distinct edge type with its own attention/message projection.

Outputs:
  data/processed/hetero_data.pt   serialized HeteroData
  data/processed/hetero_meta.json node/edge counts + feature dims
"""

import json
import numpy as np
import h5py
import torch
from pathlib import Path
from neo4j import GraphDatabase
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

NEO4J_URI  = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "Niveau99")
NEO4J_DB   = "goodreads-poetry"

DATA_DIR    = Path("data/processed")
SPLITS_DIR  = DATA_DIR / "splits"
OUT_PATH    = DATA_DIR / "hetero_data.pt"
META_PATH   = DATA_DIR / "hetero_meta.json"

EDGE_TYPE_NAMES = {0: "RATED_HIGH", 1: "READ_UNRATED", 2: "RATED_LOW"}


# ── 1. Node features ─────────────────────────────────────────────────────────
def load_idx(name):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def load_h5_embed(fname):
    with h5py.File(DATA_DIR / fname, "r") as hf:
        emb = hf["embedding"][:]
        id_key = [k for k in hf.keys() if k != "embedding"][0]
        ids = [x.decode() for x in hf[id_key][:]]
    return emb, ids


def build_tn_features(text_emb, numeric_arr) -> torch.Tensor:
    assert text_emb.shape[0] == numeric_arr.shape[0], "row count mismatch"
    return torch.from_numpy(np.concatenate([text_emb, numeric_arr], axis=1)).float()


# ── 2. Edge fetching ─────────────────────────────────────────────────────────
def fetch_edges(driver, cypher) -> list[tuple]:
    with driver.session(database=NEO4J_DB) as s:
        return [(r["src"], r["dst"]) for r in s.run(cypher)]


def to_edge_index(edges, src_map, dst_map) -> tuple[torch.Tensor, int]:
    """Map ID-pairs to integer indices; drop any pair where either side is unknown."""
    src_idx, dst_idx, dropped = [], [], 0
    for s, d in edges:
        s_key = str(s); d_key = str(d)
        if s_key in src_map and d_key in dst_map:
            src_idx.append(src_map[s_key])
            dst_idx.append(dst_map[d_key])
        else:
            dropped += 1
    if not src_idx:
        return torch.empty((2, 0), dtype=torch.long), dropped
    return torch.tensor([src_idx, dst_idx], dtype=torch.long), dropped


# ── 3. Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print("Building HeteroData")
    print("=" * 70)

    # ── Idx maps ─────────────────────────────────────────────────────────────
    print("\n[1/5] Loading idx maps...")
    idx = {
        "user":      load_idx("user_to_idx.json"),
        "book":      load_idx("book_id_to_idx.json"),
        "author":    load_idx("author_to_idx.json"),
        "work":      load_idx("work_to_idx.json"),
        "review":    load_idx("review_to_idx.json"),
        "genre":     load_idx("genre_to_idx.json"),
        "shelf":     load_idx("shelf_to_idx.json"),
        "language":  load_idx("language_to_idx.json"),
        "format":    load_idx("format_to_idx.json"),
        "publisher": load_idx("publisher_to_idx.json"),
    }
    for k, v in idx.items():
        print(f"      {k:<10} {len(v):>9,}")

    # ── Node features ────────────────────────────────────────────────────────
    print("\n[2/5] Loading node features...")
    data = HeteroData()

    # U-category (no x)
    for node in ["user", "genre", "shelf", "language", "format", "publisher"]:
        data[node].num_nodes = len(idx[node])

    # TN/MN — text ∥ numeric
    for node, (h5_name, npy_name) in {
        "book":   ("books_bge_base_768.h5",   "book_numeric.npy"),
        "review": ("reviews_bge_base_768.h5", "review_numeric.npy"),
        "author": ("authors_bge_base_768.h5", "author_numeric.npy"),
        "work":   ("works_bge_base_768.h5",   "work_numeric.npy"),
    }.items():
        emb, _ = load_h5_embed(h5_name)
        num    = np.load(DATA_DIR / npy_name)
        x      = build_tn_features(emb, num)
        data[node].x = x
        print(f"      {node:<10} x={tuple(x.shape)}  dtype={x.dtype}")

    # ── Structural edges from Neo4j ──────────────────────────────────────────
    print("\n[3/5] Fetching structural edges from Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    STRUCTURAL_EDGES = [
        # (src_type, relation, dst_type, cypher)
        ("book",   "AUTHORED_BY",  "author",
         "MATCH (b:Book)-[:AUTHORED_BY]->(a:Author) RETURN b.book_id AS src, a.author_id AS dst"),
        ("book",   "EDITION_OF",   "work",
         "MATCH (b:Book)-[:EDITION_OF]->(w:Work) RETURN b.book_id AS src, w.work_id AS dst"),
        ("book",   "HAS_GENRE",    "genre",
         "MATCH (b:Book)-[:HAS_GENRE]->(g:Genre) RETURN b.book_id AS src, g.name AS dst"),
        ("book",   "HAS_SHELF",    "shelf",
         "MATCH (b:Book)-[:HAS_SHELF]->(s:Shelf) RETURN b.book_id AS src, s.name AS dst"),
        ("book",   "PUBLISHED_BY", "publisher",
         "MATCH (b:Book)-[:PUBLISHED_BY]->(p:Publisher) RETURN b.book_id AS src, p.name AS dst"),
        ("book",   "IN_LANGUAGE",  "language",
         "MATCH (b:Book)-[:IN_LANGUAGE]->(l:Language) RETURN b.book_id AS src, l.code AS dst"),
        ("book",   "IN_FORMAT",    "format",
         "MATCH (b:Book)-[:IN_FORMAT]->(f:Format) RETURN b.book_id AS src, f.name AS dst"),
        ("user",   "WROTE",        "review",
         "MATCH (u:User)-[:WROTE]->(r:Review) RETURN u.user_id AS src, r.review_id AS dst"),
        ("review", "REVIEWS",      "book",
         "MATCH (r:Review)-[:REVIEWS]->(b:Book) RETURN r.review_id AS src, b.book_id AS dst"),
        ("user",   "SHELVED",      "book",
         "MATCH (u:User)-[:SHELVED]->(b:Book) RETURN u.user_id AS src, b.book_id AS dst"),
    ]

    for src_type, rel, dst_type, cypher in STRUCTURAL_EDGES:
        edges = fetch_edges(driver, cypher)
        ei, dropped = to_edge_index(edges, idx[src_type], idx[dst_type])
        data[src_type, rel, dst_type].edge_index = ei
        warn = f"  ⚠ dropped={dropped:,}" if dropped else ""
        print(f"      ({src_type:<6}, {rel:<13}, {dst_type:<10}) edges={ei.shape[1]:>9,}{warn}")

    driver.close()

    # ── Training supervision edges from splits/train.npz ─────────────────────
    print("\n[4/5] Loading training supervision edges...")
    train = np.load(SPLITS_DIR / "train.npz")
    user_idx_arr = train["user_idx"]
    book_idx_arr = train["book_idx"]
    etype_arr    = train["edge_type"]

    for code, name in EDGE_TYPE_NAMES.items():
        mask = etype_arr == code
        ei = torch.from_numpy(
            np.stack([user_idx_arr[mask], book_idx_arr[mask]])
        ).long()
        data["user", name, "book"].edge_index = ei
        print(f"      (user, {name:<13}, book) edges={ei.shape[1]:>9,}")

    # ── Reverse edges ────────────────────────────────────────────────────────
    print("\n[5/5] Adding reverse edges (T.ToUndirected)...")
    data = T.ToUndirected(merge=False)(data)
    print(f"      total edge types now: {len(data.edge_types)}")

    # ── Validate ─────────────────────────────────────────────────────────────
    print("\nValidating...")
    data.validate(raise_on_error=True)
    print("      ✅ HeteroData.validate() passed")

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"\nSaving to {OUT_PATH}...")
    torch.save(data, OUT_PATH)

    meta = {
        "node_types": {nt: int(data[nt].num_nodes) for nt in data.node_types},
        "node_feature_dims": {nt: (int(data[nt].x.shape[1]) if "x" in data[nt] else 0)
                              for nt in data.node_types},
        "edge_counts": {f"{s}__{r}__{d}": int(data[s, r, d].edge_index.shape[1])
                        for (s, r, d) in data.edge_types},
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved meta to {META_PATH}")

    print(f"\n✅ Done. File size: {OUT_PATH.stat().st_size / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
