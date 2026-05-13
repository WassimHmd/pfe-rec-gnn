"""
Build review_numeric.npy for Review nodes (TN category).
Text embeddings already exist at data/processed/reviews_bge_base_768.h5.

Source: Neo4j goodreads-poetry Review nodes, row order matches the HDF5.

Outputs (data/processed/):
  review_numeric.npy          (154392, 4) float32
  review_numeric_columns.json

Numeric block (4 dims):
  rating              raw [0,5]; 0 when unrated
  log1p(n_votes)
  log1p(n_comments)
  is_rated            1 if rating > 0 else 0

Run from repo root:
  python preprocessing/build_review_numeric.py
"""

import json
import numpy as np
import h5py
from pathlib import Path
from neo4j import GraphDatabase

NEO4J_URI  = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "Niveau99")
NEO4J_DB   = "goodreads-poetry"
OUT_DIR    = Path("data/processed")

NUMERIC_COLUMNS = ["rating", "log1p_n_votes", "log1p_n_comments", "is_rated"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load review_id order from HDF5 (canonical row order) ─────────────
    h5_path = OUT_DIR / "reviews_bge_base_768.h5"
    print(f"Loading review_id order from {h5_path}...")
    with h5py.File(h5_path, "r") as hf:
        review_ids = [x.decode() for x in hf["review_id"][:]]
    print(f"  {len(review_ids)} reviews")

    # ── 2. Fetch numeric fields from Neo4j ───────────────────────────────────
    print("Fetching Review nodes from Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with driver.session(database=NEO4J_DB) as s:
        rows = list(s.run("""
            MATCH (r:Review)
            RETURN r.review_id  AS review_id,
                   r.rating     AS rating,
                   r.n_votes    AS n_votes,
                   r.n_comments AS n_comments
        """))
    driver.close()
    print(f"  {len(rows)} records fetched")

    # ── 3. Build lookup dict ─────────────────────────────────────────────────
    neo4j_map = {r["review_id"]: r for r in rows}
    missing = sum(1 for rid in review_ids if rid not in neo4j_map)
    if missing:
        print(f"  WARNING: {missing} review_ids in HDF5 not found in Neo4j")

    # ── 4. Build numeric array in HDF5 row order ─────────────────────────────
    print("Building numeric array...")
    numeric = []
    for rid in review_ids:
        r = neo4j_map.get(rid, {})
        rating     = int(r.get("rating") or 0)
        n_votes    = max(int(r.get("n_votes") or 0), 0)
        n_comments = max(int(r.get("n_comments") or 0), 0)
        is_rated   = 1.0 if rating > 0 else 0.0
        numeric.append([
            float(rating),
            float(np.log1p(n_votes)),
            float(np.log1p(n_comments)),
            is_rated,
        ])

    numeric_arr = np.array(numeric, dtype=np.float32)
    print(f"  shape: {numeric_arr.shape}")
    print(f"  is_rated=1: {int(numeric_arr[:, 3].sum()):,}  "
          f"is_rated=0: {int((numeric_arr[:, 3] == 0).sum()):,}")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    npy_path = OUT_DIR / "review_numeric.npy"
    np.save(npy_path, numeric_arr)
    print(f"Saved {npy_path}")

    col_path = OUT_DIR / "review_numeric_columns.json"
    col_path.write_text(json.dumps(NUMERIC_COLUMNS, indent=2))

    # ── 6. Verify ─────────────────────────────────────────────────────────────
    back = np.load(npy_path)
    print(f"\n=== Verification ===")
    print(f"  shape : {back.shape}  dtype={back.dtype}")
    print(f"  sample (first 3 rows):")
    for i in range(3):
        print(f"    {review_ids[i]}  {back[i]}")
    print(f"  rating range : [{back[:,0].min():.0f}, {back[:,0].max():.0f}]")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
