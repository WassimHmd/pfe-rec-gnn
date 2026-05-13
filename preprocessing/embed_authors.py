"""
Featurize Author nodes (MN category).

Sources:
  - Author IDs:  Neo4j goodreads-poetry (canonical — only poetry-slice authors)
  - Author data: MongoDB goodreads.authors

Outputs (data/processed/):
  authors_bge_base_768.h5   (23105, 768) float32, L2-normalized, sorted by author_id asc
  author_numeric.npy        (23105, 4)   float32, same row order
  author_numeric_columns.json

Numeric block (4 dims):
  average_rating            raw [0,5]; imputed 0.0 when ratings_count == 0
  log1p(ratings_count)
  log1p(text_reviews_count)
  is_avg_rating_meaningful  1 if ratings_count > 0 else 0

Run from repo root:
  python preprocessing/embed_authors.py
"""

import json
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from neo4j import GraphDatabase
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 256
NEO4J_URI  = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "Niveau99")
NEO4J_DB   = "goodreads-poetry"
MONGO_URI  = "mongodb://localhost:27017/"
OUT_DIR    = Path("data/processed")

NUMERIC_COLUMNS = [
    "average_rating",
    "log1p_ratings_count",
    "log1p_text_reviews_count",
    "is_avg_rating_meaningful",
]
# ─────────────────────────────────────────────────────────────────────────────


def get_neo4j_author_ids() -> list[str]:
    """Return author_ids present in Neo4j, sorted ascending."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with driver.session(database=NEO4J_DB) as session:
        rows = session.run(
            "MATCH (a:Author) RETURN a.author_id AS id ORDER BY a.author_id"
        )
        ids = [r["id"] for r in rows]
    driver.close()
    return ids


def fetch_authors_from_mongo(author_ids: list[str]) -> dict:
    """Return dict author_id -> record for all requested ids."""
    client = MongoClient(MONGO_URI)
    cursor = client["goodreads"]["authors"].find(
        {"author_id": {"$in": author_ids}},
        {"_id": 0},
    )
    return {doc["author_id"]: doc for doc in cursor}


def build_numeric(record: dict) -> list[float]:
    ratings_count       = max(int(record.get("ratings_count") or 0), 0)
    text_reviews_count  = max(int(record.get("text_reviews_count") or 0), 0)
    is_meaningful       = 1.0 if ratings_count > 0 else 0.0
    average_rating      = float(record.get("average_rating") or 0.0) if is_meaningful else 0.0

    return [
        average_rating,
        float(np.log1p(ratings_count)),
        float(np.log1p(text_reviews_count)),
        is_meaningful,
    ]


def main() -> None:
    import torch
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Author IDs from Neo4j ─────────────────────────────────────────────
    print("Fetching author IDs from Neo4j...")
    author_ids = get_neo4j_author_ids()
    print(f"  {len(author_ids)} authors")

    # ── 2. Author records from MongoDB ──────────────────────────────────────
    print("Fetching author records from MongoDB...")
    author_map = fetch_authors_from_mongo(author_ids)
    missing = [aid for aid in author_ids if aid not in author_map]
    if missing:
        print(f"  WARNING: {len(missing)} author_ids not found in MongoDB — will use fallback")
    print(f"  {len(author_map)} records fetched")

    # ── 3. Build ordered arrays (row order = sorted author_id) ───────────────
    print("Building text and numeric arrays...")
    texts   = []
    numeric = []

    for aid in author_ids:
        rec  = author_map.get(aid, {})
        name = (rec.get("name") or "").strip() or "[unknown author]"
        texts.append(name)
        numeric.append(build_numeric(rec))

    numeric_arr = np.array(numeric, dtype=np.float32)
    print(f"  numeric shape: {numeric_arr.shape}")

    # ── 4. Text embeddings ───────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {MODEL_NAME} on {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    print("Encoding author names...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"  embeddings shape: {embeddings.shape}")

    # ── 5. Save HDF5 ─────────────────────────────────────────────────────────
    h5_path = OUT_DIR / "authors_bge_base_768.h5"
    print(f"\nSaving {h5_path}...")
    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("author_id", data=np.array(author_ids, dtype="S30"))
        hf.create_dataset("embedding", data=embeddings, dtype="float32",
                          compression="gzip", compression_opts=4)

    # ── 6. Save numeric ───────────────────────────────────────────────────────
    npy_path = OUT_DIR / "author_numeric.npy"
    print(f"Saving {npy_path}...")
    np.save(npy_path, numeric_arr)

    col_path = OUT_DIR / "author_numeric_columns.json"
    col_path.write_text(json.dumps(NUMERIC_COLUMNS, indent=2))

    # ── 7. Verify ─────────────────────────────────────────────────────────────
    print("\n=== Verification ===")
    with h5py.File(h5_path, "r") as hf:
        ids_back = hf["author_id"][:]
        emb_back = hf["embedding"][:]
    npy_back = np.load(npy_path)

    print(f"  HDF5 ids    : {len(ids_back)}")
    print(f"  HDF5 shape  : {emb_back.shape}  dtype={emb_back.dtype}")
    norms = np.linalg.norm(emb_back[:5], axis=1)
    print(f"  L2 norms (5): {norms.round(4)}")
    print(f"  npy shape   : {npy_back.shape}  dtype={npy_back.dtype}")
    print(f"  numeric sample (first 3 rows):")
    for i in range(3):
        print(f"    {author_ids[i]}  {npy_back[i]}")

    # Row-order alignment check: first id in h5 matches author_to_idx
    idx_map = json.loads((OUT_DIR / "author_to_idx.json").read_text())
    first_id = ids_back[0].decode()
    assert idx_map[first_id] == 0, f"Row 0 mismatch: {first_id} -> {idx_map[first_id]}"
    print(f"  Row-order alignment with author_to_idx.json: OK")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
