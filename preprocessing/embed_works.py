"""
Featurize Work nodes (MN category).

Sources:
  - Work IDs + data: Neo4j goodreads-poetry (canonical)
  - Fallback titles:  Neo4j Book nodes via best_book_id

Outputs (data/processed/):
  works_bge_base_768.h5     (25552, 768) float32, L2-normalized, sorted by work_id asc
  work_numeric.npy          (25552, 11)  float32, same row order
  work_numeric_columns.json

Numeric block (11 dims):
  original_year_normalized      z-score; 0.0 when missing
  log1p(books_count)
  log1p(ratings_count)
  log1p(text_reviews_count)
  rd_5_norm                     rd_k / rd_total; 0.0 when rd_total == 0
  rd_4_norm
  rd_3_norm
  rd_2_norm
  rd_1_norm
  is_year_missing               1 if original_publication_year is null
  is_avg_rating_meaningful      1 if rd_total > 0

Text:
  original_title if available, else best_book title, else '[no title]'

Run from repo root:
  python preprocessing/embed_works.py
"""

import json
import numpy as np
import h5py
from pathlib import Path
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 256
NEO4J_URI  = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "Niveau99")
NEO4J_DB   = "goodreads-poetry"
OUT_DIR    = Path("data/processed")

NUMERIC_COLUMNS = [
    "original_year_normalized",
    "log1p_books_count",
    "log1p_ratings_count",
    "log1p_text_reviews_count",
    "rd_5_norm",
    "rd_4_norm",
    "rd_3_norm",
    "rd_2_norm",
    "rd_1_norm",
    "is_year_missing",
    "is_avg_rating_meaningful",
]
# ─────────────────────────────────────────────────────────────────────────────


def fetch_works() -> list[dict]:
    """Fetch all Work nodes sorted by work_id, with best_book title fallback."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with driver.session(database=NEO4J_DB) as s:
        rows = list(s.run("""
            MATCH (w:Work)
            OPTIONAL MATCH (b:Book {book_id: w.best_book_id})
            RETURN
                w.work_id                      AS work_id,
                w.original_title               AS original_title,
                b.title                        AS book_title,
                w.original_publication_year    AS pub_year,
                w.books_count                  AS books_count,
                w.ratings_count                AS ratings_count,
                w.text_reviews_count           AS text_reviews_count,
                w.rd_1                         AS rd_1,
                w.rd_2                         AS rd_2,
                w.rd_3                         AS rd_3,
                w.rd_4                         AS rd_4,
                w.rd_5                         AS rd_5,
                w.rd_total                     AS rd_total
            ORDER BY w.work_id
        """))
    driver.close()
    return [dict(r) for r in rows]


def resolve_title(row: dict) -> str:
    t = (row.get("original_title") or "").strip()
    if t:
        return t
    t = (row.get("book_title") or "").strip()
    return t if t else "[no title]"


def build_numeric(row: dict, year_mean: float, year_std: float) -> list[float]:
    pub_year = row.get("pub_year")
    is_year_missing = 1.0 if pub_year is None else 0.0
    year_norm = float((pub_year - year_mean) / year_std) if pub_year is not None else 0.0

    books_count         = max(int(row.get("books_count") or 0), 0)
    ratings_count       = max(int(row.get("ratings_count") or 0), 0)
    text_reviews_count  = max(int(row.get("text_reviews_count") or 0), 0)

    rd_total = max(int(row.get("rd_total") or 0), 0)
    is_meaningful = 1.0 if rd_total > 0 else 0.0

    def rd_norm(key):
        v = row.get(key)
        if rd_total == 0 or v is None:
            return 0.0
        return float(v) / rd_total

    return [
        year_norm,
        float(np.log1p(books_count)),
        float(np.log1p(ratings_count)),
        float(np.log1p(text_reviews_count)),
        rd_norm("rd_5"),
        rd_norm("rd_4"),
        rd_norm("rd_3"),
        rd_norm("rd_2"),
        rd_norm("rd_1"),
        is_year_missing,
        is_meaningful,
    ]


def main() -> None:
    import torch
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Fetch from Neo4j ──────────────────────────────────────────────────
    print("Fetching Work nodes from Neo4j...")
    works = fetch_works()
    print(f"  {len(works)} works")

    work_ids = [str(w["work_id"]) for w in works]

    # ── 2. Year normalization stats (full corpus, Phase 1 shortcut) ──────────
    years = [w["pub_year"] for w in works if w.get("pub_year") is not None]
    year_mean = float(np.mean(years))
    year_std  = float(np.std(years))
    print(f"  Publication year: mean={year_mean:.1f}  std={year_std:.1f}  "
          f"missing={len(works)-len(years)} ({(len(works)-len(years))/len(works)*100:.1f}%)")

    # ── 3. Build text and numeric arrays ────────────────────────────────────
    print("Building arrays...")
    texts   = []
    numeric = []
    fallback_count = 0
    no_title_count = 0

    for w in works:
        title = resolve_title(w)
        if not (w.get("original_title") or "").strip():
            if title == "[no title]":
                no_title_count += 1
            else:
                fallback_count += 1
        texts.append(title)
        numeric.append(build_numeric(w, year_mean, year_std))

    numeric_arr = np.array(numeric, dtype=np.float32)
    print(f"  original_title used    : {len(works) - fallback_count - no_title_count}")
    print(f"  book title fallback    : {fallback_count}")
    print(f"  [no title] fallback    : {no_title_count}")
    print(f"  numeric shape          : {numeric_arr.shape}")

    # ── 4. Text embeddings ───────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {MODEL_NAME} on {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    print("Encoding work titles...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"  embeddings shape: {embeddings.shape}")

    # ── 5. Save HDF5 ─────────────────────────────────────────────────────────
    h5_path = OUT_DIR / "works_bge_base_768.h5"
    print(f"\nSaving {h5_path}...")
    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("work_id",   data=np.array(work_ids, dtype="S30"))
        hf.create_dataset("embedding", data=embeddings, dtype="float32",
                          compression="gzip", compression_opts=4)

    # ── 6. Save numeric ───────────────────────────────────────────────────────
    npy_path = OUT_DIR / "work_numeric.npy"
    print(f"Saving {npy_path}...")
    np.save(npy_path, numeric_arr)

    col_path = OUT_DIR / "work_numeric_columns.json"
    col_path.write_text(json.dumps(NUMERIC_COLUMNS, indent=2))

    stats_path = OUT_DIR / "work_norm_stats.json"
    stats_path.write_text(json.dumps({
        "year_mean": year_mean,
        "year_std":  year_std,
        "_note": "Phase 1: computed over full corpus. Tighten to training split in Phase 2.",
    }, indent=2))

    # ── 7. Verify ─────────────────────────────────────────────────────────────
    print("\n=== Verification ===")
    with h5py.File(h5_path, "r") as hf:
        ids_back = hf["work_id"][:]
        emb_back = hf["embedding"][:]
    npy_back = np.load(npy_path)

    print(f"  HDF5 ids    : {len(ids_back)}")
    print(f"  HDF5 shape  : {emb_back.shape}  dtype={emb_back.dtype}")
    norms = np.linalg.norm(emb_back[:5], axis=1)
    print(f"  L2 norms (5): {norms.round(4)}")
    print(f"  npy shape   : {npy_back.shape}  dtype={npy_back.dtype}")
    print(f"  numeric sample (first 3 rows):")
    for i in range(3):
        print(f"    {work_ids[i]:<12} {npy_back[i]}")

    idx_map = json.loads((OUT_DIR / "work_to_idx.json").read_text())
    first_id = ids_back[0].decode()
    assert idx_map[first_id] == 0, f"Row 0 mismatch: {first_id} -> {idx_map[first_id]}"
    print(f"  Row-order alignment with work_to_idx.json: OK")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
