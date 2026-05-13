"""
Temporal train/val/test split of user→book supervision edges.

Positive labels : RATED_HIGH, READ_UNRATED  (label=1)
Explicit negatives: RATED_LOW               (label=0)
Structure only  : SHELVED                   (excluded — used in HeteroData graph only)

Split boundaries (T = max date_added across all interactions):
  train : date_added < T - 2yr
  val   : T - 2yr <= date_added < T - 1yr
  test  : date_added >= T - 1yr

Null date_added → assigned to train (conservative).

Outputs (data/processed/splits/):
  train.npz / val.npz / test.npz
    user_idx   int32
    book_idx   int32
    edge_type  int8   (0=RATED_HIGH, 1=READ_UNRATED, 2=RATED_LOW)
    label      int8   (1=positive, 0=negative)

  split_stats.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from neo4j import GraphDatabase

NEO4J_URI  = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "Niveau99")
NEO4J_DB   = "goodreads-poetry"
OUT_DIR    = Path("data/processed/splits")
DATA_DIR   = Path("data/processed")

DATE_FMT   = "ISO8601"

EDGE_TYPE_CODE = {"RATED_HIGH": 0, "READ_UNRATED": 1, "RATED_LOW": 2}
LABEL          = {"RATED_HIGH": 1, "READ_UNRATED": 1, "RATED_LOW": 0}


def fetch_edges(driver) -> list[dict]:
    with driver.session(database=NEO4J_DB) as s:
        rows = list(s.run("""
            MATCH (u:User)-[r:RATED_HIGH|RATED_LOW|READ_UNRATED]->(b:Book)
            RETURN u.user_id   AS user_id,
                   b.book_id   AS book_id,
                   type(r)     AS edge_type,
                   r.date_added AS date_added
        """))
    return [dict(r) for r in rows]


def parse_dates(df: pd.DataFrame) -> pd.Series:
    parsed = pd.to_datetime(df["date_added"], format=DATE_FMT, utc=True, errors="coerce") \
               if DATE_FMT != "ISO8601" else \
               pd.to_datetime(df["date_added"], utc=True, errors="coerce")
    null_n = parsed.isna().sum()
    if null_n:
        print(f"  WARNING: {null_n:,} null/unparseable date_added → assigned to train")
    return parsed


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load idx maps ──────────────────────────────────────────────────────
    print("Loading idx maps...")
    user_to_idx = json.loads((DATA_DIR / "user_to_idx.json").read_text(encoding="utf-8"))
    book_to_idx = json.loads((DATA_DIR / "book_id_to_idx.json").read_text(encoding="utf-8"))
    print(f"  users={len(user_to_idx):,}  books={len(book_to_idx):,}")

    # ── 2. Fetch edges from Neo4j ─────────────────────────────────────────────
    print("Fetching supervision edges from Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    rows = fetch_edges(driver)
    driver.close()
    print(f"  {len(rows):,} edges fetched")

    df = pd.DataFrame(rows)
    print(f"  edge type counts:\n{df['edge_type'].value_counts().to_string()}")

    # ── 3. Map IDs to indices ─────────────────────────────────────────────────
    print("Mapping IDs to indices...")
    df["user_id"] = df["user_id"].astype(str)
    df["book_id"] = df["book_id"].astype(str)

    unknown_users = df["user_id"].apply(lambda x: x not in user_to_idx).sum()
    unknown_books = df["book_id"].apply(lambda x: x not in book_to_idx).sum()
    if unknown_users: print(f"  WARNING: {unknown_users:,} unknown user_ids — dropping")
    if unknown_books: print(f"  WARNING: {unknown_books:,} unknown book_ids — dropping")

    df = df[df["user_id"].isin(user_to_idx) & df["book_id"].isin(book_to_idx)].copy()
    df["user_idx"] = df["user_id"].map(user_to_idx).astype(np.int32)
    df["book_idx"] = df["book_id"].map(book_to_idx).astype(np.int32)
    df["edge_type_code"] = df["edge_type"].map(EDGE_TYPE_CODE).astype(np.int8)
    df["label"] = df["edge_type"].map(LABEL).astype(np.int8)
    print(f"  {len(df):,} edges after ID resolution")

    # ── 4. Parse dates & determine T ─────────────────────────────────────────
    print("Parsing dates...")
    df["date_parsed"] = parse_dates(df)
    T = df["date_parsed"].max()
    T_val  = T - pd.DateOffset(years=2)
    T_test = T - pd.DateOffset(years=1)
    print(f"  T       = {T.date()}")
    print(f"  T_val   = {T_val.date()}  (train/val boundary)")
    print(f"  T_test  = {T_test.date()}  (val/test boundary)")

    # ── 5. Assign splits ──────────────────────────────────────────────────────
    null_mask = df["date_parsed"].isna()
    df["split"] = "train"
    df.loc[~null_mask & (df["date_parsed"] >= T_val)  & (df["date_parsed"] < T_test), "split"] = "val"
    df.loc[~null_mask & (df["date_parsed"] >= T_test), "split"] = "test"

    print("\nSplit breakdown:")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        pos = (sub["label"] == 1).sum()
        neg = (sub["label"] == 0).sum()
        print(f"  {split:<6}  total={len(sub):>8,}  pos={pos:>8,}  neg={neg:>8,}"
              f"  ratio_neg/pos={neg/pos:.3f}" if pos > 0 else "")

    # ── 6. Save splits ────────────────────────────────────────────────────────
    stats = {"T": str(T.date()), "T_val": str(T_val.date()), "T_test": str(T_test.date()),
             "splits": {}}

    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        path = OUT_DIR / f"{split}.npz"
        np.savez(
            path,
            user_idx=sub["user_idx"].values,
            book_idx=sub["book_idx"].values,
            edge_type=sub["edge_type_code"].values,
            label=sub["label"].values,
        )
        pos = int((sub["label"] == 1).sum())
        neg = int((sub["label"] == 0).sum())
        stats["splits"][split] = {"total": len(sub), "pos": pos, "neg": neg}
        print(f"  Saved {path}  ({len(sub):,} edges)")

    (OUT_DIR / "split_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    # ── 7. Verify ─────────────────────────────────────────────────────────────
    print("\n=== Verification ===")
    for split in ["train", "val", "test"]:
        d = np.load(OUT_DIR / f"{split}.npz")
        assert d["user_idx"].shape == d["book_idx"].shape == d["label"].shape
        n = d["user_idx"].shape[0]
        if n == 0:
            print(f"  {split}: 0 edges ⚠️")
        else:
            print(f"  {split}: {n:,} edges  "
                  f"label∈{set(d['label'].tolist())}  "
                  f"user_idx range=[{d['user_idx'].min()},{d['user_idx'].max()}]  "
                  f"book_idx range=[{d['book_idx'].min()},{d['book_idx'].max()}]")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
