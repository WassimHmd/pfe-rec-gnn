"""
Build *_to_idx.json index mappings for all remaining node types.

Outputs (data/processed/):
  language_to_idx.json     25 nodes  (norm + UNK/OTHER bucketing, min=50)
  format_to_idx.json       12 nodes  (norm + UNK/OTHER bucketing, min=50)
  publisher_to_idx.json    86 nodes  (norm + UNK/OTHER bucketing, min=50)
  genre_to_idx.json        10 nodes  (raw genre names from goodreads.genres)
  shelf_to_idx.json        60 nodes  (stoplist + top-60 by total count)
  author_to_idx.json    23105 nodes  (sorted author_id strings)
  work_to_idx.json      25552 nodes  (sorted work_id strings)
  user_to_idx.json     ~378K nodes  (union of reviews + interactions user_ids)

Run from repo root:
  python preprocessing/build_idx_maps.py
"""

import json
from pathlib import Path

import pandas as pd
from pymongo import MongoClient

# ── Config ───────────────────────────────────────────────────────────────────
MONGO_URI = "mongodb://localhost:27017/"
OUT_DIR   = Path("data/processed")

LANG_MIN_COUNT      = 50
FORMAT_MIN_COUNT    = 50
PUBLISHER_MIN_COUNT = 50
SHELF_TOP_N         = 60

SHELF_STOPLIST = {
    "to-read", "currently-reading", "read", "owned", "own", "books-i-own",
    "favorites", "my-books", "default", "kindle", "ebook", "ebooks",
    "audiobook", "audio", "library", "wishlist", "want-to-read",
    "did-not-finish", "dnf", "abandoned", "general", "shelfari-favorites",
    "all", "maybe", "re-read", "new", "recommended", "owned-books",
    "to-buy", "my-library", "i-own", "required-reading", "home-library",
    "my-ebooks", "to-re-read",
}
# ─────────────────────────────────────────────────────────────────────────────


def save_idx(mapping: dict, path: Path) -> None:
    path.write_text(json.dumps(mapping, indent=2))
    print(f"  saved {path}  ({len(mapping)} entries)")


def normalize_language(code) -> str:
    if pd.isna(code) or code in ("", "--"):
        return "UNK"
    code = code.lower().strip()
    if code in {"eng", "en", "en-us", "en-ca", "en-gb"}:
        return "eng"
    if code in {"ger", "deu"}:
        return "ger"
    if code in {"fre", "fra", "fr"}:
        return "fre"
    if code in {"nl", "dut", "nld"}:
        return "dut"
    if code in {"gre", "ell"}:
        return "gre"
    if code in {"per", "fas", "pes"}:
        return "per"
    if code in {"nor", "nob", "nno"}:
        return "nor"
    return code


def normalize_format(fmt) -> str:
    if pd.isna(fmt) or fmt == "":
        return "UNK"
    return fmt.strip().title()


def normalize_publisher(name) -> str:
    if pd.isna(name) or name == "":
        return "UNK"
    return name.strip()


def build_categorical_idx(series: pd.Series, min_count: int) -> dict:
    """Apply UNK/OTHER bucketing and return sorted vocab → index mapping."""
    counts = series.value_counts()
    keep   = set(counts[counts >= min_count].index)
    # UNK is already in the series; ensure OTHER is included if any exist
    bucketed = series.where(series.isin(keep), "OTHER")
    vocab    = sorted(bucketed.unique())
    return {v: i for i, v in enumerate(vocab)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    client      = MongoClient(MONGO_URI)
    poetry_db   = client["poetry"]
    goodreads_db = client["goodreads"]

    # ── Load the poetry books slice ──────────────────────────────────────────
    print("Loading poetry books from MongoDB...")
    books_df = pd.DataFrame(poetry_db.books.find({}, {"_id": 0}))
    print(f"  {len(books_df)} books")

    book_ids   = books_df["book_id"].astype(str).unique().tolist()
    work_ids   = books_df["work_id"].dropna().astype(str).unique().tolist()

    # ── 1. Language ──────────────────────────────────────────────────────────
    print("\n[1/8] Language...")
    books_df["language_norm"] = books_df["language_code"].apply(normalize_language)
    lang_idx = build_categorical_idx(books_df["language_norm"], LANG_MIN_COUNT)
    save_idx(lang_idx, OUT_DIR / "language_to_idx.json")

    # ── 2. Format ────────────────────────────────────────────────────────────
    print("\n[2/8] Format...")
    books_df["format_norm"] = books_df["format"].apply(normalize_format)
    fmt_idx = build_categorical_idx(books_df["format_norm"], FORMAT_MIN_COUNT)
    save_idx(fmt_idx, OUT_DIR / "format_to_idx.json")

    # ── 3. Publisher ─────────────────────────────────────────────────────────
    print("\n[3/8] Publisher...")
    books_df["publisher_norm"] = books_df["publisher"].apply(normalize_publisher)
    pub_idx = build_categorical_idx(books_df["publisher_norm"], PUBLISHER_MIN_COUNT)
    save_idx(pub_idx, OUT_DIR / "publisher_to_idx.json")

    # ── 4. Genre ─────────────────────────────────────────────────────────────
    print("\n[4/8] Genre...")
    genres_cursor = goodreads_db.genres.find(
        {"book_id": {"$in": book_ids}},
        {"book_id": 1, "genres": 1, "_id": 0},
    )
    genres_df  = pd.DataFrame(list(genres_cursor))
    genre_names = sorted({g for row in genres_df["genres"] for g in row})
    genre_idx   = {name: i for i, name in enumerate(genre_names)}
    save_idx(genre_idx, OUT_DIR / "genre_to_idx.json")

    # ── 5. Shelf ─────────────────────────────────────────────────────────────
    print("\n[5/8] Shelf...")
    shelf_counts: dict[str, int] = {}
    for shelves in books_df["popular_shelves"]:
        if not shelves:
            continue
        for s in shelves:
            name = s["name"]
            shelf_counts[name] = shelf_counts.get(name, 0) + int(s["count"])

    shelf_df = (
        pd.DataFrame(shelf_counts.items(), columns=["shelf", "total_count"])
        .sort_values("total_count", ascending=False)
        .reset_index(drop=True)
    )
    shelf_filtered = shelf_df[~shelf_df["shelf"].isin(SHELF_STOPLIST)].head(SHELF_TOP_N)
    shelf_vocab    = sorted(shelf_filtered["shelf"].tolist())
    shelf_idx      = {name: i for i, name in enumerate(shelf_vocab)}
    save_idx(shelf_idx, OUT_DIR / "shelf_to_idx.json")

    # ── 6. Author ────────────────────────────────────────────────────────────
    print("\n[6/8] Author...")
    author_ids = sorted({
        a["author_id"]
        for authors in books_df["authors"]
        for a in (authors or [])
    })
    print(f"  fetching {len(author_ids)} authors from goodreads.authors...")
    authors_df  = pd.DataFrame(
        list(goodreads_db.authors.find({"author_id": {"$in": author_ids}}, {"author_id": 1, "_id": 0}))
    )
    # Use sorted author_ids actually present in Mongo (should be all 23105)
    present_ids = sorted(authors_df["author_id"].astype(str).unique())
    author_idx  = {aid: i for i, aid in enumerate(present_ids)}
    save_idx(author_idx, OUT_DIR / "author_to_idx.json")
    missing = len(author_ids) - len(present_ids)
    if missing:
        print(f"  WARNING: {missing} author_ids not found in goodreads.authors")

    # ── 7. Work ──────────────────────────────────────────────────────────────
    print("\n[7/8] Work...")
    print(f"  fetching {len(work_ids)} works from goodreads.works...")
    works_df = pd.DataFrame(
        list(goodreads_db.works.find({"work_id": {"$in": work_ids}}, {"work_id": 1, "_id": 0}))
    )
    present_work_ids = sorted(works_df["work_id"].astype(str).unique())
    work_idx         = {wid: i for i, wid in enumerate(present_work_ids)}
    save_idx(work_idx, OUT_DIR / "work_to_idx.json")
    missing = len(work_ids) - len(present_work_ids)
    if missing:
        print(f"  WARNING: {missing} work_ids not found in goodreads.works")

    # ── 8. User ──────────────────────────────────────────────────────────────
    print("\n[8/8] User (union of reviews + interactions)...")

    BATCH = 5_000
    review_users: set[str] = set()
    for i in range(0, len(book_ids), BATCH):
        chunk = book_ids[i : i + BATCH]
        for doc in poetry_db.reviews.find({"book_id": {"$in": chunk}}, {"user_id": 1, "_id": 0}):
            review_users.add(doc["user_id"])
    print(f"  {len(review_users)} users from reviews")

    interaction_users: set[str] = set()
    for i in range(0, len(book_ids), BATCH):
        chunk = book_ids[i : i + BATCH]
        for doc in poetry_db.interactions.find({"book_id": {"$in": chunk}}, {"user_id": 1, "_id": 0}):
            interaction_users.add(doc["user_id"])
    print(f"  {len(interaction_users)} users from interactions")

    all_user_ids = sorted(review_users | interaction_users)
    user_idx     = {uid: i for i, uid in enumerate(all_user_ids)}
    save_idx(user_idx, OUT_DIR / "user_to_idx.json")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n✅ Done. Index maps written to data/processed/")
    print(f"  language:  {len(lang_idx):>7}")
    print(f"  format:    {len(fmt_idx):>7}")
    print(f"  publisher: {len(pub_idx):>7}")
    print(f"  genre:     {len(genre_idx):>7}")
    print(f"  shelf:     {len(shelf_idx):>7}")
    print(f"  author:    {len(author_idx):>7}")
    print(f"  work:      {len(work_idx):>7}")
    print(f"  user:      {len(user_idx):>7}")


if __name__ == "__main__":
    main()
