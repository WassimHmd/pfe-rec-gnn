"""Check full MongoDB authors schema vs what's in Neo4j."""
from pymongo import MongoClient
import pandas as pd

client = MongoClient("mongodb://localhost:27017/")
authors = client["goodreads"]["authors"]

# Sample broadly - use 2000 docs to get good field coverage
docs = list(authors.find({}, {"_id": 0}).limit(2000))
df = pd.DataFrame(docs)

print(f"Sample size: {len(df)} authors")
print(f"Total fields found: {len(df.columns)}")
print()

# Neo4j loaded: name, author_id, average_rating, ratings_count, text_reviews_count
neo4j_fields = {"name", "author_id", "average_rating", "ratings_count", "text_reviews_count"}

print("=== All fields (coverage + sample value) ===")
for col in sorted(df.columns):
    series = df[col].replace("", None)
    non_null = series.notna().sum()
    pct = non_null / len(df) * 100
    sample = series.dropna().iloc[0] if non_null > 0 else None
    if isinstance(sample, str):
        sample = sample[:120]
    tag = "  [in Neo4j]" if col in neo4j_fields else "  [MISSING from Neo4j]"
    print(f"{tag} {col:<30} {non_null:>5}/{len(df)} ({pct:5.1f}%)  e.g. {repr(sample)}")

# Show a full record for an author that has the most fields populated
print("\n=== Fullest sample record ===")
best = max(docs, key=lambda d: len([v for v in d.values() if v not in (None, "", [])]))
import json
print(json.dumps(best, indent=2, default=str))
