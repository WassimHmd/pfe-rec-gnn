"""Check temporal field coverage on Review nodes (Neo4j + MongoDB)."""
from neo4j import GraphDatabase
from pymongo import MongoClient
import pandas as pd

DB = "goodreads-poetry"
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "Niveau99"))

# ── Neo4j: coverage of each time field ───────────────────────────────────────
print("=== Neo4j Review temporal fields ===")
total = driver.session(database=DB).run("MATCH (r:Review) RETURN count(r) AS c").single()["c"]
print(f"Total Review nodes: {total:,}")

for field in ["date_added", "date_updated", "read_at"]:
    row = driver.session(database=DB).run(
        f"MATCH (r:Review) WHERE r.{field} IS NOT NULL RETURN count(r) AS c"
    ).single()
    pct = row["c"] / total * 100
    print(f"  {field:<15} {row['c']:>8,}  ({pct:.1f}%)")

# Sample a few values
print("\nSample date_added values:")
rows = driver.session(database=DB).run(
    "MATCH (r:Review) WHERE r.date_added IS NOT NULL RETURN r.date_added AS d LIMIT 5"
)
for r in rows:
    print(f"  {r['d']}")

print("\nSample read_at values (non-null only):")
rows = driver.session(database=DB).run(
    "MATCH (r:Review) WHERE r.read_at IS NOT NULL RETURN r.read_at AS d LIMIT 5"
)
for r in rows:
    print(f"  {r['d']}")

driver.close()

# ── MongoDB: check for started_at + full field coverage ──────────────────────
print("\n=== MongoDB poetry.reviews temporal fields ===")
client = MongoClient("mongodb://localhost:27017/")
sample = list(client["poetry"]["reviews"].find({}, {"_id": 0}).limit(2000))
df = pd.DataFrame(sample)

time_cols = ["date_added", "date_updated", "read_at", "started_at"]
for col in time_cols:
    if col not in df.columns:
        print(f"  {col:<15} NOT IN COLLECTION")
        continue
    non_empty = df[col].replace("", None).notna().sum()
    pct = non_empty / len(df) * 100
    sample_val = df[col].replace("", None).dropna().iloc[0] if non_empty > 0 else None
    print(f"  {col:<15} {non_empty:>6}/{len(df)}  ({pct:5.1f}%)  e.g. {repr(sample_val)}")
