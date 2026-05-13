from pymongo import MongoClient
import pandas as pd

client = MongoClient("mongodb://localhost:27017/")
authors = client["goodreads"]["authors"]

docs = list(authors.find({}, {"_id": 0}).limit(500))
df = pd.DataFrame(docs)

print("=== Fields ===")
for col in df.columns:
    series = df[col].replace("", None)
    non_null = series.notna().sum()
    pct = non_null / len(df) * 100
    sample = series.dropna().iloc[0] if non_null > 0 else None
    if isinstance(sample, str):
        sample = sample[:100]
    print(f"  {col:<35} {non_null:>4}/{len(df)} ({pct:5.1f}%)  e.g. {repr(sample)}")

print("\n=== Full sample record ===")
import json
print(json.dumps({k: v for k, v in docs[1].items() if k != "_id"}, indent=2, default=str))
