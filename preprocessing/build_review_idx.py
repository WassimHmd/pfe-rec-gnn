"""
Build review_to_idx.json from the canonical HDF5 row order.
Row 0 in reviews_bge_base_768.h5 → index 0, etc.

Output: data/processed/review_to_idx.json  {review_id_str: int}
"""

import json
import h5py
from pathlib import Path

OUT_DIR = Path("data/processed")
h5_path = OUT_DIR / "reviews_bge_base_768.h5"

with h5py.File(h5_path, "r") as hf:
    review_ids = [x.decode() for x in hf["review_id"][:]]

review_to_idx = {rid: i for i, rid in enumerate(review_ids)}

out_path = OUT_DIR / "review_to_idx.json"
out_path.write_text(json.dumps(review_to_idx), encoding="utf-8")

print(f"review_to_idx.json: {len(review_to_idx):,} entries")
print(f"Sample: {list(review_to_idx.items())[:3]}")
