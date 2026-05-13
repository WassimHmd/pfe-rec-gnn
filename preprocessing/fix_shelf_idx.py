"""Rebuild shelf_to_idx.json from the Shelf nodes already in Neo4j."""
import json
from pathlib import Path
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "Niveau99"))

with driver.session(database="goodreads-poetry") as session:
    result = session.run("MATCH (s:Shelf) RETURN s.name AS name ORDER BY s.name")
    shelf_names = [r["name"] for r in result]

driver.close()

print(f"Found {len(shelf_names)} Shelf nodes in Neo4j")
print(shelf_names)

idx = {name: i for i, name in enumerate(shelf_names)}
out = Path("data/processed/shelf_to_idx.json")
out.write_text(json.dumps(idx, indent=2))
print(f"Saved {out}  ({len(idx)} entries)")
