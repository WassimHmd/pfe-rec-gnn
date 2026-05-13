# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HGT4Rec** — a heterogeneous knowledge-graph recommender (Master's thesis / PFE) built on the UCSD Goodreads poetry slice. The model is a Heterogeneous Graph Transformer (HGT, Hu et al. WWW 2020) trained for user→book link prediction. Full spec: `goodreads-hgt-project-spec-v2.md` (authoritative — follow it when this file is silent).

Pipeline stages (Phase 1 = current):
1. MongoDB → data cleaning & normalization
2. Neo4j load (exploration/visualization only, not in training loop)
3. Frozen text embeddings via BGE-base-en-v1.5 (768 dim) → HDF5
4. Numeric feature blocks → `.npy`
5. Integer index mappings → `*_to_idx.json`
6. PyG `HeteroData` construction → HGT training

## Running the Pipeline

**Generate book + review embeddings:**
```bash
python preprocessing/embed.py
```
Input: `data/raw/goodreads_{books,reviews}_poetry.json.gz`  
Output: `data/embeddings/{books,reviews}_bge_base_768.h5`  
If OOM on GPU, lower `BATCH_SIZE` from 256 → 128 in `embed.py`.

**Build the knowledge graph** (JupyterLab, run in order):
1. `preprocessing/KG.ipynb` — full graph: books, authors, works, users, reviews, interactions
2. `preprocessing/kg_mod.ipynb` — Language, Format, Publisher nodes/edges

**Sanity-check embeddings:** `preprocessing/test.ipynb` — reads the first 5 rows of any HDF5 and checks shape/dtype.

## Infrastructure

| Service | Connection | Note |
|---|---|---|
| MongoDB | `mongodb://localhost:27017/` | Cross-database `$lookup` not supported; all joins are client-side via pandas |
| Neo4j | `bolt://localhost:7687`, auth `neo4j/Niveau99`, db `goodreads-poetry` | Exploration only — NOT in the training loop |

MongoDB layout: `goodreads.{books, authors, works}` and `poetry.{reviews, interactions}`.

## Node Featurization Categories

Three patterns — **no per-type InputMLP**. Direct concatenation; dimension unification deferred to HGT layer 1.

| Category | Formula | Node types |
|---|---|---|
| **U** (ID-only) | `id_embed[idx]` | User, Genre, Shelf, Language, Format, Publisher |
| **MN** (ID + text + numeric) | `id_embed[idx] ∥ text_embed[idx] ∥ numeric[idx]` | Work, Author |
| **TN** (text + numeric) | `text_embed[idx] ∥ numeric[idx]` | Book, Review |

Book is TN (not MN) — edition metadata already defines it; an ID embedding over editions adds parameters without signal.

Book feature matrix at training time: **(36514, 777)** = 768 (text) ∥ 9 (numeric).

### Book numeric block (9 dims, frozen for Phase 1)
```
log1p(num_pages), year_normalized, average_rating, log1p(ratings_count),
log1p(text_reviews_count), is_ebook, is_num_pages_missing, is_year_missing,
is_avg_rating_meaningful
```

ID embeddings (`nn.Embedding`) are model parameters, NOT preprocessing artifacts. Only the `*_to_idx.json` index maps are precomputed.

## Knowledge Graph Schema

**Edge types** (all exist in both directions; HGT treats reverse relations as distinct):

| Edge | Status | Weight/attrs |
|---|---|---|
| `User → Book`: `RATED_HIGH`, `RATED_LOW`, `READ_UNRATED`, `SHELVED` | created | rating, date_added |
| `User → Review`: `WROTE` | created | – |
| `Review → Book`: `REVIEWS` | created | – |
| `Book → Author`: `AUTHORED_BY` | created | role, position (0-indexed) |
| `Book → Work`: `EDITION_OF` | created | – |
| `Book → Genre`: `HAS_GENRE` | created | weight (normalized fraction) |
| `Book → Shelf`: `HAS_SHELF` | created | weight (normalized fraction) |
| `Book → Publisher/Language/Format`: `PUBLISHED_BY`, `IN_LANGUAGE`, `IN_FORMAT` | created | – |
| `Book → Book`: `SIMILAR_TO` | excluded from eval by default | – |
| `Book → Series`: `IN_SERIES` | deferred | – |

`RATED_HIGH` = rating ≥ 4; `RATED_LOW` = rating 1–3; `READ_UNRATED` = read, no rating; `SHELVED` = not read.

## Model Architecture (HGT4Rec)

PyG `HGTConv` with L stacked layers. Per-edge-type attention:
```
Attention(s,e,t) = softmax[(K(s)·W^ATT_φ(e)·Q(t)ᵀ) · μ⟨τ(s),φ(e),τ(t)⟩ / √d]
Message(s,e,t)   = M(s)·W^MSG_φ(e)
H(l)[t]          = A-Linear_τ(t)(σ(⊕ Attention·Message)) + H(l-1)[t]  # residual
```

**Link prediction head** (placeholder — revise after experiments):
```python
score = MLP(h_user ∥ h_book)   # 2-3 layer, sigmoid
loss  = BCE(score, label)
```
No elementwise product, no absolute difference.

**Positive label**: `is_read=True` OR `rating ≥ 4`. **Negatives**: uniform random user–book pairs absent from all edge types, ratio 1:5.

**Evaluation**: NDCG@10, MRR, AUC. Dedup at work-level (highest-scoring book per work).

**Split**: time-based on `date_added` — train < T-2, val [T-2, T-1), test [T-1, T].

### Hyperparameter starting points
`d_model=256, num_heads=8, num_layers=3, dropout=0.2, lr=1e-3, weight_decay=1e-4, batch_size=1024, fanout=[15,10,5], neg_ratio=5, epochs=50 (early stop), id_dim=128 or 256`

## Processed Artifacts Layout

```
data/processed/
├── books_bge_base_768.h5          # (36514, 768) float32, sorted by ascending book_id
├── book_numeric.npy               # (36514, 9) float32, same row order as h5
├── book_id_to_idx.json            # book_id → row index (canonical)
├── book_norm_stats.json           # imputation/z-score stats (full-corpus, Phase 1 shortcut)
├── book_numeric_columns.json      # column order documentation
│
├── authors_bge_base_768.h5        # (planned)
├── author_numeric.npy             # (planned)
├── author_to_idx.json             # (planned)
│
├── works_bge_base_768.h5          # (planned)
├── work_numeric.npy               # (planned)
├── work_to_idx.json               # (planned)
│
├── reviews_bge_base_768.h5        # (planned)
├── review_numeric.npy             # (planned)
│
├── user_to_idx.json               # (planned) — from union of reviews + interactions user_ids
├── genre_to_idx.json              # (to confirm)
├── shelf_to_idx.json              # (to confirm)
├── language_to_idx.json           # saved
├── format_to_idx.json             # saved
└── publisher_to_idx.json          # saved
```

Raw data (`data/raw/`) and large artifacts (`*.h5`, `*.npy`) are gitignored. Small JSON mappings are tracked.

## Phase 1 Status

**Completed:** Mongo ingestion, all cleaning/normalization, Neo4j full load, book text embeddings, book numeric block, categorical `*_to_idx.json` for Language/Format/Publisher.

**Next steps (in order):**
1. Save remaining `*_to_idx.json`: User, Author, Work, Genre, Shelf
2. Compute text embeddings + numeric blocks for Author, Work, Review
3. Build PyG `HeteroData` from all artifacts
4. Implement HGT model, link prediction head, training loop, evaluation

## Key Implementation Notes

- Text and numeric are stored **separately on disk**, concatenated in CPU at `HeteroData` construction time. Rationale: enables independent updates and Phase 3 encoder fine-tuning.
- Review text is NOT stored on Neo4j `Review` nodes — metadata only in Neo4j, text in HDF5.
- All embeddings are L2-normalized at generation time → cosine similarity = dot product downstream.
- Neo4j bulk writes use 10K-row chunked `MERGE` batches.
- Timestamps use Goodreads format `"%a %b %d %H:%M:%S %z %Y"` — parse with `utc=True` to normalize timezones.
- `requirements.txt` is UTF-16 LE encoded (Windows); parse with `encoding='utf-16'`.
- `book_norm_stats.json` stats are computed over the full corpus (Phase 1 shortcut, documented in `_note` field) — tighten to training split in Phase 2.
- The five U-category categorical node types (Genre, Shelf, Language, Format, Publisher) are a deliberate design decision — do not drop them without new evidence (spec §13).

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- ALWAYS read graphify-out/GRAPH_REPORT.md before reading any source files, running grep/glob searches, or answering codebase questions. The graph is your primary map of the codebase.
- IF graphify-out/wiki/index.md EXISTS, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
