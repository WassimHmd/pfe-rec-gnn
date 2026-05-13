# Goodreads HGT Recommender — Project Reference (v2)

A heterogeneous knowledge-graph recommender built on the UCSD Goodreads dump, trained with a Heterogeneous Graph Transformer (HGT, Hu et al., WWW 2020) for link prediction between users and books.

This is v2 of the project spec. It supersedes the original `goodreads-hgt-project-spec.md` and integrates all design decisions made through the data-loading and featurization phases of Phase 1.

---

## 0. Changelog vs v1

Material changes since the original spec, with brief rationale:

- **Text embeddings: BGE-base at 768 dim**, not BGE-small at 384. File convention: `books_bge_base_768.h5` (parallel files for other node types).
- **No per-type InputMLP** at the featurization layer. Initial features are direct concatenation (`||`) of components; per-type dimension unification is handled by HGT layer 1's K/Q/M-Linear projections.
- **Three node featurization categories formalized**: U (ID-only), MN (ID + text + numeric), TN (text + numeric, no ID embedding).
- **Book moved from MN to TN** — gets text + numeric but no ID embedding. Rationale: editions are essentially defined by their metadata; an ID embedding over editions adds parameters without obvious signal.
- **Link prediction head simplified to pure concat-MLP** on `[h_u || h_v]`. No elementwise product, no absolute difference. Placeholder — revise after experiments.
- **Format and Language categorical nodes kept** (reversed earlier "drop" decision after seeing actual distributions). Five U-category categorical node types in total: Genre, Shelf, Language, Format, Publisher.
- **All five categoricals use UNK for missing + OTHER for low-count tail.** Threshold ~50 books per node, tunable per type.
- **Normalization for Phase 1 numeric stats computed over the full corpus**, not the training split. Documented Phase-1 shortcut; tighten in Phase 2.
- **Mongo split documented**: books/authors/works live in `goodreads`; reviews/interactions live in `poetry` (Phase 1 slice). Cross-database `$lookup` not supported on the deployment — joins done client-side.
- **Neo4j database**: `goodreads-poetry` (Enterprise/Aura, not Community default).
- **Approach working name**: HGT4Rec (placeholder).

---

## 1. Project Overview

### Goal
Train a recommender that predicts which books a user will meaningfully engage with, by modeling the Goodreads ecosystem as a typed multi-relational graph and learning node representations via HGT.

### Approach (Phase 1 pipeline)
1. Parse five JSON collections (books, authors, works, genres, reviews, interactions) into MongoDB and/or Parquet.
2. Filter and normalize: shelf stoplist + count threshold, language code normalization, format/publisher normalization with UNK + OTHER bucketing.
3. Load the cleaned graph into Neo4j (`goodreads-poetry` database) for exploration and visualization only.
4. Compute frozen text embeddings (BGE-base, 768 dim) for each text-bearing node type. Store as HDF5.
5. Compute numeric blocks for each text- or numeric-bearing node type. Store as `.npy`.
6. Build integer index mappings (`*_to_idx.json`) for all node types that need ID embeddings.
7. At training time: build PyG `HeteroData` from Parquet/HDF5/npy artifacts. Train HGT with neighbor sampling.
8. Supervise on user→book link prediction.

### Theoretical basis
HGT parameterizes message passing by the meta-relation triplet `⟨τ(s), φ(e), τ(t)⟩`. Weight matrices decompose into source-node projection, edge projection, and target-node projection. Parameter sharing across relations sharing node types; edge-type-specific semantics preserved. Multiple user→book edge types (`rated_high`, `rated_low`, `read_unrated`, `shelved`) share the user-to-book structural weights but differentiate through `W^ATT_φ(e)` and `W^MSG_φ(e)`.

### Scaling strategy
- **Phase 1 (current)**: Poetry genre slice — ~36K books, ~25K works, ~23K authors, ~378K users, ~154K reviews-with-text, ~2.7M interactions. Neo4j for exploration; PyG `HeteroData` for training.
- **Phase 2**: Mid-sized genre with RTE and ablations.
- **Phase 3**: Full dump.

### Hardware budget
Single GPU, 12 GB VRAM.

---

## 2. Dataset and Storage Layout

### 2.1 Source

UCSD Book Graph / Goodreads dump. Per-collection schemas unchanged from v1; see v1 §2 for raw record formats.

### 2.2 MongoDB layout (Phase 1)

Cross-database `$lookup` is not supported on the user's deployment. All joins are client-side via pandas.

| Database | Collection | Contents |
|---|---|---|
| `goodreads` | `books` | Books in the poetry slice |
| `goodreads` | `authors` | All Goodreads authors |
| `goodreads` | `works` | All Goodreads works |
| `poetry` | `reviews` | Reviews for poetry-slice books |
| `poetry` | `interactions` | Interactions for poetry-slice books |

### 2.3 Neo4j (Phase 1 exploration only)

- Database: `goodreads-poetry`.
- Driver URI: `bolt://localhost:7687` (single instance, not a cluster).
- Not in the training loop. Used solely for visualization, ad-hoc Cypher queries, and schema inspection.
- Indexes created on every node type's primary ID before bulk loads.

### 2.4 Filesystem layout for processed artifacts

```
data/processed/
├── books_bge_base_768.h5           # (n_books, 768) text + (n_books,) book_id, sorted
├── book_numeric.npy                # (n_books, 9), same row order as h5
├── book_id_to_idx.json             # canonical book_id → row index
├── book_norm_stats.json            # imputation/z-score stats
├── book_numeric_columns.json       # column-order documentation
│
├── authors_bge_base_768.h5         # (planned) Author text embeddings (name)
├── author_numeric.npy              # (planned) Author numeric block
├── author_to_idx.json              # (planned) author_id → row index
│
├── works_bge_base_768.h5           # (planned) Work text embeddings (original_title)
├── work_numeric.npy                # (planned) Work numeric block
├── work_to_idx.json                # (planned) work_id → row index
│
├── reviews_bge_base_768.h5         # (planned) Review text embeddings
├── review_numeric.npy              # (planned) Review numeric block
│
├── user_to_idx.json                # (planned) user_id → row index, for ID embedding
│
├── language_to_idx.json            # categorical → row index
├── format_to_idx.json
├── publisher_to_idx.json
├── genre_to_idx.json               # (to confirm exists)
├── shelf_to_idx.json               # (to confirm exists)
```

---

## 3. Graph Schema

### 3.1 Node types

| Node | Category | Featurization components | Phase 1 cardinality |
|---|---|---|---|
| User | U | ID embedding only | ~378K |
| Book | TN | text(title + description) + numeric | 36,514 |
| Work | MN | ID + text(original_title) + numeric (incl. rating_dist 5-bin) | 25,552 |
| Author | MN | ID + text(name) + numeric | 23,105 |
| Review | TN | text(review_text) + numeric | 154,392 |
| Genre | U | ID embedding only | 10 |
| Shelf | U | ID embedding only (post-filter) | 60 |
| Language | U | ID embedding only | ~15-25 (after normalization + threshold) |
| Format | U | ID embedding only | ~15-25 (after normalization + threshold) |
| Publisher | U | ID embedding only | varies with threshold |

### 3.2 Node featurization categories

Three patterns, no per-type InputMLP. Initial features are direct concatenation; dimension unification deferred to HGT layer 1.

**U (ID-only):**
```
h_init = id_embed[node_idx]
```

**MN (ID + text + numeric):**
```
h_init = id_embed[node_idx] || text_embed[node_idx] || numeric[node_idx]
```

**TN (text + numeric, no ID embedding):**
```
h_init = text_embed[node_idx] || numeric[node_idx]
```

Concatenation symbol is `||` consistently in code and thesis.

### 3.3 Edge types

Edges from v1 §3.2 retained. All edges exist in both directions; HGT treats reverse relations as distinct edge types.

| Source → Target | Edge type | Phase 1 status | Attributes |
|---|---|---|---|
| User → Book | `RATED_HIGH` | created | rating, date_added |
| User → Book | `RATED_LOW` | created | rating, date_added |
| User → Book | `READ_UNRATED` | created | date_added |
| User → Book | `SHELVED` | created | date_added |
| User → Review | `WROTE` | created | – |
| Review → Book | `REVIEWS` | created | – |
| Book → Author | `AUTHORED_BY` | created | role, position |
| Book → Work | `EDITION_OF` | created | – |
| Book → Genre | `HAS_GENRE` | created | weight |
| Book → Shelf | `HAS_SHELF` | created | weight |
| Book → Publisher | `PUBLISHED_BY` | created | – |
| Book → Language | `IN_LANGUAGE` | created | – |
| Book → Format | `IN_FORMAT` | created | – |
| Book → Series | `IN_SERIES` | (deferred / TBD) | – |
| Book → Book | `SIMILAR_TO` | excluded if used in eval; default off | – |

Edge weights on `HAS_GENRE` and `HAS_SHELF` are per-book normalized proportions (count / sum of counts within that book), not raw counts.

### 3.4 Design decisions retained from v1

Reviews as nodes (not edge attrs); Book and Work both kept; Genre AND Shelf both as node types; rating as edge-type split. See v1 §3.4.

### 3.5 New design decisions

- **`is_read` semantics**: empirically verified for the poetry slice that `is_read=false` with `rating>0` does not occur. The four-way edge split is therefore exhaustive and unambiguous. The exact provenance of `is_read=true` (shelf-state vs derived flag) is not documented by UCSD; for modeling purposes it means "read or claimed-read" vs "shelved with intent only."
- **`AUTHORED_BY.position`**: zero-indexed position in the book's author list, stored as an edge property alongside `role`. Reserved for Phase 2 author-position subsplit ablation.

---

## 4. Data Cleaning and Normalization

### 4.1 Shelves

Two-stage filter:

1. **Stoplist** (functional/status shelves): `to-read`, `currently-reading`, `read`, `owned`, `kindle`, `favorites`, etc. Extended during inspection.
2. **Global count threshold**: drop shelves with fewer than ~10K total uses across the corpus. Phase 1 retains 60 shelves after both filters.

Normalization at the per-(book, shelf) level: keep all shelves above filter, compute weights as `count / sum(counts within book)`. Stored as `:HAS_SHELF.weight` edge property.

### 4.2 Genres

UCSD-derived. ~10 top-level labels. Commas inside a single key string are part of the label, not a delimiter. Per-book normalization same pattern as shelves: weight = `count / sum(counts within book)`.

### 4.3 Language

Multiple ISO 639 variants in the raw data. Normalization function collapses code variants only (`eng`/`en`/`en-US`/`en-CA`/`en-GB` → `eng`); historical variants kept distinct (`enm` Middle English, `ang` Old English, `gmh` Middle High German, `fro` Old French, `frm` Middle French, `dum` Middle Dutch, `grc` Ancient Greek, `peo` Old Persian).

NaN/empty/`--` → `UNK`. Languages with fewer than 50 books bucketed into `OTHER`.

### 4.4 Format

NaN/empty → `UNK`. Whitespace stripped. Title-case applied. Formats with fewer than 50 books bucketed into `OTHER`.

### 4.5 Publisher

NaN/empty → `UNK`. Whitespace stripped. Publishers with fewer than 50 books bucketed into `OTHER`. (Threshold may be lowered to 20 in Phase 2 to better capture small-press structure characteristic of poetry.)

Phase 1 does not canonicalize imprints (Penguin Books vs Penguin Classics vs Penguin Books Ltd remain separate). Acceptable because the threshold bucket absorbs spelling variants of small publishers; large publishers dominate their own buckets even when split.

### 4.6 Date parsing

Goodreads timestamps are in the format `"%a %b %d %H:%M:%S %z %Y"` (e.g., `"Tue Jun 12 08:59:04 -0700 2012"`). Parsing uses `pd.to_datetime(..., format=..., errors="coerce", utc=True)` to normalize all timezones to UTC. ISO 8601 strings stored as Neo4j edge properties.

---

## 5. Featurization

### 5.1 Text embeddings

- Model: BGE-base (768 dim), frozen.
- Precomputed once. Stored in HDF5 files keyed by node ID, with two top-level datasets: `embedding` (n × 768 float32) and `book_id` / `author_id` / etc. (n strings).
- Both arrays in matching row order. Canonical ordering is ascending string sort of the node ID. Verified via row-by-row alignment check before training.
- File naming convention: `<type>s_bge_base_768.h5`.

### 5.2 Book numeric block (frozen for Phase 1)

9 dimensions:

```
[
    log1p(num_pages),                 # imputed with train median when missing
    year_normalized,                  # (year - mean) / std on train set
    average_rating,                   # in [0, 5], imputed 0 when missing
    log1p(ratings_count),
    log1p(text_reviews_count),
    is_ebook,                         # 0 or 1
    is_num_pages_missing,             # flag (raised BEFORE imputation)
    is_year_missing,                  # flag
    is_avg_rating_meaningful,         # 1 iff ratings_count > 0
]
```

Stored as `book_numeric.npy`, shape `(36514, 9)`, float32. Row order matches `books_bge_base_768.h5`'s book_id ordering. `book_id_to_idx.json` bridges the two.

### 5.3 Author numeric block (planned)

Per v1 §4.3:
```
[average_rating, log1p(ratings_count), log1p(text_reviews_count)]
```
Plus missingness flags as needed.

### 5.4 Work numeric block (planned)

Per v1 §4.3:
```
[original_year_normalized, log1p(books_count), log1p(ratings_count),
 log1p(text_reviews_count), rd_5_norm, rd_4_norm, rd_3_norm, rd_2_norm, rd_1_norm]
```
The 5-bin distribution is parsed from `rating_dist` (`"5:1|4:1|3:1|2:0|1:0|total:3"` → dict, then normalized by `total`).

### 5.5 Review numeric block (planned)

Per v1 §4.3:
```
[rating_normalized, log1p(n_votes), log1p(n_comments)]
```

### 5.6 Concatenation at training time

Text + numeric are kept **separate on disk**, concatenated in CPU memory once at `HeteroData` construction. Rationale: text embeddings are frozen and computed by a separate model; numeric block can be tweaked independently; Phase 3 fine-tuning of the text encoder will require text to become a function of encoder weights rather than a static cache.

Book feature matrix at training time: `(36514, 777)` = `(36514, 768)` text || `(36514, 9)` numeric. ~110 MB CPU RAM. Same pattern for Author/Work/Review with their respective text + numeric dims plus optional ID embedding (MN nodes).

### 5.7 ID embeddings

Created as `nn.Embedding` inside the model — NOT a preprocessing artifact. Only the integer-index mapping (`*_to_idx.json`) is precomputed.

Preprocessing checklist for ID embeddings:
- `user_to_idx.json` — from union of all `user_id`s in reviews + interactions
- `author_to_idx.json` — from authors_df
- `work_to_idx.json` — from works_df
- `genre_to_idx.json`, `shelf_to_idx.json`, `language_to_idx.json`, `format_to_idx.json`, `publisher_to_idx.json` — from filtered/normalized vocabularies

At model construction time:
```python
self.user_id_embedding = nn.Embedding(num_users, id_dim)
# etc.
```

`id_dim` is a hyperparameter, same value across types for Phase 1 simplicity.

### 5.8 Phase 1 normalization stats shortcut

Train/val/test split is on interactions (time-based, per §6 below). Book-level stats (medians, means, stds for numeric block) computed over the **full corpus** in Phase 1, with a documented shortcut. Leakage is minor because Book features are metadata, not labels. Tighten in Phase 2 by restricting stats computation to books that appear in training-window interactions.

---

## 6. Supervision and Link Prediction

### 6.1 Target

Binary link prediction between User and Book.

**Positive**: there exists a user→book interaction with `is_read = true` OR `rating ≥ 4`.

### 6.2 Negative sampling

Random user–book pairs not present in **any** user→book edge type for that user. Default: uniform sampling, 1:5 positive:negative ratio per batch.

### 6.3 Prediction head

Pure concat-MLP placeholder:
```
score = MLP( h_user || h_book )      # 2-3 layer MLP, sigmoid at end
loss  = BCE(score, label)
```

**Explicit decisions:**
- No elementwise product term.
- No absolute difference term.
- Revise after experiments — head is a known limitation.

### 6.4 Evaluation

NDCG@10, MRR, AUC. Dedup at work-level by collapsing predicted book_ids to work_ids and keeping the highest-scoring book per work.

### 6.5 Split

Time-based on interaction `date_added`. Train < year T-2; Val in [T-2, T-1); Test in [T-1, T]. Exact cutoff years depend on the dataset slice.

### 6.6 Temporal leakage constraint

When predicting a user→book edge with timestamp `T_edge`, the sampled subgraph used to compute user and book embeddings must not contain any edge with timestamp > `T_edge`. Custom sampler in Phase 2 enforces this; Phase 1 (non-temporal) does not.

---

## 7. Architecture (HGT4Rec)

Working name. Four modules:

1. **Heterogeneous Knowledge Graph Module** — schema design including edge subsplits.
2. **Node Featurization Module** — U/MN/TN categories.
3. **HGT Encoder Module** — L stacked HGT layers.
4. **Link Prediction Module** — concat-MLP head + BCE.

### 7.1 HGT encoder

Per-edge attention from the HGT paper:
```
Attention(s, e, t) = softmax[ (K(s) · W^ATT_φ(e) · Q(t)ᵀ) · μ⟨τ(s),φ(e),τ(t)⟩ / √d ]
Message(s, e, t)   = M(s) · W^MSG_φ(e)
H̃(l)[t]           = ⊕ Attention(s,e,t) · Message(s,e,t)
H(l)[t]            = A-Linear_τ(t)(σ(H̃(l)[t])) + H(l-1)[t]    # residual
```

K-Linear / Q-Linear / M-Linear / A-Linear are per-type. `W^ATT_φ(e)`, `W^MSG_φ(e)` are per-edge-type. `μ` is the learnable triplet prior.

Implementation: PyG's `HGTConv`.

### 7.2 Edge features

Not implemented in Phase 1. Rating handled via edge-type split (`RATED_HIGH`/`RATED_LOW`). Genre/shelf weights stored in graph but not consumed by the model. Phase 2 plan:
- Option A: edge-type subsplits for count tiers (low/med/high).
- Option B: message multiplier `g(w_e)` (cheapest non-vanilla modification).
- Option C: inject edge feature into message via per-edge-type projection (custom `HGTConv` subclass).
- Option D: inject into attention as well.

Default for Phase 2 ablations: A or B. C/D deferred to Phase 3.

### 7.3 Hyperparameter starting points

| Param | Value |
|---|---|
| d_model | 256 |
| num_heads | 8 |
| num_layers | 3 |
| dropout | 0.2 |
| learning_rate | 1e-3 |
| weight_decay | 1e-4 |
| batch_size | 1024 |
| fanout per layer | [15, 10, 5] |
| neg_sampling_ratio | 5 |
| epochs | 50 (early stop on val NDCG) |
| id_dim (all ID embeddings) | 128 or 256 (tunable) |

---

## 8. Implementation Stack

- Python 3.10+, PyTorch 2.x.
- `torch-geometric` (PyG) for `HeteroData`, `HGTConv`, `NeighborLoader`.
- `sentence-transformers` for text embedding (BGE-base, frozen).
- `pandas` + `pyarrow` for data wrangling and Parquet I/O.
- `pymongo` for the Mongo source.
- `h5py` for embedding storage.
- `neo4j` (Phase 1) for exploration only.

Stoplists and other small artifacts in `data/processed/` as JSON.

---

## 9. Thesis

Memoir structure follows the M2 SII "SENG-SoRec" template (`Mémoire_M2_SII_012-2024.pdf`):

- **Chapter 3 Conception** — general-purpose framing, NOT Goodreads-specific. Goodreads is the evaluation dataset only.
  - 3.1 Introduction
  - 3.2 Proposed Approach
    - 3.2.1 General Objective
    - 3.2.2 General Description of Modules
    - 3.2.3 Detailed Description
  - 3.3 Training Phase (3.3.1 Datasets, 3.3.2 Model Training)
  - 3.4 Recommendation Phase
  - 3.5 Conclusion

Language: English.

TikZ architecture diagram (`goodreads_hgt_architecture.tex`) is multi-panel:
- A — Pipeline
- B — Node featurization (U/MN/TN)
- C — HGT layer internals (K/Q/M-Linear, W^ATT, W^MSG, μ, A-Linear)
- D — Link prediction head
- E — Schema example with **Review→Book** edge (NOT Review→Work)

Color palette: trainable=blue, frozen=gray, data=yellow, loss=red, goal=green. **No HGT-specific color.**

Dataset-specific names (`rated_high`, `is_read`, etc.) stay out of the Conception chapter; they appear only in the Training/Evaluation chapter.

---

## 10. Open Questions / TODOs

Deferred until after the base pipeline trains:

1. Author `role` subsplit (`first_author_of` / `illustrator_of` / etc.) — Phase 2.
2. Edge weights for genre/shelf: message multiplier vs count-tier subsplit — Phase 2 ablation.
3. Multi-task link prediction (one head per user→book edge type) — Phase 2/3.
4. Fine-tuning the text encoder — Phase 3.
5. `similar_books` inclusion — Phase 2 ablation.
6. Cold-start evaluation — Phase 3.
7. RTE timestamp granularity (year vs month/day) — Phase 2.
8. Negative sampling strategy (uniform vs popularity-weighted vs in-batch) — Phase 1 after base trains.
9. Publisher imprint canonicalization for top-30 publishers — Phase 2 if threshold-bucket coverage proves insufficient.
10. Revisit link-prediction head structure (currently concat-MLP placeholder) — Phase 1 after base trains.
11. Tighten normalization-stats computation to training split only — Phase 2.

---

## 11. Key References

Unchanged from v1. Hu et al. 2020 (HGT) is the primary model reference; Vaswani et al. 2017 (Transformer); Schlichtkrull et al. 2018 (RGCN baseline); Wan & McAuley 2018 / Wan et al. 2019 (UCSD Goodreads dataset).

---

## 12. Phase 1 Status (as of this writing)

Completed:
- Mongo ingestion of poetry slice.
- Data cleaning: shelf stoplist + threshold, genre normalization, language normalization with historical variants preserved, format/publisher normalization with UNK + OTHER.
- Neo4j load (`goodreads-poetry` DB) of all node types and edge types listed in §3.3 except `IN_SERIES` and `SIMILAR_TO`.
- Book text embeddings computed (BGE-base, 768 dim) and stored in `books_bge_base_768.h5`, sorted by ascending `book_id`.
- Book numeric block computed and stored as `book_numeric.npy`. Row-by-row alignment with the h5 verified.
- Categorical `*_to_idx.json` mappings for Language, Format, Publisher saved.

Next steps in order:
1. Save remaining `*_to_idx.json` mappings: User, Author, Work, Genre, Shelf.
2. Compute text embeddings + numeric blocks for Author, Work, Review.
3. Build PyG `HeteroData` from the artifacts.
4. Wire up the HGT model, link prediction head, training loop, and evaluation.

---

## 13. Session Continuity Note

Decisions with explicit rationale (in v1 §3.4 and the v2 changelog above) should not be relitigated without new evidence. The five categorical-as-U-node decision in particular was deliberate after empirical inspection — earlier "drop format and language" guidance is obsolete.

When in doubt, follow this file. When this file is silent, follow v1. When both are silent, ask before deciding.
