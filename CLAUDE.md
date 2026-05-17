# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HGT4Rec** — a heterogeneous knowledge-graph recommender (Master's thesis / PFE) built on the UCSD Goodreads poetry slice. The model is a Heterogeneous Graph Transformer (HGT, Hu et al. WWW 2020) trained for user→book link prediction. Full spec: `goodreads-hgt-project-spec-v2.md` (authoritative — follow it when this file is silent).

Pipeline stages (Phase 1 complete; training runs reproducible end-to-end):
1. **Preprocessing** — `preprocessing/preprocess.ipynb`: one notebook, Mongo → `data/processed/`. Produces every `*_to_idx.json`, `*_bge_base_768.h5`, `*_numeric.npy`, `splits/{train,val,test}.npz`, and `hetero_data.pt`. Parameterize `SLICE_DB` / `META_DB` at the top to point at any Goodreads slice.
2. **Training** — `training/train.py`: HGT4Rec link prediction with HeteroData input, checkpointing, resume, per-epoch memory-leak reset.
3. **Neo4j** — optional visualization only (last cell of the preprocess notebook, off by default).

## Running the Pipeline

**Preprocess a Mongo slice → `data/processed/`** (one notebook, top-to-bottom):
```
preprocessing/preprocess.ipynb
```
Set `SLICE_DB` / `META_DB` in the first cell to point at any partial Goodreads slice (default `poetry` / `goodreads`). Produces every `*_to_idx.json`, every `*_bge_base_768.h5`, every `*_numeric.npy`, the temporal `splits/{train,val,test}.npz`, and the assembled `hetero_data.pt`. Reads Mongo only — Neo4j is an optional final-cell export for visualization (`PUSH_TO_NEO4J = True`). GPU bound: ~30 min for the 4 BGE-768 embedding cells; rest is pandas + small I/O.

**Legacy scripts (deprecated by the notebook, kept for reference)**: `KG.ipynb`, `kg_mod.ipynb`, `embed.py`, `embed_authors.py`, `embed_works.py`, `build_idx_maps.py`, `build_review_idx.py`, `build_review_numeric.py`, `build_splits.py`, `build_hetero_data.py`, `fix_shelf_idx.py`. The notebook supersedes all of them — every step from raw Mongo to `hetero_data.pt` is in one place.

**Train** (canonical command lives in `default_command.txt`):
```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python -m training.train --epochs 20 --batch_size 256 --fanout 5 3 3 `
    --num_layers 3 --d_model 256 --id_dim 128 --num_workers 4 `
    --no_shelved --eval_max_users 5000 --note "<run label>"
```
Every knob has a `Config` default in `training/config.py`; CLI flags override. Common flags: `--resume <path/to/best.pt>`, `--max_train_edges N` (smoke test), `--persistent_workers` (off by default on Windows). Run output: `training/runs/<timestamp>/{best.pt, last.pt, metrics.csv, summary.json}`.

**Sanity-check embeddings:** the last cell of `preprocess.ipynb` does in-place verification (shapes, dtypes, idx-map round-trips, edge-index bounds, split totals). The standalone `preprocessing/sanity_check.ipynb` is still around if you need to re-verify without rerunning the full pipeline.

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

All artifacts below exist on disk.

```
data/processed/
├── books_bge_base_768.h5          # (36514, 768) float32, sorted by ascending book_id
├── book_numeric.npy               # (36514, 9) float32, same row order as h5
├── book_id_to_idx.json            # book_id → row index (canonical)
├── book_norm_stats.json           # imputation/z-score stats (full-corpus, Phase 1 shortcut)
├── book_numeric_columns.json      # column order documentation
│
├── authors_bge_base_768.h5
├── author_numeric.npy
├── author_numeric_columns.json
├── author_to_idx.json
│
├── works_bge_base_768.h5
├── work_numeric.npy
├── work_numeric_columns.json
├── work_norm_stats.json
├── work_to_idx.json
│
├── reviews_bge_base_768.h5
├── review_numeric.npy
├── review_numeric_columns.json
├── review_to_idx.json
│
├── user_to_idx.json               # union of reviews + interactions user_ids
├── genre_to_idx.json
├── shelf_to_idx.json
├── language_to_idx.json
├── format_to_idx.json
├── publisher_to_idx.json
│
├── hetero_data.pt                 # assembled PyG HeteroData (input to training)
├── hetero_meta.json               # node/edge type metadata
└── splits/
    ├── train.npz                  # user_idx, book_idx, label  (label ∈ {0,1})
    ├── val.npz
    ├── test.npz
    └── split_stats.json           # date boundaries, counts
```

Raw data (`data/raw/`) and large artifacts (`*.h5`, `*.npy`, `*.pt`) are gitignored. Small JSON mappings + splits are tracked.

## Status

**Phase 1 complete.** All preprocessing artifacts on disk; `HeteroData` + temporal splits built; modular HGT training pipeline runs end-to-end with checkpointing, resume, and per-epoch reset for memory-leak mitigation. Smoke + multi-epoch runs verified on laptop (6 GB GPU) and queued for full-spec runs on A4000 (16 GB).

**Training pipeline layout** (`training/`):
- `train.py` — entry point, CLI, loop, checkpointing
- `config.py` — single `Config` dataclass; every knob lives here
- `featurizer.py` — per-node-type U / MN / TN assembly at batch time
- `graph_filter.py` — applies edge-type toggles from `Config`
- `model.py` — `HGT4Rec` wrapper (encoder + head); `SUPERVISION_KEY` synthetic edge type
- `eval.py` — full-candidate-set eval with work-level dedup
- `utils.py` — run dir, CSV logger, AMP context, perf settings
- `components/{encoders,heads,samplers}/` — swappable modules for ablation (e.g. `HGTEncoder` ↔ alternatives; `MLPHead` ↔ alternatives)

**Next:**
1. Full-spec baseline run on A4000 (all edges incl. SHELVED, fanout [15,10,5], d_model=256, num_layers=3)
2. Ablation sweep (see `goodreads-hgt-project-spec-v2.md` §17)
3. Final test-set numbers from best checkpoint with `eval_max_users=0`

## Key Implementation Notes

### Data / featurization
- Text and numeric are stored **separately on disk**, concatenated in CPU at `HeteroData` construction time. Rationale: enables independent updates and Phase 3 encoder fine-tuning.
- Review text is NOT stored on Neo4j `Review` nodes — metadata only in Neo4j, text in HDF5.
- All embeddings are L2-normalized at generation time → cosine similarity = dot product downstream.
- Neo4j bulk writes use 10K-row chunked `MERGE` batches.
- Timestamps use Goodreads format `"%a %b %d %H:%M:%S %z %Y"` — parse with `utc=True` to normalize timezones.
- `*_norm_stats.json` stats are computed over the full corpus (Phase 1 shortcut, documented in `_note` field) — tighten to training split in Phase 2.
- The five U-category categorical node types (Genre, Shelf, Language, Format, Publisher) are a deliberate design decision — do not drop them without new evidence (spec §13).

### Training operational gotchas
- **PyG `NegativeSampling(mode="binary")` shifts labels by +1.** Input `{0, 1}` becomes batch `{0, 1, 2}` where `2` = original positive, `1` = explicit negative (RATED_LOW), `0` = sampled random negative. BCE target = `(edge_label == 2).float()`. Already handled in `train.py`.
- **AMP is OFF by default** (`cfg.amp_dtype="none"`). `pyg_lib.grouped_matmul` is fp32-only — bf16/fp16 autocast crashes inside HGTConv. TF32 is enabled via `enable_perf_settings()` and is safe.
- **Eval encode falls back to CPU** (`cfg.eval_encode_on_cpu=True`). The full-graph forward exceeds 6 GB GPU; encoding on CPU + scoring on GPU is the safe default. On a 16 GB+ GPU you can flip this off for faster eval.
- **Per-epoch reset is mandatory on long runs.** Worker subprocesses + PyTorch's caching allocator leak (~3 GB RAM/epoch on Windows). `train.py` does `del loader; gc.collect(); empty_cache(); build_loader()` at the end of each epoch — costs ~30 s/epoch, prevents the ~40-min throughput collapse.
- **Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** before launching on the A4000 to avoid VRAM-fragmentation spillover into shared system memory.
- **`persistent_workers=False` on Windows** (default). Setting it True causes the working-set trimmer to SIGKILL idle workers silently, after which PyG falls back to main-thread sampling at ~7× slowdown.
- `requirements.txt` is plain ASCII (pipreqs-generated, 10 lines). The conda env name for this project is `pfe`.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- ALWAYS read graphify-out/GRAPH_REPORT.md before reading any source files, running grep/glob searches, or answering codebase questions. The graph is your primary map of the codebase.
- IF graphify-out/wiki/index.md EXISTS, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
