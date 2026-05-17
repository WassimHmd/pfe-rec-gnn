"""Full-candidate-set eval with work-level dedup.

For each user with ≥1 positive eval edge:
  1. Score every book (or every work after dedup).
  2. Mask out books the user has any training-period interaction with.
  3. Compute NDCG@K, MRR, AUC against the user's positive eval edges.

Aggregate: mean over eligible users.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from training.utils import amp_ctx


def _add_n_id(data: HeteroData) -> HeteroData:
    """Attach n_id=arange so the featurizer treats `data` as one big batch."""
    out = data.clone()
    for nt in out.node_types:
        out[nt].n_id = torch.arange(out[nt].num_nodes)
    return out


def _build_seen_mask(
    train_user: np.ndarray,
    train_book: np.ndarray,
    shelved_edge_index: Tensor | None,
    num_users: int,
    num_books: int,
) -> Dict[int, set]:
    """Per-user set of books seen during training (any interaction)."""
    seen: Dict[int, set] = defaultdict(set)
    for u, b in zip(train_user.tolist(), train_book.tolist()):
        seen[u].add(b)
    if shelved_edge_index is not None and shelved_edge_index.numel() > 0:
        for u, b in zip(shelved_edge_index[0].tolist(), shelved_edge_index[1].tolist()):
            seen[u].add(b)
    return seen


def _build_book_to_work(data: HeteroData) -> Tensor:
    """Vector of shape (num_books,) mapping book_idx → work_idx (-1 if no edition)."""
    n_books = data["book"].num_nodes
    book_to_work = torch.full((n_books,), -1, dtype=torch.long)
    ei = data["book", "EDITION_OF", "work"].edge_index
    book_to_work[ei[0]] = ei[1]
    return book_to_work


def _ndcg_at_k(rank_positions: Tensor, k: int) -> float:
    """DCG@K / IDCG@K where rank_positions = 0-indexed ranks of positives."""
    in_topk = rank_positions[rank_positions < k]
    if in_topk.numel() == 0:
        return 0.0
    dcg  = (1.0 / torch.log2(in_topk.float() + 2.0)).sum().item()
    n_pos = rank_positions.numel()
    idcg = (1.0 / torch.log2(torch.arange(min(n_pos, k)).float() + 2.0)).sum().item()
    return dcg / idcg if idcg > 0 else 0.0


@torch.no_grad()
def evaluate(
    model,
    data: HeteroData,
    splits_dir: str | Path,
    split: str,
    cfg,
    device: torch.device,
) -> dict:
    splits_dir = Path(splits_dir)
    model.eval()

    train_npz = np.load(splits_dir / "train.npz")
    eval_npz  = np.load(splits_dir / f"{split}.npz")

    # Per-user positive eval edges (label=1 only)
    pos_mask = eval_npz["label"] == 1
    eval_u = eval_npz["user_idx"][pos_mask]
    eval_b = eval_npz["book_idx"][pos_mask]
    user_pos: Dict[int, list[int]] = defaultdict(list)
    for u, b in zip(eval_u.tolist(), eval_b.tolist()):
        user_pos[u].append(b)

    # Training-seen books per user (for masking from candidates)
    shelved_ei = data["user", "SHELVED", "book"].edge_index \
                 if ("user", "SHELVED", "book") in data.edge_types else None
    seen = _build_seen_mask(
        train_npz["user_idx"], train_npz["book_idx"],
        shelved_ei,
        num_users=data["user"].num_nodes,
        num_books=data["book"].num_nodes,
    )

    # Encode the full graph ONCE. On 6GB GPUs the per-relation attention activations
    # for the full graph exceed memory; fall back to CPU encode in that case.
    if cfg.eval_encode_on_cpu and device.type == "cuda":
        encode_dev = torch.device("cpu")
        model_orig_dev = next(model.parameters()).device
        model.to(encode_dev)
    else:
        encode_dev = device
        model_orig_dev = None

    full = _add_n_id(data).to(encode_dev)
    with amp_ctx(cfg.amp_dtype, encode_dev):
        h_dict = model.encode(full)

    if model_orig_dev is not None:
        model.to(model_orig_dev)
        torch.cuda.empty_cache()

    # Cast to fp32 and move to scoring device for fast per-user matmul.
    h_user_all = h_dict["user"].float().to(device)
    h_book_all = h_dict["book"].float().to(device)

    n_books = h_book_all.shape[0]

    # Work-level dedup setup
    if cfg.work_level_dedup:
        book_to_work = _build_book_to_work(data).to(device)
        n_works = data["work"].num_nodes
    else:
        book_to_work = None
        n_works = n_books

    # Iterate users
    users = sorted(user_pos.keys())
    if cfg.eval_max_users and cfg.eval_max_users > 0:
        users = users[: cfg.eval_max_users]

    ndcgs, mrrs, aucs = [], [], []
    K = cfg.eval_k

    for u in tqdm(users, desc=f"eval/{split}"):
        positives = user_pos[u]
        if not positives:
            continue

        # Score all books
        h_u = h_user_all[u].unsqueeze(0).expand(n_books, -1)
        scores = model.head(h_u, h_book_all)   # (N_books,)

        # Mask out training-seen books
        u_seen = seen.get(u, set())
        if u_seen:
            scores_masked = scores.clone()
            scores_masked[list(u_seen)] = float("-inf")
        else:
            scores_masked = scores

        # Work-level dedup: collapse to work-level scores (max per work)
        if book_to_work is not None:
            work_scores = torch.full((n_works,), float("-inf"), device=device)
            work_scores.scatter_reduce_(0, book_to_work, scores_masked, reduce="amax", include_self=True)
            scores_for_rank = work_scores
            positive_ids = set(book_to_work[b].item() for b in positives)
            # discard -1 (no-work-mapping books, shouldn't happen but safe)
            positive_ids.discard(-1)
        else:
            scores_for_rank = scores_masked
            positive_ids = set(positives)

        if not positive_ids:
            continue

        # Rank positions of positives (0-indexed)
        sorted_idx = torch.argsort(scores_for_rank, descending=True)
        rank_of = torch.full((scores_for_rank.shape[0],), -1, dtype=torch.long, device=device)
        rank_of[sorted_idx] = torch.arange(scores_for_rank.shape[0], device=device)
        pos_ranks = rank_of[torch.tensor(list(positive_ids), device=device)]

        # NDCG@K
        ndcgs.append(_ndcg_at_k(pos_ranks.cpu(), K))
        # MRR (over the first positive)
        mrrs.append(float((1.0 / (pos_ranks.float() + 1.0)).max().item()))

        # AUC against all eligible negatives
        y_score = scores_for_rank.cpu().numpy()
        finite_mask = np.isfinite(y_score)
        if finite_mask.sum() < 2:
            continue
        y_true = np.zeros_like(y_score)
        y_true[list(positive_ids)] = 1
        y_true = y_true[finite_mask]
        y_score = y_score[finite_mask]
        if y_true.sum() == 0 or y_true.sum() == y_true.size:
            continue
        aucs.append(roc_auc_score(y_true, y_score))

    return {
        f"ndcg{K}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "mrr":      float(np.mean(mrrs)) if mrrs else 0.0,
        "auc":      float(np.mean(aucs)) if aucs else 0.0,
        "n_users":  len(ndcgs),
    }
