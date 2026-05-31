"""Shared utilities for all baseline training scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ── Common CLI args ───────────────────────────────────────────────────────────

def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--d_model",    type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int,   default=2048)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--early_stop_patience", type=int, default=5)
    p.add_argument("--eval_max_users", type=int, default=0)
    p.add_argument("--eval_k",     type=int,   default=10)
    p.add_argument("--data_path",  type=str,
                   default="preprocessing/data/processed/hetero_data.pt")
    p.add_argument("--splits_dir", type=str,
                   default="preprocessing/data/processed/splits")
    p.add_argument("--runs_root",  type=str,   default="training/runs")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--note",       type=str,   default="")


# ── BPR loss (graph-based baselines) ─────────────────────────────────────────

def bpr_loss(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
    user_emb:  torch.Tensor,
    pos_emb:   torch.Tensor,
    neg_emb:   torch.Tensor,
    weight_decay: float,
) -> torch.Tensor:
    loss = -F.logsigmoid(pos_score - neg_score).mean()
    reg  = (user_emb.norm(dim=-1).pow(2).mean()
            + pos_emb.norm(dim=-1).pow(2).mean()
            + neg_emb.norm(dim=-1).pow(2).mean()) / 3
    return loss + weight_decay * reg


# ── BPR triplet sampler (graph-based baselines) ───────────────────────────────

class BPRSampler:
    """Epoch iterator of (user, pos_book, neg_book) tensors from train positives."""

    def __init__(self, splits_dir: Path, n_books: int, batch_size: int, seed: int):
        z = np.load(splits_dir / "train.npz")
        mask = z["label"] == 1
        self.users      = z["user_idx"][mask].astype(np.int64)
        self.books      = z["book_idx"][mask].astype(np.int64)
        self.n_books    = n_books
        self.batch_size = batch_size
        self.rng        = np.random.default_rng(seed)

    def __len__(self) -> int:
        return max(1, len(self.users) // self.batch_size)

    def __iter__(self):
        perm = self.rng.permutation(len(self.users))
        for start in range(0, len(self.users) - self.batch_size + 1, self.batch_size):
            idx = perm[start : start + self.batch_size]
            u   = torch.from_numpy(self.users[idx]).long()
            pos = torch.from_numpy(self.books[idx]).long()
            neg = torch.from_numpy(
                self.rng.integers(0, self.n_books, size=len(idx), dtype=np.int64)
            ).long()
            yield u, pos, neg


# ── BCE pair sampler (NCF) ────────────────────────────────────────────────────

class BCESampler:
    """Epoch iterator of (user, book, label) tensors. Negatives sampled uniformly."""

    def __init__(self, splits_dir: Path, n_books: int, batch_size: int,
                 seed: int, neg_ratio: int = 4):
        z = np.load(splits_dir / "train.npz")
        self.pos_u   = z["user_idx"][z["label"] == 1].astype(np.int64)
        self.pos_b   = z["book_idx"][z["label"] == 1].astype(np.int64)
        self.n_books    = n_books
        self.batch_size = batch_size
        self.neg_ratio  = neg_ratio
        self.rng        = np.random.default_rng(seed)

    def __len__(self) -> int:
        n = len(self.pos_u) * (1 + self.neg_ratio)
        return max(1, n // self.batch_size)

    def __iter__(self):
        n_pos = len(self.pos_u)
        neg_u = np.repeat(self.pos_u, self.neg_ratio)
        neg_b = self.rng.integers(0, self.n_books, size=n_pos * self.neg_ratio,
                                  dtype=np.int64)
        all_u = np.concatenate([self.pos_u, neg_u])
        all_b = np.concatenate([self.pos_b, neg_b])
        all_y = np.concatenate([np.ones(n_pos, dtype=np.float32),
                                 np.zeros(n_pos * self.neg_ratio, dtype=np.float32)])
        perm = self.rng.permutation(len(all_u))
        all_u, all_b, all_y = all_u[perm], all_b[perm], all_y[perm]

        for start in range(0, len(all_u) - self.batch_size + 1, self.batch_size):
            sl = slice(start, start + self.batch_size)
            yield (torch.from_numpy(all_u[sl]).long(),
                   torch.from_numpy(all_b[sl]).long(),
                   torch.from_numpy(all_y[sl]))


# ── Minimal eval config ───────────────────────────────────────────────────────

class EvalCfg:
    def __init__(self, args):
        self.eval_k             = args.eval_k
        self.eval_max_users     = args.eval_max_users
        self.work_level_dedup   = True
        self.eval_encode_on_cpu = False
        self.amp_dtype          = "none"


# ── Test-set evaluation (called after training) ───────────────────────────────

def eval_test_set(model, data, splits_dir: Path, run_dir: Path,
                  eval_cfg, device: torch.device, logger=None) -> dict:
    """Load best.pt → evaluate on test split → save test_results.json."""
    from training.eval import evaluate

    best_pt = run_dir / "best.pt"
    if not best_pt.exists():
        print("No best.pt found — skipping test evaluation.")
        return {}

    ckpt = torch.load(best_pt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state"])
    print(f"\nLoaded best checkpoint (epoch {ckpt['epoch']}) for test evaluation.")

    # Use eval_max_users=0 for test (full set, no cap)
    test_cfg = EvalCfg.__new__(EvalCfg)
    test_cfg.__dict__.update(eval_cfg.__dict__)
    test_cfg.eval_max_users = 0

    test_m = evaluate(model, data, splits_dir, split="test",
                      cfg=test_cfg, device=device)

    k = eval_cfg.eval_k
    print(f"\n{'='*50}")
    print(f"TEST SET RESULTS  (best epoch = {ckpt['epoch']})")
    print(f"  NDCG@{k}  : {test_m[f'ndcg{k}']:.4f}")
    print(f"  MRR     : {test_m['mrr']:.4f}")
    print(f"  AUC     : {test_m['auc']:.4f}")
    print(f"  Users   : {test_m['n_users']:,}")
    print(f"{'='*50}")

    result = {
        "best_epoch": ckpt["epoch"],
        f"test_ndcg{k}": test_m[f"ndcg{k}"],
        "test_mrr":   test_m["mrr"],
        "test_auc":   test_m["auc"],
        "test_users": test_m["n_users"],
    }
    (run_dir / "test_results.json").write_text(json.dumps(result, indent=2))

    if logger is not None:
        logger.log({"epoch": f"TEST(e{ckpt['epoch']})",
                    "loss": "", "ndcg10": test_m[f"ndcg{k}"],
                    "mrr": test_m["mrr"], "auc": test_m["auc"],
                    "val_users": test_m["n_users"],
                    "train_sec": "", "val_sec": ""})
    return result
