"""LightGCN baseline training script.

Run from repo root:
    python -m training.baselines.lightgcn_train

Key differences vs HGT training:
- Full-graph propagation each step (no NeighborLoader).
- BPR loss with uniform negative sampling.
- No side-information — ID embeddings only.
- Dot-product scoring head.
- eval.py is reused as-is (same encode/head interface).
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from tqdm import tqdm

from training.components.encoders.lightgcn import LightGCN, build_norm_adj
from training.eval import evaluate
from training.utils import (
    CsvLogger, device_auto, enable_perf_settings, make_run_dir, set_seed,
)
from training.baselines.utils import eval_test_set


# ── config ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d_model",    type=int,   default=256)
    p.add_argument("--num_layers", type=int,   default=3)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="L2 reg on ID embeddings (lambda in BPR)")
    p.add_argument("--batch_size", type=int,   default=2048)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--early_stop_patience", type=int, default=5)
    p.add_argument("--eval_max_users", type=int, default=0)
    p.add_argument("--eval_k",     type=int,   default=10)
    p.add_argument("--data_path",  type=str,   default="preprocessing/data/processed/hetero_data.pt")
    p.add_argument("--splits_dir", type=str,   default="preprocessing/data/processed/splits")
    p.add_argument("--runs_root",  type=str,   default="training/runs")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--note",       type=str,   default="")
    return p.parse_args()


# ── BPR loss ─────────────────────────────────────────────────────────────────

def bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor,
             user_emb: torch.Tensor, pos_emb: torch.Tensor,
             neg_emb: torch.Tensor, weight_decay: float) -> torch.Tensor:
    loss = -F.logsigmoid(pos_score - neg_score).mean()
    reg  = (user_emb.norm(dim=-1).pow(2).mean()
            + pos_emb.norm(dim=-1).pow(2).mean()
            + neg_emb.norm(dim=-1).pow(2).mean()) / 3
    return loss + weight_decay * reg


# ── adjacency construction from train split ───────────────────────────────────

def build_train_graph(splits_dir: Path, n_users: int, n_books: int):
    """Return norm_adj built from positive train edges only."""
    z = np.load(splits_dir / "train.npz")
    mask = z["label"] == 1
    return build_norm_adj(
        z["user_idx"][mask].astype(np.int64),
        z["book_idx"][mask].astype(np.int64),
        n_users, n_books,
    )


# ── sampler ───────────────────────────────────────────────────────────────────

class BPRSampler:
    """Uniform BPR triplet sampler from train positives."""

    def __init__(self, splits_dir: Path, n_books: int, batch_size: int, seed: int):
        z = np.load(splits_dir / "train.npz")
        mask = z["label"] == 1
        self.users  = z["user_idx"][mask].astype(np.int64)
        self.books  = z["book_idx"][mask].astype(np.int64)
        self.n_books = n_books
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return max(1, len(self.users) // self.batch_size)

    def __iter__(self):
        n = len(self.users)
        perm = self.rng.permutation(n)
        for start in range(0, n - self.batch_size + 1, self.batch_size):
            idx = perm[start : start + self.batch_size]
            u   = torch.from_numpy(self.users[idx]).long()
            pos = torch.from_numpy(self.books[idx]).long()
            neg = torch.from_numpy(
                self.rng.integers(0, self.n_books, size=len(idx), dtype=np.int64)
            ).long()
            yield u, pos, neg


# ── eval shim ─────────────────────────────────────────────────────────────────

class _EvalCfg:
    """Minimal cfg object for evaluate()."""
    def __init__(self, args):
        self.eval_k             = args.eval_k
        self.eval_max_users     = args.eval_max_users
        self.work_level_dedup   = True
        self.eval_encode_on_cpu = False   # LightGCN is cheap; always on GPU
        self.amp_dtype          = "none"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = device_auto()
    set_seed(args.seed)
    enable_perf_settings()

    splits_dir = Path(args.splits_dir)
    runs_root  = Path(args.runs_root)

    # Minimal cfg-like object for make_run_dir
    class _Cfg:
        encoder = "lightgcn"
        def __init__(self, a):
            self.__dict__.update(vars(a))
    cfg_obj = _Cfg(args)

    run_dir = runs_root / ("lightgcn_" + time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    if args.note:
        (run_dir / "note.txt").write_text(args.note)
    logger = CsvLogger(run_dir / "metrics.csv")

    print(f"Run dir : {run_dir}")
    print(f"Device  : {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    print(f"\nLoading {args.data_path} ...")
    data: HeteroData = torch.load(args.data_path, weights_only=False)
    n_users = data["user"].num_nodes
    n_books = data["book"].num_nodes
    print(f"  users={n_users:,}  books={n_books:,}")

    norm_adj = build_train_graph(splits_dir, n_users, n_books).to(device)

    # ── Model ───────────────────────────────────────────────────────────────
    model = LightGCN(
        n_users=n_users,
        n_books=n_books,
        d_model=args.d_model,
        num_layers=args.num_layers,
        norm_adj=norm_adj,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LightGCN: {n_params:,} parameters  "
          f"(d={args.d_model}, L={args.num_layers})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sampler   = BPRSampler(splits_dir, n_books, args.batch_size, args.seed)
    eval_cfg  = _EvalCfg(args)

    best_ndcg  = -1.0
    best_epoch = -1
    patience   = 0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss, n_batches = 0.0, 0

        pbar = tqdm(sampler, desc=f"epoch {epoch} train", leave=False, dynamic_ncols=True)
        for u, pos, neg in pbar:
            u, pos, neg = u.to(device), pos.to(device), neg.to(device)
            optimizer.zero_grad()

            pos_score, neg_score = model(u, pos, neg)

            # Grab the embeddings used in this batch for L2 reg
            h_all_u, h_all_b = model._propagate()
            loss = bpr_loss(
                pos_score, neg_score,
                h_all_u[u], h_all_b[pos], h_all_b[neg],
                args.weight_decay,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.close()

        train_loss = total_loss / max(n_batches, 1)
        train_t    = time.time() - t0

        # ── Eval ────────────────────────────────────────────────────────────
        t0    = time.time()
        val_m = evaluate(model, data, splits_dir, split="val",
                         cfg=eval_cfg, device=device)
        val_t = time.time() - t0

        row = {
            "epoch":     epoch,
            "loss":      round(train_loss, 6),
            "ndcg10":    round(val_m[f"ndcg{args.eval_k}"], 6),
            "mrr":       round(val_m["mrr"], 6),
            "auc":       round(val_m["auc"], 6),
            "val_users": val_m["n_users"],
            "train_sec": round(train_t, 2),
            "val_sec":   round(val_t, 2),
        }
        logger.log(row)
        print(f"epoch {epoch:>3} | loss={train_loss:.4f}  "
              f"ndcg@{args.eval_k}={val_m[f'ndcg{args.eval_k}']:.4f}  "
              f"mrr={val_m['mrr']:.4f}  auc={val_m['auc']:.4f}  "
              f"({train_t:.1f}s train / {val_t:.1f}s val)")

        ckpt = {
            "epoch":  epoch,
            "state":  model.state_dict(),
            "optim":  optimizer.state_dict(),
            "config": vars(args),
            "metrics": val_m,
        }
        torch.save(ckpt, run_dir / "last.pt")

        if val_m[f"ndcg{args.eval_k}"] > best_ndcg:
            best_ndcg  = val_m[f"ndcg{args.eval_k}"]
            best_epoch = epoch
            patience   = 0
            torch.save(ckpt, run_dir / "best.pt")
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print(f"Early stop (patience={args.early_stop_patience})")
                break

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (run_dir / "summary.json").write_text(json.dumps({
        "best_epoch":  best_epoch,
        "best_ndcg10": best_ndcg,
    }, indent=2))
    print(f"\nBest val ndcg@{args.eval_k} = {best_ndcg:.4f} at epoch {best_epoch}")
    print(f"Best checkpoint: {run_dir / 'best.pt'}")

    eval_test_set(model, data, splits_dir, run_dir, eval_cfg, device, logger)


if __name__ == "__main__":
    main()
