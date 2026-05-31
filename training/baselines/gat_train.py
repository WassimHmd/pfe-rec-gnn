"""GAT baseline training script.

Run from repo root:
    python -m training.baselines.gat_train

Differences from GCN:
- Attention-weighted aggregation instead of fixed symmetric normalisation.
- Uses PyG GATConv with multi-head attention.
- edge_index (COO) instead of pre-normalised sparse adjacency.
- Same BPR loss + dot-product scoring.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from training.components.encoders.gat import GATRec, build_bipartite_edge_index
from training.eval import evaluate
from training.utils import CsvLogger, device_auto, enable_perf_settings, set_seed
from training.baselines.utils import (
    add_common_args, bpr_loss, BPRSampler, EvalCfg, eval_test_set,
)


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p)
    # GAT-specific defaults: d=64 keeps params ~27M (vs 106M at d=256).
    # The attention mechanism provides expressiveness even at lower dim.
    p.set_defaults(d_model=64, batch_size=4096)
    p.add_argument("--num_layers", type=int,   default=3)
    p.add_argument("--num_heads",  type=int,   default=4,
                   help="Attention heads (d_model must be divisible by num_heads)")
    p.add_argument("--dropout",    type=float, default=0.1)
    return p.parse_args()


def main():
    args   = parse_args()
    device = device_auto()
    set_seed(args.seed)
    enable_perf_settings()

    splits_dir = Path(args.splits_dir)
    run_dir    = Path(args.runs_root) / ("gat_" + time.strftime("%Y%m%d-%H%M%S"))
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

    z    = np.load(splits_dir / "train.npz")
    mask = z["label"] == 1
    edge_index = build_bipartite_edge_index(
        z["user_idx"][mask].astype(np.int64),
        z["book_idx"][mask].astype(np.int64),
        n_users,
    ).to(device)
    print(f"  train positive edges (both dirs): {edge_index.shape[1]:,}")

    # ── Model ───────────────────────────────────────────────────────────────
    model = GATRec(
        n_users=n_users,
        n_books=n_books,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        edge_index=edge_index,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GATRec : {n_params:,} parameters  "
          f"(d={args.d_model}, L={args.num_layers}, heads={args.num_heads})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sampler   = BPRSampler(splits_dir, n_books, args.batch_size, args.seed)
    eval_cfg  = EvalCfg(args)

    best_ndcg, best_epoch, patience = -1.0, -1, 0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss, n_batches = 0.0, 0

        pbar = tqdm(sampler, desc=f"epoch {epoch} train", leave=False, dynamic_ncols=True)
        for u, pos, neg in pbar:
            u, pos, neg = u.to(device), pos.to(device), neg.to(device)
            optimizer.zero_grad()

            pos_score, neg_score = model(u, pos, neg)
            h_u_all, h_b_all    = model._propagate()
            loss = bpr_loss(pos_score, neg_score,
                            h_u_all[u], h_b_all[pos], h_b_all[neg],
                            args.weight_decay)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.close()

        train_loss = total_loss / max(n_batches, 1)
        train_t    = time.time() - t0

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

        ckpt = {"epoch": epoch, "state": model.state_dict(),
                "optim": optimizer.state_dict(), "config": vars(args),
                "metrics": val_m}
        torch.save(ckpt, run_dir / "last.pt")

        if val_m[f"ndcg{args.eval_k}"] > best_ndcg:
            best_ndcg, best_epoch, patience = val_m[f"ndcg{args.eval_k}"], epoch, 0
            torch.save(ckpt, run_dir / "best.pt")
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print(f"Early stop (patience={args.early_stop_patience})")
                break

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (run_dir / "summary.json").write_text(json.dumps(
        {"best_epoch": best_epoch, "best_ndcg10": best_ndcg}, indent=2))
    print(f"\nBest val ndcg@{args.eval_k} = {best_ndcg:.4f} at epoch {best_epoch}")
    print(f"Best checkpoint: {run_dir / 'best.pt'}")

    eval_test_set(model, data, splits_dir, run_dir, eval_cfg, device, logger)


if __name__ == "__main__":
    main()
