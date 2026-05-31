"""NCF baseline training script.

Run from repo root:
    python -m training.baselines.ncf_train

Neural Collaborative Filtering (He et al., WWW 2017) — MLP variant.
No graph structure.  BCE loss with uniform negative sampling.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from tqdm import tqdm

from training.components.encoders.ncf import NCF
from training.eval import evaluate
from training.utils import CsvLogger, device_auto, enable_perf_settings, set_seed
from training.baselines.utils import add_common_args, BCESampler, EvalCfg, eval_test_set


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--hidden",     type=int,   default=128,
                   help="MLP hidden units (spec: 128)")
    p.add_argument("--num_layers", type=int,   default=2,
                   help="MLP layers (spec: 2)")
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--neg_ratio",  type=int,   default=4,
                   help="Negative samples per positive")
    return p.parse_args()


def main():
    args   = parse_args()
    device = device_auto()
    set_seed(args.seed)
    enable_perf_settings()

    splits_dir = Path(args.splits_dir)
    run_dir    = Path(args.runs_root) / ("ncf_" + time.strftime("%Y%m%d-%H%M%S"))
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

    # ── Model ───────────────────────────────────────────────────────────────
    model = NCF(
        n_users=n_users,
        n_books=n_books,
        d_model=args.d_model,
        hidden=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"NCF    : {n_params:,} parameters  "
          f"(d={args.d_model}, hidden={args.hidden}, L={args.num_layers})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    sampler   = BCESampler(splits_dir, n_books, args.batch_size,
                           args.seed, args.neg_ratio)
    eval_cfg  = EvalCfg(args)
    loss_fn   = torch.nn.BCEWithLogitsLoss()

    best_ndcg, best_epoch, patience = -1.0, -1, 0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss, n_batches = 0.0, 0

        pbar = tqdm(sampler, desc=f"epoch {epoch} train", leave=False, dynamic_ncols=True)
        for u, b, y in pbar:
            u, b, y = u.to(device), b.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(u, b)
            loss   = loss_fn(logits, y)
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
