"""Score a saved checkpoint on the test split.

Loads a `best.pt` / `last.pt`, rebuilds the model from the config stored inside
the checkpoint, and runs the same `evaluate()` used during training — but
against `splits/test.npz` and (by default) over all users.

Run from repo root, with the conda env active:
    python -m training.eval_test --checkpoint training/runs/<ts>/best.pt
    python -m training.eval_test --checkpoint <path> --eval_max_users 5000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

from training.config import Config
from training.eval import evaluate
from training.graph_filter import filter_graph
from training.model import HGT4Rec, SUPERVISION_KEY
from training.utils import device_auto, enable_perf_settings, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="path to a best.pt / last.pt")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--eval_max_users", type=int, default=None,
                   help="cap eval users (0 or unset = all users)")
    p.add_argument("--encode_on_cpu", action="store_true", default=None,
                   help="force full-graph encode on CPU (auto from ckpt config otherwise)")
    p.add_argument("--encode_on_gpu", action="store_true",
                   help="force full-graph encode on GPU (overrides encode_on_cpu)")
    p.add_argument("--data_path", type=str, default=None,
                   help="override HeteroData path; defaults to ckpt config's data_path")
    p.add_argument("--splits_dir", type=str, default=None,
                   help="override splits dir; defaults to ckpt config's splits_dir")
    p.add_argument("--dump_per_user", action="store_true",
                   help="also write a per-user JSON (one row per eval'd user)")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    device = device_auto()
    enable_perf_settings()

    # ── Load checkpoint ────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch     = ckpt.get("epoch", "?")
    saved_cfg = ckpt.get("config", {})
    val_m     = ckpt.get("metrics", {})
    print(f"  epoch: {epoch}")
    print(f"  saved val metrics: {val_m}")

    # ── Rebuild Config from saved dict ─────────────────────────────────────────
    cfg = Config(**{k: v for k, v in saved_cfg.items() if k in Config.__dataclass_fields__})
    set_seed(cfg.seed)

    # CLI overrides
    if args.eval_max_users is not None:
        cfg.eval_max_users = args.eval_max_users
    else:
        cfg.eval_max_users = 0   # default to full user set for the headline number
    if args.encode_on_gpu:
        cfg.eval_encode_on_cpu = False
    elif args.encode_on_cpu:
        cfg.eval_encode_on_cpu = True
    if args.data_path:
        cfg.data_path = args.data_path
    if args.splits_dir:
        cfg.splits_dir = args.splits_dir

    print(f"\nConfig (effective):")
    for k in ["d_model", "num_layers", "id_dim", "use_shelved_edges",
              "eval_k", "eval_max_users", "eval_encode_on_cpu",
              "data_path", "splits_dir"]:
        print(f"  {k:<22} {getattr(cfg, k)}")

    # ── Load HeteroData + apply same edge filtering used during training ──────
    print(f"\nLoading {cfg.data_path}...")
    data: HeteroData = torch.load(cfg.data_path, weights_only=False)
    data = filter_graph(data, cfg)
    print(f"  {len(data.node_types)} node types, {len(data.edge_types)} edge types")

    # The model registers a synthetic supervision edge type. Mirror train.py.
    data[SUPERVISION_KEY].edge_index = torch.empty((2, 0), dtype=torch.long)

    # ── Build model, load state ───────────────────────────────────────────────
    model = HGT4Rec(data, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    state = ckpt["state"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  ⚠ missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  ⚠ unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if not missing and not unexpected:
        print("  state loaded clean (no missing/unexpected keys)")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print(f"\nEvaluating on split='{args.split}' "
          f"(eval_max_users={cfg.eval_max_users or 'ALL'}, "
          f"encode_on_cpu={cfg.eval_encode_on_cpu})...")
    t0 = time.time()
    metrics = evaluate(model, data, cfg.splits_dir, split=args.split,
                       cfg=cfg, device=device, return_per_user=True)
    dt = time.time() - t0

    per_user = metrics.pop("per_user")

    print("\n" + "=" * 60)
    print(f"  {args.split.upper()} RESULTS — checkpoint {ckpt_path.name} (epoch {epoch})")
    print("=" * 60)
    print(f"  Aggregate over {metrics['n_users']:,} eligible users:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:<10} {v:.6f}")
        else:
            print(f"    {k:<10} {v}")
    print(f"    elapsed   {dt:.1f}s")

    # ── Bucketed breakdown by # positives per user ─────────────────────────────
    import numpy as np
    BUCKETS = [
        ("1",      lambda n: n == 1),
        ("2",      lambda n: n == 2),
        ("3-5",    lambda n: 3 <= n <= 5),
        ("6-10",   lambda n: 6 <= n <= 10),
        ("11-20",  lambda n: 11 <= n <= 20),
        ("21+",    lambda n: n >= 21),
    ]
    K = cfg.eval_k
    print("\n  Breakdown by # held-out positives per user:")
    print(f"    {'n_pos':<7} {'users':>9} {'share':>7}   {'ndcg@'+str(K):>10} {'mrr':>8} {'auc':>8} "
          f"{'best_rank<K%':>13}")

    n_total = len(per_user)
    bucket_rows = []
    for label, predicate in BUCKETS:
        bucket = [r for r in per_user if predicate(r["n_pos"])]
        if not bucket:
            row = {"label": label, "n_users": 0, "share": 0.0,
                   f"ndcg{K}": None, "mrr": None, "auc": None, "hit@K_pct": None}
            print(f"    {label:<7} {0:>9}    0.0%   "
                  f"{'-':>10} {'-':>8} {'-':>8} {'-':>13}")
        else:
            ndcg = float(np.mean([r["ndcg"] for r in bucket]))
            mrr  = float(np.mean([r["mrr"]  for r in bucket]))
            aucs = [r["auc"] for r in bucket if r["auc"] is not None]
            auc  = float(np.mean(aucs)) if aucs else None
            hit_at_k = float(np.mean([r["best_rank"] < K for r in bucket])) * 100
            row = {
                "label": label, "n_users": len(bucket),
                "share": len(bucket) / n_total * 100,
                f"ndcg{K}": ndcg, "mrr": mrr, "auc": auc, "hit@K_pct": hit_at_k,
            }
            auc_str = f"{auc:.4f}" if auc is not None else "  -   "
            print(f"    {label:<7} {len(bucket):>9} {row['share']:>6.1f}%   "
                  f"{ndcg:>10.4f} {mrr:>8.4f} {auc_str:>8} {hit_at_k:>12.1f}%")
        bucket_rows.append(row)

    # ── Save ───────────────────────────────────────────────────────────────────
    out = ckpt_path.parent / f"{args.split}_metrics.json"
    out.write_text(json.dumps({
        "checkpoint": str(ckpt_path),
        "epoch": epoch,
        "split": args.split,
        "eval_max_users": cfg.eval_max_users,
        "elapsed_sec": dt,
        "aggregate": metrics,
        "buckets": bucket_rows,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")

    # Optionally dump the full per-user table for offline analysis
    if args.dump_per_user:
        per_user_path = ckpt_path.parent / f"{args.split}_per_user.json"
        per_user_path.write_text(json.dumps(per_user), encoding="utf-8")
        print(f"Saved {per_user_path} ({len(per_user):,} users)")


if __name__ == "__main__":
    main()
