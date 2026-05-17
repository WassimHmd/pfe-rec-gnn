"""Training entry point.

Run from repo root:
    conda run -n pfe python -m training.train

Override anything from Config by editing training/config.py or by editing the
`cfg = Config(...)` line in `main()` below.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling
from tqdm import tqdm

from training.config import Config
from training.eval import evaluate
from training.graph_filter import filter_graph
from training.model import HGT4Rec, SUPERVISION_KEY
from training.utils import CsvLogger, amp_ctx, device_auto, enable_perf_settings, make_run_dir, set_seed


def build_supervision(splits_dir: Path, use_rated_low_as_neg: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenated (2, N) supervision edge_label_index + (N,) labels from train.npz."""
    z = np.load(splits_dir / "train.npz")
    u = torch.from_numpy(z["user_idx"]).long()
    b = torch.from_numpy(z["book_idx"]).long()
    lab = torch.from_numpy(z["label"]).float()

    if not use_rated_low_as_neg:
        keep = lab > 0
        u, b, lab = u[keep], b[keep], lab[keep]

    return torch.stack([u, b], dim=0), lab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--fanout",     type=int,   nargs="+", default=None)
    parser.add_argument("--eval_max_users", type=int, default=None)
    parser.add_argument("--d_model",    type=int,   default=None)
    parser.add_argument("--num_layers", type=int,   default=None)
    parser.add_argument("--id_dim",     type=int,   default=None)
    parser.add_argument("--no_shelved", action="store_true",
                        help="drop SHELVED edges (huge: 1.4M)")
    parser.add_argument("--max_train_edges", type=int, default=0,
                        help="cap supervision edges (0 = all). Useful for smoke tests.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="NeighborLoader worker subprocesses (default 4; 0 = main process)")
    parser.add_argument("--persistent_workers", action="store_true",
                        help="keep workers alive between epochs (default off; "
                             "Windows can SIGKILL idle workers' memory over long runs)")
    parser.add_argument("--resume", type=str, default=None,
                        help="path to a best.pt to resume from (e.g. training/runs/<ts>/best.pt)")
    parser.add_argument("--note",       type=str,   default="")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs       is not None: cfg.epochs       = args.epochs
    if args.batch_size   is not None: cfg.batch_size   = args.batch_size
    if args.lr           is not None: cfg.lr           = args.lr
    if args.fanout       is not None: cfg.fanout       = args.fanout
    if args.eval_max_users is not None: cfg.eval_max_users = args.eval_max_users
    if args.d_model      is not None: cfg.d_model      = args.d_model
    if args.num_layers   is not None: cfg.num_layers   = args.num_layers
    if args.id_dim       is not None: cfg.id_dim       = args.id_dim
    if args.num_workers  is not None: cfg.num_workers  = args.num_workers
    if args.persistent_workers:       cfg.persistent_workers = True
    if args.no_shelved:               cfg.use_shelved_edges = False

    device = device_auto()
    set_seed(cfg.seed)
    enable_perf_settings()
    run_dir = make_run_dir(cfg, cfg.runs_root)
    logger  = CsvLogger(run_dir / "metrics.csv")
    if args.note:
        (run_dir / "note.txt").write_text(args.note, encoding="utf-8")

    print(f"Run dir: {run_dir}")
    print(f"Device : {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    print(f"\nLoading {cfg.data_path}...")
    data: HeteroData = torch.load(cfg.data_path, weights_only=False)
    data = filter_graph(data, cfg)
    print(f"  {len(data.node_types)} node types, {len(data.edge_types)} edge types")

    # Supervision edges
    sup_ei, sup_lab = build_supervision(Path(cfg.splits_dir), cfg.use_rated_low_as_neg)
    if args.max_train_edges > 0 and sup_ei.shape[1] > args.max_train_edges:
        perm = torch.randperm(sup_ei.shape[1], generator=torch.Generator().manual_seed(cfg.seed))
        sup_ei  = sup_ei[:, perm[: args.max_train_edges]]
        sup_lab = sup_lab[perm[: args.max_train_edges]]
        print(f"  (capped to {args.max_train_edges:,} edges for smoke test)")
    n_pos = int((sup_lab > 0).sum().item())
    n_neg = int((sup_lab == 0).sum().item())
    print(f"  Supervision: {sup_ei.shape[1]:,} edges  (pos={n_pos:,} neg={n_neg:,})")

    # Register synthetic supervision edge type (empty edge_index) — required so PyG can
    # attach edge_label_index to it. The model strips this edge type before HGTConv.
    data[SUPERVISION_KEY].edge_index = torch.empty((2, 0), dtype=torch.long)

    # ── Model ───────────────────────────────────────────────────────────────
    model = HGT4Rec(data, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn   = BCEWithLogitsLoss()

    # ── Optional resume from checkpoint ─────────────────────────────────────
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state"])
        if "optim" in ckpt:
            optimizer.load_state_dict(ckpt["optim"])
            opt_note = "with optimizer state"
        else:
            opt_note = "(no optimizer state in checkpoint — Adam will warm up fresh)"
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch} {opt_note}")
        prev = ckpt.get("metrics", {})
        if prev:
            print(f"  Previous best val: {prev}")

    # ── Loader factory ──────────────────────────────────────────────────────
    # We rebuild the loader (and respawn its workers) at the start of every
    # epoch. This forces a clean release of accumulated worker buffers and
    # PyTorch caching-allocator state, preventing the slow leak we saw
    # (~11 GB → ~30 GB RAM growth per epoch) and the resulting throughput drop.
    def build_loader():
        loader_kwargs = dict(
            data=data,
            num_neighbors=cfg.fanout,
            edge_label_index=(SUPERVISION_KEY, sup_ei),
            edge_label=sup_lab,
            neg_sampling=NegativeSampling(mode="binary", amount=cfg.neg_ratio),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
        )
        if cfg.num_workers > 0 and cfg.persistent_workers:
            loader_kwargs["persistent_workers"] = True
        return LinkNeighborLoader(**loader_kwargs)

    loader = build_loader()
    print(f"Loader: {len(loader)} batches/epoch  (batch_size={cfg.batch_size}, neg_ratio={cfg.neg_ratio})")

    # ── Loop ────────────────────────────────────────────────────────────────
    best_ndcg   = -1.0
    best_epoch  = -1
    patience    = 0
    epoch_time  = []

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0 = time.time()
        total_loss, n_batches = 0.0, 0

        pbar = tqdm(loader, desc=f"epoch {epoch} train", leave=False, dynamic_ncols=True)
        running = 0.0
        for i, batch in enumerate(pbar):
            batch = batch.to(device)
            optimizer.zero_grad()
            with amp_ctx(cfg.amp_dtype, device):
                logits = model(batch)
                # PyG's NegativeSampling(mode="binary") relabels:
                #   0 = sampled random negative
                #   1 = our explicit negative (input label 0, RATED_LOW)
                #   2 = our positive       (input label 1, RATED_HIGH ∪ READ_UNRATED)
                # BCE target = 1 iff the edge is a real positive.
                labels = (batch[SUPERVISION_KEY].edge_label == 2).float()
                loss   = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            l = loss.item()
            total_loss += l
            n_batches  += 1
            # Exponential moving average for a smoother running loss
            running = l if i == 0 else 0.97 * running + 0.03 * l
            if (i % 20) == 0:
                pbar.set_postfix(loss=f"{running:.4f}")
        pbar.close()

        train_loss = total_loss / max(n_batches, 1)
        train_t    = time.time() - t0

        # Val
        t0 = time.time()
        val_m = evaluate(model, data, cfg.splits_dir, split="val", cfg=cfg, device=device)
        val_t = time.time() - t0

        epoch_time.append(train_t + val_t)
        row = {
            "epoch":      epoch,
            "loss":       round(train_loss, 6),
            "ndcg10":     round(val_m[f"ndcg{cfg.eval_k}"], 6),
            "mrr":        round(val_m["mrr"], 6),
            "auc":        round(val_m["auc"], 6),
            "val_users":  val_m["n_users"],
            "train_sec":  round(train_t, 2),
            "val_sec":    round(val_t, 2),
        }
        logger.log(row)
        print(f"epoch {epoch:>3} | loss={train_loss:.4f}  "
              f"ndcg@10={val_m['ndcg'+str(cfg.eval_k)]:.4f}  "
              f"mrr={val_m['mrr']:.4f}  auc={val_m['auc']:.4f}  "
              f"({train_t:.1f}s train / {val_t:.1f}s val)")

        # Always save a "last.pt" recovery checkpoint, plus "best.pt" when val NDCG improves.
        last_ckpt = {
            "epoch":   epoch,
            "state":   model.state_dict(),
            "optim":   optimizer.state_dict(),
            "config":  cfg.__dict__,
            "metrics": val_m,
        }
        torch.save(last_ckpt, run_dir / "last.pt")

        if val_m[f"ndcg{cfg.eval_k}"] > best_ndcg:
            best_ndcg  = val_m[f"ndcg{cfg.eval_k}"]
            best_epoch = epoch
            patience   = 0
            torch.save(last_ckpt, run_dir / "best.pt")
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"Early stop (no val ndcg improvement for {cfg.early_stop_patience} epochs)")
                break

        # Hard reset of leaky state between epochs: tear down loader workers,
        # force GC, drop PyTorch's caching allocator. Costs ~30s of worker
        # re-spin-up next epoch; avoids the multi-GB memory growth that causes
        # the mid-run slowdown.
        del loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        loader = build_loader()

    (run_dir / "summary.json").write_text(json.dumps({
        "best_epoch":   best_epoch,
        "best_ndcg10":  best_ndcg,
        "mean_epoch_s": float(np.mean(epoch_time)) if epoch_time else 0.0,
    }, indent=2))
    print(f"\nBest val ndcg@10 = {best_ndcg:.4f} at epoch {best_epoch}")
    print(f"Best checkpoint at {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
