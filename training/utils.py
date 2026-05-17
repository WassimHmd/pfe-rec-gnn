"""Seeding, run-dir bootstrap, CSV logger."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Make torch / numpy / python deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_run_dir(cfg: Any, root: Path | str = "training/runs") -> Path:
    """Create runs/<timestamp>/ with a config.json snapshot."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = root / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg_dict = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
    (run_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2, default=str))
    return run_dir


class CsvLogger:
    """Append-only CSV writer with auto-header on first row."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._fields: list[str] | None = None

    def log(self, row: dict) -> None:
        write_header = not self.path.exists()
        if self._fields is None:
            self._fields = list(row.keys())
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._fields)
            if write_header:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in self._fields})


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enable_perf_settings() -> None:
    """TF32 for fp32 matmuls on Ampere+. Free 10-30% speedup."""
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def amp_ctx(dtype_name: str, device: torch.device):
    """Return an autocast context manager (or a no-op CM if disabled)."""
    import contextlib
    if dtype_name == "none" or device.type != "cuda":
        return contextlib.nullcontext()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    return torch.amp.autocast(device_type="cuda", dtype=dtype)
