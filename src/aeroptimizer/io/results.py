"""Run directory: config.resolved.yaml, history.csv, rho_bar.npy, provenance.json (C-IO-01, C-IO-03)."""

from __future__ import annotations

from pathlib import Path


def make_run_dir(root: Path, case: str) -> Path:
    raise NotImplementedError("phase 2")


def write_provenance(run_dir: Path) -> None:
    """git rev-parse HEAD (or 'nogit'), torch/scipy/numpy versions, device, dtype."""
    raise NotImplementedError("phase 2")
