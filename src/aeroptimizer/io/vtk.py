"""VTK export of rho_bar, u, phi, v (C-IO-02). pyvista/meshio imported lazily: no hard dependency."""

from __future__ import annotations

from pathlib import Path


def export_vti(path: Path, grid, fields: dict) -> None:  # noqa: ANN001
    raise NotImplementedError("phase 2")
