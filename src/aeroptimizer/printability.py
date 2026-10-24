"""Voxel printability metrics, independent of the overhang filter, so they judge a design made with or without it.

Build axis = LAST axis (z), layer 0 lies on the build plate. A solid voxel of layer k >= 1 is an OVERHANG when
no solid voxel of layer k-1 lies in its supporting stencil:

* "cross" (default): the cell under it and its 4 lateral neighbours, the 45 deg rule of `topopt/overhang.py`;
* "square": the 3 x 3 block under it, which also accepts diagonal steps (~54.7 deg), a lenient variant.

A slicer would add a support column under every overhang voxel, down to the plate or to the part:
`support_volume` counts those empty voxels. NumPy only, inputs are boolean arrays (e.g. rho_bar > 0.5).
"""

from __future__ import annotations

import numpy as np

from aeroptimizer.errors import ConfigError


def _supported_from_below(below: np.ndarray, stencil: str) -> np.ndarray:
    """(lateral shape) bool: some solid voxel of the layer `below` lies in the stencil of each cell."""
    p = np.pad(below, 1)
    sl = [slice(1, n + 1) for n in below.shape]
    if stencil == "square":
        out = np.zeros_like(below)
        for off in np.ndindex(*(3,) * below.ndim):
            out |= p[tuple(slice(o, o + n) for o, n in zip(off, below.shape))]
        return out
    if stencil != "cross":
        raise ConfigError(f"stencil must be 'cross' or 'square', got {stencil!r}")
    out = below.copy()
    for ax in range(below.ndim):
        for o in (0, 2):
            idx = list(sl)
            idx[ax] = slice(o, o + below.shape[ax])
            out |= p[tuple(idx)]
    return out


def overhang_mask(solid: np.ndarray, stencil: str = "cross") -> np.ndarray:
    """Bool array (shape of `solid`): the solid voxels that rest on nothing in the layer below."""
    solid = np.asarray(solid, dtype=bool)
    out = np.zeros_like(solid)
    for k in range(1, solid.shape[-1]):
        out[..., k] = solid[..., k] & ~_supported_from_below(solid[..., k - 1], stencil)
    return out


def overhang_rate(solid: np.ndarray, ignore: np.ndarray | None = None, stencil: str = "cross") -> float:
    """Fraction of the solid voxels above the plate that are overhangs. `ignore` (e.g. the passive solid zones)
    is left out of both counts."""
    solid = np.asarray(solid, dtype=bool)
    keep = np.ones_like(solid) if ignore is None else ~np.asarray(ignore, dtype=bool)
    keep[..., 0] = False  # the plate supports layer 0
    n = int((solid & keep).sum())
    return float((overhang_mask(solid, stencil) & keep).sum() / n) if n else 0.0


def support_mask(solid: np.ndarray, stencil: str = "cross") -> np.ndarray:
    """Bool array: the empty voxels a slicer fills with support, i.e. the column under each overhang voxel
    down to the plate or to the first solid voxel."""
    solid = np.asarray(solid, dtype=bool)
    hang = overhang_mask(solid, stencil)
    out = np.zeros_like(solid)
    carrying = np.zeros(solid.shape[:-1], dtype=bool)
    for k in range(solid.shape[-1] - 1, -1, -1):  # from the top down
        out[..., k] = carrying & ~solid[..., k]
        carrying = np.where(solid[..., k], hang[..., k], carrying)
    return out


def support_volume(solid: np.ndarray, voxel_volume: float = 1.0, stencil: str = "cross") -> float:
    """Volume of support material (number of support voxels times `voxel_volume`)."""
    return float(support_mask(solid, stencil).sum()) * voxel_volume
