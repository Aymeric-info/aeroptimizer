"""Structured grid, connectivity and passive-region masks (FORMALISM §1, CONTRACTS §3.1).

Conventions (C-CONV-01/04): axes are (x, z) in 2D and (x, y, z) in 3D; z is the LAST axis and is the
build direction. Row-major (C) ordering, last axis fastest. Local node a has bits (b_0..b_{d-1}),
b_0 most significant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from aeroptimizer.errors import ConfigError


@dataclass(frozen=True)
class Grid:
    shape: tuple[int, ...]  # elements per axis, e.g. (nx, nz) or (nx, ny, nz)
    h: tuple[float, ...]  # element size per axis [m]

    def __post_init__(self) -> None:  # C-GRID-01
        if len(self.shape) not in (2, 3) or len(self.h) != len(self.shape):
            raise ConfigError(f"shape/h must both have length 2 or 3, got {self.shape}, {self.h}")
        if any((not isinstance(n, int)) or n < 1 for n in self.shape):
            raise ConfigError(f"shape must be positive integers, got {self.shape}")
        if any(not (hh > 0.0) for hh in self.h):
            raise ConfigError(f"h must be > 0, got {self.h}")

    @property
    def d(self) -> int:
        return len(self.shape)

    @property
    def build_axis(self) -> int:
        return self.d - 1

    @property
    def n_elems(self) -> int:  # C-GRID-02
        return math.prod(self.shape)

    @property
    def n_nodes(self) -> int:
        return math.prod(n + 1 for n in self.shape)

    @property
    def element_volume(self) -> float:
        return math.prod(self.h)

    def n_dofs(self, ncomp: int) -> int:
        return ncomp * self.n_nodes


class Connectivity:
    """Topological plan, independent of density (C-GRID-03). Built without Python loops over elements."""

    edof: torch.Tensor  # (n_elems, ncomp * 2**d) int64

    def __init__(self, grid: Grid, ncomp: int, device: torch.device | str = "cpu") -> None:
        if ncomp < 1:
            raise ConfigError(f"ncomp must be >= 1, got {ncomp}")
        d = grid.d
        node_shape = tuple(n + 1 for n in grid.shape)
        strides = [math.prod(node_shape[k + 1 :]) for k in range(d)]  # C order, last axis fastest
        idx = torch.stack(
            torch.meshgrid(*[torch.arange(n) for n in grid.shape], indexing="ij"), dim=-1
        ).reshape(-1, d)  # (Ne, d) multi-index of each element, itself in C order
        a = torch.arange(2**d)
        bits = torch.stack([(a >> (d - 1 - k)) & 1 for k in range(d)], dim=1)  # (2**d, d), b_0 = MSB
        stride = torch.tensor(strides)
        nodes = ((idx[:, None, :] + bits[None, :, :]) * stride).sum(-1)  # (Ne, 2**d)
        dofs = ncomp * nodes[:, :, None] + torch.arange(ncomp)  # (Ne, 2**d, ncomp)
        self.grid = grid
        self.ncomp = ncomp
        self.edof = dofs.reshape(grid.n_elems, ncomp * 2**d).to(device)


@dataclass(frozen=True)
class Masks:
    """Boolean element masks on the grid shape. D: design, S: passive solid, V: passive void/fluid."""

    design: torch.Tensor
    solid: torch.Tensor
    void: torch.Tensor

    def __post_init__(self) -> None:  # C-GRID-05
        if not (self.design.shape == self.solid.shape == self.void.shape):
            raise ConfigError("masks must share the grid shape")
        cover = self.design.long() + self.solid.long() + self.void.long()
        if not bool((cover == 1).all()):
            raise ConfigError("masks D, S, V must partition the grid (disjoint, union = grid)")
