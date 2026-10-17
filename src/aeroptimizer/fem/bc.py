"""Boundary conditions and load cases (FORMALISM §3.4-3.5, CONTRACTS §3.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from aeroptimizer.errors import BoundaryConditionError, ConfigError
from aeroptimizer.grid import Grid, Masks


def node_indices(grid: Grid) -> torch.Tensor:
    """(n_nodes, d) integer multi-index of every node, in the C order of the global numbering."""
    axes = [torch.arange(n + 1) for n in grid.shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, grid.d)


def node_mask_where(grid: Grid, axis: int, index: int) -> torch.Tensor:
    """(n_nodes,) bool: the nodes of the face `axis = index` (index 0 or grid.shape[axis])."""
    return node_indices(grid)[:, axis] == index


def rigid_modes(grid: Grid) -> torch.Tensor:
    """(n_nodes * d, n_rigid) rigid-body fields: d translations plus 1 (2D) or 3 (3D) rotations."""
    d, n = grid.d, grid.n_nodes
    h = torch.tensor(grid.h, dtype=torch.get_default_dtype())
    xyz = node_indices(grid).to(h.dtype) * h  # (n, d) coordinates
    modes = []
    for c in range(d):  # translations
        m = torch.zeros(n, d, dtype=h.dtype)
        m[:, c] = 1.0
        modes.append(m)
    pairs = [(0, 1)] if d == 2 else [(0, 1), (1, 2), (2, 0)]
    for a, b in pairs:  # rotation in the (a, b) plane: u_a = -x_b, u_b = x_a
        m = torch.zeros(n, d, dtype=h.dtype)
        m[:, a], m[:, b] = -xyz[:, b], xyz[:, a]
        modes.append(m)
    return torch.stack([m.reshape(-1) for m in modes], dim=1)


def check_rigid_modes(grid: Grid, fixed_mask: torch.Tensor) -> None:
    """Raise BoundaryConditionError if some rigid-body mode vanishes on every fixed dof (C-BC-01).

    The fixed dofs prevent all rigid motions iff the rigid modes restricted to them are linearly
    independent (rank 3 in 2D, 6 in 3D); counting fixed dofs is necessary but not sufficient.
    """
    R = rigid_modes(grid)
    rank = int(torch.linalg.matrix_rank(R[fixed_mask])) if bool(fixed_mask.any()) else 0
    if rank < R.shape[1]:
        raise BoundaryConditionError(
            f"under-constrained: the fixed dofs block {rank} of {R.shape[1]} rigid-body modes"
        )


@dataclass(frozen=True)
class Dirichlet:
    """Homogeneous Dirichlet on selected nodes/components -> fixed_mask (N,) bool. C-BC-01 checks rigid modes."""

    node_mask: torch.Tensor  # (n_nodes,) bool
    components: tuple[int, ...]

    def fixed_mask(self, grid: Grid, ncomp: int, check: bool = True) -> torch.Tensor:
        """(N,) bool, dof = ncomp * node + component. The rigid-mode check applies to mechanics
        (ncomp == d); pass `check=False` for one piece of a union, and use `combine` instead."""
        if self.node_mask.shape != (grid.n_nodes,) or self.node_mask.dtype != torch.bool:
            raise ConfigError(f"node_mask must be a bool tensor of shape ({grid.n_nodes},)")
        if not self.components or any(not 0 <= c < ncomp for c in self.components):
            raise ConfigError(f"components {self.components} not valid for ncomp={ncomp}")
        selected = torch.zeros(ncomp, dtype=torch.bool, device=self.node_mask.device)
        selected[list(self.components)] = True
        mask = self.node_mask[:, None] & selected[None, :]  # (n_nodes, ncomp)
        mask = mask.reshape(-1)
        if check and ncomp == grid.d:
            check_rigid_modes(grid, mask)
        return mask


def combine(grid: Grid, ncomp: int, *dirichlets: Dirichlet) -> torch.Tensor:
    """Union of several Dirichlet conditions, with the rigid-mode check applied to the union."""
    mask = torch.zeros(grid.n_dofs(ncomp), dtype=torch.bool)
    for dc in dirichlets:
        mask |= dc.fixed_mask(grid, ncomp, check=False)
    if ncomp == grid.d:
        check_rigid_modes(grid, mask)
    return mask


class LoadCase(Protocol):
    """f = assemble(rho_bar): signature accepts rho_bar so design-dependent loads are possible (C-BC-02)."""

    def assemble(self, rho_bar: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class DistributedForce:
    """Total force `value` (vector, N) spread uniformly over the nodes of `node_mask` (C-BC-04)."""

    node_mask: torch.Tensor
    value: tuple[float, ...]
    grid: Grid

    def assemble(self, rho_bar: torch.Tensor) -> torch.Tensor:
        """Constant load (v1): `rho_bar` is accepted for the LoadCase signature and ignored."""
        d = self.grid.d
        if len(self.value) != d or self.node_mask.shape != (self.grid.n_nodes,):
            raise ConfigError(f"value needs {d} components and node_mask {self.grid.n_nodes} nodes")
        n_loaded = int(self.node_mask.sum())
        if n_loaded == 0:
            raise BoundaryConditionError("load applied to an empty set of nodes")
        f = torch.zeros(self.grid.n_nodes, d, dtype=torch.get_default_dtype())
        f[self.node_mask] = torch.tensor(self.value, dtype=f.dtype) / n_loaded
        return f.reshape(-1)


def build_case(name: str, grid: Grid):  # noqa: ANN201
    """Registry of named cases: 'mbb', 'cantilever', 'drone_arm' -> (fixed_mask, LoadCase, Masks).

    Axis 0 is x, the last axis is z (up). Unit total load, no passive zones (all elements design).
    - 'mbb' (2D): half MBB beam. Symmetry plane x = 0 (u_x = 0), roller under the right end
      (u_z = 0 at one node), point load -1 on the top-left node.
    - 'cantilever' (2D or 3D): face x = 0 clamped, load -1 along z spread over the face x = L.
    """
    d = grid.d
    nx, nz = grid.shape[0], grid.shape[-1]
    nodes = node_indices(grid)
    if name == "mbb":
        if d != 2:
            raise ConfigError("the 'mbb' case is 2D only")
        sym = Dirichlet(nodes[:, 0] == 0, (0,))
        roller = Dirichlet((nodes[:, 0] == nx) & (nodes[:, 1] == 0), (1,))
        fixed = combine(grid, 2, sym, roller)
        load = DistributedForce((nodes[:, 0] == 0) & (nodes[:, 1] == nz), (0.0, -1.0), grid)
    elif name == "cantilever":
        fixed = combine(grid, d, Dirichlet(nodes[:, 0] == 0, tuple(range(d))))
        load = DistributedForce(nodes[:, 0] == nx, (*([0.0] * (d - 1)), -1.0), grid)
    else:
        raise ConfigError(f"unknown case {name!r} ('drone_arm': phase 6)")
    all_design = torch.ones(grid.shape, dtype=torch.bool)
    none = torch.zeros(grid.shape, dtype=torch.bool)
    return fixed, load, Masks(design=all_design, solid=none, void=none)
