"""Beam cross-section stiffness from a density field (v0 structural model of the arm).

The arm is a cantilever beam along x; its cross-section lives in the plane (y, z), z vertical (thrust,
downwash and build axis). For a section made of a material of modulus E(rho_bar) per cell (SIMP), the beam
stiffnesses are
    EA   = sum_e E_e A_e
    EI_y = sum_e E_e [ (z_e - z_c)^2 A_e + A_e hz^2 / 12 ]     (vertical bending, thrust)
    EI_z = sum_e E_e [ (y_e - y_c)^2 A_e + A_e hy^2 / 12 ]     (lateral bending)
with (y_c, z_c) the modulus-weighted centroid. Three warping problems share ONE operator, the G-weighted
Laplacian K(G) of the section (`StructuredFEOperator`, one unknown per node), and one factorisation:

* Saint-Venant torsion: K omega = b_t,  b_t = int G (z, -y) . grad(N);   GJ = int G (y^2 + z^2) - b_t . omega
* shear (flexure) under a unit vertical shear force: the axial stress gradient E (z - z_c) / EI_y must be
  balanced by shear stresses tau = G grad(psi):
      K psi_z = f_z,  f_z = int E (z - z_c) / EI_y N;      S_z = 1 / (kappa G A)_z = f_z . psi_z
  and the same with y for a lateral shear force.

Why the shear terms matter: Euler-Bernoulli bending alone rates two DISCONNECTED pieces as one beam (parallel
axis theorem), and an aero-structural optimiser happily splits the section into a nose and a tail. Shear
warping is only solvable if every connected piece is in balance (int_piece E (z - z_c) = 0); a piece that is not
has to push its shear through the near-void (G ~ eps), so S explodes. Torsion likewise rewards closed cells
(tubes). Cantilever tip deflection (Timoshenko): delta = T L^3 / (3 EI) + T L S.

Bending of a section with I_yz != 0 couples the two planes; the v0 designs are mirror-symmetric in y, so
I_yz = 0 and is not computed. Everything is differentiable (`solve_spd` with 3 right-hand sides).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from aeroptimizer.fem import elements as el
from aeroptimizer.fem.adjoint import solve_spd
from aeroptimizer.fem.operators import StructuredFEOperator
from aeroptimizer.fem.solvers import LinearSolver, ScipyDirect
from aeroptimizer.grid import Connectivity, Grid


@dataclass
class SectionStiffness:
    EA: torch.Tensor
    EI_y: torch.Tensor  # about the horizontal axis y: vertical bending
    EI_z: torch.Tensor  # about the vertical axis z: lateral bending
    GJ: torch.Tensor
    S_z: torch.Tensor  # shear flexibility 1 / (kappa G A) for a vertical shear force
    S_y: torch.Tensor  # same for a lateral shear force
    z_c: torch.Tensor
    warping: torch.Tensor | None = None  # nodal (omega, psi_z, psi_y), shape (n_nodes, 3), for plots


class BeamSection:
    """Section grid (ny, nz) of cell size (hy, hz), y in [-W/2, W/2], z in [0, H]. Axis 0 = y, axis 1 = z."""

    def __init__(self, shape: tuple[int, int], h: tuple[float, float], solver: LinearSolver | None = None) -> None:
        self.grid = Grid(shape, h)
        ny, nz = shape
        hy, hz = h
        y = (torch.arange(ny) + 0.5) * hy - ny * hy / 2
        z = (torch.arange(nz) + 0.5) * hz
        self.y, self.z = (t.reshape(-1) for t in torch.meshgrid(y, z, indexing="ij"))  # element centres
        self.cell_area = hy * hz
        # warping problems: Laplace operator, one node pinned (warping is defined up to a constant)
        fixed = torch.zeros(self.grid.n_nodes, dtype=torch.bool)
        fixed[(ny // 2) * (nz + 1) + nz // 2] = True
        self.op = StructuredFEOperator(self.grid, Connectivity(self.grid, 1), el.laplace_ke0(2, h), 1, fixed)
        # element integrals at the 4 Gauss points (exact: integrands of degree <= 2 per axis)
        pts, _ = el.gauss_points(2)
        grads = el.shape_grads(2, h)  # (G, 4, 2)
        N = el.shape_values(2)  # (G, 4)
        yg = self.y[:, None] + pts[None, :, 0] * hy / 2  # (Ne, G)
        zg = self.z[:, None] + pts[None, :, 1] * hz / 2
        wdet = hy * hz / 4
        self.bt0 = torch.einsum("gak,egk->ea", grads, torch.stack([zg, -yg], dim=-1)) * wdet  # int grad N . (z,-y)
        self.N1 = N.sum(0) * wdet  # int N_a, (4,)
        self.Ny = torch.einsum("ga,eg->ea", N, yg) * wdet  # int N_a y, (Ne, 4)
        self.Nz = torch.einsum("ga,eg->ea", N, zg) * wdet  # int N_a z
        self.r2 = self.y**2 + self.z**2 + (hy**2 + hz**2) / 12  # cell mean of y^2 + z^2
        self.solver = solver or ScipyDirect()

    def _assemble(self, fe: torch.Tensor) -> torch.Tensor:
        """Element vectors (Ne, 4) -> nodal vector (n_nodes,)."""
        return torch.zeros(self.grid.n_nodes, dtype=fe.dtype).index_add(0, self.op.edof.reshape(-1), fe.reshape(-1))

    def stiffness(self, E: torch.Tensor, G: torch.Tensor) -> SectionStiffness:
        """E, G: per-cell moduli (Ne,) (already penalised, e.g. SIMP). Differentiable in both."""
        A, (hy, hz) = self.cell_area, self.grid.h
        EA = (E * A).sum()
        y_c = (E * self.y).sum() * A / EA
        z_c = (E * self.z).sum() * A / EA
        EI_y = (E * ((self.z - z_c) ** 2 + hz**2 / 12)).sum() * A
        EI_z = (E * ((self.y - y_c) ** 2 + hy**2 / 12)).sum() * A
        b_t = self._assemble(G[:, None] * self.bt0)
        f_z = self._assemble(E[:, None] * (self.Nz - z_c * self.N1)) / EI_y
        f_y = self._assemble(E[:, None] * (self.Ny - y_c * self.N1)) / EI_z
        rhs = torch.stack([b_t, f_z, f_y], dim=1)
        w = solve_spd(self.op, G, rhs, self.solver)  # (n_nodes, 3), one factorisation
        GJ = (G * self.r2).sum() * A - torch.dot(b_t, w[:, 0])
        return SectionStiffness(EA=EA, EI_y=EI_y, EI_z=EI_z, GJ=GJ, S_z=torch.dot(f_z, w[:, 1]),
                                S_y=torch.dot(f_y, w[:, 2]), z_c=z_c, warping=w)


@dataclass(frozen=True)
class ArmLoads:
    """Static load cases of a cantilever arm of length L (tip loads), in consistent units (N, mm, MPa).

    thrust:  vertical tip force (motor thrust x load factor)  -> C_T = T^2 (L^3 / (3 EI_y) + L S_z)
    lateral: horizontal tip force (yaw manoeuvre, gust, hit)   -> C_L = F^2 (L^3 / (3 EI_z) + L S_y)
    torque:  torsion moment about the arm axis (thrust offset) -> C_t = M^2 L / GJ
    Each C is the compliance f.u of the (Timoshenko) beam for that load; their sum is the usual multi-load
    compliance.
    """

    L: float = 120.0
    thrust: float = 8.0
    lateral: float = 3.0
    torque: float = 40.0
    shear: bool = True  # False = Euler-Bernoulli bending only (to show what the optimiser does without it)

    def compliances(self, s: SectionStiffness) -> dict[str, torch.Tensor]:
        L, k = self.L, 1.0 if self.shear else 0.0
        return {
            "C_T": self.thrust**2 * (L**3 / (3.0 * s.EI_y) + k * L * s.S_z),
            "C_L": self.lateral**2 * (L**3 / (3.0 * s.EI_z) + k * L * s.S_y),
            "C_t": self.torque**2 * L / s.GJ,
        }
