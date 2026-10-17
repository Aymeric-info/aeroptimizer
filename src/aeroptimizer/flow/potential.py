"""Penalised potential flow  -div(kappa(rho_bar) grad phi) = 0  (FORMALISM §4.1-4.3, lesson 14).

Reuses StructuredFEOperator with ncomp=1 and Ke0 = laplace_ke0 (C-FLOW-09): no dedicated assembler.
BCs: flux inlet on z = z_max (rotor slipstream profile), phi = 0 on z = 0, slip on lateral walls
(natural condition: nothing to impose).
Requires a passive fluid pad of >= n_pad elements at the inlet (C-FLOW-05).
"""

from __future__ import annotations

import itertools
import math

import torch

from aeroptimizer.errors import ConfigError
from aeroptimizer.fem import elements as el
from aeroptimizer.fem.adjoint import solve_spd
from aeroptimizer.fem.operators import StructuredFEOperator
from aeroptimizer.fem.solvers import LinearSolver, ScipyDirect
from aeroptimizer.flow.aero import bernoulli_pressure
from aeroptimizer.flow.base import FlowState
from aeroptimizer.grid import Connectivity, Grid


def kappa(rho_bar: torch.Tensor, kappa_min: float = 1e-6, q: float = 3.0) -> torch.Tensor:
    """Conductivity kappa = kappa_min + (1 - kappa_min)(1 - rho_bar)^q: 1 in fluid, kappa_min in solid (C-FLOW-06)."""
    return kappa_min + (1.0 - kappa_min) * (1.0 - rho_bar) ** q


def inlet_load(grid: Grid, speed: torch.Tensor) -> torch.Tensor:
    """(n_nodes,) right-hand side b_a = -int_{z=z_max} U_in N_a dGamma for an inflow `speed` >= 0.

    `speed` is per face element, shape `grid.shape[:-1]`. The outward normal on the inlet is +e_z, so the
    imposed flux is kappa d_n phi = -U_in: v = grad phi points into the domain (towards -z, C-CONV-07).
    A face element hands U * |face| / 2**(d-1) to each of its corner nodes (exact for constant U).
    """
    face_shape = grid.shape[:-1]
    if tuple(speed.shape) != tuple(face_shape):
        raise ConfigError(f"inlet speed must have shape {tuple(face_shape)}, got {tuple(speed.shape)}")
    if bool((speed < 0).any()):
        raise ConfigError("inlet speed must be >= 0 (C-CONV-07: U_in is the INWARD speed)")
    face_area = math.prod(grid.h[:-1])
    nodal = torch.zeros(tuple(n + 1 for n in face_shape), dtype=speed.dtype)
    for bits in itertools.product((0, 1), repeat=len(face_shape)):
        sl = tuple(slice(b, b + n) for b, n in zip(bits, face_shape))
        nodal[sl] += speed
    nodes = torch.zeros(*(n + 1 for n in grid.shape), dtype=speed.dtype)
    nodes[..., -1] = -nodal * face_area / 2 ** len(face_shape)
    return nodes.reshape(-1)


def slipstream_inlet_profile(grid, rotor_mask: torch.Tensor, thrust_N: float, rotor_area_m2: float,  # noqa: ANN001
                             rho_air: float, k_w: float, U_inf: float) -> torch.Tensor:
    """U_in = k_w * U_h on the rotor footprint, U_inf elsewhere; U_h = sqrt(T / (2 rho_a A))  (§4.2)."""
    if rotor_mask.shape != tuple(grid.shape[:-1]) or rotor_mask.dtype != torch.bool:
        raise ConfigError(f"rotor_mask must be a bool tensor of shape {tuple(grid.shape[:-1])}")
    if not (thrust_N > 0 and rotor_area_m2 > 0 and rho_air > 0):
        raise ConfigError("thrust, rotor area and air density must be > 0")
    u_h = math.sqrt(thrust_N / (2.0 * rho_air * rotor_area_m2))
    dtype = torch.get_default_dtype()
    return torch.where(rotor_mask, torch.tensor(k_w * u_h, dtype=dtype), torch.tensor(U_inf, dtype=dtype))


class PotentialFlow:
    """FlowModel (C-FLOW-01): solve(rho_bar) -> FlowState(phi, v, p_rel). Differentiable in rho_bar.

    `rho_bar` may be flat (n_elems,) or of the grid shape. `u_ref` defaults to the largest inlet speed.
    `n_pad` is the inlet pad of C-FLOW-05 (0 disables the check: lesson 14 shows why it exists).
    """

    def __init__(
        self,
        grid: Grid,
        inlet_profile: torch.Tensor,
        *,
        solver: LinearSolver | None = None,
        kappa_min: float = 1e-6,
        q: float = 3.0,
        rho_air: float = 1.225,
        u_ref: float | None = None,
        n_pad: int = 4,
    ) -> None:
        if not (0.0 < kappa_min < 1.0) or q <= 0.0:
            raise ConfigError("need 0 < kappa_min < 1 and q > 0")
        self.grid, self.solver = grid, solver or ScipyDirect()
        self.kappa_min, self.q, self.rho_air, self.n_pad = kappa_min, q, rho_air, n_pad
        self.u_ref = float(inlet_profile.max()) if u_ref is None else u_ref
        if not self.u_ref > 0:
            raise ConfigError("u_ref must be > 0 (all-zero inlet?)")
        dtype = inlet_profile.dtype
        self.b = inlet_load(grid, inlet_profile)
        nodes_z = torch.arange(grid.n_nodes) % (grid.shape[-1] + 1)  # last axis fastest (C-CONV-04)
        self.outlet = nodes_z == 0
        self.inlet = nodes_z == grid.shape[-1]
        conn = Connectivity(grid, 1)
        ke0 = el.laplace_ke0(grid.d, grid.h, dtype)
        self.op = StructuredFEOperator(grid, conn, ke0, 1, self.outlet)
        self.op_neumann = StructuredFEOperator(grid, conn, ke0, 1, torch.zeros_like(self.outlet))
        self._grad_c = el.center_gradients(grid.d, grid.h, dtype)  # (2**d, d)

    def conductivity(self, rho_bar: torch.Tensor) -> torch.Tensor:
        return kappa(rho_bar.reshape(-1), self.kappa_min, self.q)

    def check_inlet_pad(self, rho_bar: torch.Tensor, tol: float = 1e-3) -> None:
        """C-FLOW-05: solid in the first n_pad layers under the inlet face -> ConfigError."""
        if self.n_pad > 0:
            pad = rho_bar.detach().reshape(self.grid.shape)[..., -self.n_pad :]
            if float(pad.max()) > tol:
                raise ConfigError(
                    f"solid in the {self.n_pad} elements below the inlet (max rho_bar={float(pad.max()):.3g}): "
                    "a passive fluid pad is required (C-FLOW-05)"
                )

    def solve(self, rho_bar: torch.Tensor) -> FlowState:
        self.check_inlet_pad(rho_bar)
        w = self.conductivity(rho_bar)
        phi = solve_spd(self.op, w, self.b, self.solver)
        v = phi[self.op.edof] @ self._grad_c  # (n_elems, d) velocity at the element centres
        p_rel = bernoulli_pressure(v, self.rho_air, self.u_ref)
        return FlowState(phi=phi, v=v, p_rel=p_rel)

    def fluxes(self, rho_bar: torch.Tensor, phi: torch.Tensor | None = None) -> tuple[float, float]:
        """(flux in, flux out): sums of the reactions K phi - b on the inlet and outlet nodes (C-FLOW-03).

        K here is the matrix WITHOUT Dirichlet condition; at the outlet K phi is the flux leaving the
        domain, and at the inlet K phi - b is the residual that must vanish.
        """
        w = self.conductivity(rho_bar).detach()
        if phi is None:
            phi = self.solve(rho_bar).phi
        r = self.op_neumann.matvec(w, phi.detach())
        flux_in = -float(self.b[self.inlet].sum())  # = int U_in dGamma > 0
        flux_out = float(r[self.outlet].sum())
        return flux_in, flux_out
