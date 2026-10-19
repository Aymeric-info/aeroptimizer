"""Steady incompressible Navier-Stokes with Brinkman penalisation on a 2D tensor-product grid (v0).

Why: potential flow gives zero drag (d'Alembert). The fluid topology-optimisation literature instead solves a
viscous flow on the whole box and turns the solid into a nearly impermeable porous medium,
    rho (u . grad) u = -grad p + mu lap u - alpha(rho_bar) u,   div u = 0,
with alpha ~ 0 in the fluid and alpha = alpha_max in the solid (Borrvall & Petersson 2003). The force of the
fluid on the body is then the volume integral of the Brinkman friction, F = int alpha u dOmega (Kondoh et al. 2012).

Discretisation: Q1 velocity / Q1 pressure (equal order), stabilised by SUPG + PSPG (Tezduyar), one unknown
triple (u_0, u_1, p) per node, dof = 3 * node + c. The grid is rectilinear but not uniform (`axes` are the
node coordinates per axis): fine around the body, stretched towards the far field. Non-dimensional: density 1,
reference velocity 1, viscosity mu = 1 / Re.

The element residual is written once, for one element, with batch-agnostic `...` einsums. The global
residual calls it on all elements at once; the element Jacobians come from `torch.func.vmap(jacfwd(.))`, so
Newton's Jacobian is exact (stabilisation parameters included). The steady solve is wrapped in a
`torch.autograd.Function` whose backward is the discrete adjoint: one transposed solve with the Newton
Jacobian, then a vector-Jacobian product of the residual in alpha. This is the only hand-written backward of
this module, in the spirit of `fem/adjoint.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch.func import jacfwd, vmap

from aeroptimizer.errors import ConfigError, SolverConvergenceError
from aeroptimizer.fem import elements as el
from aeroptimizer.grid import Connectivity, Grid


def stretched_axis(core: tuple[float, float], h: float, lo: float, hi: float, ratio: float = 1.15) -> np.ndarray:
    """Node coordinates: uniform step `h` on `core`, then cells growing by `ratio` up to `lo` and `hi`.

    The last cell on each side is cut so that the axis ends exactly at `lo` / `hi`. `core` ends must be
    multiples of `h` away from each other (the uniform part is where the design lives).
    """
    a, b = core
    n_core = round((b - a) / h)
    if abs(n_core * h - (b - a)) > 1e-9 * max(1.0, abs(b - a)) or not lo <= a < b <= hi:
        raise ConfigError(f"core {core} must be a multiple of h={h} inside [{lo}, {hi}]")
    nodes = list(a + h * np.arange(n_core + 1))
    for side, end in ((-1, lo), (1, hi)):
        x, step = (a if side < 0 else b), h
        out = []
        while True:
            step *= ratio
            nxt = x + side * step
            if (side < 0 and nxt <= end + 0.3 * step) or (side > 0 and nxt >= end - 0.3 * step):
                if x != end:
                    out.append(end)
                break
            out.append(nxt)
            x = nxt
        nodes = (out[::-1] + nodes) if side < 0 else (nodes + out)
    return np.array(nodes)


def brinkman_alpha(rho_bar: torch.Tensor, alpha_max: float, q: float = 0.1) -> torch.Tensor:
    """alpha = alpha_max * q * rho / (q + 1 - rho)  (Borrvall-Petersson in the SOLID density rho = 1 - gamma).

    0 in the fluid, alpha_max in the solid, convex: an intermediate density lets flow through and is not
    worth much, which pushes towards 0/1. Smaller q = more convex.
    """
    return alpha_max * q * rho_bar / (q + 1.0 - rho_bar)


@dataclass
class FlowSolution:
    U: torch.Tensor  # (3 * n_nodes,) nodal (u_0, u_1, p)
    newton_its: int
    residual: float

    def fields(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(u_0, u_1, p) as node arrays of shape (n0 + 1, n1 + 1)."""
        V = self.U.detach().numpy().reshape(shape[0] + 1, shape[1] + 1, 3)
        return V[..., 0], V[..., 1], V[..., 2]


class BrinkmanNS:
    """Stabilised Q1/Q1 Navier-Stokes-Brinkman solver on a rectilinear grid, with Newton and adjoint.

    Boundary conditions are given as a dof mask `fixed` (3 * n_nodes,) and the imposed values `values` for
    those dofs (velocity components or pressure). Faces not mentioned are traction-free ("do nothing"):
    the natural outlet condition. `solve(alpha)` is differentiable in alpha (per-element Brinkman
    coefficient) through the adjoint.
    """

    def __init__(self, axes: tuple[np.ndarray, np.ndarray], Re: float, fixed: torch.Tensor, values: torch.Tensor,
                 *, newton_tol: float = 1e-9, max_newton: int = 40) -> None:
        a0, a1 = (np.asarray(a, dtype=float) for a in axes)
        if np.any(np.diff(a0) <= 0) or np.any(np.diff(a1) <= 0):
            raise ConfigError("node coordinates must be strictly increasing")
        self.axes = (a0, a1)
        self.shape = (len(a0) - 1, len(a1) - 1)
        self.grid = Grid(self.shape, (1.0, 1.0))  # topology only: element sizes are per element below
        self.n_nodes = self.grid.n_nodes
        self.n_dofs = 3 * self.n_nodes
        if fixed.shape != (self.n_dofs,) or values.shape != (self.n_dofs,):
            raise ConfigError(f"fixed/values must have shape ({self.n_dofs},)")
        self.mu = 1.0 / Re
        self.Re = Re
        self.edof = Connectivity(self.grid, 3).edof  # (Ne, 12), dof = 3 * node + c
        h0 = torch.tensor(np.diff(a0))
        h1 = torch.tensor(np.diff(a1))
        H0, H1 = torch.meshgrid(h0, h1, indexing="ij")
        self.h0, self.h1 = H0.reshape(-1), H1.reshape(-1)  # (Ne,) element sizes
        self.area = self.h0 * self.h1
        c0 = torch.tensor(0.5 * (a0[1:] + a0[:-1]))
        c1 = torch.tensor(0.5 * (a1[1:] + a1[:-1]))
        C0, C1 = torch.meshgrid(c0, c1, indexing="ij")
        self.centres = torch.stack([C0.reshape(-1), C1.reshape(-1)], dim=-1)  # (Ne, 2)
        self.fixed = fixed.to(torch.bool)
        self.free = (~self.fixed).to(torch.get_default_dtype())
        self.values = values.to(torch.get_default_dtype())
        self.newton_tol, self.max_newton = newton_tol, max_newton
        # reference element: values and d/dxi of the 4 shape functions at the 4 Gauss points
        self._N = el.shape_values(2)  # (G, 4)
        self._dN = el.shape_grads(2, (2.0, 2.0))  # (G, 4, 2), d/dxi since J = diag(h/2) = I here
        rows = self.edof[:, :, None].expand(-1, -1, 12).reshape(-1).numpy()
        cols = self.edof[:, None, :].expand(-1, 12, -1).reshape(-1).numpy()
        self._rows, self._cols = rows, cols
        self._jac = vmap(jacfwd(self._element_residual, argnums=0))
        self.U_last: torch.Tensor | None = None  # warm start for the next solve
        self.last: FlowSolution | None = None

    # ------------------------------------------------------------------ element level
    def _element_residual(self, Ue: torch.Tensor, alpha: torch.Tensor, h0: torch.Tensor, h1: torch.Tensor):
        """Residual of one element (or a batch, leading dims `...`): (..., 12), ordering 3 * a + c."""
        N, dNr, mu = self._N, self._dN, self.mu
        U = Ue.reshape(*Ue.shape[:-1], 4, 3)
        vel, p = U[..., :2], U[..., 2]
        scale = torch.stack([2.0 / h0, 2.0 / h1], dim=-1)  # (..., 2)
        dN = dNr * scale[..., None, None, :]  # (..., G, 4, 2)
        wdet = (h0 * h1 / 4.0)[..., None]  # Gauss weights are 1
        u = torch.einsum("ga,...ac->...gc", N, vel)  # (..., G, 2)
        Gu = torch.einsum("...gaj,...ai->...gij", dN, vel)  # du_i/dx_j
        gp = torch.einsum("...gaj,...a->...gj", dN, p)
        pg = torch.einsum("ga,...a->...g", N, p)
        conv = torch.einsum("...gij,...gj->...gi", Gu, u)
        div = Gu[..., 0, 0] + Gu[..., 1, 1]
        uc = vel.mean(dim=-2)
        unorm = torch.sqrt((uc**2).sum(-1) + 1e-12)
        h = torch.sqrt(h0 * h1)
        tau = ((2.0 * unorm / h) ** 2 + (12.0 * mu / h**2) ** 2 + alpha**2) ** -0.5  # SUPG = PSPG parameter
        rM = conv + gp + alpha[..., None, None] * u  # strong momentum residual (no viscous term for Q1)
        udN = torch.einsum("...gc,...gac->...ga", u, dN)
        Rm = (
            torch.einsum("ga,...gi->...gai", N, conv + alpha[..., None, None] * u)
            + mu * torch.einsum("...gij,...gaj->...gai", Gu, dN)
            - pg[..., None, None] * dN
            + tau[..., None, None, None] * udN[..., None] * rM[..., :, None, :]
        )
        Rc = torch.einsum("ga,...g->...ga", N, div) + tau[..., None, None] * torch.einsum("...gaj,...gj->...ga", dN, rM)
        R = torch.cat([(wdet[..., None, None] * Rm).sum(-3), (wdet[..., None] * Rc).sum(-2)[..., None]], dim=-1)
        return R.reshape(*R.shape[:-2], 12)

    # ------------------------------------------------------------------ global level
    def residual(self, U: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Full residual (n_dofs,), Dirichlet rows replaced by U - values."""
        Re = self._element_residual(U[self.edof], alpha, self.h0, self.h1)
        R = torch.zeros(self.n_dofs, dtype=Re.dtype).index_add_(0, self.edof.reshape(-1), Re.reshape(-1))
        return R * self.free + (U - self.values) * (1.0 - self.free)

    def jacobian(self, U: torch.Tensor, alpha: torch.Tensor) -> sp.csc_matrix:
        """Exact Newton Jacobian of `residual` in U (CSC): element Jacobians by forward AD, then assembly."""
        with torch.no_grad():
            Je = self._jac(U[self.edof], alpha, self.h0, self.h1)  # (Ne, 12, 12)
        J = sp.csr_matrix((Je.reshape(-1).numpy(), (self._rows, self._cols)), shape=(self.n_dofs,) * 2)
        free = self.free.numpy()
        return (sp.diags(free) @ J + sp.diags(1.0 - free)).tocsc()

    def initial_guess(self) -> torch.Tensor:
        """Imposed values on the boundary, the inlet velocity copied everywhere else (uniform stream)."""
        U = torch.zeros(self.n_dofs)
        U[self.fixed] = self.values[self.fixed]
        V = U.reshape(-1, 3)
        fx = self.fixed.reshape(-1, 3)
        for c in (0, 1):  # fill free velocity dofs with the mean imposed value of that component
            imposed = V[fx[:, c], c]
            V[~fx[:, c], c] = imposed.mean() if imposed.numel() else 0.0
        return U

    def newton(self, alpha: torch.Tensor, U0: torch.Tensor | None = None) -> FlowSolution:
        """Damped Newton (backtracking on ||R||). Falls back to a Reynolds ramp if it fails from U0."""
        alpha = alpha.detach()
        start = self.initial_guess() if U0 is None else U0.detach().clone()
        try:
            return self._newton(alpha, start)
        except SolverConvergenceError:
            # continuation in the Reynolds number from the uniform stream (robust first solve)
            Re_target, U = self.Re, self.initial_guess()
            try:
                for Re in [r for r in (1.0, 5.0, 20.0, 50.0) if r < Re_target] + [Re_target]:
                    self.mu = 1.0 / Re
                    U = self._newton(alpha, U).U
            finally:
                self.mu = 1.0 / Re_target
            return self._newton(alpha, U)

    def _newton(self, alpha: torch.Tensor, U: torch.Tensor) -> FlowSolution:
        U = U.clone()
        U[self.fixed] = self.values[self.fixed]
        R = self.residual(U, alpha)
        r0 = float(torch.linalg.norm(R))
        r = r0
        for it in range(self.max_newton):
            if r <= self.newton_tol * max(1.0, float(torch.linalg.norm(self.values))):
                return FlowSolution(U, it, r)
            lu = spla.splu(self.jacobian(U, alpha), permc_spec="COLAMD")
            dU = torch.from_numpy(lu.solve(-R.numpy()))
            t = 1.0
            while True:
                U_try = U + t * dU
                R_try = self.residual(U_try, alpha)
                r_try = float(torch.linalg.norm(R_try))
                if r_try <= (1.0 - 1e-4 * t) * r or t < 1.0 / 64:
                    break
                t *= 0.5
            if not math.isfinite(r_try) or (t < 1.0 / 64 and r_try > r):
                raise SolverConvergenceError(f"Newton stalled at |R| = {r:.3e} (Re = {1 / self.mu:g})")
            U, R, r = U_try, R_try, r_try
        if r <= 1e3 * self.newton_tol * max(1.0, float(torch.linalg.norm(self.values))):
            return FlowSolution(U, self.max_newton, r)
        raise SolverConvergenceError(f"Newton: |R| = {r:.3e} after {self.max_newton} iterations")

    def solve(self, alpha: torch.Tensor) -> torch.Tensor:
        """Differentiable U(alpha): Newton forward (warm-started from the previous solve), adjoint backward."""
        return _SteadySolve.apply(alpha, self)

    # ------------------------------------------------------------------ outputs
    def force(self, U: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Force of the fluid ON the solid, F = int alpha u dOmega (Kondoh et al. 2012), shape (2,).

        Q1: the integral of u over a rectangle is its area times the mean of the 4 nodal values.
        """
        vel = U.reshape(-1, 3)[:, :2][self.edof[:, ::3] // 3]  # (Ne, 4, 2)
        return ((alpha * self.area)[:, None] * vel.mean(dim=1)).sum(dim=0)

    def dissipation(self, U: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Total dissipated power  int mu |grad u|^2 + alpha |u|^2  (Borrvall-Petersson objective)."""
        vel = U.reshape(-1, 3)[:, :2][self.edof[:, ::3] // 3]
        scale = torch.stack([2.0 / self.h0, 2.0 / self.h1], dim=-1)
        dN = self._dN * scale[:, None, None, :]
        Gu = torch.einsum("egaj,eai->egij", dN, vel)
        u = torch.einsum("ga,eac->egc", self._N, vel)
        wdet = (self.h0 * self.h1 / 4.0)[:, None]
        return (wdet * (self.mu * (Gu**2).sum((-1, -2)) + alpha[:, None] * (u**2).sum(-1))).sum()


class _SteadySolve(torch.autograd.Function):
    """U = solution of R(U, alpha) = 0. Backward: J^T lam = g_U, grad_alpha = -lam^T dR/dalpha."""

    @staticmethod
    def forward(ctx, alpha, solver: BrinkmanNS):  # type: ignore[override]
        sol = solver.newton(alpha, solver.U_last)
        solver.U_last, solver.last = sol.U.detach(), sol
        ctx.solver = solver
        ctx.save_for_backward(alpha.detach(), sol.U.detach())
        return sol.U.clone()

    @staticmethod
    def backward(ctx, g_U):  # type: ignore[override]
        alpha, U = ctx.saved_tensors
        s: BrinkmanNS = ctx.solver
        lu = spla.splu(s.jacobian(U, alpha), permc_spec="COLAMD")
        lam = torch.from_numpy(lu.solve(g_U.detach().numpy(), trans="T"))
        with torch.enable_grad():
            a = alpha.clone().requires_grad_()
            Re = s._element_residual(U[s.edof], a, s.h0, s.h1)
            weight = (lam * s.free)[s.edof]  # Dirichlet rows do not depend on alpha
            (grad_a,) = torch.autograd.grad((Re * weight).sum(), a)
        return -grad_a, None
