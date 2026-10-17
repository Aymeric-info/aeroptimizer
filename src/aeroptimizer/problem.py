"""Problem = immutable plan + pure evaluate(x) (ARCHITECTURE §5.8)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from aeroptimizer.config import Config
from aeroptimizer.fem import elements as el
from aeroptimizer.fem.adjoint import solve_spd
from aeroptimizer.errors import ConfigError
from aeroptimizer.fem.bc import Dirichlet, DistributedForce, build_case, combine, node_indices
from aeroptimizer.fem.operators import StructuredFEOperator
from aeroptimizer.fem.solvers import LinearSolver, ScipyDirect
from aeroptimizer.flow.aero import download_force, fluid_side_pressure
from aeroptimizer.flow.potential import PotentialFlow
from aeroptimizer.grid import Connectivity, Grid, Masks
from aeroptimizer.materials import Orthotropic
from aeroptimizer.objectives.composite import Composite, EvalResult
from aeroptimizer.objectives.constraints import volume_constraint
from aeroptimizer.objectives.mechanical import compliance
from aeroptimizer.topopt.density import DensityPipeline, simp
from aeroptimizer.topopt.overhang import OverhangParams


@dataclass(frozen=True)
class Plan:
    """Everything independent of x: Grid, Connectivity, Ke0, operators, masks, kernels, inlet profile."""

    # filled by Problem.from_config; kept as a dataclass so that it is trivially immutable.


class Problem:
    plan: Plan

    @classmethod
    def from_config(cls, cfg: Config) -> Problem:
        raise NotImplementedError("phase 2")

    def evaluate(self, x: torch.Tensor) -> EvalResult:
        """rho_bar = pipeline(x); one mechanics solve, one flow solve (cached); returns J, g, raw terms."""
        raise NotImplementedError("phase 2")


class ComplianceProblem:
    """Compliance + volume on a named case (lessons 8-10): x -> pipeline -> SIMP -> solve_spd -> C.

    The v1 `Problem` (flow, composite objective, config) will generalise it. Protocol used by
    `optim.loop.run`: `initial_design()`, `set_continuation(params)`, `evaluate(x_flat) -> EvalResult`.
    `overhang` adds the self-support filter (lesson 12; `beta_s` follows the schedule).
    `C0` is the stiffness of the solid material in the grid frame (lesson 11: orthotropic, rotated).
    With `normalize=True` the objective is C / C_ref, C_ref frozen at the first evaluation (C-OBJ-03).
    """

    def __init__(
        self,
        case: str,
        shape: tuple[int, ...],
        *,
        volfrac: float,
        R: float,
        h: tuple[float, ...] | None = None,
        eps_s: float = 1e-9,
        C0: torch.Tensor | None = None,
        overhang: OverhangParams | None = None,
        solver: LinearSolver | None = None,
        normalize: bool = False,
        beta: float = 0.0,
        p: float = 3.0,
    ) -> None:
        d = len(shape)
        self.grid = Grid(shape, h or (1.0,) * d)
        fixed, load, masks = build_case(case, self.grid)
        # C0: stiffness of the full material in the grid frame (default: isotropic E=1, nu=0.3)
        self.C0 = Orthotropic.isotropic(1.0, 0.3).stiffness(d) if C0 is None else C0
        Ke0 = el.elasticity_ke0(d, self.grid.h, self.C0)
        self.op = StructuredFEOperator(self.grid, Connectivity(self.grid, d), Ke0, d, fixed)
        self.f = load.assemble(torch.ones(shape))
        self.pipeline = DensityPipeline(self.grid, masks, R=R, beta=beta, overhang=overhang)
        self.volfrac, self.eps_s, self.p = volfrac, eps_s, p
        self.solver = solver or ScipyDirect()
        self.normalize, self.c_ref = normalize, None

    def initial_design(self) -> np.ndarray:
        return np.full(self.grid.n_elems, self.volfrac)

    def set_continuation(self, params: dict[str, float]) -> None:
        self.p = params["p"]
        self.pipeline.beta = params["beta"]
        if self.pipeline.overhang is not None:
            self.pipeline.overhang.beta_s = params["beta_s"]

    def rho_bar(self, x: torch.Tensor) -> torch.Tensor:
        return self.pipeline(x.reshape(self.grid.shape)).reshape(-1)

    def evaluate(self, x: torch.Tensor) -> EvalResult:
        rho_bar = self.rho_bar(x)
        u = solve_spd(self.op, simp(rho_bar, self.p, self.eps_s), self.f, self.solver)
        C = compliance(self.f, u)
        if self.normalize and self.c_ref is None:
            self.c_ref = float(C.detach())  # frozen once (C-OBJ-03)
        J = C / self.c_ref if self.normalize else C
        vol = rho_bar.mean()
        return EvalResult(J=J, g=[vol / self.volfrac - 1.0], terms={"C": C, "vol": vol}, state=u)

    def constraint_value(self, x: np.ndarray) -> float:
        """Exact volume constraint g_V(x) = mean(rho_bar(x)) / volfrac - 1 (no gradient)."""
        with torch.no_grad():
            return float(self.rho_bar(torch.from_numpy(np.asarray(x, dtype=float))).mean() / self.volfrac - 1.0)


class CoupledProblem:
    """Console under a vertical stream: stiffness (C) against download (D) on ONE density field (lesson 17).

    Layout, in elements, axes (x, z), z up and the stream going DOWN (C-CONV-07):

        pad_top    : passive fluid (inlet buffer of the flow, C-FLOW-05)
        depth      : design region, `span` wide, clamped at x = 0, loaded at x = span
        pad_bottom : passive fluid (room for the outlet)
        margin_x   : passive fluid to the right of the console (the stream can go round it)

    Only the `span * depth` design cells are variables. One mechanical solve and one flow solve per
    evaluation (C-OBJ-01); both are differentiated through the adjoint. `J = w_c C/C_ref + w_a D/D_ref`
    (`Composite`); `g_V` is the relative volume constraint over the design region only (C-OBJ-04).
    Protocol of `optim.loop.run`: `initial_design()`, `set_continuation(params)`, `evaluate(x)`.

    `c_b`, `fluid_side` and `floor_at_base` define the download indicator (lessons 15 and 17). Unprotected,
    D is not bounded below (sharp corners give huge suction) and an optimiser drives it negative; the
    composite refuses a non-positive D_ref for the same reason.
    """

    def __init__(
        self,
        span: int,
        depth: int,
        *,
        volfrac: float,
        R: float,
        weights: dict[str, float],
        refs: dict[str, float] | None = None,
        margin_x: int = 12,
        pad_top: int = 5,
        pad_bottom: int = 5,
        c_b: float = 1.4,
        fluid_side: bool = True,
        floor_at_base: bool = True,
        pressure_kw: dict | None = None,
        rho_air: float = 1.225,
        U: float = 1.0,
        q: float = 3.0,
        kappa_min: float = 1e-6,
        eps_s: float = 1e-9,
        C0: torch.Tensor | None = None,
        solver: LinearSolver | None = None,
        flow_solver: LinearSolver | None = None,
        beta: float = 1.0,
        p: float = 3.0,
    ) -> None:
        if min(span, depth) < 2 or margin_x < 1 or pad_bottom < 1 or pad_top < 4:
            raise ConfigError("need span, depth >= 2, margin_x >= 1, pad_bottom >= 1 and pad_top >= 4 (C-FLOW-05)")
        nx, nz = span + margin_x, pad_bottom + depth + pad_top
        self.grid = Grid((nx, nz), (1.0, 1.0))
        design = torch.zeros(nx, nz, dtype=torch.bool)
        design[:span, pad_bottom : pad_bottom + depth] = True
        none = torch.zeros_like(design)
        masks = Masks(design=design, solid=none, void=~design)
        self.design_mask = design
        self.design_index = torch.nonzero(design.reshape(-1)).squeeze(1)
        # mechanics: clamp the face x = 0 and load the face x = span, over the design rows only
        nodes = node_indices(self.grid)
        rows = (nodes[:, 1] >= pad_bottom) & (nodes[:, 1] <= pad_bottom + depth)
        fixed = combine(self.grid, 2, Dirichlet(rows & (nodes[:, 0] == 0), (0, 1)))
        self.f = DistributedForce(rows & (nodes[:, 0] == span), (0.0, -1.0), self.grid).assemble(torch.ones(1))
        self.C0 = Orthotropic.isotropic(1.0, 0.3).stiffness(2) if C0 is None else C0
        Ke0 = el.elasticity_ke0(2, self.grid.h, self.C0)
        self.op = StructuredFEOperator(self.grid, Connectivity(self.grid, 2), Ke0, 2, fixed)
        self.solver = solver or ScipyDirect()
        self.pipeline = DensityPipeline(self.grid, masks, R=R, beta=beta)
        # flow: uniform stream entering by the top face
        self.flow = PotentialFlow(
            self.grid, torch.full((nx,), U), solver=flow_solver, kappa_min=kappa_min, q=q,
            rho_air=rho_air, u_ref=U, n_pad=pad_top,
        )
        self.composite = Composite(weights, refs)
        self.volfrac, self.eps_s, self.p = volfrac, eps_s, p
        self.c_b, self.fluid_side, self.rho_air, self.U = c_b, fluid_side, rho_air, U
        self.floor_at_base = floor_at_base
        self.pressure_kw = {"rho_c": 0.5} if pressure_kw is None else dict(pressure_kw)

    @property
    def n_design(self) -> int:
        return int(self.design_index.numel())

    def initial_design(self) -> np.ndarray:
        return np.full(self.n_design, self.volfrac)

    def set_continuation(self, params: dict[str, float]) -> None:
        self.p = params["p"]
        self.pipeline.beta = params["beta"]

    def rho_bar(self, x: torch.Tensor) -> torch.Tensor:
        """x (n_design,) -> rho_bar on the whole grid, shape (nx, nz)."""
        full = torch.zeros(self.grid.n_elems, dtype=x.dtype).index_put((self.design_index,), x)
        return self.pipeline(full.reshape(self.grid.shape))

    def terms(self, rho_bar: torch.Tensor) -> tuple[dict[str, torch.Tensor], object]:
        """Raw terms C, D (one solve each) and the volume fraction; the state holds u and the flow state."""
        u = solve_spd(self.op, simp(rho_bar.reshape(-1), self.p, self.eps_s), self.f, self.solver)
        st = self.flow.solve(rho_bar)
        pressure = fluid_side_pressure(st.p_rel, rho_bar, **self.pressure_kw) if self.fluid_side else st.p_rel
        D = download_force(
            rho_bar, pressure, h=self.grid.h, rho_air=self.rho_air, u_ref=self.U, c_b=self.c_b,
            floor_at_base=self.floor_at_base,
        )
        vol = rho_bar[self.design_mask].mean()
        return {"C": compliance(self.f, u), "D": D, "vol": vol}, (u, st)

    def evaluate(self, x: torch.Tensor) -> EvalResult:
        rho_bar = self.rho_bar(x)
        terms, state = self.terms(rho_bar)
        g = volume_constraint(rho_bar, self.design_mask, self.volfrac, relative=True)
        return self.composite.value(terms, [g], state)

    def constraint_value(self, x: np.ndarray) -> float:
        """Exact volume constraint (no gradient), for OC's bisection."""
        with torch.no_grad():
            return float(
                volume_constraint(
                    self.rho_bar(torch.from_numpy(np.asarray(x, dtype=float))), self.design_mask,
                    self.volfrac, relative=True,
                )
            )
