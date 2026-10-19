"""v0 drone-arm problems: the 2D cross-section (stiffness x Navier-Stokes drag) and the 3D arm (3D FEM x
strip-theory download). Both follow the protocol of `optim.loop.run`: `initial_design()`,
`set_continuation(params)`, `evaluate(x) -> EvalResult`.

Frames. Arm axis = x. Cross-section plane = (y, z), z vertical, pointing up: thrust is +z, the rotor downwash
flows towards -z, the part is printed with z as the build axis (arm lying on the bed).

Coupling (literature: weighted sum or epsilon-constraint of compliance and drag, Azevedo et al. 2023):
    mode "weighted":  min  w_c C / C_ref + w_d D / D_ref        s.t.  area <= f_V
    mode "eps":       min  D / D_ref                            s.t.  area <= f_V,  C / C_ref <= c_max
The references are those of a REFERENCE PART (round tube of the same area in 2D, full box in 3D), so a
normalised value below 1 reads "better than the reference".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from aeroptimizer.fem import elements as el
from aeroptimizer.fem.adjoint import solve_spd
from aeroptimizer.fem.bc import Dirichlet, combine, node_indices
from aeroptimizer.fem.operators import StructuredFEOperator
from aeroptimizer.fem.solvers import LinearSolver, ScipyDirect
from aeroptimizer.flow.brinkman import BrinkmanNS, brinkman_alpha, stretched_axis
from aeroptimizer.grid import Connectivity, Grid, Masks
from aeroptimizer.materials import Orthotropic
from aeroptimizer.objectives.composite import EvalResult
from aeroptimizer.section import ArmLoads, BeamSection, SectionStiffness
from aeroptimizer.topopt.density import DensityPipeline, simp
from aeroptimizer.topopt.overhang import OverhangParams


def cell_fraction(inside, y: np.ndarray, z: np.ndarray, hy: float, hz: float, sub: int = 8) -> torch.Tensor:
    """Fraction of each cell (centres y, z) inside the region `inside(Y, Z) -> bool`, by sub-sampling."""
    o = (np.arange(sub) + 0.5) / sub - 0.5
    Y = np.asarray(y)[:, None, None] + o[None, :, None] * hy
    Z = np.asarray(z)[:, None, None] + o[None, None, :] * hz
    return torch.from_numpy(inside(Y, Z).mean(axis=(1, 2)))


# ====================================================================================== 2D section


@dataclass
class SectionSetup:
    """Geometry, material, loads and flow of the 2D section problem. Lengths in mm, forces in N, E in MPa.

    The flow is solved non-dimensionally with the box height H as length scale and the downwash speed as
    velocity scale: Re = U H / nu. Only half of the section is simulated (mirror symmetry about y = 0).
    """

    W: float = 14.0  # design box width (y) [mm]
    H: float = 20.0  # design box height (z) [mm]
    h: float = 0.5  # cell size [mm]
    volfrac: float = 0.3  # area fraction of the box
    R: float = 1.0  # density filter radius [mm] (min feature ~ 2 nozzles)
    E0: float = 3000.0  # PLA-like [MPa]
    nu: float = 0.35
    eps_s: float = 1e-6
    loads: ArmLoads = field(default_factory=ArmLoads)
    Re: float = 100.0
    alpha_max: float = 2e3  # Brinkman friction of the solid, non-dimensional (Darcy number ~ 1 / (Re alpha_max))
    q_alpha: float = 0.1
    flow_box: tuple[float, float, float, float] = (3.0, 2.5, 5.0, 0.3)  # side, upstream, downstream, core margin [H]
    stretch: float = 1.15
    overhang: OverhangParams | None = None


class SectionProblem:
    """x (half section, (ny/2, nz) flattened) -> mirrored section -> filter/projection(/overhang) -> rho_bar.

    rho_bar feeds (1) the beam-section stiffnesses (SIMP, warping solve) -> C = C_T + C_L + C_t and (2) the
    Brinkman Navier-Stokes flow around the section -> D = download per unit length (non-dimensional).
    """

    def __init__(self, setup: SectionSetup | None = None, *, mode: str = "weighted",
                 weights: dict[str, float] | None = None, c_max: float = 1.0,
                 refs: dict[str, float] | None = None, p: float = 3.0, beta: float = 1.0,
                 solver: LinearSolver | None = None) -> None:
        s = self.setup = setup or SectionSetup()
        if mode not in ("weighted", "eps"):
            raise ValueError(f"mode must be 'weighted' or 'eps', got {mode!r}")
        self.mode, self.c_max = mode, c_max
        self.weights = weights or {"C": 0.5, "D": 0.5}
        self.ny, self.nz = round(s.W / s.h), round(s.H / s.h)
        if self.ny % 2 or abs(self.ny * s.h - s.W) > 1e-9 or abs(self.nz * s.h - s.H) > 1e-9:
            raise ValueError("W / h must be an even integer and H / h an integer")
        self.section = BeamSection((self.ny, self.nz), (s.h, s.h), solver=solver)
        everywhere = torch.ones(self.ny, self.nz, dtype=torch.bool)
        none = torch.zeros_like(everywhere)
        self.pipeline = DensityPipeline(self.section.grid, Masks(everywhere, none, none), R=s.R, beta=beta,
                                        overhang=s.overhang)
        self.p = p
        self._build_flow()
        self.refs = refs or self.reference_values()

    # ------------------------------------------------------------------ flow grid (units of H)
    def _build_flow(self) -> None:
        s = self.setup
        side, up, down, margin = s.flow_box
        hf = s.h / s.H
        half = self.ny // 2
        m_cells = math.ceil(margin / hf)
        ay = stretched_axis((0.0, (half + m_cells) * hf), hf, 0.0, side, s.stretch)
        az = stretched_axis((-m_cells * hf, 1.0 + m_cells * hf), hf, -down, 1.0 + up, s.stretch)
        nodes = np.stack(np.meshgrid(ay, az, indexing="ij"), -1).reshape(-1, 2)
        fixed = torch.zeros(len(nodes), 3, dtype=torch.bool)
        vals = torch.zeros(len(nodes), 3)
        sym = torch.from_numpy(np.isclose(nodes[:, 0], 0.0) | np.isclose(nodes[:, 0], ay[-1]))
        inlet = torch.from_numpy(np.isclose(nodes[:, 1], az[-1]))
        fixed[sym, 0] = True  # symmetry plane and far side: slip (u_y = 0)
        fixed[inlet, 0] = fixed[inlet, 1] = True
        vals[inlet, 1] = -1.0  # downwash: unit speed towards -z
        self.flow = BrinkmanNS((ay, az), s.Re, fixed.reshape(-1), vals.reshape(-1))
        # embedding of the right half of the section in the flow grid
        iz0 = int(np.argmin(np.abs(az - 0.0)))
        self._embed = (slice(0, half), slice(iz0, iz0 + self.nz))

    def to_flow(self, rho_bar: torch.Tensor) -> torch.Tensor:
        """Section density (ny, nz) -> flow-grid density (flattened), right half only."""
        full = torch.zeros(self.flow.shape, dtype=rho_bar.dtype)
        full = full.index_put(
            tuple(torch.meshgrid(torch.arange(self._embed[0].start, self._embed[0].stop),
                                 torch.arange(self._embed[1].start, self._embed[1].stop), indexing="ij")),
            rho_bar[self.ny // 2:, :],
        )
        return full.reshape(-1)

    # ------------------------------------------------------------------ protocol of optim.loop.run
    @property
    def n_design(self) -> int:
        return self.ny // 2 * self.nz

    def initial_design(self) -> np.ndarray:
        return np.full(self.n_design, self.setup.volfrac)

    def set_continuation(self, params: dict[str, float]) -> None:
        self.p = params["p"]
        self.pipeline.beta = params["beta"]
        if self.pipeline.overhang is not None:
            self.pipeline.overhang.beta_s = params["beta_s"]

    def rho_bar(self, x: torch.Tensor) -> torch.Tensor:
        half = x.reshape(self.ny // 2, self.nz)
        return self.pipeline(torch.cat([half.flip(0), half], dim=0))  # (ny, nz), mirror-symmetric

    def physics(self, rho_bar: torch.Tensor) -> tuple[dict[str, torch.Tensor], SectionStiffness, torch.Tensor, torch.Tensor]:
        """Raw terms of a section density: compliances, drag, area fraction (+ stiffness and flow state)."""
        s = self.setup
        E = s.E0 * simp(rho_bar.reshape(-1), self.p, s.eps_s)
        stiff = self.section.stiffness(E, E / (2.0 * (1.0 + s.nu)))
        terms = dict(s.loads.compliances(stiff))
        terms["C"] = terms["C_T"] + terms["C_L"] + terms["C_t"]
        alpha = brinkman_alpha(self.to_flow(rho_bar), s.alpha_max, s.q_alpha)
        U = self.flow.solve(alpha)
        terms["D"] = -2.0 * self.flow.force(U, alpha)[1]  # both halves; > 0 = pushed down
        terms["C_D"] = terms["D"] / (0.5 * s.W / s.H)  # drag coefficient on the box width
        terms["area"] = rho_bar.mean()
        return terms, stiff, U, alpha

    def evaluate(self, x: torch.Tensor) -> EvalResult:
        rho_bar = self.rho_bar(x)
        terms, stiff, U, alpha = self.physics(rho_bar)
        terms["C_hat"] = terms["C"] / self.refs["C"]
        terms["D_hat"] = terms["D"] / self.refs["D"]
        g = [terms["area"] / self.setup.volfrac - 1.0]
        if self.mode == "weighted":
            J = self.weights["C"] * terms["C_hat"] + self.weights["D"] * terms["D_hat"]
        else:
            J = terms["D_hat"]
            g.append(terms["C_hat"] / self.c_max - 1.0)
        return EvalResult(J=J, g=g, terms=terms, state=(rho_bar, stiff, U, alpha))

    # ------------------------------------------------------------------ reference parts
    def shape_density(self, inside) -> torch.Tensor:
        """Section density (ny, nz) of an analytic shape `inside(y, z)` (y centred, z from the box bottom)."""
        h = self.setup.h
        return cell_fraction(inside, self.section.y.numpy(), self.section.z.numpy(), h, h).reshape(self.ny, self.nz)

    def tube(self, area: float | None = None) -> torch.Tensor:
        """Round tube inscribed in the box, wall set to give `area` (default: the target area)."""
        s = self.setup
        area = s.volfrac * s.W * s.H if area is None else area
        r_out = 0.5 * min(s.W, s.H)
        r_in = math.sqrt(max(r_out**2 - area / math.pi, 0.0))
        zc = s.H / 2
        return self.shape_density(lambda Y, Z: (Y**2 + (Z - zc) ** 2 <= r_out**2) & (Y**2 + (Z - zc) ** 2 >= r_in**2))

    def reference_values(self) -> dict[str, float]:
        """C and D of the reference tube, used to normalise both terms.

        Evaluated with p = 1: a boundary cell partly inside the tube counts for its true area fraction.
        """
        p, self.p = self.p, 1.0
        try:
            with torch.no_grad():
                terms, *_ = self.physics(self.tube())
        finally:
            self.p = p
        self.flow.U_last = None  # do not warm-start the optimisation from the tube's flow
        return {"C": float(terms["C"]), "D": float(terms["D"])}


# ====================================================================================== 3D arm


@dataclass
class Arm3DSetup:
    """Box arm x in [0, L], y in [-W/2, W/2], z in [0, H] (mm); root clamped at x = 0; motor mount at the tip.

    Download proxy (strip theory, Hoerner): every vertical column (x, y) of the arm that is under the rotor
    disc blocks the downwash; its download is q * C_D * (occupied width). Occupancy of a column is the smooth
    union 1 - prod_z (1 - rho_bar), so D = q C_D sum_{(x,y) under rotor} [1 - prod_z (1 - rho_bar)] hx hy.
    C_D is taken from the 2D section study (it only scales D, and D is normalised).
    """

    L: float = 120.0
    W: float = 24.0
    H: float = 20.0
    h: float = 2.0
    volfrac: float = 0.15
    R: float = 3.0
    E0: float = 3000.0
    nu: float = 0.35
    eps_s: float = 1e-6
    thrust: float = 8.0
    lateral: float = 3.0
    torque: float = 40.0  # N mm, applied as a vertical force couple on the mount
    mount: float = 14.0  # length of the solid motor mount at the tip [mm]
    rotor_radius: float = 63.5  # 5-inch propeller
    hub_radius: float = 12.0  # no downwash right under the hub
    C_D: float = 1.0
    q_dyn: float = 1.0  # dynamic pressure (normalised away)


class Arm3DProblem:
    """3D cantilever arm: compliance of 3 load cases (one factorisation, 3 right-hand sides) + download proxy.

    Design variables: half of the arm (y >= 0), mirrored, like the 2D section.
    """

    def __init__(self, setup: Arm3DSetup | None = None, *, weights: dict[str, float] | None = None,
                 refs: dict[str, float] | None = None, p: float = 3.0, beta: float = 1.0,
                 solver: LinearSolver | None = None) -> None:
        s = self.setup = setup or Arm3DSetup()
        self.weights = weights or {"C": 0.5, "D": 0.5}
        nx, ny, nz = round(s.L / s.h), round(s.W / s.h), round(s.H / s.h)
        if ny % 2:
            raise ValueError("W / h must be even (mirror symmetry)")
        self.shape = (nx, ny, nz)
        self.grid = Grid(self.shape, (s.h,) * 3)
        n_mount = max(1, round(s.mount / s.h))
        solid = torch.zeros(self.shape, dtype=torch.bool)
        solid[nx - n_mount:, :, :] = True  # motor mount: passive solid block at the tip
        design = ~solid
        self.masks = Masks(design=design, solid=solid, void=torch.zeros_like(solid))
        self.pipeline = DensityPipeline(self.grid, self.masks, R=s.R, beta=beta)
        self.p = p
        C0 = Orthotropic.isotropic(s.E0, s.nu).stiffness(3)
        nodes = node_indices(self.grid)
        fixed = combine(self.grid, 3, Dirichlet(nodes[:, 0] == 0, (0, 1, 2)))
        self.op = StructuredFEOperator(self.grid, Connectivity(self.grid, 3), el.elasticity_ke0(3, self.grid.h, C0), 3, fixed)
        self.solver = solver or ScipyDirect()
        # loads on the top face of the mount (thrust, torque couple) and its side faces (lateral)
        top = (nodes[:, 0] >= nx - n_mount) & (nodes[:, 2] == nz)
        f = torch.zeros(3, self.grid.n_nodes, 3)
        f[0, top, 2] = s.thrust / int(top.sum())
        f[1, top, 1] = s.lateral / int(top.sum())
        y_node = nodes[:, 1].to(torch.get_default_dtype()) * s.h - s.W / 2
        lever = (y_node[top] ** 2).sum()  # sum F_i y_i = M with F_i = M y_i / sum y_j^2
        f[2, top, 2] = s.torque * y_node[top] / lever
        self.f = f.reshape(3, -1).T.contiguous()  # (N, 3)
        # rotor footprint on the columns (x, y): downwash dynamic pressure (0 outside the disc and under the hub)
        xc = (torch.arange(nx) + 0.5) * s.h
        yc = (torch.arange(ny) + 0.5) * s.h - s.W / 2
        X, Y = torch.meshgrid(xc, yc, indexing="ij")
        r = torch.sqrt((X - (s.L - s.mount / 2)) ** 2 + Y**2)
        self.q_map = s.q_dyn * ((r <= s.rotor_radius) & (r >= s.hub_radius)).to(torch.get_default_dtype())
        self.design_index = torch.nonzero(design[:, ny // 2:, :].reshape(-1)).squeeze(1)
        self.refs = refs or self.reference_values()

    @property
    def n_design(self) -> int:
        return int(self.design_index.numel())

    def initial_design(self) -> np.ndarray:
        return np.full(self.n_design, self.setup.volfrac)

    def set_continuation(self, params: dict[str, float]) -> None:
        self.p = params["p"]
        self.pipeline.beta = params["beta"]

    def rho_bar(self, x: torch.Tensor) -> torch.Tensor:
        nx, ny, nz = self.shape
        half = torch.zeros(nx * (ny // 2) * nz, dtype=x.dtype).index_put((self.design_index,), x)
        half = half.reshape(nx, ny // 2, nz)
        return self.pipeline(torch.cat([half.flip(1), half], dim=1))

    def physics(self, rho_bar: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        s = self.setup
        u = solve_spd(self.op, simp(rho_bar.reshape(-1), self.p, s.eps_s), self.f, self.solver)  # (N, 3)
        Ck = (self.f * u).sum(dim=0)
        occupancy = 1.0 - torch.prod(1.0 - rho_bar, dim=2)  # (nx, ny): smooth "is there matter in this column"
        D = s.C_D * (self.q_map * occupancy).sum() * s.h**2
        design = self.masks.design
        terms = {"C_T": Ck[0], "C_L": Ck[1], "C_t": Ck[2], "C": Ck.sum(), "D": D,
                 "vol": rho_bar[design].mean()}
        return terms, u

    def evaluate(self, x: torch.Tensor) -> EvalResult:
        rho_bar = self.rho_bar(x)
        terms, u = self.physics(rho_bar)
        terms["C_hat"] = terms["C"] / self.refs["C"]
        terms["D_hat"] = terms["D"] / self.refs["D"]
        J = self.weights["C"] * terms["C_hat"] + self.weights["D"] * terms["D_hat"]
        g = [terms["vol"] / self.setup.volfrac - 1.0]
        return EvalResult(J=J, g=g, terms=terms, state=(rho_bar, u))

    def tube(self) -> torch.Tensor:
        """Reference arm: round tube along x (outer diameter min(W, H), same volume as the target), solid mount."""
        s = self.setup
        nx, ny, nz = self.shape
        r_out = 0.5 * min(s.W, s.H)
        area = s.volfrac * s.W * s.H
        r_in = math.sqrt(max(r_out**2 - area / math.pi, 0.0))
        yc = (np.arange(ny) + 0.5) * s.h - s.W / 2
        zc = (np.arange(nz) + 0.5) * s.h
        Yc, Zc = np.meshgrid(yc, zc, indexing="ij")
        ring = lambda Y, Z: (Y**2 + (Z - s.H / 2) ** 2 <= r_out**2) & (Y**2 + (Z - s.H / 2) ** 2 >= r_in**2)  # noqa: E731
        sec = cell_fraction(ring, Yc.reshape(-1), Zc.reshape(-1), s.h, s.h).reshape(ny, nz)
        rho = sec[None].expand(nx, ny, nz).clone()
        rho[self.masks.solid] = 1.0
        return rho

    def reference_values(self) -> dict[str, float]:
        """C and D of the reference tube arm (p = 1: partial cells count for their true fraction)."""
        with torch.no_grad():
            p, self.p = self.p, 1.0
            try:
                terms, _ = self.physics(self.tube())
            finally:
                self.p = p
        return {"C": float(terms["C"]), "D": float(terms["D"])}
