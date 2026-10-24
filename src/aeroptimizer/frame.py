"""v0 drone frame: the central body of a 5-inch X quad, arms bolted underneath, printed flat (FDM).

Same protocol as `arm.Arm3DProblem` (`n_design`, `initial_design`, `set_continuation`, `rho_bar`, `physics`,
`evaluate`, `reference_values`), so `optim.loop.run` drives it unchanged.

Frame and units. x forward, y left, z up = build axis; the bottom face z = 0 lies on the build plate, the
box is centred in x and y. Lengths in mm, forces in N, moduli in MPa, so compliances are in N mm (mJ).
Masses in kg and the rigid-body dynamics are computed in SI, then converted.

Interfaces (passive zones, `Masks`):
* 4 arms bolted under the bottom face, 2 M3 screws each along the diagonal: solid bosses, void holes;
* FC/ESC stack, 30.5 x 30.5 mm M3 pattern: solid bosses on the bottom, the stack itself is a void bay above;
* battery on top: 4 solid pads at its corners (the strap is not modelled). The pads, not a full plate, carry
  it: a plate above the stack would be a 38 mm bridge, which the 45 deg overhang rule forbids by construction.
Screw holes go through the whole height (tool access).

Loads: no clamp. Each load case is a set of motor thrusts (and their reaction torques). The whole drone is
treated as rigid bodies (arm units at the motors, battery and stack as boxes, the frame mass is lumped in the
stack): Newton-Euler gives the linear and angular accelerations at the start of the manoeuvre (zero angular
velocity), and every component passes to the frame its applied loads + weight + d'Alembert inertia force,
spread over its interface nodes as a rigid patch. The loads are then self-equilibrated EXACTLY, and the
3-2-1 isostatic supports carry zero reaction: the compliance does not depend on where they are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from aeroptimizer.fem import elements as el
from aeroptimizer.fem.adjoint import compliances
from aeroptimizer.fem.bc import Dirichlet, combine, node_indices
from aeroptimizer.fem.operators import StructuredFEOperator
from aeroptimizer.fem.solvers import AMGCG, LinearSolver
from aeroptimizer.grid import Connectivity, Grid, Masks
from aeroptimizer.materials import Orthotropic
from aeroptimizer.objectives.composite import EvalResult
from aeroptimizer.topopt.density import DensityPipeline, simp
from aeroptimizer.topopt.overhang import OverhangParams

G = 9.81  # m/s^2
PLA_DENSITY = 1.24e-3  # g/mm^3

# FDM PLA, transversely isotropic about z (plausible, NOT calibrated): the bond between layers is weaker.
PLA_FDM = Orthotropic.transversely_isotropic(Ep=3300.0, Ez=2300.0, nup=0.33, nupz=0.33, Gz=900.0)
PLA_ISO = Orthotropic.isotropic(3300.0, 0.33)

# Motors in Betaflight "props in" order is irrelevant here; we number them by quadrant:
# 0 front-left (+,+), 1 rear-left (-,+), 2 rear-right (-,-), 3 front-right (+,-).
# spin seen from above: +1 counter-clockwise. Front-left and rear-right spin clockwise.
MOTOR_SIGNS = ((1, 1), (-1, 1), (-1, -1), (1, -1))
MOTOR_SPIN = (-1, 1, -1, 1)

# Load cases: thrust of each motor, as a fraction of thrust_max ("lo" = thrust_low).
LOAD_CASES: dict[str, tuple[str, str, str, str]] = {
    "poussee": ("hi", "hi", "hi", "hi"),  # full-throttle punch-out (a hard flat landing loads the same paths)
    "roulis": ("hi", "hi", "lo", "lo"),  # left motors high: roll to the right
    "tangage": ("lo", "hi", "hi", "lo"),  # rear motors high: nose down
    "lacet": ("hi", "lo", "hi", "lo"),  # clockwise props high: their reaction torque yaws the frame
}


@dataclass
class FrameSetup:
    L: float = 80.0  # box length (x) [mm]
    W: float = 76.0  # box width (y) [mm]
    H: float = 28.0  # box height (z, build axis) [mm]
    h: float = 2.0  # cell size [mm]
    volfrac: float = 0.2  # of the design cells
    R: float = 3.0  # density filter radius [mm]
    material: Orthotropic = field(default_factory=lambda: PLA_FDM)
    eps_s: float = 1e-6
    # drone
    wheelbase: float = 220.0  # motor-to-motor diagonal [mm], 5-inch class
    arm_screws: tuple[float, float] = (34.0, 46.0)  # radii of the 2 arm screws along the diagonal [mm]
    boss_radius: float = 4.5  # solid boss around each screw [mm]
    boss_height: float = 4.0
    hole_radius: float = 1.6  # M3 clearance
    fc_pitch: float = 30.5  # FC/ESC stack hole pattern [mm]
    fc_board: float = 38.0  # footprint of the stack (void bay) [mm]
    fc_bay: tuple[float, float] = (4.0, 24.0)  # z range of the stack [mm]
    battery: tuple[float, float, float] = (75.0, 35.0, 35.0)  # 4S 1300 mAh, lying on the top face [mm]
    pad: float = 8.0  # side of the 4 battery pads [mm]
    pad_height: float = 2.0
    m_battery: float = 0.18  # [kg]
    m_stack: float = 0.22  # FC, ESC, VTX, camera, wires and the frame itself, lumped on the stack [kg]
    m_arm: float = 0.05  # arm + motor + propeller, lumped at the motor [kg]; all-up mass 0.6 kg
    thrust_max: float = 11.8  # per motor [N] (1.2 kgf, thrust/weight ~ 8)
    thrust_low: float = 0.1  # fraction of thrust_max on the slow motors of the differential cases
    k_torque: float = 0.012  # propeller reaction torque / thrust [m], order of magnitude of a 5" prop
    weights: dict[str, float] = field(default_factory=lambda: {k: 1.0 for k in LOAD_CASES})
    symmetry: tuple[int, ...] = (0, 1)  # mirror planes of the DENSITY only (x = 0, y = 0); loads stay asymmetric
    overhang: OverhangParams | None = None


# ------------------------------------------------------------------------------------------ rigid bodies


@dataclass
class Body:
    """A component carried by the frame: point mass + own inertia, external loads, interface nodes."""

    name: str
    mass: float  # kg
    pos: np.ndarray  # (3,) m, centre of mass
    inertia: np.ndarray  # (3, 3) kg m^2 about its centre of mass
    nodes: np.ndarray  # (n,) int, frame nodes the body is attached to


def box_inertia(m: float, a: float, b: float, c: float) -> np.ndarray:
    """Inertia (kg m^2) of a homogeneous box of sides a, b, c (m) about its centre."""
    return m / 12.0 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])


def patch_forces(xyz: np.ndarray, force: np.ndarray, moment: np.ndarray, about: np.ndarray) -> np.ndarray:
    """Nodal forces (n, 3) on the points `xyz` (n, 3) equivalent to the wrench (force, moment about `about`).

    Rigid-patch rule: f_i = F / n + w x r_i, r_i relative to the centroid c, w = J^+ M_c with
    J = sum(|r|^2 I - r r^T). Then sum f_i = F and sum r_i x f_i = J w = M_c exactly (J invertible as soon as
    the points are not collinear). The least-norm distribution of a wrench over a rigid connection.
    """
    c = xyz.mean(axis=0)
    r = xyz - c
    Mc = moment + np.cross(about - c, force)
    J = (r * r).sum() * np.eye(3) - r.T @ r
    w = np.linalg.pinv(J) @ Mc
    return force[None, :] / len(xyz) + np.cross(w[None, :], r)


def rigid_body_loads(bodies: list[Body], thrust: list[np.ndarray], torque: list[np.ndarray]):  # noqa: ANN201
    """Loads each body passes to the frame: (forces (k, 3) N, moments (k, 3) N m about each body's pos),
    plus the accelerations (a (3,), alpha (3,)) of the drone. thrust/torque: external loads per body (SI).

    Newton-Euler at zero angular velocity: M a = sum F + M g, I_c alpha = sum (p - c) x F + sum Q.
    Body i passes F_i + m_i g - m_i (a + alpha x (p_i - c)) and Q_i - I_i alpha: the sum is zero.
    """
    m = np.array([b.mass for b in bodies])
    p = np.stack([b.pos for b in bodies])
    M = m.sum()
    c = (m[:, None] * p).sum(0) / M
    r = p - c
    Ic = sum(b.inertia + b.mass * ((ri @ ri) * np.eye(3) - np.outer(ri, ri)) for b, ri in zip(bodies, r))
    g = np.array([0.0, 0.0, -G])
    F = np.stack(thrust)
    Q = np.stack(torque)
    a = (F.sum(0) + M * g) / M
    alpha = np.linalg.solve(Ic, np.cross(r, F).sum(0) + Q.sum(0))
    acc = a[None, :] + np.cross(alpha[None, :], r)
    forces = F + m[:, None] * (g[None, :] - acc)
    moments = Q - np.stack([b.inertia @ alpha for b in bodies])
    return forces, moments, a, alpha


# ------------------------------------------------------------------------------------------ problem


class FrameProblem:
    """x (design cells of one symmetry sector) -> mirrored -> filter/projection(/overhang) -> rho_bar
    -> 3D FEM, one operator and k load cases -> J = sum_k w_k C_k / C_k,ref, s.t. vol <= volfrac.
    """

    def __init__(self, setup: FrameSetup | None = None, *, refs: dict[str, float] | None = None, p: float = 3.0,
                 beta: float = 1.0, solver: LinearSolver | None = None) -> None:
        s = self.setup = setup or FrameSetup()
        n = tuple(round(v / s.h) for v in (s.L, s.W, s.H))
        if any(abs(k * s.h - v) > 1e-9 for k, v in zip(n, (s.L, s.W, s.H))):
            raise ValueError("L, W, H must be multiples of h")
        if any(n[a] % 2 for a in s.symmetry):
            raise ValueError("the box must have an even number of cells across each symmetry plane")
        self.shape = n
        self.grid = Grid(n, (s.h,) * 3)
        self._build_masks()
        self.pipeline = DensityPipeline(self.grid, self.masks, R=s.R, beta=beta, overhang=s.overhang)
        self.p = p
        self.solver = solver or AMGCG()
        self.nodes = node_indices(self.grid)
        self.xyz = self.nodes.numpy() * s.h - np.array([s.L / 2, s.W / 2, 0.0])  # node coordinates [mm]
        self._build_supports()
        self.op = StructuredFEOperator(self.grid, Connectivity(self.grid, 3), el.elasticity_ke0(3, self.grid.h, s.material.stiffness(3)), 3, self.fixed)
        self._build_loads()
        sector = self.masks.design[self._sector]
        self.design_index = torch.nonzero(sector.reshape(-1)).squeeze(1)
        self._sector_shape = tuple(sector.shape)
        self.refs = refs or self.reference_values()

    # ------------------------------------------------------------------ geometry
    def centres(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Element centres (X, Y, Z), each of the grid shape [mm]."""
        s = self.setup
        ax = [(np.arange(k) + 0.5) * s.h - off for k, off in zip(self.shape, (s.L / 2, s.W / 2, 0.0))]
        return np.meshgrid(*ax, indexing="ij")

    def snap(self, v: float, axis: int) -> float:
        """Nearest node coordinate along x (axis 0) or y (axis 1); the box is centred."""
        h, half = self.setup.h, (self.setup.L, self.setup.W)[axis] / 2
        return round((v + half) / h) * h - half

    def screws(self) -> dict[str, list[tuple[float, float]]]:
        """Screw centres (x, y) [mm], snapped to nodes: arms 'bras0'..'bras3' and the FC stack 'fc'."""
        s = self.setup
        out = {}
        for k, (sx, sy) in enumerate(MOTOR_SIGNS):
            out[f"bras{k}"] = [(self.snap(sx * r / math.sqrt(2), 0), self.snap(sy * r / math.sqrt(2), 1)) for r in s.arm_screws]
        out["fc"] = [(self.snap(sx * s.fc_pitch / 2, 0), self.snap(sy * s.fc_pitch / 2, 1)) for sx, sy in MOTOR_SIGNS]
        return out

    def _build_masks(self) -> None:
        s = self.setup
        X, Y, Z = self.centres()
        solid = np.zeros(self.shape, bool)
        void = np.zeros(self.shape, bool)
        self.zones: dict[str, np.ndarray] = {}
        for name, pts in self.screws().items():
            zone = np.zeros(self.shape, bool)
            for cx, cy in pts:
                d2 = (X - cx) ** 2 + (Y - cy) ** 2
                zone |= (d2 <= s.boss_radius**2) & (Z < s.boss_height)
                void |= d2 <= s.hole_radius**2
            self.zones[name] = zone
        bay = (np.abs(X) <= s.fc_board / 2) & (np.abs(Y) <= s.fc_board / 2) & (Z > s.fc_bay[0]) & (Z < s.fc_bay[1])
        void |= bay
        bl, bw, _ = s.battery
        pads = np.zeros(self.shape, bool)
        for sx, sy in MOTOR_SIGNS:
            cx, cy = sx * (bl - s.pad) / 2, sy * (bw - s.pad) / 2
            pads |= (np.abs(X - cx) <= s.pad / 2) & (np.abs(Y - cy) <= s.pad / 2) & (Z > s.H - s.pad_height)
        self.zones["batterie"] = pads
        for zone in self.zones.values():
            zone &= ~void
            solid |= zone
        self.masks = Masks(design=torch.from_numpy(~(solid | void)), solid=torch.from_numpy(solid),
                           void=torch.from_numpy(void))
        self._sector = tuple(slice(self.shape[a] // 2, None) if a in s.symmetry else slice(None) for a in range(3))

    def face_nodes(self, cells: np.ndarray, top: bool) -> np.ndarray:
        """Node indices of the bottom (top=False) or top faces of the given cells."""
        nx, ny, nz = self.shape
        m = np.zeros((nx + 1, ny + 1, nz + 1), bool)
        c = 1 if top else 0
        for a in (0, 1):
            for b in (0, 1):
                m[a:a + nx, b:b + ny, c:c + nz] |= cells
        return np.flatnonzero(m)

    def interface_nodes(self, zone: str) -> np.ndarray:
        """Where a component touches the frame: bottom face of the arm bosses, top faces of the FC bosses and
        of the battery pads."""
        cells = self.zones[zone]
        layers = np.flatnonzero(cells.any(axis=(0, 1)))
        if zone.startswith("bras"):
            keep = np.zeros_like(cells)
            keep[..., layers[0]] = cells[..., layers[0]]
            return self.face_nodes(keep, top=False)
        keep = np.zeros_like(cells)
        keep[..., layers[-1]] = cells[..., layers[-1]]
        return self.face_nodes(keep, top=True)

    def _build_supports(self) -> None:
        """3-2-1 isostatic supports on the arm patches: A (bras0) x y z, B (bras1, along x from A) y z,
        C (bras3, along y from A) z. Rank checked by `combine` (C-BC-01)."""
        picks = []
        for zone in ("bras0", "bras1", "bras3"):
            idx = self.interface_nodes(zone)
            c = self.xyz[idx].mean(axis=0)
            picks.append(idx[np.argmin(((self.xyz[idx] - c) ** 2).sum(1))])
        n_nodes = self.grid.n_nodes
        masks = []
        for node, comps in zip(picks, ((0, 1, 2), (1, 2), (2,))):
            m = torch.zeros(n_nodes, dtype=torch.bool)
            m[int(node)] = True
            masks.append(Dirichlet(m, comps))
        self.support_nodes = picks
        self.fixed = combine(self.grid, 3, *masks)

    # ------------------------------------------------------------------ loads
    def bodies(self) -> list[Body]:
        s = self.setup
        mm = 1e-3
        out = []
        r_motor = s.wheelbase / 2 * mm
        for k, (sx, sy) in enumerate(MOTOR_SIGNS):
            pos = np.array([sx, sy, 0.0]) * r_motor / math.sqrt(2)  # arm under the bottom face: z ~ 0
            out.append(Body(f"bras{k}", s.m_arm, pos, np.zeros((3, 3)), self.interface_nodes(f"bras{k}")))
        bl, bw, bh = s.battery
        out.append(Body("batterie", s.m_battery, np.array([0.0, 0.0, (s.H + bh / 2) * mm]),
                        box_inertia(s.m_battery, bl * mm, bw * mm, bh * mm), self.interface_nodes("batterie")))
        z0, z1 = s.fc_bay
        out.append(Body("fc", s.m_stack, np.array([0.0, 0.0, (z0 + z1) / 2 * mm]),
                        box_inertia(s.m_stack, s.fc_board * mm, s.fc_board * mm, (z1 - z0) * mm), self.interface_nodes("fc")))
        return out

    def motor_loads(self, case: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """External loads on every body (SI) for a load case: thrust and reaction torque on the arm units."""
        s = self.setup
        levels = {"hi": 1.0, "lo": s.thrust_low}
        F, Q = [], []
        for k, lvl in enumerate(LOAD_CASES[case]):
            T = levels[lvl] * s.thrust_max
            F.append(np.array([0.0, 0.0, T]))
            Q.append(np.array([0.0, 0.0, -MOTOR_SPIN[k] * s.k_torque * T]))
        F += [np.zeros(3), np.zeros(3)]
        Q += [np.zeros(3), np.zeros(3)]
        return F, Q

    def _build_loads(self) -> None:
        """f (N, k): nodal forces of every load case [N]; `kinematics` keeps a [m/s^2] and alpha [rad/s^2]."""
        bodies = self.bodies()
        self.cases = list(LOAD_CASES)
        cols, self.kinematics = [], {}
        xyz_m = self.xyz * 1e-3
        for case in self.cases:
            F, Q = self.motor_loads(case)
            forces, moments, a, alpha = rigid_body_loads(bodies, F, Q)
            f = np.zeros((self.grid.n_nodes, 3))
            for b, Fb, Mb in zip(bodies, forces, moments):
                np.add.at(f, b.nodes, patch_forces(xyz_m[b.nodes], Fb, Mb, b.pos))
            cols.append(f.reshape(-1))
            self.kinematics[case] = (a, alpha)
        self.f = torch.from_numpy(np.stack(cols, axis=1))

    def resultant(self) -> np.ndarray:
        """(k, 6): net force [N] and moment about the origin [N mm] of each load case (zero by construction)."""
        f = self.f.numpy().T.reshape(len(self.cases), -1, 3)
        return np.stack([np.concatenate([fk.sum(0), np.cross(self.xyz, fk).sum(0)]) for fk in f])

    # ------------------------------------------------------------------ protocol of optim.loop.run
    @property
    def n_design(self) -> int:
        return int(self.design_index.numel())

    def initial_design(self) -> np.ndarray:
        return np.full(self.n_design, self.setup.volfrac)

    def set_continuation(self, params: dict[str, float]) -> None:
        self.p = params["p"]
        self.pipeline.beta = params["beta"]
        if self.pipeline.overhang is not None:
            self.pipeline.overhang.beta_s = params["beta_s"]

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """Design variables of the sector -> full grid field (mirrored), zeros on the passive cells."""
        n = math.prod(self._sector_shape)
        field_ = torch.zeros(n, dtype=x.dtype).index_put((self.design_index,), x).reshape(self._sector_shape)
        for a in self.setup.symmetry:
            field_ = torch.cat([field_.flip(a), field_], dim=a)
        return field_

    def rho_bar(self, x: torch.Tensor) -> torch.Tensor:
        return self.pipeline(self.expand(x))

    def physics(self, rho_bar: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        s = self.setup
        C, u = compliances(self.op, simp(rho_bar.reshape(-1), self.p, s.eps_s), self.f, self.solver)
        terms = {f"C_{c}": C[k] for k, c in enumerate(self.cases)}
        terms["vol"] = rho_bar[self.masks.design].mean()
        terms["masse"] = rho_bar.sum() * s.h**3 * PLA_DENSITY  # grams, passive solids included
        return terms, u

    def evaluate(self, x: torch.Tensor) -> EvalResult:
        rho_bar = self.rho_bar(x)
        terms, u = self.physics(rho_bar)
        w = self.setup.weights
        J = sum(w[c] * terms[f"C_{c}"] / self.refs[c] for c in self.cases) / sum(w.values())
        for c in self.cases:
            terms[f"C_{c}_hat"] = terms[f"C_{c}"] / self.refs[c]
        terms["C_hat"] = J
        g = [terms["vol"] / self.setup.volfrac - 1.0]
        return EvalResult(J=J, g=g, terms=terms, state=(rho_bar, u))

    # ------------------------------------------------------------------ reference
    def reference_density(self) -> torch.Tensor:
        """Same mass spread evenly: rho = volfrac on every design cell, 1 on the passive solids."""
        m = self.masks
        return self.setup.volfrac * m.design.to(torch.get_default_dtype()) + m.solid.to(torch.get_default_dtype())

    def reference_values(self) -> dict[str, float]:
        """C_k of the reference density with p = 1 (no SIMP penalty: the grey material counts at its true
        stiffness fraction). Only used to normalise the cases so that equal weights mean equal importance."""
        with torch.no_grad():
            p, self.p = self.p, 1.0
            try:
                terms, _ = self.physics(self.reference_density())
            finally:
                self.p = p
        return {c: float(terms[f"C_{c}"]) for c in self.cases}
