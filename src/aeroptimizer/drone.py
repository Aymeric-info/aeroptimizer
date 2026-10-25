"""v0 monolithic drone: central body + 4 arms printed as ONE part, flat (FDM). What the solver produces alone.

Same protocol and same physics as `frame.FrameProblem` (it is a subclass: loads, supports, objective, reference,
continuation are inherited). What changes is the domain and the passive zones. The box now spans the whole
drone; everything that is neither the central body nor an arm is a forced void:

* central body: |x| <= body[0]/2, |y| <= body[1]/2, full height (like the frame v0);
* arm corridors: a band of width `arm_width` along each diagonal, from the body to the motor, of height `arm_height`
  (the optimiser decides what to keep inside: tube, truss, rib...). Props fly above this height;
* motor mounts (forced solid): a disc of radius `mount_radius` and height `arm_height` at each corridor end, with
  the 4 motor screws (square pattern) as through voids; the motor sits on its top face and passes thrust,
  reaction torque, weight and inertia there;
* FC/ESC stack and battery: as in the frame v0 (screw bosses + void bay; 4 pads at the battery corners).

There are no separate arm screws any more: the arm-body joint is part of the optimised design, which is the whole
point (no bolted interface, no stress concentration at a screw). Aerodynamics (downwash drag on the arms) is NOT in
this v0: the arms are optimised for stiffness alone.

Screw centres are snapped to cell CENTRES (not nodes), so that a screw hole is one whole cell at h = 3 mm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from aeroptimizer.frame import MOTOR_SIGNS, Body, FrameProblem, FrameSetup, box_inertia
from aeroptimizer.grid import Masks


@dataclass
class DroneSetup(FrameSetup):
    L: float = 192.0  # box (x) [mm]: covers the 4 motor mounts
    W: float = 192.0
    H: float = 30.0
    h: float = 3.0
    volfrac: float = 0.15
    R: float = 4.5  # density filter radius [mm] = 1.5 cells
    boss_height: float = 6.0  # FC bosses [mm]
    fc_bay: tuple[float, float] = (6.0, 24.0)
    pad_height: float = 3.0
    hole_radius: float = 1.6
    # monolithic additions
    body: tuple[float, float] = (80.0, 76.0)  # footprint of the central body [mm]
    arm_width: float = 18.0
    arm_height: float = 12.0
    mount_radius: float = 15.0
    motor_screws: float = 18.0  # square pattern side [mm] (3 cells; 16 or 19 mm on a real motor)
    m_arm: float = 0.05  # motor + propeller + ESC wire etc. at the mount [kg]


class DroneProblem(FrameProblem):
    """Central body + 4 arms in one design domain (see module docstring)."""

    setup: DroneSetup

    def __init__(self, setup: DroneSetup | None = None, **kw) -> None:  # noqa: ANN003
        super().__init__(setup or DroneSetup(), **kw)

    # ------------------------------------------------------------------ geometry
    def cell_centre(self, v: float, axis: int) -> float:
        """Centre of the cell of axis x or y containing/nearest to v (the box is centred)."""
        h, half = self.setup.h, (self.setup.L, self.setup.W)[axis] / 2
        return (math.floor((v + half) / h)) * h + h / 2 - half

    def motors(self) -> list[tuple[float, float]]:
        s = self.setup
        r = s.wheelbase / 2 / math.sqrt(2)
        return [(self.cell_centre(sx * r, 0), self.cell_centre(sy * r, 1)) for sx, sy in MOTOR_SIGNS]

    def screws(self) -> dict[str, list[tuple[float, float]]]:
        """Screw centres (x, y) [mm]: the FC stack ('fc') and the 4 screws of each motor ('moteur<k>')."""
        s = self.setup
        out = {}
        for k, (mx, my) in enumerate(self.motors()):
            d = s.motor_screws / 2
            out[f"moteur{k}"] = [(self.cell_centre(mx + a * d, 0), self.cell_centre(my + b * d, 1)) for a, b in MOTOR_SIGNS]
        out["fc"] = [(self.cell_centre(sx * s.fc_pitch / 2, 0), self.cell_centre(sy * s.fc_pitch / 2, 1)) for sx, sy in MOTOR_SIGNS]
        return out

    def _build_masks(self) -> None:
        s = self.setup
        X, Y, Z = self.centres()
        mounts = self.motors()
        allowed = (np.abs(X) <= s.body[0] / 2) & (np.abs(Y) <= s.body[1] / 2)  # central body, full height
        for (sx, sy), (mx, my) in zip(MOTOR_SIGNS, mounts):  # arm corridors along the diagonals + motor discs
            ux, uy = sx / math.sqrt(2), sy / math.sqrt(2)
            along, across = X * ux + Y * uy, -X * uy + Y * ux
            band = (np.abs(across) <= s.arm_width / 2) & (along >= 0) & (along <= s.wheelbase / 2)
            disc = (X - mx) ** 2 + (Y - my) ** 2 <= s.mount_radius**2
            allowed |= (band | disc) & (Z < s.arm_height)
        void = ~allowed
        bay = (np.abs(X) <= s.fc_board / 2) & (np.abs(Y) <= s.fc_board / 2) & (Z > s.fc_bay[0]) & (Z < s.fc_bay[1])
        void |= bay
        self.zones: dict[str, np.ndarray] = {}
        for name, pts in self.screws().items():
            zone = np.zeros(self.shape, bool)
            for cx, cy in pts:
                d2 = (X - cx) ** 2 + (Y - cy) ** 2
                void |= d2 <= s.hole_radius**2  # screw holes through the whole height (tool access)
                if name == "fc":
                    zone |= (d2 <= s.boss_radius**2) & (Z < s.boss_height)
            if name.startswith("moteur"):
                mx, my = mounts[int(name[-1])]
                zone = ((X - mx) ** 2 + (Y - my) ** 2 <= s.mount_radius**2) & (Z < s.arm_height)
            self.zones[name] = zone
        bl, bw, _ = s.battery
        pads = np.zeros(self.shape, bool)
        for sx, sy in MOTOR_SIGNS:
            cx, cy = sx * (bl - s.pad) / 2, sy * (bw - s.pad) / 2
            pads |= (np.abs(X - cx) <= s.pad / 2) & (np.abs(Y - cy) <= s.pad / 2) & (Z > s.H - s.pad_height)
        self.zones["batterie"] = pads
        solid = np.zeros(self.shape, bool)
        for zone in self.zones.values():
            zone &= ~void
            solid |= zone
        self.masks = Masks(design=torch.from_numpy(~(solid | void)), solid=torch.from_numpy(solid),
                           void=torch.from_numpy(void))
        self._sector = tuple(slice(self.shape[a] // 2, None) if a in s.symmetry else slice(None) for a in range(3))

    # ------------------------------------------------------------------ supports and loads
    def _build_supports(self) -> None:
        """3-2-1 on the motor patches 0, 1, 3 (replaces the bolted arm patches of the frame)."""
        from aeroptimizer.fem.bc import Dirichlet, combine

        picks = []
        for zone in ("moteur0", "moteur1", "moteur3"):
            idx = self.interface_nodes(zone)
            c = self.xyz[idx].mean(axis=0)
            picks.append(idx[np.argmin(((self.xyz[idx] - c) ** 2).sum(1))])
        masks = []
        for node, comps in zip(picks, ((0, 1, 2), (1, 2), (2,))):
            m = torch.zeros(self.grid.n_nodes, dtype=torch.bool)
            m[int(node)] = True
            masks.append(Dirichlet(m, comps))
        self.support_nodes = picks
        self.fixed = combine(self.grid, 3, *masks)

    def bodies(self) -> list[Body]:
        s = self.setup
        mm = 1e-3
        out = []
        for k, (mx, my) in enumerate(self.motors()):
            pos = np.array([mx * mm, my * mm, s.arm_height * mm])
            out.append(Body(f"moteur{k}", s.m_arm, pos, np.zeros((3, 3)), self.interface_nodes(f"moteur{k}")))
        bl, bw, bh = s.battery
        out.append(Body("batterie", s.m_battery, np.array([0.0, 0.0, (s.H + bh / 2) * mm]),
                        box_inertia(s.m_battery, bl * mm, bw * mm, bh * mm), self.interface_nodes("batterie")))
        z0, z1 = s.fc_bay
        out.append(Body("fc", s.m_stack, np.array([0.0, 0.0, (z0 + z1) / 2 * mm]),
                        box_inertia(s.m_stack, s.fc_board * mm, s.fc_board * mm, (z1 - z0) * mm), self.interface_nodes("fc")))
        return out
