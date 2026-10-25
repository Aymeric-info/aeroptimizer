"""Monolithic drone (`drone.py`): passive zones, forced voids, self-equilibrated loads, gradients."""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from aeroptimizer.drone import DroneProblem, DroneSetup
from aeroptimizer.fem.bc import rigid_modes
from aeroptimizer.fem.solvers import ScipyDirect
from aeroptimizer.frame import LOAD_CASES
from aeroptimizer.topopt.overhang import OverhangParams

# coarse: 6 mm cells (box 192 x 192 x 30), enough for the zones to exist, direct solver stays cheap
SMALL = DroneSetup(h=6.0, R=9.0, boss_radius=7.0, hole_radius=3.1, pad_height=6.0, boss_height=6.0, fc_bay=(6.0, 24.0),
                   arm_width=18.0, mount_radius=15.0, motor_screws=12.0, pad=12.0)


@pytest.fixture(scope="module")
def small():
    torch.set_default_dtype(torch.float64)
    return DroneProblem(SMALL, solver=ScipyDirect())


def test_everything_outside_body_and_arms_is_void(small):
    X, Y, Z = small.centres()
    s = small.setup
    far = (np.abs(X) > s.body[0] / 2) & (np.abs(Y) > s.body[1] / 2)  # the four corners of the box
    ax = np.abs(np.abs(X) - np.abs(Y)) > s.arm_width  # away from the diagonals
    assert small.masks.void.numpy()[far & ax].all()
    assert small.masks.void.numpy()[Z > s.arm_height][(np.abs(X) > s.body[0] / 2)[Z > s.arm_height]].all()  # arms are low


def test_zones_are_passive_solid_and_do_not_overlap_voids(small):
    m = small.masks
    for name, zone in small.zones.items():
        assert zone.any(), name
        assert m.solid.numpy()[zone].all() and not m.void.numpy()[zone].any()
    assert {f"moteur{k}" for k in range(4)} | {"fc", "batterie"} == set(small.zones)


def test_motor_screw_holes_are_voids_through_the_mount(small):
    for k in range(4):
        zone = small.zones[f"moteur{k}"]
        for cx, cy in small.screws()[f"moteur{k}"]:
            X, Y, _ = small.centres()
            hole = ((X - cx) ** 2 + (Y - cy) ** 2) <= small.setup.hole_radius**2
            assert small.masks.void.numpy()[hole].all()


def test_masks_are_mirror_symmetric_and_printable(small):
    m = small.masks
    for t in (m.solid, m.void):
        assert torch.equal(t, t.flip(0)) and torch.equal(t, t.flip(1))
    DroneProblem(replace(SMALL, overhang=OverhangParams()), solver=ScipyDirect(), refs=small.refs)  # C-OVH-06


def test_load_cases_are_self_equilibrated(small):
    R = small.resultant()
    scale = np.abs(small.f.numpy()).sum(0)[:, None]
    assert np.all(np.abs(R) <= 1e-10 * scale * max(SMALL.L, SMALL.W))
    assert torch.allclose(rigid_modes(small.grid).T @ small.f, torch.zeros(6, len(LOAD_CASES)), atol=1e-9)


def test_loads_enter_through_the_motor_mounts_battery_and_stack(small):
    f = small.f.numpy().reshape(-1, 3, len(small.cases))
    loaded = np.flatnonzero(np.abs(f).sum((1, 2)) > 0)
    expected = np.unique(np.concatenate([small.interface_nodes(z) for z in small.zones]))
    assert set(loaded) <= set(expected.tolist())


@pytest.mark.parametrize("overhang", [False, True])
def test_drone_gradients(small, overhang):
    pb = DroneProblem(replace(SMALL, overhang=OverhangParams() if overhang else None), solver=ScipyDirect(),
                      refs=small.refs)
    pb.set_continuation({"p": 3.0, "beta": 2.0, "beta_s": 10.0})
    x0 = torch.from_numpy(pb.initial_design()) + 0.1 * torch.rand(pb.n_design)
    d, e = torch.randn_like(x0), 1e-5
    for key in ("C_hat", "vol"):
        x = x0.clone().requires_grad_()
        (g,) = torch.autograd.grad(pb.evaluate(x).terms[key], x)
        with torch.no_grad():
            fd = (pb.evaluate(x0 + e * d).terms[key] - pb.evaluate(x0 - e * d).terms[key]) / (2 * e)
        assert math.isclose(float(g @ d), float(fd), rel_tol=1e-5), key
