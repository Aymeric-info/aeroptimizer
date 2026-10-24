"""v0 drone frame (`frame.py`): self-equilibrated loads, isostatic supports, symmetry, end-to-end gradients."""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from aeroptimizer.fem import elements as el
from aeroptimizer.fem.adjoint import compliances, solve_spd
from aeroptimizer.fem.bc import Dirichlet, combine, rigid_modes
from aeroptimizer.fem.operators import StructuredFEOperator
from aeroptimizer.fem.solvers import AMGCG, ScipyDirect
from aeroptimizer.frame import Body, FrameProblem, FrameSetup, LOAD_CASES, patch_forces, rigid_body_loads
from aeroptimizer.grid import Connectivity
from aeroptimizer.topopt.overhang import OverhangParams

SMALL = FrameSetup(L=80.0, W=80.0, H=28.0, h=4.0, R=6.0, pad_height=4.0)


@pytest.fixture(scope="module")
def small():
    torch.set_default_dtype(torch.float64)
    return FrameProblem(SMALL, solver=ScipyDirect())


def test_patch_forces_reproduce_the_wrench():
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(12, 3))
    F, M, about = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
    f = patch_forces(xyz, F, M, about)
    assert np.allclose(f.sum(0), F)
    assert np.allclose(np.cross(xyz - about, f).sum(0), M)


def test_hover_has_no_acceleration_and_loads_balance():
    bodies = [Body("a", 0.3, np.array([0.1, 0.0, 0.0]), np.eye(3) * 1e-4, np.zeros(1, int)),
              Body("b", 0.3, np.array([-0.1, 0.0, 0.0]), np.eye(3) * 1e-4, np.zeros(1, int))]
    T = 0.3 * 9.81
    forces, moments, a, alpha = rigid_body_loads(bodies, [np.array([0, 0, T])] * 2, [np.zeros(3)] * 2)
    assert np.allclose(a, 0) and np.allclose(alpha, 0)
    assert np.allclose(forces, 0)  # each thrust carries its own weight: nothing passes to the frame


def test_load_cases_are_self_equilibrated(small):
    R = small.resultant()
    scale = np.abs(small.f.numpy()).sum(0)[:, None]
    assert np.all(np.abs(R) <= 1e-10 * scale * max(SMALL.L, SMALL.W))
    # equivalently: no work on any rigid-body motion, so the 3-2-1 supports carry zero reaction
    assert torch.allclose(rigid_modes(small.grid).T @ small.f, torch.zeros(6, len(LOAD_CASES)), atol=1e-9)


def test_compliance_does_not_depend_on_the_supports(small):
    w = torch.full((small.grid.n_elems,), 0.5)
    C1, u1 = compliances(small.op, w, small.f, ScipyDirect())
    # move the supports: another, equally valid, 3-2-1 set
    n = small.grid.n_nodes
    picks = [small.interface_nodes(z)[0] for z in ("bras2", "bras3", "bras1")]
    ms = []
    for node, comps in zip(picks, ((0, 1, 2), (1, 2), (2,))):
        m = torch.zeros(n, dtype=torch.bool)
        m[int(node)] = True
        ms.append(Dirichlet(m, comps))
    fixed = combine(small.grid, 3, *ms)
    op2 = StructuredFEOperator(small.grid, Connectivity(small.grid, 3),
                               el.elasticity_ke0(3, small.grid.h, SMALL.material.stiffness(3)), 3, fixed)
    C2, _ = compliances(op2, w, small.f, ScipyDirect())
    assert torch.allclose(C1, C2, rtol=1e-8)


def test_masks_are_mirror_symmetric_and_printable(small):
    m = small.masks
    for t in (m.solid, m.void):
        assert torch.equal(t, t.flip(0)) and torch.equal(t, t.flip(1))
    for zone in ("bras0", "fc", "batterie"):
        assert small.interface_nodes(zone).size > 0
    FrameProblem(replace(SMALL, overhang=OverhangParams()), solver=ScipyDirect(),
                 refs=small.refs)  # check_solid_supported (C-OVH-06) passes


def test_rho_bar_is_mirror_symmetric(small):
    rho = small.rho_bar(torch.rand(small.n_design))
    assert torch.allclose(rho, rho.flip(0)) and torch.allclose(rho, rho.flip(1))


def test_self_adjoint_compliances_match_the_generic_adjoint(small):
    w = (0.2 + torch.rand(small.grid.n_elems)).requires_grad_()
    C, _ = compliances(small.op, w, small.f, ScipyDirect())
    (g1,) = torch.autograd.grad(C.sum() + 2 * C[1], w)
    w2 = w.detach().clone().requires_grad_()
    u = solve_spd(small.op, w2, small.f, ScipyDirect())
    C2 = (small.f * u).sum(0)
    (g2,) = torch.autograd.grad(C2.sum() + 2 * C2[1], w2)
    assert torch.allclose(C, C2) and torch.allclose(g1, g2, rtol=1e-9, atol=1e-12)


def test_amg_matches_the_direct_solver(small):
    w = 0.1 + torch.rand(small.grid.n_elems)
    C1, _ = compliances(small.op, w, small.f, ScipyDirect())
    C2, _ = compliances(small.op, w, small.f, AMGCG(tol=1e-10))
    assert torch.allclose(C1, C2, rtol=1e-7)


@pytest.mark.parametrize("overhang", [False, True])
def test_frame_gradients(small, overhang):
    pb = FrameProblem(replace(SMALL, overhang=OverhangParams() if overhang else None), solver=ScipyDirect(),
                      refs=small.refs)
    pb.set_continuation({"p": 3.0, "beta": 2.0, "beta_s": 10.0})
    x0 = torch.from_numpy(pb.initial_design()) + 0.1 * torch.rand(pb.n_design)
    d, e = torch.randn_like(x0), 1e-5
    for key in ("C_hat", "C_lacet", "vol"):
        x = x0.clone().requires_grad_()
        (g,) = torch.autograd.grad(pb.evaluate(x).terms[key], x)
        with torch.no_grad():
            fd = (pb.evaluate(x0 + e * d).terms[key] - pb.evaluate(x0 - e * d).terms[key]) / (2 * e)
        assert math.isclose(float(g @ d), float(fd), rel_tol=1e-5), key
