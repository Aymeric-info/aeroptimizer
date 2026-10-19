"""Navier-Stokes-Brinkman solver (`flow/brinkman.py`): exact solutions, a benchmark, the adjoint."""

import math

import numpy as np
import pytest
import torch

from aeroptimizer.arm import cell_fraction
from aeroptimizer.flow.brinkman import BrinkmanNS, brinkman_alpha, stretched_axis


def _channel(a0, a1, *, walls: str, inlet_profile):
    """Flow towards -axis1, inlet on the top face, outlet (do nothing) at the bottom."""
    nodes = np.stack(np.meshgrid(a0, a1, indexing="ij"), -1).reshape(-1, 2)
    fixed = torch.zeros(len(nodes), 3, dtype=torch.bool)
    vals = torch.zeros(len(nodes), 3)
    side = torch.from_numpy(np.isclose(nodes[:, 0], a0[0]) | np.isclose(nodes[:, 0], a0[-1]))
    inlet = torch.from_numpy(np.isclose(nodes[:, 1], a1[-1]))
    fixed[side, 0] = True
    if walls == "noslip":
        fixed[side, 1] = True
    fixed[inlet, 0] = fixed[inlet, 1] = True
    vals[inlet & ~(side if walls == "noslip" else torch.zeros_like(side)), 1] = torch.from_numpy(
        -inlet_profile(nodes[:, 0])
    )[inlet & ~(side if walls == "noslip" else torch.zeros_like(side))]
    return fixed.reshape(-1), vals.reshape(-1)


def test_stretched_axis():
    a = stretched_axis((0.0, 1.0), 0.1, -2.0, 3.0, 1.2)
    assert a[0] == -2.0 and a[-1] == 3.0 and np.all(np.diff(a) > 0)
    core = a[(a >= 0) & (a <= 1.0 + 1e-12)]
    assert np.allclose(np.diff(core), 0.1)


def test_uniform_stream_is_exact():
    """Empty box, slip sides: the uniform stream (p = 0) solves the discrete equations."""
    a0, a1 = np.linspace(0, 1, 9), stretched_axis((0.0, 1.0), 0.125, -1.0, 2.0)
    fixed, vals = _channel(a0, a1, walls="slip", inlet_profile=lambda y: np.ones_like(y))
    ns = BrinkmanNS((a0, a1), 50.0, fixed, vals)
    sol = ns.newton(torch.zeros(ns.grid.n_elems))
    u0, u1, p = sol.fields(ns.shape)
    assert np.allclose(u0, 0, atol=1e-10) and np.allclose(u1, -1, atol=1e-10) and np.allclose(p, 0, atol=1e-10)


def test_cylinder_benchmark_coarse():
    """Schafer & Turek (1996) 2D-1, Re = 20: C_D = 5.5795. Coarse Brinkman mesh (D/h = 10): within 25 %."""
    h = 0.41 / 40
    a0 = np.linspace(0, 0.41, 41)
    a1 = stretched_axis((2.2 - 40 * h, 2.2), h, 0.0, 2.2, 1.06)
    fixed, vals = _channel(a0, a1, walls="noslip", inlet_profile=lambda y: 4 * 0.3 * y * (0.41 - y) / 0.41**2)
    ns = BrinkmanNS((a0, a1), 1000.0, fixed, vals)  # mu = 1e-3, U_mean = 0.2, D = 0.1
    c = ns.centres.numpy()
    rho = cell_fraction(lambda Y, Z: (Y - 0.2) ** 2 + (Z - 2.0) ** 2 <= 0.05**2, c[:, 0], c[:, 1], h, h)
    # (the cylinder lies in the uniform core, where cells are h x h)
    alpha = brinkman_alpha(rho, 1e5, q=1.0)
    U = ns.solve(alpha)
    CD = -float(ns.force(U, alpha)[1]) / (0.5 * 0.2**2 * 0.1)
    assert 5.5795 * 0.95 < CD < 5.5795 * 1.25


def test_adjoint_gradient():
    """d(drag)/d(alpha) through Newton + adjoint == central finite differences."""
    a0 = np.linspace(0, 1, 9)
    a1 = stretched_axis((0.0, 1.0), 0.125, -0.5, 1.5)
    fixed, vals = _channel(a0, a1, walls="slip", inlet_profile=lambda y: np.ones_like(y))
    ns = BrinkmanNS((a0, a1), 20.0, fixed, vals)
    c = ns.centres
    blob = torch.exp(-(((c[:, 0] - 0.3) ** 2 + (c[:, 1] - 0.5) ** 2) / 0.05))
    rho = (0.8 * blob).requires_grad_()

    def drag(r):
        a = brinkman_alpha(r, 50.0, 0.5)
        return -ns.force(ns.solve(a), a)[1]

    (g,) = torch.autograd.grad(drag(rho), rho)
    d, e = torch.randn_like(rho), 1e-6
    with torch.no_grad():
        fd = (drag(rho.detach() + e * d) - drag(rho.detach() - e * d)) / (2 * e)
    assert math.isclose(float(g @ d), float(fd), rel_tol=1e-6)


@pytest.mark.parametrize("q", [0.1, 1.0])
def test_alpha_interpolation(q):
    r = torch.linspace(0, 1, 11)
    a = brinkman_alpha(r, 100.0, q)
    assert float(a[0]) == 0.0 and math.isclose(float(a[-1]), 100.0) and bool((a[1:] > a[:-1]).all())
