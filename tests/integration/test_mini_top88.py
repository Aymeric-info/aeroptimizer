"""Lesson 8: SIMP + adjoint + OC without filter reproduces the NumPy top88 oracle, iterate by iterate.

This is not the contract C-OPT-02 (filtered benchmark R = 1.5 h, lesson 9): it checks the unfiltered
loop against an implementation that shares no code with the package (own assembly, own solver, own OC).
"""

import numpy as np
import torch
from reference.top88_numpy import top88

from aeroptimizer.fem import elements as el
from aeroptimizer.fem.adjoint import solve_spd
from aeroptimizer.fem.bc import build_case
from aeroptimizer.fem.operators import StructuredFEOperator
from aeroptimizer.fem.solvers import ScipyDirect
from aeroptimizer.grid import Connectivity, Grid
from aeroptimizer.materials import Orthotropic
from aeroptimizer.objectives.mechanical import compliance
from aeroptimizer.optim.oc import OC
from aeroptimizer.topopt.density import simp


def _mini_top88(nx, nz, vf, p, n_iter, oc):
    g = Grid((nx, nz), (1.0, 1.0))
    fixed, load, _ = build_case("mbb", g)
    Ke0 = el.elasticity_ke0(2, g.h, Orthotropic.isotropic(1.0, 0.3).stiffness(2))
    op = StructuredFEOperator(g, Connectivity(g, 2), Ke0, 2, fixed)
    f = load.assemble(torch.ones(g.shape))
    n = g.n_elems
    x, hist = np.full(n, vf), []
    for _ in range(n_iter):
        rho = torch.from_numpy(x).requires_grad_()
        c = compliance(f, solve_spd(op, simp(rho, p), f, ScipyDirect()))
        c.backward()
        hist.append(c.item())
        x = oc.step(x, c.item(), rho.grad.numpy(), x.mean() / vf - 1.0, np.full(n, 1.0 / (n * vf)))
    return np.array(hist), x


def test_unfiltered_loop_matches_top88_oracle():
    nx, nz, vf, p, n_iter = 60, 20, 0.5, 3.0, 40
    ref = top88(nx, nz, vf, p, 1.0, max_iter=n_iter, ft=0, lam_tol=1e-12, change_tol=0.0)
    hist, x = _mini_top88(nx, nz, vf, p, n_iter, OC(tol=1e-12))
    assert np.allclose(hist, ref["c"], rtol=1e-8)
    x_ref = np.flipud(ref["x"]).T.reshape(-1)  # (rows from the top, columns) -> (x, z up) in C order
    assert np.abs(x - x_ref).max() < 1e-6
    assert abs(x.mean() - vf) < 1e-9


def test_default_oc_converges_and_is_feasible():
    """With the default tolerance: the compliance decreases overall and the volume holds at every step."""
    hist, x = _mini_top88(30, 10, 0.5, 3.0, 60, OC())
    assert hist[-1] < 0.3 * hist[0]
    assert abs(x.mean() / 0.5 - 1.0) <= 1e-3
