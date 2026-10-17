"""Compliance C = f^T u (FORMALISM §3.6): energy identity and the sign property OC relies on."""

import pytest
import torch

from aeroptimizer.fem import elements as el
from aeroptimizer.fem.adjoint import solve_spd
from aeroptimizer.fem.bc import build_case
from aeroptimizer.fem.operators import StructuredFEOperator
from aeroptimizer.fem.solvers import ScipyDirect
from aeroptimizer.grid import Connectivity, Grid
from aeroptimizer.materials import Orthotropic
from aeroptimizer.objectives.mechanical import compliance
from aeroptimizer.problem import ComplianceProblem
from aeroptimizer.topopt.density import simp


def _mbb(nx=16, nz=8):
    g = Grid((nx, nz), (1.0, 1.0))
    fixed, load, _ = build_case("mbb", g)
    Ke0 = el.elasticity_ke0(2, g.h, Orthotropic.isotropic(1.0, 0.3).stiffness(2))
    return StructuredFEOperator(g, Connectivity(g, 2), Ke0, 2, fixed), load.assemble(torch.ones(g.shape))


def test_compliance_is_twice_the_strain_energy():
    """C = f^T u = u^T K~ u > 0 (the load is on free dofs, u vanishes on the fixed ones)."""
    op, f = _mbb()
    rho = 0.1 + 0.9 * torch.rand(op.grid.n_elems, generator=torch.Generator().manual_seed(0))
    w = simp(rho, 3.0)
    u = solve_spd(op, w, f, ScipyDirect())
    c = compliance(f, u)
    assert c.item() > 0
    assert abs(c.item() - torch.dot(u, op.matvec(w, u)).item()) <= 1e-9 * c.item()


def test_more_material_never_increases_compliance():
    """dC/drho_e <= 0 for every element (adjoint, lambda = u): the sign OC's update is built on."""
    op, f = _mbb()
    rho = (0.1 + 0.9 * torch.rand(op.grid.n_elems, generator=torch.Generator().manual_seed(2))).requires_grad_()
    c = compliance(f, solve_spd(op, simp(rho, 3.0), f, ScipyDirect()))
    c.backward()
    assert bool((rho.grad <= 0).all()) and float(rho.grad.min()) < 0


def test_anisotropic_cantilever_matches_beam_theory():
    r"""Lesson 11 — `ComplianceProblem(C0=...)` : poutre console imprimée à plat / debout.

    Euler-Bernoulli: the tip deflection scales with 1/E along the beam axis, so the ratio of
    works is E_z/E_p up to a few per cent (shear and the finite mesh).
    """
    import math

    from aeroptimizer.materials import Orthotropic, rotation_xz

    ti = Orthotropic.transversely_isotropic(Ep=1.0, Ez=0.75, nup=0.35, nupz=0.35, Gz=0.29)
    work = {}
    for name, C0 in (("flat", ti.stiffness(2)), ("up", ti.rotated_stiffness(rotation_xz(math.pi / 2), 2))):
        pb = ComplianceProblem("cantilever", (60, 6), volfrac=1.0, R=1.0, C0=C0)
        work[name] = float(pb.evaluate(torch.ones(pb.grid.n_elems)).terms["C"])
    assert work["flat"] / work["up"] == pytest.approx(0.75, rel=0.03)
    default = ComplianceProblem("cantilever", (60, 6), volfrac=1.0, R=1.0)
    assert torch.allclose(default.C0, Orthotropic.isotropic(1.0, 0.3).stiffness(2))
