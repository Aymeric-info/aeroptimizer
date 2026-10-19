"""Beam-section stiffnesses (`section.py`): analytic values, disconnection penalty, gradients."""

import math

import numpy as np
import torch

from aeroptimizer.arm import cell_fraction
from aeroptimizer.section import ArmLoads, BeamSection
from aeroptimizer.topopt.density import simp


def _section(ny: int, nz: int, h: float) -> BeamSection:
    return BeamSection((ny, nz), (h, h))


def test_rectangle_bending_shear_torsion():
    """1 x 2 rectangle: EI = b h^3 / 12, kappa = 5/6 exactly, GJ = 0.229 b^3 h (Saint-Venant table)."""
    n = 40
    sec = _section(n, 2 * n, 1.0 / n)
    one = torch.ones(2 * n * n)
    s = sec.stiffness(one, one)
    assert math.isclose(float(s.EI_y), 2.0**3 / 12, rel_tol=1e-12)
    assert math.isclose(float(s.EI_z), 2.0 / 12, rel_tol=1e-12)
    assert math.isclose(float(s.S_z), 1.0 / (5.0 / 6.0 * 2.0), rel_tol=1e-3)
    assert math.isclose(float(s.GJ), 0.2287 * 2.0, rel_tol=5e-3)


def test_ellipse_torsion():
    """GJ = pi a^3 b^3 / (a^2 + b^2) for a solid ellipse (G = 1)."""
    n = 80
    sec = _section(n, n, 1.0 / n)
    a, b = 0.45, 0.25
    rho = cell_fraction(lambda Y, Z: (Y / b) ** 2 + ((Z - 0.5) / a) ** 2 <= 1, sec.y.numpy(), sec.z.numpy(), 1 / n, 1 / n)
    E = simp(rho, 1.0, 1e-9)  # void cells must keep a tiny stiffness
    s = sec.stiffness(E, E)
    assert math.isclose(float(s.GJ), math.pi * a**3 * b**3 / (a**2 + b**2), rel_tol=2e-3)


def test_disconnected_flanges_are_penalised():
    """Two flanges without a web: EI barely changes, but shear warping cannot balance -> S explodes."""
    n = 30
    sec = _section(n, 2 * n, 1.0 / n)
    y, z = sec.y.numpy(), sec.z.numpy()
    flanges = (z < 0.6) | (z > 1.4)
    for rho, expect_huge in ((flanges, True), (flanges | (np.abs(y) < 0.05), False)):
        E = simp(torch.from_numpy(rho.astype(float)), 1.0, 1e-6)
        s = sec.stiffness(E, E)
        assert (float(s.S_z) > 1e3) == expect_huge


def test_gradients_match_finite_differences():
    n = 16
    sec = _section(n, n, 1.0 / n)
    loads = ArmLoads(L=10.0)

    def f(r):
        E = simp(r, 3.0, 1e-3)
        c = loads.compliances(sec.stiffness(E, 0.4 * E))
        return c["C_T"] + c["C_L"] + c["C_t"]

    rho = (0.3 + 0.4 * torch.rand(n * n)).requires_grad_()
    (g,) = torch.autograd.grad(f(rho), rho)
    d, e = torch.randn(n * n), 1e-6
    fd = (f(rho.detach() + e * d) - f(rho.detach() - e * d)) / (2 * e)
    assert math.isclose(float(g @ d), float(fd), rel_tol=1e-5)
