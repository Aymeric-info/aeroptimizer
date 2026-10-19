"""End-to-end gradients of the v0 arm problems (`arm.py`): 2D section x Navier-Stokes, 3D arm x proxy."""

import math

import torch

from aeroptimizer.arm import Arm3DProblem, Arm3DSetup, SectionProblem, SectionSetup


def _check(pb, keys, e=1e-5):
    pb.set_continuation({"p": 3.0, "beta": 2.0, "beta_s": 1.0})
    x0 = torch.from_numpy(pb.initial_design()) + 0.1 * torch.rand(pb.n_design)
    d = torch.randn_like(x0)
    for key in keys:
        x = x0.clone().requires_grad_()
        (g,) = torch.autograd.grad(pb.evaluate(x).terms[key], x)
        with torch.no_grad():
            fd = (pb.evaluate(x0 + e * d).terms[key] - pb.evaluate(x0 - e * d).terms[key]) / (2 * e)
        assert math.isclose(float(g @ d), float(fd), rel_tol=1e-5), key


def test_section_problem_gradients():
    setup = SectionSetup(W=8.0, H=12.0, h=1.0, R=1.5, flow_box=(2.0, 1.5, 2.5, 0.25), stretch=1.3)
    pb = SectionProblem(setup, mode="eps", c_max=1.0)
    assert pb.refs["C"] > 0 and pb.refs["D"] > 0
    _check(pb, ("C", "D", "area"))
    res = pb.evaluate(torch.from_numpy(pb.initial_design()))
    assert len(res.g) == 2  # area and stiffness constraints


def test_section_is_mirror_symmetric():
    pb = SectionProblem(SectionSetup(W=8.0, H=12.0, h=1.0, R=1.5, flow_box=(2.0, 1.5, 2.5, 0.25)))
    rho = pb.rho_bar(torch.rand(pb.n_design))
    assert torch.allclose(rho, rho.flip(0))


def test_arm3d_gradients():
    pb = Arm3DProblem(Arm3DSetup(L=48.0, W=16.0, H=12.0, h=4.0, mount=8.0, rotor_radius=40.0, hub_radius=6.0))
    _check(pb, ("C_T", "C_L", "C_t", "D", "vol"), e=1e-4)
