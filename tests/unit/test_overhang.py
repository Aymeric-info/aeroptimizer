"""Overhang (self-support) filter: C-OVH-01..09. Oracle: tests/reference/overhang_numpy.py."""

import math

import numpy as np
import pytest
import torch
from scipy import ndimage

from aeroptimizer.errors import ConfigError
from aeroptimizer.grid import Grid, Masks
from aeroptimizer.topopt.density import DensityPipeline
from aeroptimizer.topopt.overhang import (
    OverhangParams,
    check_solid_supported,
    overhang_filter,
    shifted_stack,
)
from reference.overhang_numpy import exact_support


def _random_field(shape, seed, density=0.6):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(shape, generator=g, dtype=torch.float64) < density).to(torch.float64)


def test_never_adds_material():
    r"""C-OVH-01 — rho_bar <= rho_hat + tol, on binary and on continuous fields, 2D and 3D."""
    for shape in ((10, 8), (5, 4, 6)):
        for seed in range(5):
            for params in (OverhangParams(), OverhangParams(beta_s=10.0), OverhangParams(beta_s=500.0, eps=1e-8)):
                g = torch.Generator().manual_seed(seed)
                for rho in (_random_field(shape, seed), torch.rand(shape, generator=g, dtype=torch.float64)):
                    out = overhang_filter(rho, params)
                    assert float((out - rho).max()) <= math.sqrt(params.eps) + 1e-12
                    assert float(out.min()) >= -1e-12  # the filter does not create negative densities either


def test_matches_numpy_reference():
    r"""C-OVH-02 — binary input, beta_s large: equals the exact support rule (explicit loops) to 1e-2.

    The smax bias log(n)/beta_s is paid once per layer, so beta_s must grow with the number of layers.
    """
    params = OverhangParams(beta_s=3000.0, eps=1e-10)
    for shape in ((12, 9), (7, 6, 8)):
        for seed in range(4):
            rho = _random_field(shape, seed)
            out = overhang_filter(rho, params).numpy()
            assert np.abs(out - exact_support(rho.numpy())).max() <= 1e-2


def test_binarised_output_is_self_supporting():
    r"""C-OVH-03 — the output binarised at 0.5 is already a fixed point of the exact rule (100 fields)."""
    params = OverhangParams(beta_s=50.0, eps=1e-10)
    for seed in range(100):
        shape = (9, 7) if seed % 2 == 0 else (5, 5, 6)
        g = torch.Generator().manual_seed(seed)
        rho = torch.rand(shape, generator=g, dtype=torch.float64)
        bin_out = (overhang_filter(rho, params) > 0.5).numpy()
        assert (exact_support(bin_out) == bin_out).all()


def test_stencil_cross_guarantees_45deg():
    r"""C-OVH-04 — 3 points in 2D, 5-point cross in 3D; reach per layer <= 1 voxel in every direction."""
    assert shifted_stack(torch.zeros(6), 2).shape == (6, 3)
    assert shifted_stack(torch.zeros(4, 5), 3).shape == (4, 5, 5)
    # one seed on the plate, everything above designed: the printed region is the reachable cone
    n, layers = 21, 8
    rho = np.ones((n, n, layers), dtype=bool)
    rho[..., 0] = False
    rho[n // 2, n // 2, 0] = True
    cone = exact_support(rho)
    ii, jj = np.meshgrid(np.arange(n) - n // 2, np.arange(n) - n // 2, indexing="ij")
    for ell in range(1, layers):
        reach = np.sqrt(ii**2 + jj**2)[cone[..., ell]].max()
        assert reach <= ell + 1e-12  # tan(theta) <= 1 along the axes AND along the diagonals
    # the 3x3 square would let the diagonal advance sqrt(2) per layer (54.7 degrees)
    square = np.zeros((n, n), bool)
    square[n // 2, n // 2] = True
    for _ in range(layers - 1):
        square = ndimage.binary_dilation(square, structure=np.ones((3, 3)))
    assert np.sqrt(ii**2 + jj**2)[square].max() == pytest.approx(math.sqrt(2) * (layers - 1))
    assert np.sqrt(ii**2 + jj**2)[cone[..., layers - 1]].max() == layers - 1


def test_baseplate_supported():
    r"""C-OVH-05 — layer 0 is the build plate: rho_bar[..., 0] == rho_hat[..., 0] exactly."""
    for shape in ((8, 5), (4, 4, 5)):
        rho = torch.rand(shape, dtype=torch.float64)
        assert torch.equal(overhang_filter(rho, OverhangParams())[..., 0], rho[..., 0])


def test_passive_solid_unsupported_rejected():
    r"""C-OVH-06 — a passive solid floating above void raises ConfigError (also via DensityPipeline)."""
    shape = (7, 5)
    solid = torch.zeros(shape, dtype=torch.bool)
    void = torch.zeros(shape, dtype=torch.bool)
    solid[3, 3] = True
    void[2:5, 2] = True  # nothing under (3,3) except void
    with pytest.raises(ConfigError, match="rest on"):
        check_solid_supported(solid, void)
    design = ~(solid | void)
    grid = Grid(shape, (1.0, 1.0))
    with pytest.raises(ConfigError):
        DensityPipeline(grid, Masks(design, solid, void), R=1.0, overhang=OverhangParams())
    check_solid_supported(solid, torch.zeros(shape, dtype=torch.bool))  # design cells below: accepted
    base = torch.zeros(shape, dtype=torch.bool)
    base[3, 0] = True
    check_solid_supported(base, torch.zeros(shape, dtype=torch.bool))  # on the plate: accepted


def test_gradients_finite_and_correct():
    r"""C-OVH-07 — finite gradient everywhere, also for rho_hat == 0; gradcheck 5x5 (2D) and 4x4x4 (3D)."""
    for shape in ((6, 6), (4, 4, 4)):
        for fill in (0.0, 1.0, 0.5):
            x = torch.full(shape, fill, dtype=torch.float64, requires_grad=True)
            overhang_filter(x, OverhangParams()).sum().backward()
            assert torch.isfinite(x.grad).all()
    g = torch.Generator().manual_seed(3)
    for shape in ((5, 5), (4, 4, 4)):
        x = torch.rand(shape, generator=g, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda a: overhang_filter(a, OverhangParams(beta_s=20.0)), (x,), atol=1e-6)


def test_build_axis_is_last():
    r"""C-OVH-08 — permuting the axes together with `build_axis` leaves the result unchanged."""
    params = OverhangParams(beta_s=30.0)
    for shape, perm in (((5, 4, 6), (2, 0, 1)), ((6, 5), (1, 0))):
        rho = torch.rand(shape, dtype=torch.float64)
        ref = overhang_filter(rho, params)
        rho_p = rho.permute(perm).contiguous()
        build = perm.index(len(shape) - 1)  # where the old last axis went
        out = overhang_filter(rho_p, params, build_axis=build)
        assert torch.allclose(out, ref.permute(perm), atol=1e-14)


def test_cross_layer_sensitivity():
    r"""C-OVH-09 — the top layer depends on every lower layer inside its support cone."""
    params = OverhangParams(beta_s=10.0)
    x = torch.full((21, 8), 0.7, dtype=torch.float64, requires_grad=True)
    top = overhang_filter(x, params)[10, -1]
    top.backward()
    for ell in range(0, 7):
        assert x.grad[10, ell].abs() > 1e-8, ell  # same column, every layer below
    assert x.grad[0, 0] == 0 and x.grad[20, 0] == 0  # outside the cone (radius 7 around column 10)
    assert x.grad[17, 0].abs() > 0 and x.grad[18, 0] == 0  # cone edge: exactly 45 degrees
