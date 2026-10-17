"""Density pipeline x -> rho_bar (FORMALISM §2, ARCHITECTURE §5.6, CONTRACTS §3.7)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from aeroptimizer.errors import ConfigError
from aeroptimizer.grid import Grid, Masks
from aeroptimizer.topopt.overhang import OverhangParams, check_solid_supported, overhang_filter


def simp(rho_bar: torch.Tensor, p: float, eps_s: float = 1e-9) -> torch.Tensor:
    """s = eps_s + (1 - eps_s) * rho_bar**p  (C-SIMP-01).

    `p >= 1` keeps s'(0) finite (p < 1 would give an infinite slope at rho = 0); `eps_s > 0` keeps
    K~ positive definite for void elements. Plain differentiable torch, no custom backward.
    """
    if p < 1.0:
        raise ConfigError(f"SIMP exponent p must be >= 1, got {p}")
    if not 0.0 < eps_s < 1.0:
        raise ConfigError(f"eps_s must be in (0, 1), got {eps_s}")
    return eps_s + (1.0 - eps_s) * rho_bar**p


def kappa(rho_bar: torch.Tensor, q: float, kappa_min: float = 1e-6) -> torch.Tensor:
    """Fluid conductivity kappa_min + (1 - kappa_min) * (1 - rho_bar)**q  (C-FLOW-06)."""
    raise NotImplementedError("phase 4")


def heaviside_projection(rho_tilde: torch.Tensor, beta: float, eta: float = 0.5) -> torch.Tensor:
    """Smooth Heaviside projection (C-PROJ-01/02), FORMALISM §2.3.

    (tanh(b*eta) + tanh(b*(r - eta))) / (tanh(b*eta) + tanh(b*(1 - eta))). It maps 0 to 0 and 1 to 1
    for every beta, tends to the identity when beta -> 0 (the formula is 0/0 at beta = 0, which is
    handled as the identity: top88 equivalence) and to a step at eta when beta -> infinity.
    """
    if beta < 0.0 or not 0.0 < eta < 1.0:
        raise ConfigError(f"need beta >= 0 and 0 < eta < 1, got beta={beta}, eta={eta}")
    if beta == 0.0:
        return rho_tilde
    t_eta = math.tanh(beta * eta)
    return (t_eta + torch.tanh(beta * (rho_tilde - eta))) / (t_eta + math.tanh(beta * (1.0 - eta)))


class DensityFilter(nn.Module):
    """Normalised conic convolution filter, radius R [m] (C-FILT-01..04).

    Kernel w(r) = max(0, 1 - |r| / R) sampled once on (2 r_a + 1) points per axis, r_a = ceil(R / h_a) - 1
    (so R <= h gives the identity); output = conv(rho, w) / conv(1, w), the denominator precomputed.
    Zero padding plus normalisation treats the borders without density bias. Differentiable (autograd).
    """

    def __init__(self, grid: Grid, R: float) -> None:
        super().__init__()
        if not R > 0.0:
            raise ConfigError(f"filter radius must be > 0, got {R}")
        self.grid, self.R = grid, float(R)
        radii = [max(math.ceil(R / h - 1e-12) - 1, 0) for h in grid.h]
        axes = [torch.arange(-r, r + 1, dtype=torch.get_default_dtype()) * h for r, h in zip(radii, grid.h)]
        dist = torch.sqrt(sum(a**2 for a in torch.meshgrid(*axes, indexing="ij")))
        kernel = torch.clamp(1.0 - dist / R, min=0.0)
        self.radii = tuple(radii)
        self.register_buffer("kernel", kernel[None, None])
        self.register_buffer("norm", self._conv(torch.ones(grid.shape)))

    def _conv(self, field: torch.Tensor) -> torch.Tensor:
        conv = F.conv2d if self.grid.d == 2 else F.conv3d
        return conv(field[None, None], self.kernel, padding=self.radii)[0, 0]

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        return self._conv(rho) / self.norm


class DensityPipeline(nn.Module):
    """x (grid shape) -> rho_bar (grid shape). beta is a mutable attribute (C-PROJ-03).

    rho = x on D, 1 on S, 0 on V (arithmetic masks, no torch.where: NaN-gradient trap), then density
    filter, Heaviside projection, the optional overhang filter (lesson 12) and the masks re-applied. The SIMP exponent p is not stored here: it belongs to the objective (`simp`).
    """

    def __init__(
        self,
        grid: Grid,
        masks: Masks,
        *,
        R: float,
        beta: float = 1.0,
        eta: float = 0.5,
        overhang: OverhangParams | None = None,
    ) -> None:
        super().__init__()
        if overhang is not None:
            check_solid_supported(masks.solid, masks.void)  # C-OVH-06
        self.overhang = overhang  # mutable: continuation changes overhang.beta_s
        dtype = torch.get_default_dtype()
        self.filter = DensityFilter(grid, R)
        self.beta, self.eta = float(beta), float(eta)  # plain floats: outside the graph, mutable
        self.register_buffer("m_D", masks.design.to(dtype))
        self.register_buffer("m_S", masks.solid.to(dtype))
        self.register_buffer("keep", (1.0 - masks.solid.to(dtype) - masks.void.to(dtype)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rho = x * self.m_D + self.m_S
        rho_hat = heaviside_projection(self.filter(rho), self.beta, self.eta)
        if self.overhang is not None:
            rho_hat = overhang_filter(rho_hat, self.overhang)
        return rho_hat * self.keep + self.m_S  # re-apply masks (C-MASK-01)
