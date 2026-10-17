"""Self-support (overhang) filter, layer-by-layer scan along the LAST axis (FORMALISM §2.4)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from aeroptimizer.errors import ConfigError


@dataclass
class OverhangParams:
    beta_s: float = 50.0  # log-sum-exp sharpness (mutable: continuation)
    eps: float = 1.0e-4  # smooth-min regulariser
    # Only theta = 45 deg is exactly representable with a one-voxel stencil (FORMALISM H5).
    theta_deg: float = 45.0

    def __post_init__(self) -> None:
        if self.theta_deg != 45.0:
            raise ConfigError(f"only theta = 45 deg is supported (one-voxel stencil), got {self.theta_deg}")
        if not self.beta_s > 0.0 or not self.eps > 0.0:
            raise ConfigError(f"beta_s and eps must be > 0, got {self.beta_s}, {self.eps}")


def shifted_stack(layer: torch.Tensor, d: int) -> torch.Tensor:
    """Stack the supporting-stencil neighbours of a layer: (*lateral, n_stencil), zero padded at borders.

    2D (layer is 1-D): 3 points {-1,0,+1}.  3D (layer is 2-D): 5-point CROSS {0,+-x,+-y} (C-OVH-04),
    never the 3x3 square (diagonal overhang would reach ~54.7 deg).
    """
    if layer.ndim != d - 1 or d not in (2, 3):
        raise ConfigError(f"a layer of a {d}D grid has {d - 1} dimensions, got {layer.ndim}")
    if d == 2:
        p = F.pad(layer, (1, 1))
        n = layer.shape[0]
        return torch.stack([p[0:n], p[1 : n + 1], p[2 : n + 2]], dim=-1)
    p = F.pad(layer, (1, 1, 1, 1))
    nx, ny = layer.shape
    centre = p[1 : nx + 1, 1 : ny + 1]
    return torch.stack(
        [centre, p[0:nx, 1 : ny + 1], p[2 : nx + 2, 1 : ny + 1], p[1 : nx + 1, 0:ny],
         p[1 : nx + 1, 2 : ny + 2]],
        dim=-1,
    )


def smax(y: torch.Tensor, beta_s: float) -> torch.Tensor:
    """Smooth max over the last axis: LSE_beta(y) - log(n)/beta. Bias in [-log(n)/beta, 0]."""
    n = y.shape[-1]
    return torch.logsumexp(beta_s * y, dim=-1) / beta_s - math.log(n) / beta_s


def smin(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    """Smooth min of two fields: 1/2 (a + b - sqrt((a-b)^2 + eps) + sqrt(eps)). Gradient finite for eps > 0."""
    return 0.5 * (a + b - torch.sqrt((a - b) ** 2 + eps) + math.sqrt(eps))


def overhang_filter(
    rho_hat: torch.Tensor, params: OverhangParams, build_axis: int = -1
) -> torch.Tensor:
    """rho_bar = scan over layers l=1..nz-1 of
         Xi_l   = LSE_beta_s(neighbours of xi_{l-1}) - log(n)/beta_s
         xi_l   = smin_eps(rho_hat_l, Xi_l)            with xi_0 = rho_hat_0 (build plate)
    Guarantees rho_bar <= rho_hat (+ tolerance), C-OVH-01..09. Differentiable by plain Autograd.

    The build axis is the last one by convention (C-CONV-01); `build_axis` lets a caller with another
    layout say so, the scan then runs along it (C-OVH-08).
    """
    d = rho_hat.ndim
    field = torch.movedim(rho_hat, build_axis, -1)
    layers = [field[..., 0]]  # build plate: always supported (C-OVH-05)
    for ell in range(1, field.shape[-1]):
        support = smax(shifted_stack(layers[-1], d), params.beta_s)
        layers.append(smin(field[..., ell], support, params.eps))
    return torch.movedim(torch.stack(layers, dim=-1), -1, build_axis)


def stencil_offsets(d: int) -> list[tuple[int, ...]]:
    """Lateral offsets of the supporting stencil (the same cells as `shifted_stack`), for reference code."""
    if d == 2:
        return [(-1,), (0,), (1,)]
    if d == 3:
        return [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    raise ConfigError(f"d must be 2 or 3, got {d}")


def check_solid_supported(solid: torch.Tensor, void: torch.Tensor) -> None:
    """C-OVH-06: a passive solid voxel above the plate needs a non-void cell under it in its stencil.

    The overhang filter would remove it, yet the masks put it back (§2.5): such a part would hang in
    the air. Cells of the design domain count as possible support (the optimiser may keep them);
    this is a check of the configuration, not a guarantee about the final shape.
    """
    d = solid.ndim
    available = (~void).to(torch.float64)
    for ell in range(1, solid.shape[-1]):
        under = shifted_stack(available[..., ell - 1], d).amax(dim=-1) > 0
        bad = solid[..., ell] & ~under
        if bool(bad.any()):
            where = tuple(int(i) for i in torch.nonzero(bad)[0]) + (ell,)
            raise ConfigError(f"passive solid at cell {where} has nothing to rest on (overhang)")
