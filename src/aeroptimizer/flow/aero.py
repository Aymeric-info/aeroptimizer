"""Pressure force on the diffuse solid boundary and the download surrogate D (FORMALISM §4.5-4.7).

CAUTION (FORMALISM §4.4): potential flow alone gives zero drag (d'Alembert). D is a SHAPE INDICATOR
built from Bernoulli pressure + a constant base-pressure closure c_b, not a force prediction.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def bernoulli_pressure(v: torch.Tensor, rho_air: float, u_ref: float) -> torch.Tensor:
    """p - p_inf = 0.5 * rho_a * (u_ref**2 - |v|**2), element-wise (n_elems,)."""
    return 0.5 * rho_air * (u_ref**2 - (v**2).sum(dim=-1))


def density_gradient(rho_bar: torch.Tensor, h: tuple[float, ...]) -> torch.Tensor:
    """Centred finite-difference gradient of rho_bar, shape (*grid, d), unbiased at borders.

    `rho_bar` has the grid shape. Interior: (rho[i+1] - rho[i-1]) / 2h; the two border layers use the
    one-sided difference (a zero padding would invent a fake interface at the edge of the domain).
    """
    if rho_bar.ndim != len(h):
        raise ValueError(f"rho_bar must have the grid shape (d={len(h)} axes), got {tuple(rho_bar.shape)}")
    return torch.stack(torch.gradient(rho_bar, spacing=h, edge_order=1), dim=-1)


def fluid_side_pressure(
    p_rel: torch.Tensor, rho_bar: torch.Tensor, *, m: float = 2.0, k: int = 1, eps: float = 1e-2,
    rho_c: float = 1.0,
) -> torch.Tensor:
    """Pressure re-estimated from the fluid side of the interface (lesson 15; NOT in FORMALISM).

    The cells that carry the weight |grad rho_bar| of the force are interface cells, where the FE velocity
    is biased (kappa jumps across them). Replace p_e by the average of p over the (2k+1)^d neighbourhood,
    weighted by w = max(1 - rho_bar / rho_c, 0)^m (rho_c = 1: (1 - rho_bar)^m): solid cells then hardly count (with rho_c < 1 they
    do not count at all above rho_c). The average is
    sum(w p) / (sum(w) + eps): where there is no fluid around (an interface cell inside a solid region,
    where the FE velocity is meaningless) the estimate goes smoothly to 0 instead of averaging garbage
    (lesson 17: an optimiser finds and exploits such cells). Differentiable; keeps the shape of `p_rel`.
    """
    d = rho_bar.ndim
    pool = F.avg_pool2d if d == 2 else F.avg_pool3d
    p = p_rel.reshape(rho_bar.shape)
    w = torch.clamp(1.0 - rho_bar / rho_c, min=0.0) ** m

    def avg(t: torch.Tensor) -> torch.Tensor:
        return pool(t[None, None], 2 * k + 1, 1, k, count_include_pad=False)[0, 0]

    return (avg(p * w) / (avg(w) + eps)).reshape(p_rel.shape)


def pressure_force(rho_bar: torch.Tensor, p_rel: torch.Tensor, h: tuple[float, ...]) -> torch.Tensor:
    """F = int p grad(rho_bar) dOmega (FORMALISM §4.5): the force of the fluid on the solid, shape (d,)."""
    g = density_gradient(rho_bar, h)
    p = p_rel.reshape(rho_bar.shape)
    return math.prod(h) * (p[..., None] * g).reshape(-1, len(h)).sum(dim=0)


def download_force(
    rho_bar: torch.Tensor,
    p_rel: torch.Tensor,
    *,
    h: tuple[float, ...],
    rho_air: float,
    u_ref: float,
    c_b: float,
    floor_at_base: bool = False,
) -> torch.Tensor:
    """D = -F_z ~ -sum_e h^d [ p_rel_e * min(dz rho_e, 0) + p_base * max(dz rho_e, 0) ],
    p_base = -0.5 * rho_a * u_ref**2 * c_b.   D > 0 means force pointing DOWN (C-CONV-07).

    `floor_at_base=True` (lesson 17; NOT in FORMALISM) replaces the windward pressure by max(p, p_base): a
    surface cannot be sucked harder than the wake behind it. Potential flow has infinite suction at sharp
    corners, which an optimiser finds and uses to drive D below zero; with the floor D >= p_b-term balance
    = 0 for a closed body (A_upwind = A_downwind, lesson 15), and its gradient vanishes where the floor is active.
    """
    dz = density_gradient(rho_bar, h)[..., -1]
    p = p_rel.reshape(rho_bar.shape)
    p_base = -0.5 * rho_air * u_ref**2 * c_b
    if floor_at_base:
        p = torch.clamp(p, min=p_base)
    return -math.prod(h) * (p * torch.clamp(dz, max=0.0) + p_base * torch.clamp(dz, min=0.0)).sum()


def blockage(phi: torch.Tensor, b: torch.Tensor, delta_phi: float) -> torch.Tensor:
    """Alternative closure-free indicator: effective conductance C_eff = b^T phi / delta_phi."""
    return torch.dot(b, phi) / delta_phi
