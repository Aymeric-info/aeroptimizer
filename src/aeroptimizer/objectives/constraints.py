"""Constraints g(x) <= 0 (FORMALISM §5.2)."""

from __future__ import annotations

import torch

from aeroptimizer.errors import ConfigError


def volume_fraction(rho_bar: torch.Tensor, design_or_solid_mask: torch.Tensor) -> torch.Tensor:
    """sum(rho_bar over D∪S) / |D∪S|: the denominator counts the passive solid, never the passive void."""
    mask = design_or_solid_mask.reshape(-1).to(torch.bool)
    flat = rho_bar.reshape(-1)
    if mask.shape != flat.shape:
        raise ConfigError(f"mask has {mask.numel()} cells, rho_bar {flat.numel()}")
    if not bool(mask.any()):
        raise ConfigError("the design/solid mask is empty")
    return flat[mask].sum() / mask.sum()


def volume_constraint(
    rho_bar: torch.Tensor, design_or_solid_mask: torch.Tensor, f_V: float, relative: bool = False
) -> torch.Tensor:
    """g_V = V - f_V  (C-OBJ-04); with `relative=True`, g_V = V / f_V - 1 (the scaling the optimisers use)."""
    if not 0.0 < f_V <= 1.0:
        raise ConfigError(f"f_V must be in (0, 1], got {f_V}")
    v = volume_fraction(rho_bar, design_or_solid_mask)
    return v / f_V - 1.0 if relative else v - f_V
