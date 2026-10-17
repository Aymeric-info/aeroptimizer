"""Compliance C = f^T u (FORMALISM §3.6)."""

from __future__ import annotations

import torch


def compliance(f: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """C = f^T u = u^T K~ u, twice the strain energy.

    `u` vanishes on the fixed dofs, so loads placed there do not contribute. Differentiable in `u`
    and `f`; the gradient in the density comes from `solve_spd` (adjoint, lambda = u here).
    """
    return torch.dot(f, u)
