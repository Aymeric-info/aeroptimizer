"""Relative thrust loss  D_hat = D / T_tot  (FORMALISM §4.7)."""

from __future__ import annotations

import torch

from aeroptimizer.errors import ConfigError


def relative_download(D: torch.Tensor, thrust_total_N: float) -> torch.Tensor:
    """Dimensionless loss of thrust D / T_tot; T_tot must be > 0."""
    if not thrust_total_N > 0:
        raise ConfigError(f"total thrust must be > 0, got {thrust_total_N}")
    return D / thrust_total_N
