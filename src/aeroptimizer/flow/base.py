"""FlowModel protocol (C-FLOW-01): swap potential flow for Brinkman-Stokes later without touching objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class FlowState:
    phi: torch.Tensor  # (n_nodes,) velocity potential, phi = 0 on the outlet
    v: torch.Tensor  # (n_elems, d) velocity at element centres (= grad phi)
    p_rel: torch.Tensor  # (n_elems,) p - p_inf from Bernoulli (FORMALISM §4.5)


class FlowModel(Protocol):
    def solve(self, rho_bar: torch.Tensor) -> FlowState: ...
