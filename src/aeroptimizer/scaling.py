"""SI <-> reduced units conversion (C-CONV-05). The ONLY place physical constants are scaled.

Inside the solver: E_p = 1 (reduced), lengths keep their physical value.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scaling:
    """Reference scales. `E_ref` [Pa] is the in-plane Young modulus used to normalise C0."""

    E_ref: float

    def stress_to_reduced(self, sigma_pa: float) -> float:
        raise NotImplementedError("phase 1")

    def force_to_reduced(self, f_newton: float) -> float:
        """Reduced force = f / E_ref (lengths unchanged, thickness handled in Ke0)."""
        raise NotImplementedError("phase 1")

    def compliance_to_si(self, c_reduced: float) -> float:
        raise NotImplementedError("phase 1")
