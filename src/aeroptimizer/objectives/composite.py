"""J = sum_i w_i * term_i / ref_i with frozen references (C-OBJ-02, C-OBJ-03, C-OBJ-05; lesson 17).

`Composite` only combines scalars: the problem computes each raw term ONCE per evaluation (one mechanical
solve, one flow solve: C-OBJ-01) and hands them over. Reporting every raw and normalised term, not only
the sum, is what lets a Pareto front be drawn after the fact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from aeroptimizer.errors import ConfigError


@dataclass
class EvalResult:
    J: torch.Tensor
    g: list[torch.Tensor]
    terms: dict[str, torch.Tensor]  # raw AND normalised terms are reported (C-OBJ-02)
    state: object = field(repr=False, default=None)


class Composite:
    """Weighted sum of normalised terms, e.g. weights = {"C": w_c, "D": w_a}.

    `refs` are the normalisation constants; if omitted they are frozen from the first `value` call.
    A reference must be > 0: dividing by a negative one would flip the sign of the term, so the optimiser
    would maximise it (C-OBJ-03). Weights must be finite and >= 0, at least one > 0 (C-OBJ-05).
    """

    def __init__(self, weights: dict[str, float], refs: dict[str, float] | None = None) -> None:
        if not weights or any(not math.isfinite(w) or w < 0.0 for w in weights.values()):
            raise ConfigError(f"weights must be finite and >= 0, got {weights}")
        if not any(w > 0.0 for w in weights.values()):
            raise ConfigError(f"at least one weight must be > 0, got {weights}")
        self.weights = dict(weights)
        self.refs: dict[str, float] | None = None
        if refs is not None:
            self.freeze_references({k: torch.tensor(float(v)) for k, v in refs.items()})

    @property
    def frozen(self) -> bool:
        return self.refs is not None

    def freeze_references(self, terms: dict[str, torch.Tensor]) -> None:
        """Set the references ONCE; changing them later is forbidden (C-OBJ-03)."""
        if self.refs is not None:
            raise ConfigError("references are already frozen: changing them would move the optimum")
        refs = {}
        for name in self.weights:
            if name not in terms:
                raise ConfigError(f"no value for term {name!r} to freeze its reference")
            v = float(terms[name].detach())
            if not (v > 0.0 and math.isfinite(v)):
                if self.weights[name] > 0.0:
                    raise ConfigError(f"reference for {name!r} must be > 0 and finite, got {v:.4g}")
                continue  # a zero-weight term may keep an undefined reference (reported as nan)
            refs[name] = v
        self.refs = refs

    def value(self, terms: dict[str, torch.Tensor], g: list[torch.Tensor], state: object = None) -> EvalResult:
        """J and every raw + normalised term (`C`, `C_hat`, `D`, `D_hat`, ...); extra raw terms pass through."""
        if self.refs is None:
            self.freeze_references(terms)
        out = dict(terms)
        J = torch.zeros(())
        for name, w in self.weights.items():
            ref = self.refs.get(name)
            hat = terms[name] / ref if ref is not None else torch.full_like(terms[name], float("nan"))
            out[f"{name}_hat"] = hat
            if w > 0.0:
                J = J + w * hat
        return EvalResult(J=J, g=g, terms=out, state=state)
