"""Continuation schedules for p, beta, beta_s (C-OPT-04)."""

from __future__ import annotations

from dataclasses import dataclass

from aeroptimizer.errors import ConfigError


@dataclass(frozen=True)
class Schedule:
    """Staircase schedule: stage k = it // every.

    p grows by 0.5 per stage; beta and beta_s double per stage; each stops at its final value.
    A pair (a, a) is a constant. Any change of a value between two iterations is a reason to call
    `optimizer.reset()`: the objective the optimiser sees has changed (C-OPT-04).
    """

    p: tuple[float, float] = (1.0, 3.0)
    beta: tuple[float, float] = (1.0, 16.0)
    beta_s: tuple[float, float] = (10.0, 80.0)
    every: int = 50  # iterations between steps

    def __post_init__(self) -> None:
        if self.every < 1:
            raise ConfigError(f"every must be >= 1, got {self.every}")
        for name in ("p", "beta", "beta_s"):
            a, b = getattr(self, name)
            if b < a or a < 0.0 or (name == "p" and a < 1.0):
                raise ConfigError(f"{name}: need (start, end) with 0 <= start <= end (p >= 1), got {(a, b)}")
            if a == 0.0 and b > 0.0:
                raise ConfigError(f"{name}: a doubling schedule cannot start at 0, got {(a, b)}")

    def at(self, it: int) -> dict[str, float]:
        """Parameters at iteration `it`; a change of any value must trigger optimizer.reset()."""
        k = it // self.every
        return {
            "p": min(self.p[0] + 0.5 * k, self.p[1]),
            "beta": min(self.beta[0] * 2.0**k, self.beta[1]),
            "beta_s": min(self.beta_s[0] * 2.0**k, self.beta_s[1]),
        }

    def is_final(self, it: int) -> bool:
        """True once every parameter has reached its final value."""
        cur = self.at(it)
        return cur["p"] == self.p[1] and cur["beta"] == self.beta[1] and cur["beta_s"] == self.beta_s[1]
