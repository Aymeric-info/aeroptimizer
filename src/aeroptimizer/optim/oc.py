"""Optimality Criteria update for compliance + volume (validation baseline, ARCHITECTURE §5.9)."""

from __future__ import annotations

import numpy as np

from aeroptimizer.errors import ConfigError


class OC:
    """Heuristic fixed-point update x <- x (-df / (lam dg))^damping, lam set by bisection.

    Written for one constraint g(x) <= 0 and a compliance-like objective (df <= 0, dg > 0). With
    the volume constraint g = mean(x)/V - 1 this is exactly top88's update (Andreassen et al. 2011).
    """

    def __init__(
        self, move: float = 0.2, damping: float = 0.5, tol: float = 1e-6, max_bisect: int = 200
    ) -> None:
        self.move, self.damping, self.tol, self.max_bisect = move, damping, tol, max_bisect

    def step(
        self, x: np.ndarray, f: float, df: np.ndarray, g: float, dg: np.ndarray, g_of_x=None  # noqa: ANN001
    ) -> np.ndarray:
        """Bisection on the multiplier lam; keeps x in [0,1] exactly (C-OPT-01).

        `g` is the constraint value (feasible when <= 0) and `dg` its gradient; both may be given as
        scalar / (n,) or as (1,) / (1, n), the shapes `optim.loop.run` uses. The constraint is
        linearised around `x`, which is exact for the volume of `x` itself. When the constrained
        quantity is a nonlinear function of `x` (volume of the projected density), pass
        `g_of_x(x_trial) -> float` and the bisection uses the exact constraint instead. Bisection stops when the bracket is
        narrower than `tol` (relative; top88 uses 1e-3, the default here is tighter) and the
        returned point is the feasible end of the bracket (top88 returns the last midpoint).
        """
        x = np.asarray(x, dtype=float)
        g_arr, dg = np.ravel(np.asarray(g, dtype=float)), np.asarray(dg, dtype=float)
        if g_arr.size != 1 or dg.size != x.size:
            raise ConfigError("OC handles exactly one constraint (use MMA for several)")
        g, dg = float(g_arr[0]), dg.reshape(-1)
        lo = np.clip(x - self.move, 0.0, 1.0)
        hi = np.clip(x + self.move, 0.0, 1.0)
        ratio = np.maximum(-np.asarray(df, dtype=float), 0.0) / np.maximum(dg, 1e-30)
        l1, l2 = 0.0, 1e9
        for _ in range(self.max_bisect):
            if (l2 - l1) <= self.tol * (l1 + l2):
                break
            lmid = 0.5 * (l1 + l2)
            x_try = np.clip(x * (ratio / lmid) ** self.damping, lo, hi)
            gt = g_of_x(x_try) if g_of_x is not None else g + float(np.dot(dg, x_try - x))
            if gt > 0.0:  # too much volume: raise lam
                l1 = lmid
            else:
                l2 = lmid
        return np.clip(x * (ratio / l2) ** self.damping, lo, hi)  # feasible end of the bracket

    def reset(self) -> None:
        """Called on every continuation change (C-OPT-04). OC has no history: nothing to do."""
