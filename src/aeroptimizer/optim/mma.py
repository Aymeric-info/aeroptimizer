"""Method of Moving Asymptotes (Svanberg 1987), in-house port, dual sub-problem (ARCHITECTURE §5.9, D-11).

Plain MMA (no GCMMA inner loop, no elastic variables): minimise f(x) s.t. g_i(x) <= 0, 0 <= x <= 1.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from aeroptimizer.errors import ConfigError


class MMA:
    """Each step replaces f and g_i by separable convex approximations in 1/(U_j - x_j), 1/(x_j - L_j)
    built from the gradients at the current x, then solves that sub-problem exactly through its dual
    (m multipliers, one bisection when m = 1, L-BFGS-B otherwise).

    The problem is rescaled internally (f and each g_i by the inverse of their largest gradient
    component), which changes nothing mathematically but makes Svanberg's regularisation constants
    independent of the units. Contract: C-OPT-01 (`step(x, f, df, g, dg) -> x_new`, bounds exact).
    """

    def __init__(
        self, n_constraints: int = 1, move: float = 0.5, asyinit: float = 0.5, asyincr: float = 1.2,
        asydecr: float = 0.7,
    ) -> None:
        self.m, self.move = n_constraints, move
        self.asyinit, self.asyincr, self.asydecr = asyinit, asyincr, asydecr
        self.reset()

    def reset(self) -> None:
        """Reset asymptote history (C-OPT-04)."""
        self._x1 = self._x2 = self._low = self._upp = None
        self._k = 0

    def _asymptotes(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._k < 2:
            return x - self.asyinit, x + self.asyinit
        osc = (x - self._x1) * (self._x1 - self._x2)
        gamma = np.where(osc < 0, self.asydecr, np.where(osc > 0, self.asyincr, 1.0))
        low = x - gamma * (self._x1 - self._low)
        upp = x + gamma * (self._upp - self._x1)
        return np.clip(low, x - 10.0, x - 0.01), np.clip(upp, x + 0.01, x + 10.0)

    def step(
        self, x: np.ndarray, f: float, df: np.ndarray, g: np.ndarray, dg: np.ndarray, g_of_x=None  # noqa: ANN001
    ) -> np.ndarray:
        """`g_of_x` (exact constraint evaluator, used by OC) is accepted and ignored."""
        x = np.asarray(x, dtype=float)
        n = x.size
        g = np.ravel(np.asarray(g, dtype=float))
        dg = np.asarray(dg, dtype=float).reshape(-1, n)
        df = np.asarray(df, dtype=float).reshape(n)
        if g.size != self.m or dg.shape[0] != self.m:
            raise ConfigError(f"MMA built for {self.m} constraint(s), got {g.size}")
        s_f = 1.0 / max(float(np.abs(df).max()), 1e-300)
        s_g = 1.0 / np.maximum(np.abs(dg).max(axis=1), 1e-300)
        df, g, dg = df * s_f, g * s_g, dg * s_g[:, None]

        low, upp = self._asymptotes(x)
        alpha = np.maximum.reduce([low + 0.1 * (x - low), np.zeros(n), x - self.move])
        beta = np.minimum.reduce([upp - 0.1 * (upp - x), np.ones(n), x + self.move])
        ul, xl = upp - x, x - low

        def coeffs(d):  # p, q of Svanberg's approximation (all >= 0, regularised)
            pos, neg = np.maximum(d, 0.0), np.maximum(-d, 0.0)
            return ul**2 * (1.001 * pos + 0.001 * neg + 1e-5), xl**2 * (0.001 * pos + 1.001 * neg + 1e-5)

        p0, q0 = coeffs(df)
        pq = [coeffs(dg[i]) for i in range(self.m)]
        P = np.array([a for a, _ in pq]).reshape(self.m, n)
        Q = np.array([b for _, b in pq]).reshape(self.m, n)
        # approximated constraint i:  sum_j (P_ij/(U_j-x_j) + Q_ij/(x_j-L_j)) <= b_i
        b = -(g - (P / ul + Q / xl).sum(axis=1))

        def x_of(lam):  # minimiser of the Lagrangian for given multipliers
            pj, qj = p0 + lam @ P, q0 + lam @ Q
            sp_, sq_ = np.sqrt(pj), np.sqrt(qj)
            return np.clip((sp_ * low + sq_ * upp) / (sp_ + sq_), alpha, beta)

        def dual_grad(lam):  # dW/dlam_i = constraint i of the sub-problem at x(lam)
            xx = x_of(lam)
            return (P / (upp - xx) + Q / (xx - low)).sum(axis=1) - b

        if self.m == 1:
            if dual_grad(np.zeros(1))[0] <= 0.0:
                lam = np.zeros(1)
            else:
                hi = 1.0
                while dual_grad(np.array([hi]))[0] > 0.0 and hi < 1e15:
                    hi *= 2.0
                lo = 0.0
                for _ in range(200):
                    mid = 0.5 * (lo + hi)
                    if dual_grad(np.array([mid]))[0] > 0.0:
                        lo = mid
                    else:
                        hi = mid
                    if hi - lo <= 1e-14 * hi:
                        break
                lam = np.array([hi])  # feasible end of the bracket
        else:
            def neg_w(lam):
                xx = x_of(lam)
                w = (p0 / (upp - xx) + q0 / (xx - low)).sum() + lam @ ((P / (upp - xx) + Q / (xx - low)).sum(axis=1) - b)
                return -float(w), -dual_grad(lam)

            res = minimize(neg_w, np.ones(self.m), jac=True, method="L-BFGS-B",
                           bounds=[(0.0, 1e12)] * self.m, options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-12})
            lam = res.x
        x_new = x_of(lam)
        self._x2, self._x1, self._low, self._upp = self._x1, x.copy(), low, upp
        self._k += 1
        return x_new
