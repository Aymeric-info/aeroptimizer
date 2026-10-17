"""Contract tests for `optim`: C-OPT-01, 03, 04, 05 (OC: lesson 8; MMA, continuation, loop: lesson 10).

C-OPT-02 (MBB benchmark) is in tests/integration/test_mbb_benchmark.py.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.optimize import minimize

from aeroptimizer.errors import ConfigError
from aeroptimizer.optim.continuation import Schedule
from aeroptimizer.optim.loop import run
from aeroptimizer.optim.mma import MMA
from aeroptimizer.optim.oc import OC


def _volume_problem(n=200, vf=0.4, seed=0):
    """x, a compliance-like gradient df <= 0 and the volume constraint g = mean(x)/vf - 1."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    df = -rng.uniform(0.0, 5.0, n)
    dg = np.full(n, 1.0 / (n * vf))
    return x, df, x.mean() / vf - 1.0, dg, vf


@pytest.mark.parametrize("make", [OC, MMA], ids=["OC", "MMA"])
@pytest.mark.parametrize("seed", range(5))
def test_bounds_respected(make, seed):
    r"""C-OPT-01 — Interface : `Optimizer.step(x, f, df, g, dg) -> x_new` en `numpy` ; bornes `[0,1]` respectées exactement."""
    x, df, g, dg, _ = _volume_problem(seed=seed)
    x[:10], x[10:20] = 0.0, 1.0  # starting exactly on the bounds
    df[20:25] = 0.0  # no objective gain: elements must go down to the lower side, not NaN
    df[25:28] = 3.0  # wrong-sign gradient (not expected for compliance): must still be bounded
    opt = make()
    for _ in range(3):  # MMA changes its asymptotes after the first steps
        x_new = opt.step(x, 1.0, df, np.array([g]), dg[None])
        assert x_new.shape == x.shape and not np.isnan(x_new).any()
        assert x_new.min() >= 0.0 and x_new.max() <= 1.0
        assert np.abs(x_new - x).max() <= opt.move + 1e-12  # move limit
        x = x_new


def test_volume_feasible():
    r"""C-OPT-03 (moitié OC) — Le volume est respecté : `g_V ≤ 1e-3` (OC). La moitié MMA (`1e-6`) viendra avec la leçon 10."""
    for seed in range(5):
        x, df, g, dg, vf = _volume_problem(seed=seed)
        for _ in range(20):  # g_V must hold after every step, not only the last one
            x = OC().step(x, 1.0, df, x.mean() / vf - 1.0, dg)
            g_V = x.mean() / vf - 1.0
            assert g_V <= 1e-3
            assert g_V >= -1e-3  # and the volume constraint is active (not just feasible)
    # MMA: 1e-6 once the iterates have settled (the sub-problem sees the exact linear volume)
    x, df, g, dg, vf = _volume_problem(seed=0)
    mma = MMA()
    for _ in range(30):
        x = mma.step(x, 1.0, df, np.array([x.mean() / vf - 1.0]), dg[None])
    assert x.mean() / vf - 1.0 <= 1e-6


def test_mma_series_springs_analytic():
    """Same problem as lesson 8's OC test: C = sum a_e / rho_e^2 under a volume constraint; rho* ~ a^(1/3)."""
    N, V = 10, 0.4
    a = np.linspace(1.0, 4.0, N)
    exact = N * V * a ** (1 / 3) / (a ** (1 / 3)).sum()
    x, mma = np.full(N, V), MMA()
    for _ in range(60):
        x = mma.step(x, 0.0, -2 * a / x**3, np.array([x.mean() / V - 1.0]), np.full((1, N), 1.0 / (N * V)))
    assert np.abs(x - exact).max() < 1e-9


def test_mma_quadratic_with_active_bounds_and_constraint():
    """min |x-c|^2, sum x <= b, 0<=x<=1: the exact solution is clip(c - lam/2) with lam by bisection."""
    rng = np.random.default_rng(0)
    n, b = 30, 9.0
    c = rng.uniform(0.0, 1.0, n)
    lo, hi = 0.0, 50.0
    for _ in range(200):
        lam = 0.5 * (lo + hi)
        lo, hi = (lam, hi) if np.clip(c - lam / 2, 0, 1).sum() > b else (lo, lam)
    exact = np.clip(c - hi / 2, 0.0, 1.0)
    x, mma = np.full(n, 0.5), MMA()
    for _ in range(80):
        x = mma.step(x, 0.0, 2 * (x - c), np.array([x.sum() / b - 1.0]), np.ones((1, n)) / b)
    assert np.abs(x - exact).max() < 1e-8


def test_mma_two_constraints_against_slsqp():
    """Several constraints (what OC cannot do): same optimum as SciPy's SLSQP."""
    rng = np.random.default_rng(1)
    n = 30
    c, A = rng.uniform(0, 1, n), rng.uniform(0, 1, (2, n))
    b = np.array([0.3 * A[0].sum(), 0.4 * A[1].sum()])
    cons = [{"type": "ineq", "fun": lambda x, i=i: b[i] - A[i] @ x, "jac": lambda x, i=i: -A[i]} for i in range(2)]
    ref = minimize(lambda x: ((x - c) ** 2).sum(), np.full(n, 0.5), jac=lambda x: 2 * (x - c),
                   bounds=[(0, 1)] * n, constraints=cons, method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
    x, mma = np.full(n, 0.5), MMA(n_constraints=2)
    for _ in range(100):
        x = mma.step(x, 0.0, 2 * (x - c), A @ x / b - 1.0, A / b[:, None])
    assert np.abs(x - ref.x).max() < 1e-6
    assert (A @ x / b - 1.0).max() < 1e-9


def test_mma_is_scale_invariant_and_rejects_wrong_constraint_count():
    """Rescaling f and g by constants changes nothing (the internal normalisation), and m is checked."""
    x0, df, g, dg, vf = _volume_problem()
    a = MMA().step(x0, 1.0, df, np.array([g]), dg[None])
    b = MMA().step(x0, 1.0, 1e4 * df, np.array([1e-3 * g]), 1e-3 * dg[None])
    assert np.abs(a - b).max() < 1e-9
    with pytest.raises(ConfigError):
        MMA(n_constraints=2).step(x0, 1.0, df, np.array([g]), dg[None])
    with pytest.raises(ConfigError):
        OC().step(x0, 1.0, df, np.array([g, g]), np.stack([dg, dg]))


def test_oc_tolerance_is_tunable_and_fixed_point():
    """A looser bisection (top88's 1e-3) stays within 1e-3 relative on the volume; with df = -dg*lam
    (every B_e equal to 1) the update leaves x unchanged: that is the optimality condition."""
    x, df, g, dg, vf = _volume_problem()
    x_loose = OC(tol=1e-3).step(x, 1.0, df, g, dg)
    assert abs(x_loose.mean() / vf - 1.0) <= 1e-3
    x0 = np.full(50, 0.4)
    dg0 = np.full(50, 1.0 / (50 * 0.4))
    x1 = OC().step(x0, 1.0, -2.0 * dg0, 0.0, dg0)
    assert np.allclose(x1, x0, atol=1e-6)


class _Toy:
    """min sum(x^2) s.t. mean(x) <= 0.5, with the protocol used by `run`; records the continuation calls."""

    def __init__(self):
        self.calls, self.n = [], 6

    def initial_design(self):
        return np.full(self.n, 0.5)

    def set_continuation(self, params):
        self.calls.append(dict(params))

    def evaluate(self, x):
        return SimpleNamespace(J=(x**2).sum(), g=[x.mean() - 0.5], terms={"C": (x**2).sum()})


class _Frozen:
    """Optimiser that never moves and counts resets."""

    move = 0.2

    def __init__(self):
        self.resets = 0

    def step(self, x, f, df, g, dg, g_of_x=None):
        return x.copy()

    def reset(self):
        self.resets += 1


def test_continuation_schedule():
    r"""C-OPT-04 — **Continuation** : `p`, `β`, `β_s` suivent un calendrier configuré ; la fonction d'objectif est **recalibrée** (historique MMA réinitialisé) à chaque changement de paramètre de continuation."""
    sch = Schedule(p=(1.0, 2.0), beta=(1.0, 4.0), beta_s=(10.0, 10.0), every=5)
    assert sch.at(0) == {"p": 1.0, "beta": 1.0, "beta_s": 10.0}
    assert sch.at(4) == sch.at(0) and sch.at(5) == {"p": 1.5, "beta": 2.0, "beta_s": 10.0}
    assert sch.at(10) == {"p": 2.0, "beta": 4.0, "beta_s": 10.0} and sch.at(999) == sch.at(10)
    assert not sch.is_final(9) and sch.is_final(10)
    with pytest.raises(ConfigError):
        Schedule(beta=(0.0, 4.0))  # doubling from 0 is meaningless
    with pytest.raises(ConfigError):
        Schedule(p=(3.0, 1.0))
    toy, opt = _Toy(), _Frozen()
    run(toy, opt, sch, max_iter=20, tol=1e-9)
    assert [c["beta"] for c in toy.calls] == [1.0, 2.0, 4.0]  # set at the start, then at it = 5 and 10
    assert opt.resets == 2  # one reset per CHANGE (not at the start, not when nothing changes)
    real = MMA()
    real.step(np.full(4, 0.3), 1.0, -np.ones(4), np.array([0.1]), np.ones((1, 4)))
    real.reset()
    assert real._k == 0 and real._x1 is None  # history really cleared


def test_stop_reason_reported():
    r"""C-OPT-05 — Critère d'arrêt : variation max de `x` < tol **et** `β` à sa valeur finale, ou itérations max ; jamais d'arrêt silencieux (la cause d'arrêt est retournée)."""
    sch = Schedule(p=(1.0, 1.0), beta=(1.0, 4.0), beta_s=(1.0, 1.0), every=5)
    res = run(_Toy(), _Frozen(), sch, max_iter=100, tol=1e-6)
    assert res.stop_reason == "converged" and len(res.history) == 11  # dx = 0 from the start, but beta final only at it = 10
    res = run(_Toy(), _Frozen(), sch, max_iter=7, tol=1e-6)
    assert res.stop_reason == "max_iter" and len(res.history) == 7  # beta not final: no premature stop
    mover = _Frozen()
    mover.step = lambda x, f, df, g, dg, g_of_x=None: x + 0.1
    res = run(_Toy(), mover, Schedule(p=(1.0, 1.0), beta=(1.0, 1.0), beta_s=(1.0, 1.0)), max_iter=5, tol=1e-6)
    assert res.stop_reason == "max_iter"  # moving design, final schedule: runs out of iterations
    assert {"it", "J", "g_V", "dx", "beta", "p", "t_it"} <= set(res.history[0])
