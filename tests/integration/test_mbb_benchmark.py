"""Contract C-OPT-02: OC on the MBB benchmark against the NumPy top88 oracle (recomputed here)."""

import numpy as np
import torch
from reference.top88_numpy import top88

from aeroptimizer.optim.continuation import Schedule
from aeroptimizer.optim.loop import run
from aeroptimizer.optim.oc import OC
from aeroptimizer.problem import ComplianceProblem


def test_oc_matches_reference():
    r"""C-OPT-02 — **OC** : sur le benchmark MBB `60×20`, `V=0.5`, `R=1.5·h`, `p=3` : la compliance finale est dans `±2 %` de la valeur de référence (top88 recalculée avec le script de référence du dépôt).

    Reference: `top88(..., ft=2)` (same density filter, 1e-3 bisection as in the original code, stopping
    when max|dx| < 0.01). The loop shares the stopping rule; the two compliances must agree within 2 %.
    """
    ref = top88(60, 20, 0.5, 3.0, 1.5, max_iter=200, ft=2)
    problem = ComplianceProblem("mbb", (60, 20), volfrac=0.5, R=1.5)
    res = run(problem, OC(tol=1e-3), Schedule(p=(3.0, 3.0), beta=(0.0, 0.0), beta_s=(1.0, 1.0)),
              max_iter=200, tol=0.01)
    c_ours, c_ref = res.history[-1]["C"], ref["c"][-1]
    assert abs(c_ours / c_ref - 1.0) < 0.02
    assert abs(res.history[-1]["g_V"]) < 1e-3  # and the volume holds
    assert res.stop_reason in ("converged", "max_iter")
