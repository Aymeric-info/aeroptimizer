"""Main optimisation loop (ARCHITECTURE §5.9). Returns the stop reason, never stops silently (C-OPT-05)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RunResult:
    x: np.ndarray
    history: list[dict]  # it, J, C, D, g_V, beta, p, dx, t_it (C-IO-01)
    stop_reason: str  # "converged" | "max_iter"


def _gradients(res, x_t: torch.Tensor) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """(J, dJ, g, dg) as numpy: one backward per scalar (the graph is kept for the next one)."""
    scalars = [res.J, *res.g]
    grads = [
        torch.autograd.grad(s, x_t, retain_graph=True, allow_unused=True)[0] for s in scalars
    ]
    grads = [torch.zeros_like(x_t) if gr is None else gr for gr in grads]
    J = float(res.J.detach())
    g = np.array([float(v.detach()) for v in res.g])
    dg = np.stack([gr.numpy() for gr in grads[1:]]) if res.g else np.zeros((0, x_t.numel()))
    return J, grads[0].numpy(), g, dg


def run(  # noqa: ANN001
    problem, optimizer, schedule, callbacks=(), *, max_iter: int = 200, tol: float = 1e-2, x0=None
) -> RunResult:
    """Minimise `problem` with `optimizer`, following `schedule`.

    `problem.set_continuation(params)` is called whenever a schedule value changes and
    `optimizer.reset()` right after (C-OPT-04); `problem.evaluate(x)` takes a flat float tensor
    (requires_grad) and returns an object with `.J`, `.g` (list of scalar tensors) and `.terms`.
    Stops when max|dx| < tol AND every continuation parameter is final, or after `max_iter` iterations
    (C-OPT-05); the reason is returned in `RunResult.stop_reason`.
    """
    x = np.array(problem.initial_design() if x0 is None else x0, dtype=float)
    history: list[dict] = []
    params: dict[str, float] | None = None
    stop = "max_iter"
    for it in range(max_iter):
        t0 = time.perf_counter()
        new = schedule.at(it)
        if new != params:
            problem.set_continuation(new)
            if params is not None:
                optimizer.reset()
            params = new
        x_t = torch.from_numpy(x).requires_grad_()
        res = problem.evaluate(x_t)
        J, dJ, g, dg = _gradients(res, x_t)
        exact = getattr(problem, "constraint_value", None)  # exact g(x) for OC's bisection, if offered
        x_new = optimizer.step(x, J, dJ, g, dg, g_of_x=exact)
        dx = float(np.abs(x_new - x).max())
        row = {"it": it, "J": J, "g_V": float(g[0]) if g.size else float("nan"), "dx": dx,
               "beta": params["beta"], "p": params["p"],
               **{k: float(v.detach()) for k, v in res.terms.items()}}
        row["t_it"] = time.perf_counter() - t0
        history.append(row)
        for cb in callbacks:
            cb(it, x, row)
        x = x_new
        if dx < tol and schedule.is_final(it):
            stop = "converged"
            break
    return RunResult(x=x, history=history, stop_reason=stop)
