"""The ONLY hand-written backward of the package: implicit differentiation of K~ u = P f.

See ARCHITECTURE §5.4 and FORMALISM §3.6. Everything upstream (SIMP, filters, overhang, ...) is
plain Autograd and chains onto the gradients returned here.
"""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from aeroptimizer.fem.operators import StructuredFEOperator
from aeroptimizer.fem.solvers import LinearSolver


class SolveSPD(torch.autograd.Function):
    """u = K~(w)^{-1} P f. Backward = adjoint equation: ONE extra solve with the same handle.

    The handle (factorisation, or the CG matvec) is kept on `ctx`, so the backward needs no
    refactorisation and the solver iterations are never recorded in the autograd graph. Not
    twice-differentiable (`once_differentiable`): the Hessian is not needed by the optimisers.
    """

    @staticmethod
    def forward(ctx, w, f, op: StructuredFEOperator, solver: LinearSolver):  # type: ignore[override]
        handle = solver.prepare(op, w.detach())
        u = handle.solve(f.detach() * op.free_mask)
        ctx.op, ctx.handle = op, handle
        ctx.save_for_backward(u)
        return u

    @staticmethod
    @once_differentiable
    def backward(ctx, g_u):  # type: ignore[override]
        need_w, need_f = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        if not (need_w or need_f):
            return None, None, None, None
        (u,) = ctx.saved_tensors
        op = ctx.op
        free = op.free_mask.to(g_u.dtype)
        lam = ctx.handle.solve(g_u * free)  # K~ symmetric => K~^T = K~: same handle as forward
        grad_w = -op.dK_contract(lam, u) if need_w else None  # -lam_e^T Ke0 u_e, shape (Ne,)
        grad_f = lam * free if need_f else None  # P lam
        return grad_w, grad_f, None, None


def solve_spd(
    op: StructuredFEOperator, w: torch.Tensor, f: torch.Tensor, solver: LinearSolver
) -> torch.Tensor:
    """Differentiable u = K~(w)^{-1} P f with adjoint backward (C-SOLVE-03/04)."""
    return SolveSPD.apply(w, f, op, solver)
