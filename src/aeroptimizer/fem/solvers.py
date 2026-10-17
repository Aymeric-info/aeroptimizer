"""Linear solvers for SPD K~ systems (ARCHITECTURE §5.3, CONTRACTS C-SOLVE-01/02/07/08/09).

Backends are interchangeable. A `SolveHandle` keeps the factorisation so the adjoint solve is free.

Two layers:
* algorithms that only need a matrix or a `matvec` callable (`DirectHandle`, `conjugate_gradient`,
  `CGHandle`, `jacobi_preconditioner`): they do not know the FE operator, so they can be
  tested alone;
* `ScipyDirect.prepare` / `TorchCG.prepare`: the thin wiring to `StructuredFEOperator`
  (`fem/operators.py`, lesson 6), built from the layer above.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from aeroptimizer.errors import ConfigError, SolverConvergenceError
from aeroptimizer.fem.operators import StructuredFEOperator

Preconditioner = Callable[[torch.Tensor], torch.Tensor]  # r -> M^{-1} r, M SPD


class SolveHandle(Protocol):
    def solve(self, b: torch.Tensor) -> torch.Tensor:
        """Solve K~ y = b. Reusable for the adjoint (K~ symmetric)."""
        ...


class LinearSolver(Protocol):
    def prepare(self, op: StructuredFEOperator, w: torch.Tensor) -> SolveHandle: ...


def check_symmetric(K: sp.spmatrix, tol: float = 1e-12) -> None:
    """C-SOLVE-08: ||K - K^T|| / ||K|| <= tol (Frobenius). The adjoint assumes K~ = K~^T."""
    K = sp.csr_matrix(K)
    asym = spla.norm(K - K.T) / max(spla.norm(K), 1e-300)
    if asym > tol:
        raise ConfigError(f"K~ is not symmetric: ||K-K^T||/||K|| = {asym:.2e} > {tol:.0e}")


class DirectHandle:
    """Sparse LU of a CSC matrix, kept for reuse (C-SOLVE-01). CPU, float64.

    `permc_spec` is the column ordering that limits fill-in; for a symmetric matrix `MMD_AT_PLUS_A`
    works on the pattern of K + K^T with `SymmetricMode` (the default of `ScipyDirect`).
    """

    def __init__(
        self,
        K: sp.spmatrix,
        permc_spec: str = "MMD_AT_PLUS_A",
        symmetric_mode: bool = True,
        tol: float = 1e-10,
        debug: bool = False,
    ) -> None:
        self.K = sp.csc_matrix(K, dtype=np.float64)
        if debug:
            check_symmetric(self.K)
        options = {"SymmetricMode": True} if symmetric_mode else {}
        self._lu = spla.splu(self.K, permc_spec=permc_spec, options=options)
        self.tol = tol
        self._norm1 = float(abs(self.K).sum(axis=0).max())  # ||K||_1, for the backward error
        self.n_factorisations = 1  # stays 1 however many solves follow: the handle is reused

    @property
    def fill_in(self) -> float:
        """nnz(L) + nnz(U) over nnz(K): how many zeros the factorisation filled."""
        return (self._lu.L.nnz + self._lu.U.nnz) / self.K.nnz

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        """Direct solve; raises SolverConvergenceError if the normwise backward error exceeds `tol`.

        backward error = ||K y - b|| / (||K||_1 ||y|| + ||b||): about machine epsilon for a stable LU
        whatever the conditioning. (A residual relative to ||b|| alone grows like eps * cond(K): it
        fails on slender beams and on designs with near-void islands, although the solve is as accurate
        as floating point allows.)
        """
        b_np = b.detach().cpu().numpy().astype(np.float64)
        y = self._lu.solve(b_np)
        norm_b = np.linalg.norm(b_np)
        scale = self._norm1 * np.linalg.norm(y) + norm_b
        res = np.linalg.norm(self.K @ y - b_np) / scale if scale > 0 else 0.0
        if not res <= self.tol:  # also catches NaN
            raise SolverConvergenceError(f"direct solve backward error {res:.2e} > {self.tol:.0e}")
        return torch.from_numpy(y).to(dtype=b.dtype, device=b.device)


def jacobi_preconditioner(diag: torch.Tensor) -> Preconditioner:
    """M = diag(K): z = r / diag. Requires a strictly positive diagonal (true for SPD K~)."""
    if not bool((diag > 0).all()):
        raise ConfigError("Jacobi preconditioner needs a strictly positive diagonal")
    return lambda r: r / diag


@dataclass(frozen=True)
class CGResult:
    x: torch.Tensor
    iterations: int
    converged: bool
    residuals: tuple[float, ...]  # ||b - K x_k|| / ||b|| after each iteration (index 0: x0)


def conjugate_gradient(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    *,
    x0: torch.Tensor | None = None,
    preconditioner: Preconditioner | None = None,
    tol: float = 1e-8,
    max_iter: int = 10_000,
) -> CGResult:
    """Preconditioned conjugate gradient for K x = b, K symmetric positive definite (matrix-free).

    Stops when the TRUE relative residual ||b - K x|| / ||b|| <= tol: when the cheap recursive
    residual says "converged", it is recomputed from x (one extra matvec) and, if it has drifted, replaced
    and the iteration continues. `max_iter` reached => `converged=False` (the caller decides).
    """
    norm_b = float(torch.linalg.norm(b.detach()))
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    if norm_b == 0.0:
        return CGResult(torch.zeros_like(b), 0, True, (0.0,))
    precond = preconditioner or (lambda r: r)
    r = b - matvec(x)
    history = [float(torch.linalg.norm(r.detach())) / norm_b]
    if history[0] <= tol:
        return CGResult(x, 0, True, tuple(history))
    z = precond(r)
    p = z.clone()
    rz = torch.dot(r, z)
    for k in range(1, max_iter + 1):
        Kp = matvec(p)
        pKp = torch.dot(p, Kp)
        if not bool(pKp > 0):
            raise SolverConvergenceError("CG breakdown: p^T K p <= 0 (K is not positive definite)")
        alpha = rz / pKp
        x = x + alpha * p
        r = r - alpha * Kp
        rel = float(torch.linalg.norm(r.detach())) / norm_b
        if rel <= tol:  # confirm with the true residual before stopping
            r = b - matvec(x)
            rel = float(torch.linalg.norm(r.detach())) / norm_b
            if rel <= tol:
                history.append(rel)
                return CGResult(x, k, True, tuple(history))
        history.append(rel)
        z = precond(r)
        rz_new = torch.dot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return CGResult(x, max_iter, False, tuple(history))


class CGHandle:
    """`SolveHandle` backed by CG. Matrix-free: needs `matvec` (and optionally the diagonal).

    `x0` of `solve` allows a warm start (previous solution inside a continuation phase).
    """

    def __init__(
        self,
        matvec: Callable[[torch.Tensor], torch.Tensor],
        diag: torch.Tensor | None = None,
        tol: float = 1e-8,
        max_iter: int = 10_000,
    ) -> None:
        self.matvec, self.tol, self.max_iter = matvec, tol, max_iter
        self.preconditioner = None if diag is None else jacobi_preconditioner(diag)
        self.last: CGResult | None = None

    def solve(self, b: torch.Tensor, x0: torch.Tensor | None = None) -> torch.Tensor:
        res = conjugate_gradient(
            self.matvec, b, x0=x0, preconditioner=self.preconditioner, tol=self.tol,
            max_iter=self.max_iter,
        )
        self.last = res
        if not res.converged:
            raise SolverConvergenceError(
                f"CG: residual {res.residuals[-1]:.2e} > tol {self.tol:.0e} "
                f"after {res.iterations} iterations"
            )
        return res.x


class ScipyDirect:
    """CPU float64 reference: scipy.sparse.linalg.splu on CSC (SymmetricMode, MMD_AT_PLUS_A)."""

    def factorize(self, K: sp.spmatrix, debug: bool = False) -> DirectHandle:
        """Factorise an assembled K~ (CSC/CSR); `prepare` passes `op.assemble_csr(w)`."""
        return DirectHandle(K, debug=debug)

    def prepare(self, op: StructuredFEOperator, w: torch.Tensor) -> SolveHandle:
        return self.factorize(op.assemble_csr(w))


class TorchCG:
    """Matrix-free preconditioned CG (CPU/CUDA). Jacobi in phase 2, geometric multigrid in phase 6."""

    def __init__(
        self, tol: float = 1e-8, max_iter: int = 10_000, preconditioner: str = "jacobi"
    ) -> None:
        if preconditioner not in ("none", "jacobi"):
            raise ConfigError(f"unknown preconditioner {preconditioner!r} (multigrid: phase 6)")
        self.tol, self.max_iter, self.preconditioner = tol, max_iter, preconditioner

    def handle(
        self, matvec: Callable[[torch.Tensor], torch.Tensor], diag: torch.Tensor | None
    ) -> CGHandle:
        """CG handle from a `matvec` and the diagonal of K~ (`prepare` passes the operator's)."""
        use = diag if self.preconditioner == "jacobi" else None
        return CGHandle(matvec, use, self.tol, self.max_iter)

    def prepare(self, op: StructuredFEOperator, w: torch.Tensor) -> SolveHandle:
        return self.handle(lambda v: op.matvec(w, v), op.diag(w))


def check_device_dtype(device: torch.device | str, dtype: torch.dtype) -> None:
    """C-SOLVE-09: float64 on MPS is refused with DeviceError."""
    from aeroptimizer.errors import DeviceError

    if torch.device(device).type == "mps" and dtype == torch.float64:
        raise DeviceError("MPS does not support float64; use device='cpu' (reference) or CUDA.")
