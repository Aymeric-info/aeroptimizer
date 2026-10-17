"""Contract tests for `solve` (CONTRACTS §3.5).

Done here (algorithms that need only a matrix / a matvec): C-SOLVE-01, 02, 08, 09 and the CG backend of
C-SOLVE-07 (also on the operator, MBB 60x20) and C-SOLVE-03 to 06 (`solve_spd`, the adjoint).
"""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from aeroptimizer.errors import ConfigError, DeviceError, SolverConvergenceError
from aeroptimizer.fem.elements import elasticity_ke0
from aeroptimizer.fem.adjoint import solve_spd
from aeroptimizer.fem.operators import StructuredFEOperator
from aeroptimizer.fem.solvers import (
    CGHandle,
    DirectHandle,
    ScipyDirect,
    TorchCG,
    check_device_dtype,
    check_symmetric,
    conjugate_gradient,
    jacobi_preconditioner,
)
from aeroptimizer.grid import Connectivity, Grid
from aeroptimizer.materials import Orthotropic


def _k_tilde(shape=(8, 4), contrast=1e-6, seed=0):
    """Clamped (left edge) 2D cantilever K~ = P K P + (I - P), with a random stiffness factor per element."""
    g = Grid(shape, (1.0, 1.0))
    Ke0 = elasticity_ke0(2, g.h, Orthotropic.isotropic(1.0, 0.3).stiffness(2)).numpy()
    ed = Connectivity(g, 2).edof.numpy()
    rng = np.random.default_rng(seed)
    s = contrast + (1 - contrast) * rng.uniform(size=len(ed)) ** 3
    m, N = ed.shape[1], g.n_dofs(2)
    r = np.broadcast_to(ed[:, :, None], (len(ed), m, m))
    c = np.broadcast_to(ed[:, None, :], (len(ed), m, m))
    K = sp.coo_matrix(((s[:, None, None] * Ke0).ravel(), (r.ravel(), c.ravel())), shape=(N, N)).tocsr()
    nodes = np.arange(g.n_nodes).reshape(shape[0] + 1, shape[1] + 1)
    fixed = (2 * nodes[0][:, None] + np.arange(2)).ravel()
    p = np.ones(N)
    p[fixed] = 0.0
    P = sp.diags(p)
    f = np.zeros(N)
    f[2 * nodes[-1, -1] + 1] = -1.0
    return (P @ K @ P + sp.diags(1 - p)).tocsc(), torch.from_numpy(p * f), torch.from_numpy(K.diagonal() * p + (1 - p))


def _case(shape=(6, 4), seed=0):
    """Clamped cantilever (face x = 0), unit load at the free corner, on a small grid."""
    g = Grid(shape, (1.0, 1.0))
    Ke0 = elasticity_ke0(2, g.h, Orthotropic.isotropic(1.0, 0.3).stiffness(2, "stress"))
    nz = shape[1] + 1
    fixed = torch.zeros(g.n_dofs(2), dtype=torch.bool)
    fixed[: 2 * nz] = True
    f = torch.zeros(g.n_dofs(2))
    f[2 * (g.n_nodes - 1) + 1] = -1.0
    op = StructuredFEOperator(g, Connectivity(g, 2), Ke0, 2, fixed)
    rho = 0.2 + 0.8 * torch.rand(g.n_elems, generator=torch.Generator().manual_seed(seed))
    return op, f, rho


class _CountingSolver:
    """Wraps a solver and counts `prepare` calls (factorisations) and `solve` calls."""

    def __init__(self, inner):
        self.inner, self.n_prepare, self.n_solve = inner, 0, 0

    def prepare(self, op, w):
        self.n_prepare += 1
        handle = self.inner.prepare(op, w)
        outer = self

        class _H:
            def solve(self, b, **kw):
                outer.n_solve += 1
                return handle.solve(b, **kw)

        return _H()


def test_handle_reuse():
    r"""C-SOLVE-01 — `LinearSolver.prepare(op, w) -> Handle` ; `Handle.solve(b)` résout `K̃ y = b`. La **même** `Handle` sert à la résolution directe et à l'adjoint (pas de refactorisation)."""
    K, f, _ = _k_tilde()
    h = ScipyDirect().factorize(K, debug=True)
    u = h.solve(f)  # state
    lam = h.solve(2.0 * f + 1.0)  # "adjoint" right-hand side, same factorisation
    assert h.n_factorisations == 1
    Kd = K.toarray()
    assert np.allclose(Kd @ u.numpy(), f.numpy(), atol=1e-12)
    assert np.allclose(Kd @ lam.numpy(), 2.0 * f.numpy() + 1.0, atol=1e-10)
    # symmetric K~: the adjoint solve with the same handle is also the transposed solve
    b = torch.arange(K.shape[0], dtype=torch.float64)
    assert np.allclose(h.solve(b).numpy(), np.linalg.solve(Kd.T, b.numpy()), rtol=1e-8)


def test_residual_bounds():
    r"""C-SOLVE-02 — Résidu relatif `‖K̃u−Pf‖/‖Pf‖ ≤ 1e-10` (directs) ou `≤ tol` configurable (CG, défaut `1e-8`) ; `SolverConvergenceError` sinon."""
    K, f, diag = _k_tilde()
    res = lambda u: np.linalg.norm(K @ u.numpy() - f.numpy()) / np.linalg.norm(f.numpy())  # noqa: E731
    assert res(DirectHandle(K).solve(f)) <= 1e-10
    cg = TorchCG().handle(lambda v: torch.from_numpy(K @ v.numpy()), diag)
    assert cg.tol == 1e-8 and res(cg.solve(f)) <= 1e-8
    tight = TorchCG(tol=1e-12).handle(lambda v: torch.from_numpy(K @ v.numpy()), diag)
    assert res(tight.solve(f)) <= 1e-12
    # not converged within max_iter -> error, never a silent wrong answer
    short = TorchCG(tol=1e-12, max_iter=2).handle(lambda v: torch.from_numpy(K @ v.numpy()), diag)
    with pytest.raises(SolverConvergenceError):
        short.solve(f)
    # a direct solver whose tolerance cannot be met also raises
    with pytest.raises(SolverConvergenceError):
        DirectHandle(K, tol=1e-30).solve(f)


def test_direct_backward_error_is_conditioning_free():
    """C-SOLVE-02 (direct): the normwise backward error stays ~eps on ill-conditioned K~ (stiff-to-void
    contrast 1e-12, slender beam), where a residual relative to ||f|| alone exceeds 1e-10."""
    for shape, contrast in [((60, 4), 1e-12), ((200, 4), 1e-3)]:
        K, f, _ = _k_tilde(shape=shape, contrast=contrast)
        h = DirectHandle(K)
        u = h.solve(f)  # must not raise
        berr = np.linalg.norm(K @ u.numpy() - f.numpy()) / (abs(K).sum(axis=0).max() * np.linalg.norm(u.numpy()) + np.linalg.norm(f.numpy()))
        assert berr < 1e-13


def test_backward_is_adjoint_only():
    r"""C-SOLVE-03 — `solve_spd(op, w, f)` est une `torch.autograd.Function` dont le backward est l'**équation adjointe** (une résolution supplémentaire) ; **aucune** itération du solveur dans le graphe."""
    op, f, rho = _case()
    w = (1e-3 + (1 - 1e-3) * rho**3).requires_grad_()
    for inner in (ScipyDirect(), TorchCG(tol=1e-12)):
        solver = _CountingSolver(inner)
        u = solve_spd(op, w, f, solver)
        assert (solver.n_prepare, solver.n_solve) == (1, 1)
        assert type(u.grad_fn).__name__ == "SolveSPDBackward"
        # the graph below the solve is only the leaf `w`: no solver iteration was recorded
        assert all(
            n is None or type(n).__name__ == "AccumulateGrad" for n, _ in u.grad_fn.next_functions
        )
        (f @ u).backward()
        assert (solver.n_prepare, solver.n_solve) == (1, 2)  # exactly one extra solve, no refactorisation
        w.grad = None


def test_grad_wrt_load():
    r"""C-SOLVE-04 — Le backward fournit **à la fois** `∂/∂w` ($-\lambda_e^\top K_e^0u_e$) **et** `∂/∂f` ($P\lambda$), de sorte que `f = f(rho_bar)` (poids propre, pression) est différentiable."""
    op, f, rho = _case()
    gen = torch.Generator().manual_seed(3)
    free = op.free_mask.to(f.dtype)
    w = (0.1 + torch.rand(op.grid.n_elems, generator=gen)).requires_grad_()
    ff = f.clone().requires_grad_()
    g = torch.randn(op.n_dofs, generator=gen)
    u = solve_spd(op, w, ff, ScipyDirect())
    gw, gf = torch.autograd.grad(u @ g, (w, ff))
    Kd = op.assemble_csr(w.detach()).toarray()
    lam = torch.from_numpy(np.linalg.solve(Kd, (g * free).numpy()))
    assert torch.allclose(gf, lam * free, atol=1e-10)  # dJ/df = P lam
    assert torch.allclose(gw, -op.dK_contract(lam, u.detach()), atol=1e-12)  # dJ/dw_e = -lam_e^T Ke0 u_e
    # f = f(rho_bar): a design-dependent load (linear map, e.g. self-weight) stays differentiable
    B = torch.randn(op.n_dofs, op.grid.n_elems, generator=gen)
    rho_t = rho.clone().requires_grad_()
    J = lambda r: g @ solve_spd(op, 0.1 + r, B @ r, ScipyDirect())  # noqa: E731
    J(rho_t).backward()
    eps = 1e-6
    for e in (0, 7, 20):
        d = torch.zeros_like(rho)
        d[e] = eps
        fd = (J(rho + d) - J(rho - d)) / (2 * eps)
        assert abs(float(fd) - float(rho_t.grad[e])) < 1e-7 * max(1.0, abs(float(fd)))


def test_gradcheck():
    r"""C-SOLVE-05 — `gradcheck` float64 (grille `6×4`, `p=3`) : Autograd = différences finies, `rtol ≤ 1e-6`, **pour** `w` et `f`."""
    op, f, rho = _case()
    gen = torch.Generator().manual_seed(5)
    w = (0.1 + torch.rand(op.grid.n_elems, generator=gen)).requires_grad_()
    ff = torch.randn(op.n_dofs, generator=gen).requires_grad_()
    for solver in (ScipyDirect(), TorchCG(tol=1e-13)):
        assert torch.autograd.gradcheck(
            lambda ww, fv: solve_spd(op, ww, fv, solver), (w, ff), rtol=1e-6, atol=1e-6
        )
    # through SIMP, p = 3: w = eps_s + (1 - eps_s) rho^3
    rho_t = rho.clone().requires_grad_()
    simp = lambda r, fv: solve_spd(op, 1e-3 + (1 - 1e-3) * r**3, fv, ScipyDirect())  # noqa: E731
    assert torch.autograd.gradcheck(simp, (rho_t, ff), rtol=1e-6, atol=1e-6)


def test_compliance_matches_analytic():
    r"""C-SOLVE-06 — Pour `J=fᵀu` : le gradient sortant égale la formule analytique $-p\bar\rho^{p-1}u_e^\top K_e^0u_e$ à `1e-10`."""
    op, f, rho = _case()
    p, eps_s = 3.0, 1e-3
    rho_t = rho.clone().requires_grad_()
    u = solve_spd(op, eps_s + (1 - eps_s) * rho_t**p, f, ScipyDirect())
    (f @ u).backward()  # J = f^T u (compliance)
    ue = u.detach()[op.edof]  # (Ne, m)
    ue0 = (ue @ op.Ke0 * ue).sum(-1)  # u_e^T Ke0 u_e
    analytic = -p * (1 - eps_s) * rho**(p - 1) * ue0
    assert torch.allclose(rho_t.grad, analytic, rtol=1e-10, atol=0)
    assert bool((rho_t.grad < 0).all())  # more material never increases the compliance


def test_backends_agree():
    r"""C-SOLVE-07 — Backends **interchangeables** : `scipy_direct` (CPU, référence), `torch_cg` (CPU/GPU, matrix-free, préconditionné) donnent la même `u` à `1e-6` relatif sur le cas `MBB 60×20`."""
    g = Grid((60, 20), (1.0, 1.0))
    Ke0 = elasticity_ke0(2, g.h, Orthotropic.isotropic(1.0, 0.3).stiffness(2, "stress"))
    nz = g.shape[1] + 1
    fixed = torch.zeros(g.n_dofs(2), dtype=torch.bool)
    fixed[0 : 2 * nz : 2] = True  # symmetry plane x = 0: u_x = 0 on the left edge
    fixed[2 * (g.n_nodes - nz) + 1] = True  # roller at the bottom-right corner: u_z = 0
    f = torch.zeros(g.n_dofs(2))
    f[2 * (nz - 1) + 1] = -1.0  # unit load at the top of the symmetry plane
    op = StructuredFEOperator(g, Connectivity(g, 2), Ke0, 2, fixed)
    rho = torch.rand(g.n_elems, generator=torch.Generator().manual_seed(1))
    w = 1e-3 + (1 - 1e-3) * rho**3  # SIMP stiffness with p = 3
    f = f * op.free_mask
    u_ref = ScipyDirect().prepare(op, w).solve(f)
    cg = TorchCG(tol=1e-10).prepare(op, w)
    u_cg = cg.solve(f)
    assert torch.linalg.norm(u_cg - u_ref) / torch.linalg.norm(u_ref) < 1e-6
    assert cg.last.converged


def test_backends_agree_algorithm_level():
    r"""C-SOLVE-07 (niveau algorithme, sans l'opérateur) — direct et CG Jacobi donnent la même solution à `1e-6` relatif."""
    K, f, diag = _k_tilde((24, 8), contrast=1e-3)
    u_ref = DirectHandle(K).solve(f)
    u_cg = TorchCG(tol=1e-10).handle(lambda v: torch.from_numpy(K @ v.numpy()), diag).solve(f)
    assert torch.linalg.norm(u_cg - u_ref) / torch.linalg.norm(u_ref) < 1e-6


def test_symmetry_check():
    r"""C-SOLVE-08 — Symétrie vérifiée : `‖K̃−K̃ᵀ‖/‖K̃‖ ≤ 1e-12` à l'assemblage (mode debug). Le backward suppose `K̃ᵀ=K̃`."""
    K, _, _ = _k_tilde()
    check_symmetric(K)  # passes
    DirectHandle(K, debug=True)
    bad = (K + sp.csc_matrix(([1e-6], ([0], [5])), shape=K.shape)).tocsc()  # one asymmetric entry
    with pytest.raises(ConfigError):
        check_symmetric(bad)
    with pytest.raises(ConfigError):
        DirectHandle(bad, debug=True)


def test_mps_float64_rejected():
    r"""C-SOLVE-09 — `float64` sur `mps` : refus explicite (`DeviceError`) — MPS ne supporte pas `float64`."""
    with pytest.raises(DeviceError):
        check_device_dtype("mps", torch.float64)
    check_device_dtype("mps", torch.float32)  # allowed
    check_device_dtype("cpu", torch.float64)  # the reference


# --- conjugate gradient: algorithm-level properties -------------------------------------------------


def _spd(n, cond, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return torch.from_numpy((Q * np.geomspace(1.0, cond, n)) @ Q.T)


def test_cg_finite_termination_and_agreement():
    r"""CG sur une matrice SPD dense : solution = `numpy.linalg.solve`, et convergence en un nombre d'itérations de l'ordre de `n` (au plus `n` en arithmétique exacte ; en flottant la perte d'orthogonalité en ajoute quelques-unes, d'où la borne `2n`)."""
    A = _spd(30, 50.0)
    b = torch.arange(1.0, 31.0)
    res = conjugate_gradient(lambda v: A @ v, b, tol=1e-10)
    assert res.converged and res.iterations <= 60
    assert torch.allclose(res.x, torch.linalg.solve(A, b), rtol=1e-8)
    # residual history is recorded, starts at 1 for x0 = 0
    assert res.residuals[0] == pytest.approx(1.0) and res.residuals[-1] <= 1e-10


def test_cg_trivial_cases_and_warm_start():
    r"""CG : `b = 0` donne `x = 0` sans itération ; un départ à la solution converge en 0 itération ; un départ proche converge plus vite."""
    A = _spd(40, 1e3)
    b = torch.ones(40)
    assert conjugate_gradient(lambda v: A @ v, torch.zeros(40)).iterations == 0
    x_star = torch.linalg.solve(A, b)
    assert conjugate_gradient(lambda v: A @ v, b, x0=x_star, tol=1e-8).iterations == 0
    cold = conjugate_gradient(lambda v: A @ v, b, tol=1e-10)
    warm = conjugate_gradient(lambda v: A @ v, b, x0=x_star * (1 + 1e-4), tol=1e-10)
    assert warm.iterations < cold.iterations


def test_cg_jacobi_fixes_bad_scaling():
    r"""CG : une matrice SPD dont l'échelle varie de `1e-6` à `1` ligne par ligne (comme un contraste de rigidité) est lente sans préconditionneur et ~aussi rapide que la matrice bien échelonnée avec Jacobi."""
    A0 = _spd(60, 10.0)
    d = torch.from_numpy(np.geomspace(1e-6, 1.0, 60))  # symmetric diagonal scaling D A D
    A = d[:, None] * A0 * d[None, :]
    b = torch.ones(60)
    plain = conjugate_gradient(lambda v: A @ v, b, tol=1e-8, max_iter=5000)
    jac = conjugate_gradient(lambda v: A @ v, b, preconditioner=jacobi_preconditioner(torch.diag(A)),
                             tol=1e-8, max_iter=5000)
    assert jac.converged and jac.iterations <= 60
    assert (not plain.converged) or plain.iterations > 3 * jac.iterations


def test_cg_detects_indefinite_and_bad_diag():
    r"""CG signale une matrice non définie positive (rupture `pᵀKp ≤ 0`) ; Jacobi refuse une diagonale non strictement positive."""
    A = torch.diag(torch.tensor([1.0, -1.0, 2.0]))
    with pytest.raises(SolverConvergenceError):
        conjugate_gradient(lambda v: A @ v, torch.ones(3))
    with pytest.raises(ConfigError):
        jacobi_preconditioner(torch.tensor([1.0, 0.0]))
    with pytest.raises(ConfigError):
        TorchCG(preconditioner="multigrid")


def test_cg_float32_cannot_reach_tight_tolerance():
    r"""Précision : en `float32` le CG ne descend pas sous ~1e-6 de résidu relatif ; la tolérance 1e-10 est refusée avec une erreur, jamais acceptée en silence."""
    A = _spd(60, 1e3).to(torch.float32)
    b = torch.ones(60, dtype=torch.float32)
    ok = conjugate_gradient(lambda v: A @ v, b, tol=1e-4)
    assert ok.converged and ok.x.dtype == torch.float32
    h = CGHandle(lambda v: A @ v, None, tol=1e-10, max_iter=500)
    with pytest.raises(SolverConvergenceError):
        h.solve(b)
