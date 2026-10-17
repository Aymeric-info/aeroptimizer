"""Contract tests for `operator` — stubs generated from CONTRACTS.md (scripts/contracts_index.py).

C-OP-01 to 07 are realised; they use the loop assembly `reference.assembly_numpy.assemble_dense` as oracle.
"""

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from aeroptimizer.fem.elements import elasticity_ke0
from aeroptimizer.fem.operators import StructuredFEOperator
from aeroptimizer.grid import Connectivity, Grid
from aeroptimizer.materials import Orthotropic
from reference.assembly_numpy import assemble_dense

CASES = [((5, 4), (1.0, 0.5)), ((3, 2, 2), (1.0, 0.5, 0.7))]  # 2D and 3D, non-cubic elements


def _op(shape, h, seed=0, nu=0.3):
    """Operator on a small grid: the face x = 0 clamped (all components) + one extra fixed dof."""
    g = Grid(shape, h)
    d = g.d
    mat = Orthotropic.isotropic(1.0, nu)
    C0 = mat.stiffness(d, "stress") if d == 2 else mat.stiffness(3)
    Ke0 = elasticity_ke0(d, h, C0)
    fixed = torch.zeros(g.n_dofs(d), dtype=torch.bool)
    fixed[: d * (shape[-1] + 1)] = True
    fixed[-1] = True
    op = StructuredFEOperator(g, Connectivity(g, d), Ke0, d, fixed)
    gen = torch.Generator().manual_seed(seed)
    w = 0.1 + torch.rand(g.n_elems, generator=gen)
    u = torch.randn(g.n_dofs(d), generator=gen)
    return op, w, u, gen


def _dense_k_tilde(op, w):
    """Oracle: loop assembly, then K~ = P K P + (I - P) written out densely."""
    K = assemble_dense(op.grid.shape, op.Ke0.numpy(), w.numpy(), op.ncomp)
    P = np.diag(op.free_mask.numpy().astype(float))
    return P @ K @ P + (np.eye(len(K)) - P)


def test_matvec_matches_assembled():
    r"""C-OP-01 — `matvec(w, u)` avec `w: (Nₑ,)` (facteur par élément : `s` ou `kappa`) retourne `K̃ u` (identité sur ddl bloqué…"""
    for shape, h in CASES:
        op, w, u, _ = _op(shape, h)
        y_ref = _dense_k_tilde(op, w) @ u.numpy()
        y = op.matvec(w, u).numpy()
        assert np.linalg.norm(y - y_ref) / np.linalg.norm(y_ref) < 1e-13
        fixed = ~op.free_mask
        assert torch.equal(op.matvec(w, u)[fixed], u[fixed])  # identity on fixed dofs


def test_matvec_bilinear_gradcheck():
    r"""C-OP-02 — `matvec` est **linéaire en `u`** et **linéaire en `w`** ; différentiable par rapport aux deux."""
    op, w, u, gen = _op((3, 2), (1.0, 0.5))
    v = torch.randn_like(u, generator=gen)
    w2 = torch.rand_like(w, generator=gen)
    a, b = 0.7, -1.3
    lhs, rhs = op.matvec(w, a * u + b * v), a * op.matvec(w, u) + b * op.matvec(w, v)
    assert torch.allclose(lhs, rhs, atol=1e-13)
    fixed = ~op.free_mask
    lhs = op.matvec(a * w + b * w2, u) - (1 - a - b) * u * fixed  # affine on fixed dofs
    assert torch.allclose(lhs, a * op.matvec(w, u) + b * op.matvec(w2, u), atol=1e-13)
    wg, ug = w.clone().requires_grad_(), u.clone().requires_grad_()
    assert torch.autograd.gradcheck(lambda ww, uu: op.matvec(ww, uu), (wg, ug))


def test_csr_equals_matvec():
    r"""C-OP-03 — `assemble_csr(w)` : même opérateur que `matvec` à `1e-12` relatif ; motif creux (`indptr/indices`) précalculé,…"""
    for shape, h in CASES:
        op, w, u, gen = _op(shape, h)
        K = op.assemble_csr(w)
        assert abs(K - K.T).max() < 1e-14 * abs(K).max()
        y = op.matvec(w, u).numpy()
        assert np.linalg.norm(K @ u.numpy() - y) / np.linalg.norm(y) < 1e-12
        indptr, indices = K.indptr.copy(), K.indices.copy()
        w2 = 0.1 + torch.rand(w.shape, generator=gen)
        K2 = op.assemble_csr(w2)  # new data, same pattern
        assert np.array_equal(K2.indptr, indptr) and np.array_equal(K2.indices, indices)
        assert np.allclose(K2.toarray(), _dense_k_tilde(op, w2), atol=1e-13)


def test_diag_matches_dense():
    r"""C-OP-04 — `diag(w)` = diagonale exacte de `K̃` (ddl bloqués : 1)."""
    for shape, h in CASES:
        op, w, _, _ = _op(shape, h)
        diag = op.diag(w).numpy()
        assert np.allclose(diag, np.diag(_dense_k_tilde(op, w)), rtol=1e-13, atol=0)
        assert np.all(diag[~op.free_mask.numpy()] == 1.0)


def test_dK_contract_fd():
    r"""C-OP-05 — `dK_contract(lam, u)` retourne `(Nₑ,)` : $\boldsymbol\lambda_e^\top\mathbf K_e^0\mathbf u_e$ (vecteurs **masqu…"""
    op, w, u, gen = _op((4, 3), (1.0, 0.5))
    lam = torch.randn_like(u, generator=gen)
    contracted = op.dK_contract(lam, u)
    assert contracted.shape == (op.grid.n_elems,)
    free = op.free_mask
    assert torch.allclose(op.dK_contract(lam, u), op.dK_contract(lam * free, u * free))  # masked
    eps = 1e-6
    for e in range(op.grid.n_elems):
        wp, wm = w.clone(), w.clone()
        wp[e] += eps
        wm[e] -= eps
        fd = (lam[free] @ op.matvec(wp, u)[free] - lam[free] @ op.matvec(wm, u)[free]) / (2 * eps)
        assert abs(fd - contracted[e]) < 1e-8 * max(1.0, abs(fd))


def test_spd_random_vectors():
    r"""C-OP-06 — `K̃` est SPD : $\mathbf z^\top\tilde{\mathbf K}\mathbf z>0$ pour `z≠0`, toute `w ≥ w_min > 0`."""
    for shape, h in CASES:
        op, _, _, gen = _op(shape, h)
        for w_min in (1.0, 1e-3, 1e-9):  # the SIMP floor eps_s
            w = w_min + torch.rand(op.grid.n_elems, generator=gen)
            for _ in range(20):
                z = torch.randn(op.n_dofs, generator=gen)
                assert z @ op.matvec(w, z) > 0
            np.linalg.cholesky(_dense_k_tilde(op, w))  # raises if not positive definite


def test_no_element_matrix_tensor():
    r"""C-OP-07 — Complexité mémoire de `matvec` : `O(Nₑ·d·2^d)` flottants, jamais `O(Nₑ·(d·2^d)²)` (pas de tenseur `(Nₑ,24,24)`…"""
    class Largest(TorchDispatchMode):
        biggest = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for t in tree_leaves(out):
                if isinstance(t, torch.Tensor):
                    Largest.biggest = max(Largest.biggest, t.numel())
            return out

    op, w, u, _ = _op((6, 5, 4), (1.0, 1.0, 1.0))
    ne, m = op.edof.shape
    with Largest():
        op.matvec(w, u)
        op.diag(w)
        op.dK_contract(u, u)
    assert Largest.biggest <= ne * m  # (Ne, m) gathers; a (Ne, m, m) tensor would be m times larger
