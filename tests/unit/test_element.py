"""Contract tests for `element` (CONTRACTS §3.3)."""

import itertools
import math

import numpy as np
import pytest
import torch

from aeroptimizer.fem.elements import (
    center_gradients,
    elasticity_ke0,
    gauss_points,
    laplace_ke0,
    local_node_signs,
    shape_grads,
    shape_values,
)
from aeroptimizer.materials import Orthotropic

H_ANISO = {2: (0.5, 2.0), 3: (0.5, 2.0, 1.25)}  # non-uniform h: catches a mixed-up axis


def _ti_material() -> Orthotropic:
    """Transversely isotropic placeholders (configs/drone_arm_2d.yaml) rescaled to Ep = 1."""
    return Orthotropic.transversely_isotropic(Ep=1.0, Ez=0.75, nup=0.35, nupz=0.35, Gz=0.29)


@pytest.mark.parametrize("d", [2, 3])
def test_partition_of_unity(d):
    r"""C-ELEM-01 — `shape_grads(d)` : somme des $N_a$ = 1 et somme des $\nabla N_a$ = 0 à chaque point de Gauss."""
    N = shape_values(d)
    G = shape_grads(d, H_ANISO[d])
    assert N.shape == (2**d, 2**d) and G.shape == (2**d, 2**d, d)
    assert torch.allclose(N.sum(dim=1), torch.ones(2**d))
    assert torch.allclose(G.sum(dim=1), torch.zeros(2**d, d), atol=1e-14)


@pytest.mark.parametrize("d", [2, 3])
def test_ke0_spsd(d):
    r"""C-ELEM-02 — `Ke0` symétrique à `1e-12` relatif ; valeurs propres ≥ `-1e-12·‖Ke0‖`."""
    K = elasticity_ke0(d, H_ANISO[d], _ti_material().stiffness(d))
    assert K.shape == (d * 2**d, d * 2**d)
    norm = torch.linalg.matrix_norm(K, 2)
    assert torch.linalg.matrix_norm(K - K.T) <= 1e-12 * norm
    assert torch.linalg.eigvalsh((K + K.T) / 2).min() >= -1e-12 * norm


@pytest.mark.parametrize("d", [2, 3])
def test_rigid_body_modes(d):
    r"""C-ELEM-03 — **Modes rigides** : exactement 3 (2D) / 6 (3D) valeurs propres nulles (tolérance `1e-10·‖Ke0‖`)."""
    K = elasticity_ke0(d, H_ANISO[d], _ti_material().stiffness(d))
    norm = torch.linalg.matrix_norm(K, 2)
    ev = torch.linalg.eigvalsh(K)
    tol = 1e-10 * ev.max()
    n_rigid = 3 if d == 2 else 6
    assert int((ev.abs() < tol).sum()) == n_rigid
    assert ev[n_rigid] > 1e-3 * ev.max()  # a clear gap: the next mode is genuinely deformable
    # the modes themselves: translations and infinitesimal rotations are in the kernel
    s = local_node_signs(d) * torch.tensor(H_ANISO[d]) / 2  # node coordinates, centred
    rigid = []
    for c in range(d):  # translations
        u = torch.zeros(2**d, d)
        u[:, c] = 1.0
        rigid.append(u.reshape(-1))
    for i, j in itertools.combinations(range(d), 2):  # rotations in the (i, j) plane: u = e_k x x
        u = torch.zeros(2**d, d)
        u[:, i], u[:, j] = -s[:, j], s[:, i]
        rigid.append(u.reshape(-1))
    for u in rigid:
        assert torch.linalg.norm(K @ u) <= 1e-12 * norm * torch.linalg.norm(u)


@pytest.mark.parametrize("d", [2, 3])
def test_patch_uniform_strain(d):
    r"""C-ELEM-04 — **Patch test** : champ de déplacement affine imposé aux nœuds ⇒ $\mathbf K_e^0\mathbf u_e=$ forces nodales équivalentes à la contrainte constante $\mathbf C_0\boldsymbol\varepsilon$ (`1e-10`)."""
    h = H_ANISO[d]
    C0 = _ti_material().stiffness(d)
    K = elasticity_ke0(d, h, C0)
    eps_voigt = torch.tensor([0.3, -0.1, 0.2] if d == 2 else [0.3, -0.1, 0.2, 0.15, -0.25, 0.05])
    # symmetric strain tensor from the Voigt vector (shears are engineering: eps_ij = gamma_ij / 2)
    pairs = [(0, 1)] if d == 2 else [(1, 2), (0, 2), (0, 1)]
    eps = torch.diag(eps_voigt[:d]).clone()
    for r, (i, j) in enumerate(pairs):
        eps[i, j] = eps[j, i] = eps_voigt[d + r] / 2
    x = local_node_signs(d) * torch.tensor(h) / 2  # (2**d, d) node coordinates
    u = (x @ eps).reshape(-1)  # affine field u = eps . x, node-major
    # constant stress as a tensor
    sig_voigt = C0 @ eps_voigt
    sig = torch.diag(sig_voigt[:d]).clone()
    for r, (i, j) in enumerate(pairs):
        sig[i, j] = sig[j, i] = sig_voigt[d + r]
    # equivalent nodal forces of a constant stress: f_{a,i} = sum_j sig_ij * int dN_a/dx_j dV, with
    # int dN_a/dx_j dV = V s_{a,j} / (2^{d-1} h_j)  (hand-derived, independent of shape_grads)
    V = math.prod(h)
    s = local_node_signs(d)
    int_grad = V * s / (2 ** (d - 1) * torch.tensor(h))  # (2**d, d)
    f = (int_grad @ sig.T).reshape(-1)
    assert torch.allclose(K @ u, f, rtol=1e-10, atol=1e-10 * f.abs().max())


def test_matches_top88_ke():
    r"""C-ELEM-05 — **Oracle top88** : pour un matériau isotrope, contraintes planes, `h=1`, `Ke0` égale la formule fermée de top88 à une permutation de nœuds locaux près (permutation documentée dans le test)."""
    from reference.top88_numpy import top88_ke

    E, nu = 1.0, 0.3
    K = elasticity_ke0(2, (1.0, 1.0), Orthotropic.isotropic(E, nu).stiffness(2, "stress")).numpy()
    # top88 numbers nodes counter-clockwise from lower-left: (LL, LR, UR, UL) with (x right, z up).
    # Ours: bits (b_0 on x, b_1 on z): 0 = LL, 1 = UL, 2 = LR, 3 = UR. So top88 node t is our node
    # NODE_PERM[t]; the dof of (node, component c) is 2*node + c.
    NODE_PERM = [0, 2, 3, 1]
    dof_perm = np.array([2 * n + c for n in NODE_PERM for c in range(2)])
    assert np.allclose(K[np.ix_(dof_perm, dof_perm)], top88_ke(nu, E), rtol=0, atol=1e-13)


@pytest.mark.reference
@pytest.mark.parametrize("d", [2, 3])
def test_matches_loop_oracle(d):
    r"""C-ELEM-02/03 (oracle) — `Ke0` vectorisé = intégration par boucles explicites indépendantes, 2D et 3D, matériau orthotrope complet et `h` anisotrope."""
    from reference.assembly_numpy import ke0_by_loops

    mat = Orthotropic(E1=1.0, E2=0.8, E3=0.6, G23=0.2, G13=0.25, G12=0.3, nu12=0.3, nu13=0.25, nu23=0.2)
    C0 = mat.stiffness(d, "stress")
    K = elasticity_ke0(d, H_ANISO[d], C0).numpy()
    ref = ke0_by_loops(d, H_ANISO[d], C0.numpy())
    assert np.allclose(K, ref, rtol=1e-12, atol=1e-13 * np.abs(ref).max())


def test_scaling_laws():
    r"""Homogénéité : `Ke0` est invariant par homothétie en 2D (à épaisseur fixée) et croît comme `lambda` en 3D."""
    for d, power in ((2, 0), (3, 1)):
        C0 = _ti_material().stiffness(d)
        K1 = elasticity_ke0(d, (1.0,) * d, C0)
        K2 = elasticity_ke0(d, (3.0,) * d, C0)
        assert torch.allclose(K2, 3.0**power * K1, rtol=1e-12)


def test_thickness_only_2d():
    r"""L'épaisseur multiplie `Ke0` en 2D ; elle est refusée en 3D (pas d'effet silencieux)."""
    from aeroptimizer.errors import ConfigError

    C2 = _ti_material().stiffness(2)
    assert torch.allclose(elasticity_ke0(2, (1.0, 1.0), C2, 4.0), 4.0 * elasticity_ke0(2, (1.0, 1.0), C2))
    with pytest.raises(ConfigError):
        elasticity_ke0(3, (1.0,) * 3, _ti_material().stiffness(3), thickness=2.0)
    with pytest.raises(ConfigError):  # wrong Voigt size for d
        elasticity_ke0(3, (1.0,) * 3, C2)


@pytest.mark.parametrize("d", [2, 3])
def test_laplace_ke0(d):
    r"""C-ELEM-06 — `laplace_ke0(h)` : symétrique, $\mathbf K\mathbf 1=\mathbf 0$, rang $2^d-1$."""
    h = H_ANISO[d]
    K = laplace_ke0(d, h)
    assert K.shape == (2**d, 2**d)
    assert torch.allclose(K, K.T, atol=1e-14)
    assert torch.allclose(K @ torch.ones(2**d), torch.zeros(2**d), atol=1e-14)
    assert torch.linalg.matrix_rank(K, atol=1e-12).item() == 2**d - 1
    assert float(torch.linalg.eigvalsh(K).min()) > -1e-14  # PSD
    # energy of a linear potential phi = a . x is |a|^2 * volume (exact: Q1 contains the linear fields)
    s = local_node_signs(d) / 2 * torch.tensor(h)  # node coordinates about the centre
    a = torch.tensor([0.7, -1.3, 0.4][:d])
    phi = s @ a
    assert torch.allclose(phi @ K @ phi, (a @ a) * math.prod(h), rtol=1e-12)
    # the 2D unit square is the classical matrix (1/6)[[4,-1,-1,-2],...] (diagonal 2/3, opposite -1/3)
    if d == 2:
        K1 = laplace_ke0(2, (1.0, 1.0))
        assert torch.allclose(torch.diagonal(K1), torch.full((4,), 2.0 / 3.0))
        assert torch.allclose(K1[0, 3], torch.tensor(-1.0 / 3.0))


@pytest.mark.parametrize("d", [2, 3])
def test_center_gradients(d):
    r"""Complément de C-ELEM-06 — gradients au centre : $\sum_a\nabla N_a=0$ et reproduction d'un champ linéaire."""
    h = H_ANISO[d]
    G = center_gradients(d, h)
    assert G.shape == (2**d, d)
    assert torch.allclose(G.sum(dim=0), torch.zeros(d), atol=1e-14)
    s = local_node_signs(d) / 2 * torch.tensor(h)
    a = torch.tensor([0.7, -1.3, 0.4][:d])
    assert torch.allclose((s @ a) @ G, a, atol=1e-14)


@pytest.mark.parametrize("d", [2, 3])
def test_gauss_exactness(d):
    r"""C-ELEM-07 — Quadrature : Gauss $2^d$ points, exacte pour les monômes jusqu'au degré 3 par variable."""
    pts, w = gauss_points(d)
    assert pts.shape == (2**d, d) and w.shape == (2**d,)
    for powers in itertools.product(range(4), repeat=d):
        approx = (w * (pts ** torch.tensor(powers)).prod(dim=1)).sum()
        exact = math.prod(0.0 if p % 2 else 2.0 / (p + 1) for p in powers)
        assert abs(float(approx) - exact) < 1e-14, powers
