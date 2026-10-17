"""Contract tests for `material` — stubs generated from CONTRACTS.md (scripts/contracts_index.py).

Replace a stub body when its phase is reached.
"""

import pytest
import torch

from aeroptimizer.errors import MaterialError
from aeroptimizer.materials import Orthotropic

ORTHO = dict(E1=1.0, E2=0.8, E3=0.6, G23=0.2, G13=0.25, G12=0.3, nu12=0.3, nu13=0.25, nu23=0.2)


def test_voigt_order():
    r"""C-CONV-06 — **Voigt** : ordre `(xx,yy,zz,yz,xz,xy)` en 3D, `(xx,zz,xz)` en 2D, cisaillements d'ingénierie."""
    C3 = Orthotropic(**ORTHO).stiffness(3)
    # engineering shears: no factor 2, so each shear diagonal is exactly its modulus, in the order yz, xz, xy
    assert (C3[3, 3], C3[4, 4], C3[5, 5]) == pytest.approx((0.2, 0.25, 0.3), rel=1e-14)
    assert C3[3:, :3].abs().max() == 0 and C3[:3, 3:].abs().max() == 0  # orthotropy: no normal-shear coupling
    # S is the exact inverse in the SAME order: a unit shear strain gamma_xz gives sigma_xz = G13 only
    gamma = torch.tensor([0.0, 0, 0, 0, 1.0, 0])
    assert torch.allclose(C3 @ gamma, torch.tensor([0.0, 0, 0, 0, 0.25, 0]))
    # 2D (xx, zz, xz): the shear modulus is G13 (plane x-z)
    for plane in ("stress", "strain"):
        C2 = Orthotropic(**ORTHO).stiffness(2, plane)
        assert C2.shape == (3, 3) and C2[2, 2] == pytest.approx(0.25, rel=1e-14)


def test_rejects_non_spd():
    r"""C-MAT-01 — `Orthotropic(E1,E2,E3,G23,G13,G12,nu12,nu13,nu23)` : exige `S ≻ 0` (Cholesky) ; sinon `MaterialError` avec la condition violée."""
    with pytest.raises(MaterialError, match="E2"):
        Orthotropic(**{**ORTHO, "E2": -1.0})
    with pytest.raises(MaterialError, match="G13"):
        Orthotropic(**{**ORTHO, "G13": 0.0})
    # isotropic with nu = 0.6 > 0.5: moduli are positive but S is indefinite
    with pytest.raises(MaterialError, match="positive definite"):
        Orthotropic.isotropic(1.0, 0.6)
    Orthotropic.isotropic(1.0, 0.499)  # still admissible (nearly incompressible)


def test_C_S_inverse():
    r"""C-MAT-02 — `compliance_matrix()` symétrique ; `C0 @ S == I` à `1e-12` (relatif)."""
    m = Orthotropic(**ORTHO)
    S, C = m.compliance_matrix(), m.stiffness(3)
    assert torch.allclose(S, S.T, rtol=0, atol=1e-15)
    assert torch.allclose(C @ S, torch.eye(6, dtype=torch.float64), rtol=0, atol=1e-12)
    assert torch.allclose(C, C.T, rtol=1e-12, atol=1e-14)


def test_iso_is_special_case():
    r"""C-MAT-03 — `Orthotropic.isotropic(E,nu)` ≡ `Isotropic(E,nu)` (composantes de `C0` identiques à `1e-12`)."""
    # There is no separate Isotropic class: the oracle is the closed-form Lame stiffness.
    E, nu = 2.0, 0.3
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    expected = torch.zeros(6, 6, dtype=torch.float64)
    expected[:3, :3] = lam
    expected[range(3), range(3)] = lam + 2 * mu
    expected[range(3, 6), range(3, 6)] = mu
    assert torch.allclose(Orthotropic.isotropic(E, nu).stiffness(3), expected, rtol=1e-12, atol=1e-12)


def test_transverse_isotropy():
    r"""C-MAT-04 — `TransverselyIsotropic(Ep,Ez,nup,nupz,Gz)` : `E1==E2`, `G12==Ep/(2(1+nup))`, `nu13==nu23` ; construction via `Orthotropic`."""
    m = Orthotropic.transversely_isotropic(Ep=2.4, Ez=1.8, nup=0.35, nupz=0.3, Gz=0.7)
    assert m.E1 == m.E2 == 2.4 and m.E3 == 1.8
    assert m.G12 == pytest.approx(2.4 / (2 * 1.35), rel=1e-15)
    assert m.G13 == m.G23 == 0.7 and m.nu13 == m.nu23 == 0.3
    # in-plane isotropy of the stiffness: swapping x and y leaves C unchanged
    C = m.stiffness(3)
    perm = [1, 0, 2, 4, 3, 5]  # x <-> y in Voigt order
    assert torch.allclose(C, C[perm][:, perm], rtol=1e-12)


def test_plane_reductions():
    r"""C-MAT-05 — `stiffness(plane=...)` en 2D : `"stress"` = inversion **après** extraction `{xx,zz,xz}` de `S` ; `"strain"` = extraction **après** inversion de `C`. Les deux coïncident si `ν_{xy}=ν_{yz}=0`."""
    E, nu = 2.0, 0.3
    iso = Orthotropic.isotropic(E, nu)
    cp = E / (1 - nu**2) * torch.tensor([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]], dtype=torch.float64)
    dp = E / ((1 + nu) * (1 - 2 * nu)) * torch.tensor(
        [[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu) / 2]], dtype=torch.float64
    )
    assert torch.allclose(iso.stiffness(2, "stress"), cp, rtol=1e-12)
    assert torch.allclose(iso.stiffness(2, "strain"), dp, rtol=1e-12)
    # the two reductions differ in general ...
    ti = Orthotropic.transversely_isotropic(Ep=2.4, Ez=1.8, nup=0.35, nupz=0.35, Gz=0.7)
    assert not torch.allclose(ti.stiffness(2, "stress"), ti.stiffness(2, "strain"), rtol=1e-3)
    # ... and coincide when nu12 = nu23 = 0 (the y direction is then decoupled)
    dec = Orthotropic(**{**ORTHO, "nu12": 0.0, "nu23": 0.0})
    assert torch.allclose(dec.stiffness(2, "stress"), dec.stiffness(2, "strain"), rtol=1e-12)


def test_returns_constant():
    r"""C-MAT-06 — Retourne un tenseur `(n_voigt, n_voigt)` `float64`, **sans** gradient (constant du problème)."""
    torch.set_default_dtype(torch.float32)  # the material must not depend on the ambient default dtype
    try:
        m = Orthotropic(**ORTHO)
        for C, n in ((m.stiffness(3), 6), (m.stiffness(2), 3), (m.compliance_matrix(), 6)):
            assert C.shape == (n, n) and C.dtype == torch.float64 and not C.requires_grad
    finally:
        torch.set_default_dtype(torch.float64)


def _rotate_4th_order(C: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Oracle for C-MAT-07: rotate the 4th-order tensor index by index, then go back to Voigt."""
    import numpy as np

    V = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    C4 = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(V):
        for b, (k, l) in enumerate(V):
            for ii, jj in {(i, j), (j, i)}:
                for kk, ll in {(k, l), (l, k)}:
                    C4[ii, jj, kk, ll] = float(C[a, b])
    Rn = R.numpy()
    C4r = np.einsum("ia,jb,kc,ld,abcd->ijkl", Rn, Rn, Rn, Rn, C4)
    return torch.tensor([[C4r[i, j, k, l] for (k, l) in V] for (i, j) in V])


def test_rotation_matches_tensor_oracle():
    r"""C-MAT-07 — rigidité du matériau tourné = rotation du tenseur d'ordre 4, module directionnel classique."""
    import math

    from aeroptimizer.materials import rotate_stiffness, rotation_xz

    m = Orthotropic(**ORTHO)
    C = m.stiffness(3)
    for theta in (0.0, 0.3, 1.1, math.pi / 2, -0.7):
        R = rotation_xz(theta)
        Cr = rotate_stiffness(C, R)
        assert torch.allclose(Cr, _rotate_4th_order(C, R), rtol=0, atol=1e-12)
        # modulus along x: classical off-axis formula (fibre 1 at angle theta from x, plane x-z)
        c, s = math.cos(theta), math.sin(theta)
        inv_E = (c**4 / m.E1 + (1 / m.G13 - 2 * m.nu13 / m.E1) * s**2 * c**2 + s**4 / m.E3)
        assert 1.0 / torch.linalg.inv(Cr)[0, 0] == pytest.approx(1.0 / inv_E, rel=1e-12)


def test_rotation_properties():
    r"""C-MAT-07 — groupe, invariance isotrope, échange E1/E3 à 90°, réduction 2D après rotation."""
    import math

    from aeroptimizer.materials import rotate_stiffness, rotation_xz

    m = Orthotropic(**ORTHO)
    a, b = 0.4, 0.9
    assert torch.allclose(rotation_xz(a) @ rotation_xz(b), rotation_xz(a + b), atol=1e-14)
    # isotropic material: invariant for any rotation
    iso = Orthotropic.isotropic(2.0, 0.3).stiffness(3)
    assert torch.allclose(rotate_stiffness(iso, rotation_xz(0.77)), iso, atol=1e-12)
    # 90 degrees about y: x and z swap, so E1 <-> E3 and G12 <-> G23
    Sr = torch.linalg.inv(rotate_stiffness(m.stiffness(3), rotation_xz(math.pi / 2)))
    assert (1 / Sr[0, 0], 1 / Sr[2, 2]) == pytest.approx((m.E3, m.E1), rel=1e-12)
    assert (1 / Sr[5, 5], 1 / Sr[3, 3]) == pytest.approx((m.G23, m.G12), rel=1e-12)
    # strain energy is invariant: eps^T C eps = eps'^T C' eps' with eps' = T^{-T} eps
    from aeroptimizer.materials import stress_transform

    T = stress_transform(rotation_xz(a))
    eps = torch.tensor([0.3, -0.1, 0.2, 0.05, -0.4, 0.15], dtype=torch.float64)
    eps_r = torch.linalg.inv(T).T @ eps
    e0 = eps @ m.stiffness(3) @ eps
    assert e0 == pytest.approx(float(eps_r @ rotate_stiffness(m.stiffness(3), rotation_xz(a)) @ eps_r), rel=1e-12)
    # 2D: the identity rotation leaves the reductions untouched; stress and strain still differ
    I3 = rotation_xz(0.0)
    for plane in ("stress", "strain"):
        assert torch.allclose(m.rotated_stiffness(I3, 2, plane), m.stiffness(2, plane), atol=1e-12)
    Cr = m.rotated_stiffness(rotation_xz(0.5), 2)
    assert Cr[0, 2].abs() > 1e-3 and torch.allclose(Cr, Cr.T, atol=1e-14)  # shear coupling, symmetric


def test_hill_index():
    r"""C-MAT-08 — critère de Hill : 1 aux résistances, von Mises si isotrope, aveugle à la pression."""
    from aeroptimizer.materials import Strengths, embed_plane_stress, hill_index

    st = Strengths(X=40.0, Y=40.0, Z=20.0, S23=15.0, S13=15.0, S12=25.0)
    z = torch.zeros(6, dtype=torch.float64)
    for k, strength in enumerate((40.0, 40.0, 20.0, 15.0, 15.0, 25.0)):
        s = z.clone()
        s[k] = strength
        assert hill_index(s, st) == pytest.approx(1.0, rel=1e-12)
    # isotropic strengths X=Y=Z=sy, S=sy/sqrt(3): Hill == von Mises
    sy = 30.0
    iso = Strengths(sy, sy, sy, sy / 3**0.5, sy / 3**0.5, sy / 3**0.5)
    g = torch.Generator().manual_seed(0)
    sig = torch.randn(5, 6, generator=g, dtype=torch.float64) * 10
    s1, s2, s3, t23, t13, t12 = sig.unbind(-1)
    vm2 = 0.5 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2) + 3 * (t23**2 + t13**2 + t12**2)
    assert torch.allclose(hill_index(sig, iso), vm2 / sy**2, rtol=1e-12)
    # blind to the hydrostatic part: a known limitation of the criterion
    assert hill_index(torch.tensor([5.0, 5.0, 5.0, 0, 0, 0], dtype=torch.float64), iso) == pytest.approx(0.0, abs=1e-15)
    # plane-stress embedding and quadratic scaling
    s2d = torch.tensor([[10.0, -4.0, 3.0]], dtype=torch.float64)
    f1 = hill_index(embed_plane_stress(s2d), st)
    assert hill_index(embed_plane_stress(2 * s2d), st) == pytest.approx(float(4 * f1), rel=1e-12)
    with pytest.raises(MaterialError, match="Z"):
        Strengths(X=40.0, Y=40.0, Z=0.0, S23=1.0, S13=1.0, S12=1.0)
