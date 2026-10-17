"""Contract tests for `flow` (CONTRACTS §3.6, lesson 14)."""

import math

import numpy as np
import pytest
import torch

from aeroptimizer.errors import ConfigError
from aeroptimizer.fem import elements as el
from aeroptimizer.fem.operators import StructuredFEOperator
from aeroptimizer.flow import potential
from aeroptimizer.flow.aero import download_force
from aeroptimizer.flow.base import FlowModel, FlowState
from aeroptimizer.flow.potential import PotentialFlow, inlet_load, kappa, slipstream_inlet_profile
from aeroptimizer.grid import Connectivity, Grid


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _disk(R: float, nx: int, nz: int, sub: int = 8) -> torch.Tensor:
    """Area fraction of a disk of radius R (in cells) centred in the box: the diffuse cylinder."""
    o = (np.arange(sub) + 0.5) / sub
    X = np.arange(nx)[:, None, None, None] + o[None, None, :, None]
    Z = np.arange(nz)[None, :, None, None] + o[None, None, None, :]
    return torch.tensor((((X - nx / 2) ** 2 + (Z - nz / 2) ** 2) <= R * R).mean(axis=(2, 3)))


def _speed_error_annulus(R: int, nx: int, nz: int, q: float = 3.0) -> float:
    """Mean |v| error against the unbounded-cylinder solution on the annulus 1.5R < r < 3R."""
    g = Grid((nx, nz), (1.0, 1.0))
    pf = PotentialFlow(g, torch.full((nx,), 1.0), n_pad=4, q=q)
    sp = pf.solve(_disk(R, nx, nz)).v.reshape(nx, nz, 2).pow(2).sum(-1).sqrt().numpy()
    X, Z = np.meshgrid(np.arange(nx) + 0.5, np.arange(nz) + 0.5, indexing="ij")
    r = np.hypot(X - nx / 2, Z - nz / 2)
    a = np.arctan2(np.abs(X - nx / 2), np.abs(Z - nz / 2))  # angle from the flow axis (z)
    s = R**2 / r**2
    exact = np.sqrt(1 + s**2 - 2 * s * np.cos(2 * a))
    sel = (r > 1.5 * R) & (r < 3 * R)
    return float(np.abs(sp - exact)[sel].mean())


def test_flowmodel_protocol():
    r"""C-FLOW-01 — `FlowModel` (protocole) : `solve(rho_bar) -> FlowState{phi, v, p_rel}` ; les modèles (potentiel aujourd'hui, Brinkman–Stokes plus tard) sont interchangeables **sans modifier** `objectives`."""
    g = Grid((6, 8), (1.0, 1.0))
    pf = PotentialFlow(g, torch.full((6,), 1.0), n_pad=0)
    st = pf.solve(torch.zeros(g.shape))
    assert isinstance(st, FlowState)
    assert st.phi.shape == (g.n_nodes,) and st.v.shape == (g.n_elems, 2) and st.p_rel.shape == (g.n_elems,)

    class Uniform:  # a toy model: uniform downward flow, no solve at all
        def solve(self, rho_bar):
            v = torch.zeros(g.n_elems, 2)
            v[:, 1] = -1.0
            return FlowState(phi=torch.zeros(g.n_nodes), v=v, p_rel=torch.zeros(g.n_elems))

    for model in (pf, Uniform()):  # the consumer (download_force) does not know which model it gets
        model: FlowModel
        rho = torch.zeros(g.shape)
        rho[2:4, 3:5] = 1.0
        D = download_force(rho, model.solve(rho).p_rel, h=g.h, rho_air=1.225, u_ref=1.0, c_b=0.5)
        assert torch.isfinite(D)


def test_uniform_flow_exact():
    r"""C-FLOW-02 — **Sans obstacle** (`rho_bar≡0`) : `v = −U e_z` uniforme à `1e-10` ; `phi` linéaire en `z`."""
    for shape, h in [((10, 14), (0.5, 2.0)), ((4, 5, 7), (0.5, 1.0, 2.0))]:
        g = Grid(shape, h)
        U = 2.5
        pf = PotentialFlow(g, torch.full(shape[:-1], U), n_pad=0)
        st = pf.solve(torch.zeros(shape))
        want = torch.zeros(g.n_elems, g.d)
        want[:, -1] = -U
        assert torch.allclose(st.v, want, atol=1e-10)
        z = (torch.arange(g.n_nodes) % (shape[-1] + 1)) * h[-1]
        assert torch.allclose(st.phi, -U * z, atol=1e-10)
        assert torch.allclose(st.p_rel, torch.zeros(g.n_elems), atol=1e-9)  # Bernoulli: no change of speed


def test_flux_conservation():
    r"""C-FLOW-03 — **Conservation** : flux entrant = flux sortant à `1e-8` relatif (somme des réactions en Dirichlet)."""
    g = Grid((40, 60), (1.0, 1.0))
    rho = _disk(6, 40, 60)
    prof = 1.0 + 0.5 * torch.sin(torch.arange(40) / 5.0) ** 2  # non-uniform inlet
    pf = PotentialFlow(g, prof, n_pad=4)
    fin, fout = pf.fluxes(rho)
    assert fin == pytest.approx(float(prof.sum()), rel=1e-12)  # int U dGamma, h = 1
    assert fout == pytest.approx(fin, rel=1e-8)


def test_cylinder_cp():
    r"""C-FLOW-04 — **Cylindre/disque** en veine large : $C_p(\theta)\approx1-4\sin^2\theta$ (2D) ; tolérance documentée (5–10 %, blocage)."""
    # Speed error |v|/U against the unbounded cylinder, mean over the annulus 1.5R < r < 3R, R = 8 cells.
    # The error is dominated by the blockage 2R/W of the slip walls (measured 0.16 -> 0.027 -> 0.016 -> 0.010
    # for W/2R = 1.5, 6, 8, 12), not by the mesh: it shrinks when the channel widens.
    e_narrow = _speed_error_annulus(8, 64, 160)  # blockage 1/4
    e_wide = _speed_error_annulus(8, 128, 160)  # blockage 1/8
    assert e_wide < 0.03  # 2-3 % of U (about 5 % of C_p in the annulus)
    assert e_wide < 0.5 * e_narrow  # the error is blockage, and decreases with it
    # the stagnation point: C_p -> 1 on the axis upstream of the body
    g = Grid((128, 160), (1.0, 1.0))
    st = PotentialFlow(g, torch.full((128,), 1.0), n_pad=4).solve(_disk(8, 128, 160))
    cp = (1.0 - (st.v**2).sum(-1)).reshape(128, 160)  # u_ref = 1
    cp_axis = cp[63:65, 80 + 8 + 2]  # two cells above the stagnation point
    assert float(cp_axis.mean()) > 0.6 and float(cp_axis.max()) <= 1.0 + 1e-9


def test_inlet_padding_enforced():
    r"""C-FLOW-05 — Le tampon fluide d'entrée a une épaisseur ≥ `n_pad` ; un solide en contact avec la face d'entrée lève `ConfigError`."""
    g = Grid((10, 20), (1.0, 1.0))
    pf = PotentialFlow(g, torch.full((10,), 1.0), n_pad=4)
    rho = torch.zeros(g.shape)
    rho[3:6, 5:10] = 1.0  # far from the inlet (z = 20): fine
    pf.solve(rho)
    for k in (19, 16):  # touching the inlet; and inside the 4-element pad
        bad = rho.clone()
        bad[4, k] = 0.5
        with pytest.raises(ConfigError):
            pf.solve(bad)
    ok = rho.clone()
    ok[4, 15] = 1.0  # first layer outside the pad (layers 16..19)
    pf.solve(ok)
    PotentialFlow(g, torch.full((10,), 1.0), n_pad=0).solve(bad)  # the check can be disabled (lesson 14)
    with pytest.raises(ConfigError):
        inlet_load(g, -torch.ones(10))  # U_in is the inward speed: >= 0 (C-CONV-07)


def test_kappa_properties():
    r"""C-FLOW-06 — `kappa(rho_bar)` : `kappa(0)=1`, `kappa(1)=κ_min`, strictement décroissante ; ni NaN ni Inf pour `rho_bar∈[0,1]`."""
    x = torch.linspace(0.0, 1.0, 101, requires_grad=True)
    for q in (1.0, 3.0, 5.0):
        k = kappa(x, 1e-6, q)
        assert float(k[0].detach()) == 1.0 and float(k[-1].detach()) == pytest.approx(1e-6, rel=1e-12)
        assert bool((k[1:] < k[:-1]).all())
        assert bool(torch.isfinite(k).all())
        (g,) = torch.autograd.grad(k.sum(), x)
        assert bool(torch.isfinite(g).all()) and bool((g <= 0).all())


def test_neumann_nullspace():
    r"""C-FLOW-07 — Système SPD avec ≥ 1 ddl Dirichlet (sortie) ; la matrice sans Dirichlet est singulière (rang `Nn−1`) — vérifié pour détecter une configuration CL erronée."""
    g = Grid((5, 6), (1.0, 1.0))
    pf = PotentialFlow(g, torch.full((5,), 1.0), n_pad=0)
    w = torch.ones(g.n_elems)
    K0 = pf.op_neumann.assemble_csr(w).toarray()  # pure Neumann
    assert np.linalg.matrix_rank(K0, tol=1e-10) == g.n_nodes - 1
    assert np.allclose(K0 @ np.ones(g.n_nodes), 0.0, atol=1e-12)  # the constants are the null space
    K = pf.op.assemble_csr(w).toarray()  # with the outlet fixed
    assert np.allclose(K, K.T) and float(np.linalg.eigvalsh(K).min()) > 1e-3  # SPD
    # even with the solid at kappa_min the system stays SPD (barely): no fixed dof would make it singular
    assert float(np.linalg.eigvalsh(pf.op.assemble_csr(pf.conductivity(torch.ones(g.n_elems))).toarray()).min()) > 0


def test_mesh_convergence():
    r"""C-FLOW-08 — Convergence en maillage de $\|\mathbf v\|$ en un point de la veine hors de l'obstacle (ordre ≥ 1)."""
    # Smooth exact solution of the box with slip walls: phi = -z + a cos(kx) sinh(kz)/k, k = pi/W,
    # inlet speed U_in(x) = 1 - a cosh(kL) cos(kx) (element averages). Velocity error at element centres: order 2.
    k, Lz = math.pi, 1.25
    a = 0.5 / math.cosh(k * Lz)

    def error(n: int) -> float:
        nx, nz, h = n, int(Lz * n), 1.0 / n
        xs = torch.arange(nx + 1) * h
        ue = 1.0 - a * math.cosh(k * Lz) * (torch.sin(k * xs[1:]) - torch.sin(k * xs[:-1])) / (k * h)
        g = Grid((nx, nz), (h, h))
        st = PotentialFlow(g, ue, n_pad=0, u_ref=1.0).solve(torch.zeros(nx, nz))
        X, Z = torch.meshgrid((torch.arange(nx) + 0.5) * h, (torch.arange(nz) + 0.5) * h, indexing="ij")
        vx, vz = -a * torch.sin(k * X) * torch.sinh(k * Z), -1 + a * torch.cos(k * X) * torch.cosh(k * Z)
        v = st.v.reshape(nx, nz, 2)
        return float(torch.sqrt((v[..., 0] - vx) ** 2 + (v[..., 1] - vz) ** 2).max())

    errs = [error(n) for n in (8, 16, 32)]
    for e1, e2 in zip(errs, errs[1:]):
        assert math.log2(e1 / e2) > 1.8  # measured 1.96, 1.99: order 2 (contract: >= 1)


def test_uses_generic_operator():
    r"""C-FLOW-09 — Réutilise `StructuredFEOperator` avec `ncomp=1` ; **aucun** assembleur dédié au fluide."""
    g = Grid((6, 8), (1.0, 0.5))
    pf = PotentialFlow(g, torch.full((6,), 1.0), n_pad=0)
    assert isinstance(pf.op, StructuredFEOperator) and pf.op.ncomp == 1
    assert torch.equal(pf.op.Ke0, el.laplace_ke0(2, g.h))
    assert torch.equal(pf.op.edof, Connectivity(g, 1).edof)
    assert not [n for n in dir(potential) if "assembl" in n.lower()]  # no assembler of its own


def test_slipstream_profile():
    r"""Complément de C-FLOW-02 — profil d'entrée du rotor (FORMALISM §4.2) : $U_h=\sqrt{T/(2\rho A)}$ sur l'empreinte."""
    g = Grid((8, 10), (0.01, 0.01))
    mask = torch.zeros(8, dtype=torch.bool)
    mask[2:6] = True
    prof = slipstream_inlet_profile(g, mask, thrust_N=10.0, rotor_area_m2=0.02, rho_air=1.225, k_w=1.5, U_inf=0.3)
    u_h = math.sqrt(10.0 / (2 * 1.225 * 0.02))
    assert torch.allclose(prof[mask], torch.full((4,), 1.5 * u_h))
    assert torch.allclose(prof[~mask], torch.full((4,), 0.3))
    with pytest.raises(ConfigError):
        slipstream_inlet_profile(g, mask, 0.0, 0.02, 1.225, 1.5, 0.0)
