"""Contract tests for `aero_drag` (CONTRACTS §3.6, lesson 15). C-AERO-06 waits for `io/results`."""

import numpy as np
import pytest
import torch

from aeroptimizer.errors import ConfigError
from aeroptimizer.flow.aero import (
    bernoulli_pressure,
    blockage,
    density_gradient,
    download_force,
    fluid_side_pressure,
    pressure_force,
)
from aeroptimizer.flow.potential import PotentialFlow
from aeroptimizer.grid import Grid
from aeroptimizer.objectives.aerodynamic import relative_download

RHO_AIR = 1.225
Q = 0.5 * RHO_AIR  # dynamic pressure at U_ref = 1


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _fraction(inside, nx: int, nz: int, sub: int = 8) -> torch.Tensor:
    o = (np.arange(sub) + 0.5) / sub
    X = np.arange(nx)[:, None, None, None] + o[None, None, :, None]
    Z = np.arange(nz)[None, :, None, None] + o[None, None, None, :]
    return torch.tensor(inside(X, Z).mean(axis=(2, 3)))


def _disk(R, nx, nz):
    return _fraction(lambda X, Z: (X - nx / 2) ** 2 + (Z - nz / 2) ** 2 <= R * R, nx, nz)


def _egg(R, nx, nz, front: float, rear: float):
    """Two half-ellipses of lateral semi-axis R; semi-axis `front` (upstream, z > centre) and `rear` along z."""
    cx, cz = nx / 2, nz / 2
    return _fraction(
        lambda X, Z: ((X - cx) / R) ** 2 + ((Z - cz) / (R * np.where(Z >= cz, front, rear))) ** 2 <= 1, nx, nz
    )


def _frontal_width(rho: torch.Tensor, h) -> float:
    return float(-torch.clamp(density_gradient(rho, h)[..., -1], max=0.0).sum() * h[0])


def _flow(nx: int, nz: int, U: float = 1.0) -> PotentialFlow:
    return PotentialFlow(Grid((nx, nz), (1.0, 1.0)), torch.full((nx,), U), n_pad=4, rho_air=RHO_AIR)


def test_sign_conventions():
    r"""C-CONV-07 — **Signes** : $\mathbf e_f=-\mathbf e_z$ (écoulement descendant) ; poussée moteur $+\mathbf e_z$ ; $D>0$ = force **vers le bas** ; $\mathrm{U_{in}}\ge0$ = vitesse entrante."""
    nx, nz = 48, 80
    pf = _flow(nx, nz)
    rho = _disk(6, nx, nz)
    st = pf.solve(rho)
    assert float(st.v[:, -1].mean()) < 0  # the stream goes DOWN (-e_z)
    # D > 0 means a force pointing down. With a pure base term (p_rel = 0) the leeward suction pulls the body
    # down: F_z < 0 and D = -F_z > 0.
    zero_p = torch.zeros(nx, nz)
    D = download_force(rho, zero_p, h=(1.0, 1.0), rho_air=RHO_AIR, u_ref=1.0, c_b=0.5)
    assert float(D) > 0
    # the same sign from the full pressure force of a body hit from above: stagnation pressure on the top face
    # of a solid slab pushes it down
    p_top = torch.zeros(nx, nz)
    slab = torch.zeros(nx, nz)
    slab[10:30, 20:30] = 1.0
    p_top[10:30, 29:] = 1.0  # overpressure just above the slab
    assert float(pressure_force(slab, p_top, (1.0, 1.0))[-1]) < 0
    with pytest.raises(ConfigError):
        PotentialFlow(Grid((4, 6), (1.0, 1.0)), -torch.ones(4), n_pad=0)  # U_in >= 0


def test_dalembert_and_base_closure():
    r"""C-AERO-01 — **d'Alembert (oracle)** : la force de pression **complète** $\int p\nabla\bar\rho$ d'un corps symétrique avant/arrière est nulle (`|F| ≤ 0.02·½ρU²·L`) ; $D$ à `c_b=0` n'est **pas** cette force (la moitié sous le vent est remplacée) ; $D$ est affine en `c_b`, de pente $\tfrac12\rho U^2A$."""
    nx, nz = 128, 160
    pf = _flow(nx, nz)
    rho = _disk(8, nx, nz)
    st = pf.solve(rho)
    A = _frontal_width(rho, (1.0, 1.0))
    F = pressure_force(rho, st.p_rel, (1.0, 1.0))
    assert float(F.abs().max()) <= 0.02 * Q * A  # measured 4e-4 (fore/aft symmetry cancels the interface bias)
    kw = dict(h=(1.0, 1.0), rho_air=RHO_AIR, u_ref=1.0)
    D = [float(download_force(rho, st.p_rel, c_b=c, **kw)) for c in (0.0, 0.5, 1.0)]
    assert D[1] - D[0] == pytest.approx(0.5 * Q * A, rel=1e-9)  # slope in c_b: q A (the base term only)
    assert D[2] - D[1] == pytest.approx(0.5 * Q * A, rel=1e-9)
    # with the windward pressure alone (c_b = 0) D is NOT ~ 0: potential suction on the shoulders gives D < 0
    assert D[0] < -0.1 * Q * A
    # a smooth asymmetric body breaks the cancellation: the residual force is the discretisation noise floor
    egg = _egg(8, nx, nz, 1.0, 3.0)
    Fz = float(pressure_force(egg, pf.solve(egg).p_rel, (1.0, 1.0))[-1])
    assert 0.0 < abs(Fz) < 0.3 * Q * _frontal_width(egg, (1.0, 1.0))  # measured 0.19 q A at R = 8


def test_dynamic_pressure_scaling():
    r"""C-AERO-02 — Invariance d'échelle : à `ρ_a, U_ref` fixés, `D̂=D/T_tot` ne dépend que de `c_b` et de la forme ; doubler `U_ref` multiplie `D` par 4 (± `1e-6`)."""
    nx, nz = 64, 96
    rho = _disk(6, nx, nz)
    Ds = []
    for U in (1.0, 2.0):
        pf = _flow(nx, nz, U)
        Ds.append(
            download_force(rho, pf.solve(rho).p_rel, h=(1.0, 1.0), rho_air=RHO_AIR, u_ref=pf.u_ref, c_b=0.4)
        )
    assert float(Ds[1] / Ds[0]) == pytest.approx(4.0, rel=1e-6)
    assert float(relative_download(Ds[1], 8.0)) == pytest.approx(float(Ds[1]) / 8.0, rel=1e-15)
    with pytest.raises(ConfigError):
        relative_download(Ds[0], 0.0)


def test_zero_for_trivial_fields():
    r"""C-AERO-03 — `D` est nul si `rho_bar≡0` (pas de corps) et si `rho_bar≡1` (domaine plein, `∇ρ̄=0`)."""
    p = torch.randn(12, 15)
    for fill in (0.0, 1.0):
        rho = torch.full((12, 15), fill)
        assert float(download_force(rho, p, h=(0.5, 2.0), rho_air=RHO_AIR, u_ref=3.0, c_b=0.4)) == 0.0
        assert float(pressure_force(rho, p, (0.5, 2.0)).abs().max()) == 0.0


def test_streamlining_reduces_D():
    r"""C-AERO-04 — **Ce que la fermeture voit de la forme** : le terme de culot vaut exactement $c_b\,\tfrac12\rho U^2 A$ ($A$ = largeur frontale) **quelle que soit la forme sous le vent** ; seule la face au vent (pression potentielle) et $A$ distinguent deux corps. (Le « carénage » n'est donc pas vu par un $c_b$ constant — voir leçon 15.)"""
    nx, nz = 96, 144
    h = (1.0, 1.0)
    zero_p = torch.zeros(nx, nz)
    kw = dict(h=h, rho_air=RHO_AIR, u_ref=1.0, c_b=0.4)
    blunt = _egg(10, nx, nz, 1.0, 1.0)  # round rear
    tapered = _egg(10, nx, nz, 1.0, 5.0)  # same front, long tapered rear
    flat = _fraction(lambda X, Z: (abs(X - nx / 2) <= 10) & (abs(Z - nz / 2) <= 3), nx, nz)
    for rho in (blunt, tapered, flat):
        A = _frontal_width(rho, h)
        assert float(download_force(rho, zero_p, **kw)) == pytest.approx(0.4 * Q * A, rel=1e-9)
    # same front half + same A => same windward term, whatever the rear: identical D with the same pressure
    p = torch.randn(nx, nz) * 0 + 0.3
    p[:, : nz // 2] = 0.0  # pressure only on the upstream half-plane
    Db = float(download_force(blunt, p, **kw))
    Dt = float(download_force(tapered, p, **kw))
    assert Db == pytest.approx(Dt, rel=1e-9)
    # D grows with the frontal width, all else equal
    wide = _egg(14, nx, nz, 1.0, 1.0)
    assert float(download_force(wide, zero_p, **kw)) > float(download_force(blunt, zero_p, **kw))


def test_gradcheck():
    r"""C-AERO-05 — `D` différentiable par rapport à `rho_bar` (chemin direct `∇ρ̄` + chemin via `φ`) ; `gradcheck` float64, sous-gradient accepté aux points de non-différentiabilité de `min/max`."""
    nx, nz = 8, 12
    g = Grid((nx, nz), (0.5, 0.8))
    pf = PotentialFlow(g, torch.full((nx,), 1.5), n_pad=2, rho_air=RHO_AIR)
    gen = torch.Generator().manual_seed(3)
    base = 0.8 * torch.rand(nx, nz, generator=gen)
    base[:, -2:] = 0.0  # inlet pad
    mask = torch.ones(nx, nz)
    mask[:, -2:] = 0.0

    def D(r, fluid_side=False):
        rho = r * mask  # keep the pad empty
        p = pf.solve(rho).p_rel
        if fluid_side:
            p = fluid_side_pressure(p, rho.reshape(nx, nz))
        return download_force(rho.reshape(nx, nz), p, h=g.h, rho_air=RHO_AIR, u_ref=pf.u_ref, c_b=0.4)

    for fs in (False, True):
        r = base.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(lambda x: D(x, fs), (r,), eps=1e-6, atol=1e-6, rtol=1e-4)
    # path via phi only matters: freeze the gradient of rho in the formula and differentiate the pressure
    r = base.clone().requires_grad_(True)
    (gr,) = torch.autograd.grad(D(r), r)
    assert bool(torch.isfinite(gr).all()) and float(gr.abs().max()) > 0


def test_fluid_side_pressure():
    r"""Complément de C-AERO-01 — `fluid_side_pressure` : l'identité hors corps, la moyenne pondérée par $(1-\bar\rho)^m$, et la réduction du biais d'interface sur le cylindre ($\langle C_p\rangle_{\rm face}$ analytique $=-1/3$)."""
    p = torch.randn(10, 12)
    eps = 1e-2
    # k = 0 and rho = 0: weight 1, so p / (1 + eps); an all-solid field has no fluid side: the estimate is 0
    assert torch.allclose(fluid_side_pressure(p, torch.zeros(10, 12), m=2.0, k=0), p / (1 + eps))
    assert torch.allclose(fluid_side_pressure(p, torch.ones(10, 12)), torch.zeros(10, 12))
    uniform = torch.full((10, 12), 3.0)
    assert torch.allclose(fluid_side_pressure(uniform, torch.zeros(10, 12)), uniform / (1 + eps))
    # rho_c < 1: cells with rho >= rho_c carry no weight, so a garbage pressure there cannot leak into
    # their neighbours (lesson 17: velocity inside a solid is a potential drop over a thin layer, not a speed)
    rho = torch.zeros(10, 12)
    rho[3:7, 4:8] = 0.9
    junk = torch.zeros(10, 12)
    junk[3:7, 4:8] = -1e4
    out = fluid_side_pressure(junk + 1.0, rho, rho_c=0.5)
    assert float(out[5, 6]) == 0.0 and float(out.min()) > -1.0  # solid interior: 0; fluid cells see only fluid
    assert float(fluid_side_pressure(junk, rho, rho_c=1.0).min()) < -100.0  # with rho_c = 1 the junk leaks
    # cylinder: potential-flow average of C_p on the windward projected area
    nx, nz = 256, 320
    rho = _disk(16, nx, nz)
    pf = _flow(nx, nz)
    p = pf.solve(rho).p_rel

    def mean_cp(pp):
        w = -torch.clamp(density_gradient(rho, (1.0, 1.0))[..., -1], max=0.0)
        return float((w * pp.reshape(nx, nz)).sum() / w.sum()) / Q

    raw, corrected = mean_cp(p), mean_cp(fluid_side_pressure(p, rho))
    assert raw < -0.6  # measured -0.72: the interface cells over-accelerate the flow
    assert abs(corrected + 1.0 / 3.0) < 0.1 and abs(corrected + 1.0 / 3.0) < 0.5 * abs(raw + 1.0 / 3.0)


def test_blockage_indicator():
    r"""Complément de C-AERO-05 — `blockage` : $C_{\rm eff}=\mathbf b^\top\boldsymbol\varphi/\delta\varphi$ ; égal à $UW$ sans obstacle (pour $\delta\varphi=UL$) et croissant avec l'obstruction."""
    nx, nz = 40, 60
    pf = _flow(nx, nz)
    ref = float(blockage(pf.solve(torch.zeros(nx, nz)).phi, pf.b, 1.0 * nz))
    assert ref == pytest.approx(nx * 1.0, rel=1e-10)
    vals = []
    for half in (4, 8, 12):
        rho = torch.zeros(nx, nz)
        rho[nx // 2 - half : nx // 2 + half, nz // 2 - 2 : nz // 2 + 2] = 1.0
        vals.append(float(blockage(pf.solve(rho).phi, pf.b, 1.0 * nz)))
    assert ref < vals[0] < vals[1] < vals[2]


def test_params_recorded_in_results():
    r"""C-AERO-06 — Valeur de `c_b`, de `k_w` et de `κ` **explicitement** présentes dans la sortie de résultat (traçabilité des paramètres calibrés)."""
    pytest.skip("waits for io/results.py (phase 2/4)")


def test_floor_at_base():
    r"""Complément de C-AERO-05 — `floor_at_base` : la pression au vent est bornée par la pression de culot, donc $D\ge0$ pour tout champ et le gradient est nul là où le plancher est actif ; sans plancher, une aspiration de type arête vive rend $D<0$."""
    gen = torch.Generator().manual_seed(1)
    kw = dict(h=(1.0, 1.0), rho_air=RHO_AIR, u_ref=1.0, c_b=0.8)
    for body in (_disk(5, 24, 40), _egg(4, 24, 40, 1.0, 2.0)):  # closed bodies inside the box: A_amont = A_aval
        for _ in range(5):
            p = -50.0 * torch.rand(24, 40, generator=gen) * Q  # strong suction everywhere
            assert float(download_force(body, p, floor_at_base=True, **kw)) >= -1e-9
    rho = _disk(5, 24, 30)
    suction = torch.full((24, 30), -20.0 * Q)
    assert float(download_force(rho, suction, **kw)) < 0.0  # unprotected: the optimiser's loophole
    free = suction.clone().requires_grad_(True)
    download_force(rho, free, floor_at_base=True, **kw).backward()
    assert float(free.grad.abs().max()) == 0.0  # every windward cell is on the floor: no gradient
    # above the floor nothing changes
    p_ok = torch.full((24, 30), 0.3 * Q)
    assert float(download_force(rho, p_ok, floor_at_base=True, **kw)) == float(download_force(rho, p_ok, **kw))
