"""Contract tests for `density` — stubs generated from CONTRACTS.md (scripts/contracts_index.py).

Replace a stub body when its phase is reached. Done: C-FILT-01..04, C-PROJ-01..03, C-SIMP-01, C-MASK-01.
"""

import numpy as np
import pytest
import torch
from reference.filter_numpy import apply_filter, filter_matrix

from aeroptimizer.errors import ConfigError
from aeroptimizer.grid import Grid, Masks
from aeroptimizer.topopt.density import DensityFilter, DensityPipeline, heaviside_projection, simp


SHAPES = {2: ((7, 5), (0.5, 1.25)), 3: ((4, 5, 3), (0.5, 1.0, 1.5))}  # non-uniform h: catches a mixed-up axis


def _filter(d, R):
    shape, h = SHAPES[d]
    return DensityFilter(Grid(shape, h), R), shape, h


@pytest.mark.parametrize("d", [2, 3])
def test_filter_preserves_constants(d):
    r"""C-FILT-01 — Le filtre conserve les constantes : `filter(c·1)=c·1` à `1e-12`, **y compris aux bords** (normalisation `w*1`)."""
    flt, shape, _ = _filter(d, 2.2)
    for c in (0.0, 0.3, 1.0):
        assert (flt(torch.full(shape, c)) - c).abs().max() < 1e-12
    # and the filter agrees with the dense double-loop oracle, borders included
    x = torch.rand(shape, generator=torch.Generator().manual_seed(0))
    W = filter_matrix(shape, SHAPES[d][1], 2.2)
    assert np.abs(flt(x).reshape(-1).numpy() - apply_filter(W, x.numpy())).max() < 1e-12


@pytest.mark.parametrize("d", [2, 3])
def test_filter_identity_small_radius(d):
    r"""C-FILT-02 — `R→0` (`R ≤ min h`) : identité. Le noyau est symétrique et de somme positive."""
    shape, h = SHAPES[d]
    x = torch.rand(shape, generator=torch.Generator().manual_seed(1))
    flt = DensityFilter(Grid(shape, h), min(h))
    assert torch.equal(flt(x), x)
    big = DensityFilter(Grid(shape, h), 2.0 * max(h))
    k = big.kernel[0, 0]
    assert torch.allclose(k, torch.flip(k, tuple(range(d))))  # symmetric
    assert float(k.sum()) > 0 and float(k.min()) >= 0
    with pytest.raises(ConfigError):
        DensityFilter(Grid(shape, h), 0.0)


@pytest.mark.parametrize("d", [2, 3])
def test_filter_range(d):
    r"""C-FILT-03 — Sortie dans `[0,1]` si l'entrée l'est (convexité)."""
    flt, shape, _ = _filter(d, 2.0)
    g = torch.Generator().manual_seed(2)
    for x in (torch.rand(shape, generator=g), (torch.rand(shape, generator=g) > 0.5).double()):
        y = flt(x)
        assert float(y.min()) >= -1e-15 and float(y.max()) <= 1.0 + 1e-15


@pytest.mark.parametrize("d", [2, 3])
def test_filter_gradcheck(d):
    r"""C-FILT-04 — Différentiable ; gradient = filtre transposé normalisé (vérifié par `gradcheck`)."""
    flt, shape, h = _filter(d, 1.8)
    x = torch.rand(shape, generator=torch.Generator().manual_seed(3)).requires_grad_()
    assert torch.autograd.gradcheck(flt, (x,), eps=1e-6, atol=1e-8)
    c = torch.rand(shape, generator=torch.Generator().manual_seed(4))
    (c * flt(x)).sum().backward()
    W = filter_matrix(shape, h, 1.8)
    expected = W.T @ (c.reshape(-1).numpy() / W.sum(axis=1))  # W^T diag(1 / W1) c
    assert np.abs(x.grad.reshape(-1).numpy() - expected).max() < 1e-12


def test_projection_endpoints():
    r"""C-PROJ-01 — `Heaviside(β,η)` : `f(0)=0`, `f(1)=1` (`1e-12`), croissante, `f(η)=η` pour tout `β` si `η=0.5`."""
    r = torch.linspace(0.0, 1.0, 1001)
    for beta in (0.1, 1.0, 4.0, 16.0, 64.0):
        for eta in (0.3, 0.5, 0.7):
            y = heaviside_projection(r, beta, eta)
            assert abs(y[0].item()) < 1e-12 and abs(y[-1].item() - 1.0) < 1e-12
            assert bool((y[1:] >= y[:-1]).all())
        assert abs(heaviside_projection(torch.tensor(0.5), beta, 0.5).item() - 0.5) < 1e-12
    with pytest.raises(ConfigError):
        heaviside_projection(r, -1.0)


def test_projection_limits():
    r"""C-PROJ-02 — `β=0` **désactive** la projection (identité exacte, équivalence top88) ; `β→∞` ⇒ seuil."""
    r = torch.rand(50, generator=torch.Generator().manual_seed(5))
    assert torch.equal(heaviside_projection(r, 0.0), r)
    assert (heaviside_projection(r, 1e-6) - r).abs().max() < 1e-6  # continuous at beta = 0
    y = heaviside_projection(r, 500.0)
    step = (r > 0.5).double()
    away = (r - 0.5).abs() > 0.05
    assert (y[away] - step[away]).abs().max() < 1e-10


def _pipeline_setup(shape=(8, 6), beta=2.0):
    g = Grid(shape, (1.0, 1.0))
    solid = torch.zeros(shape, dtype=torch.bool)
    void = torch.zeros(shape, dtype=torch.bool)
    solid[0, :2] = True
    void[-2:, -2:] = True
    masks = Masks(design=~(solid | void), solid=solid, void=void)
    return g, masks, DensityPipeline(g, masks, R=2.0, beta=beta)


def test_beta_continuation_no_rebuild():
    r"""C-PROJ-03 — `β` est un attribut mutable **hors graphe** (continuation) ; changer `β` ne reconstruit pas le plan."""
    _, _, pipe = _pipeline_setup()
    x = torch.rand(8, 6, generator=torch.Generator().manual_seed(6))
    kernel_ptr, norm_ptr = pipe.filter.kernel.data_ptr(), pipe.filter.norm.data_ptr()
    out1 = pipe(x)
    pipe.beta = 8.0
    out2 = pipe(x)
    assert not torch.allclose(out1, out2)  # beta is used...
    assert (pipe.filter.kernel.data_ptr(), pipe.filter.norm.data_ptr()) == (kernel_ptr, norm_ptr)  # ...plan untouched
    assert list(pipe.parameters()) == [] and "beta" not in pipe.state_dict()  # not a learnable / saved tensor
    pipe.beta = 0.0
    assert torch.allclose(pipe(x), pipe.filter(x * pipe.m_D + pipe.m_S) * pipe.keep + pipe.m_S)  # identity at 0


def test_masks_enforced():
    r"""C-MASK-01 — Après le pipeline : `rho_bar[S]=1` et `rho_bar[V]=0` **exactement** ; les masques sont réappliqués après le filtre de surplomb (FORMALISM §2.5)."""
    _, masks, pipe = _pipeline_setup()
    x = torch.rand(8, 6, generator=torch.Generator().manual_seed(7)).requires_grad_()
    out = pipe(x)
    assert bool((out[masks.solid] == 1.0).all()) and bool((out[masks.void] == 0.0).all())
    assert float(out.detach().min()) >= 0.0 and float(out.detach().max()) <= 1.0
    out.sum().backward()
    assert not torch.isnan(x.grad).any()
    assert bool((x.grad[masks.solid | masks.void] == 0).all())  # no sensitivity through passive zones
    # a design value written on a passive zone has no effect at all
    x2 = x.detach().clone()
    x2[masks.solid] = 0.123
    assert torch.equal(pipe(x2), out.detach())
    # solid passive zones still influence their neighbours through the filter
    empty = pipe(torch.zeros(8, 6))
    assert float(empty[1, 0]) > 0.0 and float(empty[5, 0]) == 0.0  # next to the solid zone vs far from it


@pytest.mark.parametrize("p", [1.0, 2.0, 3.0, 4.5])
def test_simp_properties(p):
    r"""C-SIMP-01 — `s(0)=ε_s`, `s(1)=1`, strictement croissante, `s'≥0`."""
    eps_s = 1e-9
    assert simp(torch.tensor(0.0), p, eps_s).item() == eps_s
    assert abs(simp(torch.tensor(1.0), p, eps_s).item() - 1.0) < 1e-15
    rho = torch.linspace(0.0, 1.0, 201).requires_grad_()
    s = simp(rho, p, eps_s)
    assert bool((s[1:] > s[:-1]).all())
    s.sum().backward()
    assert bool((rho.grad >= 0).all())
    # the slope at rho = 1 is (1 - eps_s) p: this is what the sensitivities use
    assert abs(rho.grad[-1].item() - (1 - eps_s) * p) < 1e-12


def test_simp_rejects_bad_parameters():
    """p < 1 (infinite slope at 0) and eps_s outside (0, 1) (singular or ill-posed K~) are config errors."""
    with pytest.raises(ConfigError):
        simp(torch.rand(3), 0.5)
    with pytest.raises(ConfigError):
        simp(torch.rand(3), 3.0, eps_s=0.0)
