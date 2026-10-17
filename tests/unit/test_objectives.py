"""Contract tests for `objectives` (CONTRACTS §3.7, lesson 17)."""

import numpy as np
import pytest
import torch

from aeroptimizer.errors import ConfigError
from aeroptimizer.fem.solvers import ScipyDirect
from aeroptimizer.objectives.composite import Composite
from aeroptimizer.objectives.constraints import volume_constraint, volume_fraction
from aeroptimizer.optim.continuation import Schedule
from aeroptimizer.optim.loop import run
from aeroptimizer.optim.mma import MMA
from aeroptimizer.problem import CoupledProblem

SMALL = dict(margin_x=6, pad_top=4, pad_bottom=2, R=1.0, volfrac=0.4)


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


class CountingSolver:
    """ScipyDirect that counts factorisations (`prepare`) and solves."""

    def __init__(self):
        self.inner, self.prepares, self.solves = ScipyDirect(), 0, 0

    def prepare(self, op, w):
        self.prepares += 1
        handle = self.inner.prepare(op, w)
        outer = self

        class Handle:
            def solve(self, b):
                outer.solves += 1
                return handle.solve(b)

        return Handle()


def _problem(wa=0.5, refs=None, **kw):
    return CoupledProblem(8, 4, weights={"C": 1.0 - wa, "D": wa}, refs=refs, **{**SMALL, **kw})


def _x(pb, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (0.2 + 0.6 * torch.rand(pb.n_design, generator=g)).requires_grad_(True)


def test_single_solve_per_eval():
    r"""C-OBJ-01 — Toute fonction objectif est `f(rho_bar, state) -> scalar` avec `state` le cache des solves (une seule résolution mécanique et une seule résolution fluide par évaluation)."""
    mech, flow = CountingSolver(), CountingSolver()
    pb = _problem(solver=mech, flow_solver=flow)
    x = _x(pb)
    res = pb.evaluate(x)
    assert (mech.prepares, flow.prepares) == (1, 1)  # one factorisation each, however many terms use them
    assert (mech.solves, flow.solves) == (1, 1)
    u, st = res.state  # the cache handed to the terms
    assert u.shape == (pb.grid.n_dofs(2),) and st.phi.shape == (pb.grid.n_nodes,)
    torch.autograd.grad(res.J, x)  # the backward is the adjoint: ONE more solve per system, same handle
    assert (mech.prepares, flow.prepares) == (1, 1)
    assert (mech.solves, flow.solves) == (2, 2)


def test_terms_reported():
    r"""C-OBJ-02 — `Composite.value()` renvoie `J`, ainsi que **chaque terme brut et normalisé** (`C`, `Ĉ`, `D`, `D̂`) — pas seulement la somme."""
    pb = _problem(wa=0.3)
    res = pb.evaluate(_x(pb))
    for k in ("C", "D", "C_hat", "D_hat", "vol"):
        assert k in res.terms and bool(torch.isfinite(res.terms[k]))
    assert float(res.terms["C_hat"].detach()) == pytest.approx(1.0) and float(res.terms["D_hat"].detach()) == pytest.approx(1.0)  # first call freezes
    assert float(res.J) == pytest.approx(0.7 * float(res.terms["C_hat"].detach()) + 0.3 * float(res.terms["D_hat"].detach()))
    # a zero weight still reports the term; its normalised value is just not in J
    only_d = _problem(wa=1.0).evaluate(_x(pb))
    assert float(only_d.J) == pytest.approx(float(only_d.terms["D_hat"]))
    assert bool(torch.isfinite(only_d.terms["C_hat"]))


def test_refs_frozen():
    r"""C-OBJ-03 — Normalisation $\mathcal C_{\rm ref}$, $D_{\rm ref}$ fixée **une fois** (au démarrage ou à la fin de la 1ʳᵉ phase de continuation) et figée ensuite ; changer de référence en cours de route est interdit (déplacerait l'optimum) ; une référence $\le0$ est refusée (elle retournerait le signe du terme)."""
    pb = _problem()
    pb.evaluate(_x(pb, 0))
    refs = dict(pb.composite.refs)
    pb.evaluate(_x(pb, 1))  # a different design: the references do not move
    assert pb.composite.refs == refs
    with pytest.raises(ConfigError):
        pb.composite.freeze_references({"C": torch.tensor(1.0), "D": torch.tensor(1.0)})
    explicit = _problem(refs={"C": 10.0, "D": 4.0})
    assert explicit.composite.refs == {"C": 10.0, "D": 4.0}
    for bad in ({"C": 1.0, "D": -3.0}, {"C": 0.0, "D": 1.0}, {"C": float("nan"), "D": 1.0}):
        with pytest.raises(ConfigError):
            Composite({"C": 0.5, "D": 0.5}, refs=bad)
    Composite({"C": 1.0, "D": 0.0}, refs={"C": 2.0, "D": -1.0})  # an unused term may have no usable reference


def test_volume_constraint():
    r"""C-OBJ-04 — `volume_fraction(rho_bar)` correct sur zones passives (dénominateur : $|\Omega_{\mathcal D\cup\mathcal S}|$) ; `g_V` du bon signe."""
    rho = torch.zeros(6, 4)
    mask = torch.zeros(6, 4, dtype=torch.bool)
    mask[:4, :] = True  # D ∪ S: 16 cells, the last 2 columns are passive void
    rho[:2, :] = 1.0  # includes 8 cells of full density
    rho[4:, :] = 1.0  # a (wrong) density on the void: must not count
    assert float(volume_fraction(rho, mask)) == pytest.approx(0.5)
    assert float(volume_constraint(rho, mask, 0.4)) == pytest.approx(0.1)  # too much material: g > 0
    assert float(volume_constraint(rho, mask, 0.5)) == pytest.approx(0.0)
    assert float(volume_constraint(rho, mask, 0.8)) < 0.0
    assert float(volume_constraint(rho, mask, 0.4, relative=True)) == pytest.approx(0.5 / 0.4 - 1.0)
    with pytest.raises(ConfigError):
        volume_fraction(rho, torch.zeros(6, 4, dtype=torch.bool))
    with pytest.raises(ConfigError):
        volume_constraint(rho, mask, 1.5)


def test_weights_validation():
    r"""C-OBJ-05 — Poids `w_c,w_a ≥ 0`, au moins un > 0 ; sinon `ConfigError`."""
    for bad in ({"C": 0.0, "D": 0.0}, {"C": -1.0, "D": 1.0}, {"C": float("nan"), "D": 1.0}, {}):
        with pytest.raises(ConfigError):
            Composite(bad)
    Composite({"C": 1.0, "D": 0.0})
    Composite({"C": 0.0, "D": 2.5})  # weights need not sum to 1


@pytest.mark.parametrize("wa", [0.0, 0.5, 1.0])
def test_composite_gradient(wa):
    r"""C-OBJ-06 — Gradient du coût composite : $\nabla_x J$ (Autograd à travers les deux adjoints, le filtre, la projection, la pression côté fluide) égal aux différences finies centrées en float64."""
    pb = _problem(wa=wa)
    x = _x(pb, 3)
    pb.evaluate(x.detach().clone().requires_grad_(False))  # freeze the references at this design
    pb.set_continuation({"p": 3.0, "beta": 2.0})
    (g,) = torch.autograd.grad(pb.evaluate(x).J, x)
    gen = torch.Generator().manual_seed(7)
    for _ in range(3):
        d = torch.randn(pb.n_design, generator=gen)
        d /= d.norm()
        h = 1e-6
        fd = (float(pb.evaluate(x.detach() + h * d).J) - float(pb.evaluate(x.detach() - h * d).J)) / (2 * h)
        assert float(g @ d) == pytest.approx(fd, rel=1e-4, abs=1e-8)


def test_coupled_descent():
    r"""C-OBJ-07 — Bout en bout : quelques itérations de MMA sur le problème couplé baissent $J$, respectent le volume, et un poids plus grand sur $D$ donne un $D$ plus faible et un $C$ plus grand (à budget égal, mêmes références) ; les références viennent du design compliance **convergé**, pas du premier itéré."""
    out = {}
    refs = None
    sched = Schedule(p=(3.0, 3.0), beta=(1.0, 4.0), beta_s=(1.0, 1.0), every=20)
    for wa in (0.0, 0.9):
        pb = _problem(wa=wa, refs=refs)
        r = run(pb, MMA(), sched, max_iter=60, tol=1e-6)
        first, last = r.history[0], r.history[-1]
        assert last["J"] < first["J"]
        assert last["g_V"] <= 1e-2  # volume constraint satisfied (relative g)
        out[wa] = last
        if refs is None:
            # references from the CONVERGED compliance design, not from the first iterate: the uniform fuzzy
            # design has a meaningless D (D / D_ref then reaches thousands once the design sharpens)
            pb.set_continuation(sched.at(59))
            with torch.no_grad():
                t = pb.terms(pb.rho_bar(torch.from_numpy(r.x)))[0]
            refs = {"C": float(t["C"]), "D": float(t["D"])}
    assert out[0.9]["D"] < out[0.0]["D"] and out[0.9]["C"] > out[0.0]["C"]


def test_layout_validation():
    r"""Complément de C-FLOW-05 — le problème couplé refuse un tampon d'entrée de moins de 4 éléments et une région de conception trop petite."""
    with pytest.raises(ConfigError):
        CoupledProblem(8, 4, weights={"C": 1.0}, volfrac=0.4, R=1.0, pad_top=3)
    with pytest.raises(ConfigError):
        CoupledProblem(1, 4, weights={"C": 1.0}, volfrac=0.4, R=1.0)
    pb = _problem()
    assert pb.n_design == 32 and pb.initial_design().shape == (32,) and np.allclose(pb.initial_design(), 0.4)
    rho = pb.rho_bar(torch.full((32,), 0.4))
    assert float(rho[:, -4:].abs().max()) == 0.0 and float(rho[8:, :].abs().max()) == 0.0  # pads and margin stay empty
