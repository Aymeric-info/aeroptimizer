"""Contract tests for `bc` (CONTRACTS §3.5): C-BC-01..04, plus the registry of named cases."""

import pytest
import torch

from aeroptimizer.errors import BoundaryConditionError, ConfigError
from aeroptimizer.fem import elements as el
from aeroptimizer.fem.adjoint import solve_spd
from aeroptimizer.fem.bc import (
    Dirichlet,
    DistributedForce,
    build_case,
    combine,
    node_indices,
)
from aeroptimizer.fem.operators import StructuredFEOperator
from aeroptimizer.fem.solvers import ScipyDirect
from aeroptimizer.grid import Connectivity, Grid
from aeroptimizer.materials import Orthotropic


def _nodes(grid, where):
    """Node mask from a predicate on the integer multi-index columns."""
    return where(node_indices(grid).T)


def test_detects_rigid_body_unconstrained():
    r"""C-BC-01 — `Dirichlet(nodes_mask, components)` produit un `fixed_mask: (Nu,) bool`. Au moins 1 ddl bloqué par mode rigide (3 en 2D, 6 en 3D) sinon `BoundaryConditionError`."""
    g = Grid((4, 3), (1.0, 1.0))
    left = _nodes(g, lambda ix: ix[0] == 0)
    fm = Dirichlet(left, (0, 1)).fixed_mask(g, 2)  # clamped edge: fine
    assert fm.shape == (g.n_dofs(2),) and fm.dtype == torch.bool and int(fm.sum()) == 2 * 4
    with pytest.raises(BoundaryConditionError):  # u_x only: translation in z and rotation are free
        Dirichlet(left, (0,)).fixed_mask(g, 2)
    one = _nodes(g, lambda ix: (ix[0] == 0) & (ix[1] == 0))
    with pytest.raises(BoundaryConditionError):  # one node: the rotation about it is free
        Dirichlet(one, (0, 1)).fixed_mask(g, 2)
    two = _nodes(g, lambda ix: (ix[1] == 0) & ((ix[0] == 0) | (ix[0] == 4)))
    assert int(Dirichlet(two, (0, 1)).fixed_mask(g, 2).sum()) == 4  # two nodes block all 3 modes


def test_rigid_body_check_is_not_a_dof_count():
    """3D: six fixed dofs are not enough if they are all on a line (rotation about it is free)."""
    g = Grid((3, 3, 3), (1.0, 1.0, 1.0))
    on_line = _nodes(g, lambda ix: (ix[1] == 0) & (ix[2] == 0) & (ix[0] <= 1))
    with pytest.raises(BoundaryConditionError):
        Dirichlet(on_line, (0, 1, 2)).fixed_mask(g, 3)
    triangle = _nodes(g, lambda ix: ((ix[0] == 0) & (ix[1] == 0) & (ix[2] == 0))
                      | ((ix[0] == 3) & (ix[1] == 0) & (ix[2] == 0))
                      | ((ix[0] == 0) & (ix[1] == 3) & (ix[2] == 0)))
    assert int(Dirichlet(triangle, (0, 1, 2)).fixed_mask(g, 3).sum()) == 9
    face = _nodes(g, lambda ix: ix[0] == 0)
    assert int(Dirichlet(face, (0, 1, 2)).fixed_mask(g, 3).sum()) == 3 * 16


def test_combine_checks_the_union_and_validates_input():
    g = Grid((4, 3), (1.0, 1.0))
    ix = node_indices(g).T
    sym = Dirichlet(ix[0] == 0, (0,))
    roller = Dirichlet((ix[0] == 4) & (ix[1] == 0), (1,))
    with pytest.raises(BoundaryConditionError):
        sym.fixed_mask(g, 2)  # alone, not enough
    assert int(combine(g, 2, sym, roller).sum()) == 4 + 1  # together: symmetric MBB
    with pytest.raises(ConfigError):
        Dirichlet(ix[0] == 0, (2,)).fixed_mask(g, 2)  # no component 2 in 2D
    with pytest.raises(ConfigError):
        Dirichlet(torch.ones(3, dtype=torch.bool), (0,)).fixed_mask(g, 2)  # wrong node count
    # scalar field (potential flow): no rigid-body check
    assert int(Dirichlet(ix[0] == 0, (0,)).fixed_mask(g, 1).sum()) == 4


def test_loadcase_signature():
    r"""C-BC-02 — `LoadCase.assemble(rho_bar) -> f: (Nu,)` ; en v1 constante, mais la signature accepte `rho_bar`."""
    g = Grid((5, 4), (1.0, 1.0))
    _, load, _ = build_case("cantilever", g)
    f1 = load.assemble(torch.rand(g.shape))
    f2 = load.assemble(torch.zeros(g.shape))
    assert f1.shape == (g.n_dofs(2),)
    assert torch.equal(f1, f2)  # constant in v1


@pytest.mark.parametrize("case,shape", [("mbb", (12, 6)), ("cantilever", (10, 4)), ("cantilever", (6, 3, 3))])
def test_reaction_equilibrium(case, shape):
    r"""C-BC-03 — Bilan d'équilibre : somme des forces par direction = somme des réactions aux ddl bloqués au signe près, à `1e-8·‖f‖`."""
    g = Grid(shape, (1.0,) * len(shape))
    d = g.d
    fixed, load, _ = build_case(case, g)
    f = load.assemble(torch.ones(g.shape))
    Ke0 = el.elasticity_ke0(d, g.h, Orthotropic.isotropic(1.0, 0.3).stiffness(d))
    conn = Connectivity(g, d)
    op = StructuredFEOperator(g, conn, Ke0, d, fixed)
    free_op = StructuredFEOperator(g, conn, Ke0, d, torch.zeros_like(fixed))  # no constraint: raw K
    w = 1e-3 + torch.rand(g.n_elems, generator=torch.Generator().manual_seed(1)) ** 3
    u = solve_spd(op, w, f, ScipyDirect())
    reactions = torch.where(fixed, free_op.matvec(w, u) - f, torch.zeros_like(f))  # K u - f on fixed dofs
    total = (f + reactions).reshape(-1, d).sum(0)  # applied + reaction = 0 per direction
    assert float(total.abs().max()) <= 1e-8 * float(f.norm())
    assert float(f.reshape(-1, d).sum(0).abs().max()) > 0.5  # the test is not vacuous


def test_distributed_load_invariant():
    r"""C-BC-04 — Charges réparties : le total ne dépend pas de la finesse du maillage (même résultante pour `n` et `2n`)."""
    for shape in [(6, 4), (12, 8), (24, 16)]:
        g = Grid(shape, (1.0,) * 2)
        _, load, _ = build_case("cantilever", g)
        f = load.assemble(torch.ones(g.shape)).reshape(-1, 2).sum(0)
        assert torch.allclose(f, torch.tensor([0.0, -1.0]), atol=1e-14)


def test_load_errors_and_registry():
    g = Grid((4, 3), (1.0, 1.0))
    with pytest.raises(BoundaryConditionError):  # load on no node: outside the domain
        DistributedForce(torch.zeros(g.n_nodes, dtype=torch.bool), (0.0, -1.0), g).assemble(torch.ones(g.shape))
    with pytest.raises(ConfigError):
        DistributedForce(torch.ones(g.n_nodes, dtype=torch.bool), (1.0,), g).assemble(torch.ones(g.shape))
    with pytest.raises(ConfigError):
        build_case("unknown", g)
    with pytest.raises(ConfigError):
        build_case("mbb", Grid((3, 3, 3), (1.0,) * 3))
    fixed, load, masks = build_case("mbb", Grid((6, 2), (1.0, 1.0)))
    assert int(fixed.sum()) == 3 + 1 and bool(masks.design.all())
    f = load.assemble(torch.ones(6, 2)).reshape(-1, 2)
    assert int((f != 0).any(1).sum()) == 1 and f[:, 1].sum() == -1.0  # one point load
