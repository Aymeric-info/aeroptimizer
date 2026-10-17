"""Contract tests for `grid` — stubs generated from CONTRACTS.md (scripts/contracts_index.py).

Replace a stub body when its phase is reached.
"""

import pytest


def test_node_and_dof_ordering():
    r"""C-CONV-04 — **Ordonnancement C (row-major), dernier axe le plus rapide** pour éléments, nœuds ; ddl globaux $= d\cdot n+c$…"""
    import torch

    from aeroptimizer.grid import Connectivity, Grid

    # 3D grid 2 x 3 x 4 elements -> nodes 3 x 4 x 5. Element (i,j,k) has number (i*3 + j)*4 + k.
    g = Grid((2, 3, 4), (1.0, 1.0, 1.0))
    edof = Connectivity(g, 3).edof
    n_of = lambda i, j, k: (i * 4 + j) * 5 + k  # noqa: E731  C-order node number
    e = (1 * 3 + 2) * 4 + 3  # element (1, 2, 3)
    assert edof.shape == (24, 24) and edof.dtype == torch.int64
    nodes = [n_of(1 + b0, 2 + b1, 3 + b2) for b0 in (0, 1) for b1 in (0, 1) for b2 in (0, 1)]  # b_0 slowest
    assert edof[e].tolist() == [3 * n + c for n in nodes for c in range(3)]
    # scalar field (potential): dof = node
    assert Connectivity(g, 1).edof[e].tolist() == nodes


def test_grid_validation():
    r"""C-GRID-01 — `Grid(shape, h)` : `shape` entiers ≥ 1, `len(shape)∈{2,3}`, `h` réels > 0 de même longueur ; sinon `ConfigErro…"""
    from aeroptimizer.errors import ConfigError
    from aeroptimizer.grid import Grid

    Grid((6, 4), (1.0, 1.0))
    for shape, h in [((6,), (1.0,)), ((6, 4), (1.0,)), ((0, 4), (1.0, 1.0)), ((6, 4), (1.0, -1.0))]:
        with pytest.raises(ConfigError):
            Grid(shape, h)


def test_counts():
    r"""C-GRID-02 — `n_elems == ∏shape`, `n_nodes == ∏(shape+1)`, `n_dofs(ncomp) == ncomp·n_nodes`."""
    from aeroptimizer.grid import Grid

    g = Grid((6, 4, 3), (1.0, 2.0, 3.0))
    assert g.n_elems == 72 and g.n_nodes == 7 * 5 * 4
    assert g.n_dofs(3) == 3 * g.n_nodes and g.build_axis == 2
    assert g.element_volume == 6.0


@pytest.mark.parametrize("shape", [(5, 4), (3, 2, 4)])
def test_edof_vectorised_matches_loop(shape):
    r"""C-GRID-03 — `Connectivity.edof` est **indépendant de la densité**, calculé par vectorisation pure (pas de boucle Python su…"""
    import itertools

    import torch

    from aeroptimizer.grid import Connectivity, Grid

    g = Grid(shape, (1.0,) * len(shape))
    d, ncomp = g.d, g.d
    edof = Connectivity(g, ncomp).edof
    node_shape = [n + 1 for n in shape]
    rows = []
    for idx in itertools.product(*[range(n) for n in shape]):  # naive loops: the oracle
        row = []
        for bits in itertools.product((0, 1), repeat=d):  # lexicographic: b_0 slowest
            n = 0
            for k in range(d):
                n = n * node_shape[k] + idx[k] + bits[k]
            row += [ncomp * n + c for c in range(ncomp)]
        rows.append(row)
    assert torch.equal(edof, torch.tensor(rows))
    assert all(len(set(r.tolist())) == len(r) for r in edof)  # no duplicate inside a row
    assert int(edof.min()) == 0 and int(edof.max()) == g.n_dofs(ncomp) - 1


@pytest.mark.parametrize("shape", [(5, 4), (3, 2, 4)])
def test_element_volume(shape):
    r"""C-GRID-04 — Les coordonnées nodales reconstruites depuis `edof` et `Ke0` donnent un élément de volume $\prod h_a$ (somme d…"""
    import math

    import torch

    from aeroptimizer.fem.elements import gauss_points, shape_grads
    from aeroptimizer.grid import Connectivity, Grid

    d = len(shape)
    h = (0.5, 2.0, 1.25)[:d]
    g = Grid(shape, h)
    edof = Connectivity(g, 1).edof  # potential numbering: dof = node
    # node coordinates rebuilt from the node numbers (C-order unravel), not from the grid object
    node_shape = torch.tensor([n + 1 for n in shape])
    nodes = torch.arange(g.n_nodes)
    idx = torch.stack([(nodes // math.prod(node_shape[k + 1 :].tolist())) % node_shape[k] for k in range(d)], 1)
    coords = idx * torch.tensor(h)  # (Nn, d)
    for e in (0, g.n_elems - 1):
        xe = coords[edof[e]]  # (2**d, d)
        # Jacobian at each Gauss point from the nodes and the reference gradients (d x / d xi)
        _, w = gauss_points(d)
        dN_dx = shape_grads(d, h)  # grad_x N: with x = sum N_a x_a, sum_a x_a (grad_x N_a)^T = I
        eye = torch.einsum("ai,gaj->gij", xe, dN_dx)
        assert torch.allclose(eye, torch.eye(d).expand(len(w), d, d), atol=1e-13)
        vol = float(w.sum()) * math.prod(hk / 2 for hk in h)
        assert vol == pytest.approx(g.element_volume, rel=1e-14)
        assert float((xe.max(0).values - xe.min(0).values).prod()) == pytest.approx(g.element_volume)


def test_masks_partition():
    r"""C-GRID-05 — Masques `D`, `S`, `V` : partition de la grille (disjoints, union = grille) ; sinon `ConfigError`."""
    import torch

    from aeroptimizer.errors import ConfigError
    from aeroptimizer.grid import Masks

    D = torch.tensor([[True, False], [False, False]])
    S = torch.tensor([[False, True], [False, False]])
    V = torch.tensor([[False, False], [True, True]])
    Masks(D, S, V)
    with pytest.raises(ConfigError):
        Masks(D, S, torch.zeros(2, 2, dtype=torch.bool))  # not a cover
    with pytest.raises(ConfigError):
        Masks(D, D, V)  # overlap
