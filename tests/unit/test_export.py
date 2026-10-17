"""Export of a density field to a printable surface (lesson 13): C-EXP-01..05."""

import math

import numpy as np
import pytest
from scipy import ndimage

from aeroptimizer.io.mesh import (
    extrude,
    face_normals,
    marching_squares,
    marching_tetrahedra,
    mesh_report,
    polygon_area,
    read_stl,
    thin_fraction,
    write_stl,
)


def _ball(n, r, ndim=3):
    x = np.arange(n) + 0.5 - n / 2
    grids = np.meshgrid(*([x] * ndim), indexing="ij")
    rad = np.sqrt(sum(g**2 for g in grids))
    return np.clip(0.5 + (r - rad), 0.0, 1.0), rad


def test_marching_squares_contour():
    r"""C-EXP-01 — closed contours, vertices at the level, signed area = area of the matter."""
    disk, rad = _ball(60, 15.0, 2)
    loops = marching_squares(disk, 0.5)
    assert len(loops) == 1
    assert polygon_area(loops[0]) == pytest.approx(math.pi * 15.0**2, rel=0.01)
    ring = np.clip(0.5 + np.minimum(20.0 - rad, rad - 8.0), 0.0, 1.0)
    loops = marching_squares(ring, 0.5)
    areas = sorted(polygon_area(loop) for loop in loops)
    assert len(loops) == 2 and areas[0] < 0 < areas[1]  # a hole counts negatively
    assert sum(areas) == pytest.approx(math.pi * (20.0**2 - 8.0**2), rel=0.01)
    # every vertex lies where the interpolated field equals the level (padded index = coordinate + 1/2)
    xy = marching_squares(disk, 0.5)[0]
    vals = ndimage.map_coordinates(np.pad(disk, 1), (xy + 0.5).T, order=1)
    assert np.abs(vals - 0.5).max() < 1e-9
    # saddle: two diagonal inside corners; the centre value decides whether they join
    for corner_value, n_loops in ((0.0, 2), (1.0, 1)):
        two_by_two = np.zeros((4, 4))
        two_by_two[1, 1] = two_by_two[2, 2] = 1.0
        two_by_two[1, 2] = two_by_two[2, 1] = corner_value  # the other diagonal: low (cut) or high (joined)
        assert len(marching_squares(two_by_two, 0.5)) == n_loops


def test_marching_tetrahedra_watertight():
    r"""C-EXP-02 — closed, oriented, Euler 2 (ball) and 0 (torus), also on random fields."""
    ball, _ = _ball(24, 8.0)
    rep = mesh_report(*marching_tetrahedra(ball, 0.5))
    assert rep["watertight"] and rep["euler"] == 2 and rep["degenerate"] == 0
    # samples exactly at the level (odd n, integer radius): zero-area triangles must not break orientation
    exact, _ = _ball(13, 4.0)
    assert np.isclose(exact, 0.5).any()
    assert mesh_report(*marching_tetrahedra(exact, 0.5))["watertight"]
    n = 28
    x = np.arange(n) + 0.5 - n / 2
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    torus = np.clip(0.5 + (4.0 - np.sqrt((np.sqrt(X**2 + Y**2) - 9.0) ** 2 + Z**2)), 0.0, 1.0)
    rep = mesh_report(*marching_tetrahedra(torus, 0.5))
    assert rep["watertight"] and rep["euler"] == 0
    rng = np.random.default_rng(0)
    for _ in range(5):
        field = rng.random((7, 6, 8))
        rep = mesh_report(*marching_tetrahedra(field, 0.5))
        assert rep["watertight"] and rep["degenerate"] == 0 and rep["volume"] > 0
    # a field with a corner touching the border is still closed (the padding closes it)
    full = np.ones((4, 4, 4))
    assert mesh_report(*marching_tetrahedra(full, 0.5))["watertight"]


def test_mesh_volume_and_orientation():
    r"""C-EXP-03 — signed volume > 0 (outward normals), analytic to 2 %, converging under refinement."""
    errs = []
    for n, r in ((16, 5.0), (32, 10.0), (64, 20.0)):
        ball, _ = _ball(n, r)
        V, F = marching_tetrahedra(ball, 0.5)
        rep = mesh_report(V, F)
        # the physical ball has radius r / (n / 2) in units where the box is 2 wide: rescale to cells
        errs.append(abs(rep["volume"] / (4 / 3 * math.pi * r**3) - 1.0))
        assert rep["volume"] > 0
        # outward: normals have a positive component along (centroid - centre)
        centre = np.full(3, n / 2.0)
        ctr = V[F].mean(1) - centre
        assert (np.einsum("ij,ij->i", face_normals(V, F), ctr) > 0).mean() > 0.99
    assert errs[-1] < 0.02 and errs[0] > errs[-1]


def test_stl_roundtrip(tmp_path):
    r"""C-EXP-04 — binary size, binary and ASCII round trip, unit outward normals."""
    ball, _ = _ball(20, 6.0)
    V, F = marching_tetrahedra(ball, 0.5, h=(0.5, 0.5, 0.5))
    pb, pa = tmp_path / "b.stl", tmp_path / "a.stl"
    write_stl(pb, V, F)
    write_stl(pa, V, F, binary=False)
    assert pb.stat().st_size == 84 + 50 * len(F)
    tri_ref = np.sort(V[F].astype(np.float32).reshape(len(F), -1), axis=0)
    for path in (pb, pa):
        V2, F2, N2 = read_stl(path)
        assert len(F2) == len(F)
        assert np.allclose(np.sort(V2[F2].astype(np.float32).reshape(len(F), -1), axis=0), tri_ref, atol=1e-5)
        assert np.allclose(np.linalg.norm(N2, axis=1), 1.0, atol=1e-5)
        assert mesh_report(V2, F2)["watertight"]  # welding on identical float32 coordinates restores the topology
        assert np.einsum("ij,ij->i", N2, face_normals(V2, F2)).min() > 0.999


def test_thin_fraction():
    r"""C-EXP-05 — a bar wider than 2r loses nothing (up to its corners); a thinner one is lost."""
    bar = np.zeros((60, 40), bool)
    bar[5:55, 10:20] = True  # 10 cells wide
    assert thin_fraction(bar, 4.0) < 0.08  # a disk of radius 4 (diameter 8) fits; only the corners are rounded
    assert thin_fraction(bar, 6.0) > 0.95  # diameter 12 > 10: the bar is too thin
    fractions = [thin_fraction(bar, r) for r in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]
    assert all(a <= b + 1e-12 for a, b in zip(fractions, fractions[1:]))
    assert extrude(np.ones((3, 4)), 5).shape == (3, 5, 4)
