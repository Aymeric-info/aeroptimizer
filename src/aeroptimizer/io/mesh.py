"""From a density field to a printable surface: threshold, contour, triangle mesh, STL (lesson 13).

NumPy only, no mandatory dependency. Fields are element-centred: sample point (i, j[, k]) sits at
((i + 1/2) h_0, (j + 1/2) h_1, ...). Every field is padded with one layer of void so surfaces close.

* `marching_squares`: 2D contour (16 cases, saddles decided by the centre value).
* `marching_tetrahedra`: 3D surface. Each cube is cut into 6 tetrahedra along the same diagonal, so
  neighbouring cubes agree on every face and the surface is closed by construction; the 256-case
  table of marching cubes (and its ambiguous cases) is not needed.
* `mesh_report`, `write_stl`, `read_stl`, `thin_fraction`.
"""

from __future__ import annotations

import itertools
import struct
from pathlib import Path

import numpy as np
from scipy import ndimage

from aeroptimizer.errors import ConfigError

# ----------------------------------------------------------------------------------------- 2D

# Corners of a square: 0=(0,0) 1=(1,0) 2=(1,1) 3=(0,1). Edges: 0=bottom (0-1) 1=right (1-2) 2=top (2-3) 3=left (3-0).
_EDGE_CORNERS = ((0, 1), (1, 2), (2, 3), (3, 0))
# For each case (bit k set when corner k is inside), the list of segments as pairs of edges.
_SQUARE_CASES: dict[int, list[tuple[int, int]]] = {
    0: [], 15: [],
    1: [(3, 0)], 14: [(3, 0)],
    2: [(0, 1)], 13: [(0, 1)],
    3: [(3, 1)], 12: [(3, 1)],
    4: [(1, 2)], 11: [(1, 2)],
    6: [(0, 2)], 9: [(0, 2)],
    7: [(3, 2)], 8: [(3, 2)],
}  # saddles 5 and 10 are handled in the code


def _pad(field: np.ndarray, level: float) -> np.ndarray:
    # outside the box there is no matter: density 0 (a full cell then ends exactly at its border)
    out = np.pad(np.asarray(field, dtype=float), 1, constant_values=0.0 if level > 0.0 else level - 1.0)
    # A sample exactly at the level would give zero-area triangles whose orientation is undefined:
    # treat it as an hair below the level (1e-9 moves the surface by 1e-9 cell).
    return np.where(out == level, level - 1e-9, out)


def marching_squares(field: np.ndarray, level: float = 0.5, h: tuple[float, float] = (1.0, 1.0)):
    """Contour lines of `field` at `level`: list of closed polylines, each an (m, 2) array (x, z).

    Inside = value > level. Vertices sit on cell edges by linear interpolation, so the field equals
    `level` there exactly. Each loop is oriented with the matter on its LEFT (outer boundaries
    counter-clockwise, holes clockwise), so the shoelace areas add up to the area of the matter.
    """
    f = _pad(field, level)
    nx, nz = f.shape
    inside = f > level
    corner = [inside[:-1, :-1], inside[1:, :-1], inside[1:, 1:], inside[:-1, 1:]]  # corners 0..3
    vals = [f[:-1, :-1], f[1:, :-1], f[1:, 1:], f[:-1, 1:]]
    case = sum(c.astype(int) << k for k, c in enumerate(corner))
    centre = 0.25 * (vals[0] + vals[1] + vals[2] + vals[3])
    pos = [(0, 0), (1, 0), (1, 1), (0, 1)]

    def vertex(i, j, e):
        a, b = _EDGE_CORNERS[e]
        va, vb = vals[a][i, j], vals[b][i, j]
        t = (level - va) / (vb - va)
        (ax, az), (bx, bz) = pos[a], pos[b]
        # the key identifies the edge globally so that two cells share the same vertex
        p = (i + ax + t * (bx - ax), j + az + t * (bz - az))
        key = (min((i + ax, j + az), (i + bx, j + bz)), max((i + ax, j + az), (i + bx, j + bz)))
        return key, p

    segs = []
    for i, j in zip(*np.nonzero((case > 0) & (case < 15))):
        c = int(case[i, j])
        if c in (5, 10):
            # saddle: corners 0,2 (case 5) or 1,3 (case 10) inside. If the centre is inside, the two
            # inside corners are connected and the OUTSIDE corners are cut off; otherwise the reverse.
            joined = bool(centre[i, j] > level)
            cut_01_23 = [(0, 1), (2, 3)]  # isolates corners 1 and 3
            cut_30_12 = [(3, 0), (1, 2)]  # isolates corners 0 and 2
            if c == 5:
                pairs = cut_01_23 if joined else cut_30_12
            else:
                pairs = cut_30_12 if joined else cut_01_23
        else:
            pairs = _SQUARE_CASES[c]
        for e0, e1 in pairs:
            segs.append((vertex(i, j, e0), vertex(i, j, e1)))

    # link the segments into loops through the shared edge keys
    adj: dict = {}
    pts: dict = {}
    for (k0, p0), (k1, p1) in segs:
        pts[k0], pts[k1] = p0, p1
        adj.setdefault(k0, []).append(k1)
        adj.setdefault(k1, []).append(k0)
    loops, seen = [], set()
    for start in adj:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxts = [n for n in adj[cur] if n != prev] or adj[cur]
            nxt = nxts[0]
            if nxt == start:
                break
            if nxt in seen:  # defensive: malformed topology
                break
            loop.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        loops.append(loop)

    out = []
    hx, hz = h
    for loop in loops:
        xy = np.array([pts[k] for k in loop], dtype=float)
        # orient: sample the field just left of the first segment
        d = xy[1] - xy[0]
        left = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-300)
        probe = 0.5 * (xy[0] + xy[1]) + 0.25 * left
        value = ndimage.map_coordinates(f, [[probe[0]], [probe[1]]], order=1, mode="nearest")[0]
        if not value > level:
            xy = xy[::-1]
        # padded index p corresponds to element p-1: centre of element e at (e + 1/2) h
        out.append(np.column_stack([(xy[:, 0] - 1 + 0.5) * hx, (xy[:, 1] - 1 + 0.5) * hz]))
    return out


def polygon_area(loop: np.ndarray) -> float:
    """Signed shoelace area of a closed polyline (counter-clockwise positive)."""
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ----------------------------------------------------------------------------------------- 3D

def _kuhn_tetrahedra() -> np.ndarray:
    """6 tetrahedra of the unit cube, all sharing the diagonal (0,0,0)-(1,1,1): (6, 4, 3) offsets."""
    tets = []
    for perm in itertools.permutations(range(3)):
        p = np.zeros(3, dtype=int)
        path = [p.copy()]
        for axis in perm:
            p[axis] = 1
            path.append(p.copy())
        tets.append(path)
    return np.array(tets)


def marching_tetrahedra(
    field: np.ndarray, level: float = 0.5, h: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> tuple[np.ndarray, np.ndarray]:
    """Closed triangle mesh of the surface `field == level` of a 3D field. Returns (vertices, faces).

    Inside = value > level. Vertices are shared between triangles (welded through the edge of the
    sampling grid they lie on) and faces are oriented with the outward normal pointing away from the
    matter (right-hand rule).
    """
    f = _pad(field, level)
    if f.ndim != 3:
        raise ConfigError("marching_tetrahedra needs a 3D field")
    idx = np.arange(f.size).reshape(f.shape)
    cells = np.stack(np.meshgrid(*[np.arange(s - 1) for s in f.shape], indexing="ij"), -1).reshape(-1, 3)
    corners = cells[:, None, None, :] + _kuhn_tetrahedra()[None]  # (cells, 6, 4, 3)
    gid = idx[corners[..., 0], corners[..., 1], corners[..., 2]].reshape(-1, 4)  # (T, 4)
    flat = f.reshape(-1)
    coords = np.stack(np.unravel_index(np.arange(f.size), f.shape), -1).astype(float)

    ins = flat[gid] > level
    n_in = ins.sum(1)
    keep = (n_in > 0) & (n_in < 4)
    gid, ins, n_in = gid[keep], ins[keep], n_in[keep]

    ea, eb, direction = [], [], []  # per triangle: 3 edges (a, b) and the inside -> outside direction

    def emit(tri_a, tri_b, g, m):
        xyz = coords[g]
        w_in = m[..., None].astype(float)
        c_in = (xyz * w_in).sum(1) / w_in.sum(1)
        c_out = (xyz * (1 - w_in)).sum(1) / (1 - w_in).sum(1)
        ea.append(tri_a); eb.append(tri_b); direction.append(c_out - c_in)

    for n_minor in (1, 3):  # one vertex alone on its side: one triangle
        sel = n_in == n_minor
        if not sel.any():
            continue
        g, m = gid[sel], ins[sel]
        alone = m if n_minor == 1 else ~m
        k = np.argmax(alone, axis=1)
        rows = np.arange(len(g))
        others = np.stack([[c for c in range(4) if c != kk] for kk in k])
        emit(np.repeat(g[rows, k][:, None], 3, 1), np.take_along_axis(g, others, 1), g, m)
    sel = n_in == 2  # two against two: a quadrilateral, cut in two triangles
    if sel.any():
        g, m = gid[sel], ins[sel]
        order = np.argsort(~m, axis=1, kind="stable")  # the two inside vertices first
        A, B, C, D = (np.take_along_axis(g, order[:, [k]], 1)[:, 0] for k in range(4))
        emit(np.stack([A, A, B], 1), np.stack([C, D, D], 1), g, m)  # e_AC, e_AD, e_BD
        emit(np.stack([A, B, B], 1), np.stack([C, D, C], 1), g, m)  # e_AC, e_BD, e_BC

    if not ea:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    ea, eb, direction = np.concatenate(ea), np.concatenate(eb), np.concatenate(direction)
    lo, hi = np.minimum(ea, eb), np.maximum(ea, eb)
    key = lo.astype(np.int64) * f.size + hi
    uniq, inverse = np.unique(key.reshape(-1), return_inverse=True)
    u_lo, u_hi = uniq // f.size, uniq % f.size
    t = (level - flat[u_lo]) / (flat[u_hi] - flat[u_lo])
    P = coords[u_lo] + t[:, None] * (coords[u_hi] - coords[u_lo])  # padded-index coordinates
    F = inverse.reshape(-1, 3)
    normal = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    flip = np.einsum("ij,ij->i", normal, direction) < 0
    F[flip] = F[flip][:, ::-1]
    V = (P - 1.0 + 0.5) * np.array(h)  # padded index p is element p - 1, centred at +1/2
    proper = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    F = F[proper]
    used = np.unique(F)
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[F]


# ------------------------------------------------------------------------------- mesh checks


def mesh_report(V: np.ndarray, F: np.ndarray) -> dict:
    """Topology and geometry of a triangle mesh.

    - `boundary_edges`: edges used by one face only (holes); `nonmanifold_edges`: used by more than two;
    - `oriented`: every edge is traversed once in each direction (consistent orientation);
    - `watertight`: no boundary edge, no non-manifold edge, and oriented;
    - `euler`: V - E + F (2 for a ball, 0 for a torus, 2 per closed component of genus 0);
    - `volume`: signed volume by the divergence theorem (positive when the normals point outward);
    - `degenerate`: faces of zero area.
    """
    F = np.asarray(F)
    directed = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    und = np.sort(directed, axis=1)
    keys, counts = np.unique(und[:, 0] * (len(V) + 1) + und[:, 1], return_counts=True)
    dkeys, dcounts = np.unique(directed[:, 0] * (len(V) + 1) + directed[:, 1], return_counts=True)
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    boundary, nonman = int((counts == 1).sum()), int((counts > 2).sum())
    oriented = bool((dcounts == 1).all())
    return {
        "n_vertices": len(V), "n_faces": len(F), "n_edges": len(keys),
        "boundary_edges": boundary, "nonmanifold_edges": nonman, "oriented": oriented,
        "watertight": boundary == 0 and nonman == 0 and oriented,
        "euler": len(V) - len(keys) + len(F),
        "volume": float(np.einsum("ij,ij->", p0, np.cross(p1, p2)) / 6.0),
        "area": float(0.5 * np.linalg.norm(cross, axis=1).sum()),
        "degenerate": int((np.linalg.norm(cross, axis=1) == 0).sum()),
    }


def face_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(norm == 0, 1.0, norm)


# ----------------------------------------------------------------------------------------- STL


def write_stl(path: str | Path, V: np.ndarray, F: np.ndarray, *, binary: bool = True, name: str = "part") -> None:
    """STL, binary (80-byte header, uint32 count, 50 bytes per triangle) or ASCII.

    Facet normals are recomputed from the vertex order (outward for a mesh from this module);
    degenerate triangles get a zero normal. Coordinates are written as float32, as the format asks.
    """
    V, F = np.asarray(V, dtype=np.float32), np.asarray(F)
    tri = V[F]  # (m, 3, 3)
    normals = face_normals(V.astype(np.float64), F).astype(np.float32)
    path = Path(path)
    if binary:
        rec = np.zeros(len(F), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        rec["n"], rec["v"] = normals, tri
        with open(path, "wb") as fh:
            fh.write(name.encode("ascii", "replace")[:80].ljust(80, b" "))
            fh.write(struct.pack("<I", len(F)))
            fh.write(rec.tobytes())
    else:
        with open(path, "w", encoding="ascii") as fh:
            fh.write(f"solid {name}\n")
            for n, t in zip(normals, tri):
                fh.write(f"  facet normal {n[0]:.9e} {n[1]:.9e} {n[2]:.9e}\n    outer loop\n")
                for p in t:
                    fh.write(f"      vertex {p[0]:.9e} {p[1]:.9e} {p[2]:.9e}\n")
                fh.write("    endloop\n  endfacet\n")
            fh.write(f"endsolid {name}\n")


def read_stl(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a binary or ASCII STL. Returns (vertices welded on identical float32 coordinates, faces, normals)."""
    raw = Path(path).read_bytes()
    count = struct.unpack("<I", raw[80:84])[0] if len(raw) >= 84 else -1
    if len(raw) == 84 + 50 * count:  # binary: the size gives it away (an ASCII file may start with "solid" too)
        rec = np.frombuffer(raw[84:], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        tri, normals = rec["v"].astype(np.float32), rec["n"].astype(np.float32)
    else:
        text = raw.decode("ascii").split()
        normals = np.array(
            [[float(text[i + k]) for k in (2, 3, 4)] for i, w in enumerate(text) if w == "facet"], dtype=np.float32
        )
        tri = np.array(
            [[float(text[i + k]) for k in (1, 2, 3)] for i, w in enumerate(text) if w == "vertex"], dtype=np.float32
        ).reshape(-1, 3, 3)
    pts = tri.reshape(-1, 3)
    V, inverse = np.unique(pts, axis=0, return_inverse=True)
    return V.astype(np.float64), inverse.reshape(-1, 3), normals


# --------------------------------------------------------------------------- printability


def disk(radius: float, d: int = 2) -> np.ndarray:
    """Structuring element: the cells whose centre lies within `radius` (in cells) of the origin."""
    r = int(np.ceil(radius))
    ax = np.arange(-r, r + 1)
    grids = np.meshgrid(*([ax] * d), indexing="ij")
    return sum(g.astype(float) ** 2 for g in grids) <= radius**2 + 1e-12


def thin_fraction(solid: np.ndarray, radius: float) -> float:
    """Fraction of the material that no disk (ball) of `radius` cells fits inside: too thin to print.

    Morphological opening of the binary field by a disk, then the material lost over the material.
    Convex corners are always rounded by the opening: a small fraction is lost on a perfectly fine part.
    """
    solid = np.asarray(solid, dtype=bool)
    opened = ndimage.binary_opening(np.pad(solid, int(np.ceil(radius)) + 1), structure=disk(radius, solid.ndim))
    opened = opened[tuple(slice(int(np.ceil(radius)) + 1, -int(np.ceil(radius)) - 1) for _ in solid.shape)]
    total = solid.sum()
    return float((solid & ~opened).sum() / total) if total else 0.0


def extrude(field2d: np.ndarray, layers: int) -> np.ndarray:
    """2D field (nx, nz) -> 3D field (nx, layers, nz): the section repeated along y (the part's thickness)."""
    return np.repeat(np.asarray(field2d)[:, None, :], layers, axis=1)
