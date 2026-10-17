"""NumPy port of top88 (Andreassen et al. 2011): closed-form KE, density filter, OC, MBB.

Oracle for C-ELEM-05 (element matrix up to a node permutation), for the unfiltered MBB loop of
lesson 8 (`ft=0`) and for C-OPT-02 (reference compliance, to be RECOMPUTED here, not hard-coded
from memory). The code is independent of the package: its own assembly, solver and OC.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def top88_ke(nu: float = 0.3, E: float = 1.0) -> np.ndarray:
    """Plane-stress unit-square Q1 element matrix, top88 node ordering (counter-clockwise)."""
    k = np.array([
        1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8,
        -1 / 4 + nu / 12, -1 / 8 - nu / 8, nu / 6, 1 / 8 - 3 * nu / 8,
    ])
    idx = [
        [0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 7, 6, 5, 4, 3, 2], [2, 7, 0, 5, 6, 3, 4, 1],
        [3, 6, 5, 0, 7, 2, 1, 4], [4, 5, 6, 7, 0, 1, 2, 3], [5, 4, 3, 2, 1, 0, 7, 6],
        [6, 3, 4, 1, 2, 7, 0, 5], [7, 2, 1, 4, 3, 6, 5, 0],
    ]
    return E / (1 - nu**2) * k[np.array(idx)]


def _filter_matrix(nelx: int, nely: int, rmin: float) -> sp.csr_matrix:
    """Linear (conic) filter weights max(0, rmin - dist) between element centres, top88 numbering."""
    rows, cols, vals = [], [], []
    r = int(np.ceil(rmin))
    for i1 in range(nelx):
        for j1 in range(nely):
            for i2 in range(max(i1 - (r - 1), 0), min(i1 + r, nelx)):
                for j2 in range(max(j1 - (r - 1), 0), min(j1 + r, nely)):
                    rows.append(i1 * nely + j1)
                    cols.append(i2 * nely + j2)
                    vals.append(max(0.0, rmin - np.hypot(i1 - i2, j1 - j2)))
    return sp.csr_matrix((vals, (rows, cols)), shape=(nelx * nely,) * 2)


def top88(
    nelx: int,
    nely: int,
    volfrac: float,
    penal: float,
    rmin: float,
    max_iter: int = 200,
    ft: int = 1,
    lam_tol: float = 1e-3,
    change_tol: float = 1e-2,
) -> dict:
    """Returns {'c': compliance history, 'x': final density (nely, nelx), 'vol': volume history}.

    Faithful port of the 88-line MBB code (E0 = 1, Emin = 1e-9, nu = 0.3, move 0.2, damping 1/2).
    Row 0 is the TOP of the beam; nodes and elements are numbered column by column (Matlab order).
    `ft`: 0 no filter (added here), 1 sensitivity filter, 2 density filter.
    `lam_tol`: relative width of the bisection bracket (top88 uses 1e-3). `change_tol` = 0 runs
    exactly `max_iter` iterations.
    """
    E0, Emin, nu, move = 1.0, 1e-9, 0.3, 0.2
    n_el, n_dof = nelx * nely, 2 * (nelx + 1) * (nely + 1)
    KE = top88_ke(nu)
    nodenrs = np.arange((nelx + 1) * (nely + 1)).reshape(nely + 1, nelx + 1, order="F")
    r, c = np.meshgrid(np.arange(nely), np.arange(nelx), indexing="ij")
    ll, lr = nodenrs[r + 1, c], nodenrs[r + 1, c + 1]
    ur, ul = nodenrs[r, c + 1], nodenrs[r, c]
    nodes = np.stack([ll, lr, ur, ul], axis=-1).reshape(n_el, 4, order="F")  # counter-clockwise
    edof = (2 * nodes[:, :, None] + np.arange(2)).reshape(n_el, 8)
    iK = np.repeat(edof[:, :, None], 8, axis=2).ravel()
    jK = np.repeat(edof[:, None, :], 8, axis=1).ravel()

    F = np.zeros(n_dof)
    F[1] = -1.0  # top-left node, vertical, downwards
    fixed = np.union1d(np.arange(0, 2 * (nely + 1), 2), [n_dof - 1])
    free = np.setdiff1d(np.arange(n_dof), fixed)

    H = _filter_matrix(nelx, nely, rmin) if ft in (1, 2) else None
    Hs = np.asarray(H.sum(axis=1)).ravel() if H is not None else None

    x = volfrac * np.ones(n_el)
    xphys = x.copy()
    hist_c, hist_v = [], []
    change, loop = 1.0, 0
    while change > change_tol and loop < max_iter:
        loop += 1
        s = Emin + xphys**penal * (E0 - Emin)
        K = sp.csc_matrix(((KE[None] * s[:, None, None]).ravel(), (iK, jK)), shape=(n_dof, n_dof))
        U = np.zeros(n_dof)
        U[free] = spla.spsolve(K[free][:, free].tocsc(), F[free])
        ce = np.einsum("ei,ij,ej->e", U[edof], KE, U[edof])
        hist_c.append(float(np.sum(s * ce)))
        dc = -penal * (E0 - Emin) * xphys ** (penal - 1) * ce
        dv = np.ones(n_el)
        if ft == 1:
            dc = H @ (x * dc) / Hs / np.maximum(1e-3, x)
        elif ft == 2:
            dc = H @ (dc / Hs)
            dv = H @ (dv / Hs)
        l1, l2 = 0.0, 1e9
        while (l2 - l1) / (l1 + l2) > lam_tol:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(
                x + move, x * np.sqrt(-dc / dv / lmid)))))
            xphys = H @ xnew / Hs if ft == 2 else xnew
            if xphys.sum() > volfrac * n_el:
                l1 = lmid
            else:
                l2 = lmid
        change = float(np.abs(xnew - x).max())
        x = xnew
        hist_v.append(float(xphys.mean()))
    return {"c": hist_c, "x": xphys.reshape(nely, nelx, order="F"), "vol": hist_v}
