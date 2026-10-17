"""Naive dense assembly and element-matrix integration by explicit loops: oracle for C-OP-*, C-ELEM-*."""

import numpy as np


def assemble_dense(shape: tuple[int, ...], Ke0: np.ndarray, w: np.ndarray, ncomp: int) -> np.ndarray:
    """K = sum_e A_e^T (w_e Ke0) A_e with Python loops over elements and local dofs (tiny grids only).

    Independent of `Connectivity`: node numbers come from explicit multi-indices (C order, last axis
    fastest) and local node a has the bits of `format(a, "0{d}b")`, b_0 most significant.
    """
    import itertools

    d = len(shape)
    n_nodes = [n + 1 for n in shape]
    N = ncomp * int(np.prod(n_nodes))
    K = np.zeros((N, N))
    for e, idx in enumerate(itertools.product(*[range(n) for n in shape])):
        dofs = []
        for a in range(2**d):
            bits = [int(c) for c in format(a, f"0{d}b")]
            node = 0
            for k in range(d):
                node = node * n_nodes[k] + idx[k] + bits[k]
            dofs += [ncomp * node + c for c in range(ncomp)]
        for i, gi in enumerate(dofs):
            for j, gj in enumerate(dofs):
                K[gi, gj] += w[e] * Ke0[i, j]
    return K


def ke0_by_loops(d: int, h: tuple[float, ...], C0: np.ndarray) -> np.ndarray:
    """Gauss integration with explicit loops, independent of aeroptimizer.fem.elements.

    Nothing is shared with the package: node bits come from `format`, shape-function derivatives from
    explicit products, and the B matrix rows are written out by hand (2D: xx,zz,xz; 3D: xx,yy,zz,yz,xz,xy).
    """
    import itertools

    g = 1.0 / np.sqrt(3.0)
    nn = 2**d
    bits = [[int(c) for c in format(a, f"0{d}b")] for a in range(nn)]  # b_0 = most significant
    ke = np.zeros((d * nn, d * nn))
    det_j = np.prod([hk / 2.0 for hk in h])
    for signs in itertools.product((-g, g), repeat=d):  # Gauss points, weight 1 each
        grad = np.zeros((nn, d))  # dN_a/dx_k
        for a in range(nn):
            for k in range(d):
                val = 1.0
                for m in range(d):
                    sm = 2 * bits[a][m] - 1
                    val *= (sm / 2.0) if m == k else (1 + sm * signs[m]) / 2.0
                grad[a, k] = val * 2.0 / h[k]
        B = np.zeros((3 if d == 2 else 6, d * nn))
        for a in range(nn):
            c = d * a
            if d == 2:
                gx, gz = grad[a]
                B[0, c], B[1, c + 1], B[2, c], B[2, c + 1] = gx, gz, gz, gx
            else:
                gx, gy, gz = grad[a]
                B[0, c], B[1, c + 1], B[2, c + 2] = gx, gy, gz
                B[3, c + 1], B[3, c + 2] = gz, gy  # gamma_yz
                B[4, c], B[4, c + 2] = gz, gx  # gamma_xz
                B[5, c], B[5, c + 1] = gy, gx  # gamma_xy
        ke += det_j * B.T @ C0 @ B
    return ke
