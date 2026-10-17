"""Q1 reference element, Gauss quadrature and element matrices (FORMALISM §1, §3.3, §4.3).

Local node ordering: bits (b_0..b_{d-1}), b_0 most significant (C-CONV-04).
"""

from __future__ import annotations

import math

import torch

from aeroptimizer.errors import ConfigError

# Index pairs (i, j) of the engineering shears, in Voigt order after the d normal strains
# (C-CONV-06): 3D (yz, xz, xy), 2D (xz). Axes are the grid ones: 2D (x, z) = (0, 1),
# 3D (x, y, z) = (0, 1, 2).
_SHEAR_PAIRS = {2: ((0, 1),), 3: ((1, 2), (0, 2), (0, 1))}


def local_node_signs(d: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    """(2**d, d) tensor of s_{a,k} = 2*b_k - 1, bits ordered with b_0 most significant."""
    if d not in (2, 3):
        raise ConfigError(f"d must be 2 or 3, got {d}")
    a = torch.arange(2**d)
    bits = torch.stack([(a >> (d - 1 - k)) & 1 for k in range(d)], dim=1)
    return (2 * bits - 1).to(dtype or torch.get_default_dtype())


def gauss_points(d: int, dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """2**d-point Gauss rule on [-1,1]^d: (points (G,d), weights (G,)). Exact to degree 3 per axis."""
    pts = local_node_signs(d, dtype) / math.sqrt(3.0)  # tensor product of {-1/sqrt3, +1/sqrt3}
    return pts, torch.ones(2**d, dtype=pts.dtype)  # all weights equal 1


def shape_values(d: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    """N_a at each Gauss point: (G, 2**d). Product of 1D factors (1 + s_{a,k} xi_k) / 2."""
    xi, _ = gauss_points(d, dtype)
    s = local_node_signs(d, xi.dtype)
    factors = (1.0 + s[None, :, :] * xi[:, None, :]) / 2.0  # (G, 2**d, d)
    return factors.prod(dim=-1)


def shape_grads(d: int, h: tuple[float, ...], dtype: torch.dtype | None = None) -> torch.Tensor:
    """grad_x N_a at each Gauss point: (G, 2**d, d). Uses J = diag(h/2)."""
    if len(h) != d:
        raise ConfigError(f"h must have length d={d}, got {h}")
    xi, _ = gauss_points(d, dtype)
    s = local_node_signs(d, xi.dtype)
    factors = (1.0 + s[None, :, :] * xi[:, None, :]) / 2.0  # (G, 2**d, d)
    grads = torch.empty_like(factors)
    for k in range(d):  # d(prod_m f_m)/dxi_k = (s_k/2) * prod_{m != k} f_m
        others = torch.cat([factors[..., :k], factors[..., k + 1 :]], dim=-1).prod(dim=-1)
        grads[..., k] = s[None, :, k] / 2.0 * others
    inv_j = 2.0 / torch.tensor(h, dtype=xi.dtype)  # J^{-1} = diag(2/h)
    return grads * inv_j


def strain_displacement(grads: torch.Tensor) -> torch.Tensor:
    """B matrices from shape gradients (G, 2**d, d): (G, n_voigt, d*2**d), dof = d*a + c.

    Rows: d normal strains eps_kk = d u_k / d x_k, then the engineering shears
    gamma_ij = du_i/dx_j + du_j/dx_i in the Voigt order of C-CONV-06.
    """
    G, n_nodes, d = grads.shape
    pairs = _SHEAR_PAIRS[d]
    B = torch.zeros(G, d + len(pairs), d * n_nodes, dtype=grads.dtype)
    cols = d * torch.arange(n_nodes)
    for k in range(d):
        B[:, k, cols + k] = grads[:, :, k]
    for r, (i, j) in enumerate(pairs, start=d):
        B[:, r, cols + i] = grads[:, :, j]
        B[:, r, cols + j] = grads[:, :, i]
    return B


def elasticity_ke0(
    d: int, h: tuple[float, ...], C0: torch.Tensor, thickness: float = 1.0
) -> torch.Tensor:
    """Mechanical element matrix, shape (d*2**d, d*2**d). Symmetric PSD, rank deficit 3 (2D) / 6 (3D).

    Ke0 = t * sum_g w_g detJ B_g^T C0 B_g   (thickness t only for 2D plane stress/strain).
    """
    n_voigt = 3 if d == 2 else 6
    if C0.shape != (n_voigt, n_voigt):
        raise ConfigError(f"C0 must be {n_voigt}x{n_voigt} for d={d}, got {tuple(C0.shape)}")
    if d == 3 and thickness != 1.0:
        raise ConfigError("thickness only applies to 2D (plane stress/strain); got it with d=3")
    _, w = gauss_points(d, C0.dtype)
    B = strain_displacement(shape_grads(d, h, C0.dtype))  # (G, n_voigt, m)
    det_j = math.prod(hk / 2.0 for hk in h)
    return thickness * det_j * torch.einsum("g,gvi,vw,gwj->ij", w, B, C0, B)


def laplace_ke0(d: int, h: tuple[float, ...], dtype: torch.dtype | None = None) -> torch.Tensor:
    """Scalar diffusion element matrix, shape (2**d, 2**d), one unknown per node (C-ELEM-06).

    Ke0 = sum_g w_g detJ (grad N)_g^T (grad N)_g: the mechanical formula with B replaced by the
    gradient itself. Symmetric PSD, K @ 1 = 0 (a constant potential has no gradient), rank 2**d - 1.
    """
    grads = shape_grads(d, h, dtype)  # (G, 2**d, d)
    _, w = gauss_points(d, grads.dtype)
    det_j = math.prod(hk / 2.0 for hk in h)
    return det_j * torch.einsum("g,gak,gbk->ab", w, grads, grads)


def center_gradients(d: int, h: tuple[float, ...], dtype: torch.dtype | None = None) -> torch.Tensor:
    """grad_x N_a at the element centre, shape (2**d, d): v_e = sum_a phi_a * center_gradients[a].

    Q1 gradients are affine in xi, so the mean over the symmetric Gauss points is the centre value.
    """
    return shape_grads(d, h, dtype).mean(dim=0)
