"""Exact (non-smooth) self-support rule by explicit loops: oracle for C-OVH-02/03.

A voxel in layer l >= 1 is printed iff it is designed AND at least one stencil neighbour in layer
l-1 is printed. 2D stencil: {-1,0,+1}; 3D stencil: 5-point cross. Layer 0 is the build plate.
"""

import numpy as np


def exact_support(rho_binary: np.ndarray) -> np.ndarray:
    """rho_binary has the build axis LAST. Returns the binary printed field."""
    rho = np.asarray(rho_binary) > 0.5
    out = np.zeros_like(rho)
    out[..., 0] = rho[..., 0]
    if rho.ndim == 2:
        offsets = [(-1,), (0,), (1,)]
    elif rho.ndim == 3:
        offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        raise ValueError("2D or 3D field expected")
    lateral = rho.shape[:-1]
    for ell in range(1, rho.shape[-1]):
        for idx in np.ndindex(*lateral):
            if not rho[idx + (ell,)]:
                continue
            for off in offsets:
                nb = tuple(i + o for i, o in zip(idx, off))
                if all(0 <= n < s for n, s in zip(nb, lateral)) and out[nb + (ell - 1,)]:
                    out[idx + (ell,)] = True
                    break
    return out
