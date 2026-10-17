"""Independent dense density filter (explicit double loop over elements), oracle for C-FILT-01..04.

W[e, f] = max(0, 1 - dist(e, f) / R) with dist between element centres (index differences times h);
filter(x) = W x / W 1. No convolution: nothing shared with `DensityFilter`.
"""

import itertools

import numpy as np


def filter_matrix(shape: tuple[int, ...], h: tuple[float, ...], R: float) -> np.ndarray:
    idx = list(itertools.product(*[range(n) for n in shape]))  # C order, like the package
    W = np.zeros((len(idx), len(idx)))
    for a, ia in enumerate(idx):
        for b, ib in enumerate(idx):
            dist = np.sqrt(sum(((i - j) * hh) ** 2 for i, j, hh in zip(ia, ib, h)))
            W[a, b] = max(0.0, 1.0 - dist / R)
    return W


def apply_filter(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (W @ x.reshape(-1)) / W.sum(axis=1)
