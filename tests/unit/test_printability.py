"""Voxel printability metrics (`printability.py`): overhang rate and support volume."""

import numpy as np

from aeroptimizer.printability import overhang_mask, overhang_rate, support_mask, support_volume


def _box(shape=(7, 7, 6)):
    return np.zeros(shape, bool)


def test_cube_on_the_plate_has_no_overhang():
    s = _box()
    s[2:5, 2:5, 0:4] = True
    assert overhang_rate(s) == 0.0
    assert support_volume(s) == 0.0


def test_45_deg_pyramid_upside_down_is_self_supported():
    s = _box((11, 11, 5))
    i, j = np.meshgrid(np.arange(11), np.arange(11), indexing="ij")
    for k in range(5):  # a diamond one cell wider per layer: 45 deg along x and y (the square's corners would not be)
        s[:, :, k] = abs(i - 5) + abs(j - 5) <= k
    assert overhang_rate(s) == 0.0


def test_diagonal_step_is_an_overhang_for_the_cross_only():
    s = _box((5, 5, 2))
    s[1, 1, 0] = True
    s[2, 2, 1] = True  # diagonal neighbour of the cell below: ~54.7 deg
    assert overhang_mask(s, "cross")[2, 2, 1]
    assert not overhang_mask(s, "square")[2, 2, 1]


def test_upside_down_t_needs_supports_down_to_the_plate():
    s = _box((9, 3, 5))
    s[4, :, 0:4] = True  # stem
    s[:, :, 4] = True  # bar at the top: cells 0..2 and 6..8 hang (cells 3 and 5 rest at 45 deg on the stem)
    hang = overhang_mask(s)
    assert hang[:, :, 4].sum() == 6 * 3
    assert overhang_rate(s) == hang.sum() / s[:, :, 1:].sum()
    sup = support_mask(s)
    assert sup.sum() == 6 * 3 * 4  # 4 empty layers under each hanging cell
    assert not (sup & s).any()
    assert support_volume(s, voxel_volume=8.0) == 8.0 * sup.sum()


def test_support_stops_on_the_part():
    s = _box((3, 3, 6))
    s[:, :, 0] = True  # plate-wide base
    s[1, 1, 5] = True  # floating cell 4 layers above it
    assert support_mask(s).sum() == 4


def test_ignored_cells_leave_the_rate():
    s = _box((5, 5, 4))
    s[2, 2, 0:2] = True
    s[0, 0, 3] = True  # floating
    ignore = np.zeros_like(s)
    ignore[0, 0, 3] = True
    assert overhang_rate(s) > 0
    assert overhang_rate(s, ignore=ignore) == 0.0
