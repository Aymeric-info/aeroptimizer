"""Shared fixtures. Tests run in float64 on CPU with deterministic algorithms (C-CONV-03, C-CONV-09)."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _float64_cpu_deterministic():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(previous)


@pytest.fixture
def small_grid_2d():
    from aeroptimizer.grid import Grid

    return Grid(shape=(6, 4), h=(1.0, 1.0))


@pytest.fixture
def small_grid_3d():
    from aeroptimizer.grid import Grid

    return Grid(shape=(4, 3, 3), h=(1.0, 1.0, 1.0))
