"""aeroptimizer — differentiable aero-structural topology optimization (FDM drone arm)."""

from aeroptimizer.errors import (
    AeroptimizerError,
    BoundaryConditionError,
    ConfigError,
    DeviceError,
    MaterialError,
    SolverConvergenceError,
)
from aeroptimizer.grid import Grid

__version__ = "0.1.0"

__all__ = [
    "AeroptimizerError",
    "BoundaryConditionError",
    "ConfigError",
    "DeviceError",
    "Grid",
    "MaterialError",
    "SolverConvergenceError",
    "__version__",
]
