"""Exception hierarchy (CONTRACTS §4.3). Never use `assert` to validate user input."""


class AeroptimizerError(Exception):
    """Base class for all package errors."""


class ConfigError(AeroptimizerError):
    """Invalid or inconsistent configuration (grid, masks, weights, unsupported passive solid...)."""


class MaterialError(AeroptimizerError):
    """Stiffness/compliance tensor is not positive definite."""


class BoundaryConditionError(AeroptimizerError):
    """Under-constrained system (rigid body modes) or load outside the domain."""


class SolverConvergenceError(AeroptimizerError):
    """Linear solver residual above tolerance."""


class DeviceError(AeroptimizerError):
    """Unsupported dtype/device combination (e.g. float64 on MPS)."""
