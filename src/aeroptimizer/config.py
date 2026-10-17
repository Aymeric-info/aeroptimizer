"""Typed, strict YAML configuration (C-CFG-01..03).

Unknown keys are errors. Calibration-dependent parameters (marked with a warning sign in
FORMALISM.md) have NO default: they must be written in the YAML (C-CFG-03).
"""

from __future__ import annotations

import types
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

from aeroptimizer.errors import ConfigError


@dataclass
class GridCfg:
    shape: list[int]
    h: list[float]


@dataclass
class MaterialCfg:
    type: str  # "isotropic" | "transversely_isotropic" | "orthotropic"
    E: float | None = None
    nu: float | None = None
    Ep: float | None = None
    Ez: float | None = None
    nup: float | None = None
    nupz: float | None = None
    Gz: float | None = None


@dataclass
class PhysicsCfg:
    plane: str = "stress"
    thickness: float = 1.0


@dataclass
class OverhangCfg:
    theta_deg: float = 45.0
    beta_s: list[float] = field(default_factory=lambda: [10.0, 80.0])
    eps: float = 1.0e-4


@dataclass
class DensityCfg:
    R: float
    p: list[float] = field(default_factory=lambda: [1.0, 3.0])
    beta: list[float] = field(default_factory=lambda: [1.0, 16.0])
    eta: float = 0.5
    overhang: OverhangCfg | None = None


@dataclass
class FlowCfg:
    k_w: float  # no default: calibration parameter
    c_b: float  # no default: calibration parameter
    model: str = "potential"
    rho_air: float = 1.225
    U_inf: float = 0.0
    n_pad: int = 6
    kappa_min: float = 1.0e-6
    q: float = 3.0


@dataclass
class LoadsCfg:
    force_N: float
    load_factor: float = 1.0


@dataclass
class ObjectiveCfg:
    volume_fraction: float
    w_c: float = 1.0
    w_a: float = 0.0


@dataclass
class OptimCfg:
    method: str = "oc"
    max_iter: int = 300
    tol: float = 1.0e-3


@dataclass
class RuntimeCfg:
    device: str = "cpu"
    dtype: str = "float64"
    seed: int = 0


@dataclass
class Config:
    case: str  # "mbb" | "cantilever" | "drone_arm" (boundary-condition builders in fem/bc.py)
    grid: GridCfg
    material: MaterialCfg
    density: DensityCfg
    objective: ObjectiveCfg
    loads: LoadsCfg
    physics: PhysicsCfg = field(default_factory=PhysicsCfg)
    flow: FlowCfg | None = None
    optim: OptimCfg = field(default_factory=OptimCfg)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)


def _coerce(tp: Any, val: Any, path: str) -> Any:
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        if val is None:
            return None
        return _coerce(args[0], val, path)
    if is_dataclass(tp):
        return _build(tp, val, path)
    if origin is list:
        if not isinstance(val, list):
            raise ConfigError(f"{path}: expected a list, got {type(val).__name__}")
        (elem,) = get_args(tp)
        return [_coerce(elem, v, f"{path}[{i}]") for i, v in enumerate(val)]
    if tp is float and isinstance(val, int) and not isinstance(val, bool):
        return float(val)
    return val


def _build(cls: type, data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{path or '<root>'}: expected a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    unknown = set(data) - set(hints)
    if unknown:
        raise ConfigError(f"{path or '<root>'}: unknown key(s) {sorted(unknown)}")  # C-CFG-01
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        here = f"{path}.{f.name}" if path else f.name
        if f.name in data:
            kwargs[f.name] = _coerce(hints[f.name], data[f.name], here)
        elif f.default is MISSING and f.default_factory is MISSING:
            raise ConfigError(f"{here}: required key is missing")
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _build(Config, raw, "")


def resolved_yaml(cfg: Config) -> str:
    """Config with all defaults made explicit (C-CFG-02)."""
    from dataclasses import asdict

    return yaml.safe_dump(asdict(cfg), sort_keys=False)
