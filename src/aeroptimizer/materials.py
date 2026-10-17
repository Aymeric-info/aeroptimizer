"""Constitutive laws (FORMALISM §3.1, CONTRACTS §3.2).

Material frame == grid frame (axes 1,2,3 == x,y,z; z = build axis). Voigt order (C-CONV-06):
3D (xx,yy,zz,yz,xz,xy), 2D (xx,zz,xz); engineering shears.

Studies of a material turned relative to the grid (lesson 11) use `rotation_xz`,
`rotate_stiffness` and `Orthotropic.rotated_stiffness`; the strength criterion is `hill_index`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from aeroptimizer.errors import MaterialError

Plane = Literal["stress", "strain"]


_SHEAR = ((1, 2), (0, 2), (0, 1))  # (yz, xz, xy) as index pairs, C-CONV-06


def _reduce(S: torch.Tensor, d: int, plane: Plane) -> torch.Tensor:
    """Stiffness in the Voigt basis of dimension d from the 3D compliance S (6x6)."""
    if d == 3:
        return torch.linalg.inv(S)
    if d != 2:
        raise ValueError(f"d must be 2 or 3, got {d}")
    keep = [0, 2, 4]  # xx, zz, xz in the 3D Voigt order
    if plane == "stress":
        return torch.linalg.inv(S[keep][:, keep])
    if plane == "strain":
        return torch.linalg.inv(S)[keep][:, keep]
    raise ValueError(f"plane must be 'stress' or 'strain', got {plane!r}")


def rotation_xz(angle: float) -> torch.Tensor:
    """Active rotation (3x3) that turns x towards z by `angle` (rad), y fixed (C-MAT-07).

    Seen in the (x right, z up) plane it is counter-clockwise: R e_x = (cos, 0, sin).
    """
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=torch.float64)


def stress_transform(R: torch.Tensor) -> torch.Tensor:
    """T (6x6) with voigt(R sigma R^T) = T voigt(sigma), Voigt order (xx,yy,zz,yz,xz,xy).

    Built column by column on the Voigt basis, so no closed formula to get wrong. For the strain
    in engineering shears, voigt(R eps R^T) = T^{-T} voigt(eps).
    """
    R = R.to(torch.float64)
    T = torch.zeros(6, 6, dtype=torch.float64)
    for col in range(6):
        sig = torch.zeros(3, 3, dtype=torch.float64)
        if col < 3:
            sig[col, col] = 1.0
        else:
            i, j = _SHEAR[col - 3]
            sig[i, j] = sig[j, i] = 1.0  # stress shear component tau_ij = sigma_ij
        rot = R @ sig @ R.T
        T[:3, col] = torch.diagonal(rot)
        for row, (i, j) in enumerate(_SHEAR, start=3):
            T[row, col] = rot[i, j]
    return T


def rotate_stiffness(C: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """C' = T C T^T (6x6 Voigt, engineering shears): the stiffness of the material turned by R."""
    T = stress_transform(R)
    return T @ C.to(torch.float64) @ T.T


@dataclass(frozen=True)
class Orthotropic:
    """9 independent constants; nu_ij = -eps_j/eps_i under uniaxial sigma_i; nu_ij/E_i = nu_ji/E_j."""

    E1: float
    E2: float
    E3: float
    G23: float
    G13: float
    G12: float
    nu12: float
    nu13: float
    nu23: float

    def __post_init__(self) -> None:
        # C-MAT-01: S must be positive definite. Check the named conditions first (clear message),
        # then the Cholesky factorisation, which is the actual criterion (it also catches the
        # coupled condition 1 - nu12 nu21 - nu23 nu32 - nu13 nu31 - 2 nu21 nu32 nu13 > 0).
        moduli = {"E1": self.E1, "E2": self.E2, "E3": self.E3, "G23": self.G23, "G13": self.G13,
                  "G12": self.G12}
        for name, value in moduli.items():
            if not value > 0.0:
                raise MaterialError(f"{name} must be > 0, got {value}")
        _, info = torch.linalg.cholesky_ex(self.compliance_matrix())
        if int(info) != 0:
            raise MaterialError(
                "compliance matrix S is not positive definite (Poisson ratios too large for these "
                "moduli: need 1 - nu12 nu21 - nu23 nu32 - nu13 nu31 - 2 nu21 nu32 nu13 > 0)"
            )

    def compliance_matrix(self) -> torch.Tensor:
        """S (6x6), float64, no grad (C-MAT-02, C-MAT-06). Voigt order (xx,yy,zz,yz,xz,xy)."""
        S = torch.zeros(6, 6, dtype=torch.float64)
        S[0, 0], S[1, 1], S[2, 2] = 1.0 / self.E1, 1.0 / self.E2, 1.0 / self.E3
        S[0, 1] = S[1, 0] = -self.nu12 / self.E1
        S[0, 2] = S[2, 0] = -self.nu13 / self.E1
        S[1, 2] = S[2, 1] = -self.nu23 / self.E2
        S[3, 3], S[4, 4], S[5, 5] = 1.0 / self.G23, 1.0 / self.G13, 1.0 / self.G12
        return S

    def stiffness(self, d: int, plane: Plane = "stress") -> torch.Tensor:
        """C0 in the Voigt basis of dimension d. 2D reductions per FORMALISM §3.1 (C-MAT-05).

        2D plane stress: extract {xx,zz,xz} from S, THEN invert.
        Plane strain: invert S, THEN extract.
        """
        return _reduce(self.compliance_matrix(), d, plane)

    def rotated_stiffness(
        self, R: torch.Tensor, d: int = 3, plane: Plane = "stress"
    ) -> torch.Tensor:
        """C0 of the material turned by the rotation R (3x3), expressed in the grid frame (C-MAT-07).

        The rotated material is no longer orthotropic in the grid frame (shear coupling), so it is
        returned as a stiffness matrix, not as an `Orthotropic`. The 2D reductions apply to the
        ROTATED 3D law.
        """
        C3 = rotate_stiffness(torch.linalg.inv(self.compliance_matrix()), R)
        return _reduce(torch.linalg.inv(C3), d, plane)

    @classmethod
    def isotropic(cls, E: float, nu: float) -> Orthotropic:  # C-MAT-03
        G = E / (2.0 * (1.0 + nu))
        return cls(E, E, E, G, G, G, nu, nu, nu)

    @classmethod
    def transversely_isotropic(
        cls, Ep: float, Ez: float, nup: float, nupz: float, Gz: float
    ) -> Orthotropic:  # C-MAT-04
        return cls(Ep, Ep, Ez, Gz, Gz, Ep / (2.0 * (1.0 + nup)), nup, nupz, nupz)


@dataclass(frozen=True)
class Strengths:
    """Strengths along the material axes (X,Y,Z tension/compression, S_ij shear), same unit as stress.

    Hill's criterion is symmetric in tension/compression: one value per direction.
    """

    X: float
    Y: float
    Z: float
    S23: float
    S13: float
    S12: float

    def __post_init__(self) -> None:
        for name, v in vars(self).items():
            if not v > 0.0:
                raise MaterialError(f"strength {name} must be > 0, got {v}")


def hill_index(sigma: torch.Tensor, strengths: Strengths) -> torch.Tensor:
    """Hill (1948) / Tsai-Hill quadratic index F(sigma) for stresses in the MATERIAL axes (C-MAT-08).

    F = Fh (s2-s3)^2 + Gh (s3-s1)^2 + Hh (s1-s2)^2 + t23^2/S23^2 + t13^2/S13^2 + t12^2/S12^2,
    with 2Fh = 1/Y^2+1/Z^2-1/X^2, 2Gh = 1/Z^2+1/X^2-1/Y^2, 2Hh = 1/X^2+1/Y^2-1/Z^2.
    `sigma` has shape (..., 6) in the Voigt order (xx,yy,zz,yz,xz,xy). F < 1 safe, F = 1 failure.
    F scales with the square of the load, so sqrt(F) scales with it.
    """
    X, Y, Z = strengths.X, strengths.Y, strengths.Z
    Fh = 0.5 * (1 / Y**2 + 1 / Z**2 - 1 / X**2)
    Gh = 0.5 * (1 / Z**2 + 1 / X**2 - 1 / Y**2)
    Hh = 0.5 * (1 / X**2 + 1 / Y**2 - 1 / Z**2)
    s1, s2, s3, t23, t13, t12 = sigma.unbind(-1)
    return (
        Fh * (s2 - s3) ** 2 + Gh * (s3 - s1) ** 2 + Hh * (s1 - s2) ** 2
        + (t23 / strengths.S23) ** 2 + (t13 / strengths.S13) ** 2 + (t12 / strengths.S12) ** 2
    )


def embed_plane_stress(sigma2d: torch.Tensor) -> torch.Tensor:
    """(..., 3) stresses (xx,zz,xz) -> (..., 6) with sigma_yy = tau_yz = tau_xy = 0."""
    sxx, szz, txz = sigma2d.unbind(-1)
    zero = torch.zeros_like(sxx)
    return torch.stack([sxx, zero, szz, zero, txz, zero], dim=-1)
