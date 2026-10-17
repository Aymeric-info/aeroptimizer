"""Generic structured-grid FE operator: serves mechanics (ncomp=d) AND potential flow (ncomp=1).

See ARCHITECTURE §5.2. K~ = P K P + (I - P) (identity on fixed dofs, FORMALISM §3.4).
`w` is the per-element factor supplied by the caller (SIMP stiffness s, or fluid conductivity kappa).
"""

from __future__ import annotations

import scipy.sparse as sp
import torch

from aeroptimizer.errors import ConfigError
from aeroptimizer.grid import Connectivity, Grid


class StructuredFEOperator:
    grid: Grid
    ncomp: int
    edof: torch.Tensor  # (n_elems, m), m = ncomp * 2**d
    Ke0: torch.Tensor  # (m, m), identical for every element (uniform grid)
    free_mask: torch.Tensor  # (N,) bool

    def __init__(
        self,
        grid: Grid,
        conn: Connectivity,
        Ke0: torch.Tensor,
        ncomp: int,
        fixed_mask: torch.Tensor,
    ) -> None:
        m = ncomp * 2**grid.d
        if conn.ncomp != ncomp or conn.edof.shape != (grid.n_elems, m):
            raise ConfigError(f"connectivity {tuple(conn.edof.shape)} does not match ncomp={ncomp}")
        if Ke0.shape != (m, m):
            raise ConfigError(f"Ke0 must be ({m}, {m}), got {tuple(Ke0.shape)}")
        if fixed_mask.shape != (grid.n_dofs(ncomp),) or fixed_mask.dtype != torch.bool:
            raise ConfigError(f"fixed_mask must be a bool tensor of shape ({grid.n_dofs(ncomp)},)")
        self.grid, self.ncomp = grid, ncomp
        self.edof = conn.edof.to(fixed_mask.device)
        self.Ke0 = Ke0.to(fixed_mask.device)
        self.free_mask = ~fixed_mask
        self._pattern: tuple[torch.Tensor, ...] | None = None  # built on the first assemble_csr

    @property
    def n_dofs(self) -> int:
        return self.grid.n_dofs(self.ncomp)

    def matvec(self, w: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """y = K~(w) u via gather -> (Ne,m)@(m,m) -> scatter-add; no (Ne,m,m). C-OP-01/02/07."""
        free = self.free_mask.to(u.dtype)
        ue = (u * free)[self.edof]  # gather (Ne, m)
        fe = (ue @ self.Ke0.T) * w[:, None]
        y = torch.zeros_like(u).index_add_(0, self.edof.reshape(-1), fe.reshape(-1))
        return y * free + u * (1.0 - free)  # identity on fixed dofs

    def diag(self, w: torch.Tensor) -> torch.Tensor:
        """Exact diagonal of K~ (1 on fixed dofs). C-OP-04."""
        fe = w[:, None] * torch.diagonal(self.Ke0)[None, :]
        d = torch.zeros(self.n_dofs, dtype=fe.dtype, device=fe.device)
        d.index_add_(0, self.edof.reshape(-1), fe.reshape(-1))
        return torch.where(self.free_mask, d, torch.ones_like(d))

    def _build_pattern(self) -> tuple[torch.Tensor, ...]:
        """CSR pattern and slot table, computed once (depends on the mesh only, never on w).

        Returns (indptr, indices, pos, keep, fixed_slots): `pos` maps each of the Ne*m*m element
        entries to its CSR slot, `keep` flags the entries whose row and column are both free, and
        `fixed_slots` are the diagonal slots of the fixed dofs.
        """
        N, m = self.n_dofs, self.edof.shape[1]
        rows = self.edof[:, :, None].expand(-1, -1, m).reshape(-1)
        cols = self.edof[:, None, :].expand(-1, m, -1).reshape(-1)
        diag = torch.arange(N, device=rows.device)
        keys = torch.cat([rows * N + cols, diag * N + diag])  # every diagonal entry is stored
        uniq, inverse = torch.unique(keys, return_inverse=True)  # sorted: row-major order
        pos, diag_slot = inverse[: rows.numel()], inverse[rows.numel() :]
        indices = uniq % N
        counts = torch.bincount(uniq // N, minlength=N)
        zero = torch.zeros(1, dtype=counts.dtype, device=counts.device)
        indptr = torch.cat([zero, counts.cumsum(0)])
        keep = self.free_mask[rows] & self.free_mask[cols]
        return indptr, indices, pos, keep, diag_slot[~self.free_mask]

    def assemble_csr(self, w: torch.Tensor) -> sp.csr_matrix:
        """Symmetric CSR of K~ (scipy, CPU). C-OP-03.

        Sparsity pattern and slot table are precomputed; only `data` is rebuilt. The (Ne, m, m)
        tensor exists only here, for the direct solvers; not differentiable (the adjoint uses
        `dK_contract`, not the matrix).
        """
        if self._pattern is None:
            self._pattern = self._build_pattern()
        indptr, indices, pos, keep, fixed_slots = self._pattern
        vals = (w[:, None, None] * self.Ke0[None]).reshape(-1) * keep
        data = torch.zeros(indices.numel(), dtype=vals.dtype, device=vals.device)
        data.index_add_(0, pos, vals)
        data[fixed_slots] = 1.0
        return sp.csr_matrix(
            (data.cpu().numpy(), indices.cpu().numpy(), indptr.cpu().numpy()),
            shape=(self.n_dofs,) * 2,
        )

    def dK_contract(self, lam: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """(n_elems,) entries lam_e^T Ke0 u_e (vectors masked at fixed dofs). C-OP-05.

        The adjoint's only contraction.
        """
        free = self.free_mask.to(u.dtype)
        le, ue = (lam * free)[self.edof], (u * free)[self.edof]
        return ((le @ self.Ke0) * ue).sum(-1)

