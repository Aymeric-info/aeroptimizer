"""Contract tests for `io` — stubs generated from CONTRACTS.md (scripts/contracts_index.py).

Replace a stub body when its phase is reached.
"""

import pytest


@pytest.mark.skip(reason="phase 2 — C-IO-01")
def test_run_artifacts():
    r"""C-IO-01 — Chaque exécution écrit `config.resolved.yaml`, `history.csv` (`it, J, C, D, g_V, β, p, ‖Δx‖, t_it`) et `rho_ba…"""
    raise NotImplementedError


@pytest.mark.skip(reason="phase 2 — C-IO-02")
def test_vtk_export():
    r"""C-IO-02 — Export VTK (`.vti`/`.vtu`) de `rho_bar`, `u`, `phi`, `v` ; lisible par ParaView ; aucune dépendance obligatoir…"""
    raise NotImplementedError


@pytest.mark.skip(reason="phase 2 — C-IO-03")
def test_provenance_recorded():
    r"""C-IO-03 — Le dossier de run enregistre `git rev-parse HEAD` (ou `"nogit"`), versions `torch/scipy/numpy`, device, dtype."""
    raise NotImplementedError
