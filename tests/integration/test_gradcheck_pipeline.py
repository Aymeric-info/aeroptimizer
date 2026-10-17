"""Contract tests for `gradcheck_pipeline` — stubs generated from CONTRACTS.md (scripts/contracts_index.py).

Replace a stub body when its phase is reached.
"""

import pytest


@pytest.mark.skip(reason="phase 6 — C-DIFF-01")
def test_full_chain():
    r"""C-DIFF-01 — **Pipeline complet** : `gradcheck` (float64, petites grilles `6×4` 2D, `4×3×3` 3D) de `J(x)` pour **chaque ter…"""
    raise NotImplementedError


@pytest.mark.skip(reason="phase 6 — C-DIFF-02")
def test_no_nan_trajectory():
    r"""C-DIFF-02 — Aucun `NaN/Inf` dans les gradients sur une trajectoire d'optimisation de 50 itérations (2D) ; test de non-régr…"""
    raise NotImplementedError


@pytest.mark.skip(reason="phase 2 — C-DIFF-04")
def test_where_nan_trap():
    r"""C-DIFF-04 — Les opérateurs non différentiables en un point (`min/max`, ReLU) documentent leur sous-gradient choisi ; aucun…"""
    raise NotImplementedError
