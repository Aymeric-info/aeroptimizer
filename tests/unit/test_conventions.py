"""Contract tests for `conventions` — stubs generated from CONTRACTS.md (scripts/contracts_index.py).

Replace a stub body when its phase is reached.
"""

import pytest


@pytest.mark.skip(reason="phase 1 — C-CONV-01")
def test_build_axis_is_last():
    r"""C-CONV-01 — **Axes** : en 2D les axes sont `(x, z)`, en 3D `(x, y, z)`. L'axe `z` est le **dernier axe** (`axis = d-1`) et…"""
    raise NotImplementedError


@pytest.mark.skip(reason="phase 1 — C-CONV-02")
def test_si_roundtrip():
    r"""C-CONV-02 — **Unités** : SI strict aux interfaces publiques (m, N, Pa, kg, s). Pas de mm/MPa implicites. Toute grandeur sa…"""
    raise NotImplementedError


@pytest.mark.skip(reason="phase 1 — C-CONV-03")
def test_dtype_device_consistency():
    r"""C-CONV-03 — **Types** : calcul en `torch.float64` par défaut. `float32` est autorisé uniquement via `config.dtype` explici…"""
    raise NotImplementedError


@pytest.mark.skip(reason="phase 1 — C-CONV-05")
def test_scaling_inverse():
    r"""C-CONV-05 — **Mise à l'échelle** : le solveur travaille en grandeurs réduites ($E_p=1$, $h$ physique conservé). La convers…"""
    raise NotImplementedError


@pytest.mark.skip(reason="phase 2 — C-CONV-09")
def test_determinism_cpu():
    r"""C-CONV-09 — **Déterminisme** : à graine et device fixés, deux exécutions donnent la même suite de designs à `1e-12` près (…"""
    raise NotImplementedError


@pytest.mark.skip(reason="phase 2 — C-DIFF-03")
def test_no_handwritten_grads_outside_solve():
    r"""C-DIFF-03 — Aucune différentiation « à la main » d'un terme de couplage : tout gradient de production provient d'Autograd,…"""
    raise NotImplementedError
