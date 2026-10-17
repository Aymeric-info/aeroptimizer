"""Contract tests for `pipeline` — stubs generated from CONTRACTS.md (scripts/contracts_index.py).

Replace a stub body when its phase is reached.
"""

import pytest


@pytest.mark.skip(reason="phase 1 — C-CONV-08")
def test_no_inplace_leaf():
    r"""C-CONV-08 — **Pas de mutation en place** d'un tenseur qui participe au graphe Autograd (`x.add_`, `x[i]=…` sur un tenseur …"""
    raise NotImplementedError
