"""Contract tests for `perf` — stubs generated from CONTRACTS.md (scripts/contracts_index.py).

Replace a stub body when its phase is reached.
"""

import pytest


@pytest.mark.skip(reason="phase 2 — C-PERF-01")
def test_no_python_loops_hot_path():
    r"""C-PERF-01 — Aucune boucle Python sur éléments, nœuds ou ddl dans le chemin chaud (`matvec`, `filter`, `projection`, `drag`…"""
    raise NotImplementedError


def test_matvec_memory():
    r"""C-PERF-02 — Budget mémoire du `matvec` 3D `120×40×20` en float64 : pic ≤ 1.5 GB (hors solveur)."""
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src"
    script = textwrap.dedent(f"""
        import resource, sys
        sys.path.insert(0, {str(src)!r})
        import torch
        torch.set_default_dtype(torch.float64)
        from aeroptimizer.fem.elements import elasticity_ke0
        from aeroptimizer.fem.operators import StructuredFEOperator
        from aeroptimizer.grid import Connectivity, Grid
        from aeroptimizer.materials import Orthotropic

        g = Grid((120, 40, 20), (1.0, 1.0, 1.0))
        Ke0 = elasticity_ke0(3, g.h, Orthotropic.isotropic(1.0, 0.3).stiffness(3))
        fixed = torch.zeros(g.n_dofs(3), dtype=torch.bool)
        fixed[: 3 * 21 * 41] = True
        op = StructuredFEOperator(g, Connectivity(g, 3), Ke0, 3, fixed)
        w, v = torch.ones(g.n_elems), torch.randn(g.n_dofs(3))
        for _ in range(3):
            op.matvec(w, v)
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    peak = int(out.stdout.split()[-1])  # bytes on macOS, kilobytes on Linux
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    assert peak_bytes <= 1.5e9  # whole process (PyTorch itself ~0.2 GB), operator build included
