"""Runs D / E of the monolithic drone (body + 4 arms in one part), shared by the notebook and the command line.

    python notebooks/research/chassis/assemblage.py D E      # computes the missing runs, in parallel
    python notebooks/research/chassis/assemblage.py D --force

Results are cached in results/: drone_<run>_x.npy (design variables) and drone_<run>_hist.json (history).
D: 45 deg overhang filter along z (what we print). E: same without the filter (reference).
Everything else is identical (domain, loads, volfrac, filter, continuation, MMA, starting point).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from multiprocessing import Process
from pathlib import Path

import numpy as np
import torch

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parents[2] / "src"))

from aeroptimizer.drone import DroneProblem, DroneSetup  # noqa: E402
from aeroptimizer.fem.solvers import AMGCG  # noqa: E402
from aeroptimizer.optim.continuation import Schedule  # noqa: E402
from aeroptimizer.optim.loop import run  # noqa: E402
from aeroptimizer.optim.mma import MMA  # noqa: E402
from aeroptimizer.topopt.overhang import OverhangParams  # noqa: E402

CACHE = ICI / "results"
BASE = DroneSetup()
RUNS = {
    "D": replace(BASE, overhang=OverhangParams(beta_s=10.0)),
    "E": replace(BASE),
}
ITS = 150
SCHEDULE = Schedule(p=(3.0, 3.0), beta=(1.0, 16.0), beta_s=(10.0, 80.0), every=25)
FINAL = {"p": 3.0, "beta": 16.0, "beta_s": 80.0}


def problem(name: str, refs: dict[str, float] | None = None) -> DroneProblem:
    torch.set_default_dtype(torch.float64)
    # the first MMA steps can cut an arm off (near-singular K): AMG-CG gives up after 300 iterations, SuperLU takes over
    return DroneProblem(replace(RUNS[name]), refs=refs, solver=AMGCG(max_iter=300, fallback=True))


def optimise(name: str, force: bool = False, its: int = ITS) -> tuple[np.ndarray, list[dict]]:
    """Design variables and history of run `name`, from the cache or computed (then cached)."""
    fx, fh = CACHE / f"drone_{name}_x.npy", CACHE / f"drone_{name}_hist.json"
    if fx.exists() and fh.exists() and not force:
        return np.load(fx), json.loads(fh.read_text())
    CACHE.mkdir(exist_ok=True)
    torch.manual_seed(0)
    pb = problem(name)
    t0 = time.perf_counter()

    def log(it, x, row):  # noqa: ANN001
        if it % 10 == 0 or it < 3:
            print(f"[{name}] it {it:3d}  J {row['J']:.4f}  vol {row['vol']:.3f}  beta {row['beta']:g}  "
                  f"{row['t_it']:.1f} s/it  CG {getattr(getattr(pb.solver, 'last', None), 'iterations', '-')}", flush=True)

    r = run(pb, MMA(n_constraints=1, move=0.2), SCHEDULE, (log,), max_iter=its, tol=1e-3)
    np.save(fx, r.x)
    fh.write_text(json.dumps(r.history))
    print(f"[{name}] {len(r.history)} iterations in {time.perf_counter() - t0:.0f} s ({r.stop_reason})", flush=True)
    return r.x, r.history


def final(pb: DroneProblem, x: np.ndarray):  # noqa: ANN201
    """Evaluation of a design at the final continuation settings, without gradient."""
    pb.set_continuation(FINAL)
    with torch.no_grad():
        return pb.evaluate(torch.from_numpy(np.asarray(x)))


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(RUNS)
    force = "--force" in sys.argv
    torch.set_num_threads(4)
    procs = [Process(target=optimise, args=(n, force)) for n in names]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
