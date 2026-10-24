#!/usr/bin/env python
"""Résultat d'optimisation (.npy) -> STL lissé, en mm, à glisser dans Fusion.

    python scripts/npy_vers_stl.py a3d_aero03          # bras 3D, rigidité + souffle
    python scripts/npy_vers_stl.py a3d_rigidite        # bras 3D, rigidité seule
    python scripts/npy_vers_stl.py s2d_eps1_surplomb   # section 2D auto-supportée, extrudée sur la longueur du bras

Le .npy contient les variables de conception : on les repasse par le filtre et la projection (comme
`final` dans research/v0_pipeline.ipynb), on maille le champ à 0,5, on lisse (Taubin, volume conservé).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from aeroptimizer.arm import Arm3DProblem, Arm3DSetup, SectionProblem, SectionSetup  # noqa: E402
from aeroptimizer.io.mesh import marching_tetrahedra, mesh_report, smooth_taubin, write_stl  # noqa: E402
from aeroptimizer.topopt.overhang import OverhangParams  # noqa: E402

torch.set_default_dtype(torch.float64)
DOSSIER = next(p for p in (RACINE / "notebooks" / "research" / "bras" / "results", RACINE / "research" / "results") if p.is_dir())
N_LONG = 60  # tranches le long du bras pour la section extrudée


def champ_3d(nom: str):
    S = Arm3DSetup()
    w = {"C": 1.0, "D": 0.0} if nom == "a3d_rigidite" else {"C": 0.7, "D": 0.3}
    pb = Arm3DProblem(S, weights=w)
    pb.set_continuation({"p": 3.0, "beta": 8.0, "beta_s": 80.0})
    with torch.no_grad():
        rho = pb.evaluate(torch.from_numpy(np.load(DOSSIER / f"{nom}_x.npy"))).state[0].numpy()
    lo, hi = np.zeros(3), np.array([S.L, S.W, S.H])
    return rho, (S.h,) * 3, (lo, hi)


def champ_section(nom: str):
    base = SectionSetup(W=12.0, H=24.0)
    S3 = Arm3DSetup()
    pb = SectionProblem(replace(base, overhang=OverhangParams(beta_s=10.0)), mode="eps", c_max=1.0)
    pb.set_continuation({"p": 3.0, "beta": 16.0, "beta_s": 80.0})
    with torch.no_grad():
        rho = pb.evaluate(torch.from_numpy(np.load(DOSSIER / f"{nom}_x.npy"))).state[0].numpy()
    champ = np.repeat(rho[None], N_LONG, axis=0)  # (x, y, z)
    lo, hi = np.array([0.0, -np.inf, -np.inf]), np.array([S3.L, np.inf, np.inf])  # seuls les bouts restent plans
    return champ, (S3.L / N_LONG, base.h, base.h), (lo, hi)


PIECES = {"a3d_aero03": champ_3d, "a3d_rigidite": champ_3d, "s2d_eps1_surplomb": champ_section}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("nom", choices=PIECES)
    ap.add_argument("--iterations", type=int, default=10, help="passes de lissage (0 : maillage brut)")
    args = ap.parse_args()

    champ, h, box = PIECES[args.nom](args.nom)
    V, F = marching_tetrahedra(champ, 0.5, h)
    v0 = mesh_report(V, F)["volume"]
    if args.iterations:
        V = smooth_taubin(V, F, iterations=args.iterations, box=box)
    rep = mesh_report(V, F)
    sortie = DOSSIER / f"{args.nom}_fusion.stl"
    write_stl(sortie, V, F, name=args.nom)
    print(f"{sortie}\n  {rep['n_faces']} triangles, étanche : {rep['watertight']}, "
          f"volume {rep['volume']:.0f} mm³ ({100 * (rep['volume'] / v0 - 1):+.2f} % par le lissage)")
    print("  boîte (mm) :", V.min(0).round(1), "->", V.max(0).round(1))


if __name__ == "__main__":
    main()
