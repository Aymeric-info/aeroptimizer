# aeroptimizer

Solveur d'optimisation topologique **différentiable** (PyTorch) pour le bras d'un drone imprimé en FDM :
compromis **rigidité / traînée dans le souffle du rotor**, filtre de surplomb auto-supportant.

**État : v0 de bout en bout.** La pipeline complète tourne et est présentée dans
[notebooks/research/bras/v0_pipeline.ipynb](notebooks/research/bras/v0_pipeline.ipynb) (résultats 2D et 3D, validation de chaque brique) :

* **2D, section du bras** (`arm.SectionProblem`) : une densité dans le plan $(y, z)$ de la section ;
  structure = rigidités de poutre de Timoshenko (flexion, cisaillement, torsion par EF de gauchissement,
  `section.py`) ; fluide = Navier–Stokes stationnaire avec pénalisation de Brinkman (`flow/brinkman.py`),
  traînée = intégrale de la force de Brinkman ; couplage par ε-contrainte ou somme pondérée ; MMA.
* **3D, bras complet** (`arm.Arm3DProblem`) : EF 3D, trois cas de charge (une factorisation),
  traînée par théorie des tranches sous le disque du rotor.

FORMALISM/CONTRACTS/ARCHITECTURE décrivent la conception initiale (écoulement potentiel) ; la v0 s'en écarte
sur le modèle fluide (d'Alembert : le potentiel ne donne aucune traînée) et sur le plan 2D (section plutôt que
vue de côté). Les leçons de `notebooks/lessons` restent la base pédagogique des briques communes.

```bash
pip install -e ".[dev]"
pytest                                          # ~15 s
jupyter nbconvert --to notebook --execute --inplace notebooks/research/bras/v0_pipeline.ipynb --ExecutePreprocessor.timeout=-1
```

**Châssis (sans les bras), v0** : [notebooks/research/chassis/v0_chassis_sans_bras.ipynb](notebooks/research/chassis/v0_chassis_sans_bras.ipynb),
expérience A/B du filtre de surplomb décrite dans [CDC_CHASSIS.md](notebooks/research/chassis/CDC_CHASSIS.md)
(`frame.FrameProblem`, `printability.py`, solveur `fem.solvers.AMGCG` qui demande `pip install pyamg`) :

```bash
python notebooks/research/chassis/build_notebook.py   # régénère le notebook depuis son script
python notebooks/research/chassis/runs.py A B C       # les 3 optimisations, en parallèle, mises en cache (~45 min)
jupyter nbconvert --to notebook --execute --inplace notebooks/research/chassis/v0_chassis_sans_bras.ipynb
```

## Documents (à lire dans cet ordre)

| Document | Rôle |
|---|---|
| [FORMALISM.md](FORMALISM.md) | Formulation mathématique exacte, hypothèses, limites |
| [CONTRACTS.md](CONTRACTS.md) | Contrats d'interface/invariants, chacun relié à un test |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Conception logicielle, feuille de route, risques, décisions |

## Démarrage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/contracts_index.py --check   # chaque contrat de CONTRACTS.md a son test
```

Plateforme de référence : **CPU, float64** (MPS ne supporte pas float64, voir ARCHITECTURE §7).

## Conventions à retenir

* Axes : 2D `(x, z)` (ou `(y, z)` pour la section du bras), 3D `(x, y, z)` ; **`z` = dernier axe = direction de fabrication = verticale = écoulement**.
* SI aux interfaces, `float64`, ordre C (dernier axe le plus rapide).
* Les gradients viennent d'Autograd ; seuls les solveurs (`solve_spd`, Newton de Navier–Stokes) ont un backward manuel (adjoint).
