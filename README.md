# aeroptimizer

Solveur d'optimisation topologique **différentiable** (PyTorch) pour le bras et le châssis central
d'un drone imprimé en FDM : couplage compliance + téléchargement aérodynamique (écoulement potentiel),
loi orthotrope pour l'anisotropie inter-couches, filtre de surplomb auto-supportant.

**État : squelette (phase 0).** Aucun solveur n'est encore implémenté ; les contrats sont des tests `skip`.

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
pytest                                   # tout est skip tant que les phases ne sont pas atteintes
python scripts/contracts_index.py --check   # chaque contrat de CONTRACTS.md a son test
```

Plateforme de référence : **CPU, float64** (MPS ne supporte pas float64, voir ARCHITECTURE §7).

## Conventions à retenir

* Axes : 2D `(x, z)`, 3D `(x, y, z)` ; **`z` = dernier axe = direction de fabrication = verticale = écoulement**.
* SI aux interfaces, `float64`, ordre C (dernier axe le plus rapide).
* Les gradients viennent d'Autograd ; seuls les solves linéaires ont un backward manuel (adjoint).
