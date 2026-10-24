# Cahier des charges temporaire : passage au châssis (v1)

> Document de travail pour un agent de code. À supprimer ou à fusionner dans FORMALISM/ARCHITECTURE une fois le
> châssis stabilisé. Langue : français pour les textes et notebooks, anglais pour le code et les identifiants.
> Le formalisme/les contrats existants peuvent être contournés (cf. README). Précision physique non requise :
> les nombres doivent être plausibles, pas calés.

## 1. Question à trancher

**Le filtre de surplomb (`topopt/overhang.py`) change-t-il réellement le résultat, et la pièce obtenue est-elle
imprimable en FDM ?** C'est la raison d'être d'un solveur maison : sans effet mesurable du filtre, rien ne le
distingue d'un outil existant (les logiciels commerciaux de design génératif proposent déjà des contraintes
d'impression, à vérifier avant de s'en prévaloir). Un résultat négatif est acceptable, mais il doit être
documenté tel quel. L'aérodynamique est **hors périmètre** (le centre du châssis est hors du souffle des rotors).

## 2. Protocole de l'expérience (le livrable principal)

Deux optimisations à iso-tout (même domaine, charges, `volfrac`, filtre de densité, continuation, graine, MMA) :

| Run | Filtre de surplomb | Matériau |
|---|---|---|
| A | non | orthotrope FDM (`Ez < Ep`) |
| B | oui (45°, axe de fabrication = dernier axe `z`) | orthotrope FDM |

Un run C optionnel (isotrope, sans filtre) isole l'effet de l'orthotropie de celui du filtre.

Métriques à reporter pour chaque run, dans un tableau :

* compliance par cas de charge et masse (`vol`) ;
* **taux de surplomb** : fraction des voxels solides (`rho_bar > 0.5`, hors zones `solid` passives) de la couche
  `k` sans voxel solide dans le stencil 3×3 de la couche `k-1` (couche 0 = plateau, exclue) ;
* **volume de supports nécessaire** : voxels vides sous un voxel solide en surplomb, jusqu'au plateau ;
* épaisseur minimale (`io.mesh.thin_fraction`) et rapport de maillage (`io.mesh.mesh_report`) ;
* temps et nombre d'itérations.

Imprimabilité : exporter le STL (`marching_tetrahedra` + `write_stl`), puis, si un slicer en ligne de commande est
disponible (PrusaSlicer/Cura), relever le volume de supports estimé. Sinon, le taux de surplomb voxel suffit.
**Test final, hors code : imprimer B** (et A si le temps le permet).

## 3. Piège à éviter : l'axe de fabrication

Le filtre agit le long du **dernier axe**. Une plaque plate imprimée à plat est extrudée selon `z` : il n'y a
aucun surplomb, donc **aucun effet du filtre**, et l'expérience ne prouve rien. Le domaine doit être
réellement 3D le long de `z` : plaques superposées avec entretoises, plateau batterie surélevé, capot, ou plaque
imprimée sur chant. Un prototype 2D vue de dessus (x,y) ne teste donc **pas** le filtre ; s'il faut du 2D, utiliser
une coupe (x,z) (cf. `configs/drone_arm_2d.yaml`).

## 4. Modèle du châssis (v1 minimal)

* **Domaine** : boîte 3D, grille structurée (`Grid`), `h` ~ 2–3 mm au début. Symétrie miroir sur la densité
  uniquement (comme `Arm3DProblem.rho_bar`) ; les charges restent asymétriques.
* **Masques** (`Masks`) : `solid` = 4 embases moteur, trous de pile FC (30,5 × 30,5 mm, M3), appuis de la batterie ;
  `void` = baie batterie et passage de câbles. `check_solid_supported` doit passer pour les zones `solid`.
* **Appuis** : pas d'encastrement. Charges auto-équilibrées (poussée des 4 moteurs = poids des composants,
  appliqué en forces nodales sur leurs zones passives) + 3 appuis ponctuels isostatiques (3-2-1) à réaction nulle
  (`fem/bc.py::check_rigid_modes`).
* **Cas de charge** (une factorisation, plusieurs seconds membres) : hover, tangage/roulis (poussées
  différentielles), lacet (couples alternés), choc à l'atterrissage (poids batterie × 5–10 g).
* **Objectif** : minimiser `sum_k w_k C_k` à `volfrac` fixé (ou masse sous contrainte de compliance, ε-contrainte
  comme en v0). **Référence** : plaque pleine de même masse, pour normaliser `C`.
* **Matériau** : `materials.Orthotropic` avec `Ez` < `Ep` (valeurs plausibles du PLA/PETG, non calées).
* **Solveur** : `ScipyDirect` si la grille reste petite (< 50 000 éléments), sinon `TorchCG` avec Jacobi.

## 5. Plan de code

Réutiliser sans modifier : `topopt.density.DensityPipeline`, `topopt.overhang`, `fem.*`, `optim.mma`,
`optim.continuation`, `io.mesh`. Modèle à suivre : `arm.Arm3DProblem` (même interface : `n_design`,
`initial_design`, `set_continuation`, `rho_bar`, `physics`, `evaluate`, `reference_values`).

À créer :

1. `src/aeroptimizer/frame.py` : `FrameSetup` et `FrameProblem` (option `overhang: bool`).
2. `src/aeroptimizer/printability.py` : taux de surplomb voxel et volume de supports (fonctions pures sur un tableau
   booléen, indépendantes du filtre, donc utilisables pour juger A comme B).
3. `tests/unit/test_frame.py`, `tests/unit/test_printability.py` : équilibre des charges (somme des forces nulle),
   réactions nulles aux appuis 3-2-1, taux de surplomb nul sur un cube posé et non nul sur un T renversé,
   `FrameProblem.evaluate` différentiable (gradient contre différences finies sur quelques variables).
4. Notebook `notebooks/research/chassis_v1.ipynb` (script générateur dans le même esprit que `build_notebook.py`),
   résultats mis en cache dans `notebooks/research/results/` (`frame_A_*`, `frame_B_*`).

## 6. Hors périmètre (ne pas faire en v1)

Aérodynamique (Brinkman, traînée), contrainte de fréquence propre, contrainte de contrainte (`hill_index`),
variante monolithique bras + châssis, calibrage matériau. À reprendre selon le résultat de la section 2.

## 7. Critère de fin

Tableau A/B (section 2) rempli, images des deux formes (coupes + 3D), STL des deux, et une conclusion en trois
phrases : le filtre a-t-il un effet ? à quel coût en compliance ? B est-il imprimable sans supports ?
`pytest` vert.
