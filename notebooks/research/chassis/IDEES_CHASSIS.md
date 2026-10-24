# Idées et état de l'art pour le châssis (notes de travail)

> Complète [CDC_CHASSIS.md](CDC_CHASSIS.md). Légende : **[lit.]** vient d'une recherche web (résumés lus, pas les articles
> complets : à vérifier avant de citer) ; **[idée]** est une idée à nous, non tirée d'un papier.

## 1. État de l'art (recherche du 2026-10-21)

* **[lit.] Aéro dans l'optimisation de châssis** : quasi absente. Les papiers de châssis de quadricoptère font de la
  SIMP (parfois + level-set) sous poussée et couple, puis imprimante ; la CFD n'intervient qu'après coup (traînée,
  fatigue, soufflerie). Aucun travail trouvé qui couple la topologie au souffle du rotor. Un papier récent note que le
  downwash est rarement pris en compte en conception ([arXiv 2602.18344](https://arxiv.org/pdf/2602.18344)).
  Sources : [MDPI 2024](https://www.mdpi.com/2411-9660/8/3/58), [MDPI unibody](https://www.mdpi.com/2411-9660/6/1/8),
  [IOP](https://iopscience.iop.org/article/10.1088/1757-899X/1305/1/012021/pdf).
* **[lit.] Surplomb** : Langelaar 2017 ([doi 10.1007/s00158-016-1522-2](https://link.springer.com/content/pdf/10.1007/s00158-016-1522-2.pdf)),
  Gaynor & Guest 2016, Garaigordobil et al., Qian 2017, méthodes par pénalisation/EDP.
  `topopt/overhang.py` est du type Langelaar (couches + stencil 45°), non comparé ligne à ligne à l'article.
* **[lit.] Anisotropie FDM** : orthotropie SOMP, trajectoires de dépôt, orientation de trame, Tsai–Wu
  ([Add. Manuf. 2023](https://www.sciencedirect.com/science/article/abs/pii/S2214860423003433),
  [MDPI](https://www.mdpi.com/2072-666X/11/8/709)). Surplomb + anisotropie ensemble : plus proche =
  [arXiv 2502.20343](https://arxiv.org/pdf/2502.20343) (multi-axes).
* **[lit.] Fusion (design génératif)** a déjà la contrainte de surplomb (mode Additive, 45° par défaut, épaisseur
  minimale) ; Autodesk reconnaît que les supports sont réduits, pas supprimés
  ([aide](https://help.autodesk.com/cloudhelp/ENU/Fusion-GenerativeDesign/files/GD-TERM-OVERHANG-ANGLE.htm),
  [blog](https://www.autodesk.com/products/fusion-360/blog/unlocking-better-additive-manufacturing-outcomes-generative-design/)).
  À vérifier : Fusion gère-t-il un matériau anisotrope ?

## 2. Positionnement face à Fusion

Le surplomb seul n'est pas une nouveauté. Ce qui reste défendable :

1. **[idée]** effet **combiné** surplomb + orthotropie FDM (Z faible) sur un châssis de drone, mesuré (A/B du CDC) ;
2. **[idée]** solveur ouvert et différentiable : effet du filtre isolé, contraintes ajoutables (fréquence, rupture),
   résultats reproductibles, contre une boîte noire cloud ;
3. **[idée]** **run de référence Fusion** : même domaine, charges et masse, Additive 45° ; comparer compliance, volume de
   supports et pièce imprimée à notre run B. Documenter aussi le cas où Fusion fait mieux.

Pas de nouveauté méthodologique à prétendre : c'est un cas d'étude rigoureux avec un solveur ouvert.

## 3. Spécialiser le solveur aux drones

| Spécificité | Apport au solveur | État du code |
|---|---|---|
| **[idée]** Charges issues de l'enveloppe de vol | Poussée max, facteur de charge, couple moteur, rafale, atterrissage → cas de charge générés | Cas posés à la main dans `Arm3DSetup` |
| **[idée]** Fréquence propre | 1er mode loin des fréquences moteur/filtre gyro (souvent la contrainte réelle d'un châssis FPV) ; dérivée de valeur propre à implémenter (Du & Olhoff 2007, à revérifier) | Absent |
| **[idée]** Masse → autonomie | Convertir le gain de masse en minutes de vol (puissance de vol stationnaire) | Absent |
| **[idée]** Intégration composants | Primitives `solid`/`void` : moteurs, batterie, pile FC, câbles, **cylindres balayés par les hélices** ; contrainte de centre de gravité | Masques existent, pas les primitives |
| Aéro localisée | Bras sous le disque rotor ; traînée du corps en vol avant | Bras fait (v0), châssis non |
| Fabrication FDM | Surplomb, orthotropie, épaisseur minimale, rupture | Surplomb, orthotropie ; `hill_index` à brancher |
| Choc | Statique équivalent (batterie × n g) ; vrai choc = autre problème | Absent |

**[idée]** Une couche « description du drone » (empattement, taille d'hélice, masse, positions des composants) qui génère
domaine, masques, appuis et cas de charge, avec sorties en unités métier (1er mode, flèche en bout de bras, masse,
autonomie) et classes de taille (ex. 5" FPV) via `scaling.py`. C'est ce qui rend le solveur spécifique.

## 4. Priorités pour un mois

1. Génération du problème depuis les paramètres du drone. 2. Expérience A/B du filtre (+ référence Fusion si
l'accès est possible). 3. Fréquence propre si le reste est stable. En perspective, sans implémenter : autonomie, choc.

## 5. Validation physique (sans elle, la spécialisation reste une affirmation)

* Essai en flexion de la pièce imprimée (A et B). * Mesure de vibration à l'accéléromètre pour la fréquence propre.
* Éprouvettes de traction dans le plan et selon Z sur **notre** imprimante et notre bobine (les fiches matériau varient
beaucoup d'un lot à l'autre).
