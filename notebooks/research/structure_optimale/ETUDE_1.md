# Étude 1 : vers une structure de drone optimale (plan)

> Document de travail, suite de la phase d'exploration (`notebooks/research/introduction/`). Bibliographie, codes
> open source et ordre de lecture : [BIBLIOGRAPHIE.md](BIBLIOGRAPHIE.md). Légende : **[lit.]** vient de la
> littérature (résumés lus, pas les articles complets), **[idée]** est une proposition à nous, **[calcul]** est un
> ordre de grandeur calculé ici, à vérifier. Langue : français pour les textes, anglais pour le code.

## 0. Point de départ : ce que l'exploration a établi

| Résultat | Chiffre | Source |
|---|---|---|
| Bras sous le souffle : à rigidité ~égale, la matière devient verticale | traînée −36 % pour +3 % de souplesse (3D) ; section −35 % vs tube | `introduction/bras/v0_pipeline` |
| Châssis seul : le filtre de surplomb change la forme | +62 % de compliance, supports ÷ 100 | `introduction/chassis` §7 |
| Orthotropie FDM ($E_z/E_p$ = 0,7) | ~2 % sur la raideur : **pas** un argument | `introduction/chassis` §5.1 |
| Monolithe corps + bras (D/E) | filtre ×3,2 en compliance ; **D non convergé** ; embases = 1/3 de la masse ; les hélices traversent le corps | `introduction/chassis` §8 |

Deux leçons pour la suite : (1) la **raideur à volume fixé** est un objectif trop pauvre (J(D) > 1 alors que la
pièce est imprimable : on ne sait pas dire si elle est « bonne pour un drone ») ; (2) un run 3D coûte ~1 h sur CPU,
donc une étude à 50 runs doit se faire en grande partie sur des problèmes réduits.

## 1. Le plancher à dépasser : ce que fait Fusion

**[lit.]** D'après l'aide Autodesk ([objectifs](https://help.autodesk.com/cloudhelp/ENU/Fusion-GenerativeDesign/files/GD-OBJECTIVES.htm),
[contraintes](https://help.autodesk.com/cloudhelp/ENU/Fusion-GenerativeDesign/files/GD-SPECIFY-OBJECTIVES.htm),
[surplomb](https://help.autodesk.com/cloudhelp/ENU/Fusion-GenerativeDesign/files/GD-TERM-OVERHANG-ANGLE.htm)) :

| Fusion sait faire | Fusion ne fait pas (à notre connaissance, à revérifier sur la version actuelle) |
|---|---|
| min masse **ou** max raideur à masse cible | objectif exprimé en grandeurs de vol (temps de vol, perte de poussée, agilité) |
| coefficient de sécurité en contrainte, déplacement global/local | charges **dépendant de la forme** (souffle du rotor, traînée en vol avant) |
| **1re fréquence propre minimale** | bande interdite de fréquences (fenêtre moteur), centre de gravité, inerties |
| flambement (coefficient de sécurité) | résistance **anisotrope** entre couches (délaminage FDM) |
| additif : angle de surplomb + épaisseur mini | ponts tolérés, orientation de fabrication par pièce, couches courbes |
| plusieurs « outcomes » (matériaux, procédés) | exploration contrôlée des optimums locaux, représentation au choix |

**Conséquence** : ni la fréquence propre ni le surplomb à 45° ne distinguent ce solveur. Ce qui le peut, et que
l'étude doit démontrer chiffres à l'appui : **(i) l'aéro couplée**, **(ii) un objectif « drone » et non « pièce »**,
**(iii) la physique FDM au-delà de l'angle**, **(iv) l'architecture** (où couper, quoi rendre remplaçable). Un run
de référence Fusion (même domaine, mêmes charges, Additive 45°, contrainte de fréquence) reste nécessaire.

## 2. Reformuler le problème : une fonction de mérite « drone »

**[idée]** Remplacer « compliance normalisée à volume fixé » par des grandeurs physiques de vol, ramenées si possible
à une **masse équivalente** (grammes), pour que raideur, aéro et masse se comparent dans la même unité.

**Pourquoi l'aéro n'est pas cosmétique [calcul].** Un bras plat de 10 mm de large sous un disque de 5" (rayon
63,5 mm, moyeu 10 mm), $C_D$ ≈ 1,2, vitesse locale $k\,v_i$ avec $v_i=\sqrt{T/(2\rho A)}$ et $k$ = 1 à 1,5 :

$$\frac{D}{T}=\frac{C_D\,S\,k^2}{4A}\approx 1{,}3\ \text{à}\ 2{,}9\ \%\quad\text{(indépendant des gaz)},$$

soit **2 à 4 g par bras en stationnaire** (8 à 17 g pour un drone de 600 g), le même ordre que la masse d'un bras
imprimé. Comme la puissance de vol stationnaire varie en $(mg + D)^{3/2}$, **1 g de traînée de souffle coûte autant
d'autonomie qu'1 g de structure**. Cohérent avec [lit.] 2 à 8 % de perte de poussée par blocage, et jusqu'à ~5 à 20 %
d'écart d'efficacité selon la forme du bras en CFD (voir BIBLIOGRAPHIE §2.8).

**Les métriques de la « fiche drone »** (calculées pour **chaque** design, quel que soit l'optimiseur qui l'a produit) :

| Famille | Métrique | Unité | État du code |
|---|---|---|---|
| Masse | masse structurelle $m_s$ ; masse équivalente aéro $m_a = \sum D_{\rm bras}/g$ | g | $m_s$ oui ; $m_a$ à généraliser (existe pour un bras) |
| Autonomie | $\Delta t_{\rm vol}/t_{\rm vol} \approx -\tfrac32\,(\Delta m_s + m_a)/m$ | % | à faire |
| Raideur utile | flèche et **inclinaison de l'axe moteur** sous chaque cas (erreur de vecteur poussée) | mm, ° | compliance seulement |
| Vibrations | 1re fréquence propre $f_1$ ; écart à la bande moteur | Hz | absent |
| Dynamique | position du CG / plan des hélices, inerties $I_{xx}, I_{yy}, I_{zz}$ | mm, kg m² | `Body` existe |
| Vol avant | traînée du corps et **moment de tangage parasite** autour du CG | N, N m | absent |
| Résistance | indice de Hill max (agrégé), dont **entre couches** ; cas de choc latéral sur un moteur | – | `materials.hill_index` non branché |
| Imprimabilité | volume de supports, ponts, épaisseur mini, temps d'impression (trancheur en ligne de commande) | cm³, mm, min | voxel oui ; trancheur non |

**Interprétation de « aérodynamiquement stable » (à valider avec toi)** : (a) peu de perte de poussée sous les
hélices ; (b) en vol avant, centre de poussée aérodynamique du corps proche du CG, donc peu de moment parasite que les
moteurs doivent compenser ; (c) pas de résonance structurelle dans la bande des moteurs (sinon bruit gyro et filtres
plus lourds). Ordre de grandeur [calcul] : un 2306 à ~2400 KV en 4S tourne à ~150–600 Hz en vol ; la **mesure**
viendra des logs Blackbox (spectre gyro) sur le drone réel.

**Formulation cible (v1)** : $\min\ m_s + m_a$ sous inclinaison moteur ≤ $\theta_{\max}$ (par cas), $f_1 \ge f_{\min}$ ou
hors bande, Hill ≤ 1, CG dans une tolérance, filtre de fabrication. En pratique on garde l'ε-contrainte de la v0
pour tracer des **fronts de Pareto** (masse équivalente / raideur) plutôt qu'un seul point.

## 3. Les axes de l'étude

Chaque axe pose une question, des hypothèses **réfutables** et une règle de décision. Les numéros de runs renvoient
au plan d'expériences (§4).

### Axe A : corps + bras ou monolithe ?

| Variante | Description | Code |
|---|---|---|
| **A1** modulaire | corps (`frame.FrameProblem`) + 4 bras (`arm.Arm3DProblem`) vissés ; interfaces = patchs rigides + contrainte de résistance aux bossages | les deux existent, couplage à faire |
| **A2** monolithe | `drone.DroneProblem` (runs D/E), corrigé (§4, phase 0) | existe |
| A3 découpage optimisé | l'optimiseur choisit où couper et l'orientation d'impression de chaque pièce (Zhou, Nomura, Saitou 2019) | perspective |

Arguments propres au drone : en FPV **les bras cassent** (A1 se répare, A2 se jette) ; en A1 chaque pièce a **sa
propre orientation d'impression** (un bras imprimé sur chant n'a plus le même surplomb, cf. la section 2D de la v0)
; en A2 pas de vis ni de concentration de contrainte à la jonction.

* **H-A1** : à masse égale, A2 est plus raide que A1 en lacet et en tangage d'au moins 20 %.
* **H-A2** : l'avantage de A2 disparaît (< 5 %) quand A1 imprime ses bras dans leur orientation optimale.
* **H-A3** [idée] : on peut rendre le bras **fusible** (il casse avant le corps en choc latéral) sans perdre plus de
  5 % de raideur ; à mesurer par le rapport des charges de rupture bras/corps.

Décision : si H-A2 est vraie, l'architecture modulaire est retenue (réparabilité gratuite).

### Axe B : bloc de matière ou structure « avancée » au départ ?

SIMP n'est pas convexe : le point de départ et la représentation choisissent l'optimum local. Variantes :

| Variante | Départ / représentation | Idée testée |
|---|---|---|
| **B0** | bloc uniforme à `volfrac` (actuel) | référence |
| **B1** | B0 + bruit, 5 graines | dispersion des optimums locaux |
| **B2** | treillis optimisé (« ground structure », programme linéaire rapide) voxelisé | partir d'un squelette mécanique |
| **B3** | cadre FPV classique (plaques + bras tubes) voxelisé | « polir » un design humain |
| **B4** | barres explicites (projection géométrique, Norato) puis densité | membrures nettes, angles contrôlés (imprimables) |
| **B5** | reparamétrisation neuronale (CNN de Hoyer 2019, ou MLP de Fourier type TOuNN) | autre paysage d'optimisation, même physique |
| **B6** | coque + **remplissage** à densité variable (Wu et al. 2018) | une pièce FDM a déjà un infill : l'optimiser |
| B7 | prédiction d'un modèle génératif (OAT, NITO) en démarrage à chaud | perspective : modèles 2D, compliance seule |

* **H-B1** : la dispersion de J entre graines (B1) est < 5 %, mais les topologies diffèrent (à mesurer : distance
  IoU entre designs seuillés).
* **H-B2** : B2/B3/B4 donnent moins de parois fines et de supports, pour un coût en J ≤ 10 %.
* **H-B3** : B5 trouve un meilleur optimum que B0 plus souvent qu'à son tour (Hoyer : « meilleur design 50 % plus
  souvent » sur 116 cas 2D, à vérifier en 3D).
* **H-B4** : B6 bat B0 à masse égale en **charge de rupture** (pas en raideur, où la théorie favorise le plein).

Décision : adopter la stratégie qui donne le meilleur J **et** la meilleure fiche d'imprimabilité sur la médiane des
graines (pas sur le meilleur run, qui serait du picorage).

### Axe C : sans ou avec aéro, et à quelle fidélité ?

Hiérarchie de modèles, du moins cher au plus cher :

| Niveau | Modèle | Coût | Usage |
|---|---|---|---|
| **C0** | sans aéro | – | référence |
| **C1** | aire projetée sous chaque disque (occupation de colonne $1-\prod_z(1-\bar\rho)$, déjà dans `Arm3DProblem`), **pondérée par le profil radial de vitesse induite** [idée] + vol avant (aire frontale, moment autour du CG) | ~0 | optimisation |
| **C2** | C1 + $C_D$ local appris de la section 2D Brinkman (profilage) | faible | optimisation |
| **C3** | Brinkman Navier–Stokes 3D bas Reynolds sur un quart de drone, rotor = disque actionneur (terme source) ; ou LBM différentiable (XLB, GPU) | élevé | optimisation ciblée |
| **C4** | CFD RANS (OpenFOAM, disque actionneur ou MRF) sur les STL finaux ; **banc de poussée** physique | vérification seule | validation |

La question scientifique centrale : **un modèle bas-fidélité (C1/C2) classe-t-il les designs comme C4 ?** Mesure :
corrélation de rang (Spearman) entre la traînée prédite et la traînée C4 sur ~10 designs. Si oui, l'aéro « bon marché »
suffit pour optimiser et c'est un résultat en soi.

* **H-C1** : sur le drone entier, ≥ 30 % de traînée de souffle en moins pour ≤ 5 % de raideur en moins (comme sur le
  bras seul).
* **H-C2** : le gain net en masse équivalente ($\Delta m_s + m_a$) est positif.
* **H-C3** : C1 et C4 ont une corrélation de rang ≥ 0,8.

Prérequis : **vide imposé dans le volume balayé par les hélices** (+ 3 mm de garde), absent de la v0.

Validation physique bon marché [idée] : banc de poussée (balance + moteur + hélice), mesure de la poussée avec et
sans bras imprimé dessous, à la même vitesse de rotation. C'est la mesure directe de $m_a$.

### Axe D : fabrication FDM au-delà de la règle des 45°

| Variante | Contenu | Statut |
|---|---|---|
| **D1** | filtre 45° actuel (Langelaar) | existe |
| **D2** | filtre tolérant les **ponts** courts entre deux appuis (portée ≤ $L_b$ ≈ 20–40 mm) | [idée] ; quasi rien trouvé dans la littérature : contribution possible |
| **D3** | orientation de fabrication par pièce (A1) | à faire |
| **D4** | résistance anisotrope (Hill, Tsai–Wu) agrégée en $P$-norme, dont traction **entre couches** | `materials.hill_index` à brancher |
| **D5** | longueur minimale robuste (projections érodée/dilatée, Wang–Lazarov–Sigmund) ≥ 2 largeurs d'extrusion | à faire |
| D6 | couches courbes / multi-axes (espace-temps, S³-Slicer) | lecture seulement (demande une imprimante multi-axes) |

* **H-D1** : D2 récupère au moins la moitié du surcoût du filtre 45° (les +62 % du châssis viennent de la plaque
  supérieure, qu'un pont imprimerait).
* **H-D2** : sans D4, les designs filtrés à 45° (parois inclinées) cassent **entre couches** à une charge plus faible
  que ce que prédit le modèle isotrope ; à vérifier sur pièce imprimée.

### Critères transverses propres au drone

Fréquence propre (contrainte $f_1 \ge f_{\min}$ ou bande interdite ; attention aux modes parasites des zones de
faible densité, interpolation de masse adaptée), CG et inerties (contrainte ou terme d'objectif), cas de choc latéral,
volume des hélices et des composants imposé. Ils s'ajoutent progressivement, pas tous dès la phase 1.

## 4. Plan d'expériences

Pas de plan factoriel complet (2 × 8 × 5 × 6 cas = trop cher). Une séquence où chaque phase fixe un choix pour la
suivante. Coûts estimés sur la base des runs de l'exploration (~1 h par run 3D à 2–3 mm sur CPU).

| Phase | Contenu | Problème | Runs | Coût estimé | Sortie |
|---|---|---|---|---|---|
| **0** prérequis | fiche drone (§2) dans un module d'évaluation unique ; fiabiliser D (continuation plus lente, premier pas MMA réduit) ; évider les embases ; vide des hélices ; références : cadre FPV humain voxelisé, run Fusion | – | 2–3 | ~1 semaine | banc de mesure commun |
| **1** axe B | B0…B6 × 3 graines (B1 : 5) | châssis seul, maille 4 mm (~8× moins de mailles) | ~23 | ~5 min/run, ~2 h en parallèle | stratégie de départ retenue ; H-B1…B4 |
| | les 2–3 meilleures stratégies à 2 mm | châssis seul | 3 | ~3 h | confirmation à résolution fine |
| **2** axe A | A1 vs A2, raideur + $f_1$, sans aéro, avec la stratégie de la phase 1 | drone entier | 4 | ~4 h | H-A1…A3, architecture retenue |
| **3** axe C | C0, C1, C2 sur l'architecture retenue ; C4 sur tous les finaux ; banc de poussée | drone entier | 3 + CFD | ~1 semaine (CFD comprise) | front de Pareto masse équivalente / raideur ; H-C1…C3 |
| **4** axe D | D1 vs D2 (+ D4, D5) ; impression et essais | architecture retenue | 4 + impressions | ~1 semaine | H-D1, H-D2 ; pièce volante |

Le niveau C3 (Navier–Stokes ou LBM 3D) n'est lancé **que si** la phase 3 montre que C1/C2 classent mal (H-C3 fausse).

## 5. Règles de comparaison (pour que les chiffres tiennent)

1. Même masse (ou même contrainte active) entre variantes comparées ; écart de masse rapporté sinon.
2. Tous les designs sont **réévalués** par le même banc : densité seuillée (0/1), maillage fin, mêmes cas de charge,
   fiche drone complète. Jamais sur leur propre objectif gris.
3. Médiane et dispersion sur les graines, pas seulement le meilleur run.
4. Tous les runs rapportés, y compris les échecs (non-convergence, îlots, divergence du solveur).
5. Temps de calcul et itérations rapportés ; graines et configs enregistrées avec les résultats.
6. La vérification haute fidélité (C4, essais) départage, pas le modèle utilisé pour optimiser.

## 6. Ce qu'il faut coder, dans l'ordre

1. `aeroptimizer/evaluation.py` : la fiche drone (§2), pure, sur une densité seuillée + un `DroneSetup`.
2. Corrections de `drone.DroneProblem` : embases évidées, vide des hélices, continuation de D.
3. `fem/modal.py` : valeurs propres généralisées (LOBPCG ou `eigsh` en mode shift-invert) + sensibilités ; contrainte
   $f_1$ (modes multiples : agrégation KS).
4. `topopt/init.py` : départs B0–B4 ; `topopt/neural.py` pour B5 (le pipeline étant en PyTorch, B5 est presque gratuit).
5. Aéro C1 généralisée au drone entier (profil radial de vitesse, vol avant, moment autour du CG).
6. `topopt/bridge.py` (D2) et branchement de `hill_index` (D4).
7. Export vers un trancheur en ligne de commande (PrusaSlicer/Orca) pour le temps d'impression et les supports réels.

## 7. Risques

| Risque | Parade |
|---|---|
| coût 3D sur CPU float64 | problèmes réduits (phase 1), symétrie, GPU CUDA (Träff et al. 2023 : multigrille sans matrice) si disponible |
| modes propres parasites / croisement de modes | interpolation de masse modifiée, agrégation KS, suivi des modes |
| aéro bas Reynolds non représentative | C4 + banc de poussée tranchent ; l'objectif est le **classement**, pas la valeur absolue |
| non-convergence avec filtres empilés (D) | continuation plus lente, critère d'arrêt sur la stabilité de la forme |
| dispersion du sujet | une phase = une question ; ne pas ouvrir C3 ni D6 sans résultat qui l'exige |

## 8. Questions ouvertes (à trancher avant la phase 0)

1. « Aérodynamiquement stable » : l'interprétation du §2 (perte de poussée, moment parasite en vol avant, vibrations)
   te convient-elle, ou pensais-tu à autre chose (stabilité en vol avant à grande vitesse, rafales) ?
2. Accès à Fusion pour le run de référence ?
3. GPU NVIDIA disponible (change la faisabilité de C3 et des maillages fins) ?
4. Banc de poussée et imprimante disponibles pour les validations des phases 3 et 4 ?
