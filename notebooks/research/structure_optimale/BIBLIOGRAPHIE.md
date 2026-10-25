# Bibliographie et codes open source : Étude 1

> Complète [ETUDE_1.md](ETUDE_1.md). Recherche du 2026-10-25. Statut de chaque référence : **(R)** résumé ou page
> consultés pendant la recherche ; **(M)** cité de mémoire, référence exacte à vérifier avant de citer. Aucun article
> n'a été lu en entier : les chiffres annoncés sont ceux des auteurs.

## 1. À lire en premier (ordre conseillé)

| # | Référence | Pourquoi pour nous | À en retirer |
|---|---|---|---|
| 1 | Alexandersen (2023), *A detailed introduction to density-based topology optimisation of fluid flow problems with implementation in MATLAB*, SMO 66. [arXiv 2207.13695](https://arxiv.org/abs/2207.13695) **(R)** | la référence pédagogique de notre `flow/brinkman.py` ; code TOPflow fourni | choix des paramètres de Brinkman, précision de l'interface diffuse (notre +6 % de traînée sur le cylindre) |
| 2 | Lundgaard, Alexandersen, Zhou, Andreasen, Sigmund (2018), *Revisiting density-based topology optimization for fluid-structure-interaction problems*, SMO 58(3). [Springer](https://link.springer.com/article/10.1007/s00158-018-1940-4) **(R)** | structure soumise à des **charges fluides qui dépendent de la forme** : exactement le souffle sur le bras | formulation unifiée, objectifs pondérés contre les îlots flottants ; limite : 2D stationnaire |
| 3 | Du, Schulz, Zhu, Bickel, Matusik (2016), *Computational Multicopter Design*, SIGGRAPH Asia. [projet + code](https://copter.csail.mit.edu/) **(R)** | le seul travail trouvé qui optimise un multirotor **sur des critères de vol** (charge utile, batterie, contrôleur) | comment passer de la pièce au drone : notre « fiche drone » |
| 4 | Langelaar (2017), *An additive manufacturing filter for topology optimization of print-ready designs*, SMO. [PDF](https://link.springer.com/content/pdf/10.1007/s00158-016-1522-2.pdf) **(R)** puis van de Ven et al., front propagation : [SMO 2018](https://link.springer.com/article/10.1007/s00158-017-1880-4), [CMAME 2020, 3D](https://reports.nlr.nl/items/0331347f-ed48-4099-8124-a0cee5ccca52) **(R)** | notre filtre actuel, et une alternative à angle libre, maillage quelconque, ~10× moins chère que l'EF | base de D2 (filtre tolérant les ponts) |
| 5 | Hoyer, Sohl-Dickstein, Greydanus (2019), *Neural reparameterization improves structural optimization*. [arXiv 1909.04240](https://arxiv.org/abs/1909.04240) **(R)** + Woldseth, Aage, Bærentzen, Sigmund (2022), *On the use of artificial neural networks in topology optimisation*. [arXiv 2208.02563](https://arxiv.org/abs/2208.02563) **(R)** | B5 ; et la revue **critique** du DTU : peu de vrais succès du ML en TO | ce qui marche (reparamétrisation, même physique) vs ce qui ne marche pas (prédiction sans itération) |
| 6 | Zhou, Nomura, Saitou (2019), *Anisotropic multicomponent topology optimization for additive manufacturing with build orientation design and stress-constrained interfaces*. [arXiv 1911.10393](https://arxiv.org/abs/1911.10393) **(R)** | axe A : l'optimiseur choisit le **découpage** en pièces et **l'orientation d'impression** de chacune, avec contrainte aux jonctions | A3 ; modélisation des interfaces de A1 |
| 7 | Mirzendehdel, Rankouhi, Suresh (2018), *Strength-based topology optimization for anisotropic parts*, Addit. Manuf. [PDF](https://ersl.wisc.edu/publications/2018/AnisotropicStress.pdf) **(R)** | FDM : c'est la **résistance** entre couches qui est anisotrope, pas tant la raideur (cohérent avec notre ~2 %) | D4 : Tsai–Wu/Hill dans la boucle |
| 8 | Wu, Aage, Westermann, Sigmund (2018), *Infill optimization for additive manufacturing — approaching bone-like porous structures*, IEEE TVCG. [arXiv 1608.04366](https://arxiv.org/abs/1608.04366) **(R)** | B6 : coque + remplissage, naturel en FDM | contrainte de volume locale, gain en robustesse |
| 9 | Wang, Munro, Wang, van Keulen, Wu (2019), *Space-time topology optimization for additive manufacturing*, SMO. [Springer](https://link.springer.com/article/10.1007/s00158-019-02420-6) **(R)** + [arXiv 2502.20343](https://arxiv.org/abs/2502.20343) (multi-axes, surplomb + anisotropie, 2025) **(R)** | le **procédé** (ordre de dépôt) comme variable d'optimisation | perspective D6, et l'idée d'un champ de pseudo-temps |
| 10 | Bayat, Li, Alexandersen (2026), RANS k-ε en densité avec loi de paroi implicite. [arXiv 2601.02202](https://arxiv.org/abs/2601.02202) **(R)** | l'état de l'art pour sortir du bas Reynolds (notre limite $Re$ = 100) | ce que coûte vraiment le turbulent en TO |

## 2. Par thème

### 2.1 Aéro et fluide en optimisation topologique

* Borrvall & Petersson (2003) ; Kondoh, Matsumori, Kawamoto (2012) ; Alexandersen & Andreasen (2020, revue *Fluids*)
  : déjà cités dans la v0 **(M)**.
* *Efficient multi-stage aerodynamic topology optimization using an operator-based analytical differentiation*, SMO
  2022 : traînée/portance 3D par intégrale de Brinkman sur grille cartésienne, −45 % de coût en 3D par
  multi-niveaux. [Springer](https://link.springer.com/article/10.1007/s00158-022-03208-x) **(R)**
* Dilgen, Dilgen, Fuhrman, Sigmund, Lazarov (2018), *Topology optimization of turbulent flows*, CMAME **(M)**.
* Turbulent Spalart–Allmaras en TO : Garcia-Rodriguez et al. (2022, Appl. Math. Modelling) ; cut-cell + SA,
  Galanos et al. (Energies 2024) **(R, via résumés)**.
* *Topology optimization for stationary fluid-structure interaction problems with turbulent flow*.
  [arXiv 2302.05288](https://arxiv.org/abs/2302.05288) **(R, titre)**.
* *Aerodynamics-driven topology optimization of compliant airfoils considering stability*, CMAME 2025 : compliance +
  traînée + **flambement** (la « stabilité » de ce titre est structurelle), RANS.
  [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0045782525007261) **(R)**
* Adjoint lattice Boltzmann multi-GPU pour la TO (Łaniewski-Wołłk & Rokicki 2015).
  [arXiv 1501.04741](https://arxiv.org/abs/1501.04741) **(R)** ; Pingen, Evgrafov, Maute (2007), LBM en TO, SMO **(M)**.
* TOPress3D (2024/2025) : charges de pression dépendant de la forme en 3D (Darcy, pas Brinkman).
  [arXiv 2405.07733](https://arxiv.org/abs/2405.07733) **(R)**
* STORX (2026) : cadre MATLAB forme + topologie, exemple de soufflerie (traînée, portance).
  [arXiv 2606.17291](https://arxiv.org/abs/2606.17291) **(R)**

### 2.2 Méthodes de fabrication additive dans l'optimisation

* Garaigordobil et al. (2018), surplomb par détection de contours.
  [ADDI](https://addi.ehu.es/handle/10810/32115) **(R)**
* Longueur minimale + surplomb + orientation (2022). [arXiv 2204.07333](https://arxiv.org/abs/2204.07333) **(R)** :
  note qu'une contrainte de pente seule peut laisser passer des pièces non auto-portantes.
* Choix libre entre auto-portant et « porté par le remplissage », CMAME 2024.
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0045782524000446) **(R)** : proche de B6/D2.
* Auto-support + découpe en couches courbes multi-axes, CMAME 2025.
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0045782525001136) **(R)**
* Co-optimisation neuronale topologie + couches + orientation des trajectoires (composites fibrés, 2025).
  [arXiv 2505.03779](https://arxiv.org/abs/2505.03779) **(R)**
* Zhang et al. (2022), *S³-Slicer*, SIGGRAPH Asia (prix du meilleur papier) ; *Neural Slicer* (2024)
  [arXiv 2404.15061](https://arxiv.org/abs/2404.15061) **(R)**.
* Wang, Lazarov, Sigmund (2011), projections et formulation robuste (érodé/dilaté), SMO **(M)** : D5.
* **Ponts** : aucune formulation de TO trouvée qui autorise explicitement une portée horizontale entre deux appuis.
  À confirmer par une recherche bibliographique plus poussée (Scopus/Google Scholar) avant de le revendiquer.

### 2.3 Représentations, points de départ, apprentissage

* Papadopoulos, Farrell, Surowiec (2021), *Computing multiple solutions of topology optimization problems* (déflation).
  [arXiv 2004.11797](https://arxiv.org/abs/2004.11797) **(R)** : trouver **plusieurs** optimums locaux distincts, utile
  pour B1.
* Benchmark de la continuation sur 375 problèmes : la continuation trouve de meilleurs designs, au prix d'itérations
  ([DTU, continuation automatique](https://orbit.dtu.dk/en/publications/automatic-penalty-continuation-in-structural-topology-optimizatio/)) **(R)**.
* TOuNN (Chandrasekhar & Suresh 2020) et Fourier-TOuNN ([arXiv 2109.01861](https://arxiv.org/abs/2109.01861)) **(R)** ;
  NTopo (Zehnder et al. 2021, [arXiv 2102.10782](https://arxiv.org/abs/2102.10782)) **(R)** : champs neuronaux.
* NITO (TMLR 2025, [arXiv 2402.05073](https://arxiv.org/abs/2402.05073)) et **OAT**, *Optimize Any Topology* (NeurIPS
  2025, [arXiv 2510.23667](https://arxiv.org/abs/2510.23667)) **(R)** : modèle de fondation (diffusion latente + champ
  neuronal) entraîné sur 2,2 M de structures optimisées ; 2D, compliance seule. Intéressant comme **démarrage à chaud**
  (B7) ou comme référence de ce que le ML sait faire, pas comme solveur pour notre physique.
* TreeTOp (2024), TO par arbres CSG : [arXiv 2409.02300](https://arxiv.org/abs/2409.02300) **(R)** ; géométrie
  directement exportable en CAO.
* Projection géométrique de barres : Smith & Norato (2020), code MATLAB 2D/3D
  ([Springer](https://link.springer.com/article/10.1007/s00158-020-02552-0)) **(R)** ; contraintes d'angle entre barres
  (Smith & Norato, SAMPE 2019) ; Guo, Zhang, Zhong (2014), *Moving Morphable Components* **(M)** : B4.
* De-homogénéisation : Groen & Sigmund (2018) **(M)** ; version évolutionnaire pour remplissage solide-poreux (2024)
  [arXiv 2412.19154](https://arxiv.org/abs/2412.19154) **(R)**.

### 2.4 Résistance, fréquences propres

* Le, Norato, Bruns, Ha, Tortorelli (2010), contraintes de contrainte en continu, SMO **(M)** ;
  Giraldo-Londoño & Paulino (2021), *PolyStress* (lagrangien augmenté), SMO **(M)**.
* Du & Olhoff (2007), valeurs propres simples et multiples, écarts de fréquence, SMO **(M)** ; Pedersen (2000), modes
  parasites dans les zones de faible densité **(M)**.
* Un papier IMAV 2018 recommande de placer les premiers modes du cadre loin des ~150 Hz moteurs (drone plus gros que
  le nôtre). [PDF](https://www.imavs.org/papers/2018/7.pdf) **(R)**

### 2.5 Spécifique drones

* Aéro bras/hélice : écoulement moteur-hélice sur le bras, expérience + CFD, *Drones* 7(3):199 (2023)
  [doi](https://doi.org/10.3390/drones7030199) **(R)** ; thèse METU : en stationnaire, bras profilé (Eppler) +4,9 %
  d'efficacité d'hélice vs bras cylindrique, +21,6 % vs bras carré
  ([PDF](https://etd.lib.metu.edu.tr/upload/12624016/index.pdf)) **(R)** ; blocage par une structure proche : −2 à
  −8 % de poussée selon la configuration (résumé de recherche, source primaire à identifier) **(R)**.
* Co-conception morphologie + contrôleur par grammaire de graphes et simulateur différentiable (IROS 2022).
  [PDF](https://cdfg.mit.edu/assets/files/Automatic_Co-Design_of_Aerial_Robots_Using_a_Graph_Grammar.pdf) **(R)**
* TO de cadres de quadricoptère « classiques » (SIMP sous poussée et couple, aéro vérifiée après coup) :
  [U.Porto J. Eng. 2026](https://journalengineering.fe.up.pt/index.php/upjeng/article/view/2183-6493_0012-001_003509),
  [ScienceDirect 2026, IA générative](https://www.sciencedirect.com/science/article/pii/S259012302600784X),
  [MDPI unibody](https://www.mdpi.com/2411-9660/6/1/8) **(R)**. Utiles comme points de comparaison, pas comme méthode.

### 2.6 Calcul haute performance, substituts neuronaux

* Aage, Andreassen, Lazarov, Sigmund (2017), *Giga-voxel computational morphogenesis*, Nature **(M)**.
* Träff, Rydahl, Karlsson, Sigmund, Aage (2023), *Simple and efficient GPU accelerated topology optimisation: codes
  and applications*, CMAME 410 : 65,5 M d'éléments en ~2 h sur un GPU. [hgpu](https://hgpu.org/?p=28174) **(R)**
* AB-UPT (TMLR 2025), substitut neuronal de CFD automobile, maillages arbitraires.
  [arXiv 2502.09692](https://arxiv.org/abs/2502.09692) **(R)** : perspective si l'on veut une aéro 3D rapide un jour
  (demande une base de CFD).

## 3. Codes open source

| Code | Langage | Ce qu'il fait | Usage pour nous |
|---|---|---|---|
| [TOPflow](https://github.com/sdu-multiphysics/topflow) | MATLAB/Octave | TO fluide 2D Navier–Stokes + Brinkman (Alexandersen) | **valider** `flow/brinkman.py` cas par cas |
| [pyMOTO](https://pymoto.readthedocs.io) ([GitHub](https://github.com/aatmdelissen/pyMOTO)), licence MIT | Python | TO modulaire, sensibilités par module, 3D, **dynamique**, contraintes de contrainte, multigrille, MMA/GCMMA | architecture très proche de la nôtre ; lire leurs modules valeurs propres et contraintes avant d'écrire les nôtres |
| [TopOpt_in_PETSc](https://github.com/topopt/TopOpt_in_PETSc) (+ wrapper Python `topoptlib`) | C++/PETSc | TO structurelle massivement parallèle (130 M d'éléments) | référence multigrille ; comparaison de nos résultats raideur |
| Codes GPU de Träff et al. (2023) | C/OpenMP, Futhark | TO linéaire et non linéaire sur GPU | si un GPU est disponible (dépôt à localiser via l'article) |
| [JAX-FEM](https://arxiv.org/abs/2212.00964) (dépôt de développement `tianjuxue/jax-am`), GPLv3 | Python/JAX | EF 3D différentiables sur GPU, non linéaire | alternative si on quitte PyTorch ; attention à la licence GPL |
| [XLB](https://github.com/Autodesk/XLB) (Autodesk) | Python (JAX, Warp) | lattice Boltzmann **différentiable**, multi-GPU, milliards de mailles | candidat pour C3 (aéro 3D différentiable) et C4 |
| OpenFOAM adjoint (`adjointOptimisationFoam`, NTUA) | C++ | adjoint RANS Spalart–Allmaras ; cadre de TO par porosité au-dessus ([présentation](https://www.esi-group.com/sites/default/files/resource/other/1671/8th_OpenFOAM_Conference_NTUA_Vaggelis.pdf)) | C4 (vérification RANS des finaux) |
| neural-structural-optimization (Google, Hoyer 2019) ; [TOuNN](https://github.com/UW-ERSL/TOuNN) | Python (JAX / PyTorch) | reparamétrisation neuronale | B5 (dépôt Google non vérifié pendant la recherche) |
| [NITO-3D](https://github.com/Lyleregenwetter/NITO-3D), [OptimizeAnyTopology](https://github.com/ahnobari/OptimizeAnyTopology) | Python | champs neuronaux, diffusion ; jeux de données OpenTO | B7, lecture |
| [GGP for Additive Manufacturing](https://github.com/topggp/GGP-for-Additive-Manufacturing) | MATLAB | projection géométrique généralisée + fabrication additive | B4 |
| S3_DeformFDM (`zhangty019`, d'après un agrégateur, non vérifié) | C++ | S³-Slicer, couches courbes | D6, lecture |
| [Computational Multicopter Design](https://copter.csail.mit.edu/) | – | conception paramétrique de multirotors + contrôleur | fiche drone, critères de vol |

Non retenus pour l'instant : FSI-Topology-Optimisation-Portable (cité par un dérivé Julia, introuvable), PeTTO
(GPU pseudo-transitoire, [arXiv 2509.06971](https://arxiv.org/abs/2509.06971), à surveiller).

## 4. Ce que la littérature ne fait pas (notre créneau)

1. **Aucun travail trouvé qui couple l'optimisation topologique d'un châssis au souffle des rotors** (recherches du
   2026-10-21 et du 2026-10-25). Les papiers de cadres de drone font de la SIMP structurelle puis vérifient l'aéro
   après coup.
2. **Pas d'objectif de vol** (autonomie, masse équivalente) dans la TO continue de drones ; le plus proche, Du et al.
   2016, est paramétrique (assemblage de pièces), pas topologique.
3. La TO couplée fluide-structure reste surtout **2D stationnaire** (Lundgaard 2018 le dit) ; le 3D turbulent arrive
   à peine (2026).
4. Les contraintes de fabrication imposent une **pente**, presque jamais une **portée de pont**.

À dire honnêtement en présentant : ces « absences » viennent de recherches web en une journée, pas d'une revue
systématique. Une recherche Scopus/Google Scholar ciblée (mots-clés : *rotor downwash*, *propeller wake*, *topology
optimization*, *multirotor frame*) est à faire avant toute revendication écrite.
