# Leçons : plan du cours

Ce dossier contient une suite de notebooks pour **comprendre les outils** du projet (discrétisation, éléments finis, différentiation automatique, optimisation topologique, filtres, anisotropie, écoulement potentiel), pas pour reproduire un article de recherche.
Chaque leçon traite un **cas simple et analogue** (une barre, une poutre console, la poutre MBB, un cylindre dans un écoulement…) dont la solution est connue, et sert à valider l'outil avant qu'on l'utilise sur le bras de drone.

**Règles du cours**
- Chaque leçon est lisible sans prérequis autre que les leçons précédentes, rédigée en français, avec ses figures et sorties **déjà exécutées**.
- Chaque résultat est **vérifié** contre une solution analytique, une référence indépendante ou un test de convergence (jamais « ça a l'air juste »).
- Une leçon est d'abord **autonome** (dans le notebook). Le code n'est promu dans `src/aeroptimizer` que lorsqu'il est utile au projet et validé ; la colonne « Code `src` » ci-dessous indique les modules concernés. À ce jour sont migrés : l'élément Q1 2D/3D (`fem/elements.py`), la loi de comportement orthotrope (`materials.py`) et la connectivité (`grid.py::Connectivity`) (leçon 4) ; la factorisation réutilisable, le gradient conjugué sans matrice et le préconditionneur de Jacobi (`fem/solvers.py`, leçon 5) ; l'opérateur sans matrice `StructuredFEOperator` et le branchement des solveurs sur cet opérateur (`fem/operators.py`, leçon 6) ; `solve_spd`, la seule dérivée écrite à la main du projet (`fem/adjoint.py`, leçon 7) ; la pénalisation SIMP, la compliance, la mise à jour OC et les conditions aux limites (appuis, charges, cas nommés `mbb` et `cantilever`) (`topopt/density.py`, `objectives/mechanical.py`, `optim/oc.py`, `fem/bc.py`, leçon 8) ; le filtre de densité, la projection de Heaviside et `DensityPipeline`, le calendrier de continuation, la boucle `run` et `ComplianceProblem` (`topopt/density.py`, `optim/continuation.py`, `optim/loop.py`, `problem.py`, leçon 9) ; MMA (`optim/mma.py`, leçon 10) ; le filtre de surplomb (`topopt/overhang.py`, leçon 12) ; l'extraction de la surface, l'export STL et le contrôle d'épaisseur (`io/mesh.py`, leçon 13) ; l'élément de Laplace, l'écoulement potentiel pénalisé `PotentialFlow` et son profil d'entrée de rotor (`fem/elements.py`, `flow/potential.py`, leçon 14) ; la force de pression sur la grille diffuse, l'indicateur de téléchargement $D$, la pression côté fluide et l'indicateur de blocage (`flow/aero.py`, `objectives/aerodynamic.py`, leçon 15) ; le coût composite à références figées, la contrainte de volume sur zones passives, le plancher de culot et le problème couplé de bout en bout `CoupledProblem` (`objectives/composite.py`, `objectives/constraints.py`, `flow/aero.py`, `problem.py`, leçon 17) ; la rotation du matériau, le critère de Hill et la rigidité `C0` passée à `ComplianceProblem` (`materials.py`, `problem.py`, leçon 11), avec leurs tests.
- Le plan est un plan de travail : l'ordre et le contenu peuvent changer au fil des lectures.

## Vue d'ensemble

| # | Leçon | Prérequis | Code `src` visé | Statut |
|---|---|---|---|---|
| **A. Fondations** | | | | |
| 1 | [Découper un objet en boîtes : la discrétisation](01_discretisation.ipynb) | | `grid.py` | fait |
| 2 | [La méthode des éléments finis en 1D](02_fem_1d.ipynb) | 1 | | fait |
| 3 | [L'élasticité 2D : déformation, contrainte, élément Q1](03_elasticite_2d.ipynb) | 1, 2 | | fait |
| **B. Passer à l'échelle** | | | | |
| 4 | [Passer en 3D : l'élément hexaédrique](04_elements_3d.ipynb) | 3 | `fem/elements.py`, `materials.py`, `grid.py` | fait |
| 5 | [Résoudre $\mathbf K\mathbf u=\mathbf f$ : direct ou itératif](05_resoudre_Ku_f.ipynb) | 3, 4 | `fem/solvers.py` | fait |
| 6 | [L'opérateur sans matrice (matrix-free)](06_operateur_sans_matrice.ipynb) | 4, 5 | `fem/operators.py`, `fem/solvers.py` | fait |
| **C. Différentiation et optimisation topologique** | | | | |
| 7 | [Dériver à travers un solveur : l'état adjoint](07_etat_adjoint.ipynb) | 5, 6 | `fem/adjoint.py` | fait |
| 8 | [SIMP et compliance : un mini top88](08_simp_mini_top88.ipynb) | 7 | `topopt/density.py`, `optim/oc.py`, `objectives/mechanical.py`, `fem/bc.py` | fait |
| 9 | [Filtres, projection et continuation](09_filtres_projection_continuation.ipynb) | 8 | `topopt/density.py`, `optim/continuation.py`, `optim/loop.py`, `problem.py` | fait |
| 10 | [Optimiseurs sous contraintes : OC, MMA](10_optimiseurs_oc_mma.ipynb) | 8, 9 | `optim/mma.py` | fait |
| **D. Matériau et fabrication FDM** | | | | |
| 11 | [Anisotropie : le matériau orthotrope](11_anisotropie_orthotrope.ipynb) | 3, 8 | `materials.py`, `problem.py` | fait |
| 12 | [Surplomb et auto-support](12_surplomb_auto_support.ipynb) | 9, 10 | `topopt/overhang.py`, `topopt/density.py`, `problem.py` | fait |
| 13 | [De la densité à une pièce imprimable](13_piece_imprimable.ipynb) | 9, 12 | `io/mesh.py` | fait |
| **E. Fluide et couplage** | | | | |
| 14 | [Écoulement potentiel : l'équation de Laplace](14_ecoulement_potentiel.ipynb) | 3, 6, 7 | `fem/elements.py`, `flow/potential.py` | fait |
| 15 | [Forces fluides et paradoxe de d'Alembert](15_forces_dalembert.ipynb) | 14 | `flow/aero.py`, `objectives/aerodynamic.py` | fait |
| 16 | *(optionnel)* Pénalisation de Brinkman et équations de Stokes | 14 | | à voir |
| 17 | [Coupler deux objectifs : compliance et aérodynamique](17_couplage_compliance_aero.ipynb) | 10, 14, 15 | `objectives/composite.py`, `objectives/constraints.py`, `problem.py`, `flow/aero.py` | fait |

## A. Fondations

**[1. Découper un objet en boîtes : la discrétisation](01_discretisation.ipynb).** Un objet continu devient un tableau fini : une densité par boîte (élément) et un déplacement par coin (nœud), numérotés de façon régulière. On apprend la notation du projet (axe $z$ = sens d'impression, numérotation en ordre C, fonctions de forme Q1) et on mesure comment le nombre d'inconnues explose en 3D.
*Cas analogue : un bras avec un trou, pixelisé.*

**[2. La méthode des éléments finis en 1D](02_fem_1d.ipynb).** De « une barre est un ressort » à $\mathbf K\mathbf u=\mathbf f$ : matrice élémentaire, assemblage (en boucle puis vectorisé), appuis, compliance comme double de l'énergie de déformation. On vérifie la convergence (ordre 2 en déplacement, ordre 1 en contrainte) et on goûte à l'optimisation avec un cas où la répartition optimale de matière se calcule à la main.
*Cas analogue : barre tirée, barre à charge répartie.*

**[3. L'élasticité 2D : déformation, contrainte, élément Q1](03_elasticite_2d.ipynb).** Déformation et contrainte en 2D, loi de Hooke par la matrice de souplesse (contraintes planes, déformations planes, premier contact avec l'orthotropie), puis construction de la matrice $\mathbf K_e^0$ de l'élément Q1 avec cinq vérifications (dont une comparaison à top88). La validation sur une console (flèche à 99,8 % de Timoshenko) révèle le verrouillage en cisaillement, et un trou dans la pièce montre que retirer de la matière coûte plus ou moins cher selon l'endroit.
*Cas analogue : poutre console, console trouée.*

## B. Passer à l'échelle

**[4. Passer en 3D : l'élément hexaédrique](04_elements_3d.ipynb).** L'hexaèdre Q1 (8 nœuds, 24 inconnues) généralise le quadrilatère avec les mêmes formules, grâce à la numérotation des coins par code binaire : un seul code vaut en 2D et en 3D. On valide la matrice $\mathbf K_e^0$ par cinq vérifications (6 modes rigides, patch test, 3D $\supset$ 2D en déformations planes, oracle à boucles indépendant), on compare une console 3D à Timoshenko (99 % à $n=10$) et on mesure ce que coûte la 3D : environ 4× plus de coefficients par ligne, temps de factorisation en $N^2$ contre $N^{1,5}$ en 2D, conditionnement en $h^{-2}$ dans les deux cas.
*Cas analogue : console 3D (type top3d, Liu & Tovar).*

**[5. Résoudre $\mathbf K\mathbf u=\mathbf f$ : direct ou itératif](05_resoudre_Ku_f.ipynb).** Un solveur direct (Cholesky ou LU creux) est robuste mais subit le remplissage en 3D, alors que le gradient conjugué n'a besoin que de produits matrice-vecteur. On mesure : le remplissage selon l'ordonnancement, la réutilisation de la factorisation (état et adjoint), le CG de la descente de gradient à l'algorithme du projet (itérations $\propto\sqrt\kappa\propto h^{-1}$), l'effet d'éléments quasi vides sur une structure en treillis (Jacobi stabilise les itérations quand $\varepsilon_s$ diminue), un multigrille minimal pour Poisson (6 itérations quel que soit $h$, **non validé pour l'élasticité à contraste**), le duel direct/CG en 3D et le plancher de précision de `float32` (`float64` sur CPU en référence).
*Cas analogue : équation de la chaleur 2D (Poisson), puis élasticité.*

**[6. L'opérateur sans matrice (matrix-free)](06_operateur_sans_matrice.ipynb).** On ne stocke jamais $\mathbf K$ : le produit $\mathbf K\mathbf v$ se calcule élément par élément (rassembler, petit produit dense avec $\mathbf K_e^0$, ajouter), grâce à la grille régulière qui permet de n'utiliser qu'**une** matrice élémentaire ; le piège de l'écriture `y[edof] += fe` (indices répétés) est montré. On le valide contre un assemblage par boucles (2D, 3D), par `gradcheck` et par différences finies, puis contre Timoshenko, on mesure mémoire et temps face à la matrice creuse (au budget 3D : quelques dizaines de Mo pour le produit contre 1,5 Go pour assembler), on le réutilise pour une inconnue par nœud (Poisson, ordre 2) et on branche `prepare(op, w)` des solveurs, avec une mesure honnête du gain du départ à chaud (modeste).
*Cas analogue : console 2D puis 3D, poutre MBB, Poisson.*

## C. Différentiation et optimisation topologique

**[7. Dériver à travers un solveur : l'état adjoint](07_etat_adjoint.ipynb).** Le coût dépend de la densité à travers la solution de $\widetilde{\mathbf K}\mathbf u=\mathbf f$ : comment le dériver pour des milliers d'éléments ? Après un rappel de la différentiation automatique (graphe, mode inverse, pièges), on dérive l'**état adjoint** (une résolution de plus avec la même factorisation), d'abord sur la barre 1D où tout est exact, puis on l'implémente comme fonction personnalisée d'autograd (`solve_spd`). On le valide par différences finies, `gradcheck` et la formule analytique de la compliance, et on le compare au déroulé des itérations du CG : aussi précis, mais il conserve la mémoire de toutes les itérations (de l'ordre de 45 Go au budget 3D par extrapolation) et il est impossible avec un solveur SciPy.
*Cas analogue : barre 1D, puis console et poutre MBB.*

**[8. SIMP et compliance : un mini top88](08_simp_mini_top88.ipynb).** La densité de chaque élément module sa rigidité, $s=\varepsilon_s+(1-\varepsilon_s)\rho^p$ ; la pénalisation $p>1$ rend les densités intermédiaires inefficaces et pousse vers du noir et blanc (deux barres en parallèle, où tout se calcule). On dérive le critère d'optimalité OC et on le teste sur un problème à solution analytique, puis on assemble la boucle complète (SIMP, adjoint, OC) et on la valide contre un top88 NumPy indépendant (compliances identiques à $10^{-9}$ sur 40 itérations) et par les conditions KKT. On observe le rôle de $p$ (le gris de $p=1$ est trompeur, le design à $p=3$ est un minimum local), puis les damiers numériquement trop raides et la dépendance au maillage qui motivent la leçon 9, et enfin le même code en 3D.
*Cas analogue : poutre MBB (Sigmund 2001, Andreassen et al. 2011).*

**[9. Filtres, projection et continuation](09_filtres_projection_continuation.ipynb).** Les damiers et la dépendance au maillage imposent une taille minimale : filtre de densité conique (convolution normalisée, validée contre un oracle dense et contre un top88 NumPy à $10^{-11}$ près), puis projection de Heaviside lisse pour retrouver des contours nets. On montre que raffiner le maillage à rayon physique fixé donne la même forme, sans damiers ; qu'un $eta$ élevé d'emblée fait échouer l'OC ; et que la continuation en $eta$ (1 à 16) donne un design net et bien meilleur, alors que la continuation en $p$ n'apporte rien sur ce cas (mesuré, pas supposé). On vérifie que toute la chaîne reste différentiable.
*Cas analogue : MBB, avec et sans filtre, à plusieurs maillages.*

**[10. Optimiseurs sous contraintes : OC, MMA](10_optimiseurs_oc_mma.ipynb).** OC est taillé pour « compliance + une contrainte de volume » (et échoue sans erreur dès que les signes des gradients changent) ; dès qu'il y a plusieurs contraintes ou un autre objectif, il faut un optimiseur général. On compare descente de gradient projetée, OC et MMA (approximations convexes séparables, asymptotes mobiles, sous-problème par le dual) sur des problèmes à solution connue (analytique, SLSQP), avec les conditions KKT comme critère indépendant, puis sur la MBB : MMA converge là où l'OC oscille. On montre pourquoi MMA doit être réinitialisé à chaque changement de la continuation, et que le critère d'arrêt ne vérifie pas la faisabilité.
*Cas analogue : problèmes quadratiques à contraintes, puis MBB et « volume minimal à compliance imposée ».*

## D. Matériau et fabrication FDM

**[11. Anisotropie : le matériau orthotrope](11_anisotropie_orthotrope.ipynb).** La rigidité d'un matériau orthotrope dépend de la direction. On construit $\mathbf S$ et $\mathbf C$ à partir des 9 constantes, on montre que les conditions « par paires » des manuels ne suffisent pas à garantir $\mathbf S\succ0$ (carte d'admissibilité), puis on **change de repère** : $\mathbf C'=\mathbf T\mathbf C\mathbf T^\top$, validé contre la rotation du tenseur d'ordre 4 et la formule classique du module hors axe à $10^{-12}$ près. Dans l'élément fini, une poutre console imprimée à plat ou debout a une flèche qui suit $E_z/E_p$ ; une poutre composite tournée de $0$ à $90^\circ$ suit le module hors axe à $7\,\%$ près. Sur la MBB, on mesure (avec le bruit des minima locaux) que l'**orientation d'impression** compte bien plus que l'adaptation de la forme : au rapport FDM l'adaptation ne rapporte rien de mesurable, pour un rapport $0{,}1$ elle gagne de $5$ à $12\,\%$. Enfin raideur et résistance sont deux choses différentes : critère de Hill (Tsai–Hill), récupération des contraintes validée par patch test, et un exemple où imprimer debout coûte $25\,\%$ de raideur mais $37\,\%$ de charge à la rupture. Les résistances sont des valeurs de remplacement.
*Cas analogue : poutre console de composite stratifié, poutre MBB.*

**[12. Surplomb et auto-support](12_surplomb_auto_support.ipynb).** Une pièce FDM s'imprime couche par couche : une cellule doit reposer sur de la matière dans la couche inférieure, avec un angle limité à 45°. On écrit la règle exacte (oracle NumPy à boucles), on la rend différentiable (max lisse par log-somme-exp, min lisse, récurrence sur les couches, stencil en croix en 3D qui garantit 45° dans toutes les directions là où le carré 3×3 donnerait 54,7°), et on la valide : majoration, gradient (`gradcheck`), cône de dépendance à 45°. On mesure que le biais du max lisse est payé à chaque couche (l'erreur suit $(n_z-1)\log n/\beta_s$) : la forme lisse n'est exacte qu'à $\beta_s$ très grand, mais sa binarisation est auto-supportante. Dans le pipeline, la forme optimale n'a plus aucune cellule sans support, pour un surcoût de compliance de 24 % (MBB) à 33 % (console) avec le calendrier par défaut, mais seulement 9 % (MBB) avec $\beta_s=200$ : une partie du coût est le biais du filtre. OC diverge avec ce filtre, MMA reste stable.
*Cas analogue : filtres de fabrication additive (type Langelaar), poutre MBB et console.*

**[13. De la densité à une pièce imprimable](13_piece_imprimable.ipynb).** Le résultat de l'optimisation est un champ de densités, pas une pièce : seuil à 0,5 (celui de la projection ; la compliance varie doucement avec le seuil, sauf artefact des nœuds chargés sans cellule), contour 2D par *marching squares* (16 cas, interpolation linéaire, selle tranchée par le centre, aire signée à moins de 1 %), surface 3D par *marching tetrahedra* (variante du marching cubes sans table de 256 cas, fermée par construction : Euler 2 pour une boule, 0 pour un tore), export STL binaire relu à l'identique. On contrôle l'étanchéité et l'orientation, la largeur minimale par ouverture morphologique (le rayon $R$ du filtre ne garantit aucune largeur), puis on recalcule la pièce extraite : compliance à 3 % de celle de l'optimiseur, volume à ±1,5 % du volume visé.
*Cas analogue : console optimisée, extrudée en pièce 3D, exportée en STL.*

## E. Fluide et couplage

**14. Écoulement potentiel : l'équation de Laplace** ([notebook](14_ecoulement_potentiel.ipynb)). Un écoulement incompressible et irrotationnel dérive d'un potentiel qui vérifie l'équation de Laplace : c'est le même problème éléments finis que l'élasticité, avec **une** inconnue par nœud sur la même grille. On en déduit vitesse et pression de Bernoulli, et on valide sur l'écoulement autour d'un cylindre, dont la solution est analytique. Le notebook sépare l'erreur due au **blocage** de la veine de l'erreur de maillage, montre pourquoi la sortie doit être imposée et le tampon d'entrée obligatoire, mesure l'effet de $\kappa_{\min}$ et de $q$ sur l'interface diffuse, et refait le calcul en 3D (sphère).
*Cas analogue : cylindre dans un écoulement uniforme.*

**15. Forces fluides et paradoxe de d'Alembert** ([notebook](15_forces_dalembert.ipynb)). Un écoulement potentiel stationnaire n'exerce aucune traînée (la pression intégrée est nulle) : on le constate numériquement (en distinguant l'annulation par symétrie de l'exactitude), puis on explique pourquoi (sillage, viscosité, décollement). On construit ensuite l'indicateur de forme du projet (pression de culot constante) et on le met à l'épreuve : décomposition exacte $D=\tfrac12\rho U^2A\,(c_b+\langle C_p\rangle_{\rm amont})$, biais d'interface et correction, ce que l'indicateur voit (et ne voit pas) de la forme, plancher de bruit. **Résultat à connaître** : le terme de culot ne dépend que de la largeur frontale, et $c_b$ doit dépasser $1/3$ pour qu'un cylindre ait $D>0$.
*Cas analogue : cylindre, plaque, œuf et coin : force nulle en potentiel, indicateur de forme comparé.*

**16. Pénalisation de Brinkman et équations de Stokes** *(optionnel)*. Pour obtenir une vraie traînée il faut de la viscosité : on voit la pénalisation de Brinkman (la matière est un milieu très résistant, le fluide un milieu libre) et les équations de Stokes sur la même grille, comme en optimisation topologique des fluides. Cette leçon est une ouverture et un plan B pour le modèle aérodynamique, non requise pour la v1.
*Cas analogue : écoulement de Stokes dans un canal avec obstacle.*

**17. Coupler deux objectifs : compliance et aérodynamique** ([notebook](17_couplage_compliance_aero.ipynb)). Combiner deux objectifs de natures différentes demande de les normaliser, de choisir des poids et de lire un front de Pareto. On assemble les leçons précédentes dans un petit problème 2D de bout en bout (`CoupledProblem` : une densité, une résolution mécanique et une résolution fluide, deux adjoints) et on vérifie le gradient du coût composite aux différences finies. Le notebook montre surtout trois choses mesurées : la référence de normalisation doit venir d'un design **convergé** ; l'indicateur $D$ **non protégé est exploité** par l'optimiseur (téléchargement négatif obtenu au prix de 9 à 20 fois la compliance), d'où un plancher de culot ; le front de Pareto est **bruité** (minima locaux : 8 points non dominés sur 18) et dit surtout que supprimer une membrure coûte cher en rigidité, dans une proportion qui dépend du $c_b$ non calibré.
*Cas analogue : compromis rigidité/indicateur de forme sur une console 2D.*

## Références pour aller plus loin

Citées de mémoire : **à vérifier avant tout usage dans un rapport.**

- Bendsøe & Sigmund, *Topology Optimization: Theory, Methods and Applications*, Springer, 2003.
- Sigmund, « A 99 line topology optimization code written in Matlab », *Structural and Multidisciplinary Optimization* 21, 2001.
- Andreassen et al., « Efficient topology optimization in MATLAB using 88 lines of code », *Struct. Multidisc. Optim.* 43, 2011.
- Liu & Tovar, « Three-dimensional topology optimization in 3D using an efficient Matlab code » (type top3d), *Struct. Multidisc. Optim.* 50, 2014.
- Díaz & Sigmund, « Checkerboard patterns in layout optimization », *Structural Optimization* 10, 1995.
- Hill, « A theory of the yielding and plastic flow of anisotropic metals », *Proc. R. Soc. Lond. A* 193, 1948 ; Tsai & Wu, « A general theory of strength for anisotropic materials », *J. Composite Materials* 5, 1971.
- Lorensen & Cline, « Marching cubes: a high resolution 3D surface construction algorithm », *SIGGRAPH* 1987 ; Gaynor & Guest, « Topology optimization considering overhang constraints », *Struct. Multidisc. Optim.* 54, 2016.
- Milne-Thomson, *Theoretical Hydrodynamics* (écoulement potentiel, paradoxe de d'Alembert) ; Batchelor, *An Introduction to Fluid Dynamics*, Cambridge, 1967 (valeurs de $C_d$ à vérifier avant usage).
- Svanberg, « The method of moving asymptotes: a new method for structural optimization », *Int. J. Numer. Methods Eng.* 24, 1987.
- Langelaar, « Topology optimization of 3D self-supporting structures for additive manufacturing », *Additive Manufacturing* 12, 2016.
- Borrvall & Petersson, « Topology optimization of fluids in Stokes flow », *Int. J. Numer. Methods Fluids* 41, 2003.
- Liens avec le projet : [FORMALISM.md](../../FORMALISM.md) (formulation exacte), [ARCHITECTURE.md](../../ARCHITECTURE.md) (feuille de route), [CONTRACTS.md](../../CONTRACTS.md) (tests).
