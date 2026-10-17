# FORMALISM.md — Formulation mathématique exacte

> Statut : **v0.1 — référence normative pour l'implémentation.**
> Tout symbole utilisé dans le code ou dans [CONTRACTS.md](CONTRACTS.md) est défini ici.
> Les choix de modélisation discutables sont isolés dans la section 9 (hypothèses et points ouverts) ;
> les cas où une valeur est un **paramètre à calibrer** sont signalés par ⚠.

---

## 0. Conventions et notations

| Symbole | Sens |
|---|---|
| $d\in\{2,3\}$ | dimension spatiale |
| axes | 2D : $(x,z)$ ; 3D : $(x,y,z)$. Indices d'axe $a=0,\dots,d-1$. **L'axe $z$ est toujours le dernier axe ($a=d-1$).** |
| $\mathbf e_z$ | direction de fabrication (build direction) **et** axe vertical ; le plateau d'impression est à $z=0$ |
| $\mathbf e_f=-\mathbf e_z$ | direction de l'écoulement (flux descendant des hélices) |
| $\mathbf n=(n_0,\dots,n_{d-1})$ | nombre d'éléments par axe ; $\mathbf h=(h_0,\dots,h_{d-1})$ taille d'élément |
| $N_e=\prod_a n_a$ | nombre d'éléments ; $N_n=\prod_a (n_a+1)$ nombre de nœuds |
| $N_u=d\,N_n$ | nombre de ddl mécaniques ; $N_\varphi=N_n$ nombre de ddl de potentiel |
| $x\in[0,1]^{N_e}$ | variables de conception (une par élément) |
| $\tilde\rho,\ \hat\rho,\ \bar\rho$ | densités filtrée, projetée, **imprimable** (après filtre de surplomb) |
| $\bar\rho_e\in[0,1]$ | 1 = matière, 0 = vide/fluide. **C'est $\bar\rho$ qui alimente la mécanique et le fluide.** |
| $p$ | exposant SIMP ; $q$ exposant de pénalisation fluide |
| $\Omega=\Omega_s\cup\Omega_f$ | $\Omega_s$ solide, $\Omega_f$ fluide, au sens de $\bar\rho$ |

**Pourquoi la direction de fabrication est le dernier axe.** Elle simplifie tout : les couches
d'impression sont les tranches `[..., k]` du tenseur de densité, l'axe faible de l'orthotropie est
$z$, l'axe de l'écoulement est $z$. Une conception dont l'axe d'impression diffère doit être **tournée
avant** d'être donnée au solveur (le solveur ne gère aucune rotation du repère matériau, voir §9, H4).

Les vecteurs de Voigt suivent l'ordre

$$
\boldsymbol\varepsilon_{3D}=(\varepsilon_{xx},\varepsilon_{yy},\varepsilon_{zz},\gamma_{yz},\gamma_{xz},\gamma_{xy})^\top,\qquad
\boldsymbol\varepsilon_{2D}=(\varepsilon_{xx},\varepsilon_{zz},\gamma_{xz})^\top ,
$$

avec cisaillements **d'ingénierie** $\gamma_{ij}=2\varepsilon_{ij}$ (donc $\boldsymbol\sigma=\mathbf C\boldsymbol\varepsilon$ avec
$C_{44}=G_{yz}$, etc., sans facteur 2).

---

## 1. Discrétisation

### 1.1 Grille et élément

Grille cartésienne uniforme **par axe** (pas $h_a$ constant sur chaque axe, éventuellement différent d'un axe à l'autre).
Élément unique **Q1** (bilinéaire en 2D, trilinéaire en 3D) pour les déplacements **et** pour le
potentiel fluide : les deux problèmes partagent la même grille, la même connectivité, les mêmes
fonctions de forme. La densité est **constante par élément** (champ $P_0$).

Élément de référence $\boldsymbol\xi\in[-1,1]^d$. Le nœud local $a\in\{0,\dots,2^d-1\}$ est repéré par ses bits
$(b_0,\dots,b_{d-1})$, **$b_0$ de poids fort** (ordre lexicographique, dernier axe le plus rapide), et

$$
N_a(\boldsymbol\xi)=\prod_{k=0}^{d-1}\frac{1+s_{a,k}\,\xi_k}{2},\qquad s_{a,k}=2b_k-1 .
$$

Le jacobien est constant : $J=\mathrm{diag}(h_k/2)$, $\det J=\prod_k (h_k/2)$ et $\nabla_{\mathbf x}N_a=J^{-1}\nabla_{\boldsymbol\xi}N_a$.

### 1.2 Intégration

Les intégrandes de $\mathbf K_e^0$ (mécanique) et $\mathbf K_{\varphi,e}^0$ (Laplace) sont de degré $\le 2$ par
variable sur un élément rectangulaire/parallélépipédique : la **quadrature de Gauss $2^d$ points est exacte**.

---

## 2. Chaîne de densités $x\mapsto\bar\rho$

$$
x\ \xrightarrow{\ \text{masques}\ }\ \rho\ \xrightarrow{\ \text{filtre densité}\ }\ \tilde\rho\ \xrightarrow{\ \text{projection}\ }\ \hat\rho\ \xrightarrow{\ \text{filtre surplomb}\ }\ \bar\rho
$$

### 2.1 Zones passives

Trois ensembles d'éléments : $\mathcal D$ (conception), $\mathcal S$ (solide passif : interface moteur, trous de vis, cadre de fixation), $\mathcal V$ (vide/fluide passif : tampon fluide autour du domaine).

$$
\rho_e=\begin{cases}x_e & e\in\mathcal D\\ 1 & e\in\mathcal S\\ 0 & e\in\mathcal V\end{cases}
$$

### 2.2 Filtre de densité (contrôle de taille minimale)

Filtre de convolution normalisé, rayon $R$ (en unités physiques), noyau conique

$$
w(\mathbf r)=\max\!\Big(0,\,1-\frac{\|\mathbf r\|}{R}\Big),\qquad
\tilde\rho=\frac{w*\rho}{w*\mathbf 1}\ \ (\text{division élément par élément}).
$$

La normalisation par $w*\mathbf 1$ traite les bords sans biais de densité. Le noyau est séparable en
distance (pas en produit), il est discrétisé une fois en un tenseur de taille $(2r_a+1)_a$ avec
$r_a=\lceil R/h_a\rceil-1$, et appliqué par `conv2d`/`conv3d`. Option future : filtre de Helmholtz $(-r^2\nabla^2+1)\tilde\rho=\rho$, $r=R/(2\sqrt3)$.

### 2.3 Projection de Heaviside lisse

$$
\hat\rho=\frac{\tanh(\beta\eta)+\tanh\!\big(\beta(\tilde\rho-\eta)\big)}{\tanh(\beta\eta)+\tanh\!\big(\beta(1-\eta)\big)},\qquad \eta=0.5,\ \ \beta\uparrow\ \text{par continuation}.
$$

Propriétés exigées : $\hat\rho(0)=0$, $\hat\rho(1)=1$, croissante, $\beta\to0$ : identité.

### 2.4 Filtre de surplomb (self-support FDM)

**Définition du critère.** Soit $\theta$ l'angle maximal admissible **mesuré par rapport à la verticale**
($\theta=45^\circ$ est la valeur FDM usuelle). Un voxel imprimé en couche $\ell\ge1$ est *supporté* s'il existe, dans la couche $\ell-1$,
un voxel imprimé situé à un décalage horizontal $\le h\tan\theta$.

Sur grille cartésienne avec un voxel de décalage, $\tan\theta=1$ : **seul $\theta=45^\circ$ est représentable
exactement** (des angles $\arctan k,\ k\in\mathbb N^*$ avec un stencil de rayon $k$ ; des angles plus raides
exigent des cellules anisotropes ou un autre filtre, hors périmètre v1).

**Stencil.** Pour $\mathcal N(c)$ l'ensemble des cellules de la couche inférieure qui peuvent supporter la cellule $c$ :

* 2D : $\{c-1,\,c,\,c+1\}$ (3 points) ;
* 3D : **croix à 5 points** $\{c,\,c\pm\mathbf e_x,\,c\pm\mathbf e_y\}$, *pas* le carré $3\times3$.

Justification : avec le carré $3\times3$, une frontière diagonale avance de $\sqrt2\,h$ par couche
($\theta\approx54.7^\circ>45^\circ$ : surplomb non conforme) ; avec la croix, elle avance de $h/\sqrt2$
($\theta\approx35.3^\circ$), et $45^\circ$ le long des axes : la contrainte $\theta\le45^\circ$ est donc
garantie **dans toutes les directions horizontales** (de façon conservatrice sur les diagonales).

**Opérateurs lisses.** Sur des valeurs $y_k\in[0,1]$ :

$$
\mathrm{smax}_\beta(y_1..y_n)=\frac1{\beta_s}\log\sum_k e^{\beta_s y_k}-\frac{\log n}{\beta_s},\qquad
\mathrm{smin}_\epsilon(a,b)=\tfrac12\Big(a+b-\sqrt{(a-b)^2+\epsilon}+\sqrt\epsilon\Big).
$$

($\beta_s\approx 50$–$100$, $\epsilon\approx10^{-4}$ ; le biais de $\mathrm{smax}$ est borné par $\log n/\beta_s$.
Alternative de la littérature : $P$-norme stabilisée de Langelaar — la forme log-sum-exp est retenue car son
gradient est partout fini, y compris quand toutes les entrées valent 0.)

**Récurrence par couches** ($\ell=0$ : plateau d'impression, toujours supporté) :

$$
\xi_0=\hat\rho_0,\qquad
\Xi_\ell(c)=\mathrm{smax}_{c'\in\mathcal N(c)}\ \xi_{\ell-1}(c'),\qquad
\xi_\ell(c)=\mathrm{smin}_\epsilon\big(\hat\rho_\ell(c),\,\Xi_\ell(c)\big),\qquad
\bar\rho:=\xi .
$$

Conséquences : $\bar\rho\le\hat\rho$ (le filtre ne crée jamais de matière, il en retire), $\bar\rho$ est
auto-supportant, et la dépendance de chaque couche sur **toutes** les couches inférieures est portée par la
récurrence : la sensibilité complète est obtenue en rétropropageant à travers la boucle de couches
(ce que fait Autograd ; voir ARCHITECTURE §5.5 pour le coût).

**Cas où $\mathcal S$ (solide passif) est au-dessus du vide :** le filtre le retirerait ; les éléments
de $\mathcal S$ sont donc **réimposés après** le filtre (masque de ré-application, §2.5), et un solide passif non supporté
est une erreur de configuration détectée à l'initialisation (CONTRACTS C-OVH-06).

### 2.5 Ré-application des masques

$\bar\rho\leftarrow\bar\rho$ avec $\bar\rho_e=1$ sur $\mathcal S$ et $\bar\rho_e=0$ sur $\mathcal V$, après surplomb.

---

## 3. Mécanique statique

### 3.1 Loi de comportement orthotrope (repère matériau = repère de la grille)

Convention de Poisson : $\nu_{ij}=-\varepsilon_j/\varepsilon_i$ sous traction uniaxiale $\sigma_i$.
Symétrie : $\nu_{ij}/E_i=\nu_{ji}/E_j$. Axes matériau $1,2,3\equiv x,y,z$.

Paramètres indépendants (9) : $E_1,E_2,E_3,\ G_{23},G_{13},G_{12},\ \nu_{12},\nu_{13},\nu_{23}$.

$$
\mathbf S=\begin{pmatrix}
\frac1{E_1} & -\frac{\nu_{12}}{E_1} & -\frac{\nu_{13}}{E_1} & 0&0&0\\[2pt]
-\frac{\nu_{12}}{E_1} & \frac1{E_2} & -\frac{\nu_{23}}{E_2} &0&0&0\\[2pt]
-\frac{\nu_{13}}{E_1} & -\frac{\nu_{23}}{E_2} & \frac1{E_3} &0&0&0\\[2pt]
0&0&0&\frac1{G_{23}}&0&0\\ 0&0&0&0&\frac1{G_{13}}&0\\ 0&0&0&0&0&\frac1{G_{12}}
\end{pmatrix},\qquad \mathbf C_0=\mathbf S^{-1}.
$$

**Condition d'admissibilité** : $\mathbf S\succ0$ (équivalent à $\mathbf C_0\succ0$). Elle est vérifiée
numériquement par factorisation de Cholesky à la construction du matériau ; elle implique en particulier
$|\nu_{ij}|<\sqrt{E_i/E_j}$ et $1-\nu_{12}\nu_{21}-\nu_{23}\nu_{32}-\nu_{31}\nu_{13}-2\nu_{21}\nu_{32}\nu_{13}>0$.

**Cas FDM : isotropie transverse d'axe $z$** (cas particulier de l'orthotrope, 5 constantes) :

$$
E_1=E_2=E_p,\quad E_3=E_z,\quad \nu_{12}=\nu_p,\quad \nu_{13}=\nu_{23}=\nu_{pz},\quad G_{12}=\frac{E_p}{2(1+\nu_p)},\quad G_{13}=G_{23}=G_{z}.
$$

Ordres de grandeur pour PLA/PETG non chargés ⚠ (**à calibrer par essais de traction sur éprouvettes
imprimées dans votre configuration, orientations XY et Z**) : $E_z/E_p\approx0.6$–$0.9$ ; la
dégradation de **résistance** inter-couches est en général beaucoup plus forte que celle de la **raideur** (voir §9, H3).

**Réductions 2D** (plan $(x,z)$, $y$ hors plan) :

* *Contraintes planes* (bras mince d'épaisseur $t$ selon $y$) :
  $\mathbf C_0^{2D}=\big(\mathbf S_{\{xx,zz,xz\}}\big)^{-1}$, où l'on extrait de $\mathbf S$ les lignes/colonnes $\{1,3,5\}$ **avant** inversion ;
* *Déformations planes* : $\mathbf C_0^{2D}=\mathbf C_{0,\{xx,zz,xz\}}$, extraction **après** inversion.

Le défaut est *contraintes planes* ; l'épaisseur $t$ multiplie $\mathbf K_e^0$.

### 3.2 Pénalisation SIMP

$$
\mathbf C(\bar\rho_e)=s(\bar\rho_e)\,\mathbf C_0,\qquad s(\bar\rho)=\varepsilon_s+(1-\varepsilon_s)\,\bar\rho^{\,p},\qquad \varepsilon_s=10^{-9},\ p=3.
$$

Un **même** facteur scalaire s'applique à toutes les composantes de $\mathbf C_0$ : l'anisotropie est
celle du matériau plein, l'interpolation ne la modifie pas (pas de DMO).

### 3.3 Matrice de rigidité élémentaire et assemblage

Comme la grille est uniforme par axe, **un seul** $\mathbf K_e^0$ sert pour tous les éléments :

$$
\mathbf K_e^0=t\sum_{g=1}^{2^d}w_g\,\det J\;\mathbf B_g^\top\mathbf C_0\,\mathbf B_g\ \in\mathbb R^{(d\,2^d)\times(d\,2^d)},
$$

avec $\mathbf B_g$ la matrice déformation–déplacement au point de Gauss $g$ (colonne du nœud $a$ :
2D $\begin{psmallmatrix}\partial_xN_a&0\\0&\partial_zN_a\\\partial_zN_a&\partial_xN_a\end{psmallmatrix}$, 3D analogue 6×3 avec l'ordre de Voigt du §0), et

$$
\mathbf K(\bar\rho)=\sum_{e=1}^{N_e}\mathbf A_e^\top\,s(\bar\rho_e)\,\mathbf K_e^0\,\mathbf A_e ,
$$

$\mathbf A_e$ = opérateur de localisation (gather) élément → ddl globaux. Numérotation globale :
nœud $n=\mathrm{ravel}(i_0,\dots,i_{d-1})$ en ordre C (dernier axe le plus rapide), ddl $=d\cdot n+c$ ($c$ = composante).

### 3.4 Conditions aux limites et système

$\mathcal F\subset\{1..N_u\}$ ddl libres, $\mathcal C$ ddl bloqués (déplacement imposé nul). Avec le masque $\mathbf P=\mathrm{diag}(\mathbb 1_{\mathcal F})$ :

$$
\widetilde{\mathbf K}=\mathbf P\mathbf K\mathbf P+(\mathbf I-\mathbf P),\qquad
\widetilde{\mathbf K}\,\mathbf u=\mathbf P\mathbf f .
$$

$\widetilde{\mathbf K}$ est SPD, de taille $N_u$ (pas de renumérotation), et vérifie $\mathbf u_{\mathcal C}=0$.
Cette forme « identité sur les ddl bloqués » est celle utilisée partout : elle est compatible avec un produit
matrice–vecteur *sans matrice* (matrix-free) et avec un adjoint symétrique.

### 3.5 Chargement du drone

Cas statique quasi-stationnaire (vol stationnaire, éventuellement facteur de charge $n_g$) :

* poussée moteur $T_i=n_g\,T_{\rm hov}$ appliquée verticalement ($+\mathbf e_z$) répartie sur la zone passive $\mathcal S_{\rm moteur}$ ;
* en 3D, couple de réaction $Q_i=k_Q T_i$ d'axe $z$, introduit en champ de forces tangentielles sur la zone d'attache moteur
  (pas de couple concentré : le Q1 n'a pas de ddl de rotation) ;
* masse du châssis/charge utile : forces nodales sur la zone passive d'attache, ou, plus simplement, **encastrement** de la zone
  de raccord avec le châssis (modèle de bras isolé) ;
* poids propre : hors v1 (force dépendante de $\bar\rho$, voir CONTRACTS C-SOLVE-04 : la différentiation par rapport à $\mathbf f$ est déjà prévue).

### 3.6 Compliance et sensibilités

$$
\mathcal C(\bar\rho)=\mathbf f^\top\mathbf u(\bar\rho)=\mathbf u^\top\mathbf K\mathbf u .
$$

Pour une fonctionnelle générale $J(\mathbf u,\bar\rho)$ et $\mathbf K$ symétrique, **état adjoint** :

$$
\widetilde{\mathbf K}\,\boldsymbol\lambda=\Big(\frac{\partial J}{\partial\mathbf u}\Big)^\top,\qquad
\frac{dJ}{d\bar\rho_e}=\frac{\partial J}{\partial\bar\rho_e}\Big|_{\mathbf u}-\ s'(\bar\rho_e)\ \boldsymbol\lambda_e^\top\mathbf K_e^0\,\mathbf u_e
+\ \boldsymbol\lambda^\top\frac{\partial\mathbf f}{\partial\bar\rho_e}.
$$

Cas particulier de la compliance : $\boldsymbol\lambda=\mathbf u$ et
$\dfrac{d\mathcal C}{d\bar\rho_e}=-p(1-\varepsilon_s)\,\bar\rho_e^{\,p-1}\,\mathbf u_e^\top\mathbf K_e^0\mathbf u_e$
(**formule analytique conservée comme oracle de test**, pas comme code de production).

---

## 4. Aérodynamique : écoulement potentiel sur la même grille

### 4.1 Modèle

Écoulement incompressible, irrotationnel, stationnaire : $\mathbf v=\nabla\varphi$, $\nabla\!\cdot\mathbf v=0$ dans $\Omega_f$,
$\partial_n\varphi=0$ sur la paroi $\partial\Omega_s$. On le règle en un **seul problème sur tout le domaine** en pénalisant
le solide par une conductivité :

$$
-\nabla\!\cdot\big(\kappa(\bar\rho)\,\nabla\varphi\big)=0\ \text{ dans }\Omega,\qquad
\kappa(\bar\rho)=\kappa_{\min}+(1-\kappa_{\min})(1-\bar\rho)^{q},\quad \kappa_{\min}=10^{-6},\ q=3\ ⚠ .
$$

Quand $\bar\rho\to1$, $\kappa\to\kappa_{\min}$ : flux nul à travers le solide, ce qui reproduit la condition de
non-pénétration $\partial_n\varphi=0$ (Neumann) sur l'interface — c'est l'analogue, pour Laplace, de la pénalisation de Brinkman/Borrvall–Petersson.

### 4.2 Conditions aux limites (rotor)

Domaine $\Omega$ = boîte contenant le domaine de conception **et un tampon fluide passif ($\mathcal V$) d'au moins
$n_{\rm pad}\ge 4$ éléments** sur la face d'entrée (impératif : un solide au contact de la face d'entrée créerait des gradients $\sim1/\kappa_{\min}$).

* Entrée $\Gamma_{\rm in}$ ($z=z_{\max}$) : flux imposé
  $\kappa\,\partial_n\varphi=g_{\rm in}(x,y)=-U_{\rm in}(x,y)$ (vitesse entrante $U_{\rm in}\ge0$) ;
* Sortie $\Gamma_{\rm out}$ ($z=0$) : $\varphi=0$ (Dirichlet) ;
* Parois latérales : $\partial_n\varphi=0$ (glissement).

Profil d'entrée : $U_{\rm in}=k_w U_h$ sur l'empreinte du disque rotor, $U_{\rm in}=U_\infty$ ailleurs (défaut 0), avec
la vitesse induite en vol stationnaire (théorie de la quantité de mouvement)

$$
U_h=\sqrt{\frac{T}{2\rho_a A_{\rm rotor}}},\qquad k_w\in[1,2]\ ⚠\ (\text{contraction du sillage : } k_w\to2 \text{ loin sous le rotor}),
$$

$\rho_a=1.225\ \mathrm{kg\,m^{-3}}$. La vitesse de référence est $U_{\rm ref}=k_wU_h$.

### 4.3 Forme faible et matrice de diffusion

Trouver $\varphi\in V$, $\varphi|_{\Gamma_{\rm out}}=0$ :

$$
\int_\Omega\kappa(\bar\rho)\,\nabla\varphi\!\cdot\!\nabla w\,d\Omega=\int_{\Gamma_{\rm in}}g_{\rm in}\,w\,d\Gamma\qquad\forall w\in V_0 .
$$

Discrétisation Q1 (mêmes nœuds et éléments que §1) :

$$
\mathbf K_{\varphi}(\bar\rho)=\sum_e\kappa(\bar\rho_e)\,\mathbf A_{\varphi,e}^\top\,\mathbf K_{\varphi,e}^0\,\mathbf A_{\varphi,e},\qquad
\mathbf K_{\varphi,e}^0=\sum_g w_g\det J\ (\nabla N)_g^\top(\nabla N)_g\in\mathbb R^{2^d\times2^d}.
$$

Même forme que §3.3 avec $d_{\rm ddl}=1$ au lieu de $d$ : **un seul opérateur générique** sert aux deux physiques.
Système : $\widetilde{\mathbf K}_\varphi\,\boldsymbol\varphi=\mathbf b$, $\mathbf b$ = intégrale de flux d'entrée (assemblage de surface
$\int_{\Gamma_{\rm in}}g\,N_a\,d\Gamma$), même technique d'identité sur Dirichlet.

Vitesse reconstruite (1 point de Gauss, au centre de l'élément) : $\mathbf v_e=\sum_a\varphi_a\,\nabla N_a(\mathbf 0)$.

Sensibilités : mêmes formules adjointes qu'au §3.6 (le système est symétrique).

### 4.4 Remarque fondamentale : paradoxe de d'Alembert

> **L'écoulement potentiel stationnaire attaché ne produit aucune traînée** : la force résultante sur un corps
> immobile est nulle (paradoxe de d'Alembert). La « traînée de téléchargement » réelle d'un bras sous un rotor
> est dominée par le **décollement** et le **sillage** (nombre de Reynolds typique $Re=U\,w/\nu\sim10^4$ pour
> $U\approx5$–$10\ \mathrm{m/s}$, largeur de bras $w\approx1$ cm — à vérifier pour votre drone), que Laplace ne représente pas.

Une fonction coût « force de traînée » calculée **uniquement** à partir de $\varphi$ vaudrait $0$ (à l'erreur de
discrétisation près) pour tout design. Le solveur potentiel est donc utilisé comme **pourvoyeur de champ de
vitesse/pression extérieure**, combiné à une **fermeture empirique** de pression de culot. Le résultat est un
**indicateur de forme** (classement relatif de designs), non une prédiction de force absolue. Cette limite est assumée et
documentée (§9, H1) ; le chemin d'amélioration est Brinkman–Stokes/RANS-lite (§9, P1).

### 4.5 Pression et force sur la frontière diffuse

Pression de Bernoulli (loin en amont $p_\infty$) :

$$
p-p_\infty=\tfrac12\rho_a\big(U_{\rm ref}^2-\|\mathbf v\|^2\big)=\tfrac12\rho_aU_{\rm ref}^2\,C_p,\qquad C_p=1-\frac{\|\mathbf v\|^2}{U_{\rm ref}^2}.
$$

Force sur le solide ($\chi_s=\bar\rho$ indicatrice, $\mathbf n$ normale sortante du solide, $\nabla\chi_s=-\mathbf n\,\delta_{\partial\Omega_s}$) :

$$
\mathbf F=-\oint_{\partial\Omega_s}p\,\mathbf n\,dS=\int_\Omega p\,\nabla\bar\rho\;d\Omega .
$$

La forme volumique est **entièrement différentiable et vectorisable** sur grille (gradient de $\bar\rho$ par différences finies centrées, somme sur
éléments). Le terme $p_\infty$ s'intègre à zéro tant que $\bar\rho$ est nul au bord (garanti par le tampon fluide $\mathcal V$ *sauf* sur la
face de raccord au châssis, où l'on retranche la contribution de $p_\infty$ par intégrale de surface, ou l'on prolonge $\bar\rho$ dans le tampon).

### 4.6 Surrogate de traînée verticale (téléchargement) retenu pour la v1

Avec $\partial_z\bar\rho<0$ sur les faces orientées vers l'écoulement entrant (faces « au vent », normale $+\mathbf e_z$)
et $\partial_z\bar\rho>0$ sur les faces sous le corps (« sous le vent »), on remplace la pression sous le vent par une
pression de culot constante $p_b-p_\infty=-\tfrac12\rho_aU_{\rm ref}^2\,c_b$ :

$$
\boxed{\,D=-F_z\approx-\sum_e h^d\Big[(p-p_\infty)_e\,\min(\partial_z\bar\rho_e,0)+(p_b-p_\infty)\,\max(\partial_z\bar\rho_e,0)\Big]\,}
$$

* $D>0$ : **force vers le bas** (perte de portance utile) ; $D$ a la dimension d'une force (N) ;
* $c_b\approx0.2$–$0.5$ ⚠ coefficient de culot (corps épais ; à calibrer sur CFD ou souffleries de référence) ;
* $\min/\max$ non lisses : sous-gradient d'Autograd (ReLU), ou `softplus` d'échelle $\tau$ si instabilité ;
* un design symétrique avant/arrière (cylindre) donne $D>0$ via $c_b$ ; un design caréné (surface « sous le vent » inclinée, $|\partial_z\bar\rho|$
  faible) voit la contribution de culot diminuer : le comportement attendu qualitativement est capturé.

**Biais connu :** dans une cellule d'interface (mélange solide/fluide) la vitesse $\nabla\varphi$ est amortie par $\kappa<1$ ;
$p$ y est surestimée. Mitigations : projection nette ($\beta$ élevé) en fin d'optimisation ; correction $\|\mathbf v\|^2\leftarrow\|\mathbf v\|^2/\max(\kappa,\kappa_c)$ (option, ⚠ non validée).

> **⚠ Mesuré dans la leçon 15 (`notebooks/lessons/15_forces_dalembert.ipynb`) — le modèle ci-dessus n'est pas modifié, mais trois de ses énoncés ne tiennent pas :**
> 1. Les deux termes de $D$ s'intègrent en largeur frontale $A$ ($\sum h^d\max(\partial_z\bar\rho,0)=\sum h^d|\min(\partial_z\bar\rho,0)|=A$), donc $D=\tfrac12\rho_aU_{\rm ref}^2A\,(c_b+\langle C_p\rangle_{\rm amont})$. Le terme de culot **ne dépend pas de la forme sous le vent** : le « design caréné voit la contribution de culot diminuer » ci-dessus est faux pour un $c_b$ constant.
> 2. Pour un cylindre $\langle C_p\rangle_{\rm amont}=-1/3$ (potentiel) : $D>0$ exige $c_b>1/3$, alors que la plage annoncée est $0.2$–$0.5$. Reproduire un $C_d\approx1.1$ demande $c_b\approx1.4$–$1.8$ (valeur de $C_d$ de mémoire).
> 3. Le biais d'interface est de sens **opposé** à celui annoncé : en fractions d'aire avec $q=3$, la vitesse est **surestimée** (jusqu'à $3.2\,U$ dans une cellule solide) et la pression trop basse ; $\langle C_p\rangle_{\rm amont}$ du cylindre vaut $-0.72$ au lieu de $-0.31$. `fluid_side_pressure` (pression prélevée côté fluide) le ramène à $-0.29$ ; ce n'est pas dans le modèle et n'est validé que sur quelques formes 2D. **Leçon 17 :** même corrigé, $D$ reste exploitable par l'optimiseur (aspiration infinie aux arêtes vives du potentiel) ; l'option `floor_at_base` (pression au vent bornée par la pression de culot) le minore par 0 pour un corps fermé. Plancher de bruit de la force numérique : $\sim0.1\,\tfrac12\rho U^2A$ (corps lisse dissymétrique), $\sim10\times$ pour un corps à arêtes vives.

**Alternative sans fermeture** (`blockage`) : conductance effective $C_{\rm eff}=\mathbf b^\top\boldsymbol\varphi/\Delta\varphi$
(énergie de Dirichlet) ; elle mesure l'obstruction globale de la veine et reste purement issue de Laplace. À utiliser comme terme
secondaire si $D$ s'avère trop dépendant de $c_b$.

### 4.7 Normalisation

$$
\hat D=\frac{D}{T_{\rm tot}}\quad(\text{perte de poussée relative, sans dimension}),\qquad T_{\rm tot}=\sum_iT_i .
$$

---

## 5. Couplage et problème d'optimisation

### 5.1 Couplage retenu

**Couplage à travers la géométrie** : $\bar\rho$ est l'unique variable d'état commune ; les deux physiques sont **résolues
indépendamment** à $\bar\rho$ donné, leurs deux coûts sont combinés. Il n'y a pas de couplage d'état (le champ de pression
n'est pas réinjecté comme chargement de la mécanique en v1) : la pression dynamique de l'écoulement ($q=\tfrac12\rho_aU^2\approx 15$–$60\ \mathrm{Pa}$ pour
$U=5$–$10\ \mathrm{m/s}$) est négligeable devant les efforts de poussée dans la structure ; en revanche son effet sur la **portance perdue** n'est pas
négligeable, d'où son entrée par la fonction coût. Un couplage de niveau 2 (pression → force nodale) est
réalisable sans changement d'architecture (chargement différentiable par rapport à $\bar\rho$, CONTRACTS C-SOLVE-04).

### 5.2 Problème

$$
\min_{x\in[0,1]^{N_e}}\ J(x)=w_c\,\frac{\mathcal C(\bar\rho(x))}{\mathcal C_{\rm ref}}+w_a\,\frac{D(\bar\rho(x))}{D_{\rm ref}}
\quad\text{s.c.}\quad g_V(x)=\frac{\sum_e\bar\rho_e\,|\Omega_e|}{|\Omega_{\mathcal D\cup\mathcal S}|}-f_V\le0 .
$$

$\mathcal C_{\rm ref},D_{\rm ref}$ : valeurs au design initial (densité uniforme $x=f_V$), ce qui rend les deux termes sans dimension
et de l'ordre de 1. Le front de Pareto s'obtient en balayant $w_a/w_c$ (pas de ε-contrainte en v1).
Remarque : au design initial uniforme (flou), $D$ n'est pas représentatif ; $D_{\rm ref}$ est recalculé après la première phase de continuation ($\beta\ge4$) ou fixé manuellement.

### 5.3 Graphe de dépendance et différentiation

$$
x\to\rho\to\tilde\rho\to\hat\rho\to\bar\rho\ \Rightarrow\
\begin{cases}
s(\bar\rho)\to\mathbf K\to\mathbf u\ (\text{solve 1})\to\mathcal C\\[2pt]
\kappa(\bar\rho)\to\mathbf K_\varphi\to\boldsymbol\varphi\ (\text{solve 2})\to\mathbf v\to p\to D\ \ (\text{avec } \nabla\bar\rho\text{ direct})\\[2pt]
\bar\rho\to g_V
\end{cases}
\ \to\ J,\ g_V
$$

Toutes les opérations sont des opérations tenseur différentiables par Autograd **sauf les deux résolutions de systèmes
linéaires**, qui sont enveloppées dans une `torch.autograd.Function` dont le backward est l'**équation adjointe** (§3.6) : une
résolution supplémentaire par système, avec la même factorisation (K symétrique). On **ne déroule pas** les itérations du solveur
linéaire dans le graphe (coût mémoire et gradients bruités).

Les dérivées des termes de couplage (normalisation, $\min/\max$, gradient de $\bar\rho$, filtre de surplomb) n'ont donc **jamais** à être
écrites à la main ; la seule dérivée analytique conservée est celle de la compliance, comme oracle de test.

### 5.4 Algorithme d'optimisation

* Baseline : OC (critère d'optimalité) pour $\mathcal C$ + volume seul (validation contre la littérature) ;
* Cas général : MMA (Svanberg), 1 contrainte de volume ;
* Continuation : $p:1\to3$ ; $\beta:1\to\beta_{\max}\approx16$–$32$ (doublement tous les $\sim50$ itérations) ; $\beta_s$ (surplomb) idem.

---

## 6. Récapitulatif des paramètres

| Groupe | Paramètre | Défaut | Statut |
|---|---|---|---|
| SIMP | $p$, $\varepsilon_s$ | 3 ; $10^{-9}$ | standard |
| Filtre | $R$ | $\ge 3\max_a h_a$ | **conception** (taille min. = largeur de buse × n) |
| Projection | $\beta,\eta$ | $1\to16$ ; 0.5 | standard |
| Surplomb | $\theta,\ \beta_s,\ \epsilon$ | $45^\circ$ ; 50 ; $10^{-4}$ | ⚠ à régler |
| Matériau | $E_p,E_z,G_z,\nu_p,\nu_{pz}$ | — | ⚠ **à calibrer** |
| Fluide | $\kappa_{\min},\ q,\ n_{\rm pad}$ | $10^{-6}$ ; 3 ; 4 | ⚠ |
| Rotor | $T,\ A_{\rm rotor},\ k_w,\ U_\infty$ | — | donnée du drone |
| Fermeture | $c_b$ | 0.3 | ⚠ **à calibrer** |
| Pondération | $w_c,w_a$ | 1 ; 0 → balayage | conception |

---

## 7. Cas de vérification analytiques / de référence

| Cas | Ce qui est vérifié | Référence |
|---|---|---|
| Patch test en déformation uniforme | $\mathbf K_e^0$, assemblage, B | exact (≤ $10^{-10}$) |
| Modes rigides | rang de $\mathbf K_e^0$ : 3 (2D) ou 6 (3D) | exact |
| $\mathbf K_e^0$ isotrope 2D contraintes planes | formule fermée de top88 (à permutation de nœuds près) | Andreassen et al. 2011 |
| Poutre console 2D | flèche ≈ Euler–Bernoulli/Timoshenko (élancement élevé) | analytique, tolérance dépendante du maillage |
| Anisotropie | $E_z/E_p\to1$ retrouve l'isotrope ; $\mathbf S\mathbf C=\mathbf I$ | exact |
| Laplace sans obstacle | $\mathbf v=-U\mathbf e_z$ uniforme, flux conservé | exact |
| Laplace, cylindre en veine large | $C_p(\theta)=1-4\sin^2\theta$ | analytique (potentiel, tolérance ~5–10 % selon blocage) |
| Surplomb, $\hat\rho$ binaire, $\beta_s\to\infty$ | coïncide avec la règle de support exacte (boucle NumPy de référence) | exact |
| Gradients | Autograd vs différences finies centrées (float64) | $\le10^{-6}$ relatif |

---

## 8. Dépendances d'échelle (résumé numérique)

* Conditionnement : $\kappa(\widetilde{\mathbf K})\sim\varepsilon_s^{-1}h^{-2}$ ; le CG non préconditionné diverge/stagne en 3D → voir ARCHITECTURE §5.3.
* Le tenseur orthotrope $\mathbf C_0$ peut mélanger des ordres de grandeur (ex. $G_z\ll E_p$) : normaliser $\mathbf C_0$ par $E_p$ avant assemblage et
  travailler en grandeurs réduites ($E_p=1$) à l'intérieur du solveur, convertir aux frontières (CONTRACTS C-CONV-05).

---

## 9. Hypothèses, limites, points ouverts

| # | Sujet | État / décision |
|---|---|---|
| **H1** | Traînée par écoulement potentiel (d'Alembert) | Forces nulles en pur Laplace → surrogate §4.6 avec pression de culot $c_b$. **Indicateur de forme, non prédictif.** Validation : CFD de référence (OpenFOAM/…) sur 2–3 formes, comparer les *classements*. |
| **H2** | Erreur d'interface (pression surestimée en cellule mixte) | Atténuée par $\beta$ élevé ; option de correction à valider. |
| **H3** | Orthotropie en **raideur** ≠ **résistance** | La raideur ne capture pas la fragilité inter-couches (σ_max selon $z$ beaucoup plus faible que dans le plan). Un modèle de rupture (Tsai–Hill/Hill, contrainte locale dans le repère matériau, agrégation $P$-norme) est prévu **comme contrainte optionnelle en phase 6** ; sans lui, l'optimum peut être raide mais fragile en pelage. |
| **H4** | Repère matériau = repère de la grille ; isotropie transverse dans le plan | Pas de dépendance à l'angle de remplissage (raster) ni de trajectoire d'impression ; les périmètres/remplissages ne sont pas distingués. |
| **H5** | Surplomb : seul $\theta=45^\circ$ exact ; ponts (bridging) interdits | Les travées horizontales courtes, imprimables en pratique, sont retirées : conservatisme assumé. |
| **H6** | Charges dynamiques, vibrations (modes), fatigue | Hors périmètre. |
| **H7** | Interactions rotor–structure : le champ d'entrée est prescrit | Pas de rétroaction de l'obstacle sur le rotor. |
| **H8** | Singularités du potentiel aux arêtes vives | $\|\mathbf v\|\to\infty$ en théorie ; régularisé par la taille de maille et le filtre ⇒ dépendance au maillage de $D$ : **vérifier la convergence en maillage**. |
| **P1** | Évolution vers un modèle visqueux | Brinkman–Stokes (pénalisation $\alpha(\bar\rho)\mathbf u$) sur la même grille ; le protocole `FlowModel` (CONTRACTS C-FLOW-01) permet l'échange sans toucher le reste. |
| **P2** | Matériaux multiples, ou répartition des périmètres | Non prévu. |

---

## 10. Références

1. Bendsøe, M. P., Sigmund, O. — *Topology Optimization: Theory, Methods and Applications*, Springer, 2003.
2. Sigmund, O. — « A 99 line topology optimization code written in Matlab », *SMO*, 2001.
3. Andreassen, E. et al. — « Efficient topology optimization in MATLAB using 88 lines of code », *SMO*, 2011.
4. Bourdin, B. — « Filters in topology optimization », *IJNME*, 2001.
5. Guest, J. K., Prévost, J. H., Belytschko, T. — « Achieving minimum length scale in topology optimization using nodal design variables and projection functions », *IJNME*, 2004.
6. Wang, F., Lazarov, B. S., Sigmund, O. — « On projection methods, convergence and robust formulations in topology optimization », *SMO*, 2011.
7. Svanberg, K. — « The method of moving asymptotes — a new method for structural optimization », *IJNME*, 1987.
8. Langelaar, M. — « An additive manufacturing filter for topology optimization of print-ready designs », *SMO*, 2017 ; et « Topology optimization of 3D self-supporting structures for additive manufacturing », *Additive Manufacturing*, 2016.
9. Gaynor, A. T., Guest, J. K. — « Topology optimization considering overhang constraints: eliminating sacrificial support material in additive manufacturing through design », *SMO*, 2016.
10. Borrvall, T., Petersson, J. — « Topology optimization of fluids in Stokes flow », *IJNME*, 2003.
11. Lazarov, B. S., Sigmund, O. — « Filters in topology optimization based on Helmholtz-type differential equations », *IJNME*, 2011.
12. Hoerner, S. F. — *Fluid-Dynamic Drag*, 1965 (ordres de grandeur de pression de culot).
13. Leishman, J. G. — *Principles of Helicopter Aerodynamics*, Cambridge UP (théorie de la quantité de mouvement, sillage du rotor).

> Les références sont citées de mémoire (auteurs/titres) : **vérifier volumes et pages avant publication.**
