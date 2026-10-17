# CONTRACTS.md — Contrats d'interface, d'invariants et de vérification

> Statut : **v0.1 — normatif.** Un contrat est un énoncé **vérifiable** sur une interface ou un invariant.
> Chaque contrat a un identifiant stable `C-<MODULE>-<NN>` et **un test** (colonne « Test »,
> nom de fonction dans `tests/`). Un contrat sans test est une dette ; un test sans contrat est suspect.
> Les symboles mathématiques renvoient à [FORMALISM.md](FORMALISM.md) ; la structure du code à [ARCHITECTURE.md](ARCHITECTURE.md).

**Vocabulaire.** MUST = obligatoire (le test échoue sinon) ; SHOULD = recommandé (avertissement) ; « phase N » = jalon de la feuille de route (ARCHITECTURE §9) à partir duquel le test cesse d'être `skip`.

---

## 1. Conventions globales

| ID | Contrat | Test |
|---|---|---|
| C-CONV-01 | **Axes** : en 2D les axes sont `(x, z)`, en 3D `(x, y, z)`. L'axe `z` est le **dernier axe** (`axis = d-1`) et c'est simultanément la direction de fabrication, la verticale, et l'axe d'écoulement. Aucun module ne suppose une autre orientation. | `test_conventions::test_build_axis_is_last` |
| C-CONV-02 | **Unités** : SI strict aux interfaces publiques (m, N, Pa, kg, s). Pas de mm/MPa implicites. Toute grandeur sans dimension (densités, $\beta$, $w_c$…) est documentée comme telle. | `test_conventions::test_si_roundtrip` |
| C-CONV-03 | **Types** : calcul en `torch.float64` par défaut. `float32` est autorisé uniquement via `config.dtype` explicite et déclenche un `UserWarning` pour les problèmes 3D (conditionnement). Tout tenseur d'un même calcul partage `dtype` **et** `device`. | `test_conventions::test_dtype_device_consistency` |
| C-CONV-04 | **Ordonnancement C (row-major), dernier axe le plus rapide** pour éléments, nœuds ; ddl globaux $= d\cdot n+c$ (nœud-majeur, composante-mineure) ; potentiel : ddl $=n$. Nœud local $a$ d'un élément : bits $(b_0..b_{d-1})$, $b_0$ de poids fort. | `test_grid::test_node_and_dof_ordering` |
| C-CONV-05 | **Mise à l'échelle** : le solveur travaille en grandeurs réduites ($E_p=1$, $h$ physique conservé). La conversion SI ↔ réduit se fait **uniquement** dans `Scaling` (aux frontières IO et objectifs). Aucune constante physique en dur ailleurs. | `test_conventions::test_scaling_inverse` |
| C-CONV-06 | **Voigt** : ordre `(xx,yy,zz,yz,xz,xy)` en 3D, `(xx,zz,xz)` en 2D, cisaillements d'ingénierie. | `test_material::test_voigt_order` |
| C-CONV-07 | **Signes** : $\mathbf e_f=-\mathbf e_z$ (écoulement descendant) ; poussée moteur $+\mathbf e_z$ ; $D>0$ = force **vers le bas** ; $\mathrm{U_{in}}\ge0$ = vitesse entrante. | `test_aero_drag::test_sign_conventions` |
| C-CONV-08 | **Pas de mutation en place** d'un tenseur qui participe au graphe Autograd (`x.add_`, `x[i]=…` sur un tenseur à gradient). Les `index_add_` sont autorisés **uniquement** sur des tenseurs fraîchement alloués locaux. | revue + `test_pipeline::test_no_inplace_leaf` |
| C-CONV-09 | **Déterminisme** : à graine et device fixés, deux exécutions donnent la même suite de designs à `1e-12` près (CPU, solveur direct). `torch.use_deterministic_algorithms(True)` activé dans les tests. | `test_conventions::test_determinism_cpu` |

---

## 2. Tenseurs canoniques

`E = Nₑ` (éléments), `Nn` (nœuds), `Nu = d·Nn`. Forme **grille** = `(n₀,…,n_{d-1})` ; forme **plate** = `(Nₑ,)`, conversion par `reshape(-1)` (ordre C, C-CONV-04).

| Nom | Forme | Dtype | Domaine / invariant | Grad ? | Producteur → consommateur |
|---|---|---|---|---|---|
| `x` | grille | float | `[0,1]` (sur `D`) | **oui** (feuille) | optimiseur → `DensityPipeline` |
| `rho_tilde` / `rho_hat` | grille | float | `[0,1]` | oui | filtre / projection |
| `rho_bar` | grille | float | `[0,1]`, `rho_bar ≤ rho_hat + 1e-12` hors masque `S` | oui | surplomb → mécanique **et** fluide |
| `s` | `(Nₑ,)` | float | `[ε_s, 1]` | oui | SIMP → opérateur K |
| `kappa` | `(Nₑ,)` | float | `[κ_min, 1]` | oui | pénalisation fluide → opérateur Kφ |
| `u` | `(Nu,)` | float | `u[fixed]==0` exactement | oui | solve 1 |
| `f` | `(Nu,)` | float | `f[fixed]` ignoré (masqué) | oui (si dépend de `rho_bar`) | `LoadCase` |
| `phi` | `(Nn,)` | float | `phi[out]==0` exactement | oui | solve 2 |
| `v` | `(Nₑ, d)` | float | $\nabla\varphi$ au centre d'élément | oui | `flow.velocity` |
| `edof` | `(Nₑ, d·2^d)` | int64 | valeurs dans `[0,Nu)`, lignes sans doublon | non | `Connectivity` |
| `Ke0` | `(d·2^d, d·2^d)` | float | symétrique, semi-défini positif, rang = `d·2^d − (3 si 2D, 6 si 3D)` | non | `elements` |
| scalaires `J`, `C`, `D`, `g_V` | `()` | float | finis | oui | objectifs |

Le **plan** (`Connectivity`, `Ke0`, masques, noyaux de filtre) est précalculé **une fois** à la création du `Problem` et ne dépend jamais de `x` (C-GRID-03).

---

## 3. Contrats par module

### 3.1 `grid` — grille et connectivité

| ID | Contrat | Test |
|---|---|---|
| C-GRID-01 | `Grid(shape, h)` : `shape` entiers ≥ 1, `len(shape)∈{2,3}`, `h` réels > 0 de même longueur ; sinon `ConfigError`. | `test_grid::test_grid_validation` |
| C-GRID-02 | `n_elems == ∏shape`, `n_nodes == ∏(shape+1)`, `n_dofs(ncomp) == ncomp·n_nodes`. | `test_grid::test_counts` |
| C-GRID-03 | `Connectivity.edof` est **indépendant de la densité**, calculé par vectorisation pure (pas de boucle Python sur les éléments), identique CPU/GPU. | `test_grid::test_edof_vectorised_matches_loop` |
| C-GRID-04 | Les coordonnées nodales reconstruites depuis `edof` et `Ke0` donnent un élément de volume $\prod h_a$ (somme des poids de Gauss × detJ). | `test_grid::test_element_volume` |
| C-GRID-05 | Masques `D`, `S`, `V` : partition de la grille (disjoints, union = grille) ; sinon `ConfigError`. | `test_grid::test_masks_partition` |

### 3.2 `materials` — loi de comportement

| ID | Contrat | Test |
|---|---|---|
| C-MAT-01 | `Orthotropic(E1,E2,E3,G23,G13,G12,nu12,nu13,nu23)` : exige `S ≻ 0` (Cholesky) ; sinon `MaterialError` avec la condition violée. | `test_material::test_rejects_non_spd` |
| C-MAT-02 | `compliance_matrix()` symétrique ; `C0 @ S == I` à `1e-12` (relatif). | `test_material::test_C_S_inverse` |
| C-MAT-03 | `Orthotropic.isotropic(E,nu)` ≡ `Isotropic(E,nu)` (composantes de `C0` identiques à `1e-12`). | `test_material::test_iso_is_special_case` |
| C-MAT-04 | `TransverselyIsotropic(Ep,Ez,nup,nupz,Gz)` : `E1==E2`, `G12==Ep/(2(1+nup))`, `nu13==nu23` ; construction via `Orthotropic`. | `test_material::test_transverse_isotropy` |
| C-MAT-05 | `stiffness(plane=...)` en 2D : `"stress"` = inversion **après** extraction `{xx,zz,xz}` de `S` ; `"strain"` = extraction **après** inversion de `C`. Les deux coïncident si `ν_{xy}=ν_{yz}=0`. | `test_material::test_plane_reductions` |
| C-MAT-06 | Retourne un tenseur `(n_voigt, n_voigt)` `float64`, **sans** gradient (constant du problème). | `test_material::test_returns_constant` |
| C-MAT-07 | `rotation_xz(θ)`, `stress_transform(R)`, `rotate_stiffness(C, R)` : la rigidité du matériau tourné par `R` coïncide à `1e-12` avec la rotation du tenseur d'ordre 4 `C'_{ijkl}=R_{ia}R_{jb}R_{kc}R_{ld}C_{abcd}` ; le module dans la direction `x` suit `1/E(θ)=cos⁴θ/E₁+(1/G₁₃−2ν₁₃/E₁)sin²θcos²θ+sin⁴θ/E₃` ; `R(a)R(b)=R(a+b)` ; l'isotrope est invariant ; `θ=90°` échange `E₁` et `E₃`. `Orthotropic.rotated_stiffness(R, d, plane)` applique la réduction 2D **après** la rotation 3D. | `test_material::test_rotation_matches_tensor_oracle`, `test_material::test_rotation_properties` |
| C-MAT-08 | `hill_index(sigma, Strengths)` (contraintes dans les axes du matériau, ordre de Voigt) : `=1` pour une traction uniaxiale égale à la résistance `X`, `Y`, `Z` et pour un cisaillement égal à `S_ij` ; `=` von Mises au carré (divisé par `X²`) si `X=Y=Z=σ_y` et `S=σ_y/√3` ; nul pour une contrainte hydrostatique ; `Strengths` refuse une résistance ≤ 0 (`MaterialError`). | `test_material::test_hill_index` |

### 3.3 `fem.elements` — éléments de référence

| ID | Contrat | Test |
|---|---|---|
| C-ELEM-01 | `shape_grads(d)` : somme des $N_a$ = 1 et somme des $\nabla N_a$ = 0 à chaque point de Gauss. | `test_element::test_partition_of_unity` |
| C-ELEM-02 | `Ke0` symétrique à `1e-12` relatif ; valeurs propres ≥ `-1e-12·‖Ke0‖`. | `test_element::test_ke0_spsd` |
| C-ELEM-03 | **Modes rigides** : exactement 3 (2D) / 6 (3D) valeurs propres nulles (tolérance `1e-10·‖Ke0‖`). | `test_element::test_rigid_body_modes` |
| C-ELEM-04 | **Patch test** : champ de déplacement affine imposé aux nœuds ⇒ $\mathbf K_e^0\mathbf u_e=$ forces nodales équivalentes à la contrainte constante $\mathbf C_0\boldsymbol\varepsilon$ (`1e-10`). | `test_element::test_patch_uniform_strain` |
| C-ELEM-05 | **Oracle top88** : pour un matériau isotrope, contraintes planes, `h=1`, `Ke0` égale la formule fermée de top88 à une permutation de nœuds locaux près (permutation documentée dans le test). | `test_element::test_matches_top88_ke` |
| C-ELEM-06 | `laplace_ke0(h)` : symétrique, $\mathbf K\mathbf 1=\mathbf 0$, rang $2^d-1$. | `test_element::test_laplace_ke0` |
| C-ELEM-07 | Quadrature : Gauss $2^d$ points, exacte pour les monômes jusqu'au degré 3 par variable. | `test_element::test_gauss_exactness` |

### 3.4 `fem.operators` — opérateur élémentaire générique

`StructuredFEOperator(grid, Ke0, ncomp, fixed_mask)` sert à la fois la mécanique (`ncomp=d`) et le potentiel (`ncomp=1`).

| ID | Contrat | Test |
|---|---|---|
| C-OP-01 | `matvec(w, u)` avec `w: (Nₑ,)` (facteur par élément : `s` ou `kappa`) retourne `K̃ u` (identité sur ddl bloqués), **sans matrice assemblée**, par gather → produit avec `Ke0` → scatter-add. Aucune boucle Python sur les éléments. | `test_operator::test_matvec_matches_assembled` |
| C-OP-02 | `matvec` est **linéaire en `u`** et **linéaire en `w`** ; différentiable par rapport aux deux. | `test_operator::test_matvec_bilinear_gradcheck` |
| C-OP-03 | `assemble_csr(w)` : même opérateur que `matvec` à `1e-12` relatif ; motif creux (`indptr/indices`) précalculé, seule `data` est recalculée ; symétrique. | `test_operator::test_csr_equals_matvec` |
| C-OP-04 | `diag(w)` = diagonale exacte de `K̃` (ddl bloqués : 1). | `test_operator::test_diag_matches_dense` |
| C-OP-05 | `dK_contract(lam, u)` retourne `(Nₑ,)` : $\boldsymbol\lambda_e^\top\mathbf K_e^0\mathbf u_e$ (vecteurs **masqués**, bloqués = 0). Doit valoir `∂/∂w_e (lam·K̃(w)u)` (vérifié par différences finies). | `test_operator::test_dK_contract_fd` |
| C-OP-06 | `K̃` est SPD : $\mathbf z^\top\tilde{\mathbf K}\mathbf z>0$ pour `z≠0`, toute `w ≥ w_min > 0`. | `test_operator::test_spd_random_vectors` |
| C-OP-07 | Complexité mémoire de `matvec` : `O(Nₑ·d·2^d)` flottants, jamais `O(Nₑ·(d·2^d)²)` (pas de tenseur `(Nₑ,24,24)` matérialisé). | `test_operator::test_no_element_matrix_tensor` |

### 3.5 `fem.bc` — conditions aux limites et chargements

| ID | Contrat | Test |
|---|---|---|
| C-BC-01 | `Dirichlet(nodes_mask, components)` produit un `fixed_mask: (Nu,) bool`. Au moins 1 ddl bloqué par mode rigide (3 en 2D, 6 en 3D) sinon `BoundaryConditionError` (système singulier). | `test_bc::test_detects_rigid_body_unconstrained` |
| C-BC-02 | `LoadCase.assemble(rho_bar) -> f: (Nu,)` ; en v1 constante, mais la signature accepte `rho_bar` (charges dépendantes du design). | `test_bc::test_loadcase_signature` |
| C-BC-03 | Bilan d'équilibre : $\sum_i f_i$ (par direction) = somme des réactions aux ddl bloqués au signe près, à `1e-8·‖f‖`. | `test_bc::test_reaction_equilibrium` |
| C-BC-04 | Charges réparties sur zone passive : le total ne dépend pas de la finesse du maillage (même résultante pour `n` et `2n`). | `test_bc::test_distributed_load_invariant` |

### 3.6 `fem.solve` — solveurs linéaires et adjoint

| ID | Contrat | Test |
|---|---|---|
| C-SOLVE-01 | `LinearSolver.prepare(op, w) -> Handle` ; `Handle.solve(b)` résout `K̃ y = b`. La **même** `Handle` sert à la résolution directe et à l'adjoint (pas de refactorisation). | `test_solve::test_handle_reuse` |
| C-SOLVE-02 | Erreur inverse normwise `‖K̃u−Pf‖/(‖K̃‖₁‖u‖+‖Pf‖) ≤ 1e-10` (directs ; un résidu relatif à `‖Pf‖` seul croît comme ε·κ et échoue à tort sur poutres élancées et designs à îlots quasi vides) ou `≤ tol` configurable (CG, défaut `1e-8`) ; `SolverConvergenceError` sinon. | `test_solve::test_residual_bounds` |
| C-SOLVE-03 | `solve_spd(op, w, f)` est une `torch.autograd.Function` dont le backward est l'**équation adjointe** (une résolution supplémentaire) ; **aucune** itération du solveur dans le graphe. | `test_solve::test_backward_is_adjoint_only` |
| C-SOLVE-04 | Le backward fournit **à la fois** `∂/∂w` ($-\lambda_e^\top K_e^0u_e$) **et** `∂/∂f` ($P\lambda$), de sorte que `f = f(rho_bar)` (poids propre, pression) est différentiable. | `test_solve::test_grad_wrt_load` |
| C-SOLVE-05 | `gradcheck` float64 (grille `6×4`, `p=3`) : Autograd = différences finies, `rtol ≤ 1e-6`, **pour** `w` et `f`. | `test_solve::test_gradcheck` |
| C-SOLVE-06 | Pour `J=fᵀu` : le gradient sortant égale la formule analytique $-p\bar\rho^{p-1}u_e^\top K_e^0u_e$ à `1e-10`. | `test_solve::test_compliance_matches_analytic` |
| C-SOLVE-07 | Backends **interchangeables** : `scipy_direct` (CPU, référence), `torch_cg` (CPU/GPU, matrix-free, préconditionné) donnent la même `u` à `1e-6` relatif sur le cas `MBB 60×20`. | `test_solve::test_backends_agree` |
| C-SOLVE-08 | Symétrie vérifiée : `‖K̃−K̃ᵀ‖/‖K̃‖ ≤ 1e-12` à l'assemblage (mode debug). Le backward suppose `K̃ᵀ=K̃`. | `test_solve::test_symmetry_check` |
| C-SOLVE-09 | `float64` sur `mps` : refus explicite (`DeviceError`) — MPS ne supporte pas `float64`. | `test_solve::test_mps_float64_rejected` |

### 3.7 `topopt.density` — masques, filtre, projection, SIMP

| ID | Contrat | Test |
|---|---|---|
| C-FILT-01 | Le filtre conserve les constantes : `filter(c·1)=c·1` à `1e-12`, **y compris aux bords** (normalisation `w*1`). | `test_density::test_filter_preserves_constants` |
| C-FILT-02 | `R→0` (`R ≤ min h`) : identité. Le noyau est symétrique et de somme positive. | `test_density::test_filter_identity_small_radius` |
| C-FILT-03 | Sortie dans `[0,1]` si l'entrée l'est (convexité). | `test_density::test_filter_range` |
| C-FILT-04 | Différentiable ; gradient = filtre transposé normalisé (vérifié par `gradcheck`). | `test_density::test_filter_gradcheck` |
| C-PROJ-01 | `Heaviside(β,η)` : `f(0)=0`, `f(1)=1` (`1e-12`), croissante, `f(η)=η` pour tout `β` si `η=0.5`. | `test_density::test_projection_endpoints` |
| C-PROJ-02 | `β=0` **désactive** la projection (identité exacte, équivalence top88) ; `β→∞` ⇒ seuil. | `test_density::test_projection_limits` |
| C-PROJ-03 | `β` est un attribut mutable **hors graphe** (continuation) ; changer `β` ne reconstruit pas le plan. | `test_density::test_beta_continuation_no_rebuild` |
| C-SIMP-01 | `s(0)=ε_s`, `s(1)=1`, strictement croissante, `s'≥0`. | `test_density::test_simp_properties` |
| C-MASK-01 | Après le pipeline : `rho_bar[S]=1` et `rho_bar[V]=0` **exactement** ; les masques sont réappliqués après le filtre de surplomb (FORMALISM §2.5). | `test_density::test_masks_enforced` |

### 3.8 `topopt.overhang` — filtre de surplomb

| ID | Contrat | Test |
|---|---|---|
| C-OVH-01 | **Majoration** : `rho_bar ≤ rho_hat + ε_tol` partout (le filtre ne crée pas de matière), `ε_tol ≈ log(n)/β_s + √ε`. | `test_overhang::test_never_adds_material` |
| C-OVH-02 | **Référence exacte** : pour `rho_hat` binaire et `β_s` grand, le résultat coïncide (`≤ 1e-2`) avec la règle de support exacte implémentée par une **boucle NumPy indépendante**. Le biais `log(n)/β_s` de `smax` est payé **à chaque couche** : il faut `β_s ≳ n_couches·log(n)/10⁻²` (mesuré : `β_s=3000`, `ε=10⁻¹⁰` pour 8 couches). | `test_overhang::test_matches_numpy_reference` |
| C-OVH-03 | Auto-support : pour toute sortie binarisée à 0.5, chaque voxel de la couche `ℓ≥1` a au moins un voxel imprimé dans son stencil en `ℓ−1` (propriété vérifiée sur 100 champs aléatoires). | `test_overhang::test_binarised_output_is_self_supporting` |
| C-OVH-04 | **Stencil** : 2D = 3 points ; 3D = **croix à 5 points** (jamais le carré 3×3). Un test géométrique vérifie que l'angle de surplomb maximal ≤ 45° pour des frontières le long des axes **et** des diagonales. | `test_overhang::test_stencil_cross_guarantees_45deg` |
| C-OVH-05 | La couche `ℓ=0` (plateau) est toujours supportée : `rho_bar[...,0]=rho_hat[...,0]`. | `test_overhang::test_baseplate_supported` |
| C-OVH-06 | Un solide passif `S` non supporté (suspendu dans le vide) lève `ConfigError` à l'initialisation. | `test_overhang::test_passive_solid_unsupported_rejected` |
| C-OVH-07 | **Gradient** non nul et fini (pas de NaN/Inf) en tout point, y compris pour `rho_hat ≡ 0` (smax stable), et `gradcheck` float64 sur `5×5` en 2D / `4×4×4` en 3D. | `test_overhang::test_gradients_finite_and_correct` |
| C-OVH-08 | Axe de fabrication = dernier axe (C-CONV-01) : permuter les axes d'un champ 3D et de l'option `build_axis` ne change pas le résultat (même propriété pour 2D). | `test_overhang::test_build_axis_is_last` |
| C-OVH-09 | Sensibilité à toutes les couches inférieures : `∂ρ̄_ℓ/∂ρ̂_{ℓ'}≠0` pour `ℓ'<ℓ` dans un cas à cône supporté (vérification de la rétropropagation à travers la récurrence). | `test_overhang::test_cross_layer_sensitivity` |

### 3.9 `flow` — écoulement potentiel

| ID | Contrat | Test |
|---|---|---|
| C-FLOW-01 | `FlowModel` (protocole) : `solve(rho_bar) -> FlowState{phi, v, p_rel}` ; les modèles (potentiel aujourd'hui, Brinkman–Stokes plus tard) sont interchangeables **sans modifier** `objectives`. | `test_flow::test_flowmodel_protocol` |
| C-FLOW-02 | **Sans obstacle** (`rho_bar≡0`) : `v = −U e_z` uniforme à `1e-10` ; `phi` linéaire en `z`. | `test_flow::test_uniform_flow_exact` |
| C-FLOW-03 | **Conservation** : flux entrant = flux sortant à `1e-8` relatif (somme des réactions en Dirichlet). | `test_flow::test_flux_conservation` |
| C-FLOW-04 | **Cylindre/disque** en veine large : $C_p(\theta)\approx1-4\sin^2\theta$ (2D) ; tolérance documentée : erreur moyenne de $\|\mathbf v\|/U$ sur l'anneau $1.5R<r<3R$ ≤ 0.03 à blocage $2R/W=1/8$ (mesuré 0.016), qui décroît avec le blocage (0.21 → 0.010 pour $W/2R$ de 1.5 à 12) ; **à moins de 2 cellules de la paroi le $C_p$ numérique est peu fiable** (écart max 0.6, leçon 14). | `test_flow::test_cylinder_cp` |
| C-FLOW-05 | Le tampon fluide d'entrée a une épaisseur ≥ `n_pad` ; un solide (`rho_bar > 1e-3`) dans les `n_pad` couches sous la face d'entrée lève `ConfigError` (un bloc collé à l'entrée donne $\|\mathbf v\|\sim10^6$). `n_pad=0` désactive le contrôle. | `test_flow::test_inlet_padding_enforced` |
| C-FLOW-06 | `kappa(rho_bar)` : `kappa(0)=1`, `kappa(1)=κ_min`, strictement décroissante ; ni NaN ni Inf pour `rho_bar∈[0,1]`. | `test_flow::test_kappa_properties` |
| C-FLOW-07 | Système SPD avec ≥ 1 ddl Dirichlet (sortie) ; la matrice sans Dirichlet est singulière (rang `Nn−1`) — vérifié pour détecter une configuration CL erronée. | `test_flow::test_neumann_nullspace` |
| C-FLOW-08 | Convergence en maillage de $\|\mathbf v\|$ : sur un cas **lisse à solution exacte** (boîte à parois glissantes, $\varphi=-z+a\cos kx\sinh kz/k$), erreur aux centres d'éléments d'ordre ≥ 1 (mesuré 1.96–2.00). Sur le cylindre diffus, l'ordre mesuré est de 0.7 à 1.4, irrégulier : **non** exigé. | `test_flow::test_mesh_convergence` |
| C-FLOW-09 | Réutilise `StructuredFEOperator` avec `ncomp=1` ; **aucun** assembleur dédié au fluide. | `test_flow::test_uses_generic_operator` |

### 3.10 `flow.aero` — force de traînée

| ID | Contrat | Test |
|---|---|---|
| C-AERO-01 | **d'Alembert (oracle)** : la force de pression **complète** $\int p\nabla\bar\rho$ d'un corps symétrique avant/arrière vérifie `|F| ≤ 0.02·½ρU²·A` (mesuré 1e-4 ; l'annulation est due à la symétrie). $D$ à `c_b=0` n'est **pas** cette force (la moitié sous le vent est remplacée par la constante) : $D=\tfrac12\rho U^2A\,(c_b+\langle C_p\rangle_{\rm amont})$, affine en `c_b` de pente $\tfrac12\rho U^2A$. Pour un cylindre $\langle C_p\rangle_{\rm amont}=-1/3$ (potentiel) donc $D>0$ exige $c_b>1/3$ (≈ 0.3 côté fluide, 0.72 en pression brute) : **"avec `c_b>0`, `D>0`" est faux** (réécrit, leçon 15). | `test_aero_drag::test_dalembert_and_base_closure` |
| C-AERO-02 | Invariance d'échelle : à `ρ_a, U_ref` fixés, `D̂=D/T_tot` ne dépend que de `c_b` et de la forme ; doubler `U_ref` multiplie `D` par 4 (± `1e-6`). | `test_aero_drag::test_dynamic_pressure_scaling` |
| C-AERO-03 | `D` est nul si `rho_bar≡0` (pas de corps) et si `rho_bar≡1` (domaine plein, `∇ρ̄=0`). | `test_aero_drag::test_zero_for_trivial_fields` |
| C-AERO-04 | **Ce que la fermeture voit de la forme** : le terme de culot vaut exactement `c_b·½ρU²·A` ($A$ = largeur frontale) **quelle que soit la forme sous le vent** (carénage non vu par un `c_b` constant) ; deux corps de même $A$ et même pression au vent ont le même $D$ ; $D$ croît avec $A$. L'ancien énoncé (« `D` décroît pour un corps plus incliné sous le vent ») est **faux** pour un `c_b` constant (réécrit, leçon 15). | `test_aero_drag::test_streamlining_reduces_D` |
| C-AERO-05 | `D` différentiable par rapport à `rho_bar` (chemin direct `∇ρ̄` + chemin via `φ`), avec et sans `fluid_side_pressure` ; `gradcheck` float64, sous-gradient accepté aux points de non-différentiabilité de `min/max`. | `test_aero_drag::test_gradcheck` |
| C-AERO-06 | Valeur de `c_b`, de `k_w` et de `κ` **explicitement** présentes dans la sortie de résultat (traçabilité des paramètres calibrés). | `test_aero_drag::test_params_recorded_in_results` |

### 3.11 `objectives` — fonctions coût et contraintes

| ID | Contrat | Test |
|---|---|---|
| C-OBJ-01 | Toute fonction objectif est `f(rho_bar, state) -> scalar` avec `state` le cache des solves (une seule résolution mécanique et une seule résolution fluide par évaluation). | `test_objectives::test_single_solve_per_eval` |
| C-OBJ-02 | `Composite.value()` renvoie `J`, ainsi que **chaque terme brut et normalisé** (`C`, `Ĉ`, `D`, `D̂`) — pas seulement la somme. | `test_objectives::test_terms_reported` |
| C-OBJ-03 | Normalisation $\mathcal C_{\rm ref}$, $D_{\rm ref}$ fixée **une fois** et figée ensuite ; changer de référence en cours de route est interdit (déplacerait l'optimum, `ConfigError`) ; une référence ≤ 0 ou non finie est refusée (elle retournerait le signe du terme). **Les références doivent venir d'un design convergé** (compliance seule), pas du premier itéré : au design uniforme flou $D$ n'est pas représentatif (leçon 17 : $\hat D$ atteint des milliers). | `test_objectives::test_refs_frozen` |
| C-OBJ-04 | `volume_fraction(rho_bar, mask)` correct sur zones passives (dénominateur : $|\Omega_{\mathcal D\cup\mathcal S}|$, le vide passif ne compte jamais) ; `g_V` du bon signe ; forme relative `V/f_V−1` disponible. | `test_objectives::test_volume_constraint` |
| C-OBJ-05 | Poids `w_c,w_a ≥ 0`, au moins un > 0 ; sinon `ConfigError`. | `test_objectives::test_weights_validation` |
| C-OBJ-06 | Gradient du coût composite : $\nabla_xJ$ (Autograd à travers les deux adjoints, le filtre, la projection, la pression côté fluide) égal aux différences finies centrées, float64, pour $w_a\in\{0,\tfrac12,1\}$. | `test_objectives::test_composite_gradient` |
| C-OBJ-07 | Bout en bout (problème couplé) : MMA baisse $J$, respecte le volume ; un poids plus grand sur $D$ donne un $D$ plus faible et un $C$ plus grand (mêmes références). | `test_objectives::test_coupled_descent` |
| C-AERO-07 | `floor_at_base=True` : pression au vent bornée par la pression de culot ; $D\ge0$ pour tout corps fermé et tout champ ; gradient nul là où le plancher est actif. Sans plancher, $D$ n'est pas minoré (aspiration d'arête vive) et l'optimiseur le descend sous 0 (leçon 17). | `test_aero_drag::test_floor_at_base` |

### 3.12 `optim` — optimiseurs

| ID | Contrat | Test |
|---|---|---|
| C-OPT-01 | Interface : `Optimizer.step(x, f, df, g, dg) -> x_new` en `numpy`/`torch` CPU ; bornes `[0,1]` respectées exactement. | `test_optim::test_bounds_respected` |
| C-OPT-02 | **OC** : sur le benchmark MBB `60×20`, `V=0.5`, `R=1.5·h`, `p=3` : la compliance finale est dans `±2 %` de la valeur de référence de la littérature (top88) **recalculée** avec le script de référence du dépôt (ne pas coder une valeur de mémoire). | `test_mbb_benchmark::test_oc_matches_reference` |
| C-OPT-03 | Le volume est respecté à la fin : `g_V ≤ 1e-3` (OC) ; à `1e-6` (MMA). | `test_optim::test_volume_feasible` |
| C-OPT-04 | **Continuation** : `p`, `β`, `β_s` suivent un calendrier configuré ; la fonction d'objectif est **recalibrée** (historique MMA réinitialisé) à chaque changement de paramètre de continuation. | `test_optim::test_continuation_schedule` |
| C-OPT-05 | Critère d'arrêt : variation max de `x` < tol **et** `β` à sa valeur finale, ou itérations max ; jamais d'arrêt silencieux (la cause d'arrêt est retournée). | `test_optim::test_stop_reason_reported` |

### 3.13 `io` — sorties et reproductibilité

| ID | Contrat | Test |
|---|---|---|
| C-IO-01 | Chaque exécution écrit `config.resolved.yaml`, `history.csv` (`it, J, C, D, g_V, β, p, ‖Δx‖, t_it`) et `rho_bar.npy` au dossier de run. | `test_io::test_run_artifacts` |
| C-IO-02 | Export VTK (`.vti`/`.vtu`) de `rho_bar`, `u`, `phi`, `v` ; lisible par ParaView ; aucune dépendance obligatoire (import paresseux). | `test_io::test_vtk_export` |
| C-IO-03 | Le dossier de run enregistre `git rev-parse HEAD` (ou `"nogit"`), versions `torch/scipy/numpy`, device, dtype. | `test_io::test_provenance_recorded` |
| C-EXP-01 | `marching_squares(field, level)` : contours **fermés**, chaque sommet est sur une arête de la grille d'échantillonnage où le champ interpolé vaut `level` (à `1e-12`) ; aire signée = aire de la matière (disque, couronne : `< 1 %` à rayon 15 cellules, trous comptés négativement) ; les selles sont tranchées par la valeur du centre. | `test_export::test_marching_squares_contour` |
| C-EXP-02 | `marching_tetrahedra(field, level)` : surface **fermée et orientée** (chaque arête partagée par exactement deux faces, parcourue une fois dans chaque sens), sans face dégénérée, sur des champs aléatoires comme sur une boule (caractéristique d'Euler 2) et un tore (0). | `test_export::test_marching_tetrahedra_watertight` |
| C-EXP-03 | Volume signé du maillage (théorème de la divergence) `> 0` (normales sortantes) et égal au volume analytique à `2 %` (boule de rayon 12 cellules) ; il **converge** vers lui quand on raffine. | `test_export::test_mesh_volume_and_orientation` |
| C-EXP-04 | STL binaire : `84 + 50·F` octets ; relecture identique (`float32`) en binaire et en ASCII ; normales de facette unitaires et sortantes. | `test_export::test_stl_roundtrip` |
| C-EXP-05 | `thin_fraction(solid, r)` : `0` pour une barre plus large que `2r`, et ≈ 1 pour une barre plus fine ; monotone en `r`. | `test_export::test_thin_fraction` |

---

## 4. Contrats transverses

### 4.1 Différentiabilité

| ID | Contrat | Test |
|---|---|---|
| C-DIFF-01 | **Pipeline complet** : `gradcheck` (float64, petites grilles `6×4` 2D, `4×3×3` 3D) de `J(x)` pour **chaque terme** (`C`, `D`, `g_V`) pris isolément, puis combinés. `rtol ≤ 1e-5`, `atol ≤ 1e-7`. | `test_gradcheck_pipeline::test_full_chain` |
| C-DIFF-02 | Aucun `NaN/Inf` dans les gradients sur une trajectoire d'optimisation de 50 itérations (2D) ; test de non-régression. | `test_gradcheck_pipeline::test_no_nan_trajectory` |
| C-DIFF-03 | Aucune différentiation « à la main » d'un terme de couplage : tout gradient de production provient d'Autograd, **sauf** le backward adjoint de `solve_spd` (C-SOLVE-03). Les formules analytiques ne figurent que dans `tests/`. | revue de code (`test_conventions::test_no_handwritten_grads_outside_solve`) |
| C-DIFF-04 | Les opérateurs non différentiables en un point (`min/max`, ReLU) documentent leur sous-gradient choisi ; aucun `torch.where` qui divise par zéro dans la branche non retenue (piège NaN d'Autograd). | revue + `test_gradcheck_pipeline::test_where_nan_trap` |

### 4.2 Performance et mémoire

| ID | Contrat | Test |
|---|---|---|
| C-PERF-01 | Aucune boucle Python sur éléments, nœuds ou ddl dans le chemin chaud (`matvec`, `filter`, `projection`, `drag`). Autorisées : boucle sur les **couches** du filtre de surplomb (≤ `n_z`), boucle sur les itérations. | `test_perf::test_no_python_loops_hot_path` (inspection AST) |
| C-PERF-02 | Budget mémoire du `matvec` 3D `120×40×20` en float64 : pic ≤ 1.5 GB (hors solveur). | `test_perf::test_matvec_memory` (marqué `slow`) |
| C-PERF-03 | Temps par itération 2D `300×100` (CPU, solveur direct) ≤ 3 s ; mesure consignée dans `history.csv` (`t_it`). | `benchmarks/bench_2d.py` (non bloquant) |

### 4.3 Erreurs

| Exception | Levée quand |
|---|---|
| `ConfigError` | configuration invalide ou incohérente (grille, masques, poids, solide passif non supporté, tampon fluide insuffisant) |
| `MaterialError` | tenseur de raideur non défini positif |
| `BoundaryConditionError` | système sous-contraint (modes rigides) ou chargement hors domaine |
| `SolverConvergenceError` | résidu au-dessus de la tolérance |
| `DeviceError` | combinaison `dtype/device` non supportée (ex. `float64` sur MPS) |

Toutes dérivent de `AeroptimizerError`. **Jamais de `assert` pour valider une entrée utilisateur** (désactivés avec `-O`) ; `assert` réservés aux invariants internes.

### 4.4 Configuration

| ID | Contrat | Test |
|---|---|---|
| C-CFG-01 | La configuration est un fichier YAML validé par des `dataclass` typées ; toute clé inconnue est une erreur (pas d'ignorance silencieuse). | `test_config::test_unknown_key_rejected` |
| C-CFG-02 | Un même YAML, graine et device ⇒ même `config.resolved.yaml` (les valeurs par défaut sont explicitées). | `test_config::test_resolved_roundtrip` |
| C-CFG-03 | Les paramètres marqués ⚠ dans FORMALISM (`c_b`, `k_w`, `E_z`, …) n'ont **pas** de valeur par défaut silencieuse : le YAML d'exemple les fournit avec un commentaire `# TODO: calibrer`. | `test_config::test_calibration_params_explicit` |

---

## 5. Politique de test

* **Pyramide** : `tests/unit/` (≈ 1 fichier par module, rapide, 2D `≤ 8×8`), `tests/integration/` (benchmarks MBB, console 2D, chaîne complète), `benchmarks/` (non bloquant, perfs).
* **Tolérances** : `float64` ; `1e-12` pour l'algèbre exacte, `1e-10` pour les solveurs directs, `1e-6` pour Autograd vs différences finies, `1e-8` pour les itératifs.
* **Marqueurs** : `@pytest.mark.phase(N)` (skip si la phase n'est pas atteinte), `@pytest.mark.slow`, `@pytest.mark.gpu`, `@pytest.mark.reference` (compare à un script de référence NumPy indépendant).
* **Oracles indépendants** : toute brique critique a une **implémentation de référence naïve en NumPy** (boucles explicites) dans `tests/reference/` ; l'implémentation vectorisée doit lui être égale. Les références ne sont pas importées par `src/`.
* **Règle d'or** : un bug corrigé ajoute d'abord un test qui échoue.

---

## 6. Traçabilité contrats → tests

L'index complet est généré par `scripts/contracts_index.py` (parcourt ce fichier, extrait les `C-*` et vérifie que chaque `Test` existe dans `tests/`).
Un contrat dont le test est introuvable fait échouer la CI. Au démarrage, **tous les tests sont `skip(reason="phase N")`** : le squelette du dépôt est la liste de travail.
