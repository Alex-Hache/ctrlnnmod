# Théorie — « gradient non nul, perte plate » sur ∂S (classe `blockp`)

Session du 2026-10-22. Fiche : `gradient_non_nul_perte_plate.md`. Bibliographie : `biblio_gradient_plat.md` (renvois [A1]…[E4]). Lean : `these-lean`, `TheseLean/Ext/ScaleInvariance.lean`.

## 0. Faits à expliquer (run P de T4 ; Adam lr 1e-2, β = (0,9 ; 0,999), clip global 10, perte pleine, pas de weight decay)

- F1 : les 8 têtes restent saturées (W sur ∂S) pendant les 3000 pas ; le maître est à l'intérieur (marge 0,009).
- F2 : au point final, le gradient en (Ã, B̃, C), projeté sur l'orthogonal des normales lisses ∇r_h, vaut 0,95 à 5,8 × ‖g(t = 0)‖ (U1). La composante normale est faible (‖g‖ ≈ ‖g_tan‖).
- F3 : la perte est plate en moyenne après t ≈ 1000–2000, mais **oscille** : sur [2000, 3000), elle monte à 41–50 % des pas ; variation médiane par pas 0,1–1,4 % ; pics de 4 à 69 % en un pas (journaux de T4, lus le 2026-10-22, sans nouveau calcul).
- F4 : r brut max passe de 0,90 à 3,8–4,6 en 10 pas, culmine à 4,8–8 vers t = 50, redescend vers 3–5 (t = 100–500), puis remonte lentement (4,1–5,9 à t = 3000).
- F5 : le gradient projeté avec rétraction radiale (θ gardé à r ≤ ρ) fait pire (test 0,77 contre 0,29).
- F6 : les mêmes équations sans contrainte réussissent (0,0018).
- Point d'échelle : à la fin, le pas d'Adam sur W vaut ≈ η·ρ/r_brut ≈ 2e-3 par coordonnée (voir § 2). Ce n'est pas un pas minuscule.

## 1. Cadre

E espace préhilbertien réel. Pour une tête h, θ_h = (Ã_h, B̃_h) et r_h(θ) = Σ_k ‖A_hk‖₂ + Σ_q ‖B_hq‖₂ · Σ_k ‖C_qk‖₂. À C fixé, r_h est positivement homogène de degré 1 en θ_h. Sur U_h = {r_h > ρ}, Φ_h(θ) = (ρ / r_h(θ)) · θ_h. Donc, à C fixé, pour tout t > ρ / r_h(θ) : Φ(… t·θ_h …) = Φ(… θ_h …). La perte f = L ∘ Φ est **localement invariante** le long du rayon de θ_h. Seul le bloc (Ã_h, B̃_h) d'une tête saturée a cette propriété (C, D, E, biais, portes, lecture : non).

## 2. Résultats exacts (démontrés en Lean)

Notation : x ∈ E, f : E → ℝ, g = ∇f(x).

**P1 (Euler).** Si f(t·x) = f(x) pour t voisin de 1 et si f a le gradient g en x, alors ⟨g, x⟩ = 0.
Preuve : φ(t) = f(t·x) a la dérivée ⟨g, x⟩ en t = 1 (règle de la chaîne) et φ est constante près de 1.
Version par bloc : si f(x + s·v) = f(x) pour s voisin de 0, alors ⟨g, v⟩ = 0. Avec v = composante radiale du bloc h, on obtient ⟨∇_h f, θ_h⟩ = 0.
Lean : `inner_grad_eq_zero_of_ray_invariant`, `inner_grad_eq_zero_of_line_invariant`.
Conséquence : **la direction radiale d'une tête saturée est dans le noyau de J**, et le gradient en θ n'a aucune composante radiale. C'est le déficit de rang de M1 (= nombre de têtes saturées), maintenant démontré en général.

**P2 (degré −1).** Si f(c·y) = f(y) pour y voisin de x (c > 0) et si f a le gradient g en x, alors f a le gradient c⁻¹·g en c·x.
Preuve : près de c·x, f = f ∘ (c⁻¹·) ; règle de la chaîne ; l'adjoint de c⁻¹·Id est c⁻¹·Id.
Lean : `hasGradientAt_smul_of_invariant` (c ≠ 0) ; version générale `hasGradientAt_comp_symm` : si f ∘ A = f près de x, A linéaire inversible symétrique, le gradient en A x vaut A⁻¹ g (sert pour une homothétie par bloc, A = diag(s)).

**P3 (Pythagore).** Si ⟨g, x⟩ = 0, alors ‖x − η·g‖² = ‖x‖² + η²‖g‖² [A1, A4]. Lean : `norm_sub_smul_sq_of_inner_eq_zero`.

**P4 (pas angulaire).** Si ⟨d, x⟩ = 0 et x ≠ 0, alors ‖(x+d)/‖x+d‖ − x/‖x‖‖ ≤ ‖d‖/‖x‖.
Preuve : avec a = ‖x‖, b = ‖d‖, n = ‖x+d‖ = √(a²+b²), le carré de la corde vaut 2 − 2a/n = 2b²/(n(n+a)) ≤ b²/a².
Avec P2 (c = ‖x‖, x̂ = x/‖x‖) : un pas de GD tourne la direction d'au plus η·‖∇f(x̂)‖/‖x‖². Le taux effectif de GD sur la direction est η/‖x‖² [A2, A3].
Lean : `norm_dir_add_sub_dir_le` ; `dir_gd_step_le` (f invariante par toute homothétie c > 0).

**P5 (GD : croissance de la norme et absorption).** Si chaque gradient est orthogonal à son itéré, alors ‖x_t‖² = ‖x_0‖² + η² Σ_{s<t} ‖g_s‖², donc ‖x_t‖ croît (Lean : `gd_norm_sq_eq`).
Absorption (Lean, version par bloc : `gd_absorbing`) : soit P le projecteur orthogonal sur un bloc, N avec N(x) ≥ α‖P x‖ (α > 0), et f localement invariante le long de P x sur {N > ρ}. Si α‖P x_0‖ > ρ, alors, pour **tout** pas η, toutes les itérations de GD vérifient N(x_t) > ρ, et ‖P x_t‖ ne décroît pas. Pour `proj` (Lean : `projBlock`, `projBlock_line_invariant`, `proj_gd_absorbing`) : Φ(y) = (y − P y) + (ρ / max(N y, ρ))·P y est invariante le long de P y dès que N y > ρ > 0 et que N est homogène de degré 1 le long du bloc ; la conclusion d'absorption suit.
Version sur tout l'espace (P = Id) : soit N positivement homogène avec N(x) ≥ α‖x‖ (α > 0), et f localement invariante le long des rayons sur {N > ρ}. Si α‖x_0‖ > ρ, alors, pour **tout** pas η, toutes les itérations de GD vérifient N(x_t) > ρ.
Lecture pour `proj` (à C fixé, N = r_h, α > 0 tant que les c_q = Σ_k‖C_qk‖ sont minorés) : sous GD, une tête saturée assez loin du bord **ne revient jamais** dans S. Le gradient ne donne aucun signal vers l'intérieur (P1), et chaque pas éloigne θ_h de l'origine (P3). C'est F1 sous GD (flots GD de M2-ter et M2-q, T3 : 15/15 contacts sans retour).

**P6 (taux effectifs : conjugaison exacte).** Coordonnées (ι → ℝ), échelle s : ι → ℝ, s_i > 0, gradient G avec G(s⊙x) = s⁻¹⊙G(x) (vrai pour un bloc invariant par P2, s = c sur le bloc, 1 ailleurs).
- GD : l'itéré de GD depuis s⊙x₀ au pas η est s⊙(itéré de GD depuis x₀ au pas η/s²).
- Adam avec ε = 0, sans clip : la mise à jour m̂/√v̂ est inchangée si tous les gradients passés sont multipliés par un même c > 0. Donc l'itéré d'Adam depuis s⊙x₀ au pas η est s⊙(itéré d'Adam depuis x₀ au pas η/s).
Lean : `gd_conj`, `adam_dir_scale`, `adam_conj` (G abstrait ; le passage « bloc invariant ⇒ G(s⊙x) = s⁻¹⊙G(x) » est `hasGradientAt_comp_symm`, non refait en coordonnées).
Lecture : sur une tête saturée, l'échelle brute ‖θ_h‖ divise le pas : η/‖θ_h‖² sous GD, **η/‖θ_h‖ sous Adam** (version exacte de l'énoncé approché de [A2]). Adam est coordonnée par coordonnée : la conjugaison vaut bloc par bloc. Le clip global couple les blocs : s'il est actif, il casse la conjugaison (facteur variable dans le temps). ε = 1e-8 ne compte que si √v̂ est de cet ordre.

**P7 (ε-sous-différentiel d'une jauge).** Pour N positivement homogène (aucune autre hypothèse ; Lean : `eps_subgrad_iff`) : g vérifie N(y) ≥ N(x) + ⟨g, y − x⟩ − ε pour tout y **si et seulement si** ⟨g, z⟩ ≤ N(z) pour tout z, et ⟨g, x⟩ ≥ N(x) − ε. (Sert au test du § 6.)

**P8 (pli : gradient tangent non nul mais point KKT).** Exemple : S = {|a| + |b| ≤ 1}, W = (1, 0), g = −(1 ; ½). La normale « lisse » prise par la dérivation automatique (signe(0) = 0) est (1, 0) ; la partie tangente de g vaut (0 ; −½) ≠ 0. Mais −g = (1 ; ½) ∈ N_S(W) = {λ(1, s) : λ ≥ 0, |s| ≤ 1} : W est un point KKT. Une projection sur la normale lisse peut donc annoncer un gradient tangent non nul en un point stationnaire. Lean : `kink_kkt_tangent_ne_zero`.

## 3. Piste 1 (invariance d'échelle) : ce qu'elle explique, ce qu'elle n'explique pas

Explique :
- le déficit de rang de M1 (P1) ;
- sous GD, l'absorption par ∂S quel que soit le pas (P5) ;
- le saut initial de r brut (F4, 0,90 → 3,8–4,6 en 10 pas) : au départ, ‖θ_h‖ ≈ 0,005 par coordonnée, et Adam fait des pas de ≈ η = 0,01 par coordonnée quelle que soit l'échelle (P6) ; la norme doit croître jusqu'à ce que le pas relatif devienne modéré (auto-stabilisation [A1]) ;
- F5 (lecture) : la rétraction radiale garde ‖θ_h‖ à l'échelle du bord, donc un taux effectif d'Adam ≈ r_brut/ρ ≈ 4–6 fois plus grand que celui de P en fin de run (P6). Candidat au régime « marche aléatoire » [B1].

N'explique pas :
- le plateau **oscillant** (F3). Avec une perte lisse, la piste 1 prédit une descente lente et monotone, et un gradient qui tend vers 0 sous GD [A4]. Or la perte monte à ≈ 45 % des pas. Le pas d'Adam sur W (≈ 2e-3 par coordonnée) n'est pas assez petit pour un arrêt par lenteur ;
- la baisse de r brut entre t ≈ 50 et t ≈ 100–500 (F4). Sous GD, P5 l'interdit pour ‖θ_h‖ ; sous Adam, rien ne l'interdit (la mise à jour d'Adam n'est pas orthogonale à θ_h). r brut n'est pas ‖θ_h‖ et C bouge aussi : pas de conclusion ;
- F2 tel quel : P1–P6 portent sur θ ; F2 est mesuré en W, où l'échelle brute n'intervient plus.

Verdict : piste 1 = **mécanisme de structure** (absorption sous GD, taux effectif, rang), pas l'explication du plateau oscillant.

## 4. Piste 2 (non-différentiabilité) : l'argument propre à cette classe

- À C fixé, S est la boule (de rayon ρ) d'une jauge : somme de normes, de type « group lasso ». La jauge est non lisse dès qu'un terme est à un pli : bloc A_hk nul, bloc A_hk à valeurs singulières égales (‖·‖₂ d'une matrice ρ·rotation), B_hq nul, C_qk nul.
- **Le maître est exactement sur ces plis** : B̃·C = 0, A hors diagonale = 0, blocs diagonaux = ρ·rotation (σ₁ = σ₂). Les boules de type « group lasso » favorisent aussi des solutions à blocs nuls.
- Un pas constant près d'un pli ne peut pas atterrir sur le pli : il le traverse d'un pas à l'autre. La valeur oscille, et le (sous-)gradient choisi reste loin de 0 [C1]. C'est la signature F2 + F3.
- P8 : en un pli, la projection « lisse » de U1 peut annoncer un gradient tangent non nul en un point KKT. F2 peut donc être un **artefact de mesure**.

Prédictions (piste 2) au point final de P : (a) plusieurs termes de r_h à moins de δ d'un pli, δ de l'ordre du pas d'Adam sur W ; (b) le résidu KKT avec le sous-différentiel relâché à δ (§ 6) est petit devant ‖g‖, alors que le résidu lisse est ≈ ‖g‖ ; (c) des directions aléatoires ne sont pas absorbées par ce cône relâché.

## 5. Pistes 3 et 4

- Piste 3 (pas trop grand face à la courbure) : l'« équilibre chaotique » [B1] a la même signature, mais il a été observé sous SGD projeté en mini-lots, pas sous Adam plein lot. EoS [B2] et AEoS [B3] prédisent un progrès à long terme, pas un plateau sur 1000 pas. Non tranché par la théorie. Si le test du § 6 réfute la piste 2, la piste 3 devient le candidat principal (test suivant : reprendre P final à lr × 0,1 ; prédiction : la descente reprend).
- Piste 4 (projection + Adam) : F5 s'explique par P6 (taux effectif plus grand de la rétraction radiale), sans invoquer de non-convergence d'Adam. [D1] concerne le cadre stochastique : non applicable tel quel.

## 6. Vérification numérique — critère écrit avant calcul (2026-10-22)

But : trancher la prédiction (b) de la piste 2 : le point final de P est-il un point KKT « à δ près des plis » ?

Points : modèles P finaux de T4 (5 graines), **régénérés** (les `.pt` ne sont pas dans git) par `m2q.py t4` (bras p), vers un dossier séparé. W = Φ(θ) (`as_free`). Coordonnées contraintes : (A, B, C) ; non contraintes : D, E, b, bx, portes, lecture.

Mesures (par graine) :
- g = ∇_W L ; g_con (A, B, C), g_unc (le reste).
- Résidu KKT relâché : R_δ = min ‖g_con + Σ_h λ_h ξ_h‖ sur λ_h ≥ 0, ξ_h ∈ ∂_δ r_h(W), où ∂_δ r_h remplace le sous-différentiel de chaque norme ‖X‖ par {G : ‖G‖_dual ≤ 1, ⟨G, X⟩ ≥ ‖X‖ − δ} (P7 ; dual de ‖·‖₂ matricielle = norme nucléaire ; produit ‖B_hq‖·c_q par la règle du produit). Problème convexe (cvxpy, après le changement de variable Z = λ·G).
- Grille δ ∈ {0 ; 1e-4 ; 1e-3 ; 1e-2}. **δ de décision = 1e-2** (raison : le pas d'Adam par bloc sur W vaut ≈ 2·η·ρ/r_brut ≈ 4e-3 ; une oscillation de quelques pas autour d'un pli a une amplitude ≈ 1e-2).
- Recensement : nombre de termes à moins de δ d'un pli (norme < δ ; σ₁ − σ₂ < δ pour les blocs A).
- ‖g_unc‖ / ‖g_con‖ (rapporté ; un point KKT exige aussi g_unc ≈ 0).

Validité :
- v1 : la régénération reproduit T4 : perte finale et test égaux au JSON (écart relatif < 1e-5).
- v2 : à δ = 0, R_0 égale le résidu NNLS avec les normales de la dérivation automatique (écart < 1e-3·‖g_con‖).
- v3 (témoin) : 20 directions gaussiennes g̃ de même norme que g_con (en A, B, C) : médiane de R_δ(g̃)/‖g̃‖ ≥ 0,5 au δ de décision. Si v3 échoue à 1e-2, le δ de décision devient le plus grand δ de la grille où v3 tient (règle fixée ici, avant calcul).

Lecture (majorité = 3/5 graines, au δ de décision) :
- **piste 2 confirmée** (lecture provisoire) : R_δ/‖g_con‖ ≤ 0,1 ;
- **piste 2 réfutée** : R_δ/‖g_con‖ ≥ 0,5 ;
- sinon ambiguë.
Portée : un point KKT à δ près dit que le « gradient non nul » de U1 est un artefact des plis et que le plateau est une oscillation autour d'un point stationnaire du problème contraint. Il ne dit pas pourquoi ce point est mauvais (question Q0, hors session). Une réfutation dit qu'une direction de descente admissible existe à l'échelle δ : le plateau vient alors de la dynamique (piste 3).
Coût : régénération 5 runs ≈ 15 min (2 cœurs) ; test < 2 min.

## 7. Résultat du test KKT relâché (2026-10-22, `gp_kkt.py`, `results_gp/kkt.json`, `gp_kkt_log.txt`)

Validité : v1 tenue (5/5 : perte et test régénérés = JSON de T4 à 1e-6 près) ; v2 tenue (5/5 : R_0 = résidu NNLS) ; v3 tenue (témoin aléatoire, médiane 0,87–0,92 ≥ 0,5 à δ = 1e-2). δ de décision = 1e-2 (pas de repli).

| graine | R_δ/‖g_con‖, δ = 0 | 1e-3 | **1e-2** | témoin (1e-2) | termes à < 1e-2 d'un pli (A nuls / A σ₁≈σ₂ / B nuls / C nuls) | ‖g_unc‖/‖g_con‖ |
|---|---|---|---|---|---|---|
| 0 | 1,00 | 0,995 | **0,99** | 0,885 | 31 / 12 / 171 / 227 | 0,25 |
| 1 | 0,997 | 0,92 | **0,63** | 0,918 | 32 / 11 / 201 / 253 | 0,13 |
| 2 | 0,997 | 0,98 | **0,95** | 0,883 | 20 / 13 / 129 / 253 | 0,28 |
| 3 | 0,988 | 0,84 | **0,65** | 0,872 | 34 / 12 / 160 / 227 | 0,44 |
| 4 | 0,976 | 0,93 | **0,62** | 0,877 | 19 / 6 / 121 / 208 | 0,37 |
(sur 64 blocs A, 256 termes B, 256 termes C.)

Verdict pré-enregistré : **piste 2 réfutée** (5/5 graines ≥ 0,5). Lecture provisoire.
- Le point final est bien **près de nombreux plis** (prédiction (a) confirmée : 19–34 blocs A presque nuls, 121–201 termes B, 208–253 termes C sur 256 ; C est presque nul, donc le chemin tanh est presque fermé).
- Mais ces plis n'absorbent qu'une petite part du gradient (en norme) : 1 % à 38 % (contre 8 à 13 % pour une direction aléatoire). Le « gradient non nul » de U1 **n'est pas** un artefact des plis : une direction de descente admissible existe à l'échelle δ = 1e-2.
- Fait supplémentaire : le gradient des paramètres **sans contrainte** (D, E, biais, portes, lecture) vaut 13 à 44 % de ‖g_con‖. Même hors de toute contrainte, le point n'est pas stationnaire. Le plateau ne vient donc pas d'abord de la géométrie de S : il vient de la **dynamique** de l'optimiseur.

## 6b. Test de la piste 3 — critère écrit avant calcul (2026-10-22)

Question : le plateau oscillant vient-il d'un pas trop grand pour la courbure locale ?
Montage (`gp_lr.py`) : départ = P final régénéré (θ brut tel quel, donc mêmes W et même r brut). Adam neuf, clip 10, perte pleine, 1000 pas. Bras : **lr 1e-2** (témoin : même pas, Adam remis à zéro) et **lr 1e-3**. 5 graines.
Mesures : baisse relative b = (L₀ − L₁₀₀₀)/L₀ (perte d'entraînement) ; fraction de pas montants u sur [500, 1000).
Validité : le témoin lr 1e-2 reproduit le plateau : b < 0,10 et u ≥ 0,30 sur ≥ 3/5. Sinon, la remise à zéro d'Adam change le régime : pas de conclusion.
Lecture (majorité 3/5, bras lr 1e-3) :
- **piste 3 confirmée** : b ≥ 0,20 et u < 0,20 ;
- **piste 3 réfutée** : b < 0,05 (un pas plus petit n'aide pas : vallée plate, lenteur) ;
- sinon **partielle** (une part du plateau est l'amplitude de l'oscillation).
Coût : 10 runs × 1000 pas ≈ 5 min sur 2 cœurs.

## 7b. Résultat du test du pas (2026-10-22, `gp_lr.py`, `results_gp/lr.json`, `gp_lr_log.txt`)

| graine | Adam neuf, lr 1e-2 (témoin) : b / u / test | Adam neuf, lr 1e-3 : b / u / test | test de P (T4) |
|---|---|---|---|
| 0 | **0,98** / 0,48 / 0,059 | 0,16 / 0,10 / 0,142 | 0,124 |
| 1 | −0,02 / 0,52 / 0,327 | 0,04 / 0,46 / 0,314 | 0,340 |
| 2 | **0,43** / 0,39 / 0,202 | 0,24 / 0,29 / 0,225 | 0,292 |
| 3 | 0,03 / 0,45 / 0,266 | 0,04 / 0,43 / 0,255 | 0,288 |
| 4 | **0,77** / 0,42 / 0,062 | 0,47 / 0,17 / 0,119 | 0,248 |
(b = baisse relative de la perte d'entraînement en 1000 pas ; u = fraction de pas montants sur [500, 1000). Tous les runs restent sur ∂S : r(W) = 0,999.)

Verdict pré-enregistré : **invalide**. Le témoin devait reproduire le plateau (b < 0,10 sur ≥ 3/5) ; il ne le fait que sur 2/5. La piste 3 n'est ni confirmée ni réfutée par ce test.

Observation (non pré-enregistrée, lecture provisoire) : à θ identique (donc W, r brut identiques) et au **même pas** 1e-2, remettre à zéro l'état d'Adam (m, v, compteur) fait sortir 3 graines sur 5 du plateau, dont 2 de façon nette (graine 0 : perte 0,057 → 0,0009, test 0,124 → 0,059 ; graine 4 : test 0,248 → 0,062). Le modèle contraint, **sur ∂S**, peut donc atteindre une perte bien plus basse depuis le point du plateau. Le plateau dépend de l'**état interne de l'optimiseur**, pas seulement de (θ, lr).
Limites : (1) pas de bras « continuer avec l'état d'Adam d'origine » (T4 ne sauvegarde pas l'état d'Adam) ; T4 montre seulement que la perte était plate depuis t ≈ 1500–2000. (2) La remise à zéro mélange deux effets : l'oubli de v (mémoire de β₂ = 0,999 ≈ 1000 pas, gonflée par les pics de perte et par les gradients de début de run, plus grands quand ‖θ_h‖ était petit, P2) et le « coup de pied » des premiers pas d'un Adam neuf (≈ η·signe(g) sur chaque coordonnée).

## 8. Bilan : ce qui est expliqué, ce qui ne l'est pas

Expliqué (démontré, Lean `Ext/ScaleInvariance`, 16 énoncés, 0 `sorry`) :
- sur une tête saturée, le gradient en θ n'a pas de composante radiale ; la direction radiale est dans le noyau de J (déficit de rang de M1) ;
- sous GD, ∂S est **absorbant** pour `proj`, quel que soit le pas : une tête saturée assez loin du bord ne revient jamais dans S ;
- l'échelle brute agit comme un diviseur du pas : η/‖θ_h‖² sous GD, η/‖θ_h‖ sous Adam (conjugaison exacte, ε = 0, sans clip). Cela explique le saut initial de r brut et, en lecture, pourquoi la rétraction radiale fait pire.

Exclu (mesuré, critères écrits avant calcul) :
- le plateau n'est **pas** un point KKT caché par les plis (piste 2 réfutée, 5/5) ; une direction de descente admissible existe ;
- le point du plateau n'est même pas stationnaire dans les coordonnées **sans contrainte** (‖g_unc‖/‖g_con‖ = 0,13–0,44).

Pas encore expliqué :
- **pourquoi la perte reste plate alors que le gradient ne s'annule pas**. Les faits pointent vers la dynamique d'Adam (oscillation entretenue, perte qui monte à ≈ 45 % des pas ; sortie du plateau par simple remise à zéro de l'état sur 3/5). Candidats : mémoire de v gonflée par les pics (pas effectifs trop petits sur certaines coordonnées, trop grands sur d'autres), ou régime d'« équilibre chaotique » [B1] dont on sort par perturbation. Aucun n'est démontré ici.
- la baisse de r brut sous Adam (t ≈ 50 → 500) ; P5 l'interdit sous GD seulement.

Réponse courte à la question de la fiche : le « gradient non nul, perte plate » n'est **pas** une propriété de l'ensemble S ni de ses plis. La paramétrisation `proj` crée une structure (absorption, taux effectif ∝ 1/‖θ_h‖) ; le plateau lui-même est un **état de l'optimiseur** sur ∂S (lecture provisoire, 3/5).

## 9. Prochain test proposé (non lancé ; critère à pousser avant calcul)

But : séparer « mémoire de v » et « coup de pied ». Régénérer P (T4, bras p) en sauvegardant l'état d'Adam à t = 3000 (≈ 8 min). Depuis t = 3000, 1000 pas, lr 1e-2, 5 graines, bras :
- **cont** : état d'origine (témoin du plateau) ;
- **reset** : état remis à zéro (reproduit § 7b) ;
- **v-court** : v remplacé par g_t² (mémoire de v effacée), m et compteur gardés ;
- **kick** : état gardé, un seul pas η·signe(g) ajouté, puis Adam d'origine.
Validité : cont plat (b < 0,10 sur ≥ 3/5) ; reset reproduit § 7b (mêmes graines sorties, à 1e-6 près).
Lecture : v-court sort (b ≥ 0,3) et kick non → mémoire de v ; kick sort et v-court non → perturbation (équilibre chaotique) ; les deux → ambigu ; aucun → l'effet vient de m ou du compteur.
