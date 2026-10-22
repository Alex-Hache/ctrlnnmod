# Théorie — « gradient non nul, perte plate » sur ∂S (classe `blockp`)

Session du 2026-10-22. Fiche : `gradient_non_nul_perte_plate.md`. Bibliographie : `biblio_gradient_plat.md` (renvois [A1]…[E4]). Lean : `these-lean`, `TheseLean/Ext/ScaleInvariance.lean`.

## 0. Faits à expliquer (run P de T4 ; Adam lr 1e-2, β = (0,9 ; 0,999), clip global 10, perte pleine, pas de weight decay)

- F1 : les 8 têtes restent saturées (W sur ∂S) pendant les 3000 pas ; le maître est à l'intérieur (marge 0,009).
- F2 : au point final, le gradient en (Ã, B̃, C), projeté sur l'orthogonal des normales lisses ∇r_h, vaut 0,95 à 5,8 × ‖g(t = 0)‖ (U1). La composante normale est faible (‖g‖ ≈ ‖g_tan‖).
- F3 : la perte est plate en moyenne après t ≈ 1000–2000, mais **oscille** : sur [2000, 3000), elle monte à 42–50 % des pas ; variation médiane par pas 0,1–1,4 % ; pics de 4 à 69 % en un pas (journaux de T4, lus le 2026-10-22, sans nouveau calcul).
- F4 : r brut max passe de 0,90 à ≈ 4–5 en 10 pas, culmine à 5–8 vers t = 50, redescend vers 3–5 (t = 100–500), puis remonte lentement (4,1–5,9 à t = 3000).
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
Conséquence : **la direction radiale d'une tête saturée est dans le noyau de J**, et le gradient en θ n'a aucune composante radiale. C'est le déficit de rang de M1 (= nombre de têtes saturées), maintenant démontré en général.

**P2 (degré −1).** Si f(c·y) = f(y) pour y voisin de x (c > 0) et si f a le gradient g en x, alors f a le gradient c⁻¹·g en c·x.
Preuve : près de c·x, f = f ∘ (c⁻¹·) ; règle de la chaîne ; l'adjoint de c⁻¹·Id est c⁻¹·Id.

**P3 (Pythagore).** Si ⟨g, x⟩ = 0, alors ‖x − η·g‖² = ‖x‖² + η²‖g‖² [A1, A4].

**P4 (pas angulaire).** Si ⟨d, x⟩ = 0 et x ≠ 0, alors ‖(x+d)/‖x+d‖ − x/‖x‖‖ ≤ ‖d‖/‖x‖.
Preuve : avec a = ‖x‖, b = ‖d‖, n = ‖x+d‖ = √(a²+b²), le carré de la corde vaut 2 − 2a/n = 2b²/(n(n+a)) ≤ b²/a².
Avec P2 (c = ‖x‖, x̂ = x/‖x‖) : un pas de GD tourne la direction d'au plus η·‖∇f(x̂)‖/‖x‖². Le taux effectif de GD sur la direction est η/‖x‖² [A2, A3].

**P5 (GD : croissance de la norme et absorption).** Si chaque gradient est orthogonal à son itéré, alors ‖x_t‖² = ‖x_0‖² + η² Σ_{s<t} ‖g_s‖², donc ‖x_t‖ croît.
Absorption : soit N positivement homogène avec N(x) ≥ α‖x‖ (α > 0), et f localement invariante le long des rayons sur {N > ρ}. Si α‖x_0‖ > ρ, alors, pour **tout** pas η, toutes les itérations de GD vérifient N(x_t) > ρ.
Lecture pour `proj` (à C fixé, N = r_h, α > 0 tant que les c_q = Σ_k‖C_qk‖ sont minorés) : sous GD, une tête saturée assez loin du bord **ne revient jamais** dans S. Le gradient ne donne aucun signal vers l'intérieur (P1), et chaque pas éloigne θ_h de l'origine (P3). C'est F1 sous GD (flots GD de M2-ter et M2-q, T3 : 15/15 contacts sans retour).

**P6 (taux effectifs : conjugaison exacte).** Coordonnées (ι → ℝ), échelle s : ι → ℝ, s_i > 0, gradient G avec G(s⊙x) = s⁻¹⊙G(x) (vrai pour un bloc invariant par P2, s = c sur le bloc, 1 ailleurs).
- GD : l'itéré de GD depuis s⊙x₀ au pas η est s⊙(itéré de GD depuis x₀ au pas η/s²).
- Adam avec ε = 0, sans clip : la mise à jour m̂/√v̂ est inchangée si tous les gradients passés sont multipliés par un même c > 0. Donc l'itéré d'Adam depuis s⊙x₀ au pas η est s⊙(itéré d'Adam depuis x₀ au pas η/s).
Lecture : sur une tête saturée, l'échelle brute ‖θ_h‖ divise le pas : η/‖θ_h‖² sous GD, **η/‖θ_h‖ sous Adam** (version exacte de l'énoncé approché de [A2]). Adam est coordonnée par coordonnée : la conjugaison vaut bloc par bloc. Le clip global couple les blocs : s'il est actif, il casse la conjugaison (facteur variable dans le temps). ε = 1e-8 ne compte que si √v̂ est de cet ordre.

**P7 (ε-sous-différentiel d'une norme).** Pour N positivement homogène et sous-additive : g vérifie N(y) ≥ N(x) + ⟨g, y − x⟩ − ε pour tout y **si et seulement si** ⟨g, z⟩ ≤ N(z) pour tout z, et ⟨g, x⟩ ≥ N(x) − ε. (Sert au test du § 6.)

**P8 (pli : gradient tangent non nul mais point KKT).** Exemple : S = {|a| + |b| ≤ 1}, W = (1, 0), g = −(1 ; ½). La normale « lisse » prise par la dérivation automatique (signe(0) = 0) est (1, 0) ; la partie tangente de g vaut (0 ; −½) ≠ 0. Mais −g = (1 ; ½) ∈ N_S(W) = {λ(1, s) : λ ≥ 0, |s| ≤ 1} : W est un point KKT. Une projection sur la normale lisse peut donc annoncer un gradient tangent non nul en un point stationnaire.

## 3. Piste 1 (invariance d'échelle) : ce qu'elle explique, ce qu'elle n'explique pas

Explique :
- le déficit de rang de M1 (P1) ;
- sous GD, l'absorption par ∂S quel que soit le pas (P5) ;
- le saut initial de r brut (F4, 0,90 → 4–5 en 10 pas) : au départ, ‖θ_h‖ ≈ 0,005 par coordonnée, et Adam fait des pas de ≈ η = 0,01 par coordonnée quelle que soit l'échelle (P6) ; la norme doit croître jusqu'à ce que le pas relatif devienne modéré (auto-stabilisation [A1]) ;
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

## 7. Résultat du test

(à remplir après calcul)

## 8. Bilan : ce qui est expliqué, ce qui ne l'est pas

(à remplir après le § 7)
