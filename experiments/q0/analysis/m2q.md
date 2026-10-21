# M2-q — tests bon marché sur les blocs de 2 (oscillateurs, `blockp`)

Suite de `m2ter.md`. Décision de l'utilisateur (2026-10-21 soir) : lancer tous les tests bon marché, sur les blocs d'abord.

## Protocole et critères (écrits avant tout calcul)

Commun : banc et données de M2 (`m2_boundary.data(seed)`), graines 0 à 4, clip 10. W₀' = init de P-int (0,9ρ, ρ = 0,999). « Réussit » : test < 0,05 ; « échoue » : test > 0,2 (médiane sur 5 graines). Majorité = au moins 3 graines sur 5. Les lectures restent **provisoires** (un banc, une classe).

### T1 — par quel bloc passe la descente ? (GD η = 0,03 depuis W₀', jusqu'au contact)

À chaque pas avant le contact :
- part de ‖∇L‖² par paramètre (Ã, B̃, C, D, E, portes, biais, lecture) ;
- vitesse relative v_X = ‖∇_X L‖ / ‖X‖ pour X ∈ {Ã, B̃, C} ;
- dérivée des parts de r (Ã diag., Ã hors diag., B̃·C) de la tête de r max, le long de −∇L.
Moyennes sur les pas avant le contact.
- H1a « la descente passe d'abord par B̃ » : part de B̃ > 0,5 sur la majorité → vraie ; < 0,5 sur la majorité → fausse.
- H1b « B̃ est petit face à son gradient » : v_B̃ / v_Ã > 3 sur la majorité → vraie ; < 1 sur la majorité → fausse ; sinon ambiguë.
Lecture : deux hypothèses descriptives. Elles ne donnent pas de mécanisme.

### T2 — gradient projeté (même S, autre géométrie de mise à jour)

Bras **PGD** : équations libres (s ≡ 1), init W₀', Adam. Après chaque pas : (Ã_h·, B̃_h·) × min(1, ρ/r_h). r_h est homogène de degré 1 en ces lignes (C fixé), donc r_h ≤ ρ exactement après le pas.
Différence avec `blockp` : dans `blockp`, θ peut dériver hors de S (r brut > ρ) et le gradient passe par Φ (perte de la direction radiale). Avec PGD, θ reste dans S et le gradient est le gradient libre.
Limites : rétraction radiale, pas la projection euclidienne (le gradient projeté de Niculae) ; états d'Adam conservés (« Adam projeté »).
Lancé avec T4 (mêmes réglages). Lecture :
- PGD réussit et P échoue → même S, issue différente : l'échec de `blockp` ne vient pas de S seul, mais de la paramétrisation Φ sur ∂S. **Contre** « le bord de S est la cause ».
- PGD échoue → compatible avec « S est l'obstacle » (pour cette projection).
- Sinon : partiel ; je rapporte la part d'écart fermée g = (P − PGD)/(P − F').

### T3 — famille β sous GD (le manque de M2-ter)

P-int, GD η = 0,03, lr de B̃ = β·η, β ∈ {0,3 ; 0,1 ; 0,03}, arrêt au contact ou à 5000 pas. Script `m2bis.py` (option `--betas`). Règles et diagnostic de l'étape G de `m2ter.md`.
- V_G(β, graine) : perte en t_c − 1 ≤ 0,5·L₀ ; sans contact : perte finale ≤ 0,5·L₀ (sinon « budget »).
- β_G* = le plus grand β avec V_G sur la majorité.
- Régime de flot : graine 0, β = 0,03, η = 0,01 ; valide si η·t_c diffère de moins de 20 %.
Lecture :
- β_G* existe → sous GD, une métrique (même S) garde le flot dans S pendant qu'il apprend ;
- aucun β et contact sur la majorité → aucune métrique de la famille ne garde le flot GD dans S ;
- « budget » sur la majorité → non concluant.

### T4 — décroissance de la marge (idée de l'utilisateur, d'après Vorontsov et al. 2017)

`blockp` avec ρ_t à la place de ρ : s = ρ_t / max(r, ρ_t). Dans S_t = {r ≤ ρ_t}, Φ = identité : le run est exactement libre tant que r < ρ_t. À la fin, ρ_T = ρ : W ∈ S par construction.
Bras (Adam lr 1e-2, comme M2 ; 3000 itér. ; init W₀' pour tous) :
- **F'** : libre (m = ∞) ;
- **P** : ρ_t = ρ (m = 0 ; réplique P-int de M2) ;
- **PGD** : T2 ;
- **fix2** : ρ_t = 2 fixe (marge fixe, comme Vorontsov ; W final hors de S) ;
- **A-lent** : ρ_t = 100 pour t < 500, décroissance géométrique jusqu'à ρ en t = 2500, puis ρ ;
- **A-rapide** : ρ_t = 100 pour t < 1500, décroissance géométrique jusqu'à ρ en t = 2000, puis ρ.
Journal toutes les 50 itér. : perte, ρ_t, r max de W, r brut max, têtes saturées. Fin : test (T = 300), r max de W, état du modèle (pour T5).

Contrôles (sinon, pas de conclusion sur le bras en cause) :
- c0 : perte à t = 0 identique pour les 6 bras (W₀' est dans tous les S_t).
- c1 : A-lent, A-rapide et fix2 identiques à F' (perte, écart < 1e-6) tant que r brut < ρ_t.
- c-fin : P, PGD, A-lent, A-rapide finissent avec r max de W ≤ ρ + 1e-5.

Validité : F' médiane < 0,05 (sinon, pas de lecture de T4 : le libre n'apprend pas depuis W₀' avec ces réglages).

Lecture (meilleur des deux bras A ; la sélection sur 2 plannings est notée) :

| A | P | lecture provisoire |
|---|---|---|
| réussit | échoue | contraindre la fin, pas le chemin, atteint un bon modèle dans S : l'obstacle est le chemin dans S, pas l'absence de bons points atteignables dans S |
| échoue | échoue | la compression perd ce que le libre a appris : finir dans S est aussi dur que rester dans S |
| autre | — | partiel ; part d'écart fermée g = (P − A)/(P − F') |

Lecture secondaire (descriptive) : perte d'entraînement au début et à la fin de la décroissance ; fix2 réussit ou échoue (un certificat deux fois plus lâche suffit-il ?).

Prédictions (confiance faible) : F' réussit ; P échoue ; PGD échoue ; A partiel (0,05 à 0,2) ; fix2 partiel.

### T5 — le libre est-il contractant ? (sur les modèles finaux de T4)

Norme : d(x, x') = maxₕ ‖xₕ − x'ₕ‖₂ (celle du certificat). 32 séquences d'entrée de 2000 pas (`inputs(32, 2000, 30 + graine)`). Pour chaque séquence, 4 paires d'états initiaux à deux échelles : grande (x₀, x₀' ~ N(0, 1) indépendants) et petite (x₀' = x₀ + 1e-3·N(0, 1)).
Mesures : E = max sur paires et sur t de d_t / d₀ (expansion) ; médiane de d₂₀₀₀ / d₀.
- Contrôle T5 : P final (certifié, ρ = 0,999) doit donner E ≤ 1 + 1e-4. Sinon : erreur de code, pas de lecture.
- F' « non expansif en pratique » si E ≤ 1,05 sur la majorité ; « expansif » si E > 2 sur la majorité ; sinon ambigu.
Lecture :
- F' non expansif avec r ≈ 50 → le certificat est lâche d'un facteur > 50 sur la zone visitée : le bord en cause est celui du certificat, pas celui de la non-expansion.
- F' expansif → la solution libre sort vraiment de l'ensemble non expansif (pour cette norme seulement ; une autre norme peut contracter).
Même mesure, descriptive, pour fix2 et PGD.

### Ordre et coût

T1 (< 1 min) ; T3 (runs courts, arrêt au contact) ; T4 (30 runs × 3000 itér., ≈ 35–45 min sur 2 cœurs) ; T5 (< 2 min). Analyse : `analyze_m2q.py`, écrit avant les résultats.

## Incident : contrôle c1 en échec pour A-lent et A-rapide (premier passage)

- c1 exige que A-lent et A-rapide soient identiques à F' tant que r brut < ρ_t. Écart observé : 4,3e-3 (A-lent) et 8,9e-6 (A-rapide). Règle : pas de conclusion sur ces bras.
- Cause trouvée : pour certaines valeurs de ρ_t (non représentables en float32, par ex. t = 504), s = ρ_t / max(r, ρ_t) vaut 1 + 6e-8 au lieu de 1. Première différence de perte : t = 504–505 (A-lent), 1501–1502 (A-rapide), soit juste après le début de la décroissance, avec un écart de 1e-8 ; l'écart grandit ensuite (dynamique sensible).
- Correction : ρ_t arrondi en float32 avant usage (s = 1 exactement dans S_t). Seuls A-lent et A-rapide sont relancés (5 graines). Règles inchangées. Les résultats du premier passage sont gardés dans `results_m2q/invalide_c1/` (avec l'analyse d'avant correction) ; ils ne servent à aucune lecture.
- T5 est relancé ensuite pour tous les bras (F', P, PGD, fix2 inchangés : runs déterministes).

### Deuxième passage : correction sans effet, diagnostic corrigé

- La relance avec ρ_t arrondi en float32 a donné des résultats identiques au bit près à ceux du premier passage, et c1 a encore échoué (même écart). Mon diagnostic était incomplet : l'effet (s = 1 ± 1 ulp) était juste, le mécanisme faux.
- Vrai mécanisme : PyTorch calcule `ρ / tenseur` comme ρ × (1/tenseur). Le produit x·(1/x) n'est pas toujours exactement 1 en float32, même si ρ est représentable.
- Correction : classe `ExactProjBlock` (dans `m2q.py`), s = min(1, ρ_t/r). Dans S_t, s = 1 exactement ; hors de S_t, même Φ à 1 ulp près. Appliquée aux seuls bras A-lent et A-rapide. La bibliothèque n'est pas modifiée.
- Vérification avant la troisième relance (graine 0, A-lent, 700 pas) : écart de perte à F' = 0 sur tout le run, r brut jusqu'à 47,6.
- Résultats du deuxième passage : dans l'historique (commit `c609be0`), identiques à `results_m2q/invalide_c1/`. Journal : `m2q_rerun_log.txt`. Règles inchangées.

## T6 — le libre est-il équivalent en entrée-sortie à un point de S ? (critères écrits avant calcul ; question de l'utilisateur, 22:50)

Idée (utilisateur) : la perte ne voit que la sortie. Un changement de coordonnées x' = T·x, avec T bloc-diagonal (un bloc 2×2 par tête), garde exactement la relation entrée-sortie : A' = TAT⁻¹, B' = TB, C' = CT⁻¹ (entrée des tanh), E' = TE, bₓ' = T·bₓ, lecture' = lecture·T⁻¹ ; D, b et les portes ne changent pas (z par tête commute avec T ; x₀ = 0). Mais r change.
Modèles : les 5 modèles libres finaux de T4 (F', r = 56 à 64).
- **S1** (convexe) : T = diag(α_h·I₂). r_h(α) = Σₖ ‖A_hk‖ α_h/α_k + Σ_q ‖B_hq‖ α_h Σₖ ‖C_qk‖/α_k est un posynôme : min_α maxₕ r_h est convexe en log α. Optimum global, au lissage près (log-sum-exp avec continuation, puis évaluation exacte du max).
- **S2** (non convexe) : T_h général 2×2, optimisation locale (Adam sur un max lissé, départ en S1, 4 départs perturbés). Donne une borne supérieure du vrai minimum.
Contrôles (sinon, pas de lecture) :
- c-IO : après transformation, sortie identique : max|y − y'| / max|y| < 1e-9 (float64, entrées de test).
- c-r : avec T = I, r = r max de W stocké en fin de T4 (écart relatif < 1e-5).
Lecture (majorité ≥ 3/5 ; r* = minimum trouvé) :
- r*(S1) ≤ ρ → le libre est exactement équivalent en entrée-sortie à un modèle de S : l'obstacle est le **système de coordonnées du certificat** (métrique fixe P = I), pas S comme famille de fonctions.
- r*(S1) > ρ et r*(S2) ≤ ρ → même lecture, avec un changement de base général par tête.
- r*(S2) > ρ → les changements de coordonnées par tête ne suffisent pas. Restent : directions peu observables, borne grossière sur le chemin tanh, ou modèle vraiment hors de toute version de S. Je rapporte r*/ρ.
Limite : seules les transformations bloc-diagonales sont testées (celles qui commutent avec les portes par tête). Une couche tanh n'a pas de symétrie d'échelle (sauf signes et permutations, qui ne changent pas r).
Pas de prédiction ferme : r(T = I) ≈ 60 ; une réduction d'un facteur 60 par mise à l'échelle me semble peu probable (confiance faible).

### Résultat T6 (règles appliquées par `m2q_coords.py`) — `m2q_T6_analysis.txt`, `results_m2q/t6.json`, `m2q_T6_parts.txt`

Contrôles c-r et c-IO passés (erreur de sortie relative ≤ 3e-15 ; r(T = I) = valeur stockée).

| graine | r(T = I) | r*(S1, échelles) | r*(S2, 2×2 général) |
|---|---|---|---|
| 0 | 55,99 | 46,44 | 44,71 |
| 1 | 60,46 | 49,20 | 47,89 |
| 2 | 64,02 | 41,82 | 40,71 |
| 3 | 59,46 | 49,25 | 47,52 |
| 4 | 63,99 | 41,51 | 40,41 |

- **Ligne T6 : les changements de coordonnées par tête ne suffisent pas.** Le meilleur changement de base réduit r de 17 à 37 % seulement ; r* reste 40 à 49 fois au-dessus de ρ (0/5 graines dans S).
- Décomposition descriptive (`m2q_T6_parts.txt`) : à l'optimum S1, la part B̃·C vaut 37 à 45 à la tête de r max ; Ã hors diagonale 3 à 5 ; Ã diagonale < 1. La part B̃·C porte 85 à 90 % de r, et les changements de coordonnées la réduisent peu.
- Restent, par la table : directions peu observables, borne grossière sur le chemin tanh, ou modèle hors de toute version de S. Non testés.
- Note : un import par erreur a relancé T6 une seconde fois (calcul déterministe, même résultat).

## T7 — observabilité : grammiens empiriques et élagage (critères écrits avant calcul ; demande de l'utilisateur, 23:06)

Hypothèse (utilisateur) : la perte ne voit que la sortie. Le libre met de grands gains (grand r) dans des directions que la sortie voit peu ; ces directions ne servent pas à la relation entrée-sortie.
Modèles : libres finaux de T4 (5 graines) ; P et fix2 en référence pour (a) et (b).

(a) **Grammiens empiriques** (Lall, Marsden, Glavaški), différences centrées, balayage : amplitude c ∈ {1e-3 ; 1e-2 ; 1e-1 ; 1}, entrées ∈ {nominales (`inputs(32, 300, 40 + graine)`), nulles}.
- W_o[i, j] = (1/4c²) moyenne_lot Σ_t ⟨Δy_i(t), Δy_j(t)⟩, avec x₀ = ±c·e_i (16 directions).
- W_c = (1/4c²) Σ_m moyenne_lot Σ_t Δx_m(t) Δx_m(t)ᵀ, avec une impulsion ±c sur l'entrée m en t = 0 (en plus de l'entrée nominale).
- Valeurs de Hankel σ_i = √λ_i(W_c·W_o) ; n_eff = nombre de σ_i > 1e-3·σ₁. Descriptif ; le maître a 8 états.
(b) **Têtes** : énergie d'observabilité o_h = trace du bloc h de W_o ; corrélation de Spearman entre o_h et r_h (8 têtes), par graine. Lecture principale à c = 1e-2, entrées nominales ; les autres points du balayage sont rapportés pour la stabilité.
- H7a : Spearman < −0,5 sur la majorité → **soutient** (les têtes à grand r sont peu observables) ; > 0 sur la majorité → **contre** ; sinon ambiguë.
(c) **Élagage des têtes** (libre seulement) : glouton itératif sur la perte d'entraînement (on retire la tête dont le retrait coûte le moins), k = 0 à 7 têtes retirées. À chaque k : erreur de test, r max des têtes gardées.
(d) **Élagage des unités tanh** : même procédé sur les 32 unités cachées (colonne de B̃, ligne de C, D, b), k = 0 à 31.
- f = r(modèle complet) / min{ r max (gardé) : test ≤ 0,05 }, sur (c) et (d) (le meilleur des deux).
- H7b : f ≥ 10 sur la majorité → **soutient** : l'essentiel de r est dans des directions dont la sortie n'a pas besoin. Si en plus r max (gardé) ≤ ρ sur la majorité → le modèle utile est dans S. f < 2 sur la majorité → **réfute** (sans réentraînement) : r est porté par ce dont la sortie a besoin. Sinon : partiel.
Contrôles (sinon, pas de lecture) :
- g0 : sur le maître linéaire, W_o empirique = W_o analytique Σ_{t=1}^{T} (Aᵗ)ᵀCᵀC Aᵗ (écart relatif < 1e-6), aux 4 amplitudes.
- g1 : élagage k = 0 → erreur de test identique à celle de T4 (écart < 1e-9, float64).
Limites : modèle non linéaire à portes ; les grammiens empiriques dépendent de c et des entrées (d'où le balayage). Élagage sans réentraînement : un « réfute » est donc conservateur (un réentraînement pourrait récupérer). Un « soutient » est une preuve forte.
Prédiction (confiance faible) : H7b réfutée (f < 2), car la part B̃·C est grande sur toutes les têtes (moyennes 35 à 43).

## Résultats T1 à T5 (règles appliquées par `analyze_m2q.py` ; tous les contrôles passés après la 3ᵉ relance) — `m2q_analysis.txt`

- **T1** : H1a FAUSSE (part de B̃ dans ‖∇L‖² : 0,10 à 0,18 ; la lecture prend 0,72 à 0,86). H1b ambiguë (v_B̃/v_Ã = 1,8 à 3,6). Le long de −∇L, la part B̃·C monte (+0,35 à +1,03), les parts Ã bougent de moins de 0,05.
- **T3** : aucun β (0,3 ; 0,1 ; 0,03) ne garde le flot GD dans S : contact 15/15 ; perte au contact 0,55 à 0,86 L₀. Régime de flot valide (1,17 contre 1,16).
- **T4** (c0, c1, c-fin passés ; 5 graines, médianes du test) : F' 0,0018 ; P 0,288 ; PGD 0,769 ; fix2 0,032 ; A-lent 0,160 ; A-rapide 0,238.
  - Ligne T4 (meilleur bras A = A-lent, sélection sur 2 plannings) : **partiel, g = 0,45**. A-rapide : échoue (g = 0,17).
  - Perte d'entraînement d'A-lent (médiane) : 0,062 au début de la décroissance, 0,082 à la fin, 0,056 à t = 3000 (F' : 0,0004).
  - Ligne T2 : PGD échoue → compatible avec « S est l'obstacle » (pour cette projection).
  - Secondaire : fix2 réussit (0,032).
  - Prédictions : 4 sur 6 justes (fix2 prédit partiel, observé réussit ; A-rapide prédit partiel, observé échoue).
- **T5** (contrôle passé) : F' E = 1,6 à 2,5 → ambigu par la règle ; d₂₀₀₀/d₀ ≈ 1e-9 (convergence en pratique).

### Incident T7 : contrôle g1 en échec (premier passage)

- g1 : avec 0 élagage, test 2,2 à 3,0 au lieu de 0,0012 à 0,0023. Règle : pas de lecture.
- Cause : le script met le type par défaut en float64 ; le générateur aléatoire tire alors d'autres nombres. Le maître et les entrées (`teacher`, `inputs`) n'étaient pas ceux du banc. L'élagage était évalué sur un autre maître. Les grammiens étaient mesurés sur une autre réalisation de la même classe d'entrées.
- Correction : données générées en float32 puis converties en float64 (`f32`). Relance complète de T7. Règles inchangées. Premier passage gardé dans `results_m2q/invalide_T7/`.
- T6 n'est pas touché : il compare deux réalisations du même modèle sur les mêmes entrées (identité entrée-sortie), et r ne dépend pas des données.

### Résultat T7 (deuxième passage) — `m2q_T7_log.txt`, `results_m2q/t7.json`, `m2q_T7_lecture_conditionnelle.txt`

- g0 passé (écart ≤ 4e-13).
- **g1 en échec par la règle écrite** : écarts 1e-8 à 5e-8. C'est l'écart attendu entre le test de T4 (float32) et le même calcul en float64. **Le seuil 1e-9 était mal posé** (aucun calcul correct ne pouvait le passer). Par la règle : pas de lecture.
- **Écart au protocole, décidé après avoir vu les résultats** : seuil de g1 porté à 1e-6 (compatible float32). Avec ce seuil, les règles codées donnent :
  - H7a **CONTRE** : Spearman(o_h, r_h) = −0,02 ; +0,52 ; +0,81 ; +0,74 ; +0,43. Les têtes à grand r sont plutôt **plus** observables. Stable sur le balayage (amplitudes, entrées).
  - H7b **RÉFUTE** (sans réentraînement) : f = 1,06 ; 1,69 ; 1,04 ; 1,02 ; 1,37. Tout élagage qui garde le test ≤ 0,05 garde r ≥ 36. Retirer une seule tête fait passer le test à 0,15–0,64.
  - Descriptif : le libre a **8 modes de Hankel dominants**, puis une chute d'un facteur 107 à 384 (σ₈/σ₉) : la dimension entrée-sortie effective est celle du maître (8). P n'a pas cette rupture (σ₈/σ₉ ≈ 1–2) ; fix2 seulement sur 1 graine sur 5.
- Lecture (conditionnelle à l'écart sur g1) : l'hypothèse « le grand r vient de directions peu observables » est **réfutée sur ce banc**. Le grand r est porté par les têtes et les unités dont la sortie a besoin.
