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
