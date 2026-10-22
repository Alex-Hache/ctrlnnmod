# M2-t — « la contrainte ferme la porte » ? (oscillateurs, `blockp`)

Écrit le 2026-10-22 vers 07:40, avant tout calcul. Suite de `m2s.md`.

## Hypothèse

Le libre apprend par un détour : B̃·C monte d'abord, puis la solution se transforme ; ce détour sort de S (T3). Le contraint suit le même début, touche ∂S avec un budget dépensé dans B̃·C, puis ne peut plus transférer ce budget vers Ã (où est la solution du maître : B̃·C = 0, Ã ≈ ρ), parce que ce transfert fait monter la perte. Il y aurait dans S une **barrière de perte** que le libre contourne par l'extérieur.
Remarque : pour C fixé, S est convexe en (Ã, B̃) (r est convexe). Une barrière viendrait de la perte, pas de la forme de S.

## Protocole et critères (écrits avant tout calcul)

Commun : banc de T4, 5 graines, majorité = 3/5, réussit < 0,05, échoue > 0,2 (médiane du test). Points de départ : modèles P finaux de T4 (`results_m2q/t4_p_s*.pt`). Lectures provisoires.

### U1 — plateau ou lenteur ? (P final, sans entraînement)
- Pente : (L(2500) − L(3000)) / L(2500) (perte d'entraînement de T4).
- Gradient tangent : g_W = ∇_W L au point final, projeté sur l'espace tangent de ∂S (orthogonal aux normales ∇_W r_h des têtes saturées, en (Ã, B̃, C)). Rapport ρ_g = ‖g_tan(fin)‖ / ‖g(t = 0)‖.
- Lecture : « quasi stationnaire » si ρ_g < 1e-2 et pente < 1 % sur la majorité ; « descente lente » si pente > 5 % sur la majorité ; sinon ambigu.

### U2 — barrière sur un chemin dans S (sans entraînement)
- Alignement de P sur la représentation du maître (`m2s.teacher_on_boundary`) : chaque oscillateur i est associé à la tête de P dont les états (entrées d'entraînement) s'ajustent le mieux à ceux de l'oscillateur (affectation gloutonne, R² de la régression), puis rotation orthogonale 2×2 par tête (Procrustes). Permutation et rotations par tête ne changent ni la sortie ni r (contrôle u-IO : sortie identique, écart relatif < 1e-5 ; r identique, < 1e-5).
- Chemin : θ(λ) = Φ((1 − λ)·θ_P' + λ·θ_maître), λ ∈ {0 ; 0,05 ; … ; 1} ; perte d'entraînement L(λ).
- Lecture : « barrière » si maxλ L(λ) > 1,1·L(0) sur la majorité ; « pas de barrière sur ce chemin » si L(λ) décroît (tolérance : chaque pas ≤ 1 % au-dessus du précédent) sur la majorité ; sinon ambigu.
- Limite : un seul chemin. Une barrière sur ce chemin ne prouve pas qu'il n'existe aucun chemin sans barrière. L'absence de barrière réfute en revanche « porte fermée » au sens d'une barrière.

### U3 — forcer le transfert (entraînement)
- P-int (init W₀'), Adam lr 1e-2, 3000 itér., perte + λ·moyenneₕ(part B̃·C de W), λ ∈ {0,03 ; 0,3}. Erreur de test sans la pénalité.
- Lecture : un λ réussit → forcer le transfert ouvre la porte (le gradient seul ne le fait pas). Les deux échouent → le transfert forcé ne suffit pas. Sinon partiel.

### U4 — rouvrir la porte depuis l'état bloqué (entraînement)
- Départ : P final. 3000 itér. de plus, Adam neuf, lr 1e-2.
  - **cont** : ρ fixe.
  - **reopen** : ρ_t = 2 pendant 500 pas, décroissance géométrique jusqu'à ρ entre t = 500 et 1500, puis ρ jusqu'à 3000.
- Contrôle : r max de W ≤ ρ + 1e-5 en fin (les deux).
- Lecture : reopen réussit et cont échoue → rouvrir suffit ; médiane(reopen) ≤ médiane(cont) − 0,05 → aide partielle ; sinon pas d'effet.

Prédictions (confiance faible) : U1 quasi stationnaire ; U2 barrière ; U3 partiel ; U4 aide partielle.
Coût : U1, U2 < 5 min ; U3, U4 : 20 runs ≈ 45 min. Lancés après M2-s.

### Précision avant calcul (test de fumée de 20 pas, dossier temporaire)
- U4 : le θ brut de P est hors de S (r brut 4 à 6) ; avec ρ_t = 2, Φ ferait sauter W à r = 2 au pas 0 (divergence observée au test de fumée). L'intention est de relâcher la borne depuis l'état bloqué. Donc, pour cont et reopen : θ est d'abord remis à W_P (même fonction, r = ρ), puis l'entraînement reprend. Règles inchangées.
