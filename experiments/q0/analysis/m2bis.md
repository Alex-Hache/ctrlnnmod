# M2-bis — P-int et libre depuis la même init intérieure, à petit pas d'Adam (oscillateurs, `blockp`)

Lire d'abord `m2_postmortem.md`.

## Protocole et critères (écrits le 2026-10-21, avant tout calcul)

### Question

À un pas d'Adam plus petit que l'échelle des paramètres :
- P-int reste-t-il dans S, et réussit-il ?
- Le libre reste-t-il dans S ?
- Si un run touche ∂S : la cause est-elle l'échelle du pas, l'orientation du pas d'Adam, ou le gradient de la perte ?

Limite connue : pour `blockp`, Φ = identité dans S, donc J = I. M2-bis ne mesure pas la métrique JJᵀ. Il mesure si le run doit toucher ∂S, à optimiseur égal.

### Écarts à la recommandation du post-mortem (validés le 2026-10-21)

1. Un seul pas ne prouve pas qu'un lr garde P-int dans S. Le gradient est en lot complet : les signes d'Adam restent stables d'un pas à l'autre, et les pas s'additionnent. Le contact se lit sur la trajectoire.
2. L'échelle de B̃ vient de S (r contient Σ_q ‖B_hq‖·c_q, c_q ≈ 2,6, 32 termes), pas seulement de l'init. Un lr global plus petit ralentit aussi Ã et C. D'où une condition de budget (V2).
3. Le libre part de W₀' (init de P-int), pas de W₀. W₀ est sur ∂S (marge −1e-7) : en M2, la sortie de S du libre était triviale.
4. Le témoin d'échelle porte sur le déplacement cumulé, pas sur un pas isolé (voir Mesures).

### Bras (16 états, 8 têtes de 2 à porte, ρ = 0,999 ; Adam, clip 10 comme M2)

- **P-int** : `blockp`, (Ã_h, B̃_h) × 0,9ρ/r_h, comme M2. W₀' = Φ(θ₀') = θ₀'.
- **F'** : mêmes équations, s ≡ 1, init W₀'.

Dans S, Φ = identité et le gradient de clamp est nul. P-int et F' font donc le même run jusqu'au premier contact t_c.
REN : pas relancé. Référence M2 : 0,0051 (lr 3e-3).

### Mesures (à chaque pas jusqu'à t_c inclus, puis toutes les 10 itér.)

- r_h brut sur (Ã, B̃, C) ; marge m(t) = ρ − maxₕ rₕ. t_c = premier pas avec m < 1e-5.
- **Témoin d'échelle.** Δ(t) = θ_t − θ₀' sur (Ã, B̃, C). Pour 20 masques fixes σ_j ∈ {±1} (tirés une fois par run) : r_max(θ₀' + σ_j ⊙ Δ(t)). Même |déplacement| par coordonnée, même histoire des signes dans le temps, orientation aléatoire. q = part des 20 masques avec r_max ≥ ρ − 1e-5 à t_c.
  Pourquoi pas un témoin pas par pas : il ne voit pas le cumul de pas de même signe (écart 1).
- **Direction du flot GD.** g_GD(t) = ⟨∇rₕ*, −∇L⟩ / ‖∇L‖ sur (Ã, B̃, C), h* = tête de r max. Signe > 0 : le flot libre GD pousse cette tête vers ∂S. Je lis la moyenne sur les 10 points avant t_c (it ∈ [t_c − 10, t_c − 1]).
- Perte d'entraînement à chaque pas ; erreur de test à la fin (T = 300), comme M2.

### Contrôles (sinon, ne pas conclure)

- c0 : r_h calculé par le script = `_block_rows()` (écart < 1e-6) ; marge initiale = 0,1ρ.
- c3 : P-int et F' de même (graine, lr) : pertes et r_max identiques pour t < t_c, et même t_c (écart max < 1e-6).

### Cause du contact (lue à t_c)

- q ≥ 0,5 : **échelle** (la taille du déplacement suffit, quelle que soit l'orientation).
- q = 0 et g_GD > 0 : **gradient** (le flot libre sort de S : terme de bord, T3).
- q = 0 et g_GD ≤ 0 : **orientation d'Adam** (préconditionneur D, pas la perte).
- Autre cas : **mixte**.

### Étape 1 — calibration (graine 0, 300 pas, P-int et F', lr ∈ {1e-3 ; 3e-4 ; 1e-4})

Règle de choix, fixée avant calcul. Elle n'utilise pas l'erreur de test, ni « la plus faible perte » (erreur 5 du post-mortem).
- L₀ = perte initiale. V(lr) : perte de P-int en t_c − 1 ≤ 0,5·L₀ (P-int apprend avant de toucher ∂S). Sans contact en 300 pas : j'utilise L(300). Sans contact et L(300) > 0,5·L₀ : je prolonge ce lr à 2000 pas et j'applique la même règle.
- lr* = le plus grand lr de la grille avec V(lr).
- Aucun lr valide : arrêt, pas d'étape 2. Je rapporte la cause du contact. Décision avec l'utilisateur (lr par bloc, ou autre).
- Si lr* < 1e-3 : je demande avant l'étape 2 (budget probablement insuffisant).

### Étape 2 — run (graines 0 à 4, lr*, N = 5000 itér.)

N = 5000 : à lr 1e-3, le déplacement cumulé max par coordonnée vaut 5, comme F en M2 à 500 itér. (perte train 0,06). Calcul local (pas de Modal dans ce conteneur) : 10 runs, ≈ 25–30 min sur 2 cœurs.

Validité :
- **V1** (P-int teste un run intérieur) : pour chaque graine, L(t_c − 1) ≤ 0,5·L₀, ou pas de contact. Si V1 échoue sur ≥ 3 graines : pas de conclusion sur P-int.
- **V2** (budget) : médiane de F' < 0,05. Sinon : pas de conclusion.
- c3 sur chaque graine.

Critères (erreur de test, médiane sur 5 graines ; réussit < 0,05 ; échoue > 0,2 ; cause = majorité des graines) :

| F' | P-int | cause du contact | lecture |
|---|---|---|---|
| réussit, m > 0 tout le run | réussit (même run) | pas de contact | à petit pas, S ne gêne pas ; le contact de M2 venait de l'échelle du pas d'Adam |
| réussit | réussit, avec contact | — | contact sans dommage à petit pas |
| réussit, sort de S | échoue | gradient | bord (T3) : le chemin libre sort de S même à petit pas ; l'ensemble compte |
| réussit, sort de S | échoue | échelle ou orientation d'Adam | encore l'optimiseur ; suite : lr par bloc |
| échoue | — | — | V2 non atteinte ; pas de conclusion |

Autre cas : ambigu. Toute lecture reste « provisoire » dans `ROADMAP.md` et `Contexte` (Adam seulement ; GD non valide en M2).

## Étape 1 — calibration (faite le 2026-10-21 ; graine 0, 300 pas) — `m2bis_calib_log.txt`, `results_m2bis/`

Contrôles :
- c0 passé : écart r / `_block_rows()` = 6e-8 ; marge initiale 0,0999 ; W₀' identique pour les deux bras ; perte initiale identique (0,8886).
- c3 passé : pertes et r_max identiques (écart 0) pour t < t_c ; même t_c ; à t_c, r_max identique et la perte diffère (P-int est projeté). Ma première lecture de c3 incluait t_c (écart 1e-4) ; le protocole dit t < t_c.

| lr | t_c | t_c·lr | L(t_c−1)/L₀ | V(lr) | q | g_GD (pts avant t_c) | cause |
|---|---|---|---|---|---|---|---|
| 1e-3 | 3 | 3,0e-3 | 0,970 | non | 0,85 | +0,53 | échelle |
| 3e-4 | 9 | 2,7e-3 | 0,964 | non | 0,55 | +0,81 | échelle |
| 1e-4 | 25 | 2,5e-3 | 0,964 | non | 0,35 | +1,13 | mixte |

**Règle : aucun lr valide. Arrêt, pas d'étape 2.**

Diagnostic ajouté (`m2bis_blocks.py`, `m2bis_blocks_log.txt`, < 1 min) : r_max de θ₀' avec un seul bloc déplacé à θ_tc.

| lr | r_max à t_c | B̃ seul (réel / méd. aléatoire) | Ã seul | C seul | \|Δ\|/\|θ₀'\| (B̃, Ã, C) |
|---|---|---|---|---|---|
| 1e-3 | 1,033 | 1,016 / 1,018 | 0,917 | 0,900 | 0,53 ; 0,30 ; 0,01 |
| 1e-4 | 0,999 | 0,992 / 0,990 | 0,913 | 0,900 | 0,44 ; 0,25 ; 0,01 |

Lecture (provisoire) :
- Le contact arrive au même point du chemin d'Adam, quel que soit lr : t_c·lr ≈ 2,5 à 3e-3 par coordonnée, et la perte n'a baissé que de 3 à 4 %. Un lr global plus petit ne change pas le chemin ; il ajoute des pas. Le levier « lr global » est vide.
- La recommandation du post-mortem (« lr 1e-3 : pas de contact ») reposait sur un seul pas. Le contact arrive au pas 3.
- B̃ porte ≈ 90 % de la hausse de r. Le déplacement réel de B̃ et des orientations aléatoires de même |Δ| donnent le même r : c'est la **taille** du déplacement de B̃ (≈ 50 % de |B̃|), pas son orientation. Ã contribue +0,015 ; C rien.
- g_GD > 0 dès le pas 1 sur la tête 2 : le gradient pousse aussi cette tête vers ∂S. Le témoin par bloc montre que cette orientation n'est pas nécessaire au contact.
- Après t_c (observation hors critères) : F' sort loin de S (r_max 3,9 à 16,5 à 300 pas) ; P-int reste sur ∂S (6 à 8 têtes).

Suite : décision avec l'utilisateur. Levier prévu : lr par bloc pour B̃ (équivalent à B̃ = β·B̂ sous Adam). Il change la métrique induite J·D·Jᵀ à S fixe : test direct de l'affirmation centrale. Critères à écrire avant calcul.
