# M2-ter — lr par bloc pour B̃ : même S, métrique J·D·Jᵀ changée (oscillateurs, `blockp`)

Suite de `m2bis.md` (levier « lr global » vide). Lire d'abord `m2_postmortem.md`.

## Protocole et critères (écrits le 2026-10-21, avant tout calcul)

### Question

À S fixe, un pas d'Adam réduit sur B̃ seul (lr_B = β·lr) garde-t-il P-int dans S pendant qu'il apprend ?
Un lr par bloc équivaut, sous Adam, à la paramétrisation B̃ = β·B̂ : la métrique induite J·D·Jᵀ change, l'ensemble S ne change pas. C'est le test de l'affirmation centrale le plus direct sur ce banc.

### Fait mesuré avant les critères (`m2ter_budget.py`, `m2ter_budget_log.txt`, graines 0 à 4)

- À W₀' (r = 0,899), la part B̃·C de r vaut 0,70 à 0,73, la part Ã 0,17 à 0,20. Le terme B̃·C occupe ≈ 80 % du budget.
- Le maître (`m2_teacher_on_boundary.py`) a une part B̃·C nulle (C interne = 0) et une part Ã ≈ ρ sur 4 têtes. Dans S, aller de W₀' au maître exige donc un **transfert de budget** : la part B̃·C doit baisser d'environ 0,7 pendant que la part Ã monte d'environ 0,8.
- Au départ, −∇L fait monter la part B̃·C sur les graines 0 et 1 (+0,35 et +0,42 par unité de pas GD), et la fait baisser sur les graines 2 à 4. La part Ã baisse ou reste stable (−0,04 à +0,02).
- |B̃|/|Ã| à W₀' : 0,48 à 0,57.

Conséquence pour le plan : un β faible freine le gonflement de B̃, mais aussi la baisse de B̃ nécessaire au transfert. Un seul β ne permet pas de conclure « ensemble » s'il échoue. D'où une grille, lue comme une courbe.

### Bras

- **P-int(β)** : `blockp`, init 0,9ρ (comme M2, M2-bis). Adam, lr = 1e-3 pour tous les paramètres, sauf B̃ : β·1e-3. Clip 10.
- **F'(β)** : mêmes équations, s ≡ 1, init W₀', même optimiseur. Identique à P-int(β) jusqu'au contact (contrôle c3).
β = 1 : données de la calibration M2-bis (lr 1e-3), non relancées.

### Mesures (comme M2-bis, plus le budget)

- t_c, perte, q (témoin d'échelle), g_GD, c3 : définitions de `m2bis.md`.
- Part Ã et part B̃·C de la tête de r max, à chaque pas jusqu'à t_c.

### Étape A — grille (graine 0, P-int, jusqu'au contact ou 2000 pas), β ∈ {0,5 ; 0,1 ; 0,03 ; 0,01}

0,5 ≈ |B̃|/|Ã| à W₀' (même déplacement relatif pour B̃ et Ã).
Règle fixée avant calcul :
- V(β) : perte en t_c − 1 ≤ 0,5·L₀. Sans contact en 2000 pas : perte en 2000 ≤ 0,5·L₀.
- β* = le plus grand β de la grille avec V(β).
- Lecture de la courbe 1 − L(t_c − 1)/L₀ en fonction de β (β = 1 : 0,03).

Prédiction écrite avant calcul : V faux à β = 0,5 (le déplacement relatif de B̃ ne baisse que d'un facteur 2 ; à β = 1, la perte n'a baissé que de 3 %).

Lecture prévue :
| résultat | lecture (provisoire) | suite |
|---|---|---|
| β* existe | une métrique (même S) garde P-int dans S pendant l'apprentissage : la métrique compte | étape B |
| aucun β, et part B̃·C encore ≥ 0,5 au contact | pas de transfert de budget, quel que soit le rythme de B̃ : l'obstacle est le chemin dans S depuis cette init (ensemble × init), pas le rythme de B̃ | décision : init avec budget dans Ã, ou marge du maître (roadmap) |
| aucun β, et part B̃·C < 0,5 au contact | transfert commencé, mais trop lent face au contact | décision |

Limite : une famille de métriques (β sur B̃) ne couvre pas toutes les métriques. « Aucun β » n'est pas une preuve pour l'ensemble.

### Étape B — seulement si β* existe (graines 0 à 4, P-int(β*) et F'(β*), 5000 itér.)

Validité V1, V2, c3 et table de lecture : celles de `m2bis.md` (étape 2), avec lr_B = β*·lr. Calcul local ≈ 25–30 min.

Calcul de l'étape A : local, 4 runs, ≤ 3 min chacun, ≈ 6 min sur 2 cœurs.
