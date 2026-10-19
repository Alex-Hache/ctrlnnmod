# Les paramétrisations garanties servent-elles en boucle fermée et pour les world models ?

Trois expériences, 19 octobre 2026. Calcul : GitHub Actions (4 vCPU par job, 67 jobs) et une
machine à 2 vCPU. Toutes les données brutes sont dans `closed_loop/results_ci/` et
`world_model/results_ci/` ; les figures et tableaux dans les dossiers `analysis/`.

## Réponse courte

| Question | Réponse | Preuve principale |
| --- | --- | --- |
| Utile en boucle fermée ? | **Oui.** La garantie tient pour tout paramètre, et elle évite une perte de stabilité réelle sous erreur de modèle. | Exp. 1 : un MLP entraîné entre en **cycle limite** avec une charge ×2 ; le PB-REN converge dans 100 % des cas. |
| Passe à l'échelle ? | **Oui, en largeur et en horizon.** | REN explicite : coût 1,1× un GRU jusqu'à n = 256, 1,9× à n = 1024 ; le REN acyclique classique est 15× à 99× plus lent que le REN explicite. |
| Utile pour les world models ? | **Oui pour apprendre un contrôleur dans le modèle.** Neutre pour la seule précision de prédiction. | Exp. 3 : contrôleur appris dans le REN : erreur de suivi réelle −56 % (meilleur des 4 modèles) ; un GRU a eu une explosion de gradient (1,7·10¹²) et un contrôleur pire que rien. |

## Brique commune : `ExplicitREN` (nouveau, prouvé en Lean)

`ctrlnmod/models/discrete/explicit_ren.py`. REN en temps discret avec un gain incrémental ≤ γ et
une contraction de taux ᾱ, **pour tout paramètre**.

- Le REN « acyclique » usuel résout ses neurones un par un ($D_{11}$ strictement triangulaire).
  Ici $D_{11} = 0$ et $2\Lambda_{ii} = \sum_j |H_{22,ij}|$. Alors $2\Lambda - H_{22}$ est à
  diagonale dominante, donc ⪰ 0 (**Lemme 3.1.1 de la thèse**). Un pas coûte des produits
  matriciels, comme un GRU.
- **Antériorité.** $D_{11} = 0$ n'est pas nouveau : R2DN (Barbara, Wang, Manchester) l'utilise
  déjà, avec un réseau 1-Lipschitz en rétroaction. L'apport ici est le choix de Λ par dominance
  diagonale (sans résolution) et la preuve Lean, pas la structure $D_{11} = 0$.
- **Lean** (`these-lean/TheseLean/Ext/ExplicitREN.lean`, 0 `sorry`, axiomes standard) :
  - `explicit_ren_param` : $H = X^\top X + \varepsilon I + (K^\top K + VV^\top)/\gamma$ ⟹ $P \succ 0$ et l'inégalité quadratique.
  - `explicit_ren_step` : $V(\Delta x^+) + \|\Delta y\|^2/\gamma \le \bar\alpha^2 V(\Delta x) + \gamma\|\Delta u\|^2$.
  - `explicit_ren_gain` : $\sum_t \|\Delta y_t\|^2 \le \gamma^2 \sum_t \|\Delta u_t\|^2 + \gamma V(\Delta x_0)$.
  - `contraction_pow` : $V(\Delta x_t) \le \bar\alpha^{2t} V(\Delta x_0)$.
- Tests liés : `tests/lean/test_lean_guarantees.py::test_explicit_ren_*`.

Deux variantes, aussi prouvées en Lean (`Ext/LayeredREN.lean`, `Ext/SelectiveREN.lean`) :

- **`LayeredREN`** : $D_{11}$ strictement triangulaire par blocs (couches), dominance diagonale
  par bloc. Calcul couche par couche ; même borne de gain.
- **`SelectiveREN`** : $x^+ = (1-g)\,x + g\,f(x,u)$, une porte scalaire $g \in [g_{min}, 1]$
  par tête, choisie par l'entrée (comme Mamba-2). La contraction tient **pour toute suite de
  portes**, par convexité de la forme quadratique. Noyau Triton : `ctrlnmod/kernels/sren_scan.py`.

## Expérience 1 — boucle fermée : bras 2 axes

**Montage.** Bras plan 2 axes avec gravité. Contrôleur de base : PD + compensation de gravité.
Tâche : contourner un obstacle que le PD seul traverse (100 % de collisions). Couple résiduel
appris de trois façons :

- **MLP libre** : $u = K(x) + \pi_\theta(x)$ (retour d'état, choix RL par défaut) ;
- **PB + RNN libre** : $u = K(x) + M_\theta(\hat w)$, $\hat w$ = perturbation reconstruite ;
- **PB + REN** : même structure, $M_\theta$ = `ExplicitREN` de gain ≤ γ (performance boosting).

Entraînement par rétropropagation sur 3 s, 600 itérations, 5 graines (2 pour le balayage en γ).

**1a. Paramètres aléatoires, sans entraînement** (100 à 150 tirages, 16 conditions initiales, 30 s).

| Politique | écart max à la trajectoire du PD | couple max | cible atteinte |
| --- | --- | --- | --- |
| MLP, poids ×1 | 0,07 rad | 3 N·m | 88 % |
| MLP, poids ×3 | 0,52 rad | 26 N·m | **0 %** |
| PB + RNN libre, ×1 / ×3 | 0,02 / 0,40 rad | 9 / 80 N·m | 40 % / **0 %** |
| PB + REN, γ = 2000, paramètres ×3 | 0,18 rad | 119 N·m | **100 %** |
| PB + REN, γ = 2000, paramètres ×10 | 0,73 rad | 447 N·m | **100 %** |

À perturbation plus forte que le MLP, le PB-REN reste stable sur tous les tirages. C'est ce que
« garanti pour tout θ » veut dire : on peut explorer l'espace des paramètres (RL, grands pas,
recherche aléatoire) sans quitter l'ensemble stabilisant.

**1b. Horizon long.** Sur 30 s (10 × l'horizon d'entraînement), les 11 PB-REN entraînés (tous γ)
convergent exactement sur la cible (erreur finale 0,000). 3 PB-RNN libres sur 5 restent bloqués à
0,31–0,40 rad : la structure PB seule ne suffit pas, il faut un opérateur stable.

**1c. Erreur de modèle (charge du 2e segment × k), 120 s.**

| | ×1 | ×1,5 | ×2 | ×3 |
| --- | --- | --- | --- | --- |
| MLP qui réussit la tâche (2 graines) | converge | converge | **cycle limite** 0,11–0,13 rad, 0,4–0,5 rad/s | **cycle limite** 0,28–0,31 rad, 1 rad/s |
| PB-REN qui réussit la tâche (4 runs) | converge | converge | converge | converge |

Figure : `closed_loop/analysis/arm_limit_cycle.png`. Les MLP qui **échouent** la tâche restent
proches du PD et convergent : ce sont les politiques performantes qui perdent la stabilité.

**Limites de l'expérience 1.**
- Trouver le contournement dépend de la graine pour toutes les politiques (sans collision : MLP 2/5,
  PB-REN γ ≥ 200 : 4/7, plus 1 run à 12 % de collisions ; PB-RNN : 1/5).
- Le PB-REN converge plus lentement : à 3 s, aucun n'est à moins de 0,05 rad de la cible.
- γ petit (10, 50) borne l'énergie de la correction : la manœuvre devient impossible. γ règle le
  compromis performance / robustesse.
- Sous charge ×3, le PB-REN reste stable mais touche plus souvent l'obstacle (28–81 %) que le MLP
  (2–3 %) : sa correction est planifiée pour le modèle nominal.
- La garantie théorique du PB suppose un modèle exact. Sous erreur de modèle, la stabilité
  observée est empirique (un argument de petit gain demanderait γ·‖Δ‖ < 1).
- Simulation 2D, pas de robot réel.

## Expérience 2 — world model du Franka (MuJoCo)

**Montage.** Franka Panda (MuJoCo Menagerie), servos de position avec limites de couple, 50 Hz.
1000 trajectoires de 5 s pour l'entraînement, 50 trajectoires de 30 s pour le test. Le vrai robot
oublie son état initial en ≈ 1 s (écart 0,79 → 0,0012 → 0) : la contraction est le bon a priori.
Modèles : MLP 1 pas (type MBPO), MLP multi-pas, GRU, REN contractant. Rollout en boucle ouverte de
1500 pas depuis $o_0$. 2 graines (écart entre graines ≤ 0,01).

| Modèle (entraînement H) | NMSE@10 | NMSE@100 | NMSE@500 | NMSE@1500 |
| --- | --- | --- | --- | --- |
| MLP 1 pas (25) | 0,183 | 0,356 | 0,435 | 0,638 |
| MLP multi-pas (100 / 250) | 0,096 / 0,097 | 0,241 / 0,245 | 0,266 / 0,274 | 0,421 / 0,421 |
| GRU (100 / 250) | 0,094 / 0,097 | 0,231 / 0,228 | 0,249 / 0,308 | 0,387 / 0,467 |
| REN (25) | 0,104 | 0,268 | 0,375 | **0,941** |
| REN (100 / 250) | 0,089 / **0,078** | 0,241 / 0,232 | 0,255 / **0,247** | 0,407 / 0,390 |

- Entraîné sur un horizon long, le REN est le meilleur à court terme (−17 % à 10 pas) et à égalité
  avec le meilleur GRU à 30 s. Il s'améliore quand H augmente ; le GRU se dégrade à H = 250.
- **Résultat négatif :** aucun modèle ne diverge, même avec des actions hors distribution (×3,
  bruit blanc, bang-bang ×5). Le GRU est borné par ses portes. Sur ce robot bien amorti, la
  bornitude ne distingue pas les modèles.
- **Contraction ≠ précision :** un REN entraîné sur 25 pas est le pire à 30 s.

**Passage à l'échelle en largeur** (s / itération, batch 64, H = 100, 4 vCPU) :

| n | 32 | 64 | 128 | 256 | 512 | 1024 |
| --- | --- | --- | --- | --- | --- | --- |
| GRU | 0,038 | 0,050 | 0,073 | 0,139 | 0,373 | 1,30 |
| REN explicite | 0,045 | 0,053 | 0,078 | 0,160 | 0,572 | 2,43 |
| REN acyclique | 0,667 | 1,60 | 4,08 | 15,9 | — | — |

Le surcoût du REN explicite vient du calcul de $X^\top X$ et de $E^{-1}$ (O(n³), une fois par
séquence) et de 3× plus de paramètres à n égal.

## Expérience 3 — apprendre un contrôleur dans le world model

**Montage.** Suivi de références articulaires rapides (sinus 0,3–1,2 Hz) sur le Franka. Sans
politique, commander $a_t = q_{ref}(t+1)$ laisse une erreur de 0,188 rad (retard et saturation
des servos). Politique MLP apprise **uniquement dans le world model** (rétropropagation sur
H pas imaginés), puis déployée sur MuJoCo (20 références inédites de 15 s).

| World model | RMSE réelle (rad), H = 250 | pire run | réel / imaginé | gradient max |
| --- | --- | --- | --- | --- |
| MLP 1 pas | 0,121 (−36 %) | 0,128 | 0,55 | 1,2 |
| MLP multi-pas | 0,105 (−44 %) | 0,109 | 2,62 | 0,04 |
| GRU | 0,132 (−30 %) | **0,270 (pire que rien)** | 1,77 | **1,7·10¹²** |
| **REN contractant** | **0,083 (−56 %)** | **0,089** | 1,65 | 0,03 |

- Le REN donne le meilleur contrôleur réel, avec les deux horizons d'imagination (H = 50 : 0,096
  contre 0,113–0,125 pour les autres) et pour chaque paire de graines.
- Le GRU entraîné sur H = 100 (graine 1) : la perte descend jusqu'à 0,0024, puis le gradient
  explose à l'itération ≈ 1200 (10⁵, 10⁷, … 1,7·10¹²) malgré l'écrêtage ; le contrôleur final est
  pire que l'absence de contrôleur. Sur les 4 runs REN, le gradient ne dépasse jamais 0,03.
  Figure : `world_model/analysis/imagine.png`.
- Le MLP multi-pas est le plus « exploité » : erreur imaginée 0,040, erreur réelle 2,6× plus grande.
- Nuance : avec un GRU entraîné sur H = 250, l'écart avec le REN est faible (0,082 contre 0,078).

## Expérience 4 — REN sélectif contre Mamba-2 (Franka, proprioceptif)

Même protocole que les expériences 2 et 3. Nouveaux modèles : `SelectiveREN` (8 têtes) et un
bloc Mamba-2 (PyTorch, égal au bloc HF à 1e-14, `tests/test_mamba2_reference.py`). 2 graines.

| Modèle (H = 250, n = 128) | NMSE@10 | NMSE@250 | NMSE@1500 | contrôleur appris : RMSE réelle (rad) |
| --- | --- | --- | --- | --- |
| REN contractant | **0,078** | **0,223** | **0,390** | **0,078** |
| GRU | 0,097 | 0,241 | 0,467 | 0,082 |
| REN sélectif | 0,087 | 0,252 | 0,418 | 0,102 |
| REN sélectif, n = 256 | 0,087 | 0,244 | 0,413 | 0,093 |
| Mamba-2 | 0,119 | 0,264 | 0,491 | 0,091 |

Sans politique : 0,188 rad.

- **Résultat négatif.** La sélectivité n'aide pas ici. Le REN sélectif prédit mieux que Mamba-2,
  mais le contrôleur appris dans Mamba-2 est meilleur (0,091 contre 0,102). Le REN plein reste
  le meilleur sur les deux critères.
- Explication probable : le Franka sous asservissement de position est presque invariant dans
  le temps. Une porte choisie par l'entrée n'a rien à sélectionner.
- Coût CPU par itération (batch 64, H = 100, 4 vCPU), `results_ci/scaling_exp4.json` :
  à n = 1024, REN sélectif 0,26 s, GRU 0,66 s, REN explicite 1,13 s, Mamba-2 1,97 s.
  Les têtes indépendantes rendent le REN sélectif le moins cher en grande largeur.

## Ce que ces expériences ne prouvent pas

- Pas encore d'images : world models proprioceptifs seulement. L'étape M3a (images 64×64 du
  Franka, encodeur CNN, 4 dynamiques latentes, GPU L4) est en cours : `experiments/pixel_wm/`.
- Pas de robot réel ; MuJoCo et un bras 2D simulé.
- Peu de graines (2 à 5) ; les conclusions s'appuient sur des écarts nets et cohérents, pas sur des tests statistiques.
- Le passage à l'échelle est mesuré sur CPU ; sur GPU les constantes changent, pas les ordres.
- L'a priori de contraction convient à un robot sous asservissement de position. Il est faux pour
  un système qui garde la mémoire de son état (objet libre, intégrateur pur).

## Suites possibles

1. World model latent sur images (encodeur + REN), même protocole que l'expérience 3.
2. Théorème de petit gain incrémental en Lean pour la boucle PB sous erreur de modèle (donne la borne sur γ).
3. γ appris avec une pénalité : estimer le gain nécessaire à partir des données (balayage déjà possible).
4. MPC avec le world model REN sur le Franka.

## Reproduire

```bash
# local
PYTHONPATH=.:experiments/closed_loop python experiments/closed_loop/run_arm.py train --kind pb_ren --seed 0
PYTHONPATH=.:experiments/world_model python experiments/world_model/wm.py --model ren --H 250
# distant (gratuit) : écrire compute/jobs.json puis pousser sur la branche `compute` ;
# les résultats arrivent sur la branche `compute-results`.
```
