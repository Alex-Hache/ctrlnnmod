# Initialisation H∞ et conditionnement : résultats

Script : `experiments/init_conditioning.py` (300 époques, 5 graines, Adam lr = 1e-2, float64).
Figures et tableau : `experiments/plot_init_conditioning.py` → `init_conditioning.png`.

## Protocole

- **Système vrai** : exemple (3.78) de la thèse, $A = \begin{pmatrix}-2 & 3\\0 & -1.5\end{pmatrix}$,
  $B = (0, 1)^\top$, $C = (1, 0)$, $\gamma^* = \|G\|_\infty = 1$.
- **Données** : 20 séquences × 500 points, $dt = 0{,}01$, sommes de sinus, bruit à 35 dB.
  Plancher de bruit : MSE = 1,57e-4.
- **Modèle** : `L2BoundedLinear(param='sqrtm')` (variante $M_s$), ou paramètres libres
  $(A, B, C, P)$ avec barrière $-\mu \log\det(-\text{BRL})$ ; dans ce cas, un pas qui sort du
  domaine est annulé, puis le pas est divisé par 2 (backtracking).
- **Scénarios d'initialisation** :
  - S1 : système vrai avec $B \times 0{,}5$ et $A$ bruité (gain initial < $\gamma^*$), γ prescrit = 1,5.
  - S2 : système stable aléatoire de faible gain, γ = 1,5.
  - S1t : comme S1, avec une contrainte serrée γ = 1,05 (optimum près du bord).
- **Mesures** : MSE, $\lambda_{\min}(M_s)$ (paramétrisation) ou $\lambda_{\min}(-\text{BRL})$
  (barrière), cond(P), pas rejetés.

## Résultats (médiane sur 5 graines)

| Scénario | Méthode | γ effectif | MSE finale / plancher | min λ_min | pas rejetés (total) | ‖G‖∞ final |
|---|---|---|---|---|---|---|
| S1 | ancienne init | 0,529 | 375 | 5,8e-4 | — | 0,526 |
| S1 | **init exacte** | 1,500 | **1,0** | 1,7 | — | 1,001 |
| S1 | barrière μ=1e-3 | 1,500 | 1,1 | 1,6e-1 | 0 | 0,993 |
| S1 | barrière μ=1e-5 | 1,500 | 1,0 | 1,4e-1 | 0 | 0,998 |
| S2 | ancienne init | 0,052 | 2 824 | 1,7e-3 | — | 0,051 |
| S2 | **init exacte** | 1,500 | **1,6** | 1,5 | — | 0,990 |
| S2 | barrière μ=1e-3 | 1,500 | 80 | 1,2e-3 | 29 | 1,115 |
| S2 | barrière μ=1e-5 | 1,500 | 26 | 8,9e-6 | 298 | 1,025 |
| S1t | ancienne init | 0,529 | 375 | 5,8e-4 | — | 0,526 |
| S1t | init exacte | 1,050 | 4,3 | 4,5e-1 | — | 0,946 |
| S1t | barrière μ=1e-3 | 1,050 | 2,7 | 1,3e-3 | 22 | 0,969 |
| S1t | barrière μ=1e-5 | 1,050 | **2,1** | 8,1e-6 | 345 | 1,000 |

Les λ_min ne se comparent pas directement entre méthodes : $M_s$ pour la paramétrisation,
$-\text{BRL}$ pour la barrière. Les deux mesurent la distance au bord.

## Conclusions

1. **L'ancienne init est le problème principal, et ce n'est pas surtout un problème de
   conditionnement.** Elle remplace γ par le gain du système initial. Si ce gain est inférieur
   au gain vrai, le vrai système est **hors de l'ensemble** : MSE 375 à 2 800 fois le plancher,
   sur toutes les graines. Le modèle reste aussi collé au bord : λ_min(M_s) décroît de 3e-3 à 6e-4.
2. **Avec l'init exacte, la paramétrisation $M_s$ ne voit jamais le bord.** λ_min(M_s) reste
   supérieur à 0,2 sur tous les runs. cond(P) reste inférieur à 15. Il n'y a ni NaN ni pas rejeté.
   Le modèle atteint le plancher de bruit sur S1 et S2.
3. **La barrière logdet confirme ton intuition.** Dès que l'optimum est près du bord (S2, S1t),
   **tous** les runs ont des pas qui sortent du domaine : 22 à 345 au total. Sans backtracking,
   le logdet devient non défini et chacun de ces runs plante. Plus μ est petit, plus le modèle
   s'approche du bord (λ_min ≈ 1e-5) et plus il y a de rejets.
4. **Coût de la paramétrisation directe** : sur S1t (contrainte active), elle converge un peu
   plus lentement que la barrière (4,3 contre 2,1 fois le plancher à 300 époques). Elle ne
   demande ni projection ni backtracking.

## Limites

- Système linéaire de dimension 2, une seule cible. À refaire sur un GNSSM et sur un
  benchmark réel.
- 300 époques : S1t n'a pas fini de converger.
- La barrière démarre au même point (centre de la LMI, via cvxpy) que l'init exacte. Ce qui est
  testé, c'est donc la dynamique d'entraînement, pas l'initialisation de la barrière.
