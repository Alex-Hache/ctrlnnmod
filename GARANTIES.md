# Garanties prouvées

Chaque paramétrisation de `ctrlnmod` repose sur un théorème. Ces théorèmes sont prouvés en
Lean 4 dans le dépôt [these-lean](https://github.com/Alex-Hache/these-lean)
(62 énoncés de la thèse + 6 énoncés d'extension, aucun `sorry`). Les tests `tests/lean/test_lean_guarantees.py` vérifient
numériquement, sur des paramètres aléatoires, que le code respecte chaque énoncé.

```bash
pytest tests/lean
```

## Correspondance code → théorème

| Classe | Garantie | Théorème Lean (`These.…`) | Test | Statut |
| --- | --- | --- | --- | --- |
| `geotorch_custom.AlphaStable` (`SSLinear(alpha=…)`) | $\Re\lambda(A) < -\alpha$ | `thm_3_3_5_corrected`, `thm_3_3_6` | `test_alpha_stable_sound` | conforme |
| `L2BoundedLinear(param='sqrtm')` | $A$ Hurwitz, $\|G\|_\infty < \gamma$ | `ms_variant_sound` | `test_hinf_sqrtm_sound` | conforme |
| `L2BoundedLinear(param='sqrtm')`, initialisation | inverse à droite exact, γ gardé | `ms_variant_complete` | `test_hinf_sqrtm_right_inverse_exact`, `test_hinf_sqrtm_keeps_prescribed_gamma` | conforme (`method='exact'`, défaut) |
| `L2BoundedLinear(param='riccati')` | Riccati avec $P \succ 0$ | `thm_3_3_8_riccati` | `test_hinf_riccati_can_be_unstable` | **écart** (voir 3) |
| `H2Linear` | $\|G\|_2 = \gamma_2$ | `thm_3_3_12_gramian`, `thm_3_3_12_trace` | `test_h2_sound` | conforme |
| `ContractingRENODE` | contraction exponentielle | `thm_1_6_10` (Th. 1.6.10, 3.5.1, 3.5.2) | `test_contracting_renode_sound` | conforme |
| `ContractingRENODE` | réseau explicite bien posé | `explicit_wellPosed` | `test_contracting_renode_sound` | conforme |
| `SandwichLayer`, `LBDN` | $\|f(x) - f(x')\| \le \|\operatorname{diag}(s)(x - x')\|$ | `sandwich_nonexpansive`, `thm_3_4_2`, `prop_3_4_3` | `test_lbdn_lipschitz` | conforme |
| `DDLayer` | pénalité nulle ⟹ $M \succeq 0$ | `IsDiagDominant.posSemidef` | — | conforme (semi-définie seulement) |
| `models.discrete.ExplicitREN` (extension, hors thèse) | gain incrémental $\le \gamma$, contraction de taux $\bar\alpha$, pour tout paramètre ; $D_{11} = 0$ (coût d'un GRU) | `explicit_ren_param`, `explicit_ren_step`, `explicit_ren_gain`, `contraction_pow` (fichier `Ext/ExplicitREN.lean`) | `test_explicit_ren_gain`, `test_explicit_ren_contraction` | conforme |
| `BetaLayer` (linéarisation) | $\beta(x)$ inversible | `prop_4_1_5` (hypothèse) | — | conforme si `func='softplus'` et $n_{in} = n_{out}$ |

## Corrections faites sur cette branche

1. `SSLinear.check_` testait $\Re\lambda \le \alpha$ au lieu de $\Re\lambda \le -\alpha$. Une
   matrice non α-stable était acceptée. Corrigé.
2. `ContractingRENODE.check` utilisait $\operatorname{diag}(D_{11})$, nul car $D_{11}$ est
   strictement triangulaire, à la place de $\Lambda$ dans le bloc $PB_1 + C_1^\top\Lambda$.
   Un modèle valide était rejeté. Corrigé.
3. **Initialisation H∞ exacte** (`L2BoundedLinear.init_weights_(…, method='exact')`, défaut
   pour `param='sqrtm'`). Construction de `ms_variant_complete` : LMI du lemme borné réel au γ
   **prescrit** (marge maximale), puis $T$, $H$, $Q^{1/2}$, $S$, $M_s$ en forme close. Erreur de
   reconstruction $\approx 10^{-15}$ ; γ gardé. L'ancienne méthode reste (`method='legacy'`).
4. **ε du frame.** En Lean, ε dépend du système : $\varepsilon \le \lambda_{\min}(-N)/(3\|P\|^2)$.
   Le code fixait $\varepsilon = 10^{-3}$ : l'image codée était strictement plus petite que l'image
   prouvée (exemple : `GNSSM` avec $\|P\| \approx 70$, LMI infaisable). L'initialisation exacte
   réduit ε si nécessaire (avertissement).
5. **`geotorch_custom` : `inv_softplus_epsilon` n'inversait pas `softplus_epsilon`** (ε = 10⁻⁸
   au lieu de 10⁻⁶). Chaque matrice définie positive initialisée était décalée de
   $9{,}9 \cdot 10^{-7} I$. Corrigé (et borné sous l'image au lieu de rendre NaN).
6. **`GNSSM.init_weights_`** calculait γ sur $A_0$ alors que le frame impose la LMI sur
   $A_0 + \alpha I$. L'ancienne init ignorait α : reconstruction fausse de $\alpha I$ sans erreur.
   Corrigé.

## Écarts ouverts (tests marqués `xfail`, à décider)

1. **`ContractingRENODE.init_weights_` remplace `alpha` par `1e-3`.** La marge demandée par
   l'utilisateur est perdue après l'initialisation.
2. **Initialisation H∞ `legacy`** (gardée pour comparaison, test `xfail`) : γ remplacé par le
   gain initial, modèle au bord ($\lambda_{\min}(M_s) \approx 10^{-3}$), erreur $\approx 10^{-4}$.
3. **Riccati sans marge.** Avec $B = 0$, $G = 0$, on obtient $A = SP$, de valeurs propres
   imaginaires pures (`thm_3_3_8_image_not_hurwitz`). Il faut `alpha > 0` ou une garde.
4. **`H2Linear` exige la minimalité** (`thm_3_3_12_needs_minimality`). L'initialisation
   résout $A^\top W + WA + C^\top C = -Y$ ; la reconstruction n'est exacte que si $Y = 0$.
5. **`BetaLayer(func='relu')`** autorise $\sigma(x) = 0$, donc $\beta(x)$ singulière.
6. **`GNSSM`** : la garantie globale (partie linéaire H∞ + résidus Lipschitz) n'a pas encore de
   théorème Lean. Candidat : petit gain incrémental.

## Hors périmètre Lean

Modèles en temps discret (`rens.py`, `HInfDisc`, `H2Disc`), schémas d'intégration,
entraînement.
