# Garanties prouvées

Chaque paramétrisation de `ctrlnmod` repose sur un théorème. Ces théorèmes sont prouvés en
Lean 4 dans le dépôt [these-lean](https://github.com/Alex-Hache/these-lean)
(62 énoncés, aucun `sorry`). Les tests `tests/lean/test_lean_guarantees.py` vérifient
numériquement, sur des paramètres aléatoires, que le code respecte chaque énoncé.

```bash
pytest tests/lean
```

## Correspondance code → théorème

| Classe | Garantie | Théorème Lean (`These.…`) | Test | Statut |
| --- | --- | --- | --- | --- |
| `geotorch_custom.AlphaStable` (`SSLinear(alpha=…)`) | $\Re\lambda(A) < -\alpha$ | `thm_3_3_5_corrected`, `thm_3_3_6` | `test_alpha_stable_sound` | conforme |
| `L2BoundedLinear(param='sqrtm')` | $A$ Hurwitz, $\|G\|_\infty < \gamma$ | `ms_variant_sound` | `test_hinf_sqrtm_sound` | conforme |
| `L2BoundedLinear(param='sqrtm')`, initialisation | inverse à droite exact | `ms_variant_complete` | `test_hinf_sqrtm_right_inverse_exact` | **écart** (voir 2) |
| `L2BoundedLinear(param='riccati')` | Riccati avec $P \succ 0$ | `thm_3_3_8_riccati` | `test_hinf_riccati_can_be_unstable` | **écart** (voir 3) |
| `H2Linear` | $\|G\|_2 = \gamma_2$ | `thm_3_3_12_gramian`, `thm_3_3_12_trace` | `test_h2_sound` | conforme |
| `ContractingRENODE` | contraction exponentielle | `thm_1_6_10` (Th. 1.6.10, 3.5.1, 3.5.2) | `test_contracting_renode_sound` | conforme |
| `ContractingRENODE` | réseau explicite bien posé | `explicit_wellPosed` | `test_contracting_renode_sound` | conforme |
| `SandwichLayer`, `LBDN` | $\|f(x) - f(x')\| \le \|\operatorname{diag}(s)(x - x')\|$ | `sandwich_nonexpansive`, `thm_3_4_2`, `prop_3_4_3` | `test_lbdn_lipschitz` | conforme |
| `DDLayer` | pénalité nulle ⟹ $M \succeq 0$ | `IsDiagDominant.posSemidef` | — | conforme (semi-définie seulement) |
| `BetaLayer` (linéarisation) | $\beta(x)$ inversible | `prop_4_1_5` (hypothèse) | — | conforme si `func='softplus'` et $n_{in} = n_{out}$ |

## Corrections faites sur cette branche

1. `SSLinear.check_` testait $\Re\lambda \le \alpha$ au lieu de $\Re\lambda \le -\alpha$. Une
   matrice non α-stable était acceptée. Corrigé.
2. `ContractingRENODE.check` utilisait $\operatorname{diag}(D_{11})$, nul car $D_{11}$ est
   strictement triangulaire, à la place de $\Lambda$ dans le bloc $PB_1 + C_1^\top\Lambda$.
   Un modèle valide était rejeté. Corrigé.

## Écarts ouverts (tests marqués `xfail`, à décider)

1. **`ContractingRENODE.init_weights_` remplace `alpha` par `1e-3`.** La marge demandée par
   l'utilisateur est perdue après l'initialisation.
2. **Initialisation H∞ (`sqrtm`).**
   - `self.gamma` devient le gain du système initial ($\gamma_{sys}$), pas le $\gamma$ demandé.
     Le modèle démarre au bord de l'ensemble admissible : $\lambda_{\min}(M_s) \approx 7 \cdot
     10^{-3}$ dans le test. C'est un candidat direct pour les problèmes de conditionnement.
   - L'inverse à droite n'est pas exact : erreur $\approx 1{,}5 \cdot 10^{-4}$ sur $A$. Le terme
     $\varepsilon I$ du frame n'est pas compensé dans $S$, et `alpha` n'est pas pris en compte.
   - Construction exacte : `ms_variant_complete` (voir
     [enonces-corriges.md, fiche 4](https://github.com/Alex-Hache/these-lean/blob/main/docs/enonces-corriges.md)).
     Elle garde $\gamma$ et une marge $\lambda_{\min}(-N)/3$ sur $M_s$.
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
