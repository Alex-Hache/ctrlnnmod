# Item 1b — la trajectoire libre sort-elle de S ? (maître–élève, 2026-10-20)

Conditions nécessaires de W ∈ S (contraction ExplicitREN) : r₀ = ρ(A) < 1 et r₁ = ρ(A + B_w Λ⁻¹ C₁) < 1.
Une valeur ≥ 1 prouve la sortie de S. Aucune valeur ≥ 1 ne prouve pas l'appartenance.
Commande : `OUT_DIR=experiments/optim/results_exit python experiments/optim/exit_check.py --run`.

| méthode | ρ maître | graine | pas | max r₀ | max r₁ | 1re sortie (itér.) | test final | statut |
|---|---|---|---|---|---|---|---|---|
| free_adam | 0.5 | 0 | 0.03 | 0.532 | 0.687 | — | 6.96e-05 | ok |
| free_adam | 0.5 | 1 | 0.03 | 0.744 | 0.692 | — | 8.49e-05 | ok |
| free_adam | 0.9 | 0 | 0.01 | 0.868 | 1.004 | 250 | 2.31e-04 | ok |
| free_adam | 0.9 | 1 | 0.01 | 0.904 | 1.016 | 75 | 5.86e-04 | ok |
| free_adam | 0.97 | 0 | 0.003 | 0.961 | 1.115 | 75 | 1.83e-03 | ok |
| free_adam | 0.97 | 1 | 0.003 | 0.981 | 1.112 | 125 | 3.29e-03 | ok |
| free_adam | 0.995 | 0 | 0.01 | 0.996 | 1.115 | 50 | 1.91e-01 | ok |
| free_adam | 0.995 | 1 | 0.01 | 0.993 | 1.178 | 150 | 1.03e+00 | ok |
| free_gd | 0.5 | 0 | 0.1 | 0.625 | 0.526 | — | 1.72e-03 | ok |
| free_gd | 0.5 | 1 | 0.1 | 0.681 | 0.545 | — | 1.34e-03 | ok |
| free_gd | 0.9 | 0 | 0.03 | 0.863 | 0.933 | — | 6.09e-02 | ok |
| free_gd | 0.9 | 1 | 0.03 | 0.948 | 0.952 | — | 2.73e-02 | ok |
| free_gd | 0.97 | 0 | 0.01 | 0.960 | 0.996 | — | 3.95e-01 | ok |
| free_gd | 0.97 | 1 | 0.01 | 0.965 | 0.988 | — | 3.72e-01 | ok |
| free_gd | 0.995 | 0 | 0.03 | 0.897 | 1.071 | 425 | 1.64e+00 | ok |
| free_gd | 0.995 | 1 | 0.03 | 0.835 | 0.718 | — | 2.21e+00 | diverged@160 |
| ren_adam | 0.995 | 0 | 0.003 | 0.996 | 0.996 | — | 2.32e-03 | ok |

## Verdict (critère écrit avant le calcul)

- Contrôle valide : REN, max r₀ = max r₁ = 0,996 < 1.
- Prédiction de T3 confirmée : à ρ = 0,5, aucune sortie (4/4 runs). Libre ≈ REN (0,9×).
- **Terme de bord confirmé** : à ρ ≥ 0,9, 7 runs sur 12 sortent de S. Adam : 6/6, tôt (itér. 50 à 250), par r₁ (pente du tanh à 1). GD : 1/6 (ρ = 0,995, itér. 425).
- Non concluant : GD à ρ = 0,9 et 0,97 (r₁ max 0,93 à 0,996, condition nécessaire seulement) ; GD ρ = 0,995 graine 1 diverge à l'itér. 160 avant toute sortie journalisée.

## Lecture avec T3

- Adam : le libre sort de S tôt. T3 prédit alors que le flot riemannien ne peut pas suivre le libre (θ diverge). Observé : cond(XᵀX) ≈ 1e301 et riemannien/libre = 2,7× à 13,7× (`analysis/table_ts_base.md`). Cohérent.
- GD à ρ = 0,9 et 0,97 : aucune sortie détectée, mais REN bat le libre (0,1× à 0,2×). Si l'appartenance est confirmée par le test LMI exact, cet écart est un terme de **métrique**.
- Limite : seuls les runs maître–élève sont couverts. Les runs oscillateurs n'ont pas de journal de trajectoire.
