# Roadmap — métrique induite et entraînabilité

## Affirmation centrale

La métrique induite par la paramétrisation (JJᵀ), plus que l'ensemble contraint, gouverne l'entraînabilité.

Refuser tout travail qui ne sert pas cette affirmation.

## Règles de session

1. Écrire un objectif au début : « Session : item N uniquement. »
2. Écrire le critère de réussite **avant** chaque calcul : quelle valeur confirme, quelle valeur réfute.
3. Faire un test rapide. Lire les diagnostics (écart du pas, rang, κ). Relancer le test rapide sur plus d'itérations avant un sweep.
4. Vérifier `nproc` et le coût. Utiliser Modal `--detach` au-delà de 15 min.
5. Vérifier la thèse et these-lean avant d'annoncer un résultat comme nouveau.
6. Une session = un item. Mettre à jour ce fichier à la fin.

## Items (ordre de priorité)

### 1. Corriger et relancer `experiments/memory/set_vs_metric.py` — PROCHAIN

État : test actuel invalide. Le pas ramené dans θ ne suit pas le pas euclidien (écart 13 à 63×). λ = 1e-6 sans contrôle du pas.

Actions :
- Régularisation relative : λ = 1e-3 · ‖JJᵀ‖.
- Accepter le pas si l'écart au pas euclidien < 10 %. Sinon diviser le pas par 2.
- Journaliser κ(JJᵀ), l'écart du pas et le rang le long de la trajectoire.
- Test rapide local, puis Modal CPU (~10 min, < 1 $).

Critère de validité (à vérifier d'abord) : écart du pas < 10 % sur ≥ 95 % des itérations, sur un test rapide **et** sur un test 10× plus long. Sinon : test invalide, ne pas interpréter.

Critère de réussite :
- `ren_eucl` ≈ REN (perte finale à ±20 %) → l'ensemble compte. Réfute l'affirmation.
- `ren_eucl` ≈ libre (à ±20 %) → la métrique compte. Confirme l'affirmation.
- Entre les deux → résultat mixte. Noter la fraction de l'écart fermée.

### 2. Préconditionnement par (JJᵀ)⁻¹ sur les oscillateurs ℓ∞/blocs

Base : REN 0,017 ; GRU 0,014 ; ℓ∞/blocs 0,16 à 0,44 (maître représentable).

Critère de réussite :
- Confirme : ℓ∞/blocs préconditionnés < 0,05 (écart avec REN fermé à > 80 %).
- Réfute : amélioration < 20 % malgré un pas valide (même critère de validité qu'item 1).

### 3. Loi κ sur les vrais modèles (H∞, REN)

Mesurer κ(JJᵀ) à la solution et le long de la trajectoire. Rappel : κ au départ ne prédit pas la vitesse (`analysis/lmi_gd.md`).

Critère de réussite :
- Confirme : itérations ∝ κ_solution^a, corrélation log-log r > 0,8 (jouet PSD : a ≈ 1,09, r = 0,98).
- Réfute : r < 0,5.

### 4. Corrections ctrlnmod

- Init P bien conditionné (P = I divise κ par 1e4–1e5).
- XᵀX au lieu de `geo.positive_definite`.
- expm au lieu de Cayley.
- H2 par la boule : σ = s·√softmax, s = sigmoid(c). Ajouter le théorème Lean associé.

Critère : tests existants verts ; κ initial des modèles H∞ < 1e2 ; rang de JJᵀ plein pour H2.

### 5. Lean (these-lean)

- Daleckii–Krein pour exp(S) (dérivée de Fréchet de l'exponentielle, absente de Mathlib).
- Loi en κ pour une perte générale.

Critère : `lake build` vert, 0 `sorry`, axiomes standard.

### 6. Baselines SOTA, rapport, article

- Sweep des baselines (différé).
- Rapport : sections petit gain, ℓ∞ pixels, géométrie.
- Article.

Ne pas commencer avant que les items 1 et 2 aient un résultat valide.

## Ménage

- Supprimer la branche `lean-guarantees` sur GitHub (fusionnée, PR #3).

## Journal

- 2026-10-20 : roadmap créée. Test set_vs_metric déclaré invalide.
