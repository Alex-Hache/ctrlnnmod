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

### Constat (2026-10-20)

Lean ne contient que des énoncés du premier ordre (`gd_induced`, `riem_step_exact`) et une vitesse en 1/K sous hypothèse de régularité. Aucun théorème ne relie κ(JJᵀ) à une vitesse.
Conséquence de `riem_step_exact` : au premier ordre, `ren_eucl` **est** le modèle libre tant que la trajectoire libre reste dans S. Le test set_vs_metric mesure donc la courbure de Φ et les sorties du bord, pas « ensemble ou métrique ».

Affirmation reformulée (cible) : l'écart contraint/libre = terme de métrique (éliminable par préconditionnement, borné par κ(JJᵀ) à la solution) + terme de bord (nul si la trajectoire libre reste dans l'intérieur de S).

### 1. Lean : T2 puis T1 (these-lean, `TheseLean/Ext/InducedRate.lean`) — FAIT

- **T2 (écart du pas)** : pour Φ de classe C² avec ‖D²Φ‖ ≤ M, le pas Δθ = Jᵀ(JJᵀ)⁻¹s vérifie ‖Φ(θ+Δθ) − (Φ(θ)+s)‖ ≤ (M/2)·‖s‖²/σ_min(J)².
- **T1 (vitesse locale)** : en un minimum W* (∇L = 0), Hess_θ(L∘Φ) = JᵀHJ ; sur l'image de Jᵀ, λ_min(H)σ_min(J)² ≤ JᵀHJ ≤ λ_max(H)σ_max(J)², donc κ_eff ≤ κ(H)·κ(JJᵀ). Contraction de la GD linéarisée en 1 − 1/κ_eff.

Critère : `lake build` vert, 0 `sorry`, axiomes standard.

### 1b. Données existantes : la trajectoire libre sort-elle de S ?

Lire ρ_A (et la norme ℓ∞ pour les classes ℓ∞) dans les JSON des runs libres existants. Coût nul.
Critère : sortie de S sur ≥ 1 graine → le terme de bord existe sur cette tâche. Aucune sortie → l'écart observé est un terme de métrique.

### 1c. set_vs_metric (suspendu)

Ne pas relancer avant T2 et T3. Prédiction de T2 à vérifier : écart du pas ∝ 1/σ_min(J)². Journaliser σ_min(J) et κ(JJᵀ).

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

- **T3 (décomposition)** : en temps continu, le flot riemannien sur θ égale le flot libre sur W tant que celui-ci reste dans l'intérieur de S. Ne s'applique pas à H2 (sphère, intérieur vide) avant la correction par la boule.
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
- 2026-10-20 : bilan Lean ↔ résultats. Items réordonnés : théorie (T2, T1) avant calcul.
- 2026-10-20 : T2 et T1 démontrés (`Ext/InducedRate`, 128 énoncés, 0 `sorry`). Prochain : 1b.
