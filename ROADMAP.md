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

### 1. Lean : T2, T1, T3 — FAIT

- **T2** (`Ext/InducedRate`, `riem_step_misfit`, `pinv_norm_sq_le`) : si DΦ est M-lipschitzienne, ‖Φ(θ+J⁺s) − (W+s)‖ ≤ (M/2)·‖s‖²/σ_min(J)².
- **T1** (`Ext/InducedRate`, `induced_linear_rate[_kappa]`, `induced_rate_iter`) : L ℓ-lisse et PL (μ), p₁I ⪯ JJᵀ, ‖Jv‖² ≤ p₂‖v‖², ηℓp₂ ≤ 1 ⇒ L⁺ − L* ≤ (1 − ημp₁)(L − L*). Avec η = 1/(ℓp₂) : facteur 1 − 1/(κ_L·κ(JJᵀ)). κ pris le long de la trajectoire.
- **T3** (`Ext/FlowDecomposition`) : flot riemannien sur θ = flot libre sur W tant qu'il existe ; le flot libre reste dans S ; sortie de S ⇒ pas de flot riemannien. Exige un inverse à droite (exclut H2-sphère et les pertes de rang).

### 1d. Expérience de validation T1/T2 (jouet PSD) — PROCHAIN

Montage : perte de Frobenius ½‖W − W*‖² (μ = ℓ = 1). Paramétrisations : euclidienne, XᵀX, exp(S). 5 graines × κ(W*) ∈ {10, 1e2, 1e3}. CPU local.
Journal par pas : p₁, p₂ de JJᵀ, f − f*, ‖s‖, écart du pas.
Validité : ηℓp₂ ≤ 1 sur ≥ 95 % des pas.
- T2 (XᵀX, exact) : confirme si écart ≤ ‖s‖²/p₁ sur 100 % des pas et écart·p₁/‖s‖² ∈ [0,1 ; 1]. Une violation réfute.
- T1 : confirme si f_k − f* ≤ (f₀ − f*)Π(1 − ηp₁(j)) sur 100 % des pas (à l'écart T2 près) et K_ε ∝ κ_traj^a, a ∈ [0,8 ; 1,2], r > 0,9. Réfute si violation > 10 % là où T2 est négligeable, ou r < 0,5.
- Discriminant : r(K_ε, κ_traj) > r(K_ε, κ_départ).
Étape 2 (Modal) : set_vs_metric REN avec σ_min(J) journalisé ; T2 explique l'invalidité si écart·σ_min²/‖s‖² reste dans un facteur 10.

### 1b. Données existantes : la trajectoire libre sort-elle de S ?

Lire ρ_A (et la norme ℓ∞ pour les classes ℓ∞) dans les JSON des runs libres existants. Coût nul.
Critère : sortie de S sur ≥ 1 graine → le terme de bord existe sur cette tâche. Aucune sortie → l'écart observé est un terme de métrique.

### 1c. set_vs_metric (suspendu)

Ne pas relancer avant 1d. Avec T3, reformuler : comparer le flot préconditionné (R = J⁺) au flot libre, et mesurer l'instant de sortie de S. Prédiction de T2 à vérifier : écart du pas ∝ 1/σ_min(J)². Journaliser σ_min(J) et κ(JJᵀ).

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

- Version discrète de T3 (T3 + T2 : écart cumulé sur K pas).
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
- 2026-10-20 : T3 démontré (`Ext/FlowDecomposition`, 132 énoncés). Expérience 1d définie avec critères. Prochain : 1d, puis 1b.
