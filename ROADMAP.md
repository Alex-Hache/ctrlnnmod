# Roadmap — métrique induite et entraînabilité

## Affirmation centrale (à tester par Q0)

La métrique induite par la paramétrisation (JJᵀ), plus que l'ensemble contraint, gouverne l'entraînabilité.

Refuser tout travail qui ne sert pas Q0 (section suivante).

## Règles de session

1. Écrire un objectif au début : « Session : item N uniquement. »
2. Écrire le critère de réussite **avant** chaque calcul : quelle valeur confirme, quelle valeur réfute.
3. Faire un test rapide. Lire les diagnostics (écart du pas, rang, κ). Relancer le test rapide sur plus d'itérations avant un sweep.
4. Vérifier `nproc` et le coût. Utiliser Modal `--detach` au-delà de 15 min.
5. Vérifier la thèse et these-lean avant d'annoncer un résultat comme nouveau.
6. Une session = un item. Mettre à jour ce fichier à la fin.

## Question centrale (Q0) — fixée le 2026-10-21

**Q0 : pourquoi une classe contrainte échoue-t-elle alors que le maître est dans S ?**
Q0 est fausse en général : le REN contraint (0,017) bat ses équations libres (0,181) sur les oscillateurs. On la pose donc classe par classe.

Chaîne (prouvée, `gd_induced`, T1, T2, T3) : le contraint diffère du libre par la métrique JJᵀ et par l'ensemble S. Trois termes :
- **métrique** : contraint face au riemannien (J⁺). Vitesse : T1, κ(JJᵀ) le long de la trajectoire ;
- **bord** : riemannien face au libre. Existe seulement si la trajectoire libre sort de S (T3) ;
- **rang** : J perd du rang. Pas de riemannien ; points stationnaires parasites possibles.
Précisions : une perte de rang structurelle (dim S < dim W, ex. H2-sphère) signale un libre mal défini ; seule une perte ponctuelle crée des points parasites. Un maître près du bord donne un terme de **métrique** (J → 0 dans la direction normale), pas un terme de bord. En pas discret, l'écart du pas (T2) s'ajoute ; avec Adam, la métrique devient J·D·Jᵀ.

Tri des résultats existants :
- valident les outils, pas Q0 : jouet PSD (T1), κ au départ en H∞, 15 directions perdues, H2, set_vs_metric invalide (T2) ;
- la contrainte aide : maître–élève près du bord, REN oscillateurs ;
- capacité, hors Q0 : **ℓ∞ sur les oscillateurs** (le maître n'est pas dans S : rotation à 45° → somme de lignes 1,40, voir docstring de `oscillator.py`), mémoire (portes par unité incompatibles avec ℓ²), world model (pas de maître) ;
- **seul cas Q0 : blocs de 2 sur les oscillateurs** (`experiments/memory/oscillator.py`, `analysis/oscillator.md`) : 0,16 à 0,44 contre REN 0,017.

## Chemin Q0 : une expérience sur les blocs de 2, trois mesures

### Préalable — FAIT (lecture du code, 2026-10-21)

`BlockGatedRNN` : x⁺ = (1 − z)x + z·f, f = Ax + B·tanh(Cx + Du + b) + Eu + bₓ. Φ ne modifie que (A, B) : W_AB = s_h·(Ã, B̃), avec r_h = Σₖ‖A_hk‖₂ + Σ_q‖B_hq‖₂ Σₖ‖C_qk‖₂ (r dépend aussi de C).
- Libre = mêmes équations avec s ≡ 1. S = {r_h(A, B, C) ≤ ρ ∀h} a un intérieur non vide : dim S = dim W. **Préalable vérifié.**
- Les variantes n'exercent pas les mêmes termes :
  - `block` (« sat », ρ = 0,99, 0,437) : s = ρ/(ρ + r), somme de lignes < ρ = 0,99. Le maître (0,99) est **sur le bord**, atteint seulement si r → ∞. Pas un cas Q0 propre ;
  - `blockp*` (« proj », ρ = 0,999, 0,16 à 0,30) : s = ρ/max(r, ρ). **Φ = identité à l'intérieur de S** : J = I, terme de métrique nul. Une tête saturée (r_h > ρ) est projetée radialement : Φ est homogène de degré 0 en (Ã_h, B̃_h), donc J perd au moins une direction par tête saturée. Cas Q0 propre (marge du maître 0,009) ;
  - `blockh*` (« homo », 0,23 à 0,31) : s = 1/(r + softplus(c)). Rang plein ; marge réglée par c.
- Portes : biais −3, z ≈ 0,05 au départ. Avec les contraintes, une tête qui doit reproduire une rotation rapide exige z proche de 1 (≈ 0,98 à 45°). Le libre peut compenser par un A de grande norme.

### M1 — Rang et κ de J, sans sweep (local, minutes) — PROCHAIN

Points : θ₀ ; θ_fin d'un run échoué (un run Adam de 2000 itér., ~70 s, aucun checkpoint n'existe) ; θ* si une préimage finie existe. Classes : `blockp_nogate`, `blockp`, `block`, `blockh` ; REN en témoin.
Mesures : rang de J (seuil σ/σ_max < 1e-6), σ_min, κ(JJᵀ) sur l'image, nombre de têtes saturées (r_h > ρ), part de g = ∇_W L hors de l'image de J en θ_fin, ‖Jᵀg‖/(σ_max‖g‖).
Prédiction (code) : `blockp*` a un déficit de rang égal au nombre de têtes saturées ; `block` a un κ qui croît comme (r/ρ)².
- Terme de rang confirmé : déficit de rang en θ_fin **et** > 50 % de ‖g‖² hors de l'image **et** ‖Jᵀg‖/(σ_max‖g‖) < 1e-3.
- Terme de rang réfuté : rang plein le long du run (toutes les têtes à r_h < ρ).
- Métrique plausible : rang plein et κ_blocs ≥ 100 × κ_REN.

### M2 — Bord : le run libre des blocs (local, ~70 s/run)

Le run libre des équations blocs n'existe pas. Lancer contraint et libre depuis le même W₀, en GD pur et en Adam, 5 graines. Journaliser la marge m(t) = ρ − max_h r_h(W) et la perte.
Validité : en GD pur, blocs > 0,2 et REN < 0,05. Sinon, l'écart vient d'Adam ; T1–T3 ne s'appliquent pas tels quels.

### M3 — Riemannien (Modal), seulement si M1–M2 sont ambigus

Si le rang est plein et si le libre reste dans S, T3 donne la réponse : riemannien = libre. M3 confirme seulement. Validité (T2) : écart du pas < 10 % sur ≥ 95 % des pas.

### Décision (seuils : « réussit » < 0,05 ; « échoue » > 0,2)

| M1 | M2 (libre) | Cause | Suite |
|---|---|---|---|
| rang plein, κ_blocs ≥ 100·κ_REN | réussit, m(t) > 0 partout | métrique (par T3) | item 2 (préconditionnement) |
| rang plein | m(t) < 0 à un instant | bord | faire varier la marge du maître |
| déficit en θ_fin, gradient hors de l'image | — | rang (point parasite) | corriger Φ (item 4) |
| — | échoue aussi | pas la contrainte (équations, portes) | Q0 réfutée pour les blocs |

## Validation des outils (hors chemin Q0)

Ces items valident T1–T3. Utiles pour l'article. Ils ne répondent pas à Q0.

### 1. Lean : T2, T1, T3 — FAIT

- **T2** (`Ext/InducedRate`, `riem_step_misfit`, `pinv_norm_sq_le`) : si DΦ est M-lipschitzienne, ‖Φ(θ+J⁺s) − (W+s)‖ ≤ (M/2)·‖s‖²/σ_min(J)².
- **T1** (`Ext/InducedRate`, `induced_linear_rate[_kappa]`, `induced_rate_iter`) : L ℓ-lisse et PL (μ), p₁I ⪯ JJᵀ, ‖Jv‖² ≤ p₂‖v‖², ηℓp₂ ≤ 1 ⇒ L⁺ − L* ≤ (1 − ημp₁)(L − L*). Avec η = 1/(ℓp₂) : facteur 1 − 1/(κ_L·κ(JJᵀ)). κ pris le long de la trajectoire.
- **T3** (`Ext/FlowDecomposition`) : flot riemannien sur θ = flot libre sur W tant qu'il existe ; le flot libre reste dans S ; sortie de S ⇒ pas de flot riemannien. Exige un inverse à droite (exclut H2-sphère et les pertes de rang).

### 1e. Métrique seule, ensemble fixé (maître–élève, ρ = 0,5)

But : isoler le terme de métrique. Le bord est exclu par construction et vérifié par LMI.
Montage : maître à ρ = 0,5 (loin du bord). Reparamétrisation linéaire θ = D·θ', D diagonale de dispersion κ_D ∈ {1, 1e1, 1e2, 1e3}. L'image de Φ (donc S) est identique ; seule JJᵀ change. Méthodes : GD sur θ', GD riemannien, libre. 3 graines. Local.
Validité : test LMI exact (cvxpy) à chaque point journalisé : le libre reste dans S sur 100 % de la trajectoire. Sinon, ne pas conclure.
Prédictions : GD sur θ' : itérations jusqu'à ε ∝ κ_D (T1) ; riemannien invariant à D, ≈ libre (T3) ; libre inchangé.
- Confirme : écart GD/libre croît avec κ_D, pente log-log ∈ [0,8 ; 1,2], et riemannien/libre < 1,5× pour tout κ_D.
- Réfute : écart GD/libre < 1,5× à κ_D = 1e3.

### 1d. Expérience de validation T1/T2 (jouet PSD)

Montage : perte de Frobenius ½‖W − W*‖² (μ = ℓ = 1). Paramétrisations : euclidienne, XᵀX, exp(S). 5 graines × κ(W*) ∈ {10, 1e2, 1e3}. CPU local.
Journal par pas : p₁, p₂ de JJᵀ, f − f*, ‖s‖, écart du pas.
Validité : ηℓp₂ ≤ 1 sur ≥ 95 % des pas.
- T2 (XᵀX, exact) : confirme si écart ≤ ‖s‖²/p₁ sur 100 % des pas et écart·p₁/‖s‖² ∈ [0,1 ; 1]. Une violation réfute.
- T1 : confirme si f_k − f* ≤ (f₀ − f*)Π(1 − ηp₁(j)) sur 100 % des pas (à l'écart T2 près) et K_ε ∝ κ_traj^a, a ∈ [0,8 ; 1,2], r > 0,9. Réfute si violation > 10 % là où T2 est négligeable, ou r < 0,5.
- Discriminant : r(K_ε, κ_traj) > r(K_ε, κ_départ).
Étape 2 (Modal) : set_vs_metric REN avec σ_min(J) journalisé ; T2 explique l'invalidité si écart·σ_min²/‖s‖² reste dans un facteur 10.

### 1b. La trajectoire libre sort-elle de S ? — FAIT, portée limitée

Résultat (`experiments/optim/analysis/exit_check.md`) : à ρ ≥ 0,9, 7/12 runs libres sortent de S en cours de route (Adam 6/6, GD 1/6) ; à ρ = 0,5, aucune sortie. Contrôle REN valide.
Correction : l'optimum (le maître) est dans S par construction. Les sorties sont des **détours** : à ρ = 0,9, Adam libre sort de S et finit au niveau du REN. 1b montre seulement que le flot riemannien ne peut pas suivre le libre. Il ne montre **pas** que le bord explique un déficit du modèle contraint (ici le contraint bat le libre).
À faire (petit) : test LMI au point final de chaque run → classer « détour puis retour » ou « fin hors de S ».

### 1c. set_vs_metric (suspendu)

Ne pas relancer avant 1d. Avec T3, reformuler : comparer le flot préconditionné (R = J⁺) au flot libre, et mesurer l'instant de sortie de S. Prédiction de T2 à vérifier : écart du pas ∝ 1/σ_min(J)². Journaliser σ_min(J) et κ(JJᵀ).

## Après Q0 (selon la décision)

### 2. Préconditionnement par (JJᵀ)⁻¹ sur les blocs (si M1–M2 indiquent la métrique)

Base : REN 0,017 ; GRU 0,014 ; blocs 0,16 à 0,44 (maître dans S pour `blockp*`, ρ = 0,999). ℓ∞ est exclu : défaut de capacité.

Critère de réussite :
- Confirme : blocs préconditionnés < 0,05 (écart avec REN fermé à > 80 %).
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

## Gelé jusqu'à la réponse à Q0

H∞/H2 : extension pour le terme de rang, après Q0.

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
- 2026-10-20 : 1b fait. Terme de bord confirmé en maître–élève (Adam sort de S dès l'itér. 50–250). Prochain : 1d.
- 2026-10-20 : conclusion de 1b corrigée (sorties = détours, pas un déficit). Ajout de 1e (métrique seule, ensemble fixé). Prochain : 1e.
- 2026-10-21 : retour à Q0. ℓ∞ reclassé en défaut de capacité. Cas Q0 = blocs de 2 (proj, ρ = 0,999). Chemin M1 → M2 → M3 avec critères. Préalable vérifié par lecture du code. Items 1b–1e déplacés en validation. Prochain : M1.
