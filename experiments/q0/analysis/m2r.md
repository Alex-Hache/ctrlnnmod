# M2-r — le certificat est-il trop grossier ? (oscillateurs, blocs de 2)

Suite de `m2q.md`. Objectif de la séance (2026-10-22 matin) : écarter ou confirmer la dernière hypothèse ouverte, « le certificat est trop grossier », sous ses deux formes : la **borne** sur le chemin tanh, et la **métrique fixe** du certificat.

## Rappel : pourquoi un certificat plus fin est valide

Pour deux états x, x' et la même entrée : f(x) − f(x') = (A + B·Δ·C)(x − x'), avec Δ = diag(δ_q), δ_q ∈ [0, 1] (accroissements finis de tanh, coordonnée par coordonnée). Donc toute borne de maxₕ Σₖ ‖[A + BΔC]_hk‖₂ valable pour **tout** Δ ∈ [0, 1]³² certifie la non-expansion en norme max-ℓ² par tête (portes comprises, comme dans `BlockGatedRNN`).
- **r** (actuel, triangulaire) : r_h = Σₖ ‖A_hk‖ + Σ_q ‖B_hq‖ Σₖ ‖C_qk‖.
- **r_mid** (centre-rayon) : Δ = ½I + ½E, E ∈ [−1, 1]³². r_mid_h = Σₖ ‖A_hk + ½ B_h C_k‖ + ½ Σ_q ‖B_hq‖ Σₖ ‖C_qk‖ ≤ r_h. Valide, différentiable, homogène de degré 1 en (Ã_h·, B̃_h·) : la projection radiale reste exacte.
- **r_box** (le plus fin de cette forme) : sup sur Δ ∈ [0, 1]³². Convexe en Δ, donc atteint en un sommet ; 2³² sommets, non calculable exactement. On en calcule une **borne inférieure** r_box_lo.
- **Métrique par tête** : norme maxₕ α_h ‖x_h‖. Dans les coordonnées x' = T x, T = diag(α_h I₂) : A_hk → (α_h/α_k) A_hk, B_h → α_h B_h, C_k → C_k/α_k. Certificat r(α) (ou r_mid(α)), α appris (log-paramétré).

## Protocole et critères (écrits avant tout calcul)

Commun : banc, données, init W₀' (0,9ρ), Adam lr 1e-2, clip 10, 3000 itér., 5 graines, comme T4 de `m2q.md`. Références prises dans T4 (mêmes réglages) : libre 0,0018 ; P (r, métrique fixe) 0,288. Réussit < 0,05 ; échoue > 0,2 (médiane sur 5 graines). Majorité = au moins 3 graines sur 5. Lectures **provisoires**.

### T8 — finesse du certificat, mesurée sur les modèles finaux de T4 (pas d'entraînement)

Modèles : libre (5), fix2 (5) ; P (5) comme contrôle.
Mesures (sur W effectif) : r ; r_mid ; R_lin = valeur en Δ = I ; r_box_lo = max(meilleur sommet trouvé, R_loc), sommets : 4000 tirages aléatoires puis montée coordonnée par coordonnée depuis les 10 meilleurs ; R_loc = max le long de trajectoires de test (32 séquences × 300 pas) de maxₕ Σₖ ‖[A + B·diag(tanh'(Cx + Du + b))·C]_hk‖.
Contrôles (sinon, pas de lecture) :
- k1 : r_box_lo ≤ r_mid ≤ r sur tous les modèles (écart < 1e-9, float64).
- k2 : pour P (certifié), r_box_lo ≤ r_mid ≤ r ≤ ρ + 1e-5.
Lecture :
- Libre : r_box_lo > ρ sur la majorité → **aucun certificat de cette forme (norme max-ℓ² par tête, pentes dans [0, 1]) n'admet la solution libre**, même le plus fin. (Attendu : T5 montre déjà une dilatation 1,6–2,5 dans cette norme.)
- Descriptif : facteur de lâcheté borné par r / r_box_lo ; part de l'écart (r − r_box_lo) fermée par r_mid : (r − r_mid) / (r − r_box_lo).
- Même chose pour fix2 : r_box_lo ≤ 1 sur la majorité → la bonne solution fix2 serait admissible par le certificat le plus fin (à ρ = 1).

### T9, T10, T11 — entraîner avec un certificat plus fin ou une métrique apprise

Bras (même Φ « proj » : s_h = min(1, ρ / c_h) avec c le certificat du bras ; dans S, Φ = identité) :
- **mid** (T9) : c = r_mid, métrique fixe.
- **alpha** (T10) : c = r(α), α_h = exp(γ_h) appris (γ = 0 au départ, même lr).
- **midalpha** (T11) : c = r_mid(α), α appris.
W₀' est dans les trois S (c ≤ r = 0,9ρ au départ).
Contrôles (sinon, pas de lecture du bras en cause) :
- d0 : perte initiale identique à celle de P (T4) (écart < 1e-6).
- d-fin : c_max en fin ≤ ρ + 1e-5.
- d-val (validité du certificat) : non-expansion empirique du modèle final dans la norme du bras (maxₕ α_h ‖Δx_h‖), mesure de T5 (32 séquences × 2000 pas, 4 paires × 2 échelles, float64) : E ≤ 1 + 1e-4. Un E plus grand veut dire que le certificat du bras est faux : erreur de code ou de dérivation, pas de lecture.
Lecture par bras X (g = (P − X)/(P − libre)) :
- X réussit → un certificat valide plus fin (mid), ou une métrique apprise (alpha), permet d'apprendre dans S. L'hypothèse correspondante est **confirmée** : l'obstacle était la finesse de la borne (mid), la métrique fixe (alpha), ou les deux ensemble (midalpha seul).
- X échoue → cette forme de l'hypothèse est **réfutée** sur ce banc.
- Sinon → partiel ; je rapporte g.
Lecture d'ensemble :
- Les trois bras échouent → ni une borne plus fine (centre-rayon) ni une métrique par tête apprise ne suffisent. Restent : un obstacle propre au chemin (les modèles traversés par la descente sont expansifs dans toute norme bloc-diagonale), ou une métrique non bloc-diagonale (hors de cette classe à portes par tête).
Journal toutes les 50 itér. : perte, c_max, r_max (triangulaire), α min/max.

Prédictions (confiance faible) : T8 : r_box_lo du libre entre 2 et 5 ; r_mid / r entre 0,5 et 0,9. T9 : partiel. T10 : échoue (T6 montre qu'une mise à l'échelle par tête ne réduit r que de 17–37 % sur la solution libre). T11 : partiel.

Coût : T8 < 5 min ; T9–T11 : 15 runs × ≈ 4,5 min sur 2 cœurs ≈ 35–40 min.

## Résultats (règles appliquées par `analyze_m2r.py`) — `m2r_T8_analysis.txt`, `m2r_analysis.txt`, `results_m2r/`

### T8 (contrôles k1, k2 passés)
- Libre : r 56–64 ; r_mid 44–50 ; r_box_lo 30–37 ; lâcheté du certificat ≤ 1,5–2,0. **Ligne : aucun certificat de cette forme n'admet la solution libre, même le plus fin.**
- fix2 : r_box_lo 1,92–2,00 (lâcheté ≤ 1,04) ; admissible à ρ = 1 sur 0/5.
- P : r_box_lo ≈ r ≈ ρ.
- Gain local le long des trajectoires (pentes réelles) du libre : R_loc = 24–30.
- Correction de la lecture du 21/10 au soir : le facteur ≈ 20 entre r ≈ 60 et la dilatation réelle (≤ 2,5, T5) ne vient **pas** de l'inégalité triangulaire sur B̃·C.

### T9–T11 (contrôles d0, d-fin, d-val passés ; E = 0,92 à 0,99 : les trois certificats sont valides)
| bras | test (5 graines) | médiane | g | lecture |
|---|---|---|---|---|
| mid (borne centre-rayon) | 0,156 0,319 0,288 0,382 0,219 | 0,288 | 0,00 | **réfutée** |
| alpha (métrique par tête apprise) | 0,102 0,110 0,234 0,279 0,190 | 0,190 | 0,34 | partiel |
| midalpha (les deux) | 0,092 0,132 0,252 0,137 0,183 | 0,137 | 0,53 | partiel |
- La métrique apprise s'étale beaucoup : α max/min de 160 à 3800 en fin de run. Le r à métrique fixe des modèles finaux vaut 2 à 7 : le certificat appris admet des modèles nettement hors de S.
- Prédictions : 1 sur 3 juste (T11 partiel) ; T9 prédit partiel, observé échoue ; T10 prédit échoue, observé partiel.
- Lecture provisoire : une borne plus fine seule ne change rien. Une métrique libre par tête aide en partie (jusqu'à la moitié de l'écart avec la borne fine), mais aucun bras n'apprend comme le libre.
