# Session parallèle — « gradient non nul, perte plate » : explication théorique et preuve Lean

Fiche écrite le 2026-10-22 (07:50) à la demande de l'utilisateur. **À lire en premier par la session qui la reçoit.**

## Objectif unique de la session

Expliquer **théoriquement**, puis **démontrer en Lean** (dépôt `these-lean`), le phénomène suivant : pendant l'entraînement d'un modèle contraint par reparamétrisation, le gradient reste non nul (même tangent au bord de l'ensemble contraint), mais la perte ne baisse plus. L'utilisateur a observé ce phénomène plusieurs fois pendant sa thèse.

Hors objectif : toute nouvelle expérience pour Q0, tout changement de banc, tout autre item de `ROADMAP.md`. Une mesure numérique n'est permise que pour vérifier une prédiction de la théorie, avec un critère écrit et poussé avant le calcul (règle du dépôt).

## Ce qui est mesuré (faits, avec fichiers)

Banc : oscillateurs (`experiments/memory/oscillator.py`), classe `blockp` (`ctrlnmod/models/discrete/linf_gated.py`, `BlockGatedRNN`, `norm="proj"`, ρ = 0,999). Run « P » de `experiments/q0/analysis/m2q.md` (T4) : Adam lr 1e-2, clip 10, 3000 itér., 5 graines ; modèles finaux dans `experiments/q0/results_m2q/t4_p_s*.pt` (à régénérer si absents : `m2q.py t4 --arms p`, les `.pt` sont ignorés par git).
- Paramétrisation : W_h = s_h·(Ã_h, B̃_h), s_h = ρ / max(r_h(θ), ρ), avec r_h = Σₖ ‖A_hk‖₂ + Σ_q ‖B_hq‖₂ Σₖ ‖C_qk‖₂. Pour une tête saturée (r_h > ρ), Φ est **homogène de degré 0** en (Ã_h, B̃_h) : le Jacobien perd la direction radiale (M1 : déficit de rang = nombre de têtes saturées).
- U1 (`experiments/q0/analysis/m2t.md`, `results_m2t/u1.json`, `m2t_log.txt`) : au point final, 8 têtes sur 8 saturées. Gradient de la perte en W, projeté sur l'espace tangent de ∂S : ‖g_tan‖ / ‖g(t = 0)‖ = 1,65 ; 5,83 ; 1,04 ; 0,95 ; 3,09. Pente de la perte entre t = 2500 et 3000 : −0,01 % ; 5,4 % ; 0,7 % ; −2,6 % ; 0,5 %.
- Dérive du paramètre brut (`results_m2q/t4_p_s*.json`, champ `rraw`) : r brut max croît pendant le run (graine 0 : 3,14 à t = 500, 3,77 à t = 1500, 4,56 à t = 3000 ; fins 4,1 à 5,9 sur les 5 graines), alors que W reste sur ∂S (r(W) = ρ).
- Perte d'entraînement de P (médiane) : ≈ 0,057 à 0,14 en fin, quasi plate après t ≈ 2000 sur 4 graines sur 5.
- Témoins utiles : le gradient projeté avec rétraction radiale (θ gardé dans S, pas de dérive du brut) fait pire (test 0,77) ; la même dynamique sans contrainte réussit (0,0018).

## Pistes à examiner (hypothèses, pas des conclusions)

1. **Invariance d'échelle et taux d'apprentissage effectif.** Si f(θ) = L(Φ(θ)) avec Φ homogène de degré 0 sur un bloc, alors ⟨∇f(θ), θ⟩ = 0 (Euler), ‖∇f(cθ)‖ = ‖∇f(θ)‖ / c, et un pas de gradient fait croître ‖θ‖ (Pythagore : ‖θ⁺‖² = ‖θ‖² + η²‖∇f‖²). Le pas effectif sur la sphère décroît comme η/‖θ‖². C'est cohérent avec la dérive mesurée du r brut. À vérifier : que devient l'argument sous Adam (préconditionnement par coordonnée) et avec le clip ?
2. **Non-différentiabilité du bord** : max(r, ρ) et les normes spectrales (valeurs singulières égales) donnent un problème non lisse ; zigzag entre faces actives.
3. **Pas trop grand face à la courbure** (régime « edge of stability ») : gradient non nul, perte non décroissante quand le pas dépasse 2/λ_max localement.
4. **Projection + optimiseur adaptatif** : non-convergence connue de certaines combinaisons.
La session doit trancher entre ces pistes par la théorie d'abord, puis par une vérification numérique minimale (critère écrit avant).

## Recherche bibliographique (obligatoire, avant toute preuve)

Faire une vraie recherche (WebSearch / WebFetch, sources primaires : arXiv, actes de conférences), citer chaque source par lien, et vérifier chaque référence avant de s'en servir. Pistes de départ, **non vérifiées**, à confirmer ou écarter :
- invariance d'échelle et taux effectif : van Laarhoven (2017) « L2 regularization versus batch and weight normalization » ; Hoffer et al. (2018) « Norm matters » ; Arora, Li, Lyu (2019) « Theoretical analysis of auto rate-tuning by batch normalization » ; Li, Arora (2020) « An exponential learning rate schedule for deep learning » ; Wan et al. (2021) « Spherical motion dynamics » ;
- pas et courbure : Cohen et al. (2021) « Gradient descent on neural networks typically occurs at the edge of stability » ;
- reparamétrisation contre projection, descente miroir : Niculae (2020, vene.ro) « Optimizing with constraints: reparametrization and geometry » (billet qui a inspiré la thèse) ; Amid, Warmuth (2020) « Reparameterizing mirror descent as gradient descent » ;
- optimisation non lisse et sous-gradient projeté ; convergence d'Adam avec projection.
Livrable : `experiments/q0/sessions/biblio_gradient_plat.md` (référence, lien, résultat utile, hypothèses du résultat, lien avec notre cas).

## Lean

- Dépôt `these-lean` (Lean 4 / Mathlib ; build : `export PATH=$HOME/.elan/bin:$PATH; lake build`). Nouveau fichier sous `TheseLean/Ext/` (par ex. `ScaleInvariance.lean`). Ajouter le dépôt à la session avec accès « push ».
- Cible minimale (si la piste 1 tient) : pour f homogène de degré 0 sur un espace préhilbertien réel, différentiable, (i) ⟨∇f(θ), θ⟩ = 0 ; (ii) ∇f(cθ) = c⁻¹ ∇f(θ) pour c > 0 ; (iii) un pas de gradient vérifie ‖θ − η∇f(θ)‖² = ‖θ‖² + η²‖∇f(θ)‖² ; (iv) conséquence sur le pas effectif angulaire. Puis, si possible, le lien avec la paramétrisation « proj » (Φ homogène de degré 0 sur les têtes saturées).
- Pas de `sorry` dans le livrable final ; axiomes standard ; énoncés relus contre la théorie écrite.

## Méthode (règles du dépôt)

- Objectif écrit en tête de session. Une seule question : celle-ci.
- Critère de réussite écrit et poussé avant chaque calcul. Lectures « provisoires » tant qu'un témoin n'exclut pas l'effet de l'optimiseur.
- Style des réponses : ASD-STE100, en français.

## Livrables

1. `experiments/q0/sessions/biblio_gradient_plat.md` (bibliographie vérifiée).
2. `experiments/q0/sessions/theorie_gradient_plat.md` (énoncés, preuves sur papier, ce qui explique les faits ci-dessus et ce qui ne les explique pas).
3. Fichier Lean dans `these-lean` (énoncés prouvés, 0 `sorry`).
4. Une ligne dans le journal de `ROADMAP.md`.
