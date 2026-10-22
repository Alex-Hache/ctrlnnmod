# M2-s — le gradient choisit-il le mauvais type de solution ? (oscillateurs, `blockp`)

Suite de `m2r.md`. Écrit le 2026-10-22 vers 07:32, avant tout calcul, pendant que T9–T11 tournent.

## Fait de départ

Le maître est dans S. Sa représentation exacte (`m2_teacher_on_boundary.py`, perte 1e-29) a B̃·C = 0 (pas de chemin tanh), des blocs Ã de norme ρ, et des portes ouvertes (z = 0,125 ; 0,618 ; 0,911 ; 0,970). Le libre fait l'inverse : B̃·C porte 85–90 % de son r, et la descente fait monter B̃·C dès le départ (T1, M2-ter). Les portes partent de z ≈ 0,05 (biais −3).

## Hypothèses

- **H-tanh** : la descente dépense le budget de r dans B̃·C ; après le contact avec ∂S, il ne reste pas de budget pour Ã.
- **H-portes** : la solution dans S demande des portes ouvertes ; depuis z ≈ 0,05, la descente ne les ouvre pas assez.
- **H-bassin** : le problème est l'accès à la bonne région de S, pas la région elle-même.

## Protocole et critères (écrits avant tout calcul)

Commun : banc et données de T4, `blockp` (Φ « proj », certificat r, ρ = 0,999, s = min(1, ρ/r)), Adam lr 1e-2, clip 10, 3000 itér., 5 graines. Réussit < 0,05 ; échoue > 0,2 (médiane). Majorité = 3/5. Références T4 : P 0,288 ; libre 0,0018. Lectures provisoires.

Bras :
- **lin** : init W₀', puis B̃ = 0, figé (pas de gradient). Modèle linéaire à portes, dans S.
- **linfree** (témoin de validité de lin) : mêmes équations, B̃ = 0 figé, sans contrainte (s ≡ 1).
- **gate0** : P-int (init W₀'), biais de porte à 0 au départ (z ≈ 0,5).
- **lingate0** : lin et gate0 ensemble.
- **basin2**, **basin1** : init = représentation exacte du maître sur ∂S (construite par graine, comme `m2_teacher_on_boundary.py`), puis bruit multiplicatif (1 + σ·N(0, 1)) sur tous les paramètres, σ = 1e-2 (basin2) ou 1e-1 (basin1), puis Φ.

Contrôles (sinon, pas de lecture du bras) :
- e-lin : ‖B̃‖ = 0 exactement en fin (lin, linfree, lingate0).
- e-basin : erreur de test à t = 0 < 0,01 pour basin2 (le départ doit être bon). basin1 : descriptif.
- e-fin : r max de W ≤ ρ + 1e-5 en fin (bras contraints).
- e-val : non-expansion empirique du modèle final (mesure de T5) : E ≤ 1 + 1e-4 (bras contraints).
- Validité de lin : linfree réussit (sinon la classe linéaire n'apprend pas ce banc : pas de lecture de H-tanh).

Lecture :
- lin réussit (et linfree réussit) → **H-tanh confirmée** : sans le chemin tanh, la classe contrainte apprend ; le chemin tanh est le piège.
- lin échoue, linfree réussit → H-tanh réfutée sous cette forme : même sans tanh, la contrainte bloque.
- gate0 réussit → **H-portes confirmée**. gate0 échoue → réfutée.
- lingate0 réussit alors que lin et gate0 échouent → les deux sont nécessaires ensemble.
- basin2 réussit → la région du maître est stable sous l'entraînement contraint : **H-bassin soutenue** (problème d'accès). basin2 échoue → même près du maître, l'entraînement contraint s'en éloigne : le paysage dans S près du maître est mauvais.
- Sinon : partiel ; je rapporte g = (P − X)/(P − libre).

Prédictions (confiance faible) : linfree réussit ; lin partiel ; gate0 échoue ; lingate0 partiel ; basin2 réussit ; basin1 partiel.

Coût : 30 runs × ≈ 4,5 min sur 2 cœurs ≈ 70 min, après la fin de T9–T11.

## Résultats (règles appliquées par `analyze_m2s.py`) — `m2s_analysis.txt`, `results_m2s/`

Contrôles : e-lin, e-fin, e-val passés pour tous les bras concernés (E = 0,96 à 0,997). **e-basin en échec** : avec 1 % de bruit multiplicatif, le test au départ vaut déjà 0,024 à 0,086 (seuil 0,01). Pas de lecture de H-bassin.

| bras | test (5 graines) | médiane | g | lecture |
|---|---|---|---|---|
| linfree (témoin) | 0,0001 0,0198 0,0002 0,0001 0,0001 | 0,0001 | 1,01 | réussit (validité de lin acquise) |
| lin (sans tanh, portes fermées au départ) | 0,294 0,335 0,244 0,414 0,109 | 0,294 | −0,02 | **H-tanh réfutée** sous cette forme |
| gate0 (portes ouvertes au départ) | 0,113 0,198 0,136 0,212 0,118 | 0,136 | 0,53 | H-portes partielle |
| **lingate0 (sans tanh et portes ouvertes)** | 0,031 0,039 0,047 0,250 0,030 | **0,039** | 0,87 | **réussit** (4 graines sur 5 < 0,05) |
| basin2 (maître + 1 %) | 0,077 0,169 0,066 0,045 0,056 | 0,066 | 0,77 | pas de lecture (e-basin) |
| basin1 (maître + 10 %) | 0,148 0,227 0,062 0,053 0,101 | 0,101 | 0,65 | descriptif : partiel |

- **Premier bras contraint qui réussit, avec r = ρ et la non-expansion vérifiée : lingate0.** Il faut à la fois retirer le chemin tanh et ouvrir les portes au départ. Aucun des deux seuls ne suffit : lin échoue, gate0 est partiel. (La ligne « les deux sont nécessaires ensemble » de la règle exige que gate0 échoue ; il est partiel, donc cette ligne ne s'applique pas strictement.)
- Portes en fin (médiane par tête) : lingate0 0,25 à 0,80 ; lin 0,08 à 0,30 ; linfree 0,05 à 0,15. Le libre n'a pas besoin d'ouvrir ses portes (il compense par un grand Ã, r = 8,5 à 12,3) ; le contraint, si.
- Descriptif bassin : depuis le maître perturbé (1 % ou 10 %), l'entraînement contraint finit à 0,05–0,23, au-dessus du point de départ pour basin2 sur 4 graines sur 5.
- Défaut de protocole à noter : le seuil e-basin (0,01) n'était pas compatible avec un bruit de 1 % sur un maître posé sur le bord.
- Prédictions : 2 sur 6 justes (linfree, basin1).
- Incident : `m2s_log.txt` a été réécrit par un `git pull --rebase` pendant que le calcul y écrivait ; le fichier visible s'arrête après la graine 0. Les résultats (30 fichiers JSON, `dval.json`) sont complets et sont la seule source de l'analyse.

## Analyse descriptive (2026-10-22, 10:45, sans entraînement, pas de règle) — `m2s_spectres.txt`

À la lumière de la session parallèle (`sessions/theorie_gradient_plat.md` : pas de composante radiale sur une tête saturée, ∂S absorbant sous GD).
- Répartition du budget en fin de run (moyenne sur les têtes) : P : B̃·C 0,004–0,043 (chemin tanh **fermé**), Ã diag. 0,24–0,42, Ã hors diag. 0,57–0,74 ; lin : Ã hors diag. 0,58–0,74 ; lingate0 : Ã diag. 0,42–0,63.
- Lemme (disques, preuve en une ligne) : si r_h ≤ ρ et des portes constantes z_h, toute valeur propre λ de x ↦ (1 − Z)x + Z·J x vérifie |λ − (1 − z_h)| ≤ ρ z_h pour une tête h (prendre h = argmax ‖x_h‖ dans le vecteur propre). Un pôle 0,99·e^{iθ} du maître exige donc une tête avec z ≥ z*(θ) : 0,125 (3°), 0,618 (10°), 0,911 (25°), 0,970 (45°). À ρ = 2 (fix2), z*(45°) ≈ 0,35.
- Mesure (linéarisation en x = 0, u = 0) : les pôles rapides (25°, 45°) ne sont atteints que si une porte dépasse le seuil. P et lin : z max 0,3–0,98, pôle 45° manqué partout (distance 0,34–0,60). lingate0 : 45° atteint sur 2/5 (z ≥ 0,98 sur deux têtes) ; sa seule graine en échec (s3, test 0,25) a z max = 0,74 et manque 25° et 45°. Le libre atteint 3–4 pôles sur 4 avec z ≈ 0,07 : il compense par r ≫ 1 (rayon z·r grand).
- Lecture provisoire : dans S, les portes portent la capacité de représenter les dynamiques rapides. Le contraint doit ouvrir ses portes ; le libre, non.
