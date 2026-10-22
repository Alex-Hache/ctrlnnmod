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
