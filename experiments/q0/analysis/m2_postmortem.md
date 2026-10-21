# Post-mortem — session M2 du 2026-10-21 (Claude Opus 5.5)

Objectif fixé : M2 seule. Résultat : les mesures sont valides, mais j'ai livré une conclusion fausse sur le mécanisme et une conclusion trop forte sur l'affirmation centrale. L'utilisateur a trouvé le problème par une question simple (« le maître est au centre de S ? pourquoi touche-t-on le bord si vite ? »). Je ne l'avais pas posée moi-même.

## Ce qui reste valide

- Protocole et critères poussés avant le premier run (`469bb84`, 08:27). Contrôles c1, c2 passés.
- Mesures (Adam, 5 graines, médiane du test) : libre 0,0015 hors de S (r final 47 à 63) ; `blockp` init saturée 0,33 ; `blockp` init à 0,9ρ 0,28 ; REN 0,0051. Fichiers : `results_m2/`, `m2_*_log.txt`, `analyze_m2.py`.
- Le maître est représentable exactement avec les 8 têtes sur ∂S (`m2_teacher_on_boundary.py`, perte 1e-29).
- GD n'est pas un banc valide à 2000 itér. (REN 0,59, graine 0).
- Diagnostic du premier pas (`m2_why_boundary.py`) : l'init à 0,9ρ touche ∂S au premier pas d'Adam à cause de l'échelle du pas (0,01 par coordonnée) face à des paramètres ≈ 0,005. Des signes aléatoires de même taille donnent le même saut (r 0,90 → 1,6–1,8).

## Ce qui n'est pas établi

- Ensemble ou métrique : **non tranché**. Le contact avec ∂S vient en partie de la métrique d'Adam (J·D·Jᵀ).
- Le libre a-t-il besoin de sortir de S pour réussir ? Inconnu : sa sortie au départ vient du même effet d'échelle.
- Pourquoi r du libre monte à 50 pendant que la perte baisse : inconnu.

## Erreurs, par gravité

1. **Conclusion forte sans témoin.** J'ai écrit « l'affirmation centrale est contredite » dans `ROADMAP.md`, `m2.md` et le document `Contexte`, et je l'ai annoncée (`7412bed`, 10:56). Aucun témoin n'excluait un effet de l'optimiseur. Corrigé à 11:36 (`cc6a47d`), seulement après la question de l'utilisateur.
2. **Mécanisme inventé.** J'ai lu « contact au pas 1 » et j'ai écrit « le gradient pousse hors de S dès le départ ». Je n'ai pas testé cette cause. Le test qui la réfute (un pas de signes aléatoires) prend quelques secondes.
3. **Diagnostic disponible, non lu.** Le test rapide montrait déjà : contact au pas 1 pour P-int, sortie de S du libre avant 50 itér., aux deux pas d'Adam. La règle 3 de la roadmap demande de lire l'écart du pas. Je n'ai pas comparé le pas à l'échelle des paramètres. Cela aurait pris une ligne de code.
4. **Bras P-int mal conçu.** J'ai choisi 0,9ρ sans vérifier qu'un seul pas d'Adam garde le modèle dans S. Les critères n'avaient pas de condition de validité pour ce bras (par ex. « P-int reste dans S au moins N pas, sinon ne pas conclure »). Le bras n'a donc pas testé un run intérieur.
5. **Règle de choix du pas biaisée.** La plus faible perte à 500 itér. favorise les grands pas. Elle a retenu lr = 1e-2 pour tous les bras blocs, soit le pas qui cause l'effet d'échelle. Je n'ai pas vu l'interaction.
6. **Marge du maître mal présentée.** Mes résumés disaient « le maître est à l'intérieur ». Le bon fait est : le maître est **près du bord** (marge 0,006 à 0,009, < 1 % de ρ). La formulation a pu suggérer « au centre ».
7. **Langue.** La plupart de mes messages intermédiaires étaient en anglais. La consigne est : la langue du dernier message de l'utilisateur (français).
8. **Bruit de session.** Install pip bloquée (≈ 10 min perdues) ; moniteurs en double ; attentes avec un mauvais motif (espaces de remplissage du log) arrêtées par la limite de temps ; une douzaine de commits partiels.

Écart au protocole, documenté et justifié : GD arrêté après la graine 0 (validité rejetée nettement). Je le garde comme décision acceptable.

## Ce que je ferais autrement

- Avant de choisir un pas d'Adam : comparer lr à l'échelle des paramètres, par bloc. Ajouter un témoin « signes aléatoires, même taille ».
- Pour chaque bras : écrire une condition de validité qui vérifie que le bras teste bien ce qu'il doit tester.
- Ne rien écrire comme conclusion dans `ROADMAP.md` ou `Contexte` sans un témoin qui exclut l'effet de l'optimiseur. Écrire « lecture provisoire » sinon.
- Pour chaque mécanisme proposé : écrire le test qui le réfute, et le lancer s'il coûte moins d'une minute.
- Me poser la question de l'utilisateur avant lui : où est le maître (marge), où est l'init, quelle taille a le pas.

## Recommandation pour la prochaine session

M2-bis, avant toute variation de la marge du maître :
- Bras : P-int (0,9ρ) et libre (même W₀), Adam à lr 1e-3 (premier pas : r 0,90 → 0,93, pas de contact). Éventuellement lr 3e-4 en témoin.
- Validité de P-int : reste dans S (marge > 0) au moins jusqu'à une baisse nette de la perte ; sinon, ne pas conclure.
- Témoin : pas de signes aléatoires de même taille, pour séparer gradient et échelle.
- Critères à écrire et pousser avant calcul. Question : P-int réussit-il en restant dans S ? Le libre reste-t-il dans S à petit pas ?

## Chronologie (commits)

| heure | commit | contenu |
|---|---|---|
| 08:27 | `469bb84` | protocole et critères, avant calcul |
| 08:42–08:48 | `2b59277` … `40eb36f` | script, test rapide, choix des pas, maître représentable sur ∂S |
| 08:57 | `67b540d` | graine 0 complète ; GD rejeté |
| 09:12–09:13 | `ab28398` … `aebcd95` | graines 1 à 4 (Adam) |
| 10:56 | `7412bed` | M2 « fait » : conclusion trop forte (erreurs 1 et 2) |
| 11:36 | `cc6a47d` | correction après la question de l'utilisateur |
