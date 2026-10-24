# Clôture de la branche « blocs de 2 » (oscillateurs) — 2026-10-24

Décision de l'utilisateur : arrêter les classes hors thèse et les problèmes jouet. Cette branche est close. Détails : `m1.md`, `m2*.md`, `m2q.md`, `m2r.md`, `m2s.md`, `m2t.md`, `sessions/theorie_gradient_plat.md`.

## Établi (critères écrits avant calcul, 5 graines ; un banc, une classe hors thèse)
- Le maître est dans S, représentable exactement avec ses têtes sur ∂S, sans chemin tanh, avec des portes ouvertes (z jusqu'à 0,97).
- Aucune métrique d'optimiseur testée (Adam à divers lr, lr par bloc, flot GD, gradient projeté) ne garde l'apprentissage dans S.
- La borne du certificat n'est pas lâche (le plus fin de même forme : 1,5–2× seulement). Une borne plus fine seule n'aide pas.
- Une métrique de certificat apprise par tête aide en partie (0,19 ; 0,14 avec la borne fine, contre 0,29).
- Sans chemin tanh **et** avec des portes ouvertes au départ, la classe contrainte réussit dans S (0,039).
- Sur une tête saturée, le gradient n'a pas de composante radiale ; ∂S est absorbant sous GD (Lean, `Ext/ScaleInvariance`).
- Au plateau, le point n'est pas KKT ; une remise à zéro d'Adam en sort sur 3/5.

## Lecture provisoire
Une fois sur ∂S (absorbant), le modèle contraint reste dans la vallée où il est entré. L'issue dépend de l'état au contact (portes, chemin tanh) et de l'optimiseur. La seule vraie propriété de S mise en évidence : avec des portes z, une tête ne peut placer ses pôles que dans le disque D(1 − z, ρz) (lemme des disques, `m2s.md`).

## Ce qui sert pour la suite
Méthode (critères avant calcul, contrôles, témoins, analyse codée avant résultats) et outils de diagnostic : contact avec ∂S, rang de J, κ, déplacement cumulé contre échelle des paramètres, témoins aléatoires, non-expansion empirique, spectre linéarisé contre pôles cibles.
