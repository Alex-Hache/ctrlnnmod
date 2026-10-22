# Bibliographie — « gradient non nul, perte plate »

Session du 2026-10-22 (fiche : `gradient_non_nul_perte_plate.md`). Chaque entrée a été ouverte le 2026-10-22 (résumé arXiv, ou texte complet si indiqué). Les résultats sont paraphrasés. « Vérifié » dit ce qui a été lu ; « non vérifié » dit ce qui ne l'a pas été.

Notation de notre cas : f(θ) = L(Φ(θ)) ; tête saturée h ⇒ Φ homogène de degré 0 en θ_h = (Ã_h, B̃_h) ; Adam lr 1e-2, β = (0,9 ; 0,999), clip global 10, perte pleine (pas de mini-lot), pas de weight decay.

## A. Invariance d'échelle et taux effectif (piste 1)

### A1. Salimans, Kingma (2016) — Weight Normalization
- Lien : [arXiv:1602.07868](https://arxiv.org/abs/1602.07868v3) ; NIPS 2016 ([métadonnées NeurIPS](https://papers.neurips.cc/paper_files/paper/2016/file/ed265bc903a5a097f61d3ec064d96d2e-Metadata.json)). Vérifié : § 2.1 du texte complet ([ar5iv](https://ar5iv.labs.arxiv.org/html/1602.07868)).
- Résultat : avec w = g·v/‖v‖, le gradient en v est orthogonal à v. Sous GD sans moment, ‖v‖ croît à chaque pas (Pythagore). Un pas trop grand fait croître ‖v‖, ce qui réduit l'échelle effective : la méthode « s'auto-stabilise ».
- Hypothèses : GD sans moment. Les auteurs disent que l'argument ne tient pas strictement pour Adam ou le moment (effet qualitatif seulement).
- Lien avec notre cas : `proj` sur une tête saturée est une « weight normalization » par la norme r_h (pas la norme euclidienne). C'est la source de la piste 1. Le saut de r brut de 0,90 à ≈ 4–5 en 10 pas (T4) est cohérent avec l'auto-stabilisation.

### A2. van Laarhoven (2017) — L2 regularization versus batch and weight normalization
- Lien : [arXiv:1706.05350](https://arxiv.org/pdf/1706.05350). Vérifié : texte (résumé et résultats principaux).
- Résultat : sous normalisation, le taux effectif vaut η/‖w‖² pour SGD et ≈ η/‖w‖ pour Adam et RMSProp. La décroissance des poids (L2) agit seulement par ce taux effectif.
- Hypothèses : le résultat pour Adam vient de modèles simplifiés (gradients gaussiens, ReLU, ηλ petit) ; approximatif.
- Lien avec notre cas : prédit que, sous Adam, la dérive de r brut divise le pas effectif sur la tête par ≈ r_brut/ρ (pas par son carré). Nous démontrons une version exacte (conjugaison, ε = 0) dans `theorie_gradient_plat.md` § 2.

### A3. Hoffer, Banner, Golan, Soudry (2018) — Norm matters
- Lien : [NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/a0160709701140704575d499c997b6ca-Abstract.html), [arXiv:1803.01814](https://ar5iv.arxiv.org/html/1803.01814). Vérifié : texte (sections sur l'invariance d'échelle).
- Résultat : pas angulaire ≈ η/‖w‖² ; sans weight decay, la norme croît sans borne et le pas effectif décroît. Proposition : borner la norme dans une weight normalization.
- Lien avec notre cas : notre run P n'a pas de weight decay ⇒ pas d'équilibre de la norme (voir A6).

### A4. Arora, Li, Lyu (2019) — Theoretical analysis of auto rate-tuning by batch normalization
- Lien : [arXiv:1812.03981](https://arxiv.org/pdf/1812.03981) ; [ICLR 2019](https://iclr.cc/virtual/2019/poster/960). Vérifié : texte (Lemme 2.4, Théorème 3.1).
- Résultat : Lemme 2.4 : un poids invariant d'échelle est orthogonal à son gradient, et un pas de GD ajoute η²‖∇‖² au carré de la norme. Théorème 3.1 : GD plein lot, avec un η_w **quelconque** pour les poids invariants, atteint un point stationnaire à la vitesse T^(−1/2) (min_t ‖∇L‖).
- Hypothèses : perte C², dérivées secondes bornées, pas réglé pour les paramètres non invariants ; GD (pas Adam).
- Lien avec notre cas : pour une perte **lisse** et GD, l'invariance d'échelle ne produit **pas** de plateau à gradient non nul : la théorie prédit un gradient qui tend vers 0. Notre cas viole deux hypothèses : r_h n'est pas lisse (somme de normes), et l'optimiseur est Adam. Attention : ‖∇f(θ)‖ = ‖∇f(θ/‖θ‖)‖/‖θ‖ (notre Lean, `grad_smul`), donc une borne sur ‖∇f(θ)‖ mélange gradient angulaire et croissance de la norme. U1 mesure en W, ce qui évite ce biais.

### A5. Li, Arora (2020) — An exponential learning rate schedule for deep learning
- Lien : [arXiv:1910.07454](https://arxiv.org/abs/1910.07454) ; [ICLR 2020](https://iclr.cc/virtual/2020/poster/1896). Vérifié : résumé.
- Résultat : avec normalisation, SGD + moment + weight decay équivaut à SGD + moment avec un taux qui croît exponentiellement.
- Lien avec notre cas : même principe (l'échelle des poids est un taux d'apprentissage caché). Pas d'usage direct.

### A6. Wan, Zhu, Zhang, Sun (2021) — Spherical Motion Dynamics
- Lien : [arXiv:2006.08419](https://arxiv.org/pdf/2006.08419v4.pdf) ; [NeurIPS 2021](https://neurips.cc/virtual/2021/poster/26433). Vérifié : texte (théorème principal).
- Résultat : avec SGD (± moment) **et weight decay**, la mise à jour angulaire converge vers √(2λη) (√(2λη/(1+α)) avec moment).
- Hypothèses : λ > 0. Sans weight decay, la norme croît toujours et il n'y a pas d'équilibre.
- Lien avec notre cas : λ = 0 dans T4 ⇒ pas d'équilibre ; le pas angulaire décroît lentement.

### A7. Heo et al. (2021) — AdamP
- Lien : [arXiv:2006.08217](https://arxiv.org/abs/2006.08217) ; [ICLR 2021](https://iclr.cc/virtual/2021/poster/2880). Vérifié : résumé.
- Résultat : le moment (SGD, Adam) accélère la décroissance du pas effectif sur les poids invariants d'échelle, par la composante radiale de la mise à jour. Correctif : retirer la composante radiale à chaque pas. Support empirique (pas de théorème annoncé dans le résumé).
- Lien avec notre cas : prédit une croissance de la norme sous Adam. T4 montre une **baisse** de r brut entre t ≈ 50 et t ≈ 100–500, puis une hausse lente. r brut n'est pas ‖θ_h‖, et C bouge aussi : la comparaison est indirecte.

### A8. Kosson, Messmer, Jaggi (2024) — Rotational equilibrium
- Lien : [PMLR v235 (ICML 2024)](https://proceedings.mlr.press/v235/kosson24a.html), [arXiv:2305.17212](https://ar5iv.labs.arxiv.org/html/2305.17212). Vérifié : texte (tableau 1).
- Résultat : sous AdamW, la taille RMS de la mise à jour par coordonnée vaut η√((1−β₁)/(1+β₁)), indépendante de l'échelle du gradient (modèle de marche aléatoire) ; rotation d'équilibre √(2ηλ(1−β₁)/(1+β₁)).
- Hypothèses : régime d'équilibre, weight decay λ > 0.
- Lien avec notre cas : confirme qu'Adam fait des pas de taille fixe en θ, quelle que soit l'échelle du gradient. Pas d'équilibre sans λ.

### A9. Roburin et al. (2020) — Spherical perspective on learning with normalization layers
- Lien : [arXiv:2006.13382](https://arxiv.org/abs/2006.13382v3). Vérifié : résumé seulement. **Non vérifié** : la formule du taux effectif d'Adam (le résumé n'en donne pas).
- Résultat annoncé : première expression du taux effectif d'Adam pour des poids invariants ; reformulation sur la sphère unité.
- Lien avec notre cas : à lire si l'on veut une dynamique d'Adam sur la sphère plus fine que notre conjugaison (§ 2 de la théorie).

## B. Pas trop grand face à la courbure (piste 3)

### B1. Kodryan, Lobacheva, Nakhodnov, Vetrov (2022) — Three regimes on the sphere
- Lien : [arXiv:2209.03695](https://arxiv.org/pdf/2209.03695) ; [NeurIPS 2022](https://neurips.cc/virtual/2022/poster/53275). Vérifié : texte (régimes, protocole, limites).
- Résultat : sur une sphère de norme fixe, selon le taux effectif η/ρ² : (1) convergence ; (2) **équilibre chaotique** : la perte oscille autour d'un plateau au-dessus du minimum, et la norme du gradient effectif reste non nulle ; (3) divergence (marche aléatoire).
- Hypothèses : SGD projeté, mini-lots (128), norme totale fixe. Adam non traité (laissé en perspective par les auteurs).
- Lien avec notre cas : le régime (2) a exactement la signature observée (plateau, perte qui monte à ≈ 45 % des pas, gradient non nul). Différences : notre run est plein lot, sous Adam, sans norme fixe. Le bras « gradient projeté » de T4 (rétraction radiale, norme petite) ressemble à leur protocole avec un taux effectif grand : candidat au régime (3) (test 0,77).

### B2. Cohen, Kaur, Li, Kolter, Talwalkar (2021) — Edge of stability
- Lien : [arXiv:2103.00065](https://arxiv.org/abs/2103.00065) ; [ICLR 2021](https://iclr.cc/virtual/2021/poster/2577). Vérifié : résumé.
- Résultat : en GD plein lot, la plus grande valeur propre du hessien monte jusqu'à ≈ 2/η et y reste. La perte n'est pas monotone à court terme, mais décroît à long terme.
- Lien avec notre cas : notre perte ne décroît plus sur 1000 pas : EoS seul ne suffit pas (il prédit un progrès à long terme).

### B3. Cohen et al. (2022) — Adaptive gradient methods at the edge of stability
- Lien : [arXiv:2207.14484](https://arxiv.org/abs/2207.14484). Vérifié : résumé.
- Résultat : en plein lot, la netteté préconditionnée d'Adam monte jusqu'à un seuil (38/η pour β₁ = 0,9) et y reste ; l'entraînement continue d'avancer.
- Lien avec notre cas : seuil 38/η = 3800 à η = 1e-2. Test possible (puissance itérée avec le préconditionneur d'Adam), non fait. Comme B2, ce mécanisme prédit un progrès lent, pas un plateau.

## C. Non-différentiabilité (piste 2)

### C1. Boyd, Mutapcic (2008) — Subgradient methods (notes EE364b)
- Lien : [Stanford EE364b](https://see.stanford.edu/materials/lsocoee364b/02-subgrad_method_notes.pdf). Vérifié : texte (§ 1, § 2.3, § 3.2).
- Résultat : la méthode du sous-gradient n'est pas une méthode de descente : la valeur peut monter. À pas constant α, la meilleure valeur atteint seulement un voisinage de l'optimum, d'écart G²α/2.
- Hypothèses : fonction convexe, sous-gradients bornés par G.
- Lien avec notre cas : r_h = Σ‖A_hk‖₂ + Σ‖B_hq‖·Σ‖C_qk‖ est une somme de normes (type « group lasso »), non lisse aux blocs nuls et aux valeurs singulières égales. Le maître est sur ces plis (B̃ = 0, A hors diagonale = 0, blocs diagonaux = ρ·rotation, donc σ₁ = σ₂). Près d'un pli, un pas constant donne un gradient qui ne tend pas vers 0 et une valeur qui oscille.

## D. Adam et projection (piste 4)

### D1. Reddi, Kale, Kumar (2018) — On the convergence of Adam and beyond
- Lien : [arXiv:1904.09237](https://arxiv.org/abs/1904.09237) ; [ICLR 2018](https://iclr.cc/virtual/2018/poster/78). Vérifié : résumé.
- Résultat : Adam peut ne pas converger, même sur un problème convexe simple ; cause : la moyenne mobile exponentielle des gradients au carré.
- Hypothèses : cadre stochastique / en ligne.
- Lien avec notre cas : avertissement général. Notre cas est déterministe ; ce contre-exemple ne s'applique pas tel quel.

## E. Reparamétrisation et géométrie (contexte de la thèse)

### E1. Niculae (2020) — Optimizing with constraints: reparametrization and geometry
- Lien : [vene.ro](https://vene.ro/blog/mirror-descent.html) (14/09/2020). Vérifié : texte.
- Résultat : sur une quadratique dans [0, 1]², la reparamétrisation sigmoïde a un jacobien x(1−x) qui écrase les pas près du bord ; la descente miroir (gradient naturel dans le dual) supprime ce ralentissement.
- Lien avec notre cas : `proj` est l'autre extrême : J = I à l'intérieur, puis, sur le bord, J perd la direction radiale et ‖J‖ ∝ ρ/r_brut. L'écrasement vient de la dérive de r_brut, pas d'une sigmoïde.

### E2. Amid, Warmuth (2020) — Reparameterizing mirror descent as gradient descent
- Lien : [NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/604b37ea63ea51fa5fb3d8a89ec056e6-Abstract.html), [arXiv:2002.10487](https://arxiv.org/abs/2002.10487). Vérifié : résumé.
- Résultat : la descente miroir en temps continu s'écrit comme un flot de gradient dans d'autres paramètres. Le temps discret reste ouvert.
- Lien avec notre cas : cadre des métriques induites (T1–T3). Notre phénomène est discret (pas, Adam) : hors de ce résultat.

### E3. Levin, Kileel, Boumal (2024) — The effect of smooth parametrizations on nonconvex optimization landscapes
- Lien : [arXiv:2207.03512](https://arxiv.org/abs/2207.03512) (Mathematical Programming, DOI 10.1007/s10107…). Vérifié : résumé.
- Résultat : cadre général reliant points critiques et minima locaux de f∘φ à ceux de f ; la relation dépend surtout de φ.
- Lien avec notre cas : la piste « point stationnaire parasite » est déjà écartée par M1 (‖Jᵀg‖/(σ_max‖g‖) = 0,15–0,9). Notre Φ n'est pas lisse (max, normes) : leur cadre lisse ne s'applique pas directement.

### E4. Leplat (2025) — Constrained optimization via constraint-induced geometry
- Lien : [arXiv:2508.18764](https://arxiv.org/abs/2508.18764). Vérifié : résumé.
- Résultat : des trajectoires intérieures convergentes retrouvent les conditions KKT (orthant, simplexe, boîte) ; pas bornés loin de 0.
- Lien avec notre cas : faible. Nos trajectoires ne sont pas intérieures (collées à ∂S).

## Synthèse pour la session

1. Piste 1 est solide comme **structure** (A1–A4, A8) : sur une tête saturée, l'échelle brute agit comme un diviseur du taux (η/‖θ‖² en GD, η/‖θ‖ en Adam). Mais, pour une perte lisse et GD, elle prédit une descente lente et un gradient qui tend vers 0 (A4), pas un plateau oscillant.
2. La signature observée (oscillation, plateau, gradient non nul, perte pleine) correspond au régime « équilibre chaotique » (B1) ou à un pas constant sur un problème non lisse (C1).
3. C1 a un argument propre à notre classe : le maître est sur les plis de r. C'est la piste à tester en premier (voir la théorie, § 4).

Après tests (même jour, `theorie_gradient_plat.md` § 7–8) : piste 2 réfutée (le plateau n'est pas un point KKT caché par les plis) ; le plateau dépend de l'état d'Adam (lecture provisoire, 3/5). Les entrées B1 et D1 restent les plus proches du phénomène observé.
