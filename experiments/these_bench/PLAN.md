# Plan de recherche — Q0 sur les classes de la thèse et les vrais benchmarks

Écrit le 2026-10-24. **À lire en premier par la session qui démarre.** Remplace la branche « blocs de 2 », close (`experiments/q0/analysis/CLOTURE_blocs.md`).

## 0. Cadre fixé par l'utilisateur
- Classes de la thèse uniquement : **GNSSM initialisé par la BLA**, partie linéaire paramétrée par les LMI de la thèse (α-stabilité, H∞ ; H2 en option).
- Vrais benchmarks uniquement : **Silverbox**, puis **Wiener-Hammerstein** (Schoukens, nonlinearbenchmark.org).
- Pas de problème jouet, pas de classe hors thèse.
- Style : ASD-STE100, en français. Rôle : pair qui contredit franchement. Avancer méthodiquement.

## 1. Question

**Q0-thèse** : à initialisation BLA égale, un GNSSM dont la partie linéaire est contrainte (α-stable, H∞) apprend-il moins bien que le même GNSSM libre ? Si oui, où l'optimisation bloque-t-elle ?

Sous-questions, dans l'ordre :
1. Reproduire la méthode de la thèse : BLA, puis GNSSM libre initialisé par la BLA. Niveau d'erreur de référence (RMS de simulation sur le test du benchmark).
2. Même chose avec la partie linéaire contrainte : α-stable, puis H∞ (γ fixé, puis γ appris).
3. Si un écart existe : diagnostic (la BLA est-elle dans S ? sa marge ? contact avec ∂S pendant l'apprentissage ? κ(JJᵀ) le long de la trajectoire ? la partie non linéaire compense-t-elle hors de S ?).
4. Seulement ensuite : leviers (paramétrisation de P, décroissance de la marge γ_t → γ, métrique).

## 2. Hypothèses à tester (aucune n'est acquise)
- H1 : la contrainte ne coûte rien quand la BLA est à l'intérieur de S avec une marge nette (contrôle positif attendu).
- H2 : la contrainte coûte quand la BLA est près du bord (marge < 1 %), comme sur les blocs.
- H3 : avec la contrainte, la partie non linéaire f(x, u) prend en charge ce que la partie linéaire ne peut plus faire. L'apprentissage se bloque dans cette répartition.
- H4 : le conditionnement de la paramétrisation de P (init geotorch, λ_min(P) petit, κ de 1e4 à 1e9 au départ, cf. `analysis/lmi_gd.md`) ralentit la descente sur les classes H∞.
Chaque hypothèse reçoit un critère « confirme / réfute » écrit **avant** le calcul.

## 3. Étapes

### É0 — Mise en place (première session)
1. Modal (§ 5). Vérifier `modal run compute/modal_job.py --cmd "python -c 'print(1)'"` avec `MODAL_GPU=none`.
2. Données : un job Modal télécharge Silverbox et Wiener-Hammerstein (paquet `nonlinear_benchmarks`, accès Internet sur Modal) et les écrit sur le volume `ctrlnnmod-data` ; rapatrier avec `compute/fetch_volume.py`. Le conteneur local n'a pas accès à ces sites (téléchargement direct en échec le 2026-10-24). Autre voie : l'utilisateur joint les fichiers.
3. Lire et vérifier le code : `ctrlnmod/models/ssmodels/continuous/general/gnssm.py` (temps continu : vérifier l'intégrateur et le pas d'échantillonnage du benchmark), `ctrlnmod/models/ssmodels/continuous/linear/` (SSLinear, L2BoundedLinear, H2Linear), `ctrlnmod/lmis/`. **Aucun code d'initialisation BLA n'existe dans le dépôt** (seulement `ctrlnmod/matlab/`). À écrire : BLA par sous-espace (N4SID / MOESP) ou par estimation fréquentielle, puis projection dans S si la BLA n'y est pas.
4. Demander à l'utilisateur les chiffres de référence de la thèse (erreurs obtenues sur ces benchmarks, réglages) pour valider la reproduction.

### É1 — Reproduction (critère : erreur du GNSSM libre initialisé BLA proche de la référence de la thèse, tolérance fixée avec l'utilisateur)
### É2 — Contraint contre libre, même init (α-stable, H∞), 5 graines, critères avant calcul
### É3 — Diagnostic, seulement si É2 montre un écart
### É4 — Leviers, seulement après É3

Une session = une étape. Mettre `ROADMAP.md` à jour à la fin.

## 4. Méthodologie (règles apprises, obligatoires)
- Objectif écrit en tête de session. Une seule question.
- **Critères écrits et poussés dans un commit avant le premier run** (confirme / réfute, seuils, majorité sur les graines). Analyse codée dans un script écrit **avant** les résultats ; aucune règle changée après coup ; tout écart est écrit en clair.
- Contrôles de validité pour chaque bras (le bras teste-t-il ce qu'il doit tester ?) ; témoins qui excluent l'effet de l'optimiseur avant toute conclusion forte. Lectures « provisoires ».
- Test rapide, lecture des diagnostics, puis seulement le balayage.
- Vérifier que la cible (BLA, solution de référence) est dans S, et sa **marge**.
- Seuils compatibles avec la précision (float32 contre float64 ≈ 1e-7 relatif) et avec le bruit.
- Le tirage aléatoire de PyTorch dépend du type par défaut : générer les données dans le type d'origine.
- `scalaire / tenseur` = scalaire × (1/tenseur) en PyTorch : pour un facteur qui doit valoir 1, écrire min(1, ρ/r).
- Sauvegarder modèles **et état de l'optimiseur** en fin de run.
- Ne jamais faire `git pull --rebase` pendant qu'un calcul écrit dans un fichier suivi par git (mettre les journaux en cours dans `.git/info/exclude`).
- Écrire heures et chiffres depuis les journaux et les commits, jamais de mémoire. Relire chaque chiffre avant d'envoyer un rapport.
- Notifier l'utilisateur à chaque série finie ; pas d'interprétation avant l'application des règles.

## 5. Modal (calcul et accès Internet)
Le conteneur d'une nouvelle session n'a pas `~/.modal.toml`. Procédure (le token n'apparaît jamais dans le chat) :
1. `pip install --break-system-packages -q 'modal[api-proxy-support]'` (le support proxy est indispensable ici).
2. `modal token new --no-verify` en arrière-plan ; il affiche une URL `https://modal.com/token-flow/...`.
3. Envoyer l'URL à l'utilisateur ; il l'ouvre, connecté à Modal, et valide.
4. Attendre `~/.modal.toml`, puis `chmod 600 ~/.modal.toml`.
Usage :
- Un job : `OUT_DIR=<dossier> MODAL_GPU=L4 modal run compute/modal_job.py --cmd "python experiments/..."` (`MODAL_GPU=none` : CPU seul, 8 cœurs ; `MODAL_CPU`, `MODAL_MEM` réglables).
- Plusieurs jobs : `compute/launch_spaced.sh` (MAXPAR = 8, `modal run --detach`). Offre Starter : 10 GPU simultanés au plus.
- Retours de fonction < 1 Mo ; gros fichiers sur le volume `ctrlnnmod-data` (monté sur `/data`), rapatriés par `compute/fetch_volume.py`.
- Coût : ≈ 20,5 $ consommés en octobre 2026 (crédits). Vérifier le coût avant un balayage.
- Remarque du 2026-10-24 : depuis ce conteneur, `modal.com` est refusé par le proxy (403) mais `api.modal.com` répond ; la procédure ci-dessus a fonctionné le 2026-10-20. Si elle échoue, le dire et demander les fichiers de données à l'utilisateur.

## 6. Ce qui reste utile de la branche close
Outils de diagnostic : contact avec ∂S, rang et κ de J (`experiments/q0/m1_rank.py`), déplacement cumulé contre échelle des paramètres, témoins aléatoires, non-expansion empirique, spectre linéarisé contre pôles cibles. Résultats Lean : `Ext/InducedRate` (T1, T2), `Ext/FlowDecomposition` (T3), `Ext/ScaleInvariance`.
