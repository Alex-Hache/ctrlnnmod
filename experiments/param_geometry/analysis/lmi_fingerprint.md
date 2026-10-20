| Modèle LMI | PD | nx | rang / dim | κ(JJᵀ) médian | κ sur l'image (médian) |
|---|---|---|---|---|---|
| hinf_sqrtm | geotorch | 2 | 12 / 12 | 1.2e+04 | 1.2e+04 |
| hinf_sqrtm | geotorch | 4 | 9 / 32 | 8.4e+06 | 8.4e+06 |
| hinf_sqrtm | geotorch | 8 | 81 / 96 | 6.5e+08 | 6.5e+08 |
| hinf_sqrtm | bm | 2 | 10 / 12 | 7.9e+02 | 7.9e+02 |
| hinf_sqrtm | bm | 4 | 32 / 32 | 4.4e+06 | 4.4e+06 |
| hinf_sqrtm | bm | 8 | 75 / 96 | 2.3e+09 | 1.3e+09 |
| hinf_riccati | geotorch | 2 | 12 / 12 | 4.3e+03 | 4.3e+03 |
| hinf_riccati | geotorch | 4 | 29 / 32 | 3.9e+06 | 3.0e+06 |
| hinf_riccati | geotorch | 8 | 93 / 96 | 3.4e+09 | 9.4e+06 |
| hinf_riccati | bm | 2 | 12 / 12 | 5.9e+03 | 5.9e+03 |
| hinf_riccati | bm | 4 | 32 / 32 | 3.4e+05 | 3.4e+05 |
| hinf_riccati | bm | 8 | 95 / 96 | 2.4e+07 | 2.4e+07 |
| h2 | geotorch | 2 | 9 / 12 | 2.4e+301 | 3.7e+02 |
| h2 | geotorch | 4 | 27 / 32 | 8.0e+301 | 1.1e+07 |
| h2 | geotorch | 8 | 92 / 96 | 2.7e+302 | 3.0e+09 |
| h2 | bm | 2 | 10 / 12 | 1.4e+303 | 3.5e+03 |
| h2 | bm | 4 | 30 / 32 | 1.2e+303 | 9.9e+05 |
| h2 | bm | 8 | 94 / 96 | 3.4e+303 | 5.7e+07 |

## Cause du défaut de rang de H2 (vérifié)

`H2Linear._frame` pose σ = √(softmax_η(L)) avec η = γ². Donc Σσᵢ² = γ² pour tout paramètre, et
‖G‖_H2 = γ exactement (4 graines : 2,0000 ± 2e-5). La classe est la sphère {‖G‖_H2 = γ}, pas la
boule {‖G‖_H2 ≤ γ}. C'est une hypersurface : une valeur propre de JJᵀ est exactement nulle.
Conséquence : le modèle ne peut pas apprendre un système de norme H2 < γ.
Correction proposée : σ = s · √(softmax_η(L)), avec s = sigmoid(c) ∈ (0, 1).
Autre constat : à l'initialisation, max Re λ(A) ∈ [−0,006 ; −0,001], donc les pôles sont presque
sur l'axe imaginaire (départ près du bord de stabilité).

Note : l'égalité ‖G‖_H2 = γ est le théorème 3.3.12 de la thèse (Lean : `thm_3_3_12_trace`). L'apport ici est sa conséquence géométrique (rang de JJᵀ = dim − 1).

## H∞ : d'où vient κ (nx = 6, 8 graines, `hinf_probe.py`)

| Paramétrisation | PD | κ médian | cond(P) médian | corr(log κ, log cond P) | max Re λ(A) médian |
|---|---|---|---|---|---|
| sqrtm | geotorch | 5,6e6 | 100 | 0,91 | −0,64 |
| sqrtm | XᵀX | 4,7e5 | 42 | 0,86 | −1,56 |
| Riccati | geotorch | 1,6e7 | 100 | 0,69 | −0,020 |
| Riccati | XᵀX | 2,0e7 | 42 | 0,13 | −0,045 |

sqrtm : κ suit cond(P) (terme −½P⁻¹M : ∂A/∂M = P⁻¹ et ∂A/∂P ∝ P⁻¹MP⁻¹). Les blocs P et M ont des
valeurs singulières 10 à 1000 fois plus grandes que G, S, Q, H : déséquilibre d'échelle entre blocs.
Riccati : pas de P⁻¹, blocs équilibrés, κ peu lié à cond(P) ; les pôles sont presque sur l'axe
imaginaire. Hypothèse (non prouvée) : κ vient du départ près du bord de stabilité.
