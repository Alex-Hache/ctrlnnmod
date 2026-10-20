## Matrices définies positives (n = 6, départ H₀ = I, perte ½‖H − H*‖²)

Itérations jusqu'à ‖H − H*‖²/‖H*‖² < 1e-10 (médiane sur 2 graines, meilleur pas). Empreinte de JJᵀ au départ : rang / 21 et κ.

| Paramétrisation | rang JJᵀ en I | κ en I | κ bien cond. | bien conditionnée (κ=10) | mal conditionnée (κ=10³) | valeurs propres groupées | près du bord (10⁻⁴…10⁻¹) |
|---|---|---|---|---|---|---|---|
| H libre + projection | 21 / 21 | 1.0 | 1 | 1 | 1 | 1 | 1 |
| XᵀX + εI (ctrlnnmod, REN) | 21 / 21 | 1.0 | 10 | 43 | 3812 | 8 | 11545 |
| LLᵀ + εI (Cholesky) | 21 / 21 | 2.0 | 31 | 223 | > 20 000 (6e-10) | 44 | > 20 000 (1e-07) |
| Cholesky, diag. exp | 21 / 21 | 2.0 | 93 | 620 | > 20 000 (1e-05) | 42 | > 20 000 (9e-04) |
| exp(S) | 21 / 21 | 1.0 | 100 | 373 | > 20 000 (2e-05) | 7 | > 20 000 (6e-03) |
| geotorch PD, Q par expm | 6 / 21 | ∞ | 333 | 2266 | > 20 000 (4e-05) | 86 | > 20 000 (7e-03) |
| geotorch PD, Q par Cayley | 6 / 21 | ∞ | 303 | 2652 | > 20 000 (6e-05) | 78 | > 20 000 (7e-03) |

## Matrices orthogonales (SO(4), deux plans tournés de φ*, départ Q = I)

| Paramétrisation | φ* = 0,1π | 0,5π | 0,8π | 0,95π |
|---|---|---|---|---|
| geotorch PD, Q par expm | 11 | 14 | 16 | 19 |
| geotorch PD, Q par Cayley | 12 | 1 | 484 | > 20 000 (2e-05) |
| Q libre + rétraction polaire | 1 | 1 | 1 | 1 |

Prédiction Cayley (Lean `cayley_induced_flow`) : vitesse angulaire × cos⁴(φ/2) ; cos⁴(φ/2) vaut 9.5e-01, 2.5e-01, 9.1e-03, 3.8e-05 pour φ = 0,1π ; 0,5π ; 0,8π ; 0,95π.
