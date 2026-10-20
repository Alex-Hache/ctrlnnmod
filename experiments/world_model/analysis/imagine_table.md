| World model | H imagination | runs | RMSE réelle moyenne (rad) | pire run | RMSE imaginée | réel / imaginé | gradient max |
|---|---|---|---|---|---|---|---|
| MLP 1 pas | 250 | 2 | 0.121 (-36 %) | 0.128 | 0.228 | 0.55 | 1.2e+00 |
| MLP multi-pas | 250 | 4 | 0.105 (-44 %) | 0.109 | 0.040 | 2.62 | 3.7e-02 |
| GRU | 250 | 4 | 0.132 (-30 %) | 0.270 | 0.085 | 1.77 | 1.7e+12 |
| REN contractant | 250 | 4 | 0.083 (-56 %) | 0.089 | 0.050 | 1.65 | 3.3e-02 |
| MLP 1 pas | 50 | 2 | 0.120 (-36 %) | 0.125 | 0.276 | 0.50 | 5.3e-02 |
| MLP multi-pas | 50 | 2 | 0.113 (-40 %) | 0.119 | 0.042 | 2.68 | 3.3e-02 |
| GRU | 50 | 2 | 0.125 (-33 %) | 0.132 | 0.061 | 2.10 | 2.8e-02 |
| REN contractant | 50 | 2 | 0.096 (-49 %) | 0.100 | 0.055 | 1.75 | 3.0e-02 |

Référence sans politique (a_t = q_ref(t+1)) : RMSE = 0.188 rad.
