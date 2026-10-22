"""Q0 M2-s — applique les règles de `analysis/m2s.md`, sans jugement. Écrit avant les résultats.
Usage : python experiments/q0/analyze_m2s.py"""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
D, D4 = os.path.join(HERE, "results_m2s"), os.path.join(HERE, "results_m2q")
RHO, SEEDS = 0.999, range(5)
ARMS = ("lin", "linfree", "gate0", "lingate0", "basin2", "basin1")
CONSTRAINED = ("lin", "gate0", "lingate0", "basin2", "basin1")
lab = lambda x: "réussit" if x < 0.05 else "échoue" if x > 0.2 else "partiel"  # noqa: E731

Pm = st.median(json.load(open(os.path.join(D4, f"t4_p_s{s}.json")))["test"] for s in SEEDS)
Fm = st.median(json.load(open(os.path.join(D4, f"t4_free_s{s}.json")))["test"] for s in SEEDS)
DV = {(r["arm"], r["seed"]): r["E"] for r in json.load(open(os.path.join(D, "dval.json")))} \
    if os.path.exists(os.path.join(D, "dval.json")) else {}
print(f"Références T4 : P {Pm:.4f} ; libre {Fm:.4f}")
M, OK = {}, {}
for arm in ARMS:
    R = [json.load(open(os.path.join(D, f"{arm}_s{s}.json"))) for s in SEEDS]
    M[arm] = st.median(r["test"] for r in R)
    ctl = []
    if arm in ("lin", "linfree", "lingate0"):
        ctl.append(("e-lin", all(r["B_norm_end"] == 0.0 for r in R)))
    if arm == "basin2":
        ctl.append(("e-basin", all(r["test0"] < 0.01 for r in R)))
    if arm in CONSTRAINED:
        ctl.append(("e-fin", all(r["rW_end"] <= RHO + 1e-5 for r in R)))
        ctl.append(("e-val", all((arm, s) in DV and DV[(arm, s)] <= 1 + 1e-4 for s in SEEDS)))
    OK[arm] = all(v for _, v in ctl)
    g = (Pm - M[arm]) / (Pm - Fm)
    print(f"  {arm:8s} test : " + " ".join(f"{r['test']:.4f}" for r in R) + f" | médiane {M[arm]:.4f} ({lab(M[arm])}) | g = {g:.2f}"
          f" | test à t=0 : " + " ".join(f"{r['test0']:.3g}" for r in R) + " | r fin : " + " ".join(f"{r['rW_end']:.3g}" for r in R)
          + f" | statuts {[r['status'] for r in R]}")
    print("    z fin (médiane par tête) : " + " ".join(f"{st.median(r['z_end'][h] for r in R):.2f}" for h in range(8))
          + " | contrôles : " + ", ".join(f"{n} {'ok' if v else 'ÉCHEC'}" for n, v in ctl)
          + ((" | E : " + " ".join(f"{DV[(arm, s)]:.6f}" for s in SEEDS if (arm, s) in DV)) if arm in CONSTRAINED else ""))

print()
L = {a: lab(M[a]) for a in ARMS}
if not OK["linfree"] or L["linfree"] != "réussit":
    print("Validité de lin : linfree ne réussit pas (ou contrôle en échec) → pas de lecture de H-tanh")
elif not OK["lin"]:
    print("H-tanh : contrôle en échec pour lin → pas de lecture")
else:
    print("H-tanh :", {"réussit": "CONFIRMÉE : sans le chemin tanh, la classe contrainte apprend ; le chemin tanh est le piège",
                       "échoue": "RÉFUTÉE sous cette forme : même sans tanh, la contrainte bloque",
                       "partiel": f"partiel (g = {(Pm - M['lin']) / (Pm - Fm):.2f})"}[L["lin"]])
if OK["gate0"]:
    print("H-portes :", {"réussit": "CONFIRMÉE", "échoue": "RÉFUTÉE",
                         "partiel": f"partiel (g = {(Pm - M['gate0']) / (Pm - Fm):.2f})"}[L["gate0"]])
else:
    print("H-portes : contrôle en échec → pas de lecture")
if OK["lingate0"] and L["lingate0"] == "réussit" and L["lin"] == "échoue" and L["gate0"] == "échoue":
    print("Combinaison : les deux sont nécessaires ensemble")
elif OK["lingate0"]:
    print(f"lingate0 : {L['lingate0']}")
if OK["basin2"]:
    print("H-bassin :", {"réussit": "SOUTENUE : la région du maître est stable sous l'entraînement contraint (problème d'accès)",
                         "échoue": "contre : même près du maître, l'entraînement contraint s'en éloigne",
                         "partiel": "partiel"}[L["basin2"]])
else:
    print("H-bassin : contrôle en échec → pas de lecture")
print(f"basin1 (descriptif) : {L['basin1']}")
print("Prédictions : linfree réussit, lin partiel, gate0 échoue, lingate0 partiel, basin2 réussit, basin1 partiel → observé :", L)
