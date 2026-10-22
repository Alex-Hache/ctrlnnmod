"""Q0 M2-r — applique les règles de `analysis/m2r.md`, sans jugement. Écrit avant les résultats.
Usage : python experiments/q0/analyze_m2r.py"""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
D, D4 = os.path.join(HERE, "results_m2r"), os.path.join(HERE, "results_m2q")
RHO, SEEDS, ARMS = 0.999, range(5), ("mid", "alpha", "midalpha")
maj = lambda flags: sum(flags) >= 3  # noqa: E731

# ------------------------------------------------------------------ T8
print("== T8 — finesse du certificat (modèles finaux de T4)")
p8 = os.path.join(D, "t8.json")
if os.path.exists(p8):
    T8 = {(r["arm"], r["seed"]): r for r in json.load(open(p8))}
    k1 = all(r["r_box_lo"] <= r["r_mid"] + 1e-9 and r["r_mid"] <= r["r"] + 1e-9 for r in T8.values())
    k2 = all(T8[("p", s)]["r"] <= RHO + 1e-5 for s in SEEDS)
    print(f"k1 (r_box_lo ≤ r_mid ≤ r partout) : {'passé' if k1 else 'ÉCHEC'} ; k2 (P : r ≤ ρ) : {'passé' if k2 else 'ÉCHEC'}")
    for arm in ("p", "free", "fix2"):
        for s in SEEDS:
            r = T8[(arm, s)]
            gap = (r["r"] - r["r_mid"]) / (r["r"] - r["r_box_lo"]) if r["r"] > r["r_box_lo"] else float("nan")
            print(f"  {arm:5s} s{s}: r {r['r']:.4g} | r_mid {r['r_mid']:.4g} (r_mid/r = {r['r_mid'] / r['r']:.3f}) | "
                  f"R_lin {r['R_lin']:.4g} | sommet {r['vertex']:.4g} | R_loc {r['R_loc']:.4g} | r_box_lo {r['r_box_lo']:.4g} | "
                  f"lâcheté ≤ r/r_box_lo = {r['r'] / r['r_box_lo']:.2f} | part de l'écart fermée par r_mid {gap:.2f}")
    if k1 and k2:
        fr = [T8[("free", s)]["r_box_lo"] > RHO for s in SEEDS]
        print("Ligne T8 (libre) :", "aucun certificat de cette forme n'admet la solution libre, même le plus fin"
              if maj(fr) else "le certificat le plus fin de cette forme pourrait admettre la solution libre (non exclu)")
        f2 = [T8[("fix2", s)]["r_box_lo"] <= 1.0 for s in SEEDS]
        print("Ligne T8 (fix2) :", "la solution fix2 serait admissible par le certificat le plus fin (ρ = 1) sur la majorité"
              if maj(f2) else f"fix2 non admissible même par le certificat le plus fin (admissible sur {sum(f2)}/5)")
    else:
        print("T8 : contrôle en échec → pas de lecture.")
else:
    print("  t8.json absent")

# ------------------------------------------------------------------ T9–T11
print("\n== T9–T11 — entraînement avec un certificat plus fin ou une métrique apprise")
ref = {a: [json.load(open(os.path.join(D4, f"t4_{a}_s{s}.json"))) for s in SEEDS] for a in ("p", "free")}
Pm, Fm = st.median(r["test"] for r in ref["p"]), st.median(r["test"] for r in ref["free"])
print(f"Références T4 : P médiane {Pm:.4f} ; libre médiane {Fm:.4f}")
pv = os.path.join(D, "dval.json")
DV = {(r["arm"], r["seed"]): r for r in json.load(open(pv))} if os.path.exists(pv) else {}
lab = lambda x: "réussit" if x < 0.05 else "échoue" if x > 0.2 else "partiel"  # noqa: E731
res = {}
for arm in ARMS:
    R = [json.load(open(os.path.join(D, f"{arm}_s{s}.json"))) for s in SEEDS]
    d0 = max(abs(R[s]["losses"][0] - ref["p"][s]["losses"][0]) for s in SEEDS) < 1e-6
    dfin = all(r["c_end"] <= RHO + 1e-5 for r in R)
    dvl = all((arm, s) in DV and DV[(arm, s)]["E"] <= 1 + 1e-4 for s in SEEDS)
    med = st.median(r["test"] for r in R)
    g = (Pm - med) / (Pm - Fm)
    ok = d0 and dfin and dvl
    res[arm] = (lab(med), ok)
    al = [r["alpha"] for r in R]
    print(f"  {arm:8s} test : " + " ".join(f"{r['test']:.4f}" for r in R) + f" | médiane {med:.4f} ({lab(med)}) | g = {g:.2f} | "
          f"c_end max {max(r['c_end'] for r in R):.5f} | r fixe en fin : " + " ".join(f"{r['r_fixe_end']:.3g}" for r in R)
          + (" | α min/max en fin : " + " ".join(f"{min(a):.2g}/{max(a):.2g}" for a in al) if al[0] else "")
          + f" | statuts {[r['status'] for r in R]}")
    print(f"    contrôles : d0 {'ok' if d0 else 'ÉCHEC'} ; d-fin {'ok' if dfin else 'ÉCHEC'} ; d-val "
          + ("ok" if dvl else "ÉCHEC") + " (E : " + " ".join(f"{DV[(arm, s)]['E']:.6f}" for s in SEEDS if (arm, s) in DV) + ")")
    if not ok:
        print(f"    → pas de lecture pour {arm}")
    else:
        text = {"mid": "la finesse de la borne", "alpha": "la métrique fixe", "midalpha": "la borne et la métrique ensemble"}[arm]
        if med < 0.05:
            print(f"    Ligne : hypothèse CONFIRMÉE pour ce bras : l'obstacle était {text}")
        elif med > 0.2:
            print("    Ligne : cette forme de l'hypothèse est RÉFUTÉE sur ce banc")
        else:
            print(f"    Ligne : partiel (g = {g:.2f})")
if all(ok for _, ok in res.values()):
    if all(l == "échoue" for l, _ in res.values()):
        print("Lecture d'ensemble : ni une borne plus fine (centre-rayon) ni une métrique par tête apprise ne suffisent. Restent : un "
              "obstacle propre au chemin, ou une métrique non bloc-diagonale.")
    else:
        print("Lecture d'ensemble : voir les lignes par bras.")
print("Prédictions écrites : T9 partiel, T10 échoue, T11 partiel → observé :", {a: l for a, (l, _) in res.items()})
